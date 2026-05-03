//! Library entry point for the `ctt` CLI.
//!
//! Tests can construct an [`Args`] via `Args::parse_from(...)` and call
//! [`run`] directly without spawning a subprocess.

pub mod args;

use std::fs;
use std::sync::OnceLock;

use ctt::input::{InputOverrides, decode_container};
use ctt::{
    AlphaMode, ColorSpace, Container, ConvertSettings, CubemapInput, Encoder, EncoderInfo, Error,
    Format, FormatExt, Image, Ktx2Supercompression, MipmapFilter, PipelineOutput, Quality, Surface,
    Swizzle, SwizzleChannel, TargetFormat, TextureKind, compiled_in_encoders, format_short_name,
    parse_format, split_cubemap,
};

pub use args::{
    AlphaModeArg, Args, ColorSpaceArg, ContainerArg, CubemapLayoutArg, MipmapFilterArg, QualityArg,
};

/// Initialize the logger at the requested verbosity. The `OnceLock` makes
/// it safe to call from many tests in the same process.
pub fn setup_logger(verbose: u8) {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let level = match verbose {
            0 => log::LevelFilter::Info,
            1 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        };
        fern::Dispatch::new()
            .format(|out, message, record| match record.level() {
                log::Level::Error => out.finish(format_args!("error: {message}")),
                log::Level::Warn => out.finish(format_args!("warning: {message}")),
                _ => out.finish(format_args!("{message}")),
            })
            .level(level)
            .chain(std::io::stderr())
            .apply()
            .expect("failed to initialize logger");
    });
}

/// Run the CLI with already-parsed arguments.
pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.list_encoders {
        print_encoder_table();
        return Ok(());
    }

    let output_path = args
        .output
        .as_ref()
        .ok_or_else(|| Error::UnsupportedFormat("missing --output".into()))?;

    // Container formats (KTX2, DDS) carry color-space and alpha metadata,
    // so leave the override `None` unless the user explicitly passes a flag.
    // Standard images (PNG, JPEG, …) have no such metadata; fall back to
    // sRGB/straight for them.
    let input_color_space_override = args.input_color_space.map(map_color_space);
    let input_alpha_override = args.input_alpha.map(map_alpha_mode);
    let overrides = InputOverrides {
        color_space: input_color_space_override,
        alpha: input_alpha_override,
    };
    let standard_image_color_space = input_color_space_override.unwrap_or(ColorSpace::Srgb);
    let standard_image_alpha = input_alpha_override.unwrap_or(AlphaMode::Straight);

    log::info!("Loading {} input image(s)", args.input.len());
    let images = load_images(
        &args.input,
        overrides,
        standard_image_color_space,
        standard_image_alpha,
    )?;

    let image = if args.cubemap {
        log::info!("Cubemap mode, layout: {:?}", args.cubemap_layout);
        build_cubemap_image(images, args.cubemap_layout)?
    } else if args.volume {
        log::info!("Volume mode: stacking {} slices", images.len());
        build_volume_image(images, &args.input)?
    } else if images.len() == 1 {
        images.into_iter().next().unwrap()
    } else {
        assemble_array(images)?
    };

    let supercompression = match (args.zstd, args.zlib) {
        (Some(level), None) => Some(Ktx2Supercompression::Zstd { level }),
        (None, Some(level)) => Some(Ktx2Supercompression::Zlib { level }),
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!("clap conflicts_with prevents this"),
    };

    let mut container = resolve_container(args.container, output_path)?;
    if let Some(sc) = supercompression {
        if let Container::Ktx2(ref mut opt) = container {
            *opt = Some(sc);
        } else {
            return Err(Error::UnsupportedFormat(
                "supercompression requires KTX2 output; use --container ktx2 or a .ktx2 extension"
                    .into(),
            )
            .into());
        }
    }

    let swizzle = args.swizzle.as_deref().map(parse_swizzle).transpose()?;
    let target_format = args
        .format
        .as_deref()
        .map(parse_format)
        .transpose()?
        .map(|tf| merge_encoder_flags(tf, &args))
        .transpose()?;

    let settings = ConvertSettings {
        format: target_format,
        container,
        quality: map_quality(args.quality),
        output_color_space: args.output_color_space.map(map_color_space),
        output_alpha: args.output_alpha.map(map_alpha_mode),
        swizzle,
        mipmap: args.mipmap,
        mipmap_count: args.mipmap_count,
        mipmap_filter: map_mipmap_filter(args.mipmap_filter),
    };

    let output_bytes = match ctt::convert(image, settings)? {
        PipelineOutput::Encoded(bytes) => bytes,
        PipelineOutput::Raw(_) => {
            return Err(Error::OutputEncoding("unexpected raw output from CLI".into()).into());
        }
    };

    fs::write(output_path, &output_bytes)?;
    log::info!(
        "Output written: {} ({} bytes)",
        output_path.display(),
        output_bytes.len()
    );
    Ok(())
}

/// Resolve the container format from the explicit flag or the output file extension.
fn resolve_container(
    explicit: Option<ContainerArg>,
    output: &std::path::Path,
) -> Result<Container, Error> {
    if let Some(container) = explicit {
        return Ok(match container {
            ContainerArg::Dds => Container::Dds,
            ContainerArg::Ktx2 => Container::Ktx2(None),
        });
    }

    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match ext.as_deref() {
        Some("dds") => Ok(Container::Dds),
        Some("ktx2") => Ok(Container::Ktx2(None)),
        Some(other) => Err(Error::UnsupportedFormat(format!(
            "cannot infer container from extension '.{other}'; use --container or a .dds/.ktx2 extension"
        ))),
        None => Err(Error::UnsupportedFormat(
            "output path has no extension; use --container or a .dds/.ktx2 extension".into(),
        )),
    }
}

fn print_encoder_table() {
    let encoders: Vec<EncoderInfo> = compiled_in_encoders();

    if encoders.is_empty() {
        println!("No encoder backends are enabled.");
        println!("Recompile with features: encoder-intel, encoder-bc7enc");
        return;
    }

    println!("{:<10} {:<12} Formats", "Encoder", "Priority");
    println!("{:<10} {:<12} -------", "-------", "--------");

    for (i, encoder) in encoders.iter().enumerate() {
        let mut formats = Vec::new();
        let mut has_astc = false;
        for &f in encoder.supported_formats {
            if f.block_size().is_some() && f.is_compressed() {
                let (bw, bh) = f.block_size().unwrap();
                let is_astc = matches!(
                    (bw, bh),
                    (4, 4)
                        | (5, 4)
                        | (5, 5)
                        | (6, 5)
                        | (6, 6)
                        | (8, 5)
                        | (8, 6)
                        | (8, 8)
                        | (10, 5)
                        | (10, 6)
                        | (10, 8)
                        | (10, 10)
                        | (12, 10)
                        | (12, 12)
                ) && !matches!(
                    f,
                    Format::BC1_RGBA_UNORM_BLOCK
                        | Format::BC2_UNORM_BLOCK
                        | Format::BC3_UNORM_BLOCK
                        | Format::BC4_UNORM_BLOCK
                        | Format::BC4_SNORM_BLOCK
                        | Format::BC5_UNORM_BLOCK
                        | Format::BC5_SNORM_BLOCK
                        | Format::BC6H_UFLOAT_BLOCK
                        | Format::BC6H_SFLOAT_BLOCK
                        | Format::BC7_UNORM_BLOCK
                        | Format::ETC2_R8G8B8_UNORM_BLOCK
                        | Format::ETC2_R8G8B8_SRGB_BLOCK
                        | Format::ETC2_R8G8B8A1_UNORM_BLOCK
                        | Format::ETC2_R8G8B8A1_SRGB_BLOCK
                        | Format::ETC2_R8G8B8A8_UNORM_BLOCK
                        | Format::ETC2_R8G8B8A8_SRGB_BLOCK
                        | Format::EAC_R11_UNORM_BLOCK
                        | Format::EAC_R11_SNORM_BLOCK
                        | Format::EAC_R11G11_UNORM_BLOCK
                        | Format::EAC_R11G11_SNORM_BLOCK
                );

                if is_astc {
                    if !has_astc {
                        formats.push("astc".to_string());
                        has_astc = true;
                    }
                } else {
                    formats.push(format_short_name(f));
                }
            } else {
                formats.push(format_short_name(f));
            }
        }
        println!("{:<10} {:<12} {}", encoder.name, i + 1, formats.join(", "));
    }

    println!();
    println!("Use a bare format name (e.g. bc7) to use the highest-priority encoder,");
    println!("or prefix with the encoder name (e.g. intel_bc7) to choose explicitly.");
    println!("ASTC formats use astc_WxH (e.g. astc_4x4, astc_8x8, astc_12x12).");
    println!("Uncompressed formats use WebGPU (rgba8unorm) or Vulkan (r8g8b8a8_unorm) names.");
}

fn load_images(
    paths: &[std::path::PathBuf],
    overrides: InputOverrides,
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Result<Vec<Image>, Box<dyn std::error::Error>> {
    profiling::scope!("load_images");
    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        profiling::scope!("load image", &path.display().to_string());
        let data = fs::read(path)?;

        let image = if let Some(img) = decode_container(&data, overrides)? {
            img
        } else {
            let surface = load_standard_image(&data, color_space, alpha)?;
            Image {
                surfaces: vec![vec![surface]],
                kind: TextureKind::Texture2D,
            }
        };

        let first = &image.surfaces[0][0];
        log::debug!(
            "Loaded {}: {}x{}, {:?}, {} layer(s), {} mip(s)",
            path.display(),
            first.width,
            first.height,
            first.format,
            image.surfaces.len(),
            image.surfaces[0].len(),
        );
        images.push(image);
    }
    Ok(images)
}

fn load_standard_image(
    data: &[u8],
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Result<Surface, Box<dyn std::error::Error>> {
    let img = image::load_from_memory(data)?;

    let surface = match img {
        image::DynamicImage::ImageLuma8(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: buf.into_raw(),
                width,
                height,
                depth: 1,
                stride: width,
                slice_stride: 0,
                format: Format::R8_UNORM,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageLumaA8(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: buf.into_raw(),
                width,
                height,
                depth: 1,
                stride: width * 2,
                slice_stride: 0,
                format: Format::R8G8_UNORM,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageRgb8(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: buf.into_raw(),
                width,
                height,
                depth: 1,
                stride: width * 3,
                slice_stride: 0,
                format: Format::R8G8B8_UNORM,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageRgba8(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: buf.into_raw(),
                width,
                height,
                depth: 1,
                stride: width * 4,
                slice_stride: 0,
                format: Format::R8G8B8A8_UNORM,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageLuma16(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: bytemuck::cast_slice(buf.as_raw()).to_vec(),
                width,
                height,
                depth: 1,
                stride: width * 2,
                slice_stride: 0,
                format: Format::R16_UNORM,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageLumaA16(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: bytemuck::cast_slice(buf.as_raw()).to_vec(),
                width,
                height,
                depth: 1,
                stride: width * 4,
                slice_stride: 0,
                format: Format::R16G16_UNORM,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageRgb16(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: bytemuck::cast_slice(buf.as_raw()).to_vec(),
                width,
                height,
                depth: 1,
                stride: width * 6,
                slice_stride: 0,
                format: Format::R16G16B16_UNORM,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageRgba16(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: bytemuck::cast_slice(buf.as_raw()).to_vec(),
                width,
                height,
                depth: 1,
                stride: width * 8,
                slice_stride: 0,
                format: Format::R16G16B16A16_UNORM,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageRgb32F(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: bytemuck::cast_slice(buf.as_raw()).to_vec(),
                width,
                height,
                depth: 1,
                stride: width * 12,
                slice_stride: 0,
                format: Format::R32G32B32_SFLOAT,
                color_space,
                alpha,
            }
        }
        image::DynamicImage::ImageRgba32F(buf) => {
            let (width, height) = buf.dimensions();
            Surface {
                data: bytemuck::cast_slice(buf.as_raw()).to_vec(),
                width,
                height,
                depth: 1,
                stride: width * 16,
                slice_stride: 0,
                format: Format::R32G32B32A32_SFLOAT,
                color_space,
                alpha,
            }
        }
        _ => {
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            Surface {
                data: rgba.into_raw(),
                width,
                height,
                depth: 1,
                stride: width * 4,
                slice_stride: 0,
                format: Format::R8G8B8A8_UNORM,
                color_space,
                alpha,
            }
        }
    };

    Ok(surface)
}

fn build_cubemap_image(
    images: Vec<Image>,
    layout_arg: CubemapLayoutArg,
) -> Result<Image, Box<dyn std::error::Error>> {
    // Single already-cubemap input (single cube or cube array): passthrough.
    if images.len() == 1 && matches!(images[0].kind, TextureKind::Cubemap) {
        return Ok(images.into_iter().next().unwrap());
    }

    // Multiple already-cubemap inputs: concatenate faces into a cubemap array.
    if images.len() > 1
        && images
            .iter()
            .all(|i| matches!(i.kind, TextureKind::Cubemap))
    {
        validate_mip_counts(&images)?;
        let mut surfaces = Vec::new();
        for img in images {
            surfaces.extend(img.surfaces);
        }
        return Ok(Image {
            surfaces,
            kind: TextureKind::Cubemap,
        });
    }

    // N face images where N is a positive multiple of 6: assemble N/6 cubes.
    // Each input must be a single-layer image (mips allowed, but uniform across
    // inputs). N=0 is excluded because clap requires at least one input.
    if !images.is_empty() && images.len().is_multiple_of(6) {
        validate_mip_counts(&images)?;

        for (i, img) in images.iter().enumerate() {
            if img.surfaces.len() != 1 {
                return Err(Error::UnsupportedFormat(format!(
                    "cubemap face input {i} must be single-layer, got {} layer(s)",
                    img.surfaces.len(),
                ))
                .into());
            }
            if matches!(img.kind, TextureKind::Cubemap) {
                return Err(Error::UnsupportedFormat(format!(
                    "input {i} is already a cubemap; mix-and-match with face inputs is not allowed"
                ))
                .into());
            }
            if matches!(img.kind, TextureKind::Texture3D) {
                return Err(Error::UnsupportedFormat(format!(
                    "input {i} is a 3D texture; cannot be a cubemap face"
                ))
                .into());
            }
        }

        let surfaces: Vec<Vec<Surface>> = images
            .into_iter()
            .map(|img| img.surfaces.into_iter().next().unwrap())
            .collect();

        return Ok(Image {
            surfaces,
            kind: TextureKind::Cubemap,
        });
    }

    if images.len() == 1 {
        let image = images.into_iter().next().unwrap();
        if image.surfaces.len() != 1 || image.surfaces[0].len() != 1 {
            return Err(Error::UnsupportedFormat(
                "cubemap layout splitting requires a single-layer, single-mip input".into(),
            )
            .into());
        }
        let surface = image
            .surfaces
            .into_iter()
            .next()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let cubemap_input = match layout_arg {
            CubemapLayoutArg::Cross => CubemapInput::Cross(surface),
            CubemapLayoutArg::Strip => CubemapInput::Strip(surface),
        };
        let faces = split_cubemap(cubemap_input)?;
        let surfaces = faces.into_iter().map(|face| vec![face]).collect();
        return Ok(Image {
            surfaces,
            kind: TextureKind::Cubemap,
        });
    }

    Err(Error::CubemapFaceCount(images.len()).into())
}

/// Stack N single-layer single-mip inputs into one 3D Surface. Each input
/// must share dimensions, format, color space, and alpha. The Z order is
/// argv order. Compressed inputs are passed through verbatim, with each
/// slice's blocks concatenated.
///
/// `paths` is parallel to `images` and is consulted only to make slice
/// mismatch errors point at a specific file.
fn build_volume_image(
    images: Vec<Image>,
    paths: &[std::path::PathBuf],
) -> Result<Image, Box<dyn std::error::Error>> {
    if images.is_empty() {
        return Err(
            Error::UnsupportedFormat("--volume requires at least one input slice".into()).into(),
        );
    }
    debug_assert_eq!(
        images.len(),
        paths.len(),
        "paths must be parallel to images"
    );

    // Single already-3D input: passthrough.
    if images.len() == 1 && matches!(images[0].kind, TextureKind::Texture3D) {
        return Ok(images.into_iter().next().unwrap());
    }

    for (i, img) in images.iter().enumerate() {
        let p = paths[i].display();
        if !matches!(img.kind, TextureKind::Texture2D) {
            return Err(Error::UnsupportedFormat(format!(
                "--volume slice {i} ({p}) must be a 2D image, got {:?}",
                img.kind,
            ))
            .into());
        }
        if img.surfaces.len() != 1 || img.surfaces[0].len() != 1 {
            return Err(Error::UnsupportedFormat(format!(
                "--volume slice {i} ({p}) must be single-layer single-mip"
            ))
            .into());
        }
    }

    let depth = images.len() as u32;
    let first = &images[0].surfaces[0][0];
    let width = first.width;
    let height = first.height;
    let stride = first.stride;
    let format = first.format;
    let color_space = first.color_space;
    let alpha = first.alpha;
    let slice_stride = first.data.len() as u32;
    let p0 = paths[0].display();

    let mut data = Vec::with_capacity((slice_stride as usize) * images.len());
    for (i, img) in images.iter().enumerate() {
        let s = &img.surfaces[0][0];
        let p = paths[i].display();
        if s.width != width || s.height != height {
            return Err(Error::UnsupportedFormat(format!(
                "--volume slice {i} ({p}): dimensions {}x{} differ from slice 0 ({p0}, {}x{})",
                s.width, s.height, width, height,
            ))
            .into());
        }
        if s.format != format {
            return Err(Error::UnsupportedFormat(format!(
                "--volume slice {i} ({p}): format {:?} differs from slice 0 ({p0}, {:?})",
                s.format, format,
            ))
            .into());
        }
        if s.color_space != color_space {
            return Err(Error::UnsupportedFormat(format!(
                "--volume slice {i} ({p}): color space {:?} differs from slice 0 ({p0}, {:?})",
                s.color_space, color_space,
            ))
            .into());
        }
        if s.alpha != alpha {
            return Err(Error::UnsupportedFormat(format!(
                "--volume slice {i} ({p}): alpha mode {:?} differs from slice 0 ({p0}, {:?})",
                s.alpha, alpha,
            ))
            .into());
        }
        if s.data.len() as u32 != slice_stride {
            return Err(Error::UnsupportedFormat(format!(
                "--volume slice {i} ({p}): payload {} bytes differs from slice 0 ({p0}, {} bytes)",
                s.data.len(),
                slice_stride,
            ))
            .into());
        }
        data.extend_from_slice(&s.data);
    }

    Ok(Image {
        surfaces: vec![vec![Surface {
            data,
            width,
            height,
            depth,
            stride,
            slice_stride,
            format,
            color_space,
            alpha,
        }]],
        kind: TextureKind::Texture3D,
    })
}

fn assemble_array(images: Vec<Image>) -> Result<Image, Box<dyn std::error::Error>> {
    validate_mip_counts(&images)?;

    for (i, img) in images.iter().enumerate() {
        if matches!(img.kind, TextureKind::Cubemap) {
            return Err(Error::UnsupportedFormat(format!(
                "input {i} is a cubemap; pass --cubemap to assemble cubemap arrays"
            ))
            .into());
        }
        if matches!(img.kind, TextureKind::Texture3D) {
            return Err(Error::UnsupportedFormat(format!(
                "input {i} is a 3D texture; 3D textures cannot be combined into an array"
            ))
            .into());
        }
    }

    let mut surfaces = Vec::new();
    for img in images {
        surfaces.extend(img.surfaces);
    }

    Ok(Image {
        surfaces,
        kind: TextureKind::Texture2D,
    })
}

fn validate_mip_counts(images: &[Image]) -> Result<(), Box<dyn std::error::Error>> {
    if images.is_empty() {
        return Ok(());
    }

    let expected_mips = images[0].surfaces[0].len();
    for (i, img) in images.iter().enumerate() {
        for (layer_idx, layer) in img.surfaces.iter().enumerate() {
            if layer.len() != expected_mips {
                return Err(Error::UnsupportedFormat(format!(
                    "input {i} layer {layer_idx} has {} mip level(s), \
                     but input 0 has {expected_mips}; all inputs must have the same mip count",
                    layer.len(),
                ))
                .into());
            }
        }
    }

    Ok(())
}

fn map_quality(q: QualityArg) -> Quality {
    match q {
        QualityArg::UltraFast => Quality::UltraFast,
        QualityArg::VeryFast => Quality::VeryFast,
        QualityArg::Fast => Quality::Fast,
        QualityArg::Basic => Quality::Basic,
        QualityArg::Slow => Quality::Slow,
        QualityArg::VerySlow => Quality::VerySlow,
    }
}

/// Merge encoder-flavor flags (`--alpha`, `--dither`, `--heuristics`) into
/// the [`Encoder`] already chosen by [`parse_format`]. Mismatched
/// combinations error out instead of silently overriding the encoder choice.
fn merge_encoder_flags(tf: TargetFormat, args: &Args) -> Result<TargetFormat, Error> {
    let TargetFormat::Compressed { format, encoder } = tf else {
        if args.alpha || args.dither || args.heuristics {
            return Err(Error::UnsupportedFormat(
                "--alpha/--dither/--heuristics require a compressed --format".into(),
            ));
        }
        return Ok(tf);
    };

    let encoder = apply_flags(encoder, args)?;
    Ok(TargetFormat::Compressed { format, encoder })
}

fn apply_flags(encoder: Encoder, args: &Args) -> Result<Encoder, Error> {
    // Each flag belongs to exactly one backend. Reject when used with the
    // wrong (or auto) encoder so the user is told explicitly which encoder
    // a flag applies to instead of having it silently ignored.
    let encoder = if args.alpha {
        match encoder {
            Encoder::Intel(mut s) => {
                s.alpha = true;
                Encoder::Intel(s)
            }
            _ => {
                return Err(Error::UnsupportedFormat(
                    "--alpha only applies when using the intel encoder \
                     (e.g. --format intel_bc7)"
                        .into(),
                ));
            }
        }
    } else {
        encoder
    };

    if args.dither || args.heuristics {
        match encoder {
            Encoder::Etcpak(mut s) => {
                s.dither = args.dither;
                s.use_heuristics = args.heuristics;
                Ok(Encoder::Etcpak(s))
            }
            _ => Err(Error::UnsupportedFormat(
                "--dither/--heuristics only apply when using the etcpak encoder \
                 (e.g. --format etcpak_etc2)"
                    .into(),
            )),
        }
    } else {
        Ok(encoder)
    }
}

fn map_color_space(cs: ColorSpaceArg) -> ColorSpace {
    match cs {
        ColorSpaceArg::Srgb => ColorSpace::Srgb,
        ColorSpaceArg::Linear => ColorSpace::Linear,
    }
}

fn map_mipmap_filter(f: MipmapFilterArg) -> MipmapFilter {
    match f {
        MipmapFilterArg::Nearest => MipmapFilter::Nearest,
        MipmapFilterArg::Triangle => MipmapFilter::Triangle,
        MipmapFilterArg::CatmullRom => MipmapFilter::CatmullRom,
        MipmapFilterArg::Gaussian => MipmapFilter::Gaussian,
        MipmapFilterArg::Lanczos3 => MipmapFilter::Lanczos3,
    }
}

fn map_alpha_mode(a: AlphaModeArg) -> AlphaMode {
    match a {
        AlphaModeArg::Straight => AlphaMode::Straight,
        AlphaModeArg::Premultiplied => AlphaMode::Premultiplied,
        AlphaModeArg::Opaque => AlphaMode::Opaque,
    }
}

fn parse_swizzle(s: &str) -> Result<Swizzle, Error> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() != 4 {
        return Err(Error::InvalidSwizzle(format!(
            "swizzle must be exactly 4 characters, got {}: \"{s}\"",
            chars.len()
        )));
    }

    let mut channels = [SwizzleChannel::R; 4];
    for (i, ch) in chars.iter().enumerate() {
        channels[i] = match ch.to_ascii_lowercase() {
            'r' => SwizzleChannel::R,
            'g' => SwizzleChannel::G,
            'b' => SwizzleChannel::B,
            'a' => SwizzleChannel::A,
            '0' => SwizzleChannel::Zero,
            '1' => SwizzleChannel::One,
            _ => return Err(Error::InvalidSwizzle(format!("unknown channel '{ch}'"))),
        };
    }

    Ok(Swizzle(channels))
}
