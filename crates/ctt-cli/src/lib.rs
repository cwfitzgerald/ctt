//! Library entry point for the `ctt` CLI.
//!
//! Tests can construct an [`Args`] via `Args::parse_from(...)` and call
//! [`run`] directly without spawning a subprocess.

pub mod args;
pub mod encoder_opts;

use std::fs;
use std::sync::OnceLock;

use ctt::encoders::{Encoder, EncoderInfo, compiled_in_encoders};
use ctt::input::{InputOverrides, decode_container};
use ctt::{
    AlphaMode, ColorSpace, Container, ConvertSettings, CubemapInput, Error, Format, FormatExt,
    Image, Ktx2Supercompression, MipmapFilter, PipelineOutput, Quality, Surface, Swizzle,
    SwizzleChannel, TargetFormat, TextureKind, format_short_name, parse_format, split_cubemap,
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

    if let Some(name) = args.help_encoder.as_deref() {
        return print_encoder_help(name);
    }

    let output_path = args
        .output
        .as_ref()
        .ok_or_else(|| Error::UnsupportedFormat("missing --output".into()))?;

    // Refuse to clobber one of our own inputs. Silent overwrite of unrelated
    // existing files is intentionally allowed.
    check_output_not_input(&args.input, output_path)?;

    // `--cubemap-layout` only applies when splitting a single input image; it
    // is meaningless (and silently ignored) with multiple face inputs.
    if args.cubemap_layout.is_some() && args.input.len() > 1 {
        return Err(Error::UnsupportedFormat(
            "--cubemap-layout applies only to a single input image; \
             it cannot be combined with multiple inputs"
                .into(),
        )
        .into());
    }

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
        let layout = args.cubemap_layout.unwrap_or(CubemapLayoutArg::Cross);
        log::info!("Cubemap mode, layout: {layout:?}");
        build_cubemap_image(images, layout)?
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
        .map(|tf| merge_encoder_opts(tf, &args))
        .transpose()?;

    let settings = ConvertSettings {
        format: target_format,
        container,
        quality: map_quality(args.quality),
        output_color_space: args.output_color_space.map(map_color_space),
        output_alpha: args.output_alpha.map(map_alpha_mode),
        allow_discarding_alpha: args.allow_discarding_alpha,
        swizzle,
        mipmap: args.mipmap,
        mipmap_count: args.mipmap_count,
        mipmap_filter: map_mipmap_filter(args.mipmap_filter),
    };

    // With the default of zero workers, convert on Rayon's lazily initialized
    // global pool instead of paying for a dedicated one.
    let converted = match args.threads {
        0 => ctt::convert(image, settings),
        n => build_thread_pool(n)?.install(|| ctt::convert(image, settings)),
    };
    let output_bytes = match converted? {
        PipelineOutput::Encoded(bytes) => bytes,
        PipelineOutput::Raw(_) => {
            return Err(Error::OutputEncoding("unexpected raw output from CLI".into()).into());
        }
    };

    fs::write(output_path, &output_bytes).map_err(|e| {
        Error::OutputEncoding(format!("failed to write {}: {e}", output_path.display()))
    })?;
    log::info!(
        "Output written: {} ({} bytes)",
        output_path.display(),
        output_bytes.len()
    );
    Ok(())
}

fn build_thread_pool(threads: usize) -> Result<rayon::ThreadPool, Error> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| Error::Compression(format!("failed to create compression thread pool: {e}")))
}

/// Error out if `output` refers to the same file as any input path, so we
/// never silently truncate an input we still need to read.
///
/// The comparison is best-effort canonicalized: inputs exist and canonicalize
/// cleanly, but the output usually does not exist yet, so we canonicalize its
/// parent directory and rejoin the file name. If even that fails we fall back
/// to the lexical path, which still catches the common `in.png -o in.png` case.
fn check_output_not_input(
    inputs: &[std::path::PathBuf],
    output: &std::path::Path,
) -> Result<(), Error> {
    let output_key = normalize_for_compare(output);
    for input in inputs {
        let same_existing_file =
            output.exists() && same_file::is_same_file(input, output).unwrap_or(false);
        if same_existing_file || normalize_for_compare(input) == output_key {
            return Err(Error::UnsupportedFormat(format!(
                "output path {} is also an input; refusing to overwrite it",
                output.display()
            )));
        }
    }
    Ok(())
}

/// Best-effort absolute, symlink-resolved key for path comparison. Falls back
/// to canonicalizing the parent (for not-yet-existing outputs), then to the
/// lexical path.
fn normalize_for_compare(path: &std::path::Path) -> std::path::PathBuf {
    if let Ok(canonical) = fs::canonicalize(path) {
        return canonical;
    }
    if let Some(name) = path.file_name() {
        let parent = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => std::path::Path::new("."),
        };
        if let Ok(canonical_parent) = fs::canonicalize(parent) {
            return canonical_parent.join(name);
        }
    }
    path.to_path_buf()
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
    print!("{}", encoder_table_string());
}

/// Build the `--list-encoders` table as a string. Exposed so tests can assert
/// on its contents without capturing stdout.
pub fn encoder_table_string() -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let encoders: Vec<EncoderInfo> = compiled_in_encoders();

    if encoders.is_empty() {
        let _ = writeln!(out, "No encoder backends are enabled.");
        let _ = writeln!(
            out,
            "Recompile with features: encoder-intel, encoder-bc7enc"
        );
        return out;
    }

    let _ = writeln!(out, "{:<10} {:<12} Formats", "Encoder", "Priority");
    let _ = writeln!(out, "{:<10} {:<12} -------", "-------", "--------");

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
        let _ = writeln!(
            out,
            "{:<10} {:<12} {}",
            encoder.name,
            i + 1,
            formats.join(", ")
        );
    }

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Use a bare format name (e.g. bc7) to use the highest-priority encoder,"
    );
    let _ = writeln!(
        out,
        "or prefix with the encoder name (e.g. intel_bc7) to choose explicitly."
    );
    let _ = writeln!(
        out,
        "ASTC formats use astc_WxH (e.g. astc_4x4, astc_8x8, astc_12x12)."
    );
    let _ = writeln!(
        out,
        "Uncompressed formats use WebGPU (rgba8unorm) or Vulkan (r8g8b8a8_unorm) names."
    );
    out
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
        let data = fs::read(path)
            .map_err(|e| Error::InputDecoding(format!("failed to read {}: {e}", path.display())))?;

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

/// Merge `--<encoder>-opts` strings into the [`Encoder`] chosen by
/// [`parse_format`]. Opts targeting a different encoder than the one selected
/// are warned about and ignored — they aren't a hard error so scripts that
/// toggle `--format` between encoders can leave the opts flag in place.
fn merge_encoder_opts(tf: TargetFormat, args: &Args) -> Result<TargetFormat, Error> {
    let TargetFormat::Compressed {
        format,
        mut encoder,
    } = tf
    else {
        for (name, raw) in opts_strings(args) {
            if raw.is_some() {
                log::warn!("--{name}-opts ignored: --format is uncompressed");
            }
        }
        return Ok(tf);
    };

    // A bare format name (e.g. `-f bc7`) yields `Encoder::Auto`. Resolve it to
    // the concrete backend that will actually run so the matching
    // `--<encoder>-opts` are applied instead of being warn-dropped. If nothing
    // supports the format, leave `Auto` in place and let the encode step raise
    // the real error.
    if matches!(encoder, Encoder::Auto)
        && let Some(resolved) = ctt::encoders::resolve_auto_encoder(format)
    {
        encoder = resolved;
    }

    if let Some(raw) = args.astcenc_opts.as_deref() {
        encoder = apply_astcenc_opts(encoder, raw)?;
    }
    if let Some(raw) = args.bc7e_opts.as_deref() {
        encoder = apply_bc7enc_opts(encoder, raw)?;
    }
    if let Some(raw) = args.intel_opts.as_deref() {
        encoder = apply_intel_opts(encoder, raw)?;
    }
    if let Some(raw) = args.etcpak_opts.as_deref() {
        encoder = apply_etcpak_opts(encoder, raw)?;
    }
    if let Some(raw) = args.amd_opts.as_deref() {
        encoder = apply_amd_opts(encoder, raw)?;
    }

    Ok(TargetFormat::Compressed { format, encoder })
}

fn opts_strings(args: &Args) -> [(&'static str, &Option<String>); 5] {
    [
        ("astcenc", &args.astcenc_opts),
        ("bc7e", &args.bc7e_opts),
        ("intel", &args.intel_opts),
        ("etcpak", &args.etcpak_opts),
        ("amd", &args.amd_opts),
    ]
}

fn apply_astcenc_opts(encoder: Encoder, raw: &str) -> Result<Encoder, Error> {
    match encoder {
        Encoder::Astcenc(_seed) => {
            let parsed = encoder_opts::parse_opts::<encoder_opts::astcenc::Opts>(raw)
                .map_err(|e| Error::UnsupportedFormat(format!("--astcenc-opts: {e}")))?;
            for w in parsed.value.warnings() {
                log::warn!("--astcenc-opts: {w}");
            }
            // astcenc has no legacy flags, so the seed is always defaults —
            // wholesale replacement is correct.
            Ok(Encoder::Astcenc(parsed.value.into_settings()))
        }
        other => {
            log::warn!("--astcenc-opts ignored: --format selected a non-astcenc encoder");
            Ok(other)
        }
    }
}

fn apply_bc7enc_opts(encoder: Encoder, raw: &str) -> Result<Encoder, Error> {
    match encoder {
        Encoder::Bc7enc(_seed) => {
            let parsed = encoder_opts::parse_opts::<encoder_opts::bc7enc::Opts>(raw)
                .map_err(|e| Error::UnsupportedFormat(format!("--bc7e-opts: {e}")))?;
            Ok(Encoder::Bc7enc(parsed.value.into_settings()))
        }
        other => {
            log::warn!("--bc7e-opts ignored: --format selected a non-bc7e encoder");
            Ok(other)
        }
    }
}

fn apply_intel_opts(encoder: Encoder, raw: &str) -> Result<Encoder, Error> {
    match encoder {
        Encoder::Intel(_seed) => {
            let parsed = encoder_opts::parse_opts::<encoder_opts::intel::Opts>(raw)
                .map_err(|e| Error::UnsupportedFormat(format!("--intel-opts: {e}")))?;
            Ok(Encoder::Intel(parsed.value.into_settings()))
        }
        other => {
            log::warn!("--intel-opts ignored: --format selected a non-intel encoder");
            Ok(other)
        }
    }
}

fn apply_etcpak_opts(encoder: Encoder, raw: &str) -> Result<Encoder, Error> {
    match encoder {
        Encoder::Etcpak(_seed) => {
            let parsed = encoder_opts::parse_opts::<encoder_opts::etcpak::Opts>(raw)
                .map_err(|e| Error::UnsupportedFormat(format!("--etcpak-opts: {e}")))?;
            Ok(Encoder::Etcpak(parsed.value.into_settings()))
        }
        other => {
            log::warn!("--etcpak-opts ignored: --format selected a non-etcpak encoder");
            Ok(other)
        }
    }
}

fn apply_amd_opts(encoder: Encoder, raw: &str) -> Result<Encoder, Error> {
    match encoder {
        Encoder::Amd(_seed) => {
            let parsed = encoder_opts::parse_opts::<encoder_opts::amd::Opts>(raw)
                .map_err(|e| Error::UnsupportedFormat(format!("--amd-opts: {e}")))?;
            Ok(Encoder::Amd(parsed.value.into_settings()))
        }
        other => {
            log::warn!("--amd-opts ignored: --format selected a non-amd encoder");
            Ok(other)
        }
    }
}

/// Render `--help-encoder NAME` for one of the compiled-in backends.
fn print_encoder_help(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    match name {
        "astcenc" => {
            encoder_opts::print_help_encoder::<encoder_opts::astcenc::Opts>("astcenc");
        }
        "bc7e" => {
            encoder_opts::print_help_encoder::<encoder_opts::bc7enc::Opts>("bc7e");
        }
        "intel" => {
            encoder_opts::print_help_encoder::<encoder_opts::intel::Opts>("intel");
        }
        "etcpak" => {
            encoder_opts::print_help_encoder::<encoder_opts::etcpak::Opts>("etcpak");
        }
        "amd" => {
            encoder_opts::print_help_encoder::<encoder_opts::amd::Opts>("amd");
        }
        other => {
            return Err(Error::UnsupportedFormat(format!(
                "unknown encoder `{other}`; run --list-encoders for the compiled-in set"
            ))
            .into());
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::build_thread_pool;

    #[test]
    fn thread_pool_uses_requested_worker_count() {
        for count in [1, 4] {
            let pool = build_thread_pool(count).unwrap();
            assert_eq!(pool.current_num_threads(), count);
        }
    }
}
