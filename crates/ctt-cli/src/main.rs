mod args;

use std::fs;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;

use ctt::encoders::EncoderRegistry;
use ctt::{
    AlphaMode, ColorSpace, Container, ConvertOutput, ConvertSettings, CubemapInput, Error, Format,
    FormatExt, Image, MipmapFilter, Quality, Surface, Swizzle, SwizzleChannel, format_short_name,
    parse_format, split_cubemap,
};

use args::{
    AlphaModeArg, Args, ColorSpaceArg, ContainerArg, CubemapLayoutArg, MipmapFilterArg, QualityArg,
};

fn main() -> ExitCode {
    let args = Args::parse();
    setup_logger(args.verbose);
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn setup_logger(verbose: u8) {
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
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let registry = Arc::new(EncoderRegistry::default_registry());

    if args.list_encoders {
        print_encoder_table(&registry);
        return Ok(());
    }

    let output_path = args.output.as_ref().unwrap();

    let input_color_space = map_color_space(args.input_color_space);
    let input_alpha = map_alpha_mode(args.input_alpha);

    // Load input images.
    log::info!("Loading {} input image(s)", args.input.len());
    let surfaces = load_images(&args.input, input_color_space, input_alpha)?;

    // Assemble the image (cubemap, array, or single).
    let image = if args.cubemap {
        log::info!("Cubemap mode, layout: {:?}", args.cubemap_layout);
        build_cubemap_image(surfaces, args.cubemap_layout)?
    } else if surfaces.len() == 1 {
        Image {
            surfaces: vec![vec![surfaces.into_iter().next().unwrap()]],
            is_cubemap: false,
        }
    } else {
        Image {
            surfaces: surfaces.into_iter().map(|s| vec![s]).collect(),
            is_cubemap: false,
        }
    };

    let container = resolve_container(args.container, output_path)?;
    let swizzle = args.swizzle.as_deref().map(parse_swizzle).transpose()?;
    let target_format = args
        .format
        .as_deref()
        .map(|s| parse_format(s, &registry))
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
        allow_lossy: args.allow_lossy_intermediates,
        encoder_settings: build_encoder_settings(args),
        registry: Some(Arc::clone(&registry)),
    };

    let output_bytes = match ctt::convert(image, settings)? {
        ConvertOutput::Encoded(bytes) => bytes,
        ConvertOutput::Raw(_) => {
            return Err(Error::OutputEncoding("unexpected raw output from CLI".into()).into());
        }
    };

    // Write output.
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
            ContainerArg::Ktx2 => Container::Ktx2,
        });
    }

    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match ext.as_deref() {
        Some("dds") => Ok(Container::Dds),
        Some("ktx2") => Ok(Container::Ktx2),
        Some(other) => Err(Error::UnsupportedFormat(format!(
            "cannot infer container from extension '.{other}'; use --container or a .dds/.ktx2 extension"
        ))),
        None => Err(Error::UnsupportedFormat(
            "output path has no extension; use --container or a .dds/.ktx2 extension".into(),
        )),
    }
}

fn print_encoder_table(registry: &EncoderRegistry) {
    let encoders = registry.encoders();

    if encoders.is_empty() {
        println!("No encoder backends are enabled.");
        println!("Recompile with features: encoder-intel, encoder-bc7enc");
        return;
    }

    // Header
    println!("{:<10} {:<12} Formats", "Encoder", "Priority");
    println!("{:<10} {:<12} -------", "-------", "--------");

    for (i, encoder) in encoders.iter().enumerate() {
        let mut formats = Vec::new();
        let mut has_astc = false;
        for &f in encoder.supported_formats() {
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
        println!(
            "{:<10} {:<12} {}",
            encoder.name(),
            i + 1,
            formats.join(", ")
        );
    }

    println!();
    println!("Use a bare format name (e.g. bc7) to use the highest-priority encoder,");
    println!("or prefix with the encoder name (e.g. intel_bc7) to choose explicitly.");
    println!("ASTC formats use astc_WxH (e.g. astc_4x4, astc_8x8, astc_12x12).");
    println!("Uncompressed formats use WebGPU (rgba8unorm) or Vulkan (r8g8b8a8_unorm) names.");
}

fn load_images(
    paths: &[std::path::PathBuf],
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Result<Vec<Surface>, Box<dyn std::error::Error>> {
    profiling::scope!("load_images");
    let mut surfaces = Vec::with_capacity(paths.len());
    for path in paths {
        profiling::scope!("load image", &path.display().to_string());
        let img = image::open(path)?;

        let surface = match img {
            image::DynamicImage::ImageLuma8(buf) => {
                let (width, height) = buf.dimensions();
                Surface {
                    data: buf.into_raw(),
                    width,
                    height,
                    stride: width,
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
                    stride: width * 2,
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
                    stride: width * 3,
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
                    stride: width * 4,
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
                    stride: width * 2,
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
                    stride: width * 4,
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
                    stride: width * 6,
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
                    stride: width * 8,
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
                    stride: width * 12,
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
                    stride: width * 16,
                    format: Format::R32G32B32A32_SFLOAT,
                    color_space,
                    alpha,
                }
            }
            // DynamicImage is #[non_exhaustive]
            _ => {
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                Surface {
                    data: rgba.into_raw(),
                    width,
                    height,
                    stride: width * 4,
                    format: Format::R8G8B8A8_UNORM,
                    color_space,
                    alpha,
                }
            }
        };

        log::debug!(
            "Loaded {}: {}x{}, {:?}",
            path.display(),
            surface.width,
            surface.height,
            surface.format,
        );
        surfaces.push(surface);
    }
    Ok(surfaces)
}

/// Build a cubemap Image from loaded surfaces.
fn build_cubemap_image(
    surfaces: Vec<Surface>,
    layout_arg: CubemapLayoutArg,
) -> Result<Image, Box<dyn std::error::Error>> {
    let cubemap_input = if surfaces.len() == 6 {
        CubemapInput::SeparateFaces(Box::new(
            surfaces
                .try_into()
                .map_err(|_| Error::CubemapFaceCount(0))?,
        ))
    } else if surfaces.len() == 1 {
        let surface = surfaces.into_iter().next().unwrap();
        match layout_arg {
            CubemapLayoutArg::Cross => CubemapInput::Cross(surface),
            CubemapLayoutArg::Strip => CubemapInput::Strip(surface),
        }
    } else {
        return Err(Error::CubemapFaceCount(surfaces.len()).into());
    };

    let faces = split_cubemap(cubemap_input)?;
    let surfaces = faces.into_iter().map(|face| vec![face]).collect();
    Ok(Image {
        surfaces,
        is_cubemap: true,
    })
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

/// Build encoder-specific settings from CLI args.
fn build_encoder_settings(args: &Args) -> Option<Box<dyn ctt::encoders::EncoderSettings>> {
    if args.alpha {
        return Some(Box::new(ctt::encoders::ispc::IspcSettings { alpha: true }));
    }
    None
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
