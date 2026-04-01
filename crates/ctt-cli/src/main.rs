mod args;

use std::fs;
use std::process::ExitCode;

use clap::Parser;

use ctt::config::{CompressConfig, OutputFormat};
use ctt::encoder::{EncoderRegistry, Quality};
use ctt::error::Error;
use ctt::format::{ChannelType, ColorSpace, CompressedFormat, PixelComponents, PixelFormat};
use ctt::image::{ImageLayout, RawImage};
use ctt::transform::cubemap::{CubemapInput, split_cubemap};
use ctt::transform::swizzle::{Swizzle, SwizzleChannel};

use args::{Args, ColorSpaceArg, ContainerArg, CubemapLayoutArg, QualityArg};

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
    let registry = EncoderRegistry::default_registry();

    if args.list_encoders {
        print_encoder_table(&registry);
        return Ok(());
    }

    let format_str = args.format.as_deref().unwrap();
    let output = args.output.as_ref().unwrap();

    let color_space = match args.color_space {
        ColorSpaceArg::Srgb => ColorSpace::Srgb,
        ColorSpaceArg::Linear => ColorSpace::Linear,
    };

    let (encoder_name, format) = parse_format(format_str, &registry)?;
    let output_format = match args.container {
        ContainerArg::Dds => OutputFormat::Dds,
        ContainerArg::Ktx2 => OutputFormat::Ktx2,
    };
    let swizzle = args.swizzle.as_deref().map(parse_swizzle).transpose()?;
    let quality = map_quality(args.quality);

    let encoder_settings = build_encoder_settings(args);

    let config = CompressConfig {
        format,
        output_format,
        swizzle,
        color_space,
        quality,
        encoder_name,
        encoder_settings,
    };

    log::info!(
        "Format: {format}, container: {output_format:?}, quality: {quality:?}, color space: {color_space:?}",
    );

    // Load input images in their native format.
    log::info!("Loading {} input image(s)", args.input.len());
    let images = load_images(&args.input, color_space)?;

    // Build layout.
    let layout = if args.cubemap {
        log::info!("Cubemap mode, layout: {:?}", args.cubemap_layout);
        build_cubemap_layout(images, args.cubemap_layout)?
    } else {
        ImageLayout {
            layers: images.into_iter().map(|img| vec![img]).collect(),
            is_cubemap: false,
        }
    };

    // Run pipeline.
    let output_bytes = ctt::pipeline::run(&config, layout)?;

    // Write output.
    fs::write(output, &output_bytes)?;
    log::info!(
        "Output written: {} ({} bytes)",
        output.display(),
        output_bytes.len()
    );
    Ok(())
}

fn print_encoder_table(registry: &EncoderRegistry) {
    let encoders = registry.encoders();

    if encoders.is_empty() {
        println!("No encoder backends are enabled.");
        println!("Recompile with features: encoder-ispc, encoder-bc7enc");
        return;
    }

    // Header
    println!("{:<10} {:<12} Formats", "Encoder", "Priority");
    println!("{:<10} {:<12} -------", "-------", "--------");

    for (i, encoder) in encoders.iter().enumerate() {
        let mut formats = Vec::new();
        let mut has_astc = false;
        for f in encoder.supported_formats() {
            match f {
                CompressedFormat::Astc { .. } => {
                    if !has_astc {
                        formats.push("astc".to_string());
                        has_astc = true;
                    }
                }
                other => formats.push(format!("{other:?}").to_lowercase()),
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
    println!("or prefix with the encoder name (e.g. ispc_bc7) to choose explicitly.");
    println!("ASTC formats use astc_WxH (e.g. astc_4x4, astc_8x8, astc_12x12).");
}

fn load_images(
    paths: &[std::path::PathBuf],
    color_space: ColorSpace,
) -> Result<Vec<RawImage>, Box<dyn std::error::Error>> {
    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        let img = image::open(path)?;

        let raw = match img.color() {
            image::ColorType::Rgb32F | image::ColorType::Rgba32F => {
                let rgba = img.to_rgba32f();
                let (width, height) = rgba.dimensions();
                let stride = width * 4 * 4; // 4 channels * 4 bytes
                RawImage {
                    data: bytemuck::cast_slice(rgba.as_raw()).to_vec(),
                    width,
                    height,
                    stride,
                    pixel_format: PixelFormat {
                        components: PixelComponents::Rgba,
                        channel_type: ChannelType::F32,
                        color_space,
                    },
                }
            }
            image::ColorType::Rgb16 | image::ColorType::Rgba16 => {
                let rgba = img.to_rgba16();
                let (width, height) = rgba.dimensions();
                let stride = width * 4 * 2; // 4 channels * 2 bytes
                RawImage {
                    data: bytemuck::cast_slice(rgba.as_raw()).to_vec(),
                    width,
                    height,
                    stride,
                    pixel_format: PixelFormat {
                        components: PixelComponents::Rgba,
                        channel_type: ChannelType::U16,
                        color_space,
                    },
                }
            }
            _ => {
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                let stride = width * 4;
                RawImage {
                    data: rgba.into_raw(),
                    width,
                    height,
                    stride,
                    pixel_format: PixelFormat {
                        components: PixelComponents::Rgba,
                        channel_type: ChannelType::U8,
                        color_space,
                    },
                }
            }
        };

        log::debug!(
            "Loaded {}: {}x{}, {}",
            path.display(),
            raw.width,
            raw.height,
            raw.pixel_format
        );
        images.push(raw);
    }
    Ok(images)
}

fn build_cubemap_layout(
    images: Vec<RawImage>,
    layout_arg: CubemapLayoutArg,
) -> Result<ImageLayout, Box<dyn std::error::Error>> {
    let cubemap_input = if images.len() == 6 {
        CubemapInput::SeparateFaces(images.try_into().map_err(|_| Error::CubemapFaceCount(0))?)
    } else if images.len() == 1 {
        let img = images.into_iter().next().unwrap();
        match layout_arg {
            CubemapLayoutArg::Cross => CubemapInput::Cross(img),
            CubemapLayoutArg::Strip => CubemapInput::Strip(img),
        }
    } else {
        return Err(Error::CubemapFaceCount(images.len()).into());
    };

    let faces = split_cubemap(cubemap_input)?;
    let layers = faces.into_iter().map(|face| vec![face]).collect();
    Ok(ImageLayout {
        layers,
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
fn build_encoder_settings(args: &Args) -> Option<Box<dyn ctt::encoder::EncoderSettings>> {
    if args.alpha {
        return Some(Box::new(ctt::encoders::ispc::IspcSettings { alpha: true }));
    }
    None
}

/// Parse a format string, optionally with an encoder prefix (e.g., "ispc_bc7", "bc7e_bc7").
///
/// Returns `(optional_encoder_name, compressed_format)`.
fn parse_format(
    s: &str,
    registry: &EncoderRegistry,
) -> Result<(Option<String>, CompressedFormat), Error> {
    let lower = s.to_lowercase();

    // Derive encoder prefixes from the registry, sorted longest-first
    // so that longer names match before shorter ones (e.g. "astcenc" before "astc").
    let mut prefixes: Vec<String> = registry.encoders().iter().map(|e| e.name().to_string()).collect();
    prefixes.sort_by_key(|p| std::cmp::Reverse(p.len()));

    for prefix in &prefixes {
        if let Some(rest) = lower.strip_prefix(prefix.as_str()).and_then(|r| r.strip_prefix('_')) {
            let format = parse_bare_format(rest, s)?;
            return Ok((Some(prefix.to_string()), format));
        }
    }

    // No prefix — parse as bare format.
    let format = parse_bare_format(&lower, s)?;
    Ok((None, format))
}

fn parse_bare_format(lower: &str, original: &str) -> Result<CompressedFormat, Error> {
    match lower {
        "bc1" => Ok(CompressedFormat::Bc1),
        "bc3" => Ok(CompressedFormat::Bc3),
        "bc4" => Ok(CompressedFormat::Bc4),
        "bc5" => Ok(CompressedFormat::Bc5),
        "bc6h" => Ok(CompressedFormat::Bc6h),
        "bc7" => Ok(CompressedFormat::Bc7),
        "etc1" => Ok(CompressedFormat::Etc1),
        other => {
            if let Some(rest) = other.strip_prefix("astc_") {
                let (w, h) = rest
                    .split_once('x')
                    .ok_or_else(|| Error::UnsupportedFormat(original.into()))?;
                let block_width: u8 = w
                    .parse()
                    .map_err(|_| Error::UnsupportedFormat(original.into()))?;
                let block_height: u8 = h
                    .parse()
                    .map_err(|_| Error::UnsupportedFormat(original.into()))?;
                Ok(CompressedFormat::Astc {
                    block_width,
                    block_height,
                })
            } else {
                Err(Error::UnsupportedFormat(original.into()))
            }
        }
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
