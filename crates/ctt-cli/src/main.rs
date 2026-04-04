mod args;

use std::fs;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;

use ctt::encoder::{EncoderRegistry, Quality};
use ctt::error::Error;
use ctt::format::{ChannelType, ColorSpace, CompressedFormat, PixelComponents, PixelFormat};
use ctt::image::RawImage;
use ctt::pipeline::{
    AssemblyNode, InputBranch, InputNode, OutputNode, Pipeline, PipelineOutput,
};
use ctt::surface::{Image, Surface};
use ctt::transform::cubemap::{CubemapInput, split_cubemap};
use ctt::transform::swizzle::{Swizzle, SwizzleChannel};
use ctt::transforms::compress::CompressTransform;
use ctt::transforms::swizzle::SwizzleTransform;
use ctt::vk_format::FormatExt;

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
    let registry = Arc::new(EncoderRegistry::default_registry());

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

    let (encoder_name, compressed_format) = parse_format(format_str, &registry)?;
    let output_node = match args.container {
        ContainerArg::Dds => OutputNode::Dds,
        ContainerArg::Ktx2 => OutputNode::Ktx2,
    };
    let swizzle = args.swizzle.as_deref().map(parse_swizzle).transpose()?;
    let quality = map_quality(args.quality);

    // Determine target format as ktx2::Format.
    let target_format = ctt::ktx2::Format::from_compressed(compressed_format, color_space);
    let (target_format, _) = target_format.normalize();

    log::info!(
        "Format: {compressed_format}, container: {output_node:?}, quality: {quality:?}, color space: {color_space:?}",
    );

    // Load input images.
    log::info!("Loading {} input image(s)", args.input.len());
    let raw_images = load_images(&args.input, color_space)?;

    // Build input branches and assembly.
    let (inputs, assembly) = if args.cubemap {
        log::info!("Cubemap mode, layout: {:?}", args.cubemap_layout);
        build_cubemap_inputs(raw_images, args.cubemap_layout)?
    } else {
        let inputs: Vec<InputBranch> = raw_images
            .into_iter()
            .map(|raw| {
                let surface = Surface::from_raw_image(raw);
                InputBranch {
                    input: InputNode::Raw(Image {
                        surfaces: vec![vec![surface]],
                        is_cubemap: false,
                    }),
                    transforms: Vec::new(),
                }
            })
            .collect();

        let assembly = if inputs.len() == 1 {
            AssemblyNode::Identity
        } else {
            AssemblyNode::Array
        };
        (inputs, assembly)
    };

    // Build post-assembly transforms.
    let mut transforms: Vec<Box<dyn ctt::transform_node::Transform>> = Vec::new();

    if let Some(ref swizzle) = swizzle {
        transforms.push(Box::new(SwizzleTransform::new(*swizzle)));
    }

    let encoder_settings = build_encoder_settings(args);
    transforms.push(Box::new(CompressTransform::new(
        target_format,
        quality,
        encoder_name,
        encoder_settings,
        Arc::clone(&registry),
    )));

    let pipeline = Pipeline {
        inputs,
        assembly,
        transforms,
        output: output_node,
    };

    // Resolve and execute.
    let resolved = pipeline.resolve().map_err(|errors| {
        let messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        Error::UnsupportedFormat(messages.join("; "))
    })?;

    let output_bytes = match resolved.execute()? {
        PipelineOutput::Encoded(bytes) => bytes,
        PipelineOutput::Raw(_) => {
            return Err(Error::OutputEncoding("unexpected raw output".into()).into())
        }
    };

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
        println!("Recompile with features: encoder-intel, encoder-bc7enc");
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
    println!("or prefix with the encoder name (e.g. intel_bc7) to choose explicitly.");
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

/// Build cubemap input branches from loaded images.
///
/// Uses the legacy `split_cubemap` for cross/strip splitting, then converts to new types.
fn build_cubemap_inputs(
    images: Vec<RawImage>,
    layout_arg: CubemapLayoutArg,
) -> Result<(Vec<InputBranch>, AssemblyNode), Box<dyn std::error::Error>> {
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
    let inputs: Vec<InputBranch> = faces
        .into_iter()
        .map(|face| {
            let surface = Surface::from_raw_image(face);
            InputBranch {
                input: InputNode::Raw(Image {
                    surfaces: vec![vec![surface]],
                    is_cubemap: false,
                }),
                transforms: Vec::new(),
            }
        })
        .collect();

    Ok((inputs, AssemblyNode::Cubemap))
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

/// Parse a format string, optionally with an encoder prefix (e.g., "intel_bc7", "bc7e_bc7").
///
/// Returns `(optional_encoder_name, compressed_format)`.
fn parse_format(
    s: &str,
    registry: &EncoderRegistry,
) -> Result<(Option<String>, CompressedFormat), Error> {
    let lower = s.to_lowercase();

    // Derive encoder prefixes from the registry, sorted longest-first
    // so that longer names match before shorter ones (e.g. "astcenc" before "astc").
    let mut prefixes: Vec<String> = registry
        .encoders()
        .iter()
        .map(|e| e.name().to_string())
        .collect();
    prefixes.sort_by_key(|p| std::cmp::Reverse(p.len()));

    for prefix in &prefixes {
        if let Some(rest) = lower
            .strip_prefix(prefix.as_str())
            .and_then(|r| r.strip_prefix('_'))
        {
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
        "bc2" => Ok(CompressedFormat::Bc2),
        "bc3" => Ok(CompressedFormat::Bc3),
        "bc4" => Ok(CompressedFormat::Bc4),
        "bc4s" => Ok(CompressedFormat::Bc4s),
        "bc5" => Ok(CompressedFormat::Bc5),
        "bc5s" => Ok(CompressedFormat::Bc5s),
        "bc6h" => Ok(CompressedFormat::Bc6h),
        "bc6hsf" | "bc6h_sf" => Ok(CompressedFormat::Bc6hSf),
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
