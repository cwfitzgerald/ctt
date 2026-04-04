mod args;

use std::fs;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;

use ctt::encoder::{EncoderRegistry, Quality};
use ctt::error::Error;
use ctt::pipeline::{
    AssemblyNode, InputBranch, InputNode, OutputNode, Pipeline, PipelineOutput,
};
use ctt::surface::{ColorSpace, Image, Surface};
use ctt::cubemap::{CubemapInput, split_cubemap};
use ctt::transforms::swizzle::{Swizzle, SwizzleChannel};
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

    let (encoder_name, target_format) = parse_format(format_str, &registry)?;
    let output_node = match args.container {
        ContainerArg::Dds => OutputNode::Dds,
        ContainerArg::Ktx2 => OutputNode::Ktx2,
    };
    let swizzle = args.swizzle.as_deref().map(parse_swizzle).transpose()?;
    let quality = map_quality(args.quality);

    log::info!(
        "Format: {target_format:?}, container: {output_node:?}, quality: {quality:?}, color space: {color_space:?}",
    );

    // Load input images.
    log::info!("Loading {} input image(s)", args.input.len());
    let surfaces = load_images(&args.input, color_space)?;

    // Build input branches and assembly.
    let (inputs, assembly) = if args.cubemap {
        log::info!("Cubemap mode, layout: {:?}", args.cubemap_layout);
        build_cubemap_inputs(surfaces, args.cubemap_layout)?
    } else {
        let inputs: Vec<InputBranch> = surfaces
            .into_iter()
            .map(|surface| {
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
        for &f in encoder.supported_formats() {
            if f.block_size().is_some() && f.is_compressed() {
                // Check if it's an ASTC format by block size > 4x4 or by name pattern
                let (bw, bh) = f.block_size().unwrap();
                let is_astc = matches!(
                    (bw, bh),
                    (4, 4) | (5, 4) | (5, 5) | (6, 5) | (6, 6) | (8, 5) | (8, 6) | (8, 8)
                    | (10, 5) | (10, 6) | (10, 8) | (10, 10) | (12, 10) | (12, 12)
                ) && !matches!(
                    f,
                    ctt::ktx2::Format::BC1_RGBA_UNORM_BLOCK
                    | ctt::ktx2::Format::BC2_UNORM_BLOCK
                    | ctt::ktx2::Format::BC3_UNORM_BLOCK
                    | ctt::ktx2::Format::BC4_UNORM_BLOCK
                    | ctt::ktx2::Format::BC4_SNORM_BLOCK
                    | ctt::ktx2::Format::BC5_UNORM_BLOCK
                    | ctt::ktx2::Format::BC5_SNORM_BLOCK
                    | ctt::ktx2::Format::BC6H_UFLOAT_BLOCK
                    | ctt::ktx2::Format::BC6H_SFLOAT_BLOCK
                    | ctt::ktx2::Format::BC7_UNORM_BLOCK
                    | ctt::ktx2::Format::ETC2_R8G8B8_UNORM_BLOCK
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
}

/// Short display name for a compressed format.
fn format_short_name(format: ctt::ktx2::Format) -> String {
    use ctt::ktx2::Format as F;
    match format {
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGB_UNORM_BLOCK => "bc1".into(),
        F::BC2_UNORM_BLOCK => "bc2".into(),
        F::BC3_UNORM_BLOCK => "bc3".into(),
        F::BC4_UNORM_BLOCK => "bc4".into(),
        F::BC4_SNORM_BLOCK => "bc4s".into(),
        F::BC5_UNORM_BLOCK => "bc5".into(),
        F::BC5_SNORM_BLOCK => "bc5s".into(),
        F::BC6H_UFLOAT_BLOCK => "bc6h".into(),
        F::BC6H_SFLOAT_BLOCK => "bc6hsf".into(),
        F::BC7_UNORM_BLOCK => "bc7".into(),
        F::ETC2_R8G8B8_UNORM_BLOCK => "etc1".into(),
        _ => format!("{format:?}").to_lowercase(),
    }
}

fn load_images(
    paths: &[std::path::PathBuf],
    color_space: ColorSpace,
) -> Result<Vec<Surface>, Box<dyn std::error::Error>> {
    let mut surfaces = Vec::with_capacity(paths.len());
    for path in paths {
        let img = image::open(path)?;

        let surface = match img.color() {
            image::ColorType::Rgb32F | image::ColorType::Rgba32F => {
                let rgba = img.to_rgba32f();
                let (width, height) = rgba.dimensions();
                let stride = width * 4 * 4; // 4 channels * 4 bytes
                Surface {
                    data: bytemuck::cast_slice(rgba.as_raw()).to_vec(),
                    width,
                    height,
                    stride,
                    format: ctt::ktx2::Format::R32G32B32A32_SFLOAT,
                    color_space,
                    alpha: ctt::alpha::AlphaMode::Straight,
                }
            }
            image::ColorType::Rgb16 | image::ColorType::Rgba16 => {
                let rgba = img.to_rgba16();
                let (width, height) = rgba.dimensions();
                let stride = width * 4 * 2; // 4 channels * 2 bytes
                Surface {
                    data: bytemuck::cast_slice(rgba.as_raw()).to_vec(),
                    width,
                    height,
                    stride,
                    format: ctt::ktx2::Format::R16G16B16A16_UNORM,
                    color_space,
                    alpha: ctt::alpha::AlphaMode::Straight,
                }
            }
            _ => {
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                let stride = width * 4;
                Surface {
                    data: rgba.into_raw(),
                    width,
                    height,
                    stride,
                    format: ctt::ktx2::Format::R8G8B8A8_UNORM,
                    color_space,
                    alpha: ctt::alpha::AlphaMode::Straight,
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

/// Build cubemap input branches from loaded surfaces.
fn build_cubemap_inputs(
    surfaces: Vec<Surface>,
    layout_arg: CubemapLayoutArg,
) -> Result<(Vec<InputBranch>, AssemblyNode), Box<dyn std::error::Error>> {
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
    let inputs: Vec<InputBranch> = faces
        .into_iter()
        .map(|face| {
            InputBranch {
                input: InputNode::Raw(Image {
                    surfaces: vec![vec![face]],
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
/// Returns `(optional_encoder_name, target_ktx2_format)`.
fn parse_format(
    s: &str,
    registry: &EncoderRegistry,
) -> Result<(Option<String>, ctt::ktx2::Format), Error> {
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

fn parse_bare_format(lower: &str, original: &str) -> Result<ctt::ktx2::Format, Error> {
    use ctt::ktx2::Format as F;
    match lower {
        "bc1" => Ok(F::BC1_RGBA_UNORM_BLOCK),
        "bc2" => Ok(F::BC2_UNORM_BLOCK),
        "bc3" => Ok(F::BC3_UNORM_BLOCK),
        "bc4" => Ok(F::BC4_UNORM_BLOCK),
        "bc4s" => Ok(F::BC4_SNORM_BLOCK),
        "bc5" => Ok(F::BC5_UNORM_BLOCK),
        "bc5s" => Ok(F::BC5_SNORM_BLOCK),
        "bc6h" => Ok(F::BC6H_UFLOAT_BLOCK),
        "bc6hsf" | "bc6h_sf" => Ok(F::BC6H_SFLOAT_BLOCK),
        "bc7" => Ok(F::BC7_UNORM_BLOCK),
        "etc1" => Ok(F::ETC2_R8G8B8_UNORM_BLOCK),
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
                astc_format(block_width, block_height)
                    .ok_or_else(|| Error::UnsupportedFormat(original.into()))
            } else {
                Err(Error::UnsupportedFormat(original.into()))
            }
        }
    }
}

fn astc_format(block_width: u8, block_height: u8) -> Option<ctt::ktx2::Format> {
    use ctt::ktx2::Format as F;
    Some(match (block_width, block_height) {
        (4, 4) => F::ASTC_4x4_UNORM_BLOCK,
        (5, 4) => F::ASTC_5x4_UNORM_BLOCK,
        (5, 5) => F::ASTC_5x5_UNORM_BLOCK,
        (6, 5) => F::ASTC_6x5_UNORM_BLOCK,
        (6, 6) => F::ASTC_6x6_UNORM_BLOCK,
        (8, 5) => F::ASTC_8x5_UNORM_BLOCK,
        (8, 6) => F::ASTC_8x6_UNORM_BLOCK,
        (8, 8) => F::ASTC_8x8_UNORM_BLOCK,
        (10, 5) => F::ASTC_10x5_UNORM_BLOCK,
        (10, 6) => F::ASTC_10x6_UNORM_BLOCK,
        (10, 8) => F::ASTC_10x8_UNORM_BLOCK,
        (10, 10) => F::ASTC_10x10_UNORM_BLOCK,
        (12, 10) => F::ASTC_12x10_UNORM_BLOCK,
        (12, 12) => F::ASTC_12x12_UNORM_BLOCK,
        _ => return None,
    })
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
