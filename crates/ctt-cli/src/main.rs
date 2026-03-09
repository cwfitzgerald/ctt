mod args;

use std::fs;
use std::process::ExitCode;

use clap::Parser;

use ctt::config::{CompressConfig, OutputFormat};
use ctt::error::Error;
use ctt::format::{ColorSpace, CompressedFormat, PixelComponents, PixelFormat};
use ctt::image::{ImageLayout, RawImage};
use ctt::transform::cubemap::{CubemapInput, split_cubemap};
use ctt::transform::swizzle::{Swizzle, SwizzleChannel};

use args::{Args, ColorSpaceArg, ContainerArg, CubemapLayoutArg};

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let color_space = match args.color_space {
        ColorSpaceArg::Srgb => ColorSpace::Srgb,
        ColorSpaceArg::Linear => ColorSpace::Linear,
    };

    let format = parse_format(&args.format)?;
    let output_format = match args.container {
        ContainerArg::Dds => OutputFormat::Dds,
        ContainerArg::Ktx2 => OutputFormat::Ktx2,
    };
    let swizzle = args
        .swizzle
        .as_deref()
        .map(parse_swizzle)
        .transpose()?;

    let config = CompressConfig {
        format,
        output_format,
        swizzle,
        color_space,
    };

    // Load input images.
    let images = load_images(&args.input, color_space)?;

    // Build layout.
    let layout = if args.cubemap {
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
    fs::write(&args.output, output_bytes)?;
    Ok(())
}

fn load_images(
    paths: &[std::path::PathBuf],
    color_space: ColorSpace,
) -> Result<Vec<RawImage>, Box<dyn std::error::Error>> {
    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        let img = image::open(path)?.to_rgba8();
        let (width, height) = img.dimensions();
        let stride = width * 4;
        images.push(RawImage {
            data: img.into_raw(),
            width,
            height,
            stride,
            pixel_format: PixelFormat {
                components: PixelComponents::Rgba,
                color_space,
            },
        });
    }
    Ok(images)
}

fn build_cubemap_layout(
    images: Vec<RawImage>,
    layout_arg: CubemapLayoutArg,
) -> Result<ImageLayout, Box<dyn std::error::Error>> {
    let cubemap_input = if images.len() == 6 {
        CubemapInput::SeparateFaces(
            images
                .try_into()
                .map_err(|_| Error::CubemapFaceCount(0))?,
        )
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

fn parse_format(s: &str) -> Result<CompressedFormat, Error> {
    match s.to_lowercase().as_str() {
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
                    .ok_or_else(|| Error::UnsupportedFormat(s.into()))?;
                let block_width: u8 = w
                    .parse()
                    .map_err(|_| Error::UnsupportedFormat(s.into()))?;
                let block_height: u8 = h
                    .parse()
                    .map_err(|_| Error::UnsupportedFormat(s.into()))?;
                Ok(CompressedFormat::Astc {
                    block_width,
                    block_height,
                })
            } else {
                Err(Error::UnsupportedFormat(s.into()))
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
