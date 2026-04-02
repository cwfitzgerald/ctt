use ctt_compressonator as cmp;

use crate::encoder::{Encoder, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::format::{ChannelType, ColorSpace, CompressedFormat, PixelComponents, PixelFormat};
use crate::image::RawImage;

pub struct CompressonatorEncoder;

impl Encoder for CompressonatorEncoder {
    fn name(&self) -> &str {
        "amd"
    }

    fn supported_formats(&self) -> &[CompressedFormat] {
        &[
            CompressedFormat::Bc1,
            CompressedFormat::Bc2,
            CompressedFormat::Bc3,
            CompressedFormat::Bc4,
            CompressedFormat::Bc4s,
            CompressedFormat::Bc5,
            CompressedFormat::Bc5s,
            CompressedFormat::Bc6h,
            CompressedFormat::Bc6hSf,
            CompressedFormat::Bc7,
        ]
    }

    fn required_input_format(
        &self,
        format: CompressedFormat,
        color_space: ColorSpace,
    ) -> PixelFormat {
        match format {
            CompressedFormat::Bc1
            | CompressedFormat::Bc2
            | CompressedFormat::Bc3
            | CompressedFormat::Bc7 => PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U8,
                color_space,
            },
            CompressedFormat::Bc4 | CompressedFormat::Bc4s => PixelFormat {
                components: PixelComponents::R,
                channel_type: ChannelType::U8,
                color_space,
            },
            CompressedFormat::Bc5 | CompressedFormat::Bc5s => PixelFormat {
                components: PixelComponents::Rg,
                channel_type: ChannelType::U8,
                color_space,
            },
            CompressedFormat::Bc6h | CompressedFormat::Bc6hSf => PixelFormat {
                components: PixelComponents::Rgb,
                channel_type: ChannelType::F16,
                color_space,
            },
            _ => unreachable!("format not in supported_formats()"),
        }
    }

    fn compress(
        &self,
        image: &RawImage,
        format: CompressedFormat,
        quality: Quality,
        _settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>> {
        let q = quality_to_float(quality);

        match format {
            CompressedFormat::Bc1 => {
                let mut opts = cmp::bc1::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc1::compress_blocks(&image.data, image.width, image.height, &opts)
                    .map_err(cmp_err)
            }
            CompressedFormat::Bc2 => {
                let mut opts = cmp::bc2::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc2::compress_blocks(&image.data, image.width, image.height, &opts)
                    .map_err(cmp_err)
            }
            CompressedFormat::Bc3 => {
                let mut opts = cmp::bc3::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc3::compress_blocks(&image.data, image.width, image.height, &opts)
                    .map_err(cmp_err)
            }
            CompressedFormat::Bc4 => {
                let mut opts = cmp::bc4::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc4::compress_blocks(&image.data, image.width, image.height, &opts)
                    .map_err(cmp_err)
            }
            CompressedFormat::Bc4s => {
                let mut opts = cmp::bc4::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let src: &[i8] = bytemuck::cast_slice(&image.data);
                cmp::bc4s::compress_blocks(src, image.width, image.height, &opts).map_err(cmp_err)
            }
            CompressedFormat::Bc5 => {
                let mut opts = cmp::bc5::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc5::compress_blocks(&image.data, image.width, image.height, &opts)
                    .map_err(cmp_err)
            }
            CompressedFormat::Bc5s => {
                let mut opts = cmp::bc5::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let src: &[i8] = bytemuck::cast_slice(&image.data);
                cmp::bc5s::compress_blocks(src, image.width, image.height, &opts).map_err(cmp_err)
            }
            CompressedFormat::Bc6h => {
                let mut opts = cmp::bc6h::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let src: &[u16] = bytemuck::cast_slice(&image.data);
                cmp::bc6h::compress_blocks(src, image.width, image.height, &opts).map_err(cmp_err)
            }
            CompressedFormat::Bc6hSf => {
                let mut opts = cmp::bc6h::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                opts.set_signed(true).map_err(cmp_err)?;
                let src: &[u16] = bytemuck::cast_slice(&image.data);
                cmp::bc6h::compress_blocks(src, image.width, image.height, &opts).map_err(cmp_err)
            }
            CompressedFormat::Bc7 => {
                let mut opts = cmp::bc7::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc7::compress_blocks(&image.data, image.width, image.height, &opts)
                    .map_err(cmp_err)
            }
            _ => unreachable!("format not in supported_formats()"),
        }
    }
}

fn quality_to_float(quality: Quality) -> f32 {
    match quality {
        Quality::UltraFast => 0.01,
        Quality::VeryFast => 0.05,
        Quality::Fast => 0.1,
        Quality::Basic => 0.5,
        Quality::Slow => 0.8,
        Quality::VerySlow => 1.0,
    }
}

fn cmp_err(e: cmp::Error) -> Error {
    Error::Compression(e.to_string())
}
