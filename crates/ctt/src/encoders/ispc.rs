use ctt_intel_texture_compressor as itc;

use crate::encoder::{Encoder, EncoderSettings, Quality};
use crate::error::Result;
use crate::format::{ChannelType, ColorSpace, CompressedFormat, PixelComponents, PixelFormat};
use crate::image::RawImage;

/// ISPC-specific encoder settings.
#[derive(Debug, Clone, Copy)]
pub struct IspcSettings {
    /// Whether to encode alpha channel (for BC7).
    pub alpha: bool,
}

impl EncoderSettings for IspcSettings {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct IspcEncoder;

impl Encoder for IspcEncoder {
    fn name(&self) -> &str {
        "ispc"
    }

    fn supported_formats(&self) -> &[CompressedFormat] {
        &[
            CompressedFormat::Bc1,
            CompressedFormat::Bc3,
            CompressedFormat::Bc4,
            CompressedFormat::Bc5,
            CompressedFormat::Bc6h,
            CompressedFormat::Bc7,
            CompressedFormat::Etc1,
        ]
    }

    fn required_input_format(
        &self,
        format: CompressedFormat,
        color_space: ColorSpace,
    ) -> PixelFormat {
        match format {
            CompressedFormat::Bc1
            | CompressedFormat::Bc3
            | CompressedFormat::Bc7
            | CompressedFormat::Etc1 => PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U8,
                color_space,
            },
            CompressedFormat::Bc4 => PixelFormat {
                components: PixelComponents::R,
                channel_type: ChannelType::U8,
                color_space,
            },
            CompressedFormat::Bc5 => PixelFormat {
                components: PixelComponents::Rg,
                channel_type: ChannelType::U8,
                color_space,
            },
            CompressedFormat::Bc6h => PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U16,
                color_space,
            },
            CompressedFormat::Astc { .. } => unreachable!("ASTC not in supported_formats()"),
        }
    }

    fn compress(
        &self,
        image: &RawImage,
        format: CompressedFormat,
        quality: Quality,
        settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>> {
        match format {
            CompressedFormat::Bc1 => {
                let surface =
                    itc::RgbaSurface::new(&image.data, image.width, image.height, image.stride);
                Ok(itc::bc1::compress_blocks(&surface))
            }
            CompressedFormat::Bc3 => {
                let surface =
                    itc::RgbaSurface::new(&image.data, image.width, image.height, image.stride);
                Ok(itc::bc3::compress_blocks(&surface))
            }
            CompressedFormat::Bc4 => {
                let surface =
                    itc::RSurface::new(&image.data, image.width, image.height, image.stride);
                Ok(itc::bc4::compress_blocks(&surface))
            }
            CompressedFormat::Bc5 => {
                let surface =
                    itc::RgSurface::new(&image.data, image.width, image.height, image.stride);
                Ok(itc::bc5::compress_blocks(&surface))
            }
            CompressedFormat::Bc6h => {
                let surface =
                    itc::Rgba16Surface::new(&image.data, image.width, image.height, image.stride);
                let settings = bc6h_settings(quality);
                Ok(itc::bc6h::compress_blocks(&settings, &surface))
            }
            CompressedFormat::Bc7 => {
                let surface =
                    itc::RgbaSurface::new(&image.data, image.width, image.height, image.stride);
                let alpha = settings
                    .and_then(|s| s.as_any().downcast_ref::<IspcSettings>())
                    .is_some_and(|s| s.alpha);
                let settings = bc7_settings(quality, alpha);
                Ok(itc::bc7::compress_blocks(&settings, &surface))
            }
            CompressedFormat::Etc1 => {
                let surface =
                    itc::RgbaSurface::new(&image.data, image.width, image.height, image.stride);
                let settings = etc1_settings();
                Ok(itc::etc1::compress_blocks(&settings, &surface))
            }
            CompressedFormat::Astc { .. } => unreachable!("ASTC not in supported_formats()"),
        }
    }
}

fn bc6h_settings(quality: Quality) -> itc::bc6h::EncodeSettings {
    match quality {
        Quality::UltraFast | Quality::VeryFast => itc::bc6h::very_fast_settings(),
        Quality::Fast => itc::bc6h::fast_settings(),
        Quality::Basic => itc::bc6h::basic_settings(),
        Quality::Slow => itc::bc6h::slow_settings(),
        Quality::VerySlow => itc::bc6h::very_slow_settings(),
    }
}

fn bc7_settings(quality: Quality, alpha: bool) -> itc::bc7::EncodeSettings {
    match (alpha, quality) {
        (false, Quality::UltraFast) => itc::bc7::opaque_ultra_fast_settings(),
        (false, Quality::VeryFast) => itc::bc7::opaque_very_fast_settings(),
        (false, Quality::Fast) => itc::bc7::opaque_fast_settings(),
        (false, Quality::Basic) => itc::bc7::opaque_basic_settings(),
        (false, Quality::Slow | Quality::VerySlow) => itc::bc7::opaque_slow_settings(),
        (true, Quality::UltraFast) => itc::bc7::alpha_ultra_fast_settings(),
        (true, Quality::VeryFast) => itc::bc7::alpha_very_fast_settings(),
        (true, Quality::Fast) => itc::bc7::alpha_fast_settings(),
        (true, Quality::Basic) => itc::bc7::alpha_basic_settings(),
        (true, Quality::Slow | Quality::VerySlow) => itc::bc7::alpha_slow_settings(),
    }
}

fn etc1_settings() -> itc::etc1::EncodeSettings {
    itc::etc1::slow_settings()
}
