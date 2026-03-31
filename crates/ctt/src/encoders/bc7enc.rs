use ctt_bc7enc_rdo::bc7e;

use crate::encoder::{Encoder, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::format::{ChannelType, ColorSpace, CompressedFormat, PixelComponents, PixelFormat};
use crate::image::{self, RawImage};

/// bc7enc-rdo-specific encoder settings.
#[derive(Debug, Clone, Copy)]
pub struct Bc7encSettings {
    /// Use perceptual quality metrics.
    pub perceptual: bool,
}

impl EncoderSettings for Bc7encSettings {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Default)]
pub struct Bc7encEncoder;

impl Encoder for Bc7encEncoder {
    fn name(&self) -> &str {
        "bc7e"
    }

    fn supported_formats(&self) -> &[CompressedFormat] {
        &[CompressedFormat::Bc7]
    }

    fn required_input_format(
        &self,
        _format: CompressedFormat,
        color_space: ColorSpace,
    ) -> PixelFormat {
        PixelFormat {
            components: PixelComponents::Rgba,
            channel_type: ChannelType::U8,
            color_space,
        }
    }

    fn compress(
        &self,
        raw_image: &RawImage,
        format: CompressedFormat,
        quality: Quality,
        settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>> {
        if format != CompressedFormat::Bc7 {
            return Err(Error::CompressionNotImplemented(format));
        }

        let perceptual = settings
            .and_then(|s| s.as_any().downcast_ref::<Bc7encSettings>())
            .map(|s| s.perceptual)
            .unwrap_or(true);

        let params = match quality {
            Quality::UltraFast => bc7e::params_init_ultrafast(perceptual),
            Quality::VeryFast => bc7e::params_init_veryfast(perceptual),
            Quality::Fast => bc7e::params_init_fast(perceptual),
            Quality::Basic => bc7e::params_init_basic(perceptual),
            Quality::Slow => bc7e::params_init_slow(perceptual),
            Quality::VerySlow => bc7e::params_init_veryslow(perceptual),
        };

        let pixels = image::tile_to_blocks(raw_image, 4, 4);
        let pixels: &[u32] = bytemuck::cast_slice(&pixels);
        let num_blocks = raw_image
            .width
            .div_ceil(4)
            .checked_mul(raw_image.height.div_ceil(4))
            .expect("block count overflow") as usize;
        let compressed = bc7e::compress_blocks_alloc(num_blocks, pixels, &params);
        Ok(bytemuck::cast_vec(compressed))
    }
}
