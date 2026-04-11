use ctt_bc7enc_rdo::bc7e;

use crate::encoder::{Encoder, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::surface::Surface;

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

    fn supported_formats(&self) -> &[ktx2::Format] {
        &[ktx2::Format::BC7_UNORM_BLOCK]
    }

    fn required_input_format(&self, _format: ktx2::Format) -> ktx2::Format {
        ktx2::Format::R8G8B8A8_UNORM
    }

    fn compress(
        &self,
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>> {
        let (base, _) = crate::vk_format::FormatExt::normalize(&format);
        if base != ktx2::Format::BC7_UNORM_BLOCK {
            return Err(Error::UnsupportedFormat(format!("{format:?}")));
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

        let pixels = surface.tile_to_blocks(4, 4);
        let pixels: &[u32] = bytemuck::cast_slice(&pixels);
        let num_blocks = surface
            .width
            .div_ceil(4)
            .checked_mul(surface.height.div_ceil(4))
            .expect("block count overflow") as usize;
        let compressed = bc7e::compress_blocks_alloc(num_blocks, pixels, &params);
        Ok(bytemuck::cast_slice(&compressed).to_vec())
    }
}
