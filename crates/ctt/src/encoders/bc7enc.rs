use crate::encoders::Quality;
use crate::encoders::backend::Encoder;
use crate::error::{Error, Result};
use crate::surface::Surface;

/// bc7enc-rdo-specific encoder settings.
#[derive(Debug, Clone, Copy)]
pub struct Bc7encSettings {
    /// Use perceptual quality metrics.
    pub perceptual: bool,
}

impl Default for Bc7encSettings {
    fn default() -> Self {
        Self { perceptual: true }
    }
}

#[derive(Default)]
pub struct Bc7encEncoder;

impl Encoder for Bc7encEncoder {
    type Settings = Bc7encSettings;

    fn name() -> &'static str {
        "bc7e"
    }

    fn supported_formats() -> &'static [ktx2::Format] {
        &[ktx2::Format::BC7_UNORM_BLOCK]
    }

    fn required_input_format(_format: ktx2::Format) -> ktx2::Format {
        ktx2::Format::R8G8B8A8_UNORM
    }

    fn compress(
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: &Bc7encSettings,
    ) -> Result<Vec<u8>> {
        let (base, _) = crate::vk_format::FormatExt::normalize(&format);
        if base != ktx2::Format::BC7_UNORM_BLOCK {
            return Err(Error::UnsupportedFormat(format!("{format:?}")));
        }

        let perceptual = settings.perceptual;

        let params = match quality {
            Quality::UltraFast => ctt_bc7enc_rdo::params_init_ultrafast(perceptual),
            Quality::VeryFast => ctt_bc7enc_rdo::params_init_veryfast(perceptual),
            Quality::Fast => ctt_bc7enc_rdo::params_init_fast(perceptual),
            Quality::Basic => ctt_bc7enc_rdo::params_init_basic(perceptual),
            Quality::Slow => ctt_bc7enc_rdo::params_init_slow(perceptual),
            Quality::VerySlow => ctt_bc7enc_rdo::params_init_veryslow(perceptual),
        };

        let pixels = surface.tile_to_blocks(4, 4);
        let pixels: &[u32] = bytemuck::cast_slice(&pixels);
        let num_blocks = surface
            .width
            .div_ceil(4)
            .checked_mul(surface.height.div_ceil(4))
            .expect("block count overflow") as usize;
        let compressed = ctt_bc7enc_rdo::compress_blocks_alloc(num_blocks, pixels, &params);
        Ok(bytemuck::cast_slice(&compressed).to_vec())
    }
}

#[cfg(all(test, feature = "encoder-amd"))]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    fn solid_red(width: u32, height: u32) -> Surface {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            data.extend_from_slice(&[255, 0, 0, 255]);
        }
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride: width * 4,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    #[test]
    fn bc7_non_aligned_5x5_edges_replicate() {
        let surface = solid_red(5, 5);
        let out = Bc7encEncoder::compress(
            &surface,
            ktx2::Format::BC7_UNORM_BLOCK,
            Quality::UltraFast,
            &Bc7encSettings::default(),
        )
        .unwrap();
        // 5×5 → 8×8 → 4 blocks × 16 bytes.
        assert_eq!(out.len(), 4 * 16);
        // Every block (including the edge-replicated ones) should decode near red.
        for chunk in out.chunks_exact(16) {
            let block: [u8; 16] = chunk.try_into().unwrap();
            let decoded = ctt_compressonator::bc7::decompress_block(&block).unwrap();
            for pixel in decoded.chunks_exact(4) {
                assert!(pixel[0] > 200, "bc7enc-rdo edge R={}", pixel[0]);
            }
        }
    }
}
