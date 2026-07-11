use std::borrow::Cow;

use crate::encoders::Quality;
use crate::encoders::backend::Encoder;
use crate::error::{Error, Result};
use crate::surface::Surface;

/// bc7enc-rdo encoder settings.
///
/// The wrapper picks a `params_init_*` preset from the pipeline's [`Quality`]
/// (UltraFast → mode-6 only, VerySlow → full search). Every field here
/// either toggles an orthogonal codec feature or overrides one field of the
/// preset; everything not overridden inherits whatever the preset set.
#[derive(Debug, Clone, Copy)]
pub struct Bc7encSettings {
    /// Use perceptual error metrics (BT.601-style luminance weighting)
    /// instead of uniform RGB error. Reduces banding in mid-tones at the
    /// cost of small color shifts — leave on for photographic / albedo
    /// content, turn off for normal maps, mask data, or anywhere the
    /// channels carry non-color signals.
    ///
    /// Defaults to `true`. Drives the preset's default channel weighting:
    /// perceptual → `[128, 64, 16, 256]`, non-perceptual → `[1, 1, 1, 1]`.
    pub perceptual: bool,

    /// Restrict the encoder to BC7 mode 6 only — RGBA, no partitions,
    /// single-subset. Mode 6 is the cheapest mode to encode, so this gives
    /// the fastest possible BC7. Quality drops noticeably on content with
    /// strong gradients or sharp two-color regions (where partitioned
    /// modes 1–3 / 7 excel).
    ///
    /// Defaults to `false`. The UltraFast quality preset already enables
    /// mode-6-only mode, so toggling this on Basic/Slow is the way to ask
    /// for that speed without dropping the rest of the quality dial.
    pub mode6_only: bool,

    /// Override the preset's parity-bit (pbit) search.
    ///
    /// BC7 endpoints carry 1–2 pbits to extend their precision. Searching
    /// them costs a small constant per block and recovers a fraction of a
    /// dB at the high end. The presets enable it from Slow upward; set
    /// `Some(true)` to force it on for faster presets, `Some(false)` to
    /// suppress it on slow presets, or `None` to keep the preset's choice.
    pub pbit_search: Option<bool>,

    /// Override the preset's "uber" refinement level (`0..=4`).
    ///
    /// Higher values run extra refinement passes on blocks whose initial
    /// fit exceeded the codec's internal error threshold, trading encode
    /// time for a small quality lift. Preset defaults: `0` for Default
    /// through Fast, `1` for Basic, `2` for VerySlow, `4` for the
    /// internal "slowest" preset (not currently used here). `None` keeps
    /// the preset's choice. Values above 4 are clamped on the codec side.
    pub uber_level: Option<u32>,

    /// Custom per-channel error weights `[r, g, b, a]`.
    ///
    /// Higher values make the encoder spend more bits on that channel. The
    /// presets pre-load these from `perceptual` (see field docs) — set
    /// this only when you need a non-standard prior, e.g. `[1, 1, 1, 4]`
    /// for an alpha-critical asset, or `[2, 2, 1, 1]` to favor the red
    /// channel on data textures. `None` keeps the preset's weights.
    pub channel_weights: Option<[u32; 4]>,
}

impl Default for Bc7encSettings {
    fn default() -> Self {
        Self {
            perceptual: true,
            mode6_only: false,
            pbit_search: None,
            uber_level: None,
            channel_weights: None,
        }
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

    fn required_input_format(_format: ktx2::Format, _settings: &Bc7encSettings) -> ktx2::Format {
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

        let mut params = match quality {
            Quality::UltraFast => ctt_bc7enc_rdo::params_init_ultrafast(perceptual),
            Quality::VeryFast => ctt_bc7enc_rdo::params_init_veryfast(perceptual),
            Quality::Fast => ctt_bc7enc_rdo::params_init_fast(perceptual),
            Quality::Basic => ctt_bc7enc_rdo::params_init_basic(perceptual),
            Quality::Slow => ctt_bc7enc_rdo::params_init_slow(perceptual),
            Quality::VerySlow => ctt_bc7enc_rdo::params_init_veryslow(perceptual),
        };

        if settings.mode6_only {
            params.m_mode6_only = true;
        }
        if let Some(pbit) = settings.pbit_search {
            params.m_pbit_search = pbit;
        }
        if let Some(level) = settings.uber_level {
            params.m_uber_level = level;
        }
        if let Some(weights) = settings.channel_weights {
            params.m_weights = weights;
        }

        let pixels = surface.tile_to_blocks(4, 4);
        // `tile_to_blocks` returns a fresh `Vec<u8>` with no alignment guarantee
        // for `u32`; fall back to a realigning copy on the (rare) under-aligned
        // allocation instead of panicking in `cast_slice`.
        let pixels: Cow<[u32]> = match bytemuck::try_cast_slice(&pixels) {
            Ok(slice) => Cow::Borrowed(slice),
            Err(_) => Cow::Owned(bytemuck::pod_collect_to_vec(&pixels)),
        };
        let num_blocks = surface
            .width
            .div_ceil(4)
            .checked_mul(surface.height.div_ceil(4))
            .expect("block count overflow") as usize;
        let compressed = ctt_bc7enc_rdo::compress_blocks_alloc(num_blocks, &pixels, &params);
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

    #[test]
    fn mode6_only_still_produces_valid_output() {
        let surface = solid_red(4, 4);
        let out = Bc7encEncoder::compress(
            &surface,
            ktx2::Format::BC7_UNORM_BLOCK,
            Quality::Slow,
            &Bc7encSettings {
                mode6_only: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out.len(), 16);
        let block: [u8; 16] = out.as_slice().try_into().unwrap();
        let decoded = ctt_compressonator::bc7::decompress_block(&block).unwrap();
        for pixel in decoded.chunks_exact(4) {
            assert!(pixel[0] > 200, "mode6-only red recovers: R={}", pixel[0]);
        }
    }
}
