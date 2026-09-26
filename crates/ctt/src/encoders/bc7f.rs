use ctt_bc7f::bindings::*;

use crate::encoders::backend::Encoder;
use crate::error::{Error, Result};
use crate::quality::Quality;
use crate::surface::Surface;
use crate::vk_format::FormatExt;

/// BC7F options applied to the quality preset.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bc7fSettings {
    /// Restrict partitions to patterns shared with ASTC LDR 4x4.
    pub astc_compatible: bool,
    /// Disable separate RGB planes. A separate alpha plane remains available.
    pub disable_rgb_dual_plane: bool,
}

/// BC7F encoder for RGBA8 surfaces.
#[derive(Default)]
pub struct Bc7fEncoder;

impl Encoder for Bc7fEncoder {
    type Settings = Bc7fSettings;

    fn name() -> &'static str {
        "bc7f"
    }

    fn supported_formats() -> &'static [ktx2::Format] {
        &[ktx2::Format::BC7_UNORM_BLOCK, ktx2::Format::BC7_SRGB_BLOCK]
    }

    fn required_input_format(_format: ktx2::Format, _settings: &Bc7fSettings) -> ktx2::Format {
        ktx2::Format::R8G8B8A8_UNORM
    }

    fn compress(
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: &Bc7fSettings,
    ) -> Result<Vec<u8>> {
        if format.normalize().0 != ktx2::Format::BC7_UNORM_BLOCK {
            return Err(Error::UnsupportedFormat(format!("{format:?}")));
        }
        let mut flags = match quality {
            Quality::UltraFast => cPackBC7FlagDefaultFastest,
            Quality::VeryFast => cPackBC7FlagDefaultFaster,
            Quality::Fast => cPackBC7FlagDefaultFast,
            Quality::Basic => cPackBC7FlagDefault,
            Quality::Slow => cPackBC7FlagDefaultPartiallyAnalytical,
            Quality::VerySlow => cPackBC7FlagDefaultNonAnalytical,
        };
        if settings.astc_compatible {
            flags |= cPackBC7FlagASTCCompatible;
        }
        if settings.disable_rgb_dual_plane {
            flags |= cPackBC7FlagDisableRGBDualPlane;
        }
        let pixels = surface.tile_to_blocks(4, 4);
        let blocks_x = surface.width.div_ceil(4) as usize;
        let mut output = vec![0; pixels.len() / 64 * 16];
        crate::encoders::parallel::for_each_row_chunk(
            &mut output,
            blocks_x * 16,
            |start_row, row_count, output| {
                let start = start_row * blocks_x * 64;
                let end = start + row_count * blocks_x * 64;
                ctt_bc7f::compress_blocks_into(&pixels[start..end], output, flags);
            },
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    fn solid_surface(alpha: u8) -> Surface {
        let mut data = vec![0xAB; 40 * 5];
        for y in 0..5 {
            for x in 0..7 {
                data[y * 40 + x * 4..y * 40 + x * 4 + 4].copy_from_slice(&[27, 129, 213, alpha]);
            }
        }
        Surface {
            data,
            width: 7,
            height: 5,
            depth: 1,
            stride: 40,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        }
    }

    #[cfg(feature = "encoder-etcpak")]
    #[test]
    fn all_presets_preserve_color_alpha_and_edges() {
        for alpha in [0, 79, 255] {
            let surface = solid_surface(alpha);
            for quality in [
                Quality::UltraFast,
                Quality::VeryFast,
                Quality::Fast,
                Quality::Basic,
                Quality::Slow,
                Quality::VerySlow,
            ] {
                let encoded = Bc7fEncoder::compress(
                    &surface,
                    ktx2::Format::BC7_UNORM_BLOCK,
                    quality,
                    &Bc7fSettings::default(),
                )
                .unwrap();
                assert_eq!(encoded.len(), 64);
                let decoded = ctt_etcpak::decode::decode_bc7(&encoded, 8, 8);
                for pixel in decoded.chunks_exact(4) {
                    for (actual, expected) in pixel.iter().copied().zip([27, 129, 213, alpha]) {
                        assert!(
                            actual.abs_diff(expected) <= 8,
                            "{quality:?}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_other_formats() {
        assert!(
            Bc7fEncoder::compress(
                &solid_surface(255),
                ktx2::Format::BC1_RGB_UNORM_BLOCK,
                Quality::Basic,
                &Bc7fSettings::default()
            )
            .is_err()
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn parallel_matches_serial() {
        let mut surface = solid_surface(79);
        for (i, byte) in surface.data.iter_mut().enumerate() {
            *byte = (i * 37) as u8;
        }
        crate::encoders::assert_parallel_matches_serial(|| {
            Bc7fEncoder::compress(
                &surface,
                ktx2::Format::BC7_SRGB_BLOCK,
                Quality::Slow,
                &Bc7fSettings::default(),
            )
            .unwrap()
        });
    }
}
