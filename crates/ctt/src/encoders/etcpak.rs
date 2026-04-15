use ctt_etcpak as ep;

use crate::encoders::{Encoder, EncoderSettings, Quality};
use crate::error::Result;
use crate::surface::Surface;
use crate::vk_format::FormatExt as _;

/// etcpak-specific encoder settings.
#[derive(Debug, Clone, Copy, Default)]
pub struct EtcpakSettings {
    /// Enable dithering for ETC1 and BC1 compression.
    pub dither: bool,
    /// Enable heuristic-based fast compression mode selection for ETC2 RGB/RGBA.
    pub use_heuristics: bool,
}

impl EncoderSettings for EtcpakSettings {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct EtcpakEncoder;

impl Encoder for EtcpakEncoder {
    fn name(&self) -> &str {
        "etcpak"
    }

    fn supported_formats(&self) -> &[ktx2::Format] {
        &[
            ktx2::Format::ETC2_R8G8B8_UNORM_BLOCK,
            ktx2::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
            ktx2::Format::EAC_R11_UNORM_BLOCK,
            ktx2::Format::EAC_R11G11_UNORM_BLOCK,
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            ktx2::Format::BC3_UNORM_BLOCK,
            ktx2::Format::BC4_UNORM_BLOCK,
            ktx2::Format::BC5_UNORM_BLOCK,
        ]
    }

    fn required_input_format(&self, format: ktx2::Format) -> ktx2::Format {
        use ktx2::Format as F;
        match format {
            // ETC/EAC codecs expect BGRA pixel layout.
            F::ETC2_R8G8B8_UNORM_BLOCK
            | F::ETC2_R8G8B8A8_UNORM_BLOCK
            | F::EAC_R11_UNORM_BLOCK
            | F::EAC_R11G11_UNORM_BLOCK => F::B8G8R8A8_UNORM,
            // BC codecs expect RGBA pixel layout.
            _ => F::R8G8B8A8_UNORM,
        }
    }

    fn compress(
        &self,
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>> {
        let (base, _) = format.normalize();
        let data = &*surface.data;
        let width = surface.width;
        let height = surface.height;

        let ep_settings = settings
            .and_then(|s| s.as_any().downcast_ref::<EtcpakSettings>())
            .copied()
            .unwrap_or_default();

        let use_heuristics = ep_settings.use_heuristics;
        let dither = ep_settings.dither;

        let ep_surface = ep::Surface::new(data, width, height);

        use ktx2::Format as F;
        match base {
            F::ETC2_R8G8B8_UNORM_BLOCK => {
                if dither {
                    Ok(ep::etc1::compress_blocks_dither(&ep_surface))
                } else {
                    match quality {
                        // ETC1 is faster, ETC2 is higher quality.
                        Quality::UltraFast | Quality::VeryFast | Quality::Fast => {
                            Ok(ep::etc1::compress_blocks(&ep_surface))
                        }
                        _ => Ok(ep::etc2_rgb::compress_blocks(&ep_surface, use_heuristics)),
                    }
                }
            }
            F::ETC2_R8G8B8A8_UNORM_BLOCK => {
                Ok(ep::etc2_rgba::compress_blocks(&ep_surface, use_heuristics))
            }
            F::EAC_R11_UNORM_BLOCK => Ok(ep::eac_r::compress_blocks(&ep_surface)),
            F::EAC_R11G11_UNORM_BLOCK => Ok(ep::eac_rg::compress_blocks(&ep_surface)),
            F::BC1_RGBA_UNORM_BLOCK => {
                if dither {
                    Ok(ep::bc1::compress_blocks_dither(&ep_surface))
                } else {
                    Ok(ep::bc1::compress_blocks(&ep_surface))
                }
            }
            F::BC3_UNORM_BLOCK => Ok(ep::bc3::compress_blocks(&ep_surface)),
            F::BC4_UNORM_BLOCK => Ok(ep::bc4::compress_blocks(&ep_surface)),
            F::BC5_UNORM_BLOCK => Ok(ep::bc5::compress_blocks(&ep_surface)),
            _ => unreachable!("format not in supported_formats()"),
        }
    }
}
