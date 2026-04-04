use ctt_intel_texture_compressor as itc;

use crate::encoder::{Encoder, EncoderSettings, Quality};
use crate::error::Result;
use crate::surface::Surface;
use crate::vk_format::FormatExt as _;

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
        "intel"
    }

    fn supported_formats(&self) -> &[ktx2::Format] {
        &[
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            ktx2::Format::BC3_UNORM_BLOCK,
            ktx2::Format::BC4_UNORM_BLOCK,
            ktx2::Format::BC5_UNORM_BLOCK,
            ktx2::Format::BC6H_UFLOAT_BLOCK,
            ktx2::Format::BC7_UNORM_BLOCK,
            ktx2::Format::ETC2_R8G8B8_UNORM_BLOCK,
        ]
    }

    fn required_input_format(&self, format: ktx2::Format) -> ktx2::Format {
        use ktx2::Format as F;
        match format {
            F::BC4_UNORM_BLOCK => F::R8_UNORM,
            F::BC5_UNORM_BLOCK => F::R8G8_UNORM,
            F::BC6H_UFLOAT_BLOCK => F::R16G16B16A16_UNORM,
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
        let (data, width, height, stride) = (
            &*surface.data,
            surface.width,
            surface.height,
            surface.stride,
        );
        use ktx2::Format as F;
        match base {
            F::BC1_RGBA_UNORM_BLOCK => {
                let surface = itc::RgbaSurface::new(data, width, height, stride);
                Ok(itc::bc1::compress_blocks(&surface))
            }
            F::BC3_UNORM_BLOCK => {
                let surface = itc::RgbaSurface::new(data, width, height, stride);
                Ok(itc::bc3::compress_blocks(&surface))
            }
            F::BC4_UNORM_BLOCK => {
                let surface = itc::RSurface::new(data, width, height, stride);
                Ok(itc::bc4::compress_blocks(&surface))
            }
            F::BC5_UNORM_BLOCK => {
                let surface = itc::RgSurface::new(data, width, height, stride);
                Ok(itc::bc5::compress_blocks(&surface))
            }
            F::BC6H_UFLOAT_BLOCK => {
                let surface = itc::Rgba16Surface::new(data, width, height, stride);
                let settings = bc6h_settings(quality);
                Ok(itc::bc6h::compress_blocks(&settings, &surface))
            }
            F::BC7_UNORM_BLOCK => {
                let surface = itc::RgbaSurface::new(data, width, height, stride);
                let alpha = settings
                    .and_then(|s| s.as_any().downcast_ref::<IspcSettings>())
                    .is_some_and(|s| s.alpha);
                let settings = bc7_settings(quality, alpha);
                Ok(itc::bc7::compress_blocks(&settings, &surface))
            }
            F::ETC2_R8G8B8_UNORM_BLOCK => {
                let surface = itc::RgbaSurface::new(data, width, height, stride);
                let settings = etc1_settings();
                Ok(itc::etc1::compress_blocks(&settings, &surface))
            }
            _ => unreachable!("format not in supported_formats()"),
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
