use crate::config::{Bc6hQuality, Bc7Quality, Bc7Settings, EncodeSettings, Etc1Quality};
use crate::error::{Error, Result};
use crate::format::{ColorSpace, CompressedFormat};
use crate::image::{ImageLayout, RawImage};

use ctt_intel_texture_compressor as itc;

/// The compressed output for a single image.
#[derive(Debug, Clone)]
pub struct CompressedData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: CompressedFormat,
}

/// A full compressed texture, potentially with multiple layers and mip levels.
#[derive(Debug, Clone)]
pub struct CompressedTexture {
    pub layers: Vec<Vec<CompressedData>>,
    pub is_cubemap: bool,
    pub color_space: ColorSpace,
}

/// Compress a single raw image into the given format.
///
/// The image must already be in the format expected by the compressor
/// (see [`CompressedFormat::required_input_format`]).
pub fn compress(
    image: &RawImage,
    format: CompressedFormat,
    settings: &EncodeSettings,
) -> Result<CompressedData> {
    let data = match (format, settings) {
        (CompressedFormat::Bc1, _) => {
            let surface =
                itc::RgbaSurface::new(&image.data, image.width, image.height, image.stride);
            itc::bc1::compress_blocks(&surface)
        }
        (CompressedFormat::Bc3, _) => {
            let surface =
                itc::RgbaSurface::new(&image.data, image.width, image.height, image.stride);
            itc::bc3::compress_blocks(&surface)
        }
        (CompressedFormat::Bc4, _) => {
            let surface = itc::RSurface::new(&image.data, image.width, image.height, image.stride);
            itc::bc4::compress_blocks(&surface)
        }
        (CompressedFormat::Bc5, _) => {
            let surface = itc::RgSurface::new(&image.data, image.width, image.height, image.stride);
            itc::bc5::compress_blocks(&surface)
        }
        (CompressedFormat::Bc6h, EncodeSettings::Bc6h(quality)) => {
            let surface =
                itc::Rgba16Surface::new(&image.data, image.width, image.height, image.stride);
            let settings = bc6h_settings(*quality);
            itc::bc6h::compress_blocks(&settings, &surface)
        }
        (CompressedFormat::Bc7, EncodeSettings::Bc7(bc7)) => {
            let surface =
                itc::RgbaSurface::new(&image.data, image.width, image.height, image.stride);
            let settings = bc7_settings(*bc7);
            itc::bc7::compress_blocks(&settings, &surface)
        }
        (CompressedFormat::Etc1, EncodeSettings::Etc1(quality)) => {
            let surface =
                itc::RgbaSurface::new(&image.data, image.width, image.height, image.stride);
            let settings = etc1_settings(*quality);
            itc::etc1::compress_blocks(&settings, &surface)
        }
        (CompressedFormat::Astc { .. }, _) => {
            return Err(Error::CompressionNotImplemented(format));
        }
        // Mismatched settings/format — fall back to defaults.
        _ => return compress(image, format, &EncodeSettings::default_for(format)),
    };

    Ok(CompressedData {
        data,
        width: image.width,
        height: image.height,
        format,
    })
}

/// Compress all layers and mip levels of an image layout.
pub fn compress_layout(
    layout: &ImageLayout,
    format: CompressedFormat,
    color_space: ColorSpace,
    settings: &EncodeSettings,
) -> Result<CompressedTexture> {
    let mut layers = Vec::with_capacity(layout.layers.len());
    for layer in &layout.layers {
        let mut mips = Vec::with_capacity(layer.len());
        for image in layer {
            mips.push(compress(image, format, settings)?);
        }
        layers.push(mips);
    }
    Ok(CompressedTexture {
        layers,
        is_cubemap: layout.is_cubemap,
        color_space,
    })
}

fn bc6h_settings(quality: Bc6hQuality) -> itc::bc6h::EncodeSettings {
    match quality {
        Bc6hQuality::VeryFast => itc::bc6h::very_fast_settings(),
        Bc6hQuality::Fast => itc::bc6h::fast_settings(),
        Bc6hQuality::Basic => itc::bc6h::basic_settings(),
        Bc6hQuality::Slow => itc::bc6h::slow_settings(),
        Bc6hQuality::VerySlow => itc::bc6h::very_slow_settings(),
    }
}

fn bc7_settings(s: Bc7Settings) -> itc::bc7::EncodeSettings {
    match (s.alpha, s.quality) {
        (false, Bc7Quality::UltraFast) => itc::bc7::opaque_ultra_fast_settings(),
        (false, Bc7Quality::VeryFast) => itc::bc7::opaque_very_fast_settings(),
        (false, Bc7Quality::Fast) => itc::bc7::opaque_fast_settings(),
        (false, Bc7Quality::Basic) => itc::bc7::opaque_basic_settings(),
        (false, Bc7Quality::Slow) => itc::bc7::opaque_slow_settings(),
        (true, Bc7Quality::UltraFast) => itc::bc7::alpha_ultra_fast_settings(),
        (true, Bc7Quality::VeryFast) => itc::bc7::alpha_very_fast_settings(),
        (true, Bc7Quality::Fast) => itc::bc7::alpha_fast_settings(),
        (true, Bc7Quality::Basic) => itc::bc7::alpha_basic_settings(),
        (true, Bc7Quality::Slow) => itc::bc7::alpha_slow_settings(),
    }
}

fn etc1_settings(quality: Etc1Quality) -> itc::etc1::EncodeSettings {
    match quality {
        Etc1Quality::Slow => itc::etc1::slow_settings(),
    }
}
