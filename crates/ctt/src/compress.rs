use crate::encoder::{Encoder, EncoderSettings, Quality};
use crate::error::Result;
use crate::format::{ColorSpace, CompressedFormat};
use crate::image::{ImageLayout, RawImage};

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

/// Compress a single raw image using the given encoder.
pub fn compress(
    encoder: &dyn Encoder,
    image: &RawImage,
    format: CompressedFormat,
    quality: Quality,
    settings: Option<&dyn EncoderSettings>,
) -> Result<CompressedData> {
    let data = encoder.compress(image, format, quality, settings)?;
    Ok(CompressedData {
        data,
        width: image.width,
        height: image.height,
        format,
    })
}

/// Compress all layers and mip levels of an image layout.
pub fn compress_layout(
    encoder: &dyn Encoder,
    layout: &ImageLayout,
    format: CompressedFormat,
    color_space: ColorSpace,
    quality: Quality,
    settings: Option<&dyn EncoderSettings>,
) -> Result<CompressedTexture> {
    let mut layers = Vec::with_capacity(layout.layers.len());
    for (layer_idx, layer) in layout.layers.iter().enumerate() {
        let mut mips = Vec::with_capacity(layer.len());
        for (mip_idx, image) in layer.iter().enumerate() {
            log::debug!(
                "Compressing layer {layer_idx}, mip {mip_idx}: {}x{} to {format:?} using {}",
                image.width,
                image.height,
                encoder.name(),
            );
            mips.push(compress(encoder, image, format, quality, settings)?);
        }
        layers.push(mips);
    }
    Ok(CompressedTexture {
        layers,
        is_cubemap: layout.is_cubemap,
        color_space,
    })
}
