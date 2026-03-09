use crate::error::{Error, Result};
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

/// Compress a single raw image into the given format.
///
/// Currently returns [`Error::CompressionNotImplemented`] for all formats.
pub fn compress(_image: &RawImage, format: CompressedFormat) -> Result<CompressedData> {
    Err(Error::CompressionNotImplemented(format))
}

/// Compress all layers and mip levels of an image layout.
pub fn compress_layout(
    layout: &ImageLayout,
    format: CompressedFormat,
    color_space: ColorSpace,
) -> Result<CompressedTexture> {
    let mut layers = Vec::with_capacity(layout.layers.len());
    for layer in &layout.layers {
        let mut mips = Vec::with_capacity(layer.len());
        for image in layer {
            mips.push(compress(image, format)?);
        }
        layers.push(mips);
    }
    Ok(CompressedTexture {
        layers,
        is_cubemap: layout.is_cubemap,
        color_space,
    })
}
