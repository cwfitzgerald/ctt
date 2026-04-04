use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::format::ColorSpace;
use crate::image::{ImageLayout, RawImage};
use crate::vk_format::FormatExt;

/// A single 2D image surface — either raw pixels or compressed blocks.
///
/// The format field determines whether the data is uncompressed pixel data or compressed block
/// data. Use [`FormatExt::is_compressed`] to check.
#[derive(Debug, Clone)]
pub struct Surface {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// For uncompressed: bytes per row.
    /// For compressed: row-of-blocks * bytes-per-block.
    pub stride: u32,
    pub format: ktx2::Format,
    pub color_space: ColorSpace,
    pub alpha: AlphaMode,
}

/// Multi-layer, multi-mip image.
///
/// `surfaces[i][j]` is layer `i`, mip level `j`.
/// A 2D texture has 1 layer; a cubemap has 6.
#[derive(Debug, Clone)]
pub struct Image {
    pub surfaces: Vec<Vec<Surface>>,
    pub is_cubemap: bool,
}

impl Surface {
    /// Convert from the legacy [`RawImage`] type.
    ///
    /// Color space is extracted from the `PixelFormat` and stored separately. Alpha defaults to
    /// [`AlphaMode::Straight`].
    pub fn from_raw_image(raw: RawImage) -> Self {
        let format = ktx2::Format::from_pixel_format(raw.pixel_format);
        let (format, color_space) = format.normalize();
        Self {
            data: raw.data,
            width: raw.width,
            height: raw.height,
            stride: raw.stride,
            format,
            color_space,
            alpha: AlphaMode::Straight,
        }
    }

    /// Convert back to the legacy [`RawImage`] type.
    ///
    /// Fails if the format cannot be represented as a [`PixelFormat`] (e.g. compressed or packed
    /// formats).
    pub fn to_raw_image(&self) -> Result<RawImage> {
        let pixel_format = vk_to_pixel_format(self.format, self.color_space)?;
        Ok(RawImage {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
            stride: self.stride,
            pixel_format,
        })
    }

    /// Tile the surface into tightly-packed blocks for block-level encoders.
    ///
    /// Each block is `block_w * block_h * bytes_per_pixel` bytes of contiguous pixel data.
    /// Partial blocks at the image edges are padded with zeros.
    ///
    /// Panics if the format is compressed or has unknown bytes-per-pixel.
    pub fn tile_to_blocks(&self, block_w: u32, block_h: u32) -> Vec<u8> {
        let bpp = self
            .format
            .bytes_per_pixel()
            .expect("tile_to_blocks requires an uncompressed format with known bpp") as u32;

        let blocks_x = self.width.div_ceil(block_w);
        let blocks_y = self.height.div_ceil(block_h);
        let block_bytes = (block_w * block_h * bpp) as usize;
        let mut out = vec![0u8; (blocks_x * blocks_y) as usize * block_bytes];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = (by * blocks_x + bx) as usize;
                let block_start = block_idx * block_bytes;

                for py in 0..block_h {
                    let y = by * block_h + py;
                    if y >= self.height {
                        break;
                    }
                    for px in 0..block_w {
                        let x = bx * block_w + px;
                        if x >= self.width {
                            break;
                        }
                        let src = (y * self.stride + x * bpp) as usize;
                        let dst = block_start + ((py * block_w + px) * bpp) as usize;
                        let len = bpp as usize;
                        out[dst..dst + len].copy_from_slice(&self.data[src..src + len]);
                    }
                }
            }
        }

        out
    }
}

impl Image {
    /// Convert from the legacy [`ImageLayout`] type.
    pub fn from_layout(layout: ImageLayout) -> Self {
        let surfaces = layout
            .layers
            .into_iter()
            .map(|mips| mips.into_iter().map(Surface::from_raw_image).collect())
            .collect();
        Self {
            surfaces,
            is_cubemap: layout.is_cubemap,
        }
    }

    /// Convert back to the legacy [`ImageLayout`] type.
    ///
    /// Fails if any surface cannot be represented as a [`RawImage`].
    pub fn to_layout(&self) -> Result<ImageLayout> {
        let mut layers = Vec::with_capacity(self.surfaces.len());
        for layer in &self.surfaces {
            let mut mips = Vec::with_capacity(layer.len());
            for surface in layer {
                mips.push(surface.to_raw_image()?);
            }
            layers.push(mips);
        }
        Ok(ImageLayout {
            layers,
            is_cubemap: self.is_cubemap,
        })
    }
}

/// Convert a `ktx2::Format` + `ColorSpace` back to the legacy `PixelFormat`.
///
/// Only supports the 16 basic uncompressed formats (`{R,RG,RGB,RGBA} x {U8,U16,F16,F32}`).
fn vk_to_pixel_format(
    format: ktx2::Format,
    color_space: ColorSpace,
) -> Result<crate::format::PixelFormat> {
    use crate::format::{ChannelType, PixelComponents, PixelFormat};
    use ktx2::Format as F;

    let (components, channel_type) = match format {
        F::R8_UNORM => (PixelComponents::R, ChannelType::U8),
        F::R16_UNORM => (PixelComponents::R, ChannelType::U16),
        F::R16_SFLOAT => (PixelComponents::R, ChannelType::F16),
        F::R32_SFLOAT => (PixelComponents::R, ChannelType::F32),

        F::R8G8_UNORM => (PixelComponents::Rg, ChannelType::U8),
        F::R16G16_UNORM => (PixelComponents::Rg, ChannelType::U16),
        F::R16G16_SFLOAT => (PixelComponents::Rg, ChannelType::F16),
        F::R32G32_SFLOAT => (PixelComponents::Rg, ChannelType::F32),

        F::R8G8B8_UNORM => (PixelComponents::Rgb, ChannelType::U8),
        F::R16G16B16_UNORM => (PixelComponents::Rgb, ChannelType::U16),
        F::R16G16B16_SFLOAT => (PixelComponents::Rgb, ChannelType::F16),
        F::R32G32B32_SFLOAT => (PixelComponents::Rgb, ChannelType::F32),

        F::R8G8B8A8_UNORM => (PixelComponents::Rgba, ChannelType::U8),
        F::R16G16B16A16_UNORM => (PixelComponents::Rgba, ChannelType::U16),
        F::R16G16B16A16_SFLOAT => (PixelComponents::Rgba, ChannelType::F16),
        F::R32G32B32A32_SFLOAT => (PixelComponents::Rgba, ChannelType::F32),

        _ => {
            return Err(Error::UnsupportedConversion(format!(
                "cannot convert {format:?} to PixelFormat"
            )));
        }
    };

    Ok(PixelFormat {
        components,
        channel_type,
        color_space,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ChannelType, PixelComponents, PixelFormat};

    #[test]
    fn raw_image_roundtrip() {
        let raw = RawImage {
            data: vec![10, 20, 30, 255],
            width: 1,
            height: 1,
            stride: 4,
            pixel_format: PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U8,
                color_space: ColorSpace::Linear,
            },
        };

        let surface = Surface::from_raw_image(raw.clone());
        assert_eq!(surface.format, ktx2::Format::R8G8B8A8_UNORM);
        assert_eq!(surface.color_space, ColorSpace::Linear);

        let back = surface.to_raw_image().unwrap();
        assert_eq!(back.data, raw.data);
        assert_eq!(back.pixel_format, raw.pixel_format);
    }

    #[test]
    fn raw_image_srgb_roundtrip() {
        let raw = RawImage {
            data: vec![10, 20, 30, 255],
            width: 1,
            height: 1,
            stride: 4,
            pixel_format: PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U8,
                color_space: ColorSpace::Srgb,
            },
        };

        let surface = Surface::from_raw_image(raw.clone());
        assert_eq!(surface.format, ktx2::Format::R8G8B8A8_UNORM);
        assert_eq!(surface.color_space, ColorSpace::Srgb);

        let back = surface.to_raw_image().unwrap();
        assert_eq!(back.pixel_format.color_space, ColorSpace::Srgb);
    }

    #[test]
    fn layout_roundtrip() {
        let raw = RawImage {
            data: vec![1, 2, 3, 4],
            width: 1,
            height: 1,
            stride: 4,
            pixel_format: PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U8,
                color_space: ColorSpace::Linear,
            },
        };
        let layout = ImageLayout {
            layers: vec![vec![raw]],
            is_cubemap: false,
        };

        let image = Image::from_layout(layout);
        assert_eq!(image.surfaces.len(), 1);
        assert_eq!(image.surfaces[0].len(), 1);
        assert!(!image.is_cubemap);

        let back = image.to_layout().unwrap();
        assert_eq!(back.layers[0][0].data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn tile_to_blocks_basic() {
        // 2x2 RGBA8 image, tile into 4x4 blocks (padded)
        let surface = Surface {
            data: vec![
                1, 2, 3, 4, 5, 6, 7, 8, // row 0
                9, 10, 11, 12, 13, 14, 15, 16, // row 1
            ],
            width: 2,
            height: 2,
            stride: 8,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };

        let blocks = surface.tile_to_blocks(4, 4);
        // 1 block of 4x4 pixels, 4 bytes each = 64 bytes
        assert_eq!(blocks.len(), 64);
        // First pixel should be (1,2,3,4)
        assert_eq!(&blocks[0..4], &[1, 2, 3, 4]);
        // Second pixel should be (5,6,7,8)
        assert_eq!(&blocks[4..8], &[5, 6, 7, 8]);
    }
}
