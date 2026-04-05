use std::fmt;

use crate::alpha::AlphaMode;
use crate::vk_format::FormatExt;

/// Color space metadata for a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum ColorSpace {
    #[default]
    Srgb,
    Linear,
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Srgb => f.write_str("srgb"),
            Self::Linear => f.write_str("linear"),
        }
    }
}

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
    /// Tile the surface into tightly-packed blocks for block-level encoders.
    ///
    /// Each block is `block_w * block_h * bytes_per_pixel` bytes of contiguous pixel data.
    /// Partial blocks at the image edges are padded with zeros.
    ///
    /// Panics if the format is compressed or has unknown bytes-per-pixel.
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
            .expect("tile_to_blocks requires an uncompressed format with known bpp")
            as u32;

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

#[cfg(test)]
mod tests {
    use super::*;

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
