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
    /// Each block is `block_w * block_h * bytes_per_pixel` bytes of contiguous
    /// pixel data. Partial blocks at the right/bottom edges replicate the
    /// nearest edge pixel (clamp-to-edge) so encoders don't see black padding.
    ///
    /// Panics if the format is compressed or has unknown bytes-per-pixel, or
    /// if the surface is empty (width or height of 0).
    pub fn tile_to_blocks(&self, block_w: u32, block_h: u32) -> Vec<u8> {
        let bpp = self
            .format
            .bytes_per_pixel()
            .expect("tile_to_blocks requires an uncompressed format with known bpp")
            as u32;
        assert!(
            self.width > 0 && self.height > 0,
            "tile_to_blocks requires non-empty surface"
        );

        let blocks_x = self.width.div_ceil(block_w);
        let blocks_y = self.height.div_ceil(block_h);
        let block_bytes = (block_w * block_h * bpp) as usize;
        let mut out = vec![0u8; (blocks_x * blocks_y) as usize * block_bytes];

        let max_x = self.width - 1;
        let max_y = self.height - 1;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = (by * blocks_x + bx) as usize;
                let block_start = block_idx * block_bytes;

                for py in 0..block_h {
                    let y = (by * block_h + py).min(max_y);
                    for px in 0..block_w {
                        let x = (bx * block_w + px).min(max_x);
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
        // Right-padded pixel at (2,0) should replicate the pixel at (1,0) = (5,6,7,8)
        assert_eq!(&blocks[8..12], &[5, 6, 7, 8]);
        // Right-padded pixel at (3,0) should also replicate (5,6,7,8)
        assert_eq!(&blocks[12..16], &[5, 6, 7, 8]);
        // Bottom-padded row 2: replicates row 1 (pixels (9,10,11,12), (13,14,15,16), ...)
        assert_eq!(&blocks[32..36], &[9, 10, 11, 12]);
        assert_eq!(&blocks[36..40], &[13, 14, 15, 16]);
        // And the right-padded portion of that replicated row still replicates the last column.
        assert_eq!(&blocks[40..44], &[13, 14, 15, 16]);
        assert_eq!(&blocks[44..48], &[13, 14, 15, 16]);
    }

    #[test]
    fn tile_to_blocks_non_multiple_size() {
        // 3x3 RGBA8 (not a multiple of 4) — should edge-replicate into one 4x4 block.
        let mut data = Vec::new();
        for y in 0..3u8 {
            for x in 0..3u8 {
                data.extend_from_slice(&[x, y, 0, 255]);
            }
        }
        let surface = Surface {
            data,
            width: 3,
            height: 3,
            stride: 3 * 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };

        let blocks = surface.tile_to_blocks(4, 4);
        assert_eq!(blocks.len(), 64);

        // Pixel (3, 0) should replicate pixel (2, 0) = (2, 0, 0, 255).
        assert_eq!(&blocks[12..16], &[2, 0, 0, 255]);
        // Pixel (0, 3) should replicate pixel (0, 2) = (0, 2, 0, 255).
        assert_eq!(&blocks[48..52], &[0, 2, 0, 255]);
        // Corner pixel (3, 3) should replicate pixel (2, 2) = (2, 2, 0, 255).
        assert_eq!(&blocks[60..64], &[2, 2, 0, 255]);
    }
}
