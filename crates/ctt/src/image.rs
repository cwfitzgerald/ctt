use crate::format::PixelFormat;

/// A raw, uncompressed image stored as a flat pixel buffer.
#[derive(Debug, Clone)]
pub struct RawImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pixel_format: PixelFormat,
}

/// Tile an image into tightly-packed blocks for block-level encoders.
///
/// Each block is `block_w * block_h * bytes_per_pixel` bytes of contiguous pixel data.
/// Partial blocks at the image edges are padded with zeros.
pub fn tile_to_blocks(image: &RawImage, block_w: u32, block_h: u32) -> Vec<u8> {
    let bpp = image.pixel_format.bytes_per_pixel() as u32;
    let blocks_x = image.width.div_ceil(block_w);
    let blocks_y = image.height.div_ceil(block_h);
    let block_bytes = (block_w * block_h * bpp) as usize;
    let mut out = vec![0u8; (blocks_x * blocks_y) as usize * block_bytes];

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_idx = (by * blocks_x + bx) as usize;
            let block_start = block_idx * block_bytes;

            for py in 0..block_h {
                let y = by * block_h + py;
                if y >= image.height {
                    break;
                }
                for px in 0..block_w {
                    let x = bx * block_w + px;
                    if x >= image.width {
                        break;
                    }
                    let src = (y * image.stride + x * bpp) as usize;
                    let dst = block_start + ((py * block_w + px) * bpp) as usize;
                    let len = bpp as usize;
                    out[dst..dst + len].copy_from_slice(&image.data[src..src + len]);
                }
            }
        }
    }

    out
}

/// Describes the layout of one or more images for compression.
///
/// `layers[i][j]` is layer `i`, mip level `j`.
/// A 2D texture has 1 layer; a cubemap has 6.
/// Currently only a single mip level (index 0) is supported.
#[derive(Debug, Clone)]
pub struct ImageLayout {
    pub layers: Vec<Vec<RawImage>>,
    pub is_cubemap: bool,
}
