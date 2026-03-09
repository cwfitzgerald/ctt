//! BC1 (DXT1) block compression — RGB, no alpha.
//!
//! # Input format
//!
//! Expects an [`RgbaSurface`] with **`R8 G8 B8 A8` interleaved** pixel data
//! (4 bytes per pixel, little-endian RGBA). Only the **R, G, B** channels are
//! used; the alpha channel is read but ignored.
//!
//! # Output
//!
//! Each 4×4 texel block is encoded into **8 bytes** (0.5 bytes/pixel).

use crate::RgbaSurface;
use crate::bindings::kernel;

#[must_use]
pub fn calc_output_size(width: u32, height: u32) -> usize {
    // BC1 uses 8 bytes to store each 4×4 block, giving it an average data rate of 0.5 bytes per pixel.
    let block_count = (width.div_ceil(4) * height.div_ceil(4)) as usize;
    block_count * 8
}

#[must_use]
pub fn compress_blocks(surface: &RgbaSurface) -> Vec<u8> {
    let output_size = calc_output_size(surface.width, surface.height);
    let mut output = vec![0u8; output_size];
    compress_blocks_into(surface, &mut output);
    output
}

/// Compresses an [`RgbaSurface`] into BC1 blocks.
///
/// The surface must contain `R8 G8 B8 A8` interleaved pixel data (4 bytes per
/// pixel). Only the R, G, and B channels are read; the alpha byte is present
/// in the layout but ignored by the encoder.
///
/// # Panics
///
/// Panics if `blocks.len()` does not equal [`calc_output_size`] for the given
/// surface dimensions.
pub fn compress_blocks_into(surface: &RgbaSurface, blocks: &mut [u8]) {
    assert_eq!(
        blocks.len(),
        calc_output_size(surface.width, surface.height)
    );
    // SAFETY: The ISPC function does not mutate the source surface; the `*mut u8`
    // pointer type is an artifact of the C header declaration.
    let mut surface = kernel::rgba_surface {
        width: surface.width as i32,
        height: surface.height as i32,
        stride: surface.stride as i32,
        ptr: surface.data.as_ptr() as *mut u8,
    };

    unsafe {
        kernel::CompressBlocksBC1_ispc(&mut surface, blocks.as_mut_ptr());
    }
}
