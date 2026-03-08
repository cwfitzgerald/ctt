//! BC5 (RGTC2) block compression — two-channel (red + green).
//!
//! # Input format
//!
//! Expects an [`RgSurface`] with **`R8 G8` interleaved** pixel data (2 bytes
//! per pixel). The byte sequence is `[R₀, G₀, R₁, G₁, …]`. The ISPC kernel
//! reads 4 bytes at a time as a packed u32, de-interleaving two R and two G
//! samples per read. Each channel is independently encoded into its own BC4
//! block.
//!
//! # Output
//!
//! Each 4×4 texel block is encoded into **16 bytes** (1 byte/pixel):
//! 8 bytes for the red channel block followed by 8 bytes for the green channel
//! block.

use crate::bindings::kernel;
use crate::RgSurface;

#[must_use]
pub fn calc_output_size(width: u32, height: u32) -> usize {
    // BC5 uses 16 bytes to store each 4×4 block, giving it an average data rate of 1 byte per pixel.
    let block_count = (width.div_ceil(4) * height.div_ceil(4)) as usize;
    block_count * 16
}

#[must_use]
pub fn compress_blocks(surface: &RgSurface) -> Vec<u8> {
    let output_size = calc_output_size(surface.width, surface.height);
    let mut output = vec![0u8; output_size];
    compress_blocks_into(surface, &mut output);
    output
}

/// Compresses an [`RgSurface`] into BC5 blocks.
///
/// The surface must contain `R8 G8` interleaved pixel data (2 bytes per
/// pixel). The R and G channels are independently encoded into separate BC4
/// blocks.
///
/// # Panics
///
/// Panics if `blocks.len()` does not equal [`calc_output_size`] for the given
/// surface dimensions.
pub fn compress_blocks_into(surface: &RgSurface, blocks: &mut [u8]) {
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
        kernel::CompressBlocksBC5_ispc(&mut surface, blocks.as_mut_ptr());
    }
}
