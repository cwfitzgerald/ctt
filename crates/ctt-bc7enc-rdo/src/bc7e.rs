//! Safe wrapper for the ISPC bc7e block encoder.
//!
//! # Input format
//!
//! The encoder operates on individual 4x4 pixel blocks. Each block is 16
//! tightly-packed `R8 G8 B8 A8` pixels (64 bytes). The caller is responsible
//! for tiling an image into blocks before calling [`compress_blocks`].
//!
//! # Output
//!
//! Each 4x4 block produces **16 bytes** of BC7-compressed data.

use crate::bindings::bc7e as ffi;

/// Initializes the bc7e codec lookup tables.
///
/// Must be called at least once before [`compress_blocks`]. Subsequent calls
/// are no-ops. This is **not** thread-safe with respect to concurrent calls
/// to [`compress_blocks`] during the first initialization.
pub fn compress_block_init() {
    unsafe {
        ffi::bc7e_compress_block_init();
    }
}

/// Returns the compressed output size in bytes for an image of the given
/// dimensions: 16 bytes per 4x4 block.
#[must_use]
pub fn calc_output_size(width: u32, height: u32) -> usize {
    let block_count = (width.div_ceil(4) * height.div_ceil(4)) as usize;
    block_count * 16
}

/// Compresses 4x4 pixel blocks into BC7.
///
/// `pixels` must contain `num_blocks * 64` bytes of tightly-packed RGBA data
/// (16 pixels per block, 4 bytes per pixel). `blocks` must be exactly
/// `num_blocks * 16` bytes.
///
/// # Panics
///
/// Panics if `blocks` or `pixels` are not correctly sized for the given
/// number of blocks.
pub fn compress_blocks(
    blocks: &mut [u8],
    pixels: &[u8],
    params: &ffi::bc7e_compress_block_params,
) {
    assert_eq!(blocks.len() % 16, 0, "output must be a multiple of 16 bytes");
    let num_blocks = blocks.len() / 16;
    assert_eq!(
        pixels.len(),
        num_blocks * 64,
        "pixels must contain {num_blocks} blocks * 64 bytes each, got {}",
        pixels.len()
    );

    unsafe {
        ffi::bc7e_compress_blocks(
            num_blocks as u32,
            blocks.as_mut_ptr() as *mut u64,
            pixels.as_ptr() as *const u32,
            params,
        );
    }
}

/// Compresses 4x4 pixel blocks into BC7, returning a newly allocated `Vec`.
///
/// `pixels` must contain `num_blocks * 64` bytes of tightly-packed RGBA data.
#[must_use]
pub fn compress_blocks_alloc(
    num_blocks: usize,
    pixels: &[u8],
    params: &ffi::bc7e_compress_block_params,
) -> Vec<u8> {
    let mut output = vec![0u8; num_blocks * 16];
    compress_blocks(&mut output, pixels, params);
    output
}

// ---------------------------------------------------------------------------
// Preset helpers
// ---------------------------------------------------------------------------

/// Default / highest quality preset.
#[must_use]
pub fn params_init(perceptual: bool) -> ffi::bc7e_compress_block_params {
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init(&mut p, perceptual) };
    p
}

/// Ultrafast preset — mode 6 only for opaque, limited alpha modes.
#[must_use]
pub fn params_init_ultrafast(perceptual: bool) -> ffi::bc7e_compress_block_params {
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_ultrafast(&mut p, perceptual) };
    p
}

/// Very fast preset.
#[must_use]
pub fn params_init_veryfast(perceptual: bool) -> ffi::bc7e_compress_block_params {
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_veryfast(&mut p, perceptual) };
    p
}

/// Fast preset.
#[must_use]
pub fn params_init_fast(perceptual: bool) -> ffi::bc7e_compress_block_params {
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_fast(&mut p, perceptual) };
    p
}

/// Basic quality preset.
#[must_use]
pub fn params_init_basic(perceptual: bool) -> ffi::bc7e_compress_block_params {
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_basic(&mut p, perceptual) };
    p
}

/// Slow preset — higher quality, pbit search enabled.
#[must_use]
pub fn params_init_slow(perceptual: bool) -> ffi::bc7e_compress_block_params {
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_slow(&mut p, perceptual) };
    p
}

/// Very slow preset.
#[must_use]
pub fn params_init_veryslow(perceptual: bool) -> ffi::bc7e_compress_block_params {
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_veryslow(&mut p, perceptual) };
    p
}

/// Slowest preset — maximum quality.
#[must_use]
pub fn params_init_slowest(perceptual: bool) -> ffi::bc7e_compress_block_params {
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_slowest(&mut p, perceptual) };
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_single_block() {
        compress_block_init();

        // One 4x4 block of solid red pixels (RGBA).
        let mut pixels = [0u8; 64];
        for i in 0..16 {
            pixels[i * 4] = 255; // R
            pixels[i * 4 + 3] = 255; // A
        }

        let params = params_init_ultrafast(false);
        let mut output = [0u8; 16];
        compress_blocks(&mut output, &pixels, &params);

        // The output should be non-zero (a valid BC7 block).
        assert!(output.iter().any(|&b| b != 0), "expected non-zero BC7 output");
    }
}
