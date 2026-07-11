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

#[cfg(feature = "prebuilt")]
extern crate ctt_bc7enc_rdo_prebuilt;

pub mod bindings;

use std::sync::Once;

use crate::bindings as ffi;

static INIT: Once = Once::new();

/// Initializes the bc7e codec lookup tables.
///
/// Called automatically by all encoding functions. Subsequent calls are no-ops.
fn compress_block_init() {
    INIT.call_once(|| unsafe {
        ffi::bc7e_compress_block_init();
    });
}

/// Returns the compressed output size in bytes for an image of the given
/// dimensions: 16 bytes per 4x4 block.
///
/// # Panics
///
/// Panics if the block count overflows `u32`.
#[must_use]
pub fn calc_output_size(width: u32, height: u32) -> usize {
    let block_count = width
        .div_ceil(4)
        .checked_mul(height.div_ceil(4))
        .expect("block count overflow") as usize;
    block_count * 16
}

/// Compresses 4x4 pixel blocks into BC7.
///
/// `pixels` must contain `num_blocks * 16` RGBA pixels (one `u32` per pixel,
/// 16 pixels per block). `blocks` must have exactly `num_blocks * 2` elements
/// (each block is 16 bytes = 2 × `u64`).
///
/// # Panics
///
/// Panics if `blocks` or `pixels` are not correctly sized for the given
/// number of blocks.
pub fn compress_blocks(
    blocks: &mut [u64],
    pixels: &[u32],
    params: &ffi::bc7e_compress_block_params,
) {
    assert_eq!(
        blocks.len() % 2,
        0,
        "output must be a multiple of 2 u64s (16 bytes per block)"
    );
    let num_blocks = blocks.len() / 2;
    assert_eq!(
        pixels.len(),
        num_blocks * 16,
        "pixels must contain {num_blocks} blocks * 16 u32 pixels each, got {}",
        pixels.len()
    );
    let num_blocks_u32 = u32::try_from(num_blocks).expect("block count exceeds u32::MAX");

    compress_block_init();

    unsafe {
        ffi::bc7e_compress_blocks(num_blocks_u32, blocks.as_mut_ptr(), pixels.as_ptr(), params);
    }
}

/// Compresses 4x4 pixel blocks into BC7, returning a newly allocated `Vec`.
///
/// `pixels` must contain `num_blocks * 16` RGBA pixels (one `u32` per pixel).
#[must_use]
pub fn compress_blocks_alloc(
    num_blocks: usize,
    pixels: &[u32],
    params: &ffi::bc7e_compress_block_params,
) -> Vec<u64> {
    assert_eq!(
        pixels.len(),
        num_blocks * 16,
        "pixels must contain {num_blocks} blocks * 16 u32 pixels each, got {}",
        pixels.len()
    );

    // ISPC 1.30's source-built neon-i32x8 target can corrupt memory when a
    // data-dependent path runs with a partial final gang on AArch64. Prebuilt
    // libraries and other targets are unaffected. Duplicate the final block
    // to fill the gang, then discard the extra encoded blocks.
    #[cfg(all(feature = "build-from-source", target_arch = "aarch64"))]
    let gang_width = 8;
    #[cfg(not(all(feature = "build-from-source", target_arch = "aarch64")))]
    let gang_width = 1;

    let pixels = pad_pixels_to_gang(pixels, num_blocks, gang_width);
    let ispc_num_blocks = pixels.len() / 16;

    let mut output = vec![0u64; ispc_num_blocks * 2];
    compress_blocks(&mut output, &pixels, params);
    output.truncate(num_blocks * 2);
    output
}

fn pad_pixels_to_gang(
    pixels: &[u32],
    num_blocks: usize,
    gang_width: usize,
) -> std::borrow::Cow<'_, [u32]> {
    let padded_blocks = num_blocks.next_multiple_of(gang_width);
    if padded_blocks == num_blocks {
        return std::borrow::Cow::Borrowed(pixels);
    }

    let last_block = &pixels[(num_blocks - 1) * 16..num_blocks * 16];
    let mut padded = Vec::with_capacity(padded_blocks * 16);
    padded.extend_from_slice(pixels);
    for _ in num_blocks..padded_blocks {
        padded.extend_from_slice(last_block);
    }
    std::borrow::Cow::Owned(padded)
}

// ---------------------------------------------------------------------------
// Preset helpers
// ---------------------------------------------------------------------------

/// Default / highest quality preset.
#[must_use]
pub fn params_init(perceptual: bool) -> ffi::bc7e_compress_block_params {
    compress_block_init();
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init(&mut p, perceptual) };
    p
}

/// Ultrafast preset — mode 6 only for opaque, limited alpha modes.
#[must_use]
pub fn params_init_ultrafast(perceptual: bool) -> ffi::bc7e_compress_block_params {
    compress_block_init();
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_ultrafast(&mut p, perceptual) };
    p
}

/// Very fast preset.
#[must_use]
pub fn params_init_veryfast(perceptual: bool) -> ffi::bc7e_compress_block_params {
    compress_block_init();
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_veryfast(&mut p, perceptual) };
    p
}

/// Fast preset.
#[must_use]
pub fn params_init_fast(perceptual: bool) -> ffi::bc7e_compress_block_params {
    compress_block_init();
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_fast(&mut p, perceptual) };
    p
}

/// Basic quality preset.
#[must_use]
pub fn params_init_basic(perceptual: bool) -> ffi::bc7e_compress_block_params {
    compress_block_init();
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_basic(&mut p, perceptual) };
    p
}

/// Slow preset — higher quality, pbit search enabled.
#[must_use]
pub fn params_init_slow(perceptual: bool) -> ffi::bc7e_compress_block_params {
    compress_block_init();
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_slow(&mut p, perceptual) };
    p
}

/// Very slow preset.
#[must_use]
pub fn params_init_veryslow(perceptual: bool) -> ffi::bc7e_compress_block_params {
    compress_block_init();
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_veryslow(&mut p, perceptual) };
    p
}

/// Slowest preset — maximum quality.
#[must_use]
pub fn params_init_slowest(perceptual: bool) -> ffi::bc7e_compress_block_params {
    compress_block_init();
    let mut p = unsafe { std::mem::zeroed() };
    unsafe { ffi::bc7e_compress_block_params_init_slowest(&mut p, perceptual) };
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_single_block() {
        // One 4x4 block of solid red pixels (RGBA packed as u32).
        let mut pixels = [0u32; 16];
        for pixel in &mut pixels {
            *pixel = 0xFF0000FF_u32.to_le(); // R=255, G=0, B=0, A=255
        }

        let params = params_init_ultrafast(false);
        let mut output = [0u64; 2];
        compress_blocks(&mut output, &pixels, &params);

        // The output should be non-zero (a valid BC7 block).
        assert!(
            output.iter().any(|&b| b != 0),
            "expected non-zero BC7 output"
        );
    }

    #[test]
    fn gang_padding_repeats_final_block() {
        let pixels: Vec<u32> = (0..3 * 16).map(|value| value as u32).collect();
        let padded = pad_pixels_to_gang(&pixels, 3, 8);

        assert_eq!(padded.len(), 8 * 16);
        assert_eq!(&padded[..3 * 16], pixels.as_slice());
        let final_block = &pixels[2 * 16..3 * 16];
        for block in padded[3 * 16..].chunks_exact(16) {
            assert_eq!(block, final_block);
        }
    }
}
