//! BC6H block compression — RGB HDR (half-precision float).
//!
//! # Input format
//!
//! Expects an [`RgbaF16Surface`] with **`R16 G16 B16 A16` interleaved** pixel
//! data (8 bytes per pixel). Each channel is a **little-endian IEEE 754
//! binary16 (half-precision) float**. The alpha channel is present in the
//! layout but **ignored** by the encoder (it is set to zero internally).
//!
//! # Output
//!
//! Each 4×4 texel block is encoded into **16 bytes** (1 byte/pixel). The
//! format stores unsigned half-float endpoints; signed BC6H is not supported
//! by this encoder.

use crate::RgbaF16Surface;
use crate::bindings::kernel;

#[derive(Debug, Copy, Clone)]
pub struct EncodeSettings {
    pub slow_mode: bool,
    pub fast_mode: bool,
    pub refine_iterations_1p: u32,
    pub refine_iterations_2p: u32,
    pub fast_skip_threshold: u32,
}

#[must_use]
pub fn calc_output_size(width: u32, height: u32) -> usize {
    // BC6H uses a fixed block size of 16 bytes (128 bits) and a fixed tile size of 4x4 texels.
    let block_count = (width.div_ceil(4) * height.div_ceil(4)) as usize;
    block_count * 16
}

#[must_use]
pub fn compress_blocks(settings: &EncodeSettings, surface: &RgbaF16Surface) -> Vec<u8> {
    let output_size = calc_output_size(surface.width, surface.height);
    let mut output = vec![0u8; output_size];
    compress_blocks_into(settings, surface, &mut output);
    output
}

/// Compresses an [`RgbaF16Surface`] into BC6H blocks.
///
/// The surface must contain `R16 G16 B16 A16` interleaved pixel data (8 bytes
/// per pixel) where each channel is a little-endian f16. Only the R, G, and B
/// channels are encoded; the alpha channel is ignored.
///
/// # Panics
///
/// Panics if `blocks.len()` does not equal [`calc_output_size`] for the given
/// surface dimensions.
pub fn compress_blocks_into(settings: &EncodeSettings, surface: &RgbaF16Surface, blocks: &mut [u8]) {
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
    let mut settings = kernel::bc6h_enc_settings {
        slow_mode: settings.slow_mode,
        fast_mode: settings.fast_mode,
        refineIterations_1p: settings.refine_iterations_1p as i32,
        refineIterations_2p: settings.refine_iterations_2p as i32,
        fastSkipThreshold: settings.fast_skip_threshold as i32,
    };

    unsafe {
        kernel::CompressBlocksBC6H_ispc(&mut surface, blocks.as_mut_ptr(), &mut settings);
    }
}

pub fn very_fast_settings() -> EncodeSettings {
    EncodeSettings {
        slow_mode: false,
        fast_mode: true,
        fast_skip_threshold: 0,
        refine_iterations_1p: 0,
        refine_iterations_2p: 0,
    }
}

pub fn fast_settings() -> EncodeSettings {
    EncodeSettings {
        slow_mode: false,
        fast_mode: true,
        fast_skip_threshold: 2,
        refine_iterations_1p: 0,
        refine_iterations_2p: 1,
    }
}

pub fn basic_settings() -> EncodeSettings {
    EncodeSettings {
        slow_mode: false,
        fast_mode: false,
        fast_skip_threshold: 4,
        refine_iterations_1p: 2,
        refine_iterations_2p: 2,
    }
}

pub fn slow_settings() -> EncodeSettings {
    EncodeSettings {
        slow_mode: true,
        fast_mode: false,
        fast_skip_threshold: 10,
        refine_iterations_1p: 2,
        refine_iterations_2p: 2,
    }
}

pub fn very_slow_settings() -> EncodeSettings {
    EncodeSettings {
        slow_mode: true,
        fast_mode: false,
        fast_skip_threshold: 32,
        refine_iterations_1p: 2,
        refine_iterations_2p: 2,
    }
}
