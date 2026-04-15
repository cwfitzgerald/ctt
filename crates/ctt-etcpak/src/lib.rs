//! Safe Rust bindings to [etcpak](https://github.com/wolfpld/etcpak) — an extremely fast
//! ETC/EAC/BC texture compressor and decompressor.
//!
//! # ISA dispatch
//!
//! etcpak uses compile-time SIMD ISA selection. This crate compiles multiple
//! variants and selects the best one at runtime:
//!
//! - **x86_64**: SSE4.1 and AVX2 variants (selected via `std::is_x86_feature_detected!`)
//! - **aarch64**: NEON (the only variant, always available on AArch64)
//!
//! # Pixel format
//!
//! All compression functions expect 4 bytes per pixel, packed as `uint32_t`.
//! The **channel ordering within each pixel depends on the codec**:
//!
//! - **ETC1/ETC2/EAC codecs**: `B8 G8 R8 A8` (BGRA — blue in lowest byte)
//! - **BC codecs**: `R8 G8 B8 A8` (RGBA — red in lowest byte)
//!
//! Input data must be **tightly packed** (stride = width × 4 bytes).
//!
//! Decompression functions output `uint32_t` per pixel in native etcpak layout.

mod dispatch;

use dispatch::dispatch;

// ── Surface ───────────────────────────────────────────────────────────────────

/// Describes a 2D image for compression.
///
/// All pixel data must be 4 bytes per pixel, tightly packed (stride = width × 4).
/// The channel ordering depends on which codec consumes the surface — see the
/// individual format modules for details.
#[derive(Debug, Copy, Clone)]
pub struct Surface<'a> {
    /// The raw pixel data for the image (4 bytes per pixel, tightly packed).
    pub data: &'a [u8],
    /// The width of the image in texels. Must be a multiple of 4.
    pub width: u32,
    /// The height of the image in texels. Must be a multiple of 4.
    pub height: u32,
}

impl<'a> Surface<'a> {
    /// Creates a new surface, validating dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` are not multiples of 4, or if `data` is
    /// too small for the given dimensions.
    pub fn new(data: &'a [u8], width: u32, height: u32) -> Self {
        assert!(width % 4 == 0, "width {width} must be a multiple of 4");
        assert!(height % 4 == 0, "height {height} must be a multiple of 4");
        let required = (width as usize) * (height as usize) * 4;
        assert!(
            data.len() >= required,
            "data length {} is less than width * height * 4 ({required})",
            data.len()
        );
        Self {
            data,
            width,
            height,
        }
    }

    fn blocks(&self) -> u32 {
        (self.width / 4) * (self.height / 4)
    }
}

// ── Output size helpers ───────────────────────────────────────────────────────

/// Output size for formats using 64-bit (8-byte) blocks:
/// ETC1, ETC2 RGB, EAC R, BC1, BC4.
#[must_use]
pub fn output_size_64bpb(width: u32, height: u32) -> usize {
    let block_count = (width.div_ceil(4) * height.div_ceil(4)) as usize;
    block_count * 8
}

/// Output size for formats using 128-bit (16-byte) blocks:
/// ETC2 RGBA, EAC RG, BC3, BC5.
#[must_use]
pub fn output_size_128bpb(width: u32, height: u32) -> usize {
    let block_count = (width.div_ceil(4) * height.div_ceil(4)) as usize;
    block_count * 16
}

// ── ETC1 ──────────────────────────────────────────────────────────────────────

/// ETC1 block compression — RGB, no alpha.
///
/// Expects BGRA pixel data (blue in lowest byte of each `uint32_t`).
/// Each 4×4 block encodes to 8 bytes.
pub mod etc1 {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface) -> Vec<u8> {
        let size = output_size_64bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_64bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_etc1_rgb)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }

    #[must_use]
    pub fn compress_blocks_dither(surface: &Surface) -> Vec<u8> {
        let size = output_size_64bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_dither_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_dither_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_64bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_etc1_rgb_dither)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }
}

// ── ETC2 RGB ──────────────────────────────────────────────────────────────────

/// ETC2 RGB block compression — RGB, no alpha.
///
/// Expects BGRA pixel data. Each 4×4 block encodes to 8 bytes.
pub mod etc2_rgb {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface, use_heuristics: bool) -> Vec<u8> {
        let size = output_size_64bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, use_heuristics, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, use_heuristics: bool, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_64bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_etc2_rgb)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
                use_heuristics,
            );
        }
    }
}

// ── ETC2 RGBA ─────────────────────────────────────────────────────────────────

/// ETC2 RGBA block compression — RGB with alpha.
///
/// Expects BGRA pixel data. Each 4×4 block encodes to 16 bytes.
pub mod etc2_rgba {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface, use_heuristics: bool) -> Vec<u8> {
        let size = output_size_128bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, use_heuristics, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, use_heuristics: bool, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_128bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_etc2_rgba)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
                use_heuristics,
            );
        }
    }
}

// ── EAC R ─────────────────────────────────────────────────────────────────────

/// EAC R11 block compression — single channel (red).
///
/// Expects BGRA pixel data. Each 4×4 block encodes to 8 bytes.
pub mod eac_r {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface) -> Vec<u8> {
        let size = output_size_64bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_64bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_eac_r)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }
}

// ── EAC RG ────────────────────────────────────────────────────────────────────

/// EAC RG11 block compression — dual channel (red + green).
///
/// Expects BGRA pixel data. Each 4×4 block encodes to 16 bytes.
pub mod eac_rg {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface) -> Vec<u8> {
        let size = output_size_128bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_128bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_eac_rg)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }
}

// ── BC1 ───────────────────────────────────────────────────────────────────────

/// BC1 (DXT1) block compression — RGB with optional 1-bit alpha.
///
/// Expects RGBA pixel data (red in lowest byte). Each 4×4 block encodes to 8 bytes.
pub mod bc1 {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface) -> Vec<u8> {
        let size = output_size_64bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_64bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_bc1)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }

    #[must_use]
    pub fn compress_blocks_dither(surface: &Surface) -> Vec<u8> {
        let size = output_size_64bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_dither_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_dither_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_64bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_bc1_dither)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }
}

// ── BC3 ───────────────────────────────────────────────────────────────────────

/// BC3 (DXT5) block compression — RGBA.
///
/// Expects RGBA pixel data. Each 4×4 block encodes to 16 bytes.
pub mod bc3 {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface) -> Vec<u8> {
        let size = output_size_128bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_128bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_bc3)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }
}

// ── BC4 ───────────────────────────────────────────────────────────────────────

/// BC4 block compression — single channel (red).
///
/// Expects RGBA pixel data. Each 4×4 block encodes to 8 bytes.
pub mod bc4 {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface) -> Vec<u8> {
        let size = output_size_64bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_64bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_bc4)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }
}

// ── BC5 ───────────────────────────────────────────────────────────────────────

/// BC5 block compression — dual channel (red + green, typically normals).
///
/// Expects RGBA pixel data. Each 4×4 block encodes to 16 bytes.
pub mod bc5 {
    use super::*;

    #[must_use]
    pub fn compress_blocks(surface: &Surface) -> Vec<u8> {
        let size = output_size_128bpb(surface.width, surface.height);
        let mut output = vec![0u8; size];
        compress_blocks_into(surface, &mut output);
        output
    }

    pub fn compress_blocks_into(surface: &Surface, output: &mut [u8]) {
        assert_eq!(
            output.len(),
            output_size_128bpb(surface.width, surface.height)
        );
        unsafe {
            (dispatch().compress_bc5)(
                surface.data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                surface.blocks(),
                surface.width as usize,
            );
        }
    }
}

// ── Decompression ─────────────────────────────────────────────────────────────

/// Decompression functions for all supported formats.
///
/// Output is `width × height` packed `uint32_t` pixels. The channel ordering
/// matches the format's native layout.
pub mod decode {
    use super::*;

    fn output_pixel_size(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize) * 4
    }

    // ── ETC/EAC decode ──

    /// Decode ETC1 or ETC2 RGB compressed data.
    #[must_use]
    pub fn decode_rgb(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_rgb_into(data, width, height, &mut output);
        output
    }

    pub fn decode_rgb_into(data: &[u8], width: u32, height: u32, output: &mut [u8]) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_64bpb(width, height));
        unsafe {
            (dispatch().decode_rgb)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
            );
        }
    }

    /// Decode ETC2 RGBA compressed data.
    #[must_use]
    pub fn decode_rgba(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_rgba_into(data, width, height, &mut output);
        output
    }

    pub fn decode_rgba_into(data: &[u8], width: u32, height: u32, output: &mut [u8]) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_128bpb(width, height));
        unsafe {
            (dispatch().decode_rgba)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
            );
        }
    }

    /// Decode EAC R11 compressed data.
    #[must_use]
    pub fn decode_r(data: &[u8], width: u32, height: u32, is_signed: bool) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_r_into(data, width, height, is_signed, &mut output);
        output
    }

    pub fn decode_r_into(data: &[u8], width: u32, height: u32, is_signed: bool, output: &mut [u8]) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_64bpb(width, height));
        unsafe {
            (dispatch().decode_r)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
                is_signed,
            );
        }
    }

    /// Decode EAC RG11 compressed data.
    #[must_use]
    pub fn decode_rg(data: &[u8], width: u32, height: u32, is_signed: bool) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_rg_into(data, width, height, is_signed, &mut output);
        output
    }

    pub fn decode_rg_into(
        data: &[u8],
        width: u32,
        height: u32,
        is_signed: bool,
        output: &mut [u8],
    ) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_128bpb(width, height));
        unsafe {
            (dispatch().decode_rg)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
                is_signed,
            );
        }
    }

    // ── BC decode ──

    /// Decode BC1 (DXT1) compressed data.
    #[must_use]
    pub fn decode_bc1(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_bc1_into(data, width, height, &mut output);
        output
    }

    pub fn decode_bc1_into(data: &[u8], width: u32, height: u32, output: &mut [u8]) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_64bpb(width, height));
        unsafe {
            (dispatch().decode_bc1)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
            );
        }
    }

    /// Decode BC3 (DXT5) compressed data.
    #[must_use]
    pub fn decode_bc3(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_bc3_into(data, width, height, &mut output);
        output
    }

    pub fn decode_bc3_into(data: &[u8], width: u32, height: u32, output: &mut [u8]) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_128bpb(width, height));
        unsafe {
            (dispatch().decode_bc3)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
            );
        }
    }

    /// Decode BC4 compressed data.
    #[must_use]
    pub fn decode_bc4(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_bc4_into(data, width, height, &mut output);
        output
    }

    pub fn decode_bc4_into(data: &[u8], width: u32, height: u32, output: &mut [u8]) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_64bpb(width, height));
        unsafe {
            (dispatch().decode_bc4)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
            );
        }
    }

    /// Decode BC5 compressed data.
    #[must_use]
    pub fn decode_bc5(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_bc5_into(data, width, height, &mut output);
        output
    }

    pub fn decode_bc5_into(data: &[u8], width: u32, height: u32, output: &mut [u8]) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_128bpb(width, height));
        unsafe {
            (dispatch().decode_bc5)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
            );
        }
    }

    /// Decode BC7 compressed data.
    #[must_use]
    pub fn decode_bc7(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let size = output_pixel_size(width, height);
        let mut output = vec![0u8; size];
        decode_bc7_into(data, width, height, &mut output);
        output
    }

    pub fn decode_bc7_into(data: &[u8], width: u32, height: u32, output: &mut [u8]) {
        assert_eq!(output.len(), output_pixel_size(width, height));
        assert!(data.len() >= output_size_128bpb(width, height));
        unsafe {
            (dispatch().decode_bc7)(
                data.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                width as i32,
                height as i32,
            );
        }
    }
}
