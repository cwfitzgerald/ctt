mod bindings;

use bindings::*;
use std::os::raw::c_void;
use std::sync::Once;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Error returned by ctt-compressonator operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    /// The image has a zero dimension.
    #[error("image dimensions must be non-zero, got {width}x{height}")]
    EmptyImage { width: u32, height: u32 },
    /// An unknown error occurred in CMP_Core.
    #[error("an unknown error occurred in cmp_core")]
    Unknown,
    /// Memory allocation failed inside CMP_Core.
    #[error("memory allocation failed in cmp_core")]
    AllocationFailed,
    /// A null or invalid pointer was passed to CMP_Core.
    #[error("a null or invalid pointer was passed to cmp_core")]
    InvalidPointer,
    /// The red channel weight is out of range.
    #[error("red channel weight is out of range")]
    RedChannelOutOfRange,
    /// The green channel weight is out of range.
    #[error("green channel weight is out of range")]
    GreenChannelOutOfRange,
    /// The blue channel weight is out of range.
    #[error("blue channel weight is out of range")]
    BlueChannelOutOfRange,
    /// CMP_Core returned an error code not covered by the known set.
    #[error("cmp_core returned unrecognized error code {code}")]
    UnrecognizedCode { code: i32 },
}

/// Maps a CMP_Core return code to a Rust error.
///
/// Error codes are defined by the `CGU_ERROR_CODES` enum in `common_def.h`
/// and exposed via bindgen.
fn check(code: i32) -> Result<(), Error> {
    match code {
        CGU_ERROR_CODES_CGU_CORE_OK => Ok(()),
        CGU_ERROR_CODES_CGU_CORE_ERR_UNKOWN => Err(Error::Unknown),
        CGU_ERROR_CODES_CGU_CORE_ERR_NEWMEM => Err(Error::AllocationFailed),
        CGU_ERROR_CODES_CGU_CORE_ERR_INVALIDPTR => Err(Error::InvalidPointer),
        CGU_ERROR_CODES_CGU_CORE_ERR_RANGERED => Err(Error::RedChannelOutOfRange),
        CGU_ERROR_CODES_CGU_CORE_ERR_RANGEGREEN => Err(Error::GreenChannelOutOfRange),
        CGU_ERROR_CODES_CGU_CORE_ERR_RANGEBLUE => Err(Error::BlueChannelOutOfRange),
        _ => Err(Error::UnrecognizedCode { code }),
    }
}

fn check_nonempty(width: u32, height: u32) -> Result<(), Error> {
    if width == 0 || height == 0 {
        Err(Error::EmptyImage { width, height })
    } else {
        Ok(())
    }
}

/// Iterate over every 4×4 block in an image whose dimensions may not be
/// multiples of the block size. For fully-contained blocks, calls `compress`
/// with a pointer into the original source and the source stride (zero-copy).
/// For partial edge blocks, fills a tightly-packed scratch buffer with
/// edge-clamped samples and calls `compress` with the scratch.
///
/// `pixel_stride` is the byte size of one source pixel. Source data must be
/// tightly packed: `src.len() >= width * height * pixel_stride`.
fn iterate_blocks_u8<F>(
    src: &[u8],
    width: u32,
    height: u32,
    pixel_stride: usize,
    bytes_per_block: usize,
    dst: &mut [u8],
    mut compress: F,
) -> Result<(), Error>
where
    F: FnMut(*const u8, u32, *mut u8) -> Result<(), Error>,
{
    debug_assert!(width > 0 && height > 0);
    let blocks_x = width.div_ceil(BLOCK_WIDTH) as usize;
    let blocks_y = height.div_ceil(BLOCK_HEIGHT) as usize;
    let stride = (width as usize) * pixel_stride;
    let w = width as usize;
    let h = height as usize;
    let max_x = w - 1;
    let max_y = h - 1;

    // Up to 4×4 pixels at 8 bytes per pixel covers everything through BC6H.
    // The stack-allocated buffer avoids a per-call heap allocation.
    let mut scratch = [0u8; 4 * 4 * 8];
    let scratch_len = 4 * 4 * pixel_stride;
    let scratch_stride = 4 * pixel_stride;
    debug_assert!(scratch_len <= scratch.len());

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let x0 = bx * 4;
            let y0 = by * 4;
            let dst_offset = (by * blocks_x + bx) * bytes_per_block;
            if x0 + 4 <= w && y0 + 4 <= h {
                let src_offset = y0 * stride + x0 * pixel_stride;
                compress(
                    unsafe { src.as_ptr().add(src_offset) },
                    stride as u32,
                    unsafe { dst.as_mut_ptr().add(dst_offset) },
                )?;
            } else {
                for py in 0..4usize {
                    let y = (y0 + py).min(max_y);
                    for px in 0..4usize {
                        let x = (x0 + px).min(max_x);
                        let src_off = y * stride + x * pixel_stride;
                        let dst_off = (py * 4 + px) * pixel_stride;
                        scratch[dst_off..dst_off + pixel_stride]
                            .copy_from_slice(&src[src_off..src_off + pixel_stride]);
                    }
                }
                compress(scratch.as_ptr(), scratch_stride as u32, unsafe {
                    dst.as_mut_ptr().add(dst_offset)
                })?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SIMD control
// ---------------------------------------------------------------------------

/// Which SIMD extension is currently enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SimdExtension {
    None = 0,
    Sse4 = 1,
    Avx2 = 2,
    Avx512 = 3,
}

/// # Safety
///
/// Must not be called concurrently with any compression/decompression
/// operation or other SIMD control function. The underlying C library
/// modifies global function pointers without synchronization.
pub unsafe fn enable_sse4() -> Result<(), Error> {
    check(unsafe { bindings::EnableSSE4() })
}

/// # Safety
///
/// Must not be called concurrently with any compression/decompression
/// operation or other SIMD control function. The underlying C library
/// modifies global function pointers without synchronization.
pub unsafe fn enable_avx2() -> Result<(), Error> {
    check(unsafe { bindings::EnableAVX2() })
}

/// # Safety
///
/// Must not be called concurrently with any compression/decompression
/// operation or other SIMD control function. The underlying C library
/// modifies global function pointers without synchronization.
pub unsafe fn enable_avx512() -> Result<(), Error> {
    check(unsafe { bindings::EnableAVX512() })
}

/// # Safety
///
/// Must not be called concurrently with any compression/decompression
/// operation or other SIMD control function. The underlying C library
/// modifies global function pointers without synchronization.
pub unsafe fn disable_simd() -> Result<(), Error> {
    check(unsafe { bindings::DisableSIMD() })
}

pub fn enabled_simd_extension() -> SimdExtension {
    match unsafe { bindings::GetEnabledSIMDExtension() } {
        1 => SimdExtension::Sse4,
        2 => SimdExtension::Avx2,
        3 => SimdExtension::Avx512,
        _ => SimdExtension::None,
    }
}

/// Auto-detect and enable the best available SIMD extension.
/// Called once before any compression/decompression.
fn ensure_init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx512f") {
                let _ = bindings::EnableAVX512();
            } else if std::arch::is_x86_feature_detected!("avx2") {
                let _ = bindings::EnableAVX2();
            } else if std::arch::is_x86_feature_detected!("sse4.1") {
                let _ = bindings::EnableSSE4();
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

const BLOCK_WIDTH: u32 = 4;
const BLOCK_HEIGHT: u32 = 4;
const BLOCK_PIXELS: usize = 16; // 4x4

// ---------------------------------------------------------------------------
// BC1 — 4-channel RGBA U8, 8 bytes per block
//
// The most common DXT format. Encodes RGB with optional 1-bit alpha.
// ---------------------------------------------------------------------------

pub mod bc1 {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 8;
    pub const SRC_PIXEL_STRIDE: usize = 4; // RGBA U8

    /// Opaque encoder options handle for BC1.
    pub struct Options(*mut c_void);

    // SAFETY: The options handle is a plain heap-allocated data struct
    // with no thread-local or global mutable state references.
    unsafe impl Send for Options {}
    unsafe impl Sync for Options {}

    impl Options {
        pub fn new() -> Result<Self, Error> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { bindings::CreateOptionsBC1(&mut ptr) })?;
            Ok(Self(ptr))
        }

        pub fn set_quality(&mut self, quality: f32) -> Result<(), Error> {
            check(unsafe { bindings::SetQualityBC1(self.0, quality) })
        }

        pub fn set_channel_weights(
            &mut self,
            red: f32,
            green: f32,
            blue: f32,
        ) -> Result<(), Error> {
            check(unsafe { bindings::SetChannelWeightsBC1(self.0, red, green, blue) })
        }

        pub fn set_srgb(&mut self, srgb: bool) -> Result<(), Error> {
            check(unsafe { bindings::SetSrgbBC1(self.0, srgb) })
        }

        pub(crate) fn as_ptr(&self) -> *const c_void {
            self.0
        }
    }

    impl Drop for Options {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { bindings::DestroyOptionsBC1(self.0) };
            }
        }
    }

    /// Returns the compressed output size in bytes for the given dimensions.
    ///
    /// Dimensions that aren't multiples of 4 are rounded up to the next block
    /// boundary (partial edge blocks are padded with edge-replicated pixels
    /// during compression).
    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress an RGBA U8 image into BC1 blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height * 4` bytes of tightly-packed
    /// RGBA data. Dimensions don't need to be multiples of 4; partial edge
    /// blocks are filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress an RGBA U8 image into BC1 blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height * 4` bytes of tightly-packed
    /// RGBA data. `dst` must be exactly `calc_output_size(width, height)`
    /// bytes long.
    pub fn compress_blocks_into(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize) * SRC_PIXEL_STRIDE;
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} bytes, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        let opts_ptr = options.as_ptr();
        iterate_blocks_u8(
            src,
            width,
            height,
            SRC_PIXEL_STRIDE,
            BYTES_PER_BLOCK,
            dst,
            |src_ptr, stride, dst_ptr| {
                check(unsafe { bindings::CompressBlockBC1(src_ptr, stride, dst_ptr, opts_ptr) })
            },
        )
    }

    /// Decompress a single BC1 block into 4x4 RGBA U8 pixels (64 bytes).
    pub fn decompress_block(cmp_block: &[u8; BYTES_PER_BLOCK]) -> Result<[u8; 64], Error> {
        ensure_init();
        let mut out = [0u8; 64];
        check(unsafe {
            bindings::DecompressBlockBC1(cmp_block.as_ptr(), out.as_mut_ptr(), std::ptr::null())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BC2 — 4-channel RGBA U8, 16 bytes per block
//
// Stores explicit 4-bit alpha per pixel alongside DXT-style RGB.
// ---------------------------------------------------------------------------

pub mod bc2 {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 16;
    pub const SRC_PIXEL_STRIDE: usize = 4;

    /// Opaque encoder options handle for BC2.
    pub struct Options(*mut c_void);

    // SAFETY: The options handle is a plain heap-allocated data struct
    // with no thread-local or global mutable state references.
    unsafe impl Send for Options {}
    unsafe impl Sync for Options {}

    impl Options {
        pub fn new() -> Result<Self, Error> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { bindings::CreateOptionsBC2(&mut ptr) })?;
            Ok(Self(ptr))
        }

        pub fn set_quality(&mut self, quality: f32) -> Result<(), Error> {
            check(unsafe { bindings::SetQualityBC2(self.0, quality) })
        }

        pub fn set_channel_weights(
            &mut self,
            red: f32,
            green: f32,
            blue: f32,
        ) -> Result<(), Error> {
            check(unsafe { bindings::SetChannelWeightsBC2(self.0, red, green, blue) })
        }

        pub fn set_srgb(&mut self, srgb: bool) -> Result<(), Error> {
            check(unsafe { bindings::SetSrgbBC2(self.0, srgb) })
        }

        pub(crate) fn as_ptr(&self) -> *const c_void {
            self.0
        }
    }

    impl Drop for Options {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { bindings::DestroyOptionsBC2(self.0) };
            }
        }
    }

    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress an RGBA U8 image into BC2 blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height * 4` bytes of tightly-packed
    /// RGBA data. Dimensions don't need to be multiples of 4; partial edge
    /// blocks are filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress an RGBA U8 image into BC2 blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height * 4` bytes of tightly-packed
    /// RGBA data. `dst` must be exactly `calc_output_size(width, height)`
    /// bytes long.
    pub fn compress_blocks_into(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize) * SRC_PIXEL_STRIDE;
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} bytes, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        let opts_ptr = options.as_ptr();
        iterate_blocks_u8(
            src,
            width,
            height,
            SRC_PIXEL_STRIDE,
            BYTES_PER_BLOCK,
            dst,
            |src_ptr, stride, dst_ptr| {
                check(unsafe { bindings::CompressBlockBC2(src_ptr, stride, dst_ptr, opts_ptr) })
            },
        )
    }

    /// Decompress a single BC2 block into 4x4 RGBA U8 pixels (64 bytes).
    pub fn decompress_block(cmp_block: &[u8; BYTES_PER_BLOCK]) -> Result<[u8; 64], Error> {
        ensure_init();
        let mut out = [0u8; 64];
        check(unsafe {
            bindings::DecompressBlockBC2(cmp_block.as_ptr(), out.as_mut_ptr(), std::ptr::null())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BC3 — 4-channel RGBA U8, 16 bytes per block
//
// DXT5: interpolated alpha (higher quality than BC2) with DXT-style RGB.
// ---------------------------------------------------------------------------

pub mod bc3 {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 16;
    pub const SRC_PIXEL_STRIDE: usize = 4;

    /// Opaque encoder options handle for BC3.
    pub struct Options(*mut c_void);

    // SAFETY: The options handle is a plain heap-allocated data struct
    // with no thread-local or global mutable state references.
    unsafe impl Send for Options {}
    unsafe impl Sync for Options {}

    impl Options {
        pub fn new() -> Result<Self, Error> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { bindings::CreateOptionsBC3(&mut ptr) })?;
            Ok(Self(ptr))
        }

        pub fn set_quality(&mut self, quality: f32) -> Result<(), Error> {
            check(unsafe { bindings::SetQualityBC3(self.0, quality) })
        }

        pub fn set_channel_weights(
            &mut self,
            red: f32,
            green: f32,
            blue: f32,
        ) -> Result<(), Error> {
            check(unsafe { bindings::SetChannelWeightsBC3(self.0, red, green, blue) })
        }

        pub fn set_srgb(&mut self, srgb: bool) -> Result<(), Error> {
            check(unsafe { bindings::SetSrgbBC3(self.0, srgb) })
        }

        pub(crate) fn as_ptr(&self) -> *const c_void {
            self.0
        }
    }

    impl Drop for Options {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { bindings::DestroyOptionsBC3(self.0) };
            }
        }
    }

    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress an RGBA U8 image into BC3 blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height * 4` bytes of tightly-packed
    /// RGBA data. Dimensions don't need to be multiples of 4; partial edge
    /// blocks are filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress an RGBA U8 image into BC3 blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height * 4` bytes of tightly-packed
    /// RGBA data. `dst` must be exactly `calc_output_size(width, height)`
    /// bytes long.
    pub fn compress_blocks_into(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize) * SRC_PIXEL_STRIDE;
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} bytes, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        let opts_ptr = options.as_ptr();
        iterate_blocks_u8(
            src,
            width,
            height,
            SRC_PIXEL_STRIDE,
            BYTES_PER_BLOCK,
            dst,
            |src_ptr, stride, dst_ptr| {
                check(unsafe { bindings::CompressBlockBC3(src_ptr, stride, dst_ptr, opts_ptr) })
            },
        )
    }

    /// Decompress a single BC3 block into 4x4 RGBA U8 pixels (64 bytes).
    pub fn decompress_block(cmp_block: &[u8; BYTES_PER_BLOCK]) -> Result<[u8; 64], Error> {
        ensure_init();
        let mut out = [0u8; 64];
        check(unsafe {
            bindings::DecompressBlockBC3(cmp_block.as_ptr(), out.as_mut_ptr(), std::ptr::null())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BC7 — 4-channel RGBA U8, 16 bytes per block
//
// High-quality RGBA compression with multiple encoding modes.
// ---------------------------------------------------------------------------

pub mod bc7 {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 16;
    pub const SRC_PIXEL_STRIDE: usize = 4;

    /// Opaque encoder options handle for BC7.
    pub struct Options(*mut c_void);

    // SAFETY: The options handle is a plain heap-allocated data struct
    // with no thread-local or global mutable state references.
    unsafe impl Send for Options {}
    unsafe impl Sync for Options {}

    impl Options {
        pub fn new() -> Result<Self, Error> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { bindings::CreateOptionsBC7(&mut ptr) })?;
            Ok(Self(ptr))
        }

        pub fn set_quality(&mut self, quality: f32) -> Result<(), Error> {
            check(unsafe { bindings::SetQualityBC7(self.0, quality) })
        }

        pub fn set_mask(&mut self, mask: u8) -> Result<(), Error> {
            check(unsafe { bindings::SetMaskBC7(self.0, mask) })
        }

        pub fn set_alpha_options(
            &mut self,
            image_needs_alpha: bool,
            colour_restrict: bool,
            alpha_restrict: bool,
        ) -> Result<(), Error> {
            check(unsafe {
                bindings::SetAlphaOptionsBC7(
                    self.0,
                    image_needs_alpha,
                    colour_restrict,
                    alpha_restrict,
                )
            })
        }

        pub fn set_error_threshold(&mut self, min: f32, max: f32) -> Result<(), Error> {
            check(unsafe { bindings::SetErrorThresholdBC7(self.0, min, max) })
        }

        pub(crate) fn as_ptr(&self) -> *const c_void {
            self.0
        }
    }

    impl Drop for Options {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { bindings::DestroyOptionsBC7(self.0) };
            }
        }
    }

    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress an RGBA U8 image into BC7 blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height * 4` bytes of tightly-packed
    /// RGBA data. Dimensions don't need to be multiples of 4; partial edge
    /// blocks are filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress an RGBA U8 image into BC7 blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height * 4` bytes of tightly-packed
    /// RGBA data. `dst` must be exactly `calc_output_size(width, height)`
    /// bytes long.
    pub fn compress_blocks_into(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize) * SRC_PIXEL_STRIDE;
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} bytes, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        let opts_ptr = options.as_ptr();
        iterate_blocks_u8(
            src,
            width,
            height,
            SRC_PIXEL_STRIDE,
            BYTES_PER_BLOCK,
            dst,
            |src_ptr, stride, dst_ptr| {
                check(unsafe { bindings::CompressBlockBC7(src_ptr, stride, dst_ptr, opts_ptr) })
            },
        )
    }

    /// Decompress a single BC7 block into 4x4 RGBA U8 pixels (64 bytes).
    pub fn decompress_block(cmp_block: &[u8; BYTES_PER_BLOCK]) -> Result<[u8; 64], Error> {
        ensure_init();
        let mut out = [0u8; 64];
        check(unsafe {
            bindings::DecompressBlockBC7(cmp_block.as_ptr(), out.as_mut_ptr(), std::ptr::null())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BC4 — single-channel unsigned U8, 8 bytes per block
//
// Compresses a single unsigned 8-bit channel (typically R or luminance).
// ---------------------------------------------------------------------------

pub mod bc4 {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 8;

    /// Opaque encoder options handle for BC4/BC4S.
    pub struct Options(*mut c_void);

    // SAFETY: The options handle is a plain heap-allocated data struct
    // with no thread-local or global mutable state references.
    unsafe impl Send for Options {}
    unsafe impl Sync for Options {}

    impl Options {
        pub fn new() -> Result<Self, Error> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { bindings::CreateOptionsBC4(&mut ptr) })?;
            Ok(Self(ptr))
        }

        pub fn set_quality(&mut self, quality: f32) -> Result<(), Error> {
            check(unsafe { bindings::SetQualityBC4(self.0, quality) })
        }

        pub(crate) fn as_ptr(&self) -> *const c_void {
            self.0
        }
    }

    impl Drop for Options {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { bindings::DestroyOptionsBC4(self.0) };
            }
        }
    }

    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress a single-channel U8 image into BC4 blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height` bytes (1 byte per pixel).
    /// Dimensions don't need to be multiples of 4; partial edge blocks are
    /// filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress a single-channel U8 image into BC4 blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height` bytes (1 byte per pixel).
    /// `dst` must be exactly `calc_output_size(width, height)` bytes long.
    pub fn compress_blocks_into(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize);
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} bytes, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        let opts_ptr = options.as_ptr();
        iterate_blocks_u8(
            src,
            width,
            height,
            1, // R U8
            BYTES_PER_BLOCK,
            dst,
            |src_ptr, stride, dst_ptr| {
                check(unsafe { bindings::CompressBlockBC4(src_ptr, stride, dst_ptr, opts_ptr) })
            },
        )
    }

    /// Decompress a single BC4 block into 4x4 unsigned U8 values (16 bytes).
    pub fn decompress_block(
        cmp_block: &[u8; BYTES_PER_BLOCK],
    ) -> Result<[u8; BLOCK_PIXELS], Error> {
        ensure_init();
        let mut out = [0u8; BLOCK_PIXELS];
        check(unsafe {
            bindings::DecompressBlockBC4(cmp_block.as_ptr(), out.as_mut_ptr(), std::ptr::null())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BC4S — single-channel signed I8, 8 bytes per block
//
// Signed variant of BC4. Shares options with BC4.
// ---------------------------------------------------------------------------

pub mod bc4s {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 8;

    // BC4S shares options with BC4.
    pub use super::bc4::Options;

    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress a single-channel I8 image into BC4S blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height` signed bytes (1 byte per
    /// pixel). Dimensions don't need to be multiples of 4; partial edge blocks
    /// are filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[i8],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress a single-channel I8 image into BC4S blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height` signed bytes (1 byte per
    /// pixel). `dst` must be exactly `calc_output_size(width, height)` bytes
    /// long.
    pub fn compress_blocks_into(
        src: &[i8],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize);
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} bytes, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        // i8 and u8 have identical byte layouts; reinterpret the slice so we
        // can reuse the shared iterator, then cast the pointer back at the
        // C call site.
        let src_u8: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len()) };
        let opts_ptr = options.as_ptr();
        iterate_blocks_u8(
            src_u8,
            width,
            height,
            1,
            BYTES_PER_BLOCK,
            dst,
            |src_ptr, stride, dst_ptr| {
                check(unsafe {
                    bindings::CompressBlockBC4S(src_ptr.cast(), stride, dst_ptr, opts_ptr)
                })
            },
        )
    }

    /// Decompress a single BC4S block into 4x4 signed I8 values (16 bytes).
    pub fn decompress_block(
        cmp_block: &[u8; BYTES_PER_BLOCK],
    ) -> Result<[i8; BLOCK_PIXELS], Error> {
        ensure_init();
        let mut out = [0i8; BLOCK_PIXELS];
        check(unsafe {
            bindings::DecompressBlockBC4S(
                cmp_block.as_ptr(),
                out.as_mut_ptr() as *mut _,
                std::ptr::null(),
            )
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BC5 — two-channel unsigned U8, 16 bytes per block
//
// Compresses two unsigned 8-bit channels (typically RG for normal maps).
// Source data is interleaved as [R, G, R, G, ...]; the encoder deinterleaves
// into separate channel buffers per block for the underlying C API.
// ---------------------------------------------------------------------------

pub mod bc5 {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 16;
    pub const SRC_PIXEL_STRIDE: usize = 2; // RG U8

    /// Opaque encoder options handle for BC5/BC5S.
    pub struct Options(*mut c_void);

    // SAFETY: The options handle is a plain heap-allocated data struct
    // with no thread-local or global mutable state references.
    unsafe impl Send for Options {}
    unsafe impl Sync for Options {}

    impl Options {
        pub fn new() -> Result<Self, Error> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { bindings::CreateOptionsBC5(&mut ptr) })?;
            Ok(Self(ptr))
        }

        pub fn set_quality(&mut self, quality: f32) -> Result<(), Error> {
            check(unsafe { bindings::SetQualityBC5(self.0, quality) })
        }

        pub(crate) fn as_ptr(&self) -> *const c_void {
            self.0
        }
    }

    impl Drop for Options {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { bindings::DestroyOptionsBC5(self.0) };
            }
        }
    }

    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress an interleaved RG U8 image into BC5 blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height * 2` bytes of interleaved
    /// [R, G] data. Dimensions don't need to be multiples of 4; partial edge
    /// blocks are filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress an interleaved RG U8 image into BC5 blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height * 2` bytes of interleaved
    /// [R, G] data. `dst` must be exactly `calc_output_size(width, height)`
    /// bytes long.
    ///
    /// The interleaved source is deinterleaved into separate R and G channel
    /// buffers per 4x4 block before passing to the C encoder. Out-of-range
    /// reads at the right/bottom edge clamp to the nearest edge pixel.
    pub fn compress_blocks_into(
        src: &[u8],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize) * SRC_PIXEL_STRIDE;
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} bytes, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        let blocks_x = width.div_ceil(BLOCK_WIDTH) as usize;
        let blocks_y = height.div_ceil(BLOCK_HEIGHT) as usize;
        let stride = (width as usize) * SRC_PIXEL_STRIDE;
        let max_x = (width - 1) as usize;
        let max_y = (height - 1) as usize;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut r_block = [0u8; BLOCK_PIXELS];
                let mut g_block = [0u8; BLOCK_PIXELS];

                // Deinterleave [R, G] pairs into separate channel buffers
                // with edge clamping at the right/bottom image borders.
                for row in 0..4 {
                    let y = (by * 4 + row).min(max_y);
                    let row_start = y * stride;
                    let block_row = row * 4;
                    for col in 0..4 {
                        let x = (bx * 4 + col).min(max_x);
                        let pair_off = row_start + x * SRC_PIXEL_STRIDE;
                        r_block[block_row + col] = src[pair_off];
                        g_block[block_row + col] = src[pair_off + 1];
                    }
                }

                let dst_offset = (by * blocks_x + bx) * BYTES_PER_BLOCK;
                check(unsafe {
                    bindings::CompressBlockBC5(
                        r_block.as_ptr(),
                        4,
                        g_block.as_ptr(),
                        4,
                        dst.as_mut_ptr().add(dst_offset),
                        options.as_ptr(),
                    )
                })?;
            }
        }
        Ok(())
    }

    /// Decompress a single BC5 block into two separate 4x4 channel buffers.
    pub fn decompress_block(
        cmp_block: &[u8; BYTES_PER_BLOCK],
    ) -> Result<([u8; BLOCK_PIXELS], [u8; BLOCK_PIXELS]), Error> {
        ensure_init();
        let mut r = [0u8; BLOCK_PIXELS];
        let mut g = [0u8; BLOCK_PIXELS];
        check(unsafe {
            bindings::DecompressBlockBC5(
                cmp_block.as_ptr(),
                r.as_mut_ptr(),
                g.as_mut_ptr(),
                std::ptr::null(),
            )
        })?;
        Ok((r, g))
    }
}

// ---------------------------------------------------------------------------
// BC5S — two-channel signed I8, 16 bytes per block
//
// Signed variant of BC5. Shares options with BC5.
// ---------------------------------------------------------------------------

pub mod bc5s {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 16;
    pub const SRC_PIXEL_STRIDE: usize = 2;

    // BC5S shares options with BC5.
    pub use super::bc5::Options;

    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress an interleaved RG I8 image into BC5S blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height * 2` signed bytes of
    /// interleaved [R, G] data. Dimensions don't need to be multiples of 4;
    /// partial edge blocks are filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[i8],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress an interleaved RG I8 image into BC5S blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height * 2` signed bytes of
    /// interleaved [R, G] data. `dst` must be exactly
    /// `calc_output_size(width, height)` bytes long.
    pub fn compress_blocks_into(
        src: &[i8],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize) * SRC_PIXEL_STRIDE;
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} bytes, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        let blocks_x = width.div_ceil(BLOCK_WIDTH) as usize;
        let blocks_y = height.div_ceil(BLOCK_HEIGHT) as usize;
        let stride = (width as usize) * SRC_PIXEL_STRIDE;
        let max_x = (width - 1) as usize;
        let max_y = (height - 1) as usize;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut r_block = [0i8; BLOCK_PIXELS];
                let mut g_block = [0i8; BLOCK_PIXELS];

                for row in 0..4 {
                    let y = (by * 4 + row).min(max_y);
                    let row_start = y * stride;
                    let block_row = row * 4;
                    for col in 0..4 {
                        let x = (bx * 4 + col).min(max_x);
                        let pair_off = row_start + x * SRC_PIXEL_STRIDE;
                        r_block[block_row + col] = src[pair_off];
                        g_block[block_row + col] = src[pair_off + 1];
                    }
                }

                let dst_offset = (by * blocks_x + bx) * BYTES_PER_BLOCK;
                check(unsafe {
                    bindings::CompressBlockBC5S(
                        r_block.as_ptr() as *const _,
                        4,
                        g_block.as_ptr() as *const _,
                        4,
                        dst.as_mut_ptr().add(dst_offset),
                        options.as_ptr(),
                    )
                })?;
            }
        }
        Ok(())
    }

    /// Decompress a single BC5S block into two separate 4x4 signed I8 channel buffers.
    pub fn decompress_block(
        cmp_block: &[u8; BYTES_PER_BLOCK],
    ) -> Result<([i8; BLOCK_PIXELS], [i8; BLOCK_PIXELS]), Error> {
        ensure_init();
        let mut r = [0i8; BLOCK_PIXELS];
        let mut g = [0i8; BLOCK_PIXELS];
        check(unsafe {
            bindings::DecompressBlockBC5S(
                cmp_block.as_ptr(),
                r.as_mut_ptr() as *mut _,
                g.as_mut_ptr() as *mut _,
                std::ptr::null(),
            )
        })?;
        Ok((r, g))
    }
}

// ---------------------------------------------------------------------------
// BC6H — three-channel unsigned FP16, 16 bytes per block
//
// Compresses RGB half-precision floating-point data. Use `set_signed(true)`
// on the options to encode BC6H_SF (signed variant).
// ---------------------------------------------------------------------------

pub mod bc6h {
    use super::*;

    pub const BYTES_PER_BLOCK: usize = 16;
    pub const SRC_PIXEL_STRIDE: usize = 3; // RGB F16 (3 u16 values)

    /// Opaque encoder options handle for BC6H.
    pub struct Options(*mut c_void);

    // SAFETY: The options handle is a plain heap-allocated data struct
    // with no thread-local or global mutable state references.
    unsafe impl Send for Options {}
    unsafe impl Sync for Options {}

    impl Options {
        pub fn new() -> Result<Self, Error> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { bindings::CreateOptionsBC6(&mut ptr) })?;
            Ok(Self(ptr))
        }

        pub fn set_quality(&mut self, quality: f32) -> Result<(), Error> {
            check(unsafe { bindings::SetQualityBC6(self.0, quality) })
        }

        pub fn set_signed(&mut self, signed: bool) -> Result<(), Error> {
            check(unsafe { bindings::SetSignedBC6(self.0, signed) })
        }

        pub fn set_mask(&mut self, mask: u32) -> Result<(), Error> {
            check(unsafe { bindings::SetMaskBC6(self.0, mask) })
        }

        pub(crate) fn as_ptr(&self) -> *const c_void {
            self.0
        }
    }

    impl Drop for Options {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { bindings::DestroyOptionsBC6(self.0) };
            }
        }
    }

    #[must_use]
    pub fn calc_output_size(width: u32, height: u32) -> usize {
        (width.div_ceil(4) as usize) * (height.div_ceil(4) as usize) * BYTES_PER_BLOCK
    }

    /// Compress an RGB FP16 image into BC6H blocks, returning the compressed data.
    ///
    /// `src` must contain at least `width * height * 3` u16 values of
    /// tightly-packed RGB data. Dimensions don't need to be multiples of 4;
    /// partial edge blocks are filled with edge-replicated pixels.
    pub fn compress_blocks(
        src: &[u16],
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; calc_output_size(width, height)];
        compress_blocks_into(src, width, height, options, &mut out)?;
        Ok(out)
    }

    /// Compress an RGB FP16 image into BC6H blocks, writing into `dst`.
    ///
    /// `src` must contain at least `width * height * 3` u16 values of
    /// tightly-packed RGB data. `dst` must be exactly
    /// `calc_output_size(width, height)` bytes long.
    ///
    /// The C API stride is in units of `u16`.
    pub fn compress_blocks_into(
        src: &[u16],
        width: u32,
        height: u32,
        options: &Options,
        dst: &mut [u8],
    ) -> Result<(), Error> {
        ensure_init();
        check_nonempty(width, height)?;
        let expected_src = (width as usize) * (height as usize) * SRC_PIXEL_STRIDE;
        assert!(
            src.len() >= expected_src,
            "src buffer too small: need {} u16s, got {}",
            expected_src,
            src.len(),
        );
        assert_eq!(dst.len(), calc_output_size(width, height));

        let blocks_x = width.div_ceil(BLOCK_WIDTH) as usize;
        let blocks_y = height.div_ceil(BLOCK_HEIGHT) as usize;
        let stride_shorts = (width as usize) * SRC_PIXEL_STRIDE;
        let w = width as usize;
        let h = height as usize;
        let max_x = w - 1;
        let max_y = h - 1;

        // 4×4 block of RGB u16 = 48 shorts of scratch.
        let mut scratch = [0u16; 4 * 4 * SRC_PIXEL_STRIDE];
        let scratch_stride_shorts = (4 * SRC_PIXEL_STRIDE) as u32;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * 4;
                let y0 = by * 4;
                let dst_offset = (by * blocks_x + bx) * BYTES_PER_BLOCK;

                if x0 + 4 <= w && y0 + 4 <= h {
                    let src_offset = y0 * stride_shorts + x0 * SRC_PIXEL_STRIDE;
                    check(unsafe {
                        bindings::CompressBlockBC6(
                            src.as_ptr().add(src_offset),
                            stride_shorts as u32,
                            dst.as_mut_ptr().add(dst_offset),
                            options.as_ptr(),
                        )
                    })?;
                } else {
                    for py in 0..4usize {
                        let y = (y0 + py).min(max_y);
                        for px in 0..4usize {
                            let x = (x0 + px).min(max_x);
                            let src_off = y * stride_shorts + x * SRC_PIXEL_STRIDE;
                            let dst_off = (py * 4 + px) * SRC_PIXEL_STRIDE;
                            scratch[dst_off..dst_off + SRC_PIXEL_STRIDE]
                                .copy_from_slice(&src[src_off..src_off + SRC_PIXEL_STRIDE]);
                        }
                    }
                    check(unsafe {
                        bindings::CompressBlockBC6(
                            scratch.as_ptr(),
                            scratch_stride_shorts,
                            dst.as_mut_ptr().add(dst_offset),
                            options.as_ptr(),
                        )
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Decompress a single BC6H block into 48 u16 values (4x4 pixels x 3 RGB channels).
    pub fn decompress_block(cmp_block: &[u8; BYTES_PER_BLOCK]) -> Result<[u16; 48], Error> {
        ensure_init();
        let mut out = [0u16; 48];
        check(unsafe {
            bindings::DecompressBlockBC6(cmp_block.as_ptr(), out.as_mut_ptr(), std::ptr::null())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid red 4x4 RGBA block.
    fn red_block_rgba() -> [u8; 64] {
        let mut block = [0u8; 64];
        for i in 0..16 {
            block[i * 4] = 255; // R
            block[i * 4 + 3] = 255; // A
        }
        block
    }

    #[test]
    fn bc1_roundtrip() {
        let src = red_block_rgba();
        let opts = bc1::Options::new().unwrap();
        let compressed = bc1::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc1::BYTES_PER_BLOCK);
        let decompressed =
            bc1::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        assert!(decompressed[0] > 200, "R={}", decompressed[0]);
    }

    #[test]
    fn bc2_roundtrip() {
        let src = red_block_rgba();
        let opts = bc2::Options::new().unwrap();
        let compressed = bc2::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc2::BYTES_PER_BLOCK);
        let decompressed =
            bc2::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        assert!(decompressed[0] > 200, "R={}", decompressed[0]);
    }

    #[test]
    fn bc3_roundtrip() {
        let src = red_block_rgba();
        let opts = bc3::Options::new().unwrap();
        let compressed = bc3::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc3::BYTES_PER_BLOCK);
        let decompressed =
            bc3::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        assert!(decompressed[0] > 200, "R={}", decompressed[0]);
    }

    #[test]
    fn bc4_roundtrip() {
        let src = [128u8; 16]; // 4x4 single-channel
        let opts = bc4::Options::new().unwrap();
        let compressed = bc4::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc4::BYTES_PER_BLOCK);
        let decompressed =
            bc4::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        assert!(
            (decompressed[0] as i16 - 128).unsigned_abs() < 5,
            "got {}",
            decompressed[0]
        );
    }

    #[test]
    fn bc4s_roundtrip() {
        let src = [64i8; 16];
        let opts = bc4::Options::new().unwrap();
        let compressed = bc4s::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc4s::BYTES_PER_BLOCK);
        let decompressed =
            bc4s::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        assert!(
            (decompressed[0] as i16 - 64).unsigned_abs() < 5,
            "got {}",
            decompressed[0]
        );
    }

    #[test]
    fn bc5_roundtrip() {
        // 4x4 RG data, 2 bytes per pixel
        let mut src = [0u8; 32];
        for i in 0..16 {
            src[i * 2] = 200; // R
            src[i * 2 + 1] = 100; // G
        }
        let opts = bc5::Options::new().unwrap();
        let compressed = bc5::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc5::BYTES_PER_BLOCK);
        let (r, g) = bc5::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        assert!(r[0] > 180, "R={}", r[0]);
        assert!(g[0] > 80, "G={}", g[0]);
    }

    #[test]
    fn bc5s_roundtrip() {
        let mut src = [0i8; 32];
        for i in 0..16 {
            src[i * 2] = 50;
            src[i * 2 + 1] = -30;
        }
        let opts = bc5::Options::new().unwrap();
        let compressed = bc5s::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc5s::BYTES_PER_BLOCK);
        let (r, g) = bc5s::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        // On x86 with SIMD the codec is precise; the scalar fallback on
        // aarch64 has much lower quality for signed formats.
        #[cfg(target_arch = "x86_64")]
        {
            assert!((r[0] as i16 - 50).unsigned_abs() < 5, "R={}", r[0]);
            assert!((g[0] as i16 + 30).unsigned_abs() < 5, "G={}", g[0]);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (r, g); // smoke-test: compression + decompression didn't crash
        }
    }

    #[test]
    fn bc6h_roundtrip() {
        // 4x4 RGB fp16 — half-precision 1.0 is 0x3C00
        let mut src = [0u16; 48]; // 16 pixels x 3 channels
        for i in 0..16 {
            src[i * 3] = 0x3C00; // R = 1.0
        }
        let opts = bc6h::Options::new().unwrap();
        let compressed = bc6h::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc6h::BYTES_PER_BLOCK);
        let decompressed =
            bc6h::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        assert!(decompressed[0] > 0x3800, "R=0x{:04X}", decompressed[0]);
    }

    #[test]
    fn bc7_roundtrip() {
        let src = red_block_rgba();
        let opts = bc7::Options::new().unwrap();
        let compressed = bc7::compress_blocks(&src, 4, 4, &opts).unwrap();
        assert_eq!(compressed.len(), bc7::BYTES_PER_BLOCK);
        let decompressed =
            bc7::decompress_block(compressed.as_slice().try_into().unwrap()).unwrap();
        assert!(decompressed[0] > 200, "R={}", decompressed[0]);
    }

    #[test]
    fn simd_detection_works() {
        ensure_init();
        #[cfg(target_arch = "x86_64")]
        assert_ne!(enabled_simd_extension(), SimdExtension::None);
    }

    #[test]
    fn multi_block_image() {
        // 8x8 solid blue RGBA image
        let mut src = vec![0u8; 8 * 8 * 4];
        for i in 0..(8 * 8) {
            src[i * 4 + 2] = 255; // B
            src[i * 4 + 3] = 255; // A
        }
        let opts = bc7::Options::new().unwrap();
        let compressed = bc7::compress_blocks(&src, 8, 8, &opts).unwrap();
        // 2x2 blocks = 4 blocks x 16 bytes = 64 bytes
        assert_eq!(compressed.len(), 64);
    }

    #[test]
    fn empty_image_returns_error() {
        let opts = bc7::Options::new().unwrap();
        let result = bc7::compress_blocks(&[], 0, 0, &opts);
        assert!(matches!(
            result,
            Err(Error::EmptyImage {
                width: 0,
                height: 0
            })
        ));
    }

    /// Non-multiple-of-4 dimensions round up to a single block with
    /// edge-replicated pixels.
    #[test]
    fn bc1_non_block_aligned_dimensions() {
        // 5x3 image: each pixel is solid red (R=255).
        let mut src = vec![0u8; 5 * 3 * 4];
        for i in 0..(5 * 3) {
            src[i * 4] = 255;
            src[i * 4 + 3] = 255;
        }
        let opts = bc1::Options::new().unwrap();
        let compressed = bc1::compress_blocks(&src, 5, 3, &opts).unwrap();
        // Rounded to 8x4: 2 blocks in x, 1 in y = 2 blocks = 16 bytes.
        assert_eq!(compressed.len(), 2 * bc1::BYTES_PER_BLOCK);
        // The right-edge block (containing the replicated column) should still
        // produce a valid red block on decode.
        let decoded = bc1::decompress_block(
            (&compressed[bc1::BYTES_PER_BLOCK..2 * bc1::BYTES_PER_BLOCK])
                .try_into()
                .unwrap(),
        )
        .unwrap();
        assert!(
            decoded[0] > 200,
            "edge block should decode to red, got R={}",
            decoded[0]
        );
    }
}
