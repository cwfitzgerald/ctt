//! Packed 32-bit `A2B10G10R10` / `A2R10G10B10` loaders.
//!
//! One little-endian u32 per pixel: three 10-bit channels at bits `[0,10)`,
//! `[10,20)`, `[20,30)` plus a 2-bit alpha at `[30,32)`. G always sits in the
//! middle slot; the channel *order* only decides whether R or B occupies the
//! low slot — `A2B10G10R10` puts R in the low bits, `A2R10G10B10` puts R in
//! the high 10-bit slot. Alpha is always the top 2 bits.
//!
//! UNORM/SNORM decode into the f32 pipeline; UINT/SINT into the u32 pipeline.
//!
//! The SIMD kernels extract the four bit-fields into planar (SoA) lane vectors
//! using only compile-time-constant shifts against the *fixed* slot positions
//! (0/10/20/30), then choose which slot feeds R vs. B based on `R_SHIFT`. This
//! keeps a single body generic over both channel orders without needing
//! generic const shift expressions. Field normalization multiplies by the f32
//! reciprocal of each divisor (`1/1023`, `1/511`, `1/3`).

use crate::error::Result;
#[cfg(target_arch = "x86_64")]
use crate::processing::x86::{
    has_avx512, store_interleaved4_epi32, store_interleaved4_ps, store_interleaved16_epi32,
    store_interleaved16_mask_epi32, store_interleaved16_mask_ps, store_interleaved16_ps,
};
use crate::surface::Surface;

use super::{Buffer, read_pixels_f32, read_pixels_u32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// R-channel bit offset for `A2B10G10R10` (R in the low 10-bit slot).
pub const A2B_R_SHIFT: u32 = 0;
/// R-channel bit offset for `A2R10G10B10` (R in the high 10-bit slot).
pub const A2R_R_SHIFT: u32 = 20;

/// Split a packed word into `(r, g, b, a)` raw fields for the given R offset.
/// The B channel takes whichever 10-bit slot R does not.
#[inline]
pub(crate) fn extract<const R_SHIFT: u32>(word: u32) -> (u32, u32, u32, u32) {
    let r = (word >> R_SHIFT) & 0x3ff;
    let g = (word >> 10) & 0x3ff;
    let b = (word >> (20 - R_SHIFT)) & 0x3ff;
    let a = (word >> 30) & 0x3;
    (r, g, b, a)
}

/// Sign-extend a 10-bit two's-complement field, then `max(x / 511, -1)`.
#[inline]
pub(crate) fn snorm10(v: u32) -> f32 {
    let s = ((v as i32) << 22) >> 22;
    (s as f32 / 511.0).max(-1.0)
}

/// Sign-extend a 2-bit two's-complement field, then `max(x, -1)`.
#[inline]
pub(crate) fn snorm2(v: u32) -> f32 {
    let s = ((v as i32) << 30) >> 30;
    (s as f32).max(-1.0)
}

/// Sign-extend a 10-bit two's-complement field into the low bits of a u32.
#[inline]
pub(crate) fn sext10(v: u32) -> u32 {
    (((v as i32) << 22) >> 22) as u32
}

/// Sign-extend a 2-bit two's-complement field into the low bits of a u32.
#[inline]
pub(crate) fn sext2(v: u32) -> u32 {
    (((v as i32) << 30) >> 30) as u32
}

// ---- Per-pixel scalar decode (golden reference, shared by serial + tails) ----

#[inline]
fn load_unorm_pixel<const R_SHIFT: u32>(word: u32) -> [f32; 4] {
    let (r, g, b, a) = extract::<R_SHIFT>(word);
    [
        r as f32 / 1023.0,
        g as f32 / 1023.0,
        b as f32 / 1023.0,
        a as f32 / 3.0,
    ]
}

#[inline]
fn load_snorm_pixel<const R_SHIFT: u32>(word: u32) -> [f32; 4] {
    let (r, g, b, a) = extract::<R_SHIFT>(word);
    [snorm10(r), snorm10(g), snorm10(b), snorm2(a)]
}

#[inline]
fn load_uint_pixel<const R_SHIFT: u32>(word: u32) -> [u32; 4] {
    let (r, g, b, a) = extract::<R_SHIFT>(word);
    [r, g, b, a]
}

#[inline]
fn load_sint_pixel<const R_SHIFT: u32>(word: u32) -> [u32; 4] {
    let (r, g, b, a) = extract::<R_SHIFT>(word);
    [sext10(r), sext10(g), sext10(b), sext2(a)]
}

// ---- UNORM (f32 pipeline) ----

pub fn load_a2b10g10r10_unorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2b10g10r10_unorm_f32");
    load_f32_dispatch::<A2B_R_SHIFT, false>(surface)
}

pub fn load_a2r10g10b10_unorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2r10g10b10_unorm_f32");
    load_f32_dispatch::<A2R_R_SHIFT, false>(surface)
}

#[doc(hidden)]
pub fn load_a2_unorm_serial<const R_SHIFT: u32>(surface: &Surface) -> Result<Buffer<f32>> {
    read_pixels_f32(surface, 1, 4, |bytes, lanes| {
        let word = u32::from_le_bytes(<[u8; 4]>::try_from(bytes).expect("4-byte pixel"));
        *lanes = load_unorm_pixel::<R_SHIFT>(word);
    })
}

// ---- SNORM (f32 pipeline) ----

pub fn load_a2b10g10r10_snorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2b10g10r10_snorm_f32");
    load_f32_dispatch::<A2B_R_SHIFT, true>(surface)
}

pub fn load_a2r10g10b10_snorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2r10g10b10_snorm_f32");
    load_f32_dispatch::<A2R_R_SHIFT, true>(surface)
}

#[doc(hidden)]
pub fn load_a2_snorm_serial<const R_SHIFT: u32>(surface: &Surface) -> Result<Buffer<f32>> {
    read_pixels_f32(surface, 1, 4, |bytes, lanes| {
        let word = u32::from_le_bytes(<[u8; 4]>::try_from(bytes).expect("4-byte pixel"));
        *lanes = load_snorm_pixel::<R_SHIFT>(word);
    })
}

// ---- UINT (u32 pipeline) ----

pub fn load_a2b10g10r10_uint_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2b10g10r10_uint_u32");
    load_u32_dispatch::<A2B_R_SHIFT, false>(surface)
}

pub fn load_a2r10g10b10_uint_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2r10g10b10_uint_u32");
    load_u32_dispatch::<A2R_R_SHIFT, false>(surface)
}

#[doc(hidden)]
pub fn load_a2_uint_serial<const R_SHIFT: u32>(surface: &Surface) -> Result<Buffer<u32>> {
    read_pixels_u32(surface, 1, 4, |bytes, lanes| {
        let word = u32::from_le_bytes(<[u8; 4]>::try_from(bytes).expect("4-byte pixel"));
        *lanes = load_uint_pixel::<R_SHIFT>(word);
    })
}

// ---- SINT (u32 pipeline) ----

pub fn load_a2b10g10r10_sint_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2b10g10r10_sint_u32");
    load_u32_dispatch::<A2B_R_SHIFT, true>(surface)
}

pub fn load_a2r10g10b10_sint_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2r10g10b10_sint_u32");
    load_u32_dispatch::<A2R_R_SHIFT, true>(surface)
}

#[doc(hidden)]
pub fn load_a2_sint_serial<const R_SHIFT: u32>(surface: &Surface) -> Result<Buffer<u32>> {
    read_pixels_u32(surface, 1, 4, |bytes, lanes| {
        let word = u32::from_le_bytes(<[u8; 4]>::try_from(bytes).expect("4-byte pixel"));
        *lanes = load_sint_pixel::<R_SHIFT>(word);
    })
}

// ---- Runtime dispatch ----

#[inline]
fn load_f32_dispatch<const R_SHIFT: u32, const SNORM: bool>(
    surface: &Surface,
) -> Result<Buffer<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: runtime check confirms avx512f + vl + bw are available.
            return unsafe { load_a2_f32_avx512::<R_SHIFT, SNORM>(surface) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime check confirms avx2 is available.
            return unsafe { load_a2_f32_avx2::<R_SHIFT, SNORM>(surface) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { load_a2_f32_sse4_1::<R_SHIFT, SNORM>(surface) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime check confirms NEON is available.
            return unsafe { load_a2_f32_neon::<R_SHIFT, SNORM>(surface) };
        }
    }

    if SNORM {
        load_a2_snorm_serial::<R_SHIFT>(surface)
    } else {
        load_a2_unorm_serial::<R_SHIFT>(surface)
    }
}

#[inline]
fn load_u32_dispatch<const R_SHIFT: u32, const SINT: bool>(
    surface: &Surface,
) -> Result<Buffer<u32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: runtime check confirms avx512f + vl + bw are available.
            return unsafe { load_a2_u32_avx512::<R_SHIFT, SINT>(surface) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime check confirms avx2 is available.
            return unsafe { load_a2_u32_avx2::<R_SHIFT, SINT>(surface) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { load_a2_u32_sse4_1::<R_SHIFT, SINT>(surface) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime check confirms NEON is available.
            return unsafe { load_a2_u32_neon::<R_SHIFT, SINT>(surface) };
        }
    }

    if SINT {
        load_a2_sint_serial::<R_SHIFT>(surface)
    } else {
        load_a2_uint_serial::<R_SHIFT>(surface)
    }
}

// =====================================================================
// x86_64 SIMD
// =====================================================================

/// Extract the four fixed bit-slots (low/mid/high 10-bit + 2-bit alpha) from a
/// vector of 4 packed words into planar, masked lane vectors.
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn extract_slots_sse4(words: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
    let mask10 = _mm_set1_epi32(0x3ff);
    let slot0 = _mm_and_si128(words, mask10);
    let slot1 = _mm_and_si128(_mm_srli_epi32::<10>(words), mask10);
    let slot2 = _mm_and_si128(_mm_srli_epi32::<20>(words), mask10);
    let alpha = _mm_srli_epi32::<30>(words);
    (slot0, slot1, slot2, alpha)
}

/// Sign-extend the low 10 bits of each lane in place.
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn sext10_sse4(v: __m128i) -> __m128i {
    _mm_srai_epi32::<22>(_mm_slli_epi32::<22>(v))
}

/// Sign-extend the low 2 bits of each lane in place.
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn sext2_sse4(v: __m128i) -> __m128i {
    _mm_srai_epi32::<30>(_mm_slli_epi32::<30>(v))
}

/// Decode four planar field vectors into `[R, G, B, A]` planar float vectors.
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn decode_f32_planes_sse4<const SNORM: bool>(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
) -> (__m128, __m128, __m128, __m128) {
    if SNORM {
        let inv511 = _mm_set1_ps(1.0 / 511.0);
        let neg1 = _mm_set1_ps(-1.0);
        let rf = _mm_max_ps(_mm_mul_ps(_mm_cvtepi32_ps(sext10_sse4(r)), inv511), neg1);
        let gf = _mm_max_ps(_mm_mul_ps(_mm_cvtepi32_ps(sext10_sse4(g)), inv511), neg1);
        let bf = _mm_max_ps(_mm_mul_ps(_mm_cvtepi32_ps(sext10_sse4(b)), inv511), neg1);
        let af = _mm_max_ps(_mm_cvtepi32_ps(sext2_sse4(a)), neg1);
        (rf, gf, bf, af)
    } else {
        let inv1023 = _mm_set1_ps(1.0 / 1023.0);
        let inv3 = _mm_set1_ps(1.0 / 3.0);
        let rf = _mm_mul_ps(_mm_cvtepi32_ps(r), inv1023);
        let gf = _mm_mul_ps(_mm_cvtepi32_ps(g), inv1023);
        let bf = _mm_mul_ps(_mm_cvtepi32_ps(b), inv1023);
        let af = _mm_mul_ps(_mm_cvtepi32_ps(a), inv3);
        (rf, gf, bf, af)
    }
}

/// Decode 4 packed words at `src` into 4 interleaved f32 pixels at `out`.
///
/// # Safety
/// * SSE4.1 must be available.
/// * `src` valid for 16 bytes, `out` valid for 16 f32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn load_block4_f32_sse4<const R_SHIFT: u32, const SNORM: bool>(
    src: *const u8,
    out: *mut f32,
) {
    // SAFETY: caller guarantees the read/write ranges.
    unsafe {
        let words = _mm_loadu_si128(src as *const __m128i);
        let (slot0, slot1, slot2, alpha) = extract_slots_sse4(words);
        let (r, b) = if R_SHIFT == 0 {
            (slot0, slot2)
        } else {
            (slot2, slot0)
        };
        let (rf, gf, bf, af) = decode_f32_planes_sse4::<SNORM>(r, slot1, b, alpha);
        store_interleaved4_ps(rf, gf, bf, af, out);
    }
}

/// Decode 4 packed words at `src` into 4 interleaved u32 pixels at `out`.
///
/// # Safety
/// * SSE4.1 must be available.
/// * `src` valid for 16 bytes, `out` valid for 16 u32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn load_block4_u32_sse4<const R_SHIFT: u32, const SINT: bool>(
    src: *const u8,
    out: *mut u32,
) {
    // SAFETY: caller guarantees the read/write ranges.
    unsafe {
        let words = _mm_loadu_si128(src as *const __m128i);
        let (slot0, slot1, slot2, alpha) = extract_slots_sse4(words);
        let (r, b) = if R_SHIFT == 0 {
            (slot0, slot2)
        } else {
            (slot2, slot0)
        };
        let (r, g, b, a) = if SINT {
            (
                sext10_sse4(r),
                sext10_sse4(slot1),
                sext10_sse4(b),
                sext2_sse4(alpha),
            )
        } else {
            (r, slot1, b, alpha)
        };
        store_interleaved4_epi32(r, g, b, a, out);
    }
}

/// Decode an arbitrary f32 row remainder with 4-pixel SSE blocks and one padded tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn load_row_f32_sse4<const R_SHIFT: u32, const SNORM: bool>(row: &[u8], dst: *mut f32) {
    debug_assert_eq!(row.len() % 4, 0);

    let mut x = 0;
    while x + 16 <= row.len() {
        unsafe { load_block4_f32_sse4::<R_SHIFT, SNORM>(row.as_ptr().add(x), dst.add(x)) };
        x += 16;
    }
    if x < row.len() {
        let mut padded = [0u8; 16];
        padded[..row.len() - x].copy_from_slice(&row[x..]);
        let mut pixels = [0.0f32; 16];
        unsafe {
            load_block4_f32_sse4::<R_SHIFT, SNORM>(padded.as_ptr(), pixels.as_mut_ptr());
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), dst.add(x), row.len() - x);
        }
    }
}

/// Decode an arbitrary u32 row remainder with 4-pixel SSE blocks and one padded tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn load_row_u32_sse4<const R_SHIFT: u32, const SINT: bool>(row: &[u8], dst: *mut u32) {
    debug_assert_eq!(row.len() % 4, 0);

    let mut x = 0;
    while x + 16 <= row.len() {
        unsafe { load_block4_u32_sse4::<R_SHIFT, SINT>(row.as_ptr().add(x), dst.add(x)) };
        x += 16;
    }
    if x < row.len() {
        let mut padded = [0u8; 16];
        padded[..row.len() - x].copy_from_slice(&row[x..]);
        let mut pixels = [0u32; 16];
        unsafe {
            load_block4_u32_sse4::<R_SHIFT, SINT>(padded.as_ptr(), pixels.as_mut_ptr());
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), dst.add(x), row.len() - x);
        }
    }
}

/// SSE4.1 loader for the f32 (UNORM/SNORM) pipeline.
///
/// **Not part of the public API.** Exposed for benches/tests; use the runtime
/// dispatch entry points instead.
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn load_a2_f32_sse4_1<const R_SHIFT: u32, const SNORM: bool>(
    surface: &Surface,
) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2_f32_sse4_1");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: sse4.1 is available per the enclosing fn's contract.
            load_row_f32_sse4::<R_SHIFT, SNORM>(row, dst)
        })
    }
}

/// SSE4.1 loader for the u32 (UINT/SINT) pipeline.
///
/// **Not part of the public API.**
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn load_a2_u32_sse4_1<const R_SHIFT: u32, const SINT: bool>(
    surface: &Surface,
) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2_u32_sse4_1");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: sse4.1 is available per the enclosing fn's contract.
            load_row_u32_sse4::<R_SHIFT, SINT>(row, dst)
        })
    }
}

// ---- AVX2 ----

/// Sign-extend the low 10 bits of each 32-bit lane (256-bit).
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn sext10_avx2(v: __m256i) -> __m256i {
    _mm256_srai_epi32::<22>(_mm256_slli_epi32::<22>(v))
}

/// Sign-extend the low 2 bits of each 32-bit lane (256-bit).
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn sext2_avx2(v: __m256i) -> __m256i {
    _mm256_srai_epi32::<30>(_mm256_slli_epi32::<30>(v))
}

/// Extract the four fixed bit-slots from 8 packed words (256-bit).
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn extract_slots_avx2(words: __m256i) -> (__m256i, __m256i, __m256i, __m256i) {
    let mask10 = _mm256_set1_epi32(0x3ff);
    let slot0 = _mm256_and_si256(words, mask10);
    let slot1 = _mm256_and_si256(_mm256_srli_epi32::<10>(words), mask10);
    let slot2 = _mm256_and_si256(_mm256_srli_epi32::<20>(words), mask10);
    let alpha = _mm256_srli_epi32::<30>(words);
    (slot0, slot1, slot2, alpha)
}

/// Decode four planar field vectors into planar float vectors (256-bit).
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn decode_f32_planes_avx2<const SNORM: bool>(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    a: __m256i,
) -> (__m256, __m256, __m256, __m256) {
    if SNORM {
        let inv511 = _mm256_set1_ps(1.0 / 511.0);
        let neg1 = _mm256_set1_ps(-1.0);
        let rf = _mm256_max_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(sext10_avx2(r)), inv511),
            neg1,
        );
        let gf = _mm256_max_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(sext10_avx2(g)), inv511),
            neg1,
        );
        let bf = _mm256_max_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(sext10_avx2(b)), inv511),
            neg1,
        );
        let af = _mm256_max_ps(_mm256_cvtepi32_ps(sext2_avx2(a)), neg1);
        (rf, gf, bf, af)
    } else {
        let inv1023 = _mm256_set1_ps(1.0 / 1023.0);
        let inv3 = _mm256_set1_ps(1.0 / 3.0);
        let rf = _mm256_mul_ps(_mm256_cvtepi32_ps(r), inv1023);
        let gf = _mm256_mul_ps(_mm256_cvtepi32_ps(g), inv1023);
        let bf = _mm256_mul_ps(_mm256_cvtepi32_ps(b), inv1023);
        let af = _mm256_mul_ps(_mm256_cvtepi32_ps(a), inv3);
        (rf, gf, bf, af)
    }
}

/// Decode 8 packed words at `src` into 8 interleaved f32 pixels at `out`.
///
/// # Safety
/// * AVX2 must be available; `src` valid for 32 bytes, `out` for 32 f32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_block8_f32_avx2<const R_SHIFT: u32, const SNORM: bool>(
    src: *const u8,
    out: *mut f32,
) {
    // SAFETY: caller guarantees the read/write ranges.
    unsafe {
        let words = _mm256_loadu_si256(src as *const __m256i);
        let (slot0, slot1, slot2, alpha) = extract_slots_avx2(words);
        let (r, b) = if R_SHIFT == 0 {
            (slot0, slot2)
        } else {
            (slot2, slot0)
        };
        let (rf, gf, bf, af) = decode_f32_planes_avx2::<SNORM>(r, slot1, b, alpha);
        store_interleaved4_ps(
            _mm256_castps256_ps128(rf),
            _mm256_castps256_ps128(gf),
            _mm256_castps256_ps128(bf),
            _mm256_castps256_ps128(af),
            out,
        );
        store_interleaved4_ps(
            _mm256_extractf128_ps::<1>(rf),
            _mm256_extractf128_ps::<1>(gf),
            _mm256_extractf128_ps::<1>(bf),
            _mm256_extractf128_ps::<1>(af),
            out.add(16),
        );
    }
}

/// Decode 8 packed words at `src` into 8 interleaved u32 pixels at `out`.
///
/// # Safety
/// * AVX2 must be available; `src` valid for 32 bytes, `out` for 32 u32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_block8_u32_avx2<const R_SHIFT: u32, const SINT: bool>(
    src: *const u8,
    out: *mut u32,
) {
    // SAFETY: caller guarantees the read/write ranges.
    unsafe {
        let words = _mm256_loadu_si256(src as *const __m256i);
        let (slot0, slot1, slot2, alpha) = extract_slots_avx2(words);
        let (r, b) = if R_SHIFT == 0 {
            (slot0, slot2)
        } else {
            (slot2, slot0)
        };
        let (r, g, b, a) = if SINT {
            (
                sext10_avx2(r),
                sext10_avx2(slot1),
                sext10_avx2(b),
                sext2_avx2(alpha),
            )
        } else {
            (r, slot1, b, alpha)
        };
        store_interleaved4_epi32(
            _mm256_castsi256_si128(r),
            _mm256_castsi256_si128(g),
            _mm256_castsi256_si128(b),
            _mm256_castsi256_si128(a),
            out,
        );
        store_interleaved4_epi32(
            _mm256_extracti128_si256::<1>(r),
            _mm256_extracti128_si256::<1>(g),
            _mm256_extracti128_si256::<1>(b),
            _mm256_extracti128_si256::<1>(a),
            out.add(16),
        );
    }
}

/// Decode a row with 8-pixel AVX2 blocks, delegating the remainder to the
/// SSE4.1 row helper (avx2 implies sse4.1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_row_f32_avx2<const R_SHIFT: u32, const SNORM: bool>(row: &[u8], dst: *mut f32) {
    let mut x = 0usize;
    while x + 32 <= row.len() {
        unsafe { load_block8_f32_avx2::<R_SHIFT, SNORM>(row.as_ptr().add(x), dst.add(x)) };
        x += 32;
    }
    unsafe { load_row_f32_sse4::<R_SHIFT, SNORM>(&row[x..], dst.add(x)) };
}

/// AVX2 loader for the f32 (UNORM/SNORM) pipeline. **Not part of the public API.**
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn load_a2_f32_avx2<const R_SHIFT: u32, const SNORM: bool>(
    surface: &Surface,
) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2_f32_avx2");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: avx2 is available per the enclosing fn's contract.
            load_row_f32_avx2::<R_SHIFT, SNORM>(row, dst)
        })
    }
}

/// Decode a row with 8-pixel AVX2 blocks, delegating the remainder to the
/// SSE4.1 row helper (avx2 implies sse4.1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_row_u32_avx2<const R_SHIFT: u32, const SINT: bool>(row: &[u8], dst: *mut u32) {
    let mut x = 0usize;
    while x + 32 <= row.len() {
        unsafe { load_block8_u32_avx2::<R_SHIFT, SINT>(row.as_ptr().add(x), dst.add(x)) };
        x += 32;
    }
    unsafe { load_row_u32_sse4::<R_SHIFT, SINT>(&row[x..], dst.add(x)) };
}

/// AVX2 loader for the u32 (UINT/SINT) pipeline. **Not part of the public API.**
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn load_a2_u32_avx2<const R_SHIFT: u32, const SINT: bool>(
    surface: &Surface,
) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2_u32_avx2");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: avx2 is available per the enclosing fn's contract.
            load_row_u32_avx2::<R_SHIFT, SINT>(row, dst)
        })
    }
}

// ---- AVX-512 ----

/// Sign-extend the low 10 bits of each 32-bit lane (512-bit).
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn sext10_avx512(v: __m512i) -> __m512i {
    _mm512_srai_epi32::<22>(_mm512_slli_epi32::<22>(v))
}

/// Sign-extend the low 2 bits of each 32-bit lane (512-bit).
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn sext2_avx512(v: __m512i) -> __m512i {
    _mm512_srai_epi32::<30>(_mm512_slli_epi32::<30>(v))
}

/// Extract the four fixed bit-slots from 16 packed words (512-bit).
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn extract_slots_avx512(words: __m512i) -> (__m512i, __m512i, __m512i, __m512i) {
    let mask10 = _mm512_set1_epi32(0x3ff);
    let slot0 = _mm512_and_si512(words, mask10);
    let slot1 = _mm512_and_si512(_mm512_srli_epi32::<10>(words), mask10);
    let slot2 = _mm512_and_si512(_mm512_srli_epi32::<20>(words), mask10);
    let alpha = _mm512_srli_epi32::<30>(words);
    (slot0, slot1, slot2, alpha)
}

/// Decode four planar field vectors into planar float vectors (512-bit).
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn decode_f32_planes_avx512<const SNORM: bool>(
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
) -> (__m512, __m512, __m512, __m512) {
    if SNORM {
        let inv511 = _mm512_set1_ps(1.0 / 511.0);
        let neg1 = _mm512_set1_ps(-1.0);
        let rf = _mm512_max_ps(
            _mm512_mul_ps(_mm512_cvtepi32_ps(sext10_avx512(r)), inv511),
            neg1,
        );
        let gf = _mm512_max_ps(
            _mm512_mul_ps(_mm512_cvtepi32_ps(sext10_avx512(g)), inv511),
            neg1,
        );
        let bf = _mm512_max_ps(
            _mm512_mul_ps(_mm512_cvtepi32_ps(sext10_avx512(b)), inv511),
            neg1,
        );
        let af = _mm512_max_ps(_mm512_cvtepi32_ps(sext2_avx512(a)), neg1);
        (rf, gf, bf, af)
    } else {
        let inv1023 = _mm512_set1_ps(1.0 / 1023.0);
        let inv3 = _mm512_set1_ps(1.0 / 3.0);
        let rf = _mm512_mul_ps(_mm512_cvtepi32_ps(r), inv1023);
        let gf = _mm512_mul_ps(_mm512_cvtepi32_ps(g), inv1023);
        let bf = _mm512_mul_ps(_mm512_cvtepi32_ps(b), inv1023);
        let af = _mm512_mul_ps(_mm512_cvtepi32_ps(a), inv3);
        (rf, gf, bf, af)
    }
}

/// Decode 16 packed words into planar `[R, G, B, A]` f32 planes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn decode16_f32_avx512<const R_SHIFT: u32, const SNORM: bool>(
    words: __m512i,
) -> (__m512, __m512, __m512, __m512) {
    let (slot0, slot1, slot2, alpha) = extract_slots_avx512(words);
    let (r, b) = if R_SHIFT == 0 {
        (slot0, slot2)
    } else {
        (slot2, slot0)
    };
    decode_f32_planes_avx512::<SNORM>(r, slot1, b, alpha)
}

/// Decode 16 packed words into planar `[R, G, B, A]` u32 planes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn decode16_u32_avx512<const R_SHIFT: u32, const SINT: bool>(
    words: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let (slot0, slot1, slot2, alpha) = extract_slots_avx512(words);
    let (r, b) = if R_SHIFT == 0 {
        (slot0, slot2)
    } else {
        (slot2, slot0)
    };
    if SINT {
        (
            sext10_avx512(r),
            sext10_avx512(slot1),
            sext10_avx512(b),
            sext2_avx512(alpha),
        )
    } else {
        (r, slot1, b, alpha)
    }
}

/// Decode a row with 16-pixel AVX-512 blocks and a masked 1-15 pixel tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn load_row_f32_avx512<const R_SHIFT: u32, const SNORM: bool>(row: &[u8], dst: *mut f32) {
    // SAFETY: masked loads/stores only touch enabled lanes; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;
        while x + 64 <= row.len() {
            let words = _mm512_loadu_si512(row.as_ptr().add(x) as *const __m512i);
            let (rf, gf, bf, af) = decode16_f32_avx512::<R_SHIFT, SNORM>(words);
            store_interleaved16_ps(rf, gf, bf, af, dst.add(x));
            x += 64;
        }
        let rem = (row.len() - x) / 4;
        if rem > 0 {
            let words =
                _mm512_maskz_loadu_epi32((1u16 << rem) - 1, row.as_ptr().add(x) as *const i32);
            let (rf, gf, bf, af) = decode16_f32_avx512::<R_SHIFT, SNORM>(words);
            store_interleaved16_mask_ps(rf, gf, bf, af, dst.add(x), rem);
        }
    }
}

/// AVX-512 loader for the f32 (UNORM/SNORM) pipeline. **Not part of the public API.**
///
/// # Safety
/// * AVX-512 F/VL/BW must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn load_a2_f32_avx512<const R_SHIFT: u32, const SNORM: bool>(
    surface: &Surface,
) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2_f32_avx512");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: avx512 is available per the enclosing fn's contract.
            load_row_f32_avx512::<R_SHIFT, SNORM>(row, dst)
        })
    }
}

/// Decode a row with 16-pixel AVX-512 blocks and a masked 1-15 pixel tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn load_row_u32_avx512<const R_SHIFT: u32, const SINT: bool>(row: &[u8], dst: *mut u32) {
    // SAFETY: masked loads/stores only touch enabled lanes; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;
        while x + 64 <= row.len() {
            let words = _mm512_loadu_si512(row.as_ptr().add(x) as *const __m512i);
            let (r, g, b, a) = decode16_u32_avx512::<R_SHIFT, SINT>(words);
            store_interleaved16_epi32(r, g, b, a, dst.add(x));
            x += 64;
        }
        let rem = (row.len() - x) / 4;
        if rem > 0 {
            let words =
                _mm512_maskz_loadu_epi32((1u16 << rem) - 1, row.as_ptr().add(x) as *const i32);
            let (r, g, b, a) = decode16_u32_avx512::<R_SHIFT, SINT>(words);
            store_interleaved16_mask_epi32(r, g, b, a, dst.add(x), rem);
        }
    }
}

/// AVX-512 loader for the u32 (UINT/SINT) pipeline. **Not part of the public API.**
///
/// # Safety
/// * AVX-512 F/VL/BW must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn load_a2_u32_avx512<const R_SHIFT: u32, const SINT: bool>(
    surface: &Surface,
) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2_u32_avx512");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: avx512 is available per the enclosing fn's contract.
            load_row_u32_avx512::<R_SHIFT, SINT>(row, dst)
        })
    }
}

// =====================================================================
// AArch64 NEON (compile-guarded; mirrors the scalar reference)
// =====================================================================

/// Sign-extend the low 10 bits of each lane.
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sext10_neon(v: uint32x4_t) -> int32x4_t {
    vshrq_n_s32::<22>(vshlq_n_s32::<22>(vreinterpretq_s32_u32(v)))
}

/// Sign-extend the low 2 bits of each lane.
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sext2_neon(v: uint32x4_t) -> int32x4_t {
    vshrq_n_s32::<30>(vshlq_n_s32::<30>(vreinterpretq_s32_u32(v)))
}

/// Extract the four fixed bit-slots from 4 packed words.
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn extract_slots_neon(
    words: uint32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
    let mask10 = vdupq_n_u32(0x3ff);
    let slot0 = vandq_u32(words, mask10);
    let slot1 = vandq_u32(vshrq_n_u32::<10>(words), mask10);
    let slot2 = vandq_u32(vshrq_n_u32::<20>(words), mask10);
    let alpha = vshrq_n_u32::<30>(words);
    (slot0, slot1, slot2, alpha)
}

/// Decode a row with 4-pixel NEON blocks and a per-pixel scalar tail.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_row_f32_neon<const R_SHIFT: u32, const SNORM: bool>(row: &[u8], dst: *mut f32) {
    // SAFETY: NEON enabled by the enclosing target_feature; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;
        while x + 16 <= row.len() {
            let words = vld1q_u32(row.as_ptr().add(x) as *const u32);
            let (slot0, slot1, slot2, alpha) = extract_slots_neon(words);
            let (r, b) = if R_SHIFT == 0 {
                (slot0, slot2)
            } else {
                (slot2, slot0)
            };
            let (rf, gf, bf, af) = if SNORM {
                let inv511 = vdupq_n_f32(1.0 / 511.0);
                let neg1 = vdupq_n_f32(-1.0);
                (
                    vmaxq_f32(vmulq_f32(vcvtq_f32_s32(sext10_neon(r)), inv511), neg1),
                    vmaxq_f32(vmulq_f32(vcvtq_f32_s32(sext10_neon(slot1)), inv511), neg1),
                    vmaxq_f32(vmulq_f32(vcvtq_f32_s32(sext10_neon(b)), inv511), neg1),
                    vmaxq_f32(vcvtq_f32_s32(sext2_neon(alpha)), neg1),
                )
            } else {
                let inv1023 = vdupq_n_f32(1.0 / 1023.0);
                let inv3 = vdupq_n_f32(1.0 / 3.0);
                (
                    vmulq_f32(vcvtq_f32_u32(r), inv1023),
                    vmulq_f32(vcvtq_f32_u32(slot1), inv1023),
                    vmulq_f32(vcvtq_f32_u32(b), inv1023),
                    vmulq_f32(vcvtq_f32_u32(alpha), inv3),
                )
            };
            vst4q_f32(dst.add(x), float32x4x4_t(rf, gf, bf, af));
            x += 16;
        }
        while x < row.len() {
            let word = u32::from_le_bytes([row[x], row[x + 1], row[x + 2], row[x + 3]]);
            let px = if SNORM {
                load_snorm_pixel::<R_SHIFT>(word)
            } else {
                load_unorm_pixel::<R_SHIFT>(word)
            };
            for (k, &lane) in px.iter().enumerate() {
                dst.add(x + k).write(lane);
            }
            x += 4;
        }
    }
}

/// NEON loader for the f32 (UNORM/SNORM) pipeline. **Not part of the public API.**
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn load_a2_f32_neon<const R_SHIFT: u32, const SNORM: bool>(
    surface: &Surface,
) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2_f32_neon");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: NEON is available per the enclosing fn's contract.
            load_row_f32_neon::<R_SHIFT, SNORM>(row, dst)
        })
    }
}

/// Decode a row with 4-pixel NEON blocks and a per-pixel scalar tail.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_row_u32_neon<const R_SHIFT: u32, const SINT: bool>(row: &[u8], dst: *mut u32) {
    // SAFETY: NEON enabled by the enclosing target_feature; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;
        while x + 16 <= row.len() {
            let words = vld1q_u32(row.as_ptr().add(x) as *const u32);
            let (slot0, slot1, slot2, alpha) = extract_slots_neon(words);
            let (r, b) = if R_SHIFT == 0 {
                (slot0, slot2)
            } else {
                (slot2, slot0)
            };
            let (r, g, b, a) = if SINT {
                (
                    vreinterpretq_u32_s32(sext10_neon(r)),
                    vreinterpretq_u32_s32(sext10_neon(slot1)),
                    vreinterpretq_u32_s32(sext10_neon(b)),
                    vreinterpretq_u32_s32(sext2_neon(alpha)),
                )
            } else {
                (r, slot1, b, alpha)
            };
            vst4q_u32(dst.add(x), uint32x4x4_t(r, g, b, a));
            x += 16;
        }
        while x < row.len() {
            let word = u32::from_le_bytes([row[x], row[x + 1], row[x + 2], row[x + 3]]);
            let px = if SINT {
                load_sint_pixel::<R_SHIFT>(word)
            } else {
                load_uint_pixel::<R_SHIFT>(word)
            };
            for (k, &lane) in px.iter().enumerate() {
                dst.add(x + k).write(lane);
            }
            x += 4;
        }
    }
}

/// NEON loader for the u32 (UINT/SINT) pipeline. **Not part of the public API.**
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn load_a2_u32_neon<const R_SHIFT: u32, const SINT: bool>(
    surface: &Surface,
) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2_u32_neon");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: NEON is available per the enclosing fn's contract.
            load_row_u32_neon::<R_SHIFT, SINT>(row, dst)
        })
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::{ColorSpace, Surface};

    fn a2_surface(data: Vec<u8>, width: u32, height: u32, stride: u32) -> Surface {
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::A2B10G10R10_UNORM_PACK32,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        }
    }

    /// Pack a pixel's raw fields into a little-endian word for the given order.
    fn pack_word<const R_SHIFT: u32>(r: u32, g: u32, b: u32, a: u32) -> [u8; 4] {
        let word = (a << 30) | (b << (20 - R_SHIFT)) | (g << 10) | (r << R_SHIFT);
        word.to_le_bytes()
    }

    /// Every 10-bit value in the R slot and (separately) the B slot, crossed
    /// with all 4 alpha values, plus mid-range G — one pixel per row entry.
    fn full_domain_words<const R_SHIFT: u32>() -> Vec<u8> {
        let mut data = Vec::new();
        for field in 0u32..1024 {
            for a in 0u32..4 {
                // field exercised in the R slot.
                data.extend_from_slice(&pack_word::<R_SHIFT>(
                    field,
                    (field ^ 0x155) & 0x3ff,
                    512,
                    a,
                ));
                // field exercised in the B slot.
                data.extend_from_slice(&pack_word::<R_SHIFT>(
                    511,
                    (field ^ 0x2aa) & 0x3ff,
                    field,
                    a,
                ));
            }
        }
        data
    }

    /// Per-lane closeness: finite scalar lanes within a small relative
    /// tolerance, non-finite lanes matched by class.
    fn assert_pixels_close(simd: &[[f32; 4]], scalar: &[[f32; 4]], label: &str) {
        assert_eq!(simd.len(), scalar.len(), "{label} length mismatch");
        for (i, (s, r)) in simd.iter().zip(scalar).enumerate() {
            for lane in 0..4 {
                let (sv, rv) = (s[lane], r[lane]);
                if rv.is_finite() {
                    assert!(
                        (sv - rv).abs() <= 1e-4 * (1.0 + rv.abs()),
                        "{label} pixel {i} lane {lane}: simd {sv} vs scalar {rv}"
                    );
                } else {
                    assert!(
                        sv.is_nan() == rv.is_nan() && sv.is_finite() == rv.is_finite(),
                        "{label} pixel {i} lane {lane}: simd {sv} vs scalar {rv}"
                    );
                }
            }
        }
    }

    fn assert_f32_close<const R_SHIFT: u32, const SNORM: bool>(surface: &Surface) {
        let scalar = if SNORM {
            load_a2_snorm_serial::<R_SHIFT>(surface).unwrap()
        } else {
            load_a2_unorm_serial::<R_SHIFT>(surface).unwrap()
        };
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
                let simd = unsafe { load_a2_f32_sse4_1::<R_SHIFT, SNORM>(surface).unwrap() };
                assert_pixels_close(&simd.pixels, &scalar.pixels, "sse4.1 f32");
            }
            if is_x86_feature_detected!("avx2") {
                let simd = unsafe { load_a2_f32_avx2::<R_SHIFT, SNORM>(surface).unwrap() };
                assert_pixels_close(&simd.pixels, &scalar.pixels, "avx2 f32");
            }
            if has_avx512() {
                let simd = unsafe { load_a2_f32_avx512::<R_SHIFT, SNORM>(surface).unwrap() };
                assert_pixels_close(&simd.pixels, &scalar.pixels, "avx512 f32");
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            let simd = unsafe { load_a2_f32_neon::<R_SHIFT, SNORM>(surface).unwrap() };
            assert_pixels_close(&simd.pixels, &scalar.pixels, "neon f32");
        }
    }

    fn assert_u32_bit_exact<const R_SHIFT: u32, const SINT: bool>(surface: &Surface) {
        let scalar = if SINT {
            load_a2_sint_serial::<R_SHIFT>(surface).unwrap()
        } else {
            load_a2_uint_serial::<R_SHIFT>(surface).unwrap()
        };
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
                let simd = unsafe { load_a2_u32_sse4_1::<R_SHIFT, SINT>(surface).unwrap() };
                assert_eq!(simd.pixels, scalar.pixels, "sse4.1 u32 mismatch");
            }
            if is_x86_feature_detected!("avx2") {
                let simd = unsafe { load_a2_u32_avx2::<R_SHIFT, SINT>(surface).unwrap() };
                assert_eq!(simd.pixels, scalar.pixels, "avx2 u32 mismatch");
            }
            if has_avx512() {
                let simd = unsafe { load_a2_u32_avx512::<R_SHIFT, SINT>(surface).unwrap() };
                assert_eq!(simd.pixels, scalar.pixels, "avx512 u32 mismatch");
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            let simd = unsafe { load_a2_u32_neon::<R_SHIFT, SINT>(surface).unwrap() };
            assert_eq!(simd.pixels, scalar.pixels, "neon u32 mismatch");
        }
    }

    #[test]
    fn full_domain_f32_both_orders() {
        // width chosen so height=1 and many wide iterations plus SSE + scalar tails.
        let a2b = full_domain_words::<A2B_R_SHIFT>();
        let count = (a2b.len() / 4) as u32;
        let surf = a2_surface(a2b, count, 1, count * 4);
        assert_f32_close::<A2B_R_SHIFT, false>(&surf);
        assert_f32_close::<A2B_R_SHIFT, true>(&surf);

        let a2r = full_domain_words::<A2R_R_SHIFT>();
        let surf = a2_surface(a2r, count, 1, count * 4);
        assert_f32_close::<A2R_R_SHIFT, false>(&surf);
        assert_f32_close::<A2R_R_SHIFT, true>(&surf);
    }

    #[test]
    fn full_domain_u32_both_orders() {
        let a2b = full_domain_words::<A2B_R_SHIFT>();
        let count = (a2b.len() / 4) as u32;
        let surf = a2_surface(a2b, count, 1, count * 4);
        assert_u32_bit_exact::<A2B_R_SHIFT, false>(&surf);
        assert_u32_bit_exact::<A2B_R_SHIFT, true>(&surf);

        let a2r = full_domain_words::<A2R_R_SHIFT>();
        let surf = a2_surface(a2r, count, 1, count * 4);
        assert_u32_bit_exact::<A2R_R_SHIFT, false>(&surf);
        assert_u32_bit_exact::<A2R_R_SHIFT, true>(&surf);
    }

    #[test]
    fn odd_widths_exercise_all_tails() {
        // widths spanning each kernel's main-loop + narrower-block + scalar tail.
        for &wpx in &[1usize, 3, 5, 7, 11, 13, 17, 19, 23, 31, 33] {
            let mut data = Vec::new();
            for i in 0..wpx as u32 {
                let r = (i.wrapping_mul(37)) & 0x3ff;
                let g = (i.wrapping_mul(101).wrapping_add(7)) & 0x3ff;
                let b = (i.wrapping_mul(211).wrapping_add(3)) & 0x3ff;
                let a = i & 0x3;
                data.extend_from_slice(&pack_word::<A2B_R_SHIFT>(r, g, b, a));
            }
            let surf = a2_surface(data, wpx as u32, 1, (wpx * 4) as u32);
            assert_f32_close::<A2B_R_SHIFT, false>(&surf);
            assert_f32_close::<A2B_R_SHIFT, true>(&surf);
            assert_u32_bit_exact::<A2B_R_SHIFT, false>(&surf);
            assert_u32_bit_exact::<A2B_R_SHIFT, true>(&surf);
        }
    }

    #[test]
    fn stride_padding_is_skipped() {
        // 5×3 with a full extra pixel of junk padding per row.
        let w = 5u32;
        let h = 3u32;
        let stride = w * 4 + 4;
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let v = (y * w + x) & 0x3ff;
                data.extend_from_slice(&pack_word::<A2R_R_SHIFT>(
                    v,
                    (v + 1) & 0x3ff,
                    (v + 2) & 0x3ff,
                    x & 0x3,
                ));
            }
            data.extend_from_slice(&[0xEE, 0xEE, 0xEE, 0xEE]);
        }
        let surf = a2_surface(data, w, h, stride);
        assert_f32_close::<A2R_R_SHIFT, false>(&surf);
        assert_u32_bit_exact::<A2R_R_SHIFT, true>(&surf);
    }
}
