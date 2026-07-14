//! Packed 32-bit `A2B10G10R10` / `A2R10G10B10` storers.
//!
//! Encodes `[R, G, B, A]` lanes into one little-endian u32 per pixel: three
//! 10-bit channels plus a 2-bit alpha. G occupies the middle slot; the channel
//! order decides whether R goes to the low slot (`A2B10G10R10`) or the high
//! 10-bit slot (`A2R10G10B10`). Alpha is always the top 2 bits.
//!
//! UNORM/SNORM come from the f32 pipeline; UINT/SINT from the u32 pipeline.
//!
//! The SIMD kernels de-interleave 4/8/16 pixels into planar lane vectors,
//! encode each channel, then pack lane-wise into words at the *fixed* slot
//! positions (0/10/20/30), choosing which slot receives R vs. B from
//! `R_SHIFT`. UNORM/SNORM rounding is round-half-away-from-zero via
//! `trunc(x + copysign(0.5, x))`, which matches `f32::round` except for inputs
//! within 1 ULP of a halfway point, where intermediate rounding may tip the
//! result by one code. The native `_mm_round_ps` rounds ties to even instead,
//! which would diverge on the far more common exact-`.5` scale points.

use crate::processing::buffer::Buffer;
#[cfg(target_arch = "x86_64")]
use crate::processing::x86::{
    has_avx512, load_planar4_epi32, load_planar4_ps, load_planar8_epi32, load_planar8_ps,
    load_planar16_epi32, load_planar16_mask_epi32, load_planar16_mask_ps, load_planar16_ps,
};

use super::{write_pixels, write_pixels_u32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// R-channel bit offset for `A2B10G10R10` (R in the low 10-bit slot).
pub(crate) const A2B_R_SHIFT: u32 = 0;
/// R-channel bit offset for `A2R10G10B10` (R in the high 10-bit slot).
pub(crate) const A2R_R_SHIFT: u32 = 20;

/// Pack raw `(r, g, b, a)` fields (already masked to their bit widths) into a
/// word for the given R offset.
#[inline]
pub(crate) fn pack<const R_SHIFT: u32>(r: u32, g: u32, b: u32, a: u32) -> u32 {
    (a << 30) | (b << (20 - R_SHIFT)) | (g << 10) | (r << R_SHIFT)
}

#[inline]
fn unorm10(v: f32) -> u32 {
    (v.clamp(0.0, 1.0) * 1023.0).round() as u32 & 0x3ff
}

#[inline]
fn snorm10(v: f32) -> u32 {
    ((v.clamp(-1.0, 1.0) * 511.0).round() as i32) as u32 & 0x3ff
}

#[inline]
fn sint10(v: u32) -> u32 {
    ((v as i32).clamp(-512, 511)) as u32 & 0x3ff
}

// ---- Per-pixel scalar encode (golden reference, shared by serial + tails) ----

#[inline]
fn store_unorm_word<const R_SHIFT: u32>(lanes: &[f32; 4]) -> u32 {
    let r = unorm10(lanes[0]);
    let g = unorm10(lanes[1]);
    let b = unorm10(lanes[2]);
    let a = (lanes[3].clamp(0.0, 1.0) * 3.0).round() as u32 & 0x3;
    pack::<R_SHIFT>(r, g, b, a)
}

#[inline]
fn store_snorm_word<const R_SHIFT: u32>(lanes: &[f32; 4]) -> u32 {
    let r = snorm10(lanes[0]);
    let g = snorm10(lanes[1]);
    let b = snorm10(lanes[2]);
    // The 2-bit snorm scale factor is 1, so no multiply is needed.
    let a = (lanes[3].clamp(-1.0, 1.0).round() as i32) as u32 & 0x3;
    pack::<R_SHIFT>(r, g, b, a)
}

#[inline]
fn store_uint_word<const R_SHIFT: u32>(lanes: &[u32; 4]) -> u32 {
    let r = lanes[0].min(0x3ff);
    let g = lanes[1].min(0x3ff);
    let b = lanes[2].min(0x3ff);
    let a = lanes[3].min(0x3);
    pack::<R_SHIFT>(r, g, b, a)
}

#[inline]
fn store_sint_word<const R_SHIFT: u32>(lanes: &[u32; 4]) -> u32 {
    let r = sint10(lanes[0]);
    let g = sint10(lanes[1]);
    let b = sint10(lanes[2]);
    let a = ((lanes[3] as i32).clamp(-2, 1)) as u32 & 0x3;
    pack::<R_SHIFT>(r, g, b, a)
}

// ---- UNORM (f32 pipeline) ----

pub fn store_a2b10g10r10_unorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_a2b10g10r10_unorm_f32");
    store_f32_dispatch::<A2B_R_SHIFT, false>(buf)
}

pub fn store_a2r10g10b10_unorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_a2r10g10b10_unorm_f32");
    store_f32_dispatch::<A2R_R_SHIFT, false>(buf)
}

#[doc(hidden)]
pub fn store_a2_unorm_serial<const R_SHIFT: u32>(buf: &Buffer<f32>) -> Vec<u8> {
    write_pixels(buf, 1, 4, |lanes, bytes| {
        bytes.copy_from_slice(&store_unorm_word::<R_SHIFT>(lanes).to_le_bytes());
    })
}

// ---- SNORM (f32 pipeline) ----

pub fn store_a2b10g10r10_snorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_a2b10g10r10_snorm_f32");
    store_f32_dispatch::<A2B_R_SHIFT, true>(buf)
}

pub fn store_a2r10g10b10_snorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_a2r10g10b10_snorm_f32");
    store_f32_dispatch::<A2R_R_SHIFT, true>(buf)
}

#[doc(hidden)]
pub fn store_a2_snorm_serial<const R_SHIFT: u32>(buf: &Buffer<f32>) -> Vec<u8> {
    write_pixels(buf, 1, 4, |lanes, bytes| {
        bytes.copy_from_slice(&store_snorm_word::<R_SHIFT>(lanes).to_le_bytes());
    })
}

// ---- UINT (u32 pipeline) ----

pub fn store_a2b10g10r10_uint_u32(buf: &Buffer<u32>) -> Vec<u8> {
    profiling::scope!("store_a2b10g10r10_uint_u32");
    store_u32_dispatch::<A2B_R_SHIFT, false>(buf)
}

pub fn store_a2r10g10b10_uint_u32(buf: &Buffer<u32>) -> Vec<u8> {
    profiling::scope!("store_a2r10g10b10_uint_u32");
    store_u32_dispatch::<A2R_R_SHIFT, false>(buf)
}

#[doc(hidden)]
pub fn store_a2_uint_serial<const R_SHIFT: u32>(buf: &Buffer<u32>) -> Vec<u8> {
    write_pixels_u32(buf, 1, 4, |lanes, bytes| {
        bytes.copy_from_slice(&store_uint_word::<R_SHIFT>(lanes).to_le_bytes());
    })
}

// ---- SINT (u32 pipeline) ----

pub fn store_a2b10g10r10_sint_u32(buf: &Buffer<u32>) -> Vec<u8> {
    profiling::scope!("store_a2b10g10r10_sint_u32");
    store_u32_dispatch::<A2B_R_SHIFT, true>(buf)
}

pub fn store_a2r10g10b10_sint_u32(buf: &Buffer<u32>) -> Vec<u8> {
    profiling::scope!("store_a2r10g10b10_sint_u32");
    store_u32_dispatch::<A2R_R_SHIFT, true>(buf)
}

#[doc(hidden)]
pub fn store_a2_sint_serial<const R_SHIFT: u32>(buf: &Buffer<u32>) -> Vec<u8> {
    write_pixels_u32(buf, 1, 4, |lanes, bytes| {
        bytes.copy_from_slice(&store_sint_word::<R_SHIFT>(lanes).to_le_bytes());
    })
}

// ---- Runtime dispatch ----

#[inline]
fn store_f32_dispatch<const R_SHIFT: u32, const SNORM: bool>(buf: &Buffer<f32>) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: runtime check confirms avx512f + vl + bw are available.
            return unsafe { store_a2_f32_avx512::<R_SHIFT, SNORM>(buf) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime check confirms avx2 + fma are available.
            return unsafe { store_a2_f32_avx2_fma::<R_SHIFT, SNORM>(buf) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { store_a2_f32_sse4_1::<R_SHIFT, SNORM>(buf) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime check confirms NEON is available.
            return unsafe { store_a2_f32_neon::<R_SHIFT, SNORM>(buf) };
        }
    }

    if SNORM {
        store_a2_snorm_serial::<R_SHIFT>(buf)
    } else {
        store_a2_unorm_serial::<R_SHIFT>(buf)
    }
}

#[inline]
fn store_u32_dispatch<const R_SHIFT: u32, const SINT: bool>(buf: &Buffer<u32>) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: runtime check confirms avx512f + vl + bw are available.
            return unsafe { store_a2_u32_avx512::<R_SHIFT, SINT>(buf) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime check confirms avx2 is available.
            return unsafe { store_a2_u32_avx2::<R_SHIFT, SINT>(buf) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { store_a2_u32_sse4_1::<R_SHIFT, SINT>(buf) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime check confirms NEON is available.
            return unsafe { store_a2_u32_neon::<R_SHIFT, SINT>(buf) };
        }
    }

    if SINT {
        store_a2_sint_serial::<R_SHIFT>(buf)
    } else {
        store_a2_uint_serial::<R_SHIFT>(buf)
    }
}

// =====================================================================
// x86_64 SIMD
// =====================================================================

// ---- SSE4.1 ----

/// Round each lane to the nearest integer, ties away from zero, via
/// `trunc(x + copysign(0.5, x))` — matches `f32::round` except within 1 ULP of
/// halfway points (see the module doc).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn round_ties_away_sse4(x: __m128) -> __m128i {
    let sign = _mm_and_ps(x, _mm_set1_ps(-0.0));
    let biased = _mm_add_ps(x, _mm_or_ps(sign, _mm_set1_ps(0.5)));
    _mm_cvttps_epi32(biased)
}

/// Clamp, scale, round and mask a float channel plane into a packed field.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn encode_f32_field_sse4<const SNORM: bool>(x: __m128, scale: f32, mask: i32) -> __m128i {
    let lo = _mm_set1_ps(if SNORM { -1.0 } else { 0.0 });
    let hi = _mm_set1_ps(1.0);
    let clamped = _mm_min_ps(_mm_max_ps(x, lo), hi);
    let scaled = _mm_mul_ps(clamped, _mm_set1_ps(scale));
    _mm_and_si128(round_ties_away_sse4(scaled), _mm_set1_epi32(mask))
}

/// Unsigned-min a u32 channel plane into a packed field.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn encode_uint_field_sse4(x: __m128i, maxv: i32) -> __m128i {
    _mm_min_epu32(x, _mm_set1_epi32(maxv))
}

/// Signed-clamp and mask a u32 channel plane into a packed field.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn encode_sint_field_sse4(x: __m128i, lo: i32, hi: i32, mask: i32) -> __m128i {
    let clamped = _mm_max_epi32(_mm_min_epi32(x, _mm_set1_epi32(hi)), _mm_set1_epi32(lo));
    _mm_and_si128(clamped, _mm_set1_epi32(mask))
}

/// Pack four masked planar field vectors into a word vector (one word/pixel).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn pack_sse4<const R_SHIFT: u32>(r: __m128i, g: __m128i, b: __m128i, a: __m128i) -> __m128i {
    let (low, high) = if R_SHIFT == 0 { (r, b) } else { (b, r) };
    _mm_or_si128(
        _mm_or_si128(_mm_slli_epi32::<30>(a), _mm_slli_epi32::<20>(high)),
        _mm_or_si128(_mm_slli_epi32::<10>(g), low),
    )
}

/// Encode 4 interleaved f32 pixels at `src` into 4 packed words at `out`.
///
/// # Safety
/// * SSE4.1 must be available; `src` valid for 16 f32, `out` for 4 u32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_block4_f32_sse4<const R_SHIFT: u32, const SNORM: bool>(
    src: *const f32,
    out: *mut u32,
) {
    // SAFETY: caller guarantees the read/write ranges.
    unsafe {
        let (r, g, b, a) = load_planar4_ps(src);
        let rgb_scale = if SNORM { 511.0 } else { 1023.0 };
        let a_scale = if SNORM { 1.0 } else { 3.0 };
        let ri = encode_f32_field_sse4::<SNORM>(r, rgb_scale, 0x3ff);
        let gi = encode_f32_field_sse4::<SNORM>(g, rgb_scale, 0x3ff);
        let bi = encode_f32_field_sse4::<SNORM>(b, rgb_scale, 0x3ff);
        let ai = encode_f32_field_sse4::<SNORM>(a, a_scale, 0x3);
        _mm_storeu_si128(out as *mut __m128i, pack_sse4::<R_SHIFT>(ri, gi, bi, ai));
    }
}

/// Encode 4 interleaved u32 pixels at `src` into 4 packed words at `out`.
///
/// # Safety
/// * SSE4.1 must be available; `src` valid for 16 u32, `out` for 4 u32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_block4_u32_sse4<const R_SHIFT: u32, const SINT: bool>(
    src: *const u32,
    out: *mut u32,
) {
    // SAFETY: caller guarantees the read/write ranges.
    unsafe {
        let (r, g, b, a) = load_planar4_epi32(src);
        let (ri, gi, bi, ai) = if SINT {
            (
                encode_sint_field_sse4(r, -512, 511, 0x3ff),
                encode_sint_field_sse4(g, -512, 511, 0x3ff),
                encode_sint_field_sse4(b, -512, 511, 0x3ff),
                encode_sint_field_sse4(a, -2, 1, 0x3),
            )
        } else {
            (
                encode_uint_field_sse4(r, 0x3ff),
                encode_uint_field_sse4(g, 0x3ff),
                encode_uint_field_sse4(b, 0x3ff),
                encode_uint_field_sse4(a, 0x3),
            )
        };
        _mm_storeu_si128(out as *mut __m128i, pack_sse4::<R_SHIFT>(ri, gi, bi, ai));
    }
}

/// Encode an arbitrary f32 remainder with 4-pixel SSE blocks and one padded tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_remainder_f32_sse4<const R_SHIFT: u32, const SNORM: bool>(
    mut src: *const f32,
    mut dst: *mut u32,
    mut count: usize,
) {
    while count >= 4 {
        unsafe { store_block4_f32_sse4::<R_SHIFT, SNORM>(src, dst) };
        src = unsafe { src.add(16) };
        dst = unsafe { dst.add(4) };
        count -= 4;
    }
    if count > 0 {
        let mut padded = [0.0f32; 16];
        unsafe { std::ptr::copy_nonoverlapping(src, padded.as_mut_ptr(), count * 4) };
        let mut words = [0u32; 4];
        unsafe {
            store_block4_f32_sse4::<R_SHIFT, SNORM>(padded.as_ptr(), words.as_mut_ptr());
            std::ptr::copy_nonoverlapping(words.as_ptr() as *const u8, dst as *mut u8, count * 4);
        }
    }
}

/// Encode an arbitrary u32 remainder with 4-pixel SSE blocks and one padded tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_remainder_u32_sse4<const R_SHIFT: u32, const SINT: bool>(
    mut src: *const u32,
    mut dst: *mut u32,
    mut count: usize,
) {
    while count >= 4 {
        unsafe { store_block4_u32_sse4::<R_SHIFT, SINT>(src, dst) };
        src = unsafe { src.add(16) };
        dst = unsafe { dst.add(4) };
        count -= 4;
    }
    if count > 0 {
        let mut padded = [0u32; 16];
        unsafe { std::ptr::copy_nonoverlapping(src, padded.as_mut_ptr(), count * 4) };
        let mut words = [0u32; 4];
        unsafe {
            store_block4_u32_sse4::<R_SHIFT, SINT>(padded.as_ptr(), words.as_mut_ptr());
            std::ptr::copy_nonoverlapping(words.as_ptr() as *const u8, dst as *mut u8, count * 4);
        }
    }
}

/// SSE4.1 storer for the f32 (UNORM/SNORM) pipeline. **Not part of the public API.**
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn store_a2_f32_sse4_1<const R_SHIFT: u32, const SNORM: bool>(
    buf: &Buffer<f32>,
) -> Vec<u8> {
    profiling::scope!("store_a2_f32_sse4_1");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: sse4.1 is available per the enclosing fn's contract; the
        // helper writes all `n` words.
        unsafe { store_remainder_f32_sse4::<R_SHIFT, SNORM>(src, dst, n) }
    })
}

/// SSE4.1 storer for the u32 (UINT/SINT) pipeline. **Not part of the public API.**
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn store_a2_u32_sse4_1<const R_SHIFT: u32, const SINT: bool>(
    buf: &Buffer<u32>,
) -> Vec<u8> {
    profiling::scope!("store_a2_u32_sse4_1");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: sse4.1 is available per the enclosing fn's contract; the
        // helper writes all `n` words.
        unsafe { store_remainder_u32_sse4::<R_SHIFT, SINT>(src, dst, n) }
    })
}

// ---- AVX2 ----

/// Clamp, then fuse `clamp*scale + copysign(0.5, clamp)` and truncate — the
/// FMA form of round-half-away-from-zero (`sign(clamp*scale) == sign(clamp)`
/// since `scale > 0`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn encode_f32_field_avx2<const SNORM: bool>(x: __m256, scale: f32, mask: i32) -> __m256i {
    let lo = _mm256_set1_ps(if SNORM { -1.0 } else { 0.0 });
    let hi = _mm256_set1_ps(1.0);
    let clamped = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    let bias = _mm256_or_ps(
        _mm256_and_ps(clamped, _mm256_set1_ps(-0.0)),
        _mm256_set1_ps(0.5),
    );
    let rounded = _mm256_cvttps_epi32(_mm256_fmadd_ps(clamped, _mm256_set1_ps(scale), bias));
    _mm256_and_si256(rounded, _mm256_set1_epi32(mask))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn encode_uint_field_avx2(x: __m256i, maxv: i32) -> __m256i {
    _mm256_min_epu32(x, _mm256_set1_epi32(maxv))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn encode_sint_field_avx2(x: __m256i, lo: i32, hi: i32, mask: i32) -> __m256i {
    let clamped = _mm256_max_epi32(
        _mm256_min_epi32(x, _mm256_set1_epi32(hi)),
        _mm256_set1_epi32(lo),
    );
    _mm256_and_si256(clamped, _mm256_set1_epi32(mask))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn pack_avx2<const R_SHIFT: u32>(r: __m256i, g: __m256i, b: __m256i, a: __m256i) -> __m256i {
    let (low, high) = if R_SHIFT == 0 { (r, b) } else { (b, r) };
    _mm256_or_si256(
        _mm256_or_si256(_mm256_slli_epi32::<30>(a), _mm256_slli_epi32::<20>(high)),
        _mm256_or_si256(_mm256_slli_epi32::<10>(g), low),
    )
}

/// Encode 8 interleaved f32 pixels at `src` into 8 packed words at `out`.
///
/// # Safety
/// * AVX2 + FMA must be available; `src` valid for 32 f32, `out` for 8 u32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_block8_f32_avx2<const R_SHIFT: u32, const SNORM: bool>(
    src: *const f32,
    out: *mut u32,
) {
    // SAFETY: caller guarantees the read/write ranges.
    unsafe {
        let (r, g, b, a) = load_planar8_ps(src);
        let rgb_scale = if SNORM { 511.0 } else { 1023.0 };
        let a_scale = if SNORM { 1.0 } else { 3.0 };
        let ri = encode_f32_field_avx2::<SNORM>(r, rgb_scale, 0x3ff);
        let gi = encode_f32_field_avx2::<SNORM>(g, rgb_scale, 0x3ff);
        let bi = encode_f32_field_avx2::<SNORM>(b, rgb_scale, 0x3ff);
        let ai = encode_f32_field_avx2::<SNORM>(a, a_scale, 0x3);
        _mm256_storeu_si256(out as *mut __m256i, pack_avx2::<R_SHIFT>(ri, gi, bi, ai));
    }
}

/// Encode 8 interleaved u32 pixels at `src` into 8 packed words at `out`.
///
/// # Safety
/// * AVX2 must be available; `src` valid for 32 u32, `out` for 8 u32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_block8_u32_avx2<const R_SHIFT: u32, const SINT: bool>(
    src: *const u32,
    out: *mut u32,
) {
    // SAFETY: caller guarantees the read/write ranges.
    unsafe {
        let (r, g, b, a) = load_planar8_epi32(src);
        let (ri, gi, bi, ai) = if SINT {
            (
                encode_sint_field_avx2(r, -512, 511, 0x3ff),
                encode_sint_field_avx2(g, -512, 511, 0x3ff),
                encode_sint_field_avx2(b, -512, 511, 0x3ff),
                encode_sint_field_avx2(a, -2, 1, 0x3),
            )
        } else {
            (
                encode_uint_field_avx2(r, 0x3ff),
                encode_uint_field_avx2(g, 0x3ff),
                encode_uint_field_avx2(b, 0x3ff),
                encode_uint_field_avx2(a, 0x3),
            )
        };
        _mm256_storeu_si256(out as *mut __m256i, pack_avx2::<R_SHIFT>(ri, gi, bi, ai));
    }
}

/// AVX2+FMA storer for the f32 (UNORM/SNORM) pipeline. **Not part of the public API.**
///
/// # Safety
/// * AVX2 + FMA must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn store_a2_f32_avx2_fma<const R_SHIFT: u32, const SNORM: bool>(
    buf: &Buffer<f32>,
) -> Vec<u8> {
    profiling::scope!("store_a2_f32_avx2_fma");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: avx2+fma are available per the enclosing fn's contract (avx2
        // implies sse4.1 for the remainder); all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 8 <= n {
                store_block8_f32_avx2::<R_SHIFT, SNORM>(src.add(i * 4), dst.add(i));
                i += 8;
            }
            store_remainder_f32_sse4::<R_SHIFT, SNORM>(src.add(i * 4), dst.add(i), n - i);
        }
    })
}

/// AVX2 storer for the u32 (UINT/SINT) pipeline. **Not part of the public API.**
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn store_a2_u32_avx2<const R_SHIFT: u32, const SINT: bool>(
    buf: &Buffer<u32>,
) -> Vec<u8> {
    profiling::scope!("store_a2_u32_avx2");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: avx2 is available per the enclosing fn's contract (avx2
        // implies sse4.1 for the remainder); all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 8 <= n {
                store_block8_u32_avx2::<R_SHIFT, SINT>(src.add(i * 4), dst.add(i));
                i += 8;
            }
            store_remainder_u32_sse4::<R_SHIFT, SINT>(src.add(i * 4), dst.add(i), n - i);
        }
    })
}

// ---- AVX-512 ----

/// Clamp, then fuse `clamp*scale + copysign(0.5, clamp)` and truncate — the
/// FMA form of round-half-away-from-zero. The sign is applied in the integer
/// domain to avoid needing AVX-512 DQ (`_mm512_and_ps`). UNORM clamps to
/// `[+0.0, 1.0]` (`maxps` returns its second operand on `-0.0` and NaN), so
/// its bias is a plain `+0.5`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn encode_f32_field_avx512<const SNORM: bool>(x: __m512, scale: f32, mask: i32) -> __m512i {
    let lo = _mm512_set1_ps(if SNORM { -1.0 } else { 0.0 });
    let hi = _mm512_set1_ps(1.0);
    let clamped = _mm512_min_ps(_mm512_max_ps(x, lo), hi);
    let half = _mm512_set1_ps(0.5);
    let bias = if SNORM {
        let signbit = _mm512_set1_epi32(0x8000_0000u32 as i32);
        let sign = _mm512_and_si512(_mm512_castps_si512(clamped), signbit);
        _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_castps_si512(half)))
    } else {
        half
    };
    let rounded = _mm512_cvttps_epi32(_mm512_fmadd_ps(clamped, _mm512_set1_ps(scale), bias));
    _mm512_and_si512(rounded, _mm512_set1_epi32(mask))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn encode_uint_field_avx512(x: __m512i, maxv: i32) -> __m512i {
    _mm512_min_epu32(x, _mm512_set1_epi32(maxv))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn encode_sint_field_avx512(x: __m512i, lo: i32, hi: i32, mask: i32) -> __m512i {
    let clamped = _mm512_max_epi32(
        _mm512_min_epi32(x, _mm512_set1_epi32(hi)),
        _mm512_set1_epi32(lo),
    );
    _mm512_and_si512(clamped, _mm512_set1_epi32(mask))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn pack_avx512<const R_SHIFT: u32>(r: __m512i, g: __m512i, b: __m512i, a: __m512i) -> __m512i {
    let (low, high) = if R_SHIFT == 0 { (r, b) } else { (b, r) };
    // 0xFE = three-way OR.
    let hi3 = _mm512_ternarylogic_epi32::<0xFE>(
        _mm512_slli_epi32::<30>(a),
        _mm512_slli_epi32::<20>(high),
        _mm512_slli_epi32::<10>(g),
    );
    _mm512_or_si512(hi3, low)
}

/// Encode 16 pixels' planar f32 planes into 16 packed words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn encode16_f32_avx512<const R_SHIFT: u32, const SNORM: bool>(
    r: __m512,
    g: __m512,
    b: __m512,
    a: __m512,
) -> __m512i {
    let rgb_scale = if SNORM { 511.0 } else { 1023.0 };
    let a_scale = if SNORM { 1.0 } else { 3.0 };
    let ri = encode_f32_field_avx512::<SNORM>(r, rgb_scale, 0x3ff);
    let gi = encode_f32_field_avx512::<SNORM>(g, rgb_scale, 0x3ff);
    let bi = encode_f32_field_avx512::<SNORM>(b, rgb_scale, 0x3ff);
    let ai = encode_f32_field_avx512::<SNORM>(a, a_scale, 0x3);
    pack_avx512::<R_SHIFT>(ri, gi, bi, ai)
}

/// Encode 16 pixels' planar u32 planes into 16 packed words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn encode16_u32_avx512<const R_SHIFT: u32, const SINT: bool>(
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
) -> __m512i {
    let (ri, gi, bi, ai) = if SINT {
        (
            encode_sint_field_avx512(r, -512, 511, 0x3ff),
            encode_sint_field_avx512(g, -512, 511, 0x3ff),
            encode_sint_field_avx512(b, -512, 511, 0x3ff),
            encode_sint_field_avx512(a, -2, 1, 0x3),
        )
    } else {
        (
            encode_uint_field_avx512(r, 0x3ff),
            encode_uint_field_avx512(g, 0x3ff),
            encode_uint_field_avx512(b, 0x3ff),
            encode_uint_field_avx512(a, 0x3),
        )
    };
    pack_avx512::<R_SHIFT>(ri, gi, bi, ai)
}

/// AVX-512 storer for the f32 (UNORM/SNORM) pipeline. **Not part of the public API.**
///
/// # Safety
/// * AVX-512 F/VL/BW must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn store_a2_f32_avx512<const R_SHIFT: u32, const SNORM: bool>(
    buf: &Buffer<f32>,
) -> Vec<u8> {
    profiling::scope!("store_a2_f32_avx512");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: avx512 is available per the enclosing fn's contract; masked
        // loads/stores only touch enabled lanes; all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 16 <= n {
                let (r, g, b, a) = load_planar16_ps(src.add(i * 4));
                let words = encode16_f32_avx512::<R_SHIFT, SNORM>(r, g, b, a);
                _mm512_storeu_si512(dst.add(i) as *mut __m512i, words);
                i += 16;
            }
            if i < n {
                let tail = n - i;
                let (r, g, b, a) = load_planar16_mask_ps(src.add(i * 4), tail);
                let words = encode16_f32_avx512::<R_SHIFT, SNORM>(r, g, b, a);
                _mm512_mask_storeu_epi32(dst.add(i) as *mut i32, (1u16 << tail) - 1, words);
            }
        }
    })
}

/// AVX-512 storer for the u32 (UINT/SINT) pipeline. **Not part of the public API.**
///
/// # Safety
/// * AVX-512 F/VL/BW must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn store_a2_u32_avx512<const R_SHIFT: u32, const SINT: bool>(
    buf: &Buffer<u32>,
) -> Vec<u8> {
    profiling::scope!("store_a2_u32_avx512");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: avx512 is available per the enclosing fn's contract; masked
        // loads/stores only touch enabled lanes; all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 16 <= n {
                let (r, g, b, a) = load_planar16_epi32(src.add(i * 4));
                let words = encode16_u32_avx512::<R_SHIFT, SINT>(r, g, b, a);
                _mm512_storeu_si512(dst.add(i) as *mut __m512i, words);
                i += 16;
            }
            if i < n {
                let tail = n - i;
                let (r, g, b, a) = load_planar16_mask_epi32(src.add(i * 4), tail);
                let words = encode16_u32_avx512::<R_SHIFT, SINT>(r, g, b, a);
                _mm512_mask_storeu_epi32(dst.add(i) as *mut i32, (1u16 << tail) - 1, words);
            }
        }
    })
}

// =====================================================================
// AArch64 NEON (compile-guarded; mirrors the scalar reference)
// =====================================================================

/// Clamp, then fuse `clamp*scale + copysign(0.5, clamp)` and truncate — the
/// FMA form of round-half-away-from-zero — and mask into a packed field.
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn encode_f32_field_neon<const SNORM: bool>(
    x: float32x4_t,
    scale: f32,
    mask: u32,
) -> uint32x4_t {
    let lo = vdupq_n_f32(if SNORM { -1.0 } else { 0.0 });
    let hi = vdupq_n_f32(1.0);
    let clamped = vminq_f32(vmaxq_f32(x, lo), hi);
    let bias = vbslq_f32(vdupq_n_u32(0x8000_0000), clamped, vdupq_n_f32(0.5));
    let rounded = vcvtq_s32_f32(vfmaq_f32(bias, clamped, vdupq_n_f32(scale)));
    vandq_u32(vreinterpretq_u32_s32(rounded), vdupq_n_u32(mask))
}

/// Signed-clamp and mask a u32 channel plane into a packed field (NEON).
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn encode_sint_field_neon(x: uint32x4_t, lo: i32, hi: i32, mask: u32) -> uint32x4_t {
    let xi = vreinterpretq_s32_u32(x);
    let clamped = vmaxq_s32(vminq_s32(xi, vdupq_n_s32(hi)), vdupq_n_s32(lo));
    vandq_u32(vreinterpretq_u32_s32(clamped), vdupq_n_u32(mask))
}

/// Pack four masked planar field vectors into a word vector (NEON).
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn pack_neon<const R_SHIFT: u32>(
    r: uint32x4_t,
    g: uint32x4_t,
    b: uint32x4_t,
    a: uint32x4_t,
) -> uint32x4_t {
    let (low, high) = if R_SHIFT == 0 { (r, b) } else { (b, r) };
    vorrq_u32(
        vorrq_u32(vshlq_n_u32::<30>(a), vshlq_n_u32::<20>(high)),
        vorrq_u32(vshlq_n_u32::<10>(g), low),
    )
}

/// NEON storer for the f32 (UNORM/SNORM) pipeline. **Not part of the public API.**
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn store_a2_f32_neon<const R_SHIFT: u32, const SNORM: bool>(
    buf: &Buffer<f32>,
) -> Vec<u8> {
    profiling::scope!("store_a2_f32_neon");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: NEON is available per the enclosing fn's contract; all `n`
        // words get written.
        unsafe {
            let mut i = 0usize;
            while i + 4 <= n {
                let px = vld4q_f32(src.add(i * 4));
                let rgb_scale = if SNORM { 511.0 } else { 1023.0 };
                let a_scale = if SNORM { 1.0 } else { 3.0 };
                let ri = encode_f32_field_neon::<SNORM>(px.0, rgb_scale, 0x3ff);
                let gi = encode_f32_field_neon::<SNORM>(px.1, rgb_scale, 0x3ff);
                let bi = encode_f32_field_neon::<SNORM>(px.2, rgb_scale, 0x3ff);
                let ai = encode_f32_field_neon::<SNORM>(px.3, a_scale, 0x3);
                vst1q_u32(dst.add(i), pack_neon::<R_SHIFT>(ri, gi, bi, ai));
                i += 4;
            }
            while i < n {
                let word = if SNORM {
                    store_snorm_word::<R_SHIFT>(&buf.pixels[i])
                } else {
                    store_unorm_word::<R_SHIFT>(&buf.pixels[i])
                };
                dst.add(i).write_unaligned(word);
                i += 1;
            }
        }
    })
}

/// NEON storer for the u32 (UINT/SINT) pipeline. **Not part of the public API.**
///
/// # Safety
/// * NEON must be available (enforced by `target_feature`).
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn store_a2_u32_neon<const R_SHIFT: u32, const SINT: bool>(
    buf: &Buffer<u32>,
) -> Vec<u8> {
    profiling::scope!("store_a2_u32_neon");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: NEON is available per the enclosing fn's contract; all `n`
        // words get written.
        unsafe {
            let mut i = 0usize;
            while i + 4 <= n {
                let px = vld4q_u32(src.add(i * 4));
                let (ri, gi, bi, ai) = if SINT {
                    (
                        encode_sint_field_neon(px.0, -512, 511, 0x3ff),
                        encode_sint_field_neon(px.1, -512, 511, 0x3ff),
                        encode_sint_field_neon(px.2, -512, 511, 0x3ff),
                        encode_sint_field_neon(px.3, -2, 1, 0x3),
                    )
                } else {
                    (
                        vminq_u32(px.0, vdupq_n_u32(0x3ff)),
                        vminq_u32(px.1, vdupq_n_u32(0x3ff)),
                        vminq_u32(px.2, vdupq_n_u32(0x3ff)),
                        vminq_u32(px.3, vdupq_n_u32(0x3)),
                    )
                };
                vst1q_u32(dst.add(i), pack_neon::<R_SHIFT>(ri, gi, bi, ai));
                i += 4;
            }
            while i < n {
                let word = if SINT {
                    store_sint_word::<R_SHIFT>(&buf.pixels[i])
                } else {
                    store_uint_word::<R_SHIFT>(&buf.pixels[i])
                };
                dst.add(i).write_unaligned(word);
                i += 1;
            }
        }
    })
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    fn buf_f32(pixels: Vec<[f32; 4]>) -> Buffer<f32> {
        let width = pixels.len() as u32;
        Buffer {
            pixels,
            width,
            height: 1,
        }
    }

    fn buf_u32(pixels: Vec<[u32; 4]>) -> Buffer<u32> {
        let width = pixels.len() as u32;
        Buffer {
            pixels,
            width,
            height: 1,
        }
    }

    fn assert_f32_bit_exact<const R_SHIFT: u32, const SNORM: bool>(buf: &Buffer<f32>) {
        let scalar = if SNORM {
            store_a2_snorm_serial::<R_SHIFT>(buf)
        } else {
            store_a2_unorm_serial::<R_SHIFT>(buf)
        };
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
                let simd = unsafe { store_a2_f32_sse4_1::<R_SHIFT, SNORM>(buf) };
                assert_eq!(simd, scalar, "sse4.1 f32 store mismatch");
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd = unsafe { store_a2_f32_avx2_fma::<R_SHIFT, SNORM>(buf) };
                assert_eq!(simd, scalar, "avx2 f32 store mismatch");
            }
            if has_avx512() {
                let simd = unsafe { store_a2_f32_avx512::<R_SHIFT, SNORM>(buf) };
                assert_eq!(simd, scalar, "avx512 f32 store mismatch");
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            let simd = unsafe { store_a2_f32_neon::<R_SHIFT, SNORM>(buf) };
            assert_eq!(simd, scalar, "neon f32 store mismatch");
        }
    }

    fn assert_u32_bit_exact<const R_SHIFT: u32, const SINT: bool>(buf: &Buffer<u32>) {
        let scalar = if SINT {
            store_a2_sint_serial::<R_SHIFT>(buf)
        } else {
            store_a2_uint_serial::<R_SHIFT>(buf)
        };
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
                let simd = unsafe { store_a2_u32_sse4_1::<R_SHIFT, SINT>(buf) };
                assert_eq!(simd, scalar, "sse4.1 u32 store mismatch");
            }
            if is_x86_feature_detected!("avx2") {
                let simd = unsafe { store_a2_u32_avx2::<R_SHIFT, SINT>(buf) };
                assert_eq!(simd, scalar, "avx2 u32 store mismatch");
            }
            if has_avx512() {
                let simd = unsafe { store_a2_u32_avx512::<R_SHIFT, SINT>(buf) };
                assert_eq!(simd, scalar, "avx512 u32 store mismatch");
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            let simd = unsafe { store_a2_u32_neon::<R_SHIFT, SINT>(buf) };
            assert_eq!(simd, scalar, "neon u32 store mismatch");
        }
    }

    /// A wide sweep of float inputs that lands each channel on exact halfway
    /// values (to exercise round-half-away-from-zero), out-of-range clamps, and
    /// ordinary values — for both UNORM and SNORM scales.
    fn f32_sweep() -> Vec<[f32; 4]> {
        let mut pixels = Vec::new();
        // Exact halfway points for the UNORM scale (k+0.5)/1023 rounds to k+1.
        for k in 0..1023u32 {
            let v = (k as f32 + 0.5) / 1023.0;
            pixels.push([v, v, v, ((k % 3) as f32 + 0.5) / 3.0]);
        }
        // Exact halfway points for the SNORM scale (±(k+0.5)/511).
        for k in 0..511u32 {
            let v = (k as f32 + 0.5) / 511.0;
            pixels.push([v, -v, v, 0.5]);
            pixels.push([-v, v, -v, -0.5]);
        }
        // Out-of-range and boundary values.
        for &v in &[
            -2.0f32,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            f32::MIN,
            f32::MAX,
        ] {
            pixels.push([v, v, v, v]);
        }
        pixels
    }

    fn u32_sweep() -> Vec<[u32; 4]> {
        let mut pixels = Vec::new();
        for field in 0u32..1024 {
            for a in 0u32..4 {
                pixels.push([field, (field ^ 0x155) & 0xffff, field.wrapping_add(700), a]);
            }
        }
        // Values above the 10-bit range to exercise clamping/masking.
        for &v in &[1023u32, 1024, 2000, 0xffff_ffff, 0x8000_0000, 512, 511] {
            pixels.push([v, v, v, v & 0xf]);
        }
        pixels
    }

    #[test]
    fn f32_stores_are_bit_exact_for_both_orders() {
        let buf = buf_f32(f32_sweep());
        assert_f32_bit_exact::<A2B_R_SHIFT, false>(&buf);
        assert_f32_bit_exact::<A2B_R_SHIFT, true>(&buf);
        assert_f32_bit_exact::<A2R_R_SHIFT, false>(&buf);
        assert_f32_bit_exact::<A2R_R_SHIFT, true>(&buf);
    }

    #[test]
    fn u32_stores_bit_exact_both_orders() {
        let buf = buf_u32(u32_sweep());
        assert_u32_bit_exact::<A2B_R_SHIFT, false>(&buf);
        assert_u32_bit_exact::<A2B_R_SHIFT, true>(&buf);
        assert_u32_bit_exact::<A2R_R_SHIFT, false>(&buf);
        assert_u32_bit_exact::<A2R_R_SHIFT, true>(&buf);
    }

    #[test]
    fn odd_counts_exercise_all_tails() {
        for &count in &[1usize, 3, 5, 7, 11, 13, 17, 19, 23, 31, 33] {
            let f: Vec<[f32; 4]> = (0..count)
                .map(|i| {
                    let x = i as f32;
                    [x * 0.013, 1.0 - x * 0.017, x * 0.031, (x * 0.007) % 1.0]
                })
                .collect();
            let buf = buf_f32(f);
            assert_f32_bit_exact::<A2B_R_SHIFT, false>(&buf);
            assert_f32_bit_exact::<A2R_R_SHIFT, true>(&buf);

            let u: Vec<[u32; 4]> = (0..count as u32)
                .map(|i| [i * 37, i * 101 + 7, i * 211 + 3, i])
                .collect();
            let buf = buf_u32(u);
            assert_u32_bit_exact::<A2B_R_SHIFT, false>(&buf);
            assert_u32_bit_exact::<A2R_R_SHIFT, true>(&buf);
        }
    }
}
