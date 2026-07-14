//! `B10G11R11_UFLOAT_PACK32` loader (packed unsigned floats).
//!
//! One little-endian u32 per pixel: R is an 11-bit float at bits `[0,11)`, G
//! an 11-bit float at `[11,22)`, B a 10-bit float at `[22,32)`. All are
//! unsigned (no sign bit) with a 5-bit exponent (bias 15); the 11-bit forms
//! carry 6 mantissa bits, the 10-bit form 5. Alpha loads as 1.0.

use crate::error::Result;
#[cfg(target_arch = "x86_64")]
use crate::processing::x86::{
    has_avx512, store_interleaved_ps, store_interleaved16_mask_ps, store_interleaved16_ps,
};
use crate::surface::Surface;

use super::{Buffer, read_pixels_f32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Decode an unsigned float with `MANT_BITS` mantissa bits and a 5-bit,
/// bias-15 exponent into f32. Matches the IEEE small-float rules used by
/// `B10G11R11`: exp 0 is zero/denormal, exp 31 is Inf/NaN.
#[inline]
pub(crate) fn decode_small_float<const MANT_BITS: u32>(v: u32) -> f32 {
    let mant_max = 1u32 << MANT_BITS;
    let exp = (v >> MANT_BITS) & 0x1f;
    let mant = v & (mant_max - 1);
    if exp == 0 {
        // Zero (mant == 0) or denormal: 2^-14 * (mant / 2^MANT_BITS).
        f32::from_bits(0x3800_0000) * (mant as f32 / mant_max as f32) // 0x3800_0000 = 2^-14
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        // Normal: 2^(exp - 15) * (1 + mant / 2^MANT_BITS).
        let scale = f32::from_bits((exp + 112) << 23); // 2^(exp - 15)
        scale * (1.0 + mant as f32 / mant_max as f32)
    }
}

/// Decode one packed word into linear `[R, G, B]` (alpha handled by the caller).
#[inline]
pub(crate) fn decode(word: u32) -> [f32; 3] {
    let r = decode_small_float::<6>(word & 0x7ff);
    let g = decode_small_float::<6>((word >> 11) & 0x7ff);
    let b = decode_small_float::<5>((word >> 22) & 0x3ff);
    [r, g, b]
}

pub fn load_b10g11r11_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_b10g11r11_f32");

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: runtime check confirms avx512f + vl + bw are available.
            return unsafe { load_b10g11r11_f32_avx512(surface) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime check confirms avx2 + fma are available.
            return unsafe { load_b10g11r11_f32_avx2_fma(surface) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { load_b10g11r11_f32_sse4_1(surface) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime check confirms NEON is available.
            return unsafe { load_b10g11r11_f32_neon(surface) };
        }
    }

    load_b10g11r11_f32_serial(surface)
}

/// Serial reference path.
///
/// **Not part of the public API.** Exposed for benches/tests to compare
/// against the SIMD kernels.
#[doc(hidden)]
pub fn load_b10g11r11_f32_serial(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_b10g11r11_f32_serial");
    read_pixels_f32(surface, 1, 4, |bytes, lanes| {
        let word = u32::from_le_bytes(<[u8; 4]>::try_from(bytes).expect("4-byte pixel"));
        let [r, g, b] = decode(word);
        lanes[0] = r;
        lanes[1] = g;
        lanes[2] = b;
        // lanes[3] stays at the default 1.0.
    })
}

// ---- x86_64 SIMD ----
//
// All SIMD paths replicate `decode_small_float` branchlessly per channel:
//   * `frac = mant * (1 / 2^MANT_BITS)` (exact, `1 / 2^MANT_BITS` is a power of
//     two)
//   * denormal form `2^-14 * frac`, normal form `2^(exp-15) * (1 + frac)`
//   * select the denormal form where `exp == 0`
//   * overlay Inf (mant == 0) or NaN (mant != 0) where `exp == 31`
// FMA tiers (avx2_fma, avx512, neon) fuse the normal form as `fma(scale, frac,
// scale)`, so their normal lanes may differ from the scalar path by <=1 ULP;
// the SSE4.1 tier keeps the separate multiply and stays bit-identical.
//
// Channels are decoded SoA (one vector per channel across N pixels), then a
// 4x4 transpose interleaves them back into `[R, G, B, A]` pixels with A = 1.0.

/// Bit pattern of `+Inf`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const F32_INF_BITS: u32 = 0x7f80_0000;
/// Bit pattern of a quiet NaN (matches `f32::NAN`).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const F32_NAN_BITS: u32 = 0x7fc0_0000;
/// Bit pattern of `2^-14`, the smallest positive normal small-float value.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const F32_2POW_M14_BITS: u32 = 0x3800_0000;

/// Decode a vector of right-aligned channel codes (uniform `M` mantissa bits).
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn decode_codes_sse4_1<const M: i32>(codes: __m128i) -> __m128 {
    let mant_max = 1i32 << M;
    let mant = _mm_and_si128(codes, _mm_set1_epi32(mant_max - 1));
    let exp = _mm_and_si128(_mm_srli_epi32::<M>(codes), _mm_set1_epi32(0x1f));
    let mant_f = _mm_cvtepi32_ps(mant);
    let frac = _mm_mul_ps(mant_f, _mm_set1_ps(1.0 / mant_max as f32));
    let denorm = _mm_mul_ps(_mm_set1_ps(f32::from_bits(F32_2POW_M14_BITS)), frac);
    let scale = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(
        exp,
        _mm_set1_epi32(112),
    )));
    let normal = _mm_mul_ps(scale, _mm_add_ps(_mm_set1_ps(1.0), frac));
    let is_denorm = _mm_castsi128_ps(_mm_cmpeq_epi32(exp, _mm_setzero_si128()));
    let val = _mm_blendv_ps(normal, denorm, is_denorm);
    let is_max = _mm_castsi128_ps(_mm_cmpeq_epi32(exp, _mm_set1_epi32(0x1f)));
    let mant_nonzero = _mm_castsi128_ps(_mm_cmpgt_epi32(mant, _mm_setzero_si128()));
    let special = _mm_blendv_ps(
        _mm_castsi128_ps(_mm_set1_epi32(F32_INF_BITS as i32)),
        _mm_castsi128_ps(_mm_set1_epi32(F32_NAN_BITS as i32)),
        mant_nonzero,
    );
    _mm_blendv_ps(val, special, is_max)
}

/// Decode four packed words into channel vectors `(R, G, B)`.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn decode_words_sse4_1(words: __m128i) -> (__m128, __m128, __m128) {
    let r = _mm_and_si128(words, _mm_set1_epi32(0x7ff));
    let g = _mm_and_si128(_mm_srli_epi32::<11>(words), _mm_set1_epi32(0x7ff));
    let b = _mm_and_si128(_mm_srli_epi32::<22>(words), _mm_set1_epi32(0x3ff));
    // SAFETY: sse4.1 is enabled by the enclosing target_feature.
    unsafe {
        (
            decode_codes_sse4_1::<6>(r),
            decode_codes_sse4_1::<6>(g),
            decode_codes_sse4_1::<5>(b),
        )
    }
}

/// Interleave `(R, G, B)` channel vectors (4 pixels) with A = 1.0 and store the
/// first `count` (1..=4) pixels at `dst`.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
/// * `dst` must be valid for `count * 4` f32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_pixels_sse4_1(r: __m128, g: __m128, b: __m128, dst: *mut f32, count: usize) {
    // SAFETY: sse4.1 enabled; `dst` valid for `count * 4` f32 per caller.
    unsafe {
        let a = _mm_set1_ps(1.0);
        store_interleaved_ps(r, g, b, a, dst, count);
    }
}

/// Decode a row remainder with 4-pixel SSE blocks and one padded partial block.
///
/// # Safety
/// * SSE4.1 must be available.
/// * `dst` must be valid for `row.len()` f32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn load_row_sse4_1(row: &[u8], dst: *mut f32) {
    debug_assert_eq!(row.len() % 4, 0);

    let mut x = 0;
    while x + 16 <= row.len() {
        let words = unsafe { _mm_loadu_si128(row.as_ptr().add(x).cast()) };
        let (r, g, b) = unsafe { decode_words_sse4_1(words) };
        unsafe { store_pixels_sse4_1(r, g, b, dst.add(x), 4) };
        x += 16;
    }

    if x < row.len() {
        let count = (row.len() - x) / 4;
        let mut padded = [0u8; 16];
        padded[..count * 4].copy_from_slice(&row[x..]);
        let words = unsafe { _mm_loadu_si128(padded.as_ptr().cast()) };
        let (r, g, b) = unsafe { decode_words_sse4_1(words) };
        unsafe { store_pixels_sse4_1(r, g, b, dst.add(x), count) };
    }
}

/// SSE4.1 path for `B10G11R11_UFLOAT_PACK32`.
///
/// Processes four pixels (16 bytes → 16 f32) per iteration; a 1-3 pixel tail is
/// decoded with the same 4-wide helper on a zero-padded word block.
///
/// **Not part of the public API.** Exposed as `pub` + `doc(hidden)` only so
/// `benches/` can measure this kernel directly. Use [`load_b10g11r11_f32`] for
/// the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn load_b10g11r11_f32_sse4_1(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_b10g11r11_f32_sse4_1");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: sse4.1 is available per the enclosing fn's contract.
            load_row_sse4_1(row, dst)
        })
    }
}

/// Decode a vector of right-aligned channel codes (AVX2+FMA, 8 lanes).
///
/// # Safety
/// * AVX2 and FMA must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn decode_codes_avx2<const M: i32>(codes: __m256i) -> __m256 {
    let mant_max = 1i32 << M;
    let mant = _mm256_and_si256(codes, _mm256_set1_epi32(mant_max - 1));
    let exp = _mm256_and_si256(_mm256_srli_epi32::<M>(codes), _mm256_set1_epi32(0x1f));
    let mant_f = _mm256_cvtepi32_ps(mant);
    let frac = _mm256_mul_ps(mant_f, _mm256_set1_ps(1.0 / mant_max as f32));
    let denorm = _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(F32_2POW_M14_BITS)), frac);
    let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        exp,
        _mm256_set1_epi32(112),
    )));
    // scale * (1 + frac) fused as scale * frac + scale.
    let normal = _mm256_fmadd_ps(scale, frac, scale);
    let is_denorm = _mm256_castsi256_ps(_mm256_cmpeq_epi32(exp, _mm256_setzero_si256()));
    let val = _mm256_blendv_ps(normal, denorm, is_denorm);
    let is_max = _mm256_castsi256_ps(_mm256_cmpeq_epi32(exp, _mm256_set1_epi32(0x1f)));
    let mant_nonzero = _mm256_castsi256_ps(_mm256_cmpgt_epi32(mant, _mm256_setzero_si256()));
    let special = _mm256_blendv_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(F32_INF_BITS as i32)),
        _mm256_castsi256_ps(_mm256_set1_epi32(F32_NAN_BITS as i32)),
        mant_nonzero,
    );
    _mm256_blendv_ps(val, special, is_max)
}

/// Decode a row with 8-pixel AVX2 blocks, delegating the remainder to the
/// SSE4.1 row helper (avx2 implies sse4.1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn load_row_avx2_fma(row: &[u8], dst: *mut f32) {
    // SAFETY: avx2+fma enabled by the enclosing target_feature; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;

        // 8 pixels (32 input bytes, 32 output f32s) per iteration.
        while x + 32 <= row.len() {
            let words = _mm256_loadu_si256(row.as_ptr().add(x) as *const __m256i);
            let r = _mm256_and_si256(words, _mm256_set1_epi32(0x7ff));
            let g = _mm256_and_si256(_mm256_srli_epi32::<11>(words), _mm256_set1_epi32(0x7ff));
            let b = _mm256_and_si256(_mm256_srli_epi32::<22>(words), _mm256_set1_epi32(0x3ff));
            let rf = decode_codes_avx2::<6>(r);
            let gf = decode_codes_avx2::<6>(g);
            let bf = decode_codes_avx2::<5>(b);
            store_pixels_sse4_1(
                _mm256_castps256_ps128(rf),
                _mm256_castps256_ps128(gf),
                _mm256_castps256_ps128(bf),
                dst.add(x),
                4,
            );
            store_pixels_sse4_1(
                _mm256_extractf128_ps::<1>(rf),
                _mm256_extractf128_ps::<1>(gf),
                _mm256_extractf128_ps::<1>(bf),
                dst.add(x + 16),
                4,
            );
            x += 32;
        }

        load_row_sse4_1(&row[x..], dst.add(x));
    }
}

/// AVX2 path for `B10G11R11_UFLOAT_PACK32`.
///
/// Processes eight pixels per iteration; the 8-wide channel vectors are split
/// into 128-bit halves and interleaved with the shared SSE4.1 transpose/store.
/// The 0-7 pixel remainder is handled with the SSE4.1 4-wide block and its
/// zero-padded tail.
///
/// **Not part of the public API.** See [`load_b10g11r11_f32_sse4_1`]; use
/// [`load_b10g11r11_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn load_b10g11r11_f32_avx2_fma(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_b10g11r11_f32_avx2_fma");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: avx2+fma are available per the enclosing fn's contract.
            load_row_avx2_fma(row, dst)
        })
    }
}

/// Decode a vector of right-aligned channel codes (AVX-512, 16 lanes).
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn decode_codes_avx512<const M: u32>(codes: __m512i) -> __m512 {
    let mant_max = 1i32 << M;
    let mant = _mm512_and_si512(codes, _mm512_set1_epi32(mant_max - 1));
    let exp = _mm512_and_si512(_mm512_srli_epi32::<M>(codes), _mm512_set1_epi32(0x1f));
    let mant_f = _mm512_cvtepi32_ps(mant);
    let frac = _mm512_mul_ps(mant_f, _mm512_set1_ps(1.0 / mant_max as f32));
    let denorm = _mm512_mul_ps(_mm512_set1_ps(f32::from_bits(F32_2POW_M14_BITS)), frac);
    let scale = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
        exp,
        _mm512_set1_epi32(112),
    )));
    // scale * (1 + frac) fused as scale * frac + scale.
    let normal = _mm512_fmadd_ps(scale, frac, scale);
    let is_denorm = _mm512_cmpeq_epi32_mask(exp, _mm512_setzero_si512());
    let val = _mm512_mask_blend_ps(is_denorm, normal, denorm);
    let is_max = _mm512_cmpeq_epi32_mask(exp, _mm512_set1_epi32(0x1f));
    let mant_nonzero = _mm512_cmpgt_epi32_mask(mant, _mm512_setzero_si512());
    let special = _mm512_mask_blend_ps(
        mant_nonzero,
        _mm512_castsi512_ps(_mm512_set1_epi32(F32_INF_BITS as i32)),
        _mm512_castsi512_ps(_mm512_set1_epi32(F32_NAN_BITS as i32)),
    );
    _mm512_mask_blend_ps(is_max, val, special)
}

/// Decode 16 packed words into planar `(R, G, B)` f32 planes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn decode16_words_avx512(words: __m512i) -> (__m512, __m512, __m512) {
    let r = _mm512_and_si512(words, _mm512_set1_epi32(0x7ff));
    let g = _mm512_and_si512(_mm512_srli_epi32::<11>(words), _mm512_set1_epi32(0x7ff));
    let b = _mm512_and_si512(_mm512_srli_epi32::<22>(words), _mm512_set1_epi32(0x3ff));
    // SAFETY: avx512 is enabled by the enclosing target_feature.
    unsafe {
        (
            decode_codes_avx512::<6>(r),
            decode_codes_avx512::<6>(g),
            decode_codes_avx512::<5>(b),
        )
    }
}

/// Decode a row with 16-pixel AVX-512 blocks and a masked 1-15 pixel tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn load_row_avx512(row: &[u8], dst: *mut f32) {
    // SAFETY: masked loads/stores only touch enabled lanes; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let one = _mm512_set1_ps(1.0);
        let mut x = 0usize;

        // 16 pixels (64 input bytes, 64 output f32s) per iteration.
        while x + 64 <= row.len() {
            let words = _mm512_loadu_si512(row.as_ptr().add(x) as *const __m512i);
            let (rf, gf, bf) = decode16_words_avx512(words);
            store_interleaved16_ps(rf, gf, bf, one, dst.add(x));
            x += 64;
        }

        let rem = (row.len() - x) / 4;
        if rem > 0 {
            let words =
                _mm512_maskz_loadu_epi32((1u16 << rem) - 1, row.as_ptr().add(x) as *const i32);
            let (rf, gf, bf) = decode16_words_avx512(words);
            store_interleaved16_mask_ps(rf, gf, bf, one, dst.add(x), rem);
        }
    }
}

/// AVX-512 path for `B10G11R11_UFLOAT_PACK32`.
///
/// Processes sixteen pixels per iteration; the 16-wide channel vectors are
/// interleaved with two levels of `vpermt2` permutes and stored 512 bits at a
/// time. The 0-15 pixel remainder runs the same block behind masked
/// loads/stores.
///
/// **Not part of the public API.** See [`load_b10g11r11_f32_sse4_1`]; use
/// [`load_b10g11r11_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn load_b10g11r11_f32_avx512(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_b10g11r11_f32_avx512");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: avx512 is available per the enclosing fn's contract.
            load_row_avx512(row, dst)
        })
    }
}

// ---- AArch64 NEON ----

/// Decode a vector of right-aligned channel codes (NEON, 4 lanes).
///
/// See the module-level SIMD note for the branchless decode; the math is
/// identical, just mapped to AArch64 NEON.
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_codes_neon<const M: i32>(codes: uint32x4_t) -> float32x4_t {
    let mant_max = 1i32 << M;
    let mant = vandq_u32(codes, vdupq_n_u32((mant_max - 1) as u32));
    let exp = vandq_u32(vshrq_n_u32::<M>(codes), vdupq_n_u32(0x1f));
    let mant_f = vcvtq_f32_u32(mant);
    let frac = vmulq_f32(mant_f, vdupq_n_f32(1.0 / mant_max as f32));
    let denorm = vmulq_f32(vdupq_n_f32(f32::from_bits(F32_2POW_M14_BITS)), frac);
    let scale = vreinterpretq_f32_u32(vshlq_n_u32::<23>(vaddq_u32(exp, vdupq_n_u32(112))));
    // scale * (1 + frac) fused as scale + scale * frac.
    let normal = vfmaq_f32(scale, scale, frac);
    let is_denorm = vceqq_u32(exp, vdupq_n_u32(0));
    let val = vbslq_f32(is_denorm, denorm, normal);
    let is_max = vceqq_u32(exp, vdupq_n_u32(0x1f));
    let mant_nonzero = vtstq_u32(mant, mant);
    let inf = vreinterpretq_f32_u32(vdupq_n_u32(F32_INF_BITS));
    let nan = vreinterpretq_f32_u32(vdupq_n_u32(F32_NAN_BITS));
    let special = vbslq_f32(mant_nonzero, nan, inf);
    vbslq_f32(is_max, special, val)
}

/// Decode four packed words into interleaved `[R, G, B, A]` pixels (A = 1.0).
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_words_neon(words: uint32x4_t) -> float32x4x4_t {
    let r = vandq_u32(words, vdupq_n_u32(0x7ff));
    let g = vandq_u32(vshrq_n_u32::<11>(words), vdupq_n_u32(0x7ff));
    let b = vandq_u32(vshrq_n_u32::<22>(words), vdupq_n_u32(0x3ff));
    // SAFETY: NEON enabled by the enclosing target_feature.
    unsafe {
        float32x4x4_t(
            decode_codes_neon::<6>(r),
            decode_codes_neon::<6>(g),
            decode_codes_neon::<5>(b),
            vdupq_n_f32(1.0),
        )
    }
}

/// Decode a row with 4-pixel NEON blocks; a 1-3 pixel tail reuses the same
/// 4-wide decode on a zero-padded word block.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_row_neon(row: &[u8], dst: *mut f32) {
    // SAFETY: NEON enabled by the enclosing target_feature; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;

        // 4 pixels (16 input bytes, 16 output f32s) per iteration.
        while x + 16 <= row.len() {
            let words = vreinterpretq_u32_u8(vld1q_u8(row.as_ptr().add(x)));
            let decoded = decode_words_neon(words);
            vst4q_f32(dst.add(x), decoded);
            x += 16;
        }

        // 1-3 pixel tail: zero-pad a 4-word block, decode, and copy out the
        // valid pixels via a scratch buffer.
        if x < row.len() {
            let rem = (row.len() - x) / 4;
            let mut tmp = [0u8; 16];
            tmp[..rem * 4].copy_from_slice(&row[x..x + rem * 4]);
            let words = vreinterpretq_u32_u8(vld1q_u8(tmp.as_ptr()));
            let decoded = decode_words_neon(words);
            let mut scratch = [0.0f32; 16];
            vst4q_f32(scratch.as_mut_ptr(), decoded);
            std::ptr::copy_nonoverlapping(scratch.as_ptr(), dst.add(x), rem * 4);
        }
    }
}

/// NEON path for `B10G11R11_UFLOAT_PACK32`.
///
/// Processes four pixels per iteration; `vst4q_f32` interleaves the channel
/// vectors straight into `[R, G, B, A]` output. A 1-3 pixel tail reuses the
/// same 4-wide decode on a zero-padded word block.
///
/// **Not part of the public API.** See [`load_b10g11r11_f32_sse4_1`]; use
/// [`load_b10g11r11_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn load_b10g11r11_f32_neon(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_b10g11r11_f32_neon");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: NEON is available per the enclosing fn's contract.
            load_row_neon(row, dst)
        })
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::{ColorSpace, Surface};

    fn b10_surface(data: Vec<u8>, width: u32, height: u32, stride: u32) -> Surface {
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::B10G11R11_UFLOAT_PACK32,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    /// Compare a SIMD decode against the scalar reference. The FMA-fused normal
    /// path can differ by <=1 ULP, so finite lanes are checked to a relative
    /// tolerance; non-finite lanes (exp==31 decodes to Inf/NaN) must agree on
    /// their NaN-ness and finiteness.
    fn assert_load_matches(simd: &[[f32; 4]], surface: &Surface) {
        let reference = load_b10g11r11_f32_serial(surface).unwrap().pixels;
        assert_eq!(simd.len(), reference.len(), "pixel count");
        for (i, (got, want)) in simd.iter().zip(&reference).enumerate() {
            for c in 0..4 {
                let g = got[c];
                let w = want[c];
                if w.is_finite() {
                    assert!(
                        (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                        "pixel {i} lane {c}: got {g} want {w}"
                    );
                } else {
                    assert!(
                        g.is_nan() == w.is_nan() && g.is_finite() == w.is_finite(),
                        "pixel {i} lane {c}: got {g} want {w}"
                    );
                }
            }
        }
    }

    /// A row that sweeps all 2^11 codes for R and G and all 2^10 for B, each in
    /// its real bit slot.
    fn sweep_surface() -> Surface {
        let n = 2048u32; // covers 0..=2047 (11-bit); B wraps its 10-bit code.
        let mut data = Vec::with_capacity((n * 4) as usize);
        for i in 0..n {
            let r = i & 0x7ff;
            let g = i & 0x7ff;
            let b = i & 0x3ff;
            let word = r | (g << 11) | (b << 22);
            data.extend_from_slice(&word.to_le_bytes());
        }
        b10_surface(data, n, 1, n * 4)
    }

    /// A multi-row surface with an odd width (exercises every tail residue) and
    /// junk stride padding that must be skipped.
    fn tail_stride_surface(width: u32) -> Surface {
        let h = 3u32;
        let row_bytes = width * 4;
        let stride = row_bytes + 12; // 3 words of junk padding per row
        let mut data = vec![0u8; (stride * h) as usize];
        for y in 0..h {
            for xp in 0..width {
                let seed = y.wrapping_mul(9973).wrapping_add(xp.wrapping_mul(7));
                let r = seed & 0x7ff;
                let g = seed.wrapping_mul(3) & 0x7ff;
                let b = seed.wrapping_mul(5) & 0x3ff;
                let word = r | (g << 11) | (b << 22);
                let off = (y * stride + xp * 4) as usize;
                data[off..off + 4].copy_from_slice(&word.to_le_bytes());
            }
            // Fill the padding with a non-zero junk pattern.
            let pad_off = (y * stride + row_bytes) as usize;
            for byte in &mut data[pad_off..pad_off + 12] {
                *byte = 0xAB;
            }
        }
        b10_surface(data, width, h, stride)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sse4_load_sweep_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let s = sweep_surface();
        let simd = unsafe { load_b10g11r11_f32_sse4_1(&s).unwrap() };
        assert_load_matches(&simd.pixels, &s);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sse4_load_tails_match_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        for width in 1..=17u32 {
            let s = tail_stride_surface(width);
            let simd = unsafe { load_b10g11r11_f32_sse4_1(&s).unwrap() };
            assert_load_matches(&simd.pixels, &s);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_load_sweep_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let s = sweep_surface();
        let simd = unsafe { load_b10g11r11_f32_avx2_fma(&s).unwrap() };
        assert_load_matches(&simd.pixels, &s);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_load_tails_match_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for width in 1..=25u32 {
            let s = tail_stride_surface(width);
            let simd = unsafe { load_b10g11r11_f32_avx2_fma(&s).unwrap() };
            assert_load_matches(&simd.pixels, &s);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_load_sweep_matches_scalar() {
        if !has_avx512() {
            return;
        }
        let s = sweep_surface();
        let simd = unsafe { load_b10g11r11_f32_avx512(&s).unwrap() };
        assert_load_matches(&simd.pixels, &s);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_load_tails_match_scalar() {
        if !has_avx512() {
            return;
        }
        for width in 1..=49u32 {
            let s = tail_stride_surface(width);
            let simd = unsafe { load_b10g11r11_f32_avx512(&s).unwrap() };
            assert_load_matches(&simd.pixels, &s);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_load_sweep_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let s = sweep_surface();
        let simd = unsafe { load_b10g11r11_f32_neon(&s).unwrap() };
        assert_load_matches(&simd.pixels, &s);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_load_tails_match_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for width in 1..=17u32 {
            let s = tail_stride_surface(width);
            let simd = unsafe { load_b10g11r11_f32_neon(&s).unwrap() };
            assert_load_matches(&simd.pixels, &s);
        }
    }
}
