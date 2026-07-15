//! `E5B9G9R9_UFLOAT_PACK32` storer (shared-exponent RGB).
//!
//! Encodes linear `[R, G, B]` into a single packed word using the Khronos /
//! OpenGL shared-exponent algorithm: clamp to the representable range, pick the
//! shared exponent from the largest channel, then round each channel's mantissa
//! against that exponent. Alpha is dropped (the format has none).

use crate::processing::buffer::Buffer;
use crate::processing::dispatch::dispatch_simd;
#[cfg(target_arch = "x86_64")]
use crate::processing::x86::{
    load_planar4_ps, load_planar8_ps, load_planar16_mask_ps, load_planar16_ps,
};

use super::write_pixels;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const MANTISSA_BITS: i32 = 9;
const EXP_BIAS: i32 = 15;
const MAX_EXP: i32 = 31;

/// Largest finite value representable: `(2^9 - 1) / 2^9 * 2^(31 - 15)`.
const SHARED_EXP_MAX: f32 = ((1i32 << MANTISSA_BITS) - 1) as f32 / (1i32 << MANTISSA_BITS) as f32
    * (1u32 << (MAX_EXP - EXP_BIAS)) as f32;

/// Clamp a channel into `[0, SHARED_EXP_MAX]`, mapping negatives and NaN to 0
/// and +Inf to the max (matching the spec's clamp on finite inputs).
#[inline]
fn clamp_channel(c: f32) -> f32 {
    if c.is_nan() || c <= 0.0 {
        0.0
    } else if c >= SHARED_EXP_MAX {
        SHARED_EXP_MAX
    } else {
        c
    }
}

/// Encode linear `[R, G, B]` into a packed `E5B9G9R9` word.
#[inline]
pub(crate) fn encode(rgb: [f32; 3]) -> u32 {
    let rc = clamp_channel(rgb[0]);
    let gc = clamp_channel(rgb[1]);
    let bc = clamp_channel(rgb[2]);
    let max_c = rc.max(gc).max(bc);

    // Preliminary shared exponent from the largest channel's magnitude.
    let exp_shared_p = if max_c > 0.0 {
        // floor(log2(max_c)) == unbiased IEEE exponent for normal floats; any
        // denormal/tiny value lands below -EXP_BIAS-1 and is clamped identically.
        let e = ((max_c.to_bits() >> 23) & 0xff) as i32 - 127;
        e.max(-EXP_BIAS - 1) + 1 + EXP_BIAS
    } else {
        0
    };

    // Round the max channel; if it rounds up into the next binade, bump the
    // shared exponent so all mantissas stay in range.
    let denom_p = exp2i(exp_shared_p - EXP_BIAS - MANTISSA_BITS);
    let max_s = (max_c / denom_p + 0.5).floor() as i32;
    let exp_shared = if max_s == (1 << MANTISSA_BITS) {
        exp_shared_p + 1
    } else {
        exp_shared_p
    };

    let denom = exp2i(exp_shared - EXP_BIAS - MANTISSA_BITS);
    let rs = (rc / denom + 0.5).floor() as u32;
    let gs = (gc / denom + 0.5).floor() as u32;
    let bs = (bc / denom + 0.5).floor() as u32;

    (((exp_shared as u32) & 0x1f) << 27) | ((bs & 0x1ff) << 18) | ((gs & 0x1ff) << 9) | (rs & 0x1ff)
}

/// `2^n` as f32 for the exponent range this encoder produces (always a valid
/// normal exponent, so a direct bit construction is exact).
#[inline]
fn exp2i(n: i32) -> f32 {
    f32::from_bits(((n + 127) as u32) << 23)
}

pub fn store_e5b9g9r9_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_e5b9g9r9_f32");

    dispatch_simd! {
        x86_64: {
            avx512: store_e5b9g9r9_f32_avx512(buf),
            avx2_fma: store_e5b9g9r9_f32_avx2_fma(buf),
            sse4_1: store_e5b9g9r9_f32_sse4_1(buf),
        },
        aarch64: {
            neon: store_e5b9g9r9_f32_neon(buf),
        },
    }

    store_e5b9g9r9_f32_serial(buf)
}

/// Serial reference path.
///
/// **Not part of the public API.** Exposed for benches/tests to compare
/// against the SIMD kernels.
#[doc(hidden)]
pub fn store_e5b9g9r9_f32_serial(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_e5b9g9r9_f32_serial");
    write_pixels(buf, 1, 4, |lanes, bytes| {
        let word = encode([lanes[0], lanes[1], lanes[2]]);
        bytes.copy_from_slice(&word.to_le_bytes());
    })
}

// ---- SIMD kernels ----------------------------------------------------------
//
// The store algorithm has a per-pixel reduction (max of the three channels
// selects the shared exponent), so the kernels work on a *planar* layout:
// each `Buffer<[f32; 4]>` block is deinterleaved into R/G/B vectors, the
// per-pixel exponent + mantissa math runs vertically, and the packed words
// come out one-per-lane for a contiguous store.
//
// Each step mirrors the scalar `encode`:
//   * clamp = min(max(c, 0), MAX) — `maxps`/`minps` map NaN and negatives to 0
//     and +Inf/overflow to MAX (Intel semantics: NaN in the first operand
//     returns the second), matching `clamp_channel`.
//   * preliminary exponent = max((bits(max_c) >> 23 & 0xff) - 127, -16) + 16,
//     valid for max_c == 0 (bits 0 → -16 → 0) and denormals alike.
//   * scale by the reciprocal power-of-two `inv_denom = 2^(24 - exp)` (exact,
//     no divide) then round: FMA `x * inv_denom + 0.5` (add + mul on SSE4.1,
//     no FMA) and a truncating `cvttps`, which equals `.floor() as i32` because
//     every operand is non-negative. Fused rounding drops ~1 ULP of error.

/// Encode four planar pixels into four packed `E5B9G9R9` words.
///
/// # Safety
/// * SSE4.1 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn encode_planar_sse(vr: __m128, vg: __m128, vb: __m128) -> __m128i {
    let zero = _mm_setzero_ps();
    let max_lim = _mm_set1_ps(SHARED_EXP_MAX);
    let half = _mm_set1_ps(0.5);
    let c151 = _mm_set1_epi32(151);
    let mask9 = _mm_set1_epi32(0x1ff);

    let rc = _mm_min_ps(_mm_max_ps(vr, zero), max_lim);
    let gc = _mm_min_ps(_mm_max_ps(vg, zero), max_lim);
    let bc = _mm_min_ps(_mm_max_ps(vb, zero), max_lim);
    let max_c = _mm_max_ps(_mm_max_ps(rc, gc), bc);

    // Preliminary shared exponent (per lane).
    let bits = _mm_castps_si128(max_c);
    let e = _mm_sub_epi32(
        _mm_and_si128(_mm_srli_epi32::<23>(bits), _mm_set1_epi32(0xff)),
        _mm_set1_epi32(127),
    );
    let esp = _mm_add_epi32(_mm_max_epi32(e, _mm_set1_epi32(-16)), _mm_set1_epi32(16));

    // Round the max channel and bump the exponent if it overflows the mantissa.
    // inv_denom_p = 2^(24 - esp), IEEE exponent field 151 - esp.
    let inv_denom_p = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_sub_epi32(c151, esp)));
    let max_s = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(max_c, inv_denom_p), half));
    let bump = _mm_cmpeq_epi32(max_s, _mm_set1_epi32(512)); // -1 where equal
    let exp_shared = _mm_sub_epi32(esp, bump); // subtract -1 == +1

    let inv_denom = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_sub_epi32(c151, exp_shared)));
    let rs = _mm_and_si128(
        _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(rc, inv_denom), half)),
        mask9,
    );
    let gs = _mm_and_si128(
        _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(gc, inv_denom), half)),
        mask9,
    );
    let bs = _mm_and_si128(
        _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(bc, inv_denom), half)),
        mask9,
    );

    let exp_field = _mm_slli_epi32::<27>(_mm_and_si128(exp_shared, _mm_set1_epi32(0x1f)));
    _mm_or_si128(
        _mm_or_si128(exp_field, _mm_slli_epi32::<18>(bs)),
        _mm_or_si128(_mm_slli_epi32::<9>(gs), rs),
    )
}

/// SSE4.1 path for `E5B9G9R9_UFLOAT_PACK32` stores (4 pixels/iteration).
///
/// **Not part of the public API.** Exposed as `pub` + `doc(hidden)` for
/// benchmarks. Real callers should use [`store_e5b9g9r9_f32`].
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn store_e5b9g9r9_f32_sse4_1(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_e5b9g9r9_f32_sse4_1");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: sse4.1 is available per the enclosing fn's contract; the
        // tail uses a zero-padded stack buffer so block loads never read past
        // the source, and all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 4 <= n {
                let (vr, vg, vb, _) = load_planar4_ps(src.add(i * 4));
                let words = encode_planar_sse(vr, vg, vb);
                _mm_storeu_si128(dst.add(i) as *mut __m128i, words);
                i += 4;
            }

            if i < n {
                let tail = n - i;
                let mut pad = [0.0f32; 16];
                std::ptr::copy_nonoverlapping(src.add(i * 4), pad.as_mut_ptr(), tail * 4);
                let (vr, vg, vb, _) = load_planar4_ps(pad.as_ptr());
                let words = encode_planar_sse(vr, vg, vb);
                let mut wtmp = [0u32; 4];
                _mm_storeu_si128(wtmp.as_mut_ptr() as *mut __m128i, words);
                std::ptr::copy_nonoverlapping(
                    wtmp.as_ptr() as *const u8,
                    dst.add(i) as *mut u8,
                    tail * 4,
                );
            }
        }
    })
}

/// Encode eight planar pixels into eight packed words. Mirrors
/// [`encode_planar_sse`] widened to 8 lanes, fusing the rounding `+ 0.5` via FMA.
///
/// # Safety
/// * AVX2 and FMA must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn encode_planar_avx2(vr: __m256, vg: __m256, vb: __m256) -> __m256i {
    let zero = _mm256_setzero_ps();
    let max_lim = _mm256_set1_ps(SHARED_EXP_MAX);
    let half = _mm256_set1_ps(0.5);
    let c151 = _mm256_set1_epi32(151);
    let mask9 = _mm256_set1_epi32(0x1ff);

    let rc = _mm256_min_ps(_mm256_max_ps(vr, zero), max_lim);
    let gc = _mm256_min_ps(_mm256_max_ps(vg, zero), max_lim);
    let bc = _mm256_min_ps(_mm256_max_ps(vb, zero), max_lim);
    let max_c = _mm256_max_ps(_mm256_max_ps(rc, gc), bc);

    let bits = _mm256_castps_si256(max_c);
    let e = _mm256_sub_epi32(
        _mm256_and_si256(_mm256_srli_epi32::<23>(bits), _mm256_set1_epi32(0xff)),
        _mm256_set1_epi32(127),
    );
    let esp = _mm256_add_epi32(
        _mm256_max_epi32(e, _mm256_set1_epi32(-16)),
        _mm256_set1_epi32(16),
    );

    // inv_denom_p = 2^(24 - esp), IEEE exponent field 151 - esp.
    let inv_denom_p = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_sub_epi32(c151, esp)));
    let max_s = _mm256_cvttps_epi32(_mm256_fmadd_ps(max_c, inv_denom_p, half));
    let bump = _mm256_cmpeq_epi32(max_s, _mm256_set1_epi32(512));
    let exp_shared = _mm256_sub_epi32(esp, bump);

    let inv_denom =
        _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_sub_epi32(c151, exp_shared)));
    let rs = _mm256_and_si256(
        _mm256_cvttps_epi32(_mm256_fmadd_ps(rc, inv_denom, half)),
        mask9,
    );
    let gs = _mm256_and_si256(
        _mm256_cvttps_epi32(_mm256_fmadd_ps(gc, inv_denom, half)),
        mask9,
    );
    let bs = _mm256_and_si256(
        _mm256_cvttps_epi32(_mm256_fmadd_ps(bc, inv_denom, half)),
        mask9,
    );

    let exp_field = _mm256_slli_epi32::<27>(_mm256_and_si256(exp_shared, _mm256_set1_epi32(0x1f)));
    _mm256_or_si256(
        _mm256_or_si256(exp_field, _mm256_slli_epi32::<18>(bs)),
        _mm256_or_si256(_mm256_slli_epi32::<9>(gs), rs),
    )
}

/// AVX2+FMA path for `E5B9G9R9_UFLOAT_PACK32` stores (8 pixels/iteration).
///
/// **Not part of the public API.** See [`store_e5b9g9r9_f32_sse4_1`].
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn store_e5b9g9r9_f32_avx2_fma(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_e5b9g9r9_f32_avx2_fma");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: avx2+fma are available per the enclosing fn's contract; the
        // tail uses a zero-padded stack buffer, and all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 8 <= n {
                let (vr, vg, vb, _) = load_planar8_ps(src.add(i * 4));
                let words = encode_planar_avx2(vr, vg, vb);
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, words);
                i += 8;
            }

            if i < n {
                let tail = n - i;
                let mut pad = [0.0f32; 32];
                std::ptr::copy_nonoverlapping(src.add(i * 4), pad.as_mut_ptr(), tail * 4);
                let (vr, vg, vb, _) = load_planar8_ps(pad.as_ptr());
                let words = encode_planar_avx2(vr, vg, vb);
                let mut wtmp = [0u32; 8];
                _mm256_storeu_si256(wtmp.as_mut_ptr() as *mut __m256i, words);
                std::ptr::copy_nonoverlapping(
                    wtmp.as_ptr() as *const u8,
                    dst.add(i) as *mut u8,
                    tail * 4,
                );
            }
        }
    })
}

/// Encode sixteen planar pixels into sixteen packed words. Mirrors
/// [`encode_planar_sse`] widened to 16 lanes.
///
/// # Safety
/// * AVX-512 F/VL/BW must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn encode_planar_avx512(vr: __m512, vg: __m512, vb: __m512) -> __m512i {
    let zero = _mm512_setzero_ps();
    let max_lim = _mm512_set1_ps(SHARED_EXP_MAX);
    let half = _mm512_set1_ps(0.5);
    let c151 = _mm512_set1_epi32(151);
    let mask9 = _mm512_set1_epi32(0x1ff);

    let rc = _mm512_min_ps(_mm512_max_ps(vr, zero), max_lim);
    let gc = _mm512_min_ps(_mm512_max_ps(vg, zero), max_lim);
    let bc = _mm512_min_ps(_mm512_max_ps(vb, zero), max_lim);
    let max_c = _mm512_max_ps(_mm512_max_ps(rc, gc), bc);

    let bits = _mm512_castps_si512(max_c);
    let e = _mm512_sub_epi32(
        _mm512_and_si512(_mm512_srli_epi32::<23>(bits), _mm512_set1_epi32(0xff)),
        _mm512_set1_epi32(127),
    );
    let esp = _mm512_add_epi32(
        _mm512_max_epi32(e, _mm512_set1_epi32(-16)),
        _mm512_set1_epi32(16),
    );

    // inv_denom_p = 2^(24 - esp), IEEE exponent field 151 - esp.
    let inv_denom_p = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_sub_epi32(c151, esp)));
    let max_s = _mm512_cvttps_epi32(_mm512_fmadd_ps(max_c, inv_denom_p, half));
    let bump = _mm512_cmpeq_epi32_mask(max_s, _mm512_set1_epi32(512));
    let exp_shared = _mm512_mask_add_epi32(esp, bump, esp, _mm512_set1_epi32(1));

    let inv_denom =
        _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_sub_epi32(c151, exp_shared)));
    let rs = _mm512_and_si512(
        _mm512_cvttps_epi32(_mm512_fmadd_ps(rc, inv_denom, half)),
        mask9,
    );
    let gs = _mm512_and_si512(
        _mm512_cvttps_epi32(_mm512_fmadd_ps(gc, inv_denom, half)),
        mask9,
    );
    let bs = _mm512_and_si512(
        _mm512_cvttps_epi32(_mm512_fmadd_ps(bc, inv_denom, half)),
        mask9,
    );

    let exp_field = _mm512_slli_epi32::<27>(_mm512_and_si512(exp_shared, _mm512_set1_epi32(0x1f)));
    // 0xFE = three-way OR.
    let hi3 = _mm512_ternarylogic_epi32::<0xFE>(
        exp_field,
        _mm512_slli_epi32::<18>(bs),
        _mm512_slli_epi32::<9>(gs),
    );
    _mm512_or_si512(hi3, rs)
}

/// AVX-512 path for `E5B9G9R9_UFLOAT_PACK32` stores (16 pixels/iteration).
///
/// **Not part of the public API.** See [`store_e5b9g9r9_f32_sse4_1`].
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn store_e5b9g9r9_f32_avx512(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_e5b9g9r9_f32_avx512");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: avx512 is available per the enclosing fn's contract; masked
        // loads/stores only touch enabled lanes; all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 16 <= n {
                let (vr, vg, vb, _) = load_planar16_ps(src.add(i * 4));
                let words = encode_planar_avx512(vr, vg, vb);
                _mm512_storeu_si512(dst.add(i) as *mut __m512i, words);
                i += 16;
            }

            if i < n {
                let tail = n - i;
                let (vr, vg, vb, _) = load_planar16_mask_ps(src.add(i * 4), tail);
                let words = encode_planar_avx512(vr, vg, vb);
                _mm512_mask_storeu_epi32(dst.add(i) as *mut i32, (1u16 << tail) - 1, words);
            }
        }
    })
}

/// Encode one `[R, G, B, A]` pixel into a packed `E5B9G9R9` word on NEON.
///
/// Mirrors the scalar [`encode`], scaling by the reciprocal power-of-two
/// `inv_denom = 2^(24 - exp)` (exact, no divide) with the `+ 0.5` rounding
/// fused via `vfmaq_f32`; a truncating convert (`vcvtq_u32_f32` rounds toward
/// zero) then equals `.floor() as u32` for the non-negative operands.
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn encode_e5_pixel_neon(lanes: float32x4_t) -> u32 {
    let zero = vdupq_n_f32(0.0);
    let max_lim = vdupq_n_f32(SHARED_EXP_MAX);
    let half = vdupq_n_f32(0.5);

    // Clamp all lanes; lane 3 (alpha) is forced to 0 so it never wins the max.
    let clamped = vminq_f32(vmaxq_f32(lanes, zero), max_lim);
    let c = vsetq_lane_f32::<3>(0.0, clamped);

    // Horizontal max over the four lanes (lane 3 == 0), broadcast to all lanes.
    let max_c = vmovq_n_f32(vmaxvq_f32(c));

    // Preliminary shared exponent (same in every lane).
    let bits = vreinterpretq_u32_f32(max_c);
    let e = vsubq_s32(
        vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32::<23>(bits), vdupq_n_u32(0xff))),
        vdupq_n_s32(127),
    );
    let esp = vaddq_s32(vmaxq_s32(e, vdupq_n_s32(-16)), vdupq_n_s32(16));

    // inv_denom_p = 2^(24 - esp), IEEE exponent field 151 - esp.
    let inv_denom_p = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vsubq_s32(vdupq_n_s32(151), esp)));
    let max_s = vcvtq_u32_f32(vfmaq_f32(half, max_c, inv_denom_p));
    let bump = vceqq_u32(max_s, vdupq_n_u32(512)); // all-ones lanes where equal
    // exp_shared = esp + (bump ? 1 : 0).
    let exp_shared = vsubq_s32(esp, vreinterpretq_s32_u32(bump));

    let inv_denom =
        vreinterpretq_f32_s32(vshlq_n_s32::<23>(vsubq_s32(vdupq_n_s32(151), exp_shared)));
    let mask9 = vdupq_n_u32(0x1ff);
    let scaled = vandq_u32(vcvtq_u32_f32(vfmaq_f32(half, c, inv_denom)), mask9);

    let rs = vgetq_lane_u32::<0>(scaled);
    let gs = vgetq_lane_u32::<1>(scaled);
    let bs = vgetq_lane_u32::<2>(scaled);
    let exp = vgetq_lane_s32::<0>(exp_shared) as u32;

    ((exp & 0x1f) << 27) | ((bs & 0x1ff) << 18) | ((gs & 0x1ff) << 9) | (rs & 0x1ff)
}

/// NEON path for `E5B9G9R9_UFLOAT_PACK32` stores.
///
/// **Not part of the public API.** See [`store_e5b9g9r9_f32_sse4_1`].
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn store_e5b9g9r9_f32_neon(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_e5b9g9r9_f32_neon");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: NEON is available per the enclosing fn's contract; all `n`
        // words get written.
        unsafe {
            for i in 0..n {
                let lanes = vld1q_f32(src.add(i * 4));
                dst.add(i).write_unaligned(encode_e5_pixel_neon(lanes));
            }
        }
    })
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use crate::processing::x86::has_avx512;

    fn buf_from(pixels: Vec<[f32; 4]>) -> Buffer<f32> {
        let width = pixels.len().max(1) as u32;
        Buffer {
            pixels,
            width,
            height: 1,
        }
    }

    /// A broad set of RGB triples covering every documented edge case.
    fn edge_case_pixels() -> Vec<[f32; 4]> {
        let mut v = vec![
            [0.0, 0.0, 0.0, 1.0],             // all zero
            [1e-30, 1e-30, 1e-30, 1.0],       // tiny (denormal-ish exponent)
            [1e-30, 0.0, 5e-20, 1.0],         // tiny mixed
            [0.5, 0.25, 0.125, 1.0],          // sub-1.0
            [0.3, 0.3, 0.3, 1.0],             // equal channels
            [100.0, 0.01, 1.0, 1.0],          // wildly different magnitudes
            [65408.0, 65408.0, 65408.0, 1.0], // exactly SHARED_EXP_MAX
            [70000.0, 1.0, 0.001, 1.0],       // above max → clamp/overflow
            [1e30, 2e30, 3e30, 1.0],          // far above max
            [-1.0, -0.5, -0.001, 1.0],        // negatives → 0
            [-5.0, 2.0, 0.0, 1.0],            // mixed sign
            [f32::NAN, 1.0, 2.0, 1.0],        // NaN behavior is unspecified
            [f32::INFINITY, 0.5, 0.25, 1.0],  // +Inf → max
            [1.0, f32::NAN, f32::INFINITY, 1.0],
            [511.0, 512.0, 513.0, 1.0], // near-binade rounding
            [0.9999, 1.0, 1.0001, 1.0], // rounding across 1.0
        ];
        // A pseudo-random sweep across a wide dynamic range.
        let mut state = 0x9e37_79b9u32;
        for _ in 0..6000 {
            let mut nxt = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state
            };
            let mk = |bits: u32| {
                // Map to a value in roughly [2^-30, 2^18], occasionally negative.
                let exp = ((bits >> 3) % 60) as i32 - 40;
                let mant = 1.0 + ((bits & 0x7ff) as f32) / 2048.0;
                let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
                sign * mant * 2f32.powi(exp)
            };
            v.push([mk(nxt()), mk(nxt()), mk(nxt()), 1.0]);
        }
        v
    }

    /// Decode each packed word and allow at most one mantissa step at the larger
    /// of the two shared exponents. NaN inputs are excluded because their packed
    /// representation is intentionally not specified across SIMD implementations.
    fn assert_within_one_mantissa_step(got: &[u8], want: &[u8], buf: &Buffer<f32>, ctx: &str) {
        use crate::processing::load_kernels::e5b9g9r9::decode;
        assert_eq!(got.len(), want.len(), "{ctx}: length");
        for (i, (g, w)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
            if buf.pixels[i][..3].iter().any(|channel| channel.is_nan()) {
                continue;
            }
            let gw = u32::from_le_bytes(g.try_into().unwrap());
            let ww = u32::from_le_bytes(w.try_into().unwrap());
            let gd = decode(gw);
            let wd = decode(ww);
            let g_step = f32::from_bits((((gw >> 27) & 0x1f) + 103) << 23);
            let w_step = f32::from_bits((((ww >> 27) & 0x1f) + 103) << 23);
            let tolerance = g_step.max(w_step);
            for c in 0..3 {
                assert!(
                    (gd[c] - wd[c]).abs() <= tolerance,
                    "{ctx}: pixel {i} channel {c}: simd={} scalar={} tolerance={tolerance}",
                    gd[c],
                    wd[c],
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sse4_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let buf = buf_from(edge_case_pixels());
        let got = unsafe { store_e5b9g9r9_f32_sse4_1(&buf) };
        let want = store_e5b9g9r9_f32_serial(&buf);
        assert_within_one_mantissa_step(&got, &want, &buf, "sse4.1");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let buf = buf_from(edge_case_pixels());
        let got = unsafe { store_e5b9g9r9_f32_avx2_fma(&buf) };
        let want = store_e5b9g9r9_f32_serial(&buf);
        assert_within_one_mantissa_step(&got, &want, &buf, "avx2");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_matches_scalar() {
        if !has_avx512() {
            return;
        }
        let buf = buf_from(edge_case_pixels());
        let got = unsafe { store_e5b9g9r9_f32_avx512(&buf) };
        let want = store_e5b9g9r9_f32_serial(&buf);
        assert_within_one_mantissa_step(&got, &want, &buf, "avx512");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let buf = buf_from(edge_case_pixels());
        let got = unsafe { store_e5b9g9r9_f32_neon(&buf) };
        let want = store_e5b9g9r9_f32_serial(&buf);
        assert_within_one_mantissa_step(&got, &want, &buf, "neon");
    }

    /// Every pixel count 1..=33 crosses each kernel's block/tail boundary.
    #[test]
    fn tails_match_scalar() {
        let all = edge_case_pixels();
        for n in 1..=33usize {
            let buf = buf_from(all[..n].to_vec());
            let want = store_e5b9g9r9_f32_serial(&buf);
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse4.1") {
                    let got = unsafe { store_e5b9g9r9_f32_sse4_1(&buf) };
                    assert_within_one_mantissa_step(&got, &want, &buf, &format!("sse4.1 n={n}"));
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    let got = unsafe { store_e5b9g9r9_f32_avx2_fma(&buf) };
                    assert_within_one_mantissa_step(&got, &want, &buf, &format!("avx2 n={n}"));
                }
                if has_avx512() {
                    let got = unsafe { store_e5b9g9r9_f32_avx512(&buf) };
                    assert_within_one_mantissa_step(&got, &want, &buf, &format!("avx512 n={n}"));
                }
            }
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                let got = unsafe { store_e5b9g9r9_f32_neon(&buf) };
                assert_within_one_mantissa_step(&got, &want, &buf, &format!("neon n={n}"));
            }
        }
    }
}
