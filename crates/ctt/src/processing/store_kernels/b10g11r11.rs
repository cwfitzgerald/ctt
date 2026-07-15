//! `B10G11R11_UFLOAT_PACK32` storer (packed unsigned floats).
//!
//! Encodes linear `[R, G, B]` into a single packed word: R and G as 11-bit
//! unsigned floats (6 mantissa bits), B as a 10-bit unsigned float (5 mantissa
//! bits), all with a 5-bit bias-15 exponent. Alpha is dropped.
//!
//! The per-channel packing is the standard round-to-nearest-even f32 → small
//! unsigned float conversion (as used by D3D's `R11G11B10_FLOAT`): negatives
//! flush to 0, finite overflow and +Inf map to Inf, NaN maps to NaN.

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

/// Encode a non-negative-biased f32 into a `5 + MANT_BITS`-bit unsigned float.
#[inline]
pub(crate) fn encode_small_float<const MANT_BITS: u32>(x: f32) -> u32 {
    let result_mask = (1u32 << (5 + MANT_BITS)) - 1;
    let drop = 23 - MANT_BITS;
    let inf = 0x1fu32 << MANT_BITS; // exponent all ones, zero mantissa
    // Largest finite f32 whose rounded value still fits: exponent 2^15 with a
    // full target mantissa.
    let max_bits = (142u32 << 23) | (((1u32 << MANT_BITS) - 1) << drop);

    let bits = x.to_bits();
    let mag = bits & 0x7fff_ffff;

    if mag > max_bits {
        // Overflow (finite too large or +Inf) → Inf; NaN → NaN; anything with
        // the sign bit set (incl. -Inf) → 0.
        if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
            return inf | ((1u32 << MANT_BITS) - 1); // NaN
        }
        if (bits & 0x8000_0000) != 0 {
            return 0;
        }
        return inf;
    }
    if (bits & 0x8000_0000) != 0 {
        return 0; // negatives (and -0.0) flush to zero
    }

    let mut i = mag;
    if mag < 0x3880_0000 {
        // Below the target's smallest normal (2^-14): build a subnormal.
        let shift = 113u32.wrapping_sub(mag >> 23);
        if shift >= 32 {
            return 0;
        }
        i = (0x0080_0000 | (mag & 0x007f_ffff)) >> shift;
    } else {
        i = i.wrapping_add(0xC800_0000); // rebias exponent from bias 127 to bias 15
    }

    let round = (1u32 << (drop - 1)) - 1;
    ((i + round + ((i >> drop) & 1)) >> drop) & result_mask
}

/// Encode linear `[R, G, B]` into a packed `B10G11R11` word.
#[inline]
pub(crate) fn encode(rgb: [f32; 3]) -> u32 {
    let r = encode_small_float::<6>(rgb[0]);
    let g = encode_small_float::<6>(rgb[1]);
    let b = encode_small_float::<5>(rgb[2]);
    r | (g << 11) | (b << 22)
}

pub fn store_b10g11r11_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_b10g11r11_f32");

    // No SSE4.1 tier: lacking a per-lane variable shift, the emulation
    // benchmarks slower than scalar bit-packing here (AVX2's `srlv` is what
    // makes the vector path win), so below AVX2 the scalar path handles it.
    dispatch_simd! {
        x86_64: {
            avx512: store_b10g11r11_f32_avx512(buf),
            avx2: store_b10g11r11_f32_avx2(buf),
        },
        aarch64: {
            neon: store_b10g11r11_f32_neon(buf),
        },
    }

    store_b10g11r11_f32_serial(buf)
}

/// Serial reference path.
///
/// **Not part of the public API.** Exposed for benches/tests to compare
/// against the SIMD kernels.
#[doc(hidden)]
pub fn store_b10g11r11_f32_serial(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_b10g11r11_f32_serial");
    write_pixels(buf, 1, 4, |lanes, bytes| {
        let word = encode([lanes[0], lanes[1], lanes[2]]);
        bytes.copy_from_slice(&word.to_le_bytes());
    })
}

// ---- x86_64 SIMD ----
//
// All SIMD paths replicate `encode_small_float` branchlessly per channel: the
// subnormal build (`(0x0080_0000 | mantissa) >> shift`, with `shift = 113 -
// exp`), the normal rebias (`mag + 0xC800_0000`), the round-to-nearest-even
// (`((i + round + ((i >> drop) & 1)) >> drop) & mask`), and the overflow
// selection (NaN → NaN, negative → 0, else → Inf). Every op mirrors the scalar
// integer arithmetic, so packed words are bit-identical to the scalar path.
//
// Input `[R, G, B, A]` pixels are transposed to channel vectors (AoS → SoA),
// each channel is encoded, and the results are combined as `r | (g << 11) |
// (b << 22)` into one word per pixel.

/// Per-lane logical right shift by the low 5 bits of `count` (SSE4.1 has no
/// variable shift). Counts of 32+ are the caller's responsibility.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn var_srl_sse4_1(v: __m128i, count: __m128i) -> __m128i {
    let mut r = v;
    // Conditionally shift by 16, 8, 4, 2, 1 based on each bit of the count.
    let m16 = _mm_cmpeq_epi32(_mm_and_si128(count, _mm_set1_epi32(16)), _mm_set1_epi32(16));
    r = _mm_blendv_epi8(r, _mm_srli_epi32::<16>(r), m16);
    let m8 = _mm_cmpeq_epi32(_mm_and_si128(count, _mm_set1_epi32(8)), _mm_set1_epi32(8));
    r = _mm_blendv_epi8(r, _mm_srli_epi32::<8>(r), m8);
    let m4 = _mm_cmpeq_epi32(_mm_and_si128(count, _mm_set1_epi32(4)), _mm_set1_epi32(4));
    r = _mm_blendv_epi8(r, _mm_srli_epi32::<4>(r), m4);
    let m2 = _mm_cmpeq_epi32(_mm_and_si128(count, _mm_set1_epi32(2)), _mm_set1_epi32(2));
    r = _mm_blendv_epi8(r, _mm_srli_epi32::<2>(r), m2);
    let m1 = _mm_cmpeq_epi32(_mm_and_si128(count, _mm_set1_epi32(1)), _mm_set1_epi32(1));
    r = _mm_blendv_epi8(r, _mm_srli_epi32::<1>(r), m1);
    r
}

/// Encode a channel vector into `5 + M`-bit unsigned-float codes (`DROP = 23 -
/// M`).
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn encode_codes_sse4_1<const M: i32, const DROP: i32>(x: __m128) -> __m128i {
    debug_assert_eq!(DROP, 23 - M);

    let mant_bits_mask = (1i32 << M) - 1;
    let result_mask = (1i32 << (5 + M)) - 1;
    let round = (1i32 << (DROP - 1)) - 1;
    let inf = 0x1fi32 << M;
    let max_bits = (142i32 << 23) | (mant_bits_mask << DROP);

    let bits = _mm_castps_si128(x);
    let mag = _mm_and_si128(bits, _mm_set1_epi32(0x7fff_ffff));

    let is_over = _mm_cmpgt_epi32(mag, _mm_set1_epi32(max_bits));
    let is_sub = _mm_cmpgt_epi32(_mm_set1_epi32(0x3880_0000), mag);

    // Subnormal path: (0x0080_0000 | mantissa) >> (113 - exp), zeroed at 32+.
    let numer = _mm_or_si128(
        _mm_set1_epi32(0x0080_0000),
        _mm_and_si128(mag, _mm_set1_epi32(0x007f_ffff)),
    );
    let shift = _mm_sub_epi32(_mm_set1_epi32(113), _mm_srli_epi32::<23>(mag));
    // SAFETY: sse4.1 is enabled by the enclosing target_feature.
    let sub_raw = unsafe { var_srl_sse4_1(numer, shift) };
    let shift_ge32 = _mm_cmpgt_epi32(shift, _mm_set1_epi32(31));
    let sub_i = _mm_andnot_si128(shift_ge32, sub_raw);

    // Normal path: rebias exponent from bias 127 to bias 15.
    let norm_i = _mm_add_epi32(mag, _mm_set1_epi32(0xC800_0000u32 as i32));

    let i = _mm_blendv_epi8(norm_i, sub_i, is_sub);

    // Round-to-nearest-even, then mask to the result width.
    let odd = _mm_and_si128(_mm_srli_epi32::<DROP>(i), _mm_set1_epi32(1));
    let t = _mm_add_epi32(_mm_add_epi32(i, _mm_set1_epi32(round)), odd);
    let rounded = _mm_and_si128(_mm_srli_epi32::<DROP>(t), _mm_set1_epi32(result_mask));

    // Overflow region: NaN → NaN, negative → 0, else → Inf.
    let is_inf_exp = _mm_cmpeq_epi32(
        _mm_and_si128(bits, _mm_set1_epi32(0x7f80_0000u32 as i32)),
        _mm_set1_epi32(0x7f80_0000u32 as i32),
    );
    let mant_nonzero = _mm_cmpgt_epi32(
        _mm_and_si128(bits, _mm_set1_epi32(0x007f_ffff)),
        _mm_setzero_si128(),
    );
    let is_nan = _mm_and_si128(is_inf_exp, mant_nonzero);
    let sign = _mm_srai_epi32::<31>(bits); // all-ones lane if negative

    let inf_vec = _mm_set1_epi32(inf);
    let nan_vec = _mm_set1_epi32(inf | mant_bits_mask);
    let over_res = _mm_blendv_epi8(inf_vec, _mm_setzero_si128(), sign);
    let over_res = _mm_blendv_epi8(over_res, nan_vec, is_nan);
    let nonover_res = _mm_blendv_epi8(rounded, _mm_setzero_si128(), sign);
    _mm_blendv_epi8(nonover_res, over_res, is_over)
}

/// Encode four planar RGB vectors into four packed words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn encode_planar_sse4_1(r: __m128, g: __m128, b: __m128) -> __m128i {
    let ri = unsafe { encode_codes_sse4_1::<6, 17>(r) };
    let gi = unsafe { encode_codes_sse4_1::<6, 17>(g) };
    let bi = unsafe { encode_codes_sse4_1::<5, 18>(b) };
    _mm_or_si128(
        ri,
        _mm_or_si128(_mm_slli_epi32::<11>(gi), _mm_slli_epi32::<22>(bi)),
    )
}

/// Encode exactly four pixels at `src` into four packed words at `dst`.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
/// * `src` must be valid for 16 f32 reads; `dst` for 4 u32 writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_quad_sse4_1(src: *const f32, dst: *mut u32) {
    // SAFETY: sse4.1 enabled; pointer ranges guaranteed by the caller.
    unsafe {
        let (r, g, b, _) = load_planar4_ps(src);
        let word = encode_planar_sse4_1(r, g, b);
        _mm_storeu_si128(dst as *mut __m128i, word);
    }
}

/// Encode a 1-3 pixel tail at `src` into `count` packed words at `dst`.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
/// * `src` must be valid for `count * 4` f32 reads; `dst` for `count` u32
///   writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_tail_sse4_1(src: *const f32, dst: *mut u32, count: usize) {
    // SAFETY: sse4.1 enabled; pointer ranges guaranteed by the caller.
    unsafe {
        let mut tmp = [0.0f32; 16];
        std::ptr::copy_nonoverlapping(src, tmp.as_mut_ptr(), count * 4);
        let mut words = [0u32; 4];
        store_quad_sse4_1(tmp.as_ptr(), words.as_mut_ptr());
        std::ptr::copy_nonoverlapping(words.as_ptr() as *const u8, dst as *mut u8, count * 4);
    }
}

/// Encode an arbitrary remainder with 4-pixel SSE blocks and one padded tail.
///
/// # Safety
/// * SSE4.1 must be available.
/// * `src` and `dst` must be valid for `count` pixels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_remainder_sse4_1(mut src: *const f32, mut dst: *mut u32, mut count: usize) {
    while count >= 4 {
        unsafe { store_quad_sse4_1(src, dst) };
        src = unsafe { src.add(16) };
        dst = unsafe { dst.add(4) };
        count -= 4;
    }
    if count > 0 {
        unsafe { store_tail_sse4_1(src, dst, count) };
    }
}

/// Encode a channel vector into `5 + M`-bit codes (AVX2, 8 lanes).
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn encode_codes_avx2<const M: i32, const DROP: i32>(x: __m256) -> __m256i {
    debug_assert_eq!(DROP, 23 - M);

    let mant_bits_mask = (1i32 << M) - 1;
    let result_mask = (1i32 << (5 + M)) - 1;
    let round = (1i32 << (DROP - 1)) - 1;
    let inf = 0x1fi32 << M;
    let max_bits = (142i32 << 23) | (mant_bits_mask << DROP);

    let bits = _mm256_castps_si256(x);
    let mag = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff_ffff));

    let is_over = _mm256_cmpgt_epi32(mag, _mm256_set1_epi32(max_bits));
    let is_sub = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x3880_0000), mag);

    // Subnormal path; VPSRLVD yields 0 for shift counts of 32+, matching the
    // scalar `shift >= 32 → 0` guard.
    let numer = _mm256_or_si256(
        _mm256_set1_epi32(0x0080_0000),
        _mm256_and_si256(mag, _mm256_set1_epi32(0x007f_ffff)),
    );
    let shift = _mm256_sub_epi32(_mm256_set1_epi32(113), _mm256_srli_epi32::<23>(mag));
    let sub_i = _mm256_srlv_epi32(numer, shift);

    let norm_i = _mm256_add_epi32(mag, _mm256_set1_epi32(0xC800_0000u32 as i32));
    let i = _mm256_blendv_epi8(norm_i, sub_i, is_sub);

    let odd = _mm256_and_si256(_mm256_srli_epi32::<DROP>(i), _mm256_set1_epi32(1));
    let t = _mm256_add_epi32(_mm256_add_epi32(i, _mm256_set1_epi32(round)), odd);
    let rounded = _mm256_and_si256(_mm256_srli_epi32::<DROP>(t), _mm256_set1_epi32(result_mask));

    let is_inf_exp = _mm256_cmpeq_epi32(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x7f80_0000u32 as i32)),
        _mm256_set1_epi32(0x7f80_0000u32 as i32),
    );
    let mant_nonzero = _mm256_cmpgt_epi32(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_setzero_si256(),
    );
    let is_nan = _mm256_and_si256(is_inf_exp, mant_nonzero);
    let sign = _mm256_srai_epi32::<31>(bits);

    let inf_vec = _mm256_set1_epi32(inf);
    let nan_vec = _mm256_set1_epi32(inf | mant_bits_mask);
    let over_res = _mm256_blendv_epi8(inf_vec, _mm256_setzero_si256(), sign);
    let over_res = _mm256_blendv_epi8(over_res, nan_vec, is_nan);
    let nonover_res = _mm256_blendv_epi8(rounded, _mm256_setzero_si256(), sign);
    _mm256_blendv_epi8(nonover_res, over_res, is_over)
}

/// AVX2 path for `B10G11R11_UFLOAT_PACK32`.
///
/// Processes eight pixels per iteration; the AoS → SoA deinterleave is done in
/// 128-bit halves and joined into the 256-bit encode. The 0-7 pixel remainder
/// reuses the SSE4.1 4-wide block and its zero-padded tail.
///
/// **Not part of the public API.** Use [`store_b10g11r11_f32`] for the stable,
/// runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn store_b10g11r11_f32_avx2(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_b10g11r11_f32_avx2");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: avx2 is available per the enclosing fn's contract (avx2
        // implies sse4.1 for the remainder); all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 8 <= n {
                let (r, g, b, _) = load_planar8_ps(src.add(i * 4));
                let ri = encode_codes_avx2::<6, 17>(r);
                let gi = encode_codes_avx2::<6, 17>(g);
                let bi = encode_codes_avx2::<5, 18>(b);
                let word = _mm256_or_si256(
                    ri,
                    _mm256_or_si256(_mm256_slli_epi32::<11>(gi), _mm256_slli_epi32::<22>(bi)),
                );
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, word);
                i += 8;
            }
            store_remainder_sse4_1(src.add(i * 4), dst.add(i), n - i);
        }
    })
}

/// Encode a channel vector into `5 + M`-bit codes (AVX-512, 16 lanes).
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn encode_codes_avx512<const M: u32, const DROP: u32>(x: __m512) -> __m512i {
    debug_assert_eq!(DROP, 23 - M);

    let mant_bits_mask = (1i32 << M) - 1;
    let result_mask = (1i32 << (5 + M)) - 1;
    let round = (1i32 << (DROP - 1)) - 1;
    let inf = 0x1fi32 << M;
    let max_bits = (142i32 << 23) | (mant_bits_mask << DROP);

    let bits = _mm512_castps_si512(x);
    let mag = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff_ffff));

    let is_over = _mm512_cmpgt_epi32_mask(mag, _mm512_set1_epi32(max_bits));
    let is_sub = _mm512_cmplt_epi32_mask(mag, _mm512_set1_epi32(0x3880_0000));

    // Subnormal path; VPSRLVD yields 0 for shift counts of 32+, matching the
    // scalar `shift >= 32 → 0` guard.
    let numer = _mm512_or_si512(
        _mm512_set1_epi32(0x0080_0000),
        _mm512_and_si512(mag, _mm512_set1_epi32(0x007f_ffff)),
    );
    let shift = _mm512_sub_epi32(_mm512_set1_epi32(113), _mm512_srli_epi32::<23>(mag));
    let sub_i = _mm512_srlv_epi32(numer, shift);

    let norm_i = _mm512_add_epi32(mag, _mm512_set1_epi32(0xC800_0000u32 as i32));
    let i = _mm512_mask_blend_epi32(is_sub, norm_i, sub_i);

    let odd = _mm512_and_si512(_mm512_srli_epi32::<DROP>(i), _mm512_set1_epi32(1));
    let t = _mm512_add_epi32(_mm512_add_epi32(i, _mm512_set1_epi32(round)), odd);
    let rounded = _mm512_and_si512(_mm512_srli_epi32::<DROP>(t), _mm512_set1_epi32(result_mask));

    let is_inf_exp = _mm512_cmpeq_epi32_mask(
        _mm512_and_si512(bits, _mm512_set1_epi32(0x7f80_0000u32 as i32)),
        _mm512_set1_epi32(0x7f80_0000u32 as i32),
    );
    let mant_nonzero = _mm512_cmpgt_epi32_mask(
        _mm512_and_si512(bits, _mm512_set1_epi32(0x007f_ffff)),
        _mm512_setzero_si512(),
    );
    let is_nan = is_inf_exp & mant_nonzero;
    let sign = _mm512_cmplt_epi32_mask(bits, _mm512_setzero_si512());

    let inf_vec = _mm512_set1_epi32(inf);
    let nan_vec = _mm512_set1_epi32(inf | mant_bits_mask);
    let over_res = _mm512_mask_blend_epi32(sign, inf_vec, _mm512_setzero_si512());
    let over_res = _mm512_mask_blend_epi32(is_nan, over_res, nan_vec);
    let nonover_res = _mm512_mask_blend_epi32(sign, rounded, _mm512_setzero_si512());
    _mm512_mask_blend_epi32(is_over, nonover_res, over_res)
}

/// Encode 16 pixels' planar RGB planes into 16 packed words.
///
/// # Safety
/// * AVX-512 F/VL/BW must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn encode_planar_avx512(r: __m512, g: __m512, b: __m512) -> __m512i {
    // SAFETY: avx512 is enabled by the enclosing target_feature.
    unsafe {
        let ri = encode_codes_avx512::<6, 17>(r);
        let gi = encode_codes_avx512::<6, 17>(g);
        let bi = encode_codes_avx512::<5, 18>(b);
        // 0xFE = three-way OR.
        _mm512_ternarylogic_epi32::<0xFE>(
            ri,
            _mm512_slli_epi32::<11>(gi),
            _mm512_slli_epi32::<22>(bi),
        )
    }
}

/// AVX-512 path for `B10G11R11_UFLOAT_PACK32`.
///
/// Processes sixteen pixels per iteration; the AoS → SoA deinterleave runs as
/// two levels of `vpermt2` permutes over 512-bit loads. The 0-15 pixel
/// remainder runs the same block behind masked loads/stores.
///
/// **Not part of the public API.** Use [`store_b10g11r11_f32`] for the stable,
/// runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn store_b10g11r11_f32_avx512(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_b10g11r11_f32_avx512");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: avx512 is available per the enclosing fn's contract; masked
        // loads/stores only touch enabled lanes; all `n` words get written.
        unsafe {
            let mut i = 0usize;
            while i + 16 <= n {
                let (r, g, b, _) = load_planar16_ps(src.add(i * 4));
                let words = encode_planar_avx512(r, g, b);
                _mm512_storeu_si512(dst.add(i) as *mut __m512i, words);
                i += 16;
            }
            if i < n {
                let tail = n - i;
                let (r, g, b, _) = load_planar16_mask_ps(src.add(i * 4), tail);
                let words = encode_planar_avx512(r, g, b);
                _mm512_mask_storeu_epi32(dst.add(i) as *mut i32, (1u16 << tail) - 1, words);
            }
        }
    })
}

// ---- AArch64 NEON ----

/// Encode a channel vector into `5 + M`-bit codes (NEON, 4 lanes, `DROP = 23 -
/// M`).
///
/// See the module-level SIMD note for the branchless encode; the math is
/// identical, just mapped to AArch64 NEON.
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn encode_codes_neon<const M: i32, const DROP: i32>(x: float32x4_t) -> uint32x4_t {
    debug_assert_eq!(DROP, 23 - M);

    let mant_bits_mask = ((1i32 << M) - 1) as u32;
    let result_mask = ((1i32 << (5 + M)) - 1) as u32;
    let round = ((1i32 << (DROP - 1)) - 1) as u32;
    let inf = (0x1fi32 << M) as u32;
    let max_bits = ((142i32 << 23) | (((1i32 << M) - 1) << DROP)) as u32;

    let bits = vreinterpretq_u32_f32(x);
    let mag = vandq_u32(bits, vdupq_n_u32(0x7fff_ffff));

    let is_over = vcgtq_u32(mag, vdupq_n_u32(max_bits));
    let is_sub = vcltq_u32(mag, vdupq_n_u32(0x3880_0000));

    // Subnormal path: (0x0080_0000 | mantissa) >> (113 - exp), zeroed at 32+.
    let numer = vorrq_u32(
        vdupq_n_u32(0x0080_0000),
        vandq_u32(mag, vdupq_n_u32(0x007f_ffff)),
    );
    let shift = vsubq_s32(
        vdupq_n_s32(113),
        vreinterpretq_s32_u32(vshrq_n_u32::<23>(mag)),
    );
    // Negative shift counts request a logical right shift; magnitudes of 32+
    // already yield 0, but mask explicitly to match the scalar guard.
    let sub_raw = vshlq_u32(numer, vnegq_s32(shift));
    let shift_ge32 = vcgtq_s32(shift, vdupq_n_s32(31));
    let sub_i = vbicq_u32(sub_raw, shift_ge32);

    let norm_i = vaddq_u32(mag, vdupq_n_u32(0xC800_0000));
    let i = vbslq_u32(is_sub, sub_i, norm_i);

    let odd = vandq_u32(vshrq_n_u32::<DROP>(i), vdupq_n_u32(1));
    let t = vaddq_u32(vaddq_u32(i, vdupq_n_u32(round)), odd);
    let rounded = vandq_u32(vshrq_n_u32::<DROP>(t), vdupq_n_u32(result_mask));

    let is_inf_exp = vceqq_u32(
        vandq_u32(bits, vdupq_n_u32(0x7f80_0000)),
        vdupq_n_u32(0x7f80_0000),
    );
    let mant_field = vandq_u32(bits, vdupq_n_u32(0x007f_ffff));
    let mant_nonzero = vtstq_u32(mant_field, mant_field);
    let is_nan = vandq_u32(is_inf_exp, mant_nonzero);
    // All-ones lane if the sign bit is set (arithmetic shift broadcasts it).
    let sign = vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_u32(bits)));

    let inf_vec = vdupq_n_u32(inf);
    let nan_vec = vdupq_n_u32(inf | mant_bits_mask);
    let over_res = vbslq_u32(sign, vdupq_n_u32(0), inf_vec);
    let over_res = vbslq_u32(is_nan, nan_vec, over_res);
    let nonover_res = vbslq_u32(sign, vdupq_n_u32(0), rounded);
    vbslq_u32(is_over, over_res, nonover_res)
}

/// NEON path for `B10G11R11_UFLOAT_PACK32`.
///
/// Processes four pixels per iteration; `vld4q_f32` deinterleaves the input
/// straight into channel vectors. A 1-3 pixel tail reuses the same 4-wide
/// encode on a zero-padded block.
///
/// **Not part of the public API.** Use [`store_b10g11r11_f32`] for the stable,
/// runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn store_b10g11r11_f32_neon(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_b10g11r11_f32_neon");
    super::store_packed_words(&buf.pixels, |src, dst, n| {
        // SAFETY: NEON is available per the enclosing fn's contract; all `n`
        // words get written.
        unsafe {
            let mut i = 0usize;
            while i + 4 <= n {
                let chans = vld4q_f32(src.add(i * 4));
                let ri = encode_codes_neon::<6, 17>(chans.0);
                let gi = encode_codes_neon::<6, 17>(chans.1);
                let bi = encode_codes_neon::<5, 18>(chans.2);
                let word = vorrq_u32(ri, vorrq_u32(vshlq_n_u32::<11>(gi), vshlq_n_u32::<22>(bi)));
                vst1q_u8(dst.add(i) as *mut u8, vreinterpretq_u8_u32(word));
                i += 4;
            }

            // 1-3 pixel tail via the same 4-wide encode on a zero-padded block.
            if i < n {
                let tail = n - i;
                let mut tmp = [0.0f32; 16];
                std::ptr::copy_nonoverlapping(src.add(i * 4), tmp.as_mut_ptr(), tail * 4);
                let chans = vld4q_f32(tmp.as_ptr());
                let ri = encode_codes_neon::<6, 17>(chans.0);
                let gi = encode_codes_neon::<6, 17>(chans.1);
                let bi = encode_codes_neon::<5, 18>(chans.2);
                let word = vorrq_u32(ri, vorrq_u32(vshlq_n_u32::<11>(gi), vshlq_n_u32::<22>(bi)));
                let mut words = [0u32; 4];
                vst1q_u32(words.as_mut_ptr(), word);
                for (k, &wd) in words.iter().enumerate().take(tail) {
                    dst.add(i + k).write_unaligned(wd);
                }
            }
        }
    })
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use crate::processing::x86::has_avx512;

    /// A broad set of per-channel f32 inputs spanning every encode branch.
    fn test_values() -> Vec<f32> {
        let mut v = vec![
            0.0f32,
            -0.0,
            f32::from_bits(0x0000_0001), // smallest positive f32 subnormal
            f32::from_bits(0x0080_0000), // smallest positive f32 normal
            2f32.powi(-25),
            2f32.powi(-20),
            2f32.powi(-15), // half the target min-normal
            2f32.powi(-14), // target min-normal
            2f32.powi(-14) * 1.5,
            2f32.powi(-13),
            0.001,
            0.01,
            0.1,
            0.25,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            7.0,
            100.0,
            1000.0,
            60000.0,
            64512.0, // 10-bit max representable
            65024.0, // 11-bit max representable
            65280.0, // just over 11-bit max → Inf for 11-bit
            70000.0, // over both → Inf
            1.0e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001), // NaN with minimal payload
            -1.0,
            -0.5,
            -100.0,
            -70000.0,
        ];
        // Pseudo-random sweep across a wide magnitude range, both signs.
        let mut state = 0x1234_5678u32;
        for _ in 0..2000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Map into a float in roughly [-2^17, 2^17] plus some tiny values.
            let f = f32::from_bits(state);
            if f.is_finite() {
                v.push(f % 131072.0);
            }
            v.push((state as f32 / u32::MAX as f32) * 4.0 - 1.0);
        }
        v
    }

    /// Build pixels whose channels each march through `values` (with an offset
    /// per channel so R, G, B rarely share the same branch on a given pixel).
    fn test_pixels() -> Vec<[f32; 4]> {
        let values = test_values();
        let n = values.len();
        (0..n)
            .map(|i| {
                [
                    values[i],
                    values[(i + n / 3) % n],
                    values[(i + 2 * n / 3) % n],
                    1.0,
                ]
            })
            .collect()
    }

    fn buf_from(pixels: Vec<[f32; 4]>) -> Buffer<f32> {
        let width = pixels.len() as u32;
        Buffer {
            pixels,
            width,
            height: 1,
        }
    }

    /// Packed words must be bit-identical to the scalar reference (including the
    /// NaN encoding, which is deterministic on the store side).
    fn assert_store_matches(actual: &[u8], buf: &Buffer<f32>) {
        let reference = store_b10g11r11_f32_serial(buf);
        assert_eq!(actual.len(), reference.len(), "byte count");
        for (i, (&got, &want)) in actual.iter().zip(&reference).enumerate() {
            assert_eq!(
                got,
                want,
                "pixel {} byte {}: got {got:#04x} want {want:#04x}",
                i / 4,
                i % 4
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_store_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let buf = buf_from(test_pixels());
        for n in [buf.pixels.len(), 1, 2, 3, 5, 8, 9, 13, 15, 16, 17] {
            let sub = buf_from(buf.pixels[..n].to_vec());
            let got = unsafe { store_b10g11r11_f32_avx2(&sub) };
            assert_store_matches(&got, &sub);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_store_matches_scalar() {
        if !has_avx512() {
            return;
        }
        let buf = buf_from(test_pixels());
        for n in [buf.pixels.len(), 1, 3, 7, 15, 16, 17, 31, 32, 33] {
            let sub = buf_from(buf.pixels[..n].to_vec());
            let got = unsafe { store_b10g11r11_f32_avx512(&sub) };
            assert_store_matches(&got, &sub);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_store_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let buf = buf_from(test_pixels());
        for n in [buf.pixels.len(), 1, 2, 3, 5, 8, 9, 13, 15, 16, 17] {
            let sub = buf_from(buf.pixels[..n].to_vec());
            let got = unsafe { store_b10g11r11_f32_neon(&sub) };
            assert_store_matches(&got, &sub);
        }
    }

    /// Round-trip: values that are exactly representable in the format must
    /// survive store → load and land on the same packed word the scalar picks.
    #[test]
    fn store_roundtrips_representable() {
        // Exhaustively cover every representable 11-bit R/G and 10-bit B code by
        // decoding it to f32 and re-encoding through the dispatched entry.
        use crate::processing::load_kernels::b10g11r11::decode_small_float;
        let mut pixels = Vec::new();
        for code in 0u32..2048 {
            let r = decode_small_float::<6>(code);
            let b = decode_small_float::<5>(code & 0x3ff);
            pixels.push([r, r, b, 1.0]);
        }
        let buf = buf_from(pixels);
        let got = store_b10g11r11_f32(&buf);
        assert_store_matches(&got, &buf);
    }
}
