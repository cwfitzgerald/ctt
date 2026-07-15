//! `E5B9G9R9_UFLOAT_PACK32` loader (shared-exponent RGB).
//!
//! One little-endian u32 per pixel: three 9-bit mantissas (R at bits `[0,9)`,
//! G at `[9,18)`, B at `[18,27)`) sharing a single 5-bit exponent at `[27,32)`.
//! There is no alpha channel and no sign bit — every channel is
//! `mantissa * 2^(exp - 24)` (bias 15, 9 mantissa bits). Alpha loads as 1.0.

use crate::error::Result;
use crate::processing::dispatch::dispatch_simd;
use crate::surface::Surface;

use super::{Buffer, read_pixels_f32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Decode one packed word into linear `[R, G, B]` (alpha handled by the caller).
///
/// The shared exponent has no implicit leading one and no denormal special
/// case, so the value is simply `mantissa * 2^(exp - 24)`. The scale is built
/// directly as an IEEE-754 f32: `exp` spans `0..=31`, so `exp - 24 + 127`
/// (i.e. `exp + 103`) lands in `103..=134`, always a valid normal exponent.
#[inline]
pub(crate) fn decode(word: u32) -> [f32; 3] {
    let r_mant = (word & 0x1ff) as f32;
    let g_mant = ((word >> 9) & 0x1ff) as f32;
    let b_mant = ((word >> 18) & 0x1ff) as f32;
    let exp = (word >> 27) & 0x1f;
    let scale = f32::from_bits((exp + 103) << 23);
    [r_mant * scale, g_mant * scale, b_mant * scale]
}

pub fn load_e5b9g9r9_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_e5b9g9r9_f32");

    dispatch_simd! {
        x86_64: {
            avx512: load_e5b9g9r9_f32_avx512(surface),
            avx2: load_e5b9g9r9_f32_avx2(surface),
            sse4_1: load_e5b9g9r9_f32_sse4_1(surface),
        },
        aarch64: {
            neon: load_e5b9g9r9_f32_neon(surface),
        },
    }

    load_e5b9g9r9_f32_serial(surface)
}

/// Serial reference path.
///
/// **Not part of the public API.** Exposed for benches/tests to compare
/// against the SIMD kernels.
#[doc(hidden)]
pub fn load_e5b9g9r9_f32_serial(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_e5b9g9r9_f32_serial");
    read_pixels_f32(surface, 1, 4, |bytes, lanes| {
        let word = u32::from_le_bytes(<[u8; 4]>::try_from(bytes).expect("4-byte pixel"));
        let [r, g, b] = decode(word);
        lanes[0] = r;
        lanes[1] = g;
        lanes[2] = b;
        // lanes[3] stays at the default 1.0.
    })
}

/// Decode one packed word at `ptr` into `[R, G, B, A]` f32 lanes (A = 1.0),
/// shared between the SSE4.1 main loop and the AVX2 fast path's tail.
///
/// The word is broadcast across all four lanes; per-lane fixed shifts (blended
/// into place, since SSE has no variable shift) isolate the three mantissas,
/// and the shared exponent is turned into a power-of-two scale via an integer
/// `(exp + 103) << 23` bit construction. `mantissa * scale` is exact (a ≤9-bit
/// integer times a power of two), so the result is bit-identical to [`decode`].
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
/// * `ptr` must be valid for a 4-byte read.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn decode_e5_pixel_sse4_1(ptr: *const u8) -> __m128 {
    // SAFETY: caller guarantees 4 valid bytes at ptr.
    let word = unsafe { ptr.cast::<u32>().read_unaligned() };
    let w = _mm_set1_epi32(word as i32);

    // Lane layout after the blends: [w, w >> 9, w >> 18, w]. Only lanes 0..3
    // are masked to the 9-bit mantissas; lane 3 is masked to 0 (alpha bypass).
    let sh9 = _mm_srli_epi32::<9>(w);
    let sh18 = _mm_srli_epi32::<18>(w);
    let m = _mm_blend_ps::<0b0010>(_mm_castsi128_ps(w), _mm_castsi128_ps(sh9));
    let m = _mm_blend_ps::<0b0100>(m, _mm_castsi128_ps(sh18));
    let mant = _mm_and_si128(_mm_castps_si128(m), _mm_setr_epi32(0x1ff, 0x1ff, 0x1ff, 0));
    let mantf = _mm_cvtepi32_ps(mant);

    // Shared exponent (identical across the broadcast lanes) → power-of-two scale.
    let exp = _mm_srli_epi32::<27>(w);
    let scale_bits = _mm_slli_epi32::<23>(_mm_add_epi32(exp, _mm_set1_epi32(103)));
    let scale = _mm_castsi128_ps(scale_bits);

    let res = _mm_mul_ps(mantf, scale);
    // Alpha lane → 1.0.
    _mm_blend_ps::<0b1000>(res, _mm_set1_ps(1.0))
}

/// Decode a row one pixel (4 bytes → 4 f32) at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn load_row_e5_sse4_1(row: &[u8], dst: *mut f32) {
    // SAFETY: sse4.1 enabled by the enclosing target_feature; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;
        while x + 4 <= row.len() {
            let res = decode_e5_pixel_sse4_1(row.as_ptr().add(x));
            _mm_storeu_ps(dst.add(x), res);
            x += 4;
        }
    }
}

/// SSE4.1 path for `E5B9G9R9_UFLOAT_PACK32`.
///
/// Decodes one pixel (4 bytes → 4 f32) per iteration via
/// [`decode_e5_pixel_sse4_1`], bit-identical to the scalar [`decode`].
///
/// **Not part of the public API.** Exposed as `pub` + `doc(hidden)` only so
/// `benches/` (a separate crate) can measure this kernel directly without
/// going through dispatch. Real callers should use [`load_e5b9g9r9_f32`].
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn load_e5b9g9r9_f32_sse4_1(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_e5b9g9r9_f32_sse4_1");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: sse4.1 is available per the enclosing fn's contract.
            load_row_e5_sse4_1(row, dst)
        })
    }
}

/// Decode two consecutive packed words at `ptr` into `[R,G,B,A] × 2` f32 lanes
/// (A = 1.0). Each word is broadcast to its own 4-lane group and a per-lane
/// variable shift (`srlv`) isolates the mantissas, so the interleaved result
/// falls straight out with no transpose.
///
/// # Safety
/// * AVX2 must be available (enforced by `target_feature`).
/// * `ptr` must be valid for an 8-byte read.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn decode_e5_pixels_avx2(ptr: *const u8) -> __m256 {
    // SAFETY: caller guarantees 8 valid bytes at ptr.
    let two = unsafe { _mm_loadl_epi64(ptr as *const __m128i) };
    // Broadcast word 0 to lanes 0..3 and word 1 to lanes 4..7.
    let idx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    let words = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(two), idx);

    let shift = _mm256_setr_epi32(0, 9, 18, 0, 0, 9, 18, 0);
    let mask = _mm256_setr_epi32(0x1ff, 0x1ff, 0x1ff, 0, 0x1ff, 0x1ff, 0x1ff, 0);
    let mant = _mm256_and_si256(_mm256_srlv_epi32(words, shift), mask);
    let mantf = _mm256_cvtepi32_ps(mant);

    let exp = _mm256_srli_epi32::<27>(words);
    let scale_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(exp, _mm256_set1_epi32(103)));
    let scale = _mm256_castsi256_ps(scale_bits);

    let res = _mm256_mul_ps(mantf, scale);
    let alpha = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, -1));
    _mm256_blendv_ps(res, _mm256_set1_ps(1.0), alpha)
}

/// Decode a row with 2-pixel AVX2 blocks and a 1-pixel SSE4.1 tail (avx2
/// implies sse4.1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_row_e5_avx2(row: &[u8], dst: *mut f32) {
    // SAFETY: avx2 enabled by the enclosing target_feature; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;

        // 2 pixels (8 input bytes, 8 output f32s) per iteration.
        while x + 8 <= row.len() {
            let res = decode_e5_pixels_avx2(row.as_ptr().add(x));
            _mm256_storeu_ps(dst.add(x), res);
            x += 8;
        }

        if x < row.len() {
            let res = decode_e5_pixel_sse4_1(row.as_ptr().add(x));
            _mm_storeu_ps(dst.add(x), res);
        }
    }
}

/// AVX2 path for `E5B9G9R9_UFLOAT_PACK32`.
///
/// Decodes two pixels (8 bytes → 8 f32) per iteration via
/// [`decode_e5_pixels_avx2`]; any 1-pixel remainder is handled by the SSE4.1
/// [`decode_e5_pixel_sse4_1`] helper, keeping the tail vectorized.
///
/// **Not part of the public API.** See [`load_e5b9g9r9_f32_sse4_1`]; use
/// [`load_e5b9g9r9_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn load_e5b9g9r9_f32_avx2(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_e5b9g9r9_f32_avx2");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: avx2 is available per the enclosing fn's contract.
            load_row_e5_avx2(row, dst)
        })
    }
}

/// Decode four of the sixteen packed words in `words` into `[R,G,B,A] × 4`
/// f32 lanes (A = 1.0), shared between the AVX-512 main loop and its masked
/// tail. `idx` broadcasts each selected word to its own 4-lane group via
/// `permutexvar`, then a per-lane variable shift isolates the mantissas.
///
/// # Safety
/// * AVX-512 F/BW/VL must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn decode_e5_quad_avx512(words: __m512i, idx: __m512i) -> __m512 {
    let quad = _mm512_permutexvar_epi32(idx, words);

    let shift = _mm512_broadcast_i32x4(_mm_setr_epi32(0, 9, 18, 0));
    let mask = _mm512_broadcast_i32x4(_mm_setr_epi32(0x1ff, 0x1ff, 0x1ff, 0));
    let mant = _mm512_and_si512(_mm512_srlv_epi32(quad, shift), mask);
    let mantf = _mm512_cvtepi32_ps(mant);

    let exp = _mm512_srli_epi32::<27>(quad);
    let scale_bits = _mm512_slli_epi32::<23>(_mm512_add_epi32(exp, _mm512_set1_epi32(103)));
    let scale = _mm512_castsi512_ps(scale_bits);

    let res = _mm512_mul_ps(mantf, scale);
    let alpha: __mmask16 = 0b1000_1000_1000_1000;
    _mm512_mask_blend_ps(alpha, res, _mm512_set1_ps(1.0))
}

/// Decode a row with 16-pixel AVX-512 blocks and a masked 1-15 pixel tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn load_row_e5_avx512(row: &[u8], dst: *mut f32) {
    // Broadcast word i of each 4-word group to lanes 4i..4i+3; adding a group
    // offset selects the later groups.
    let base = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    let idx = [
        base,
        _mm512_add_epi32(base, _mm512_set1_epi32(4)),
        _mm512_add_epi32(base, _mm512_set1_epi32(8)),
        _mm512_add_epi32(base, _mm512_set1_epi32(12)),
    ];

    // SAFETY: masked loads/stores only touch enabled lanes; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;

        // 16 pixels (64 input bytes, 64 output f32s) per iteration.
        while x + 64 <= row.len() {
            let words = _mm512_loadu_si512(row.as_ptr().add(x) as *const __m512i);
            for (k, &idx) in idx.iter().enumerate() {
                _mm512_storeu_ps(dst.add(x + k * 16), decode_e5_quad_avx512(words, idx));
            }
            x += 64;
        }

        // 1-15 word tail: masked-load the remaining words and masked-store
        // exactly the valid lanes of each 4-pixel group.
        let rem = (row.len() - x) / 4;
        if rem > 0 {
            let words =
                _mm512_maskz_loadu_epi32((1u16 << rem) - 1, row.as_ptr().add(x) as *const i32);
            let live = (1u64 << (rem * 4)) - 1;
            for (k, &idx) in idx.iter().enumerate() {
                let mask = (live >> (16 * k)) as __mmask16;
                if mask != 0 {
                    let res = decode_e5_quad_avx512(words, idx);
                    _mm512_mask_storeu_ps(dst.add(x + k * 16), mask, res);
                }
            }
        }
    }
}

/// AVX-512 path for `E5B9G9R9_UFLOAT_PACK32`.
///
/// Decodes sixteen pixels (64 bytes → 64 f32) per iteration — one 512-bit word
/// load feeding four broadcast/decode/store rounds via
/// [`decode_e5_quad_avx512`]. A 1-15 pixel masked tail handles any remainder
/// without dropping to a narrower width.
///
/// **Not part of the public API.** See [`load_e5b9g9r9_f32_sse4_1`]; use
/// [`load_e5b9g9r9_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn load_e5b9g9r9_f32_avx512(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_e5b9g9r9_f32_avx512");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: avx512 is available per the enclosing fn's contract.
            load_row_e5_avx512(row, dst)
        })
    }
}

/// Decode one packed word into `[R, G, B, A]` f32 lanes (A = 1.0) on NEON.
///
/// The word is broadcast to all four lanes; a per-lane variable shift
/// (`vshlq_u32` with negative counts = right shift) isolates the mantissas,
/// and the shared exponent becomes a power-of-two scale via `(exp + 103) << 23`.
/// Bit-identical to the scalar [`decode`].
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_e5_pixel_neon(word: u32) -> float32x4_t {
    let w = vdupq_n_u32(word);

    // Right-shift lanes by [0, 9, 18, 0] (negative counts shift right).
    let shift_amt: [i32; 4] = [0, -9, -18, 0];
    // SAFETY: `shift_amt` holds 4 i32s.
    let shift = unsafe { vld1q_s32(shift_amt.as_ptr()) };
    let mask_arr: [u32; 4] = [0x1ff, 0x1ff, 0x1ff, 0];
    // SAFETY: `mask_arr` holds 4 u32s.
    let mask = unsafe { vld1q_u32(mask_arr.as_ptr()) };

    let mant = vandq_u32(vshlq_u32(w, shift), mask);
    let mantf = vcvtq_f32_u32(mant);

    let exp = vshrq_n_u32::<27>(w);
    let scale_bits = vshlq_n_u32::<23>(vaddq_u32(exp, vdupq_n_u32(103)));
    let scale = vreinterpretq_f32_u32(scale_bits);

    let res = vmulq_f32(mantf, scale);
    vsetq_lane_f32::<3>(1.0, res)
}

/// Decode a row with unrolled 4-pixel NEON blocks and a per-pixel tail.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_row_e5_neon(row: &[u8], dst: *mut f32) {
    // SAFETY: NEON enabled by the enclosing target_feature; the caller
    // guarantees `row.len()` lanes of destination space.
    unsafe {
        let mut x = 0usize;

        // 4 pixels per iteration.
        while x + 16 <= row.len() {
            for k in 0..4 {
                let word = (row.as_ptr().add(x + k * 4) as *const u32).read_unaligned();
                vst1q_f32(dst.add(x + k * 4), decode_e5_pixel_neon(word));
            }
            x += 16;
        }

        // 1-3 pixel tail.
        while x < row.len() {
            let word = (row.as_ptr().add(x) as *const u32).read_unaligned();
            vst1q_f32(dst.add(x), decode_e5_pixel_neon(word));
            x += 4;
        }
    }
}

/// NEON path for `E5B9G9R9_UFLOAT_PACK32`.
///
/// Decodes four pixels per iteration (unrolled per-pixel decodes) and handles
/// any 1-3 pixel tail with the same per-pixel NEON decoder.
///
/// **Not part of the public API.** See [`load_e5b9g9r9_f32_sse4_1`]; use
/// [`load_e5b9g9r9_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn load_e5b9g9r9_f32_neon(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_e5b9g9r9_f32_neon");
    // SAFETY: the row helper writes every lane of each row it is given.
    unsafe {
        super::load_packed_rows(surface, |row, dst| {
            // SAFETY: NEON is available per the enclosing fn's contract.
            load_row_e5_neon(row, dst)
        })
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    use crate::alpha::AlphaMode;
    #[cfg(target_arch = "x86_64")]
    use crate::processing::x86::has_avx512;
    use crate::surface::{ColorSpace, Surface};

    fn e5_surface(words: &[u32], width: u32, height: u32, stride: u32) -> Surface {
        let mut data = vec![0u8; (stride * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let w = words[y * width as usize + x];
                let off = y * stride as usize + x * 4;
                data[off..off + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::E5B9G9R9_UFLOAT_PACK32,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    /// Assert two decoded buffers are bit-identical.
    fn assert_bit_exact(a: &[[f32; 4]], b: &[[f32; 4]]) {
        assert_eq!(a.len(), b.len());
        for (i, (pa, pb)) in a.iter().zip(b).enumerate() {
            for c in 0..4 {
                assert_eq!(
                    pa[c].to_bits(),
                    pb[c].to_bits(),
                    "pixel {i} lane {c}: {} vs {}",
                    pa[c],
                    pb[c],
                );
            }
        }
    }

    /// Words spanning every exponent with a spread of mantissas on each channel,
    /// plus a pseudo-random sweep.
    fn sweep_words() -> Vec<u32> {
        let mants = [0u32, 1, 255, 256, 511];
        let mut words = Vec::new();
        for exp in 0..32u32 {
            for &r in &mants {
                for &g in &mants {
                    for &b in &mants {
                        words.push((exp << 27) | (b << 18) | (g << 9) | r);
                    }
                }
            }
        }
        // Pseudo-random sweep of several thousand words.
        let mut state = 0x1234_5678u32;
        for _ in 0..8000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            words.push(state);
        }
        words
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse4_matches_scalar_full_sweep() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let words = sweep_words();
        let width = words.len() as u32;
        let surface = e5_surface(&words, width, 1, width * 4);
        let simd = unsafe { load_e5b9g9r9_f32_sse4_1(&surface).unwrap() };
        let serial = load_e5b9g9r9_f32_serial(&surface).unwrap();
        assert_bit_exact(&simd.pixels, &serial.pixels);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_full_sweep() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let words = sweep_words();
        let width = words.len() as u32;
        let surface = e5_surface(&words, width, 1, width * 4);
        let simd = unsafe { load_e5b9g9r9_f32_avx2(&surface).unwrap() };
        let serial = load_e5b9g9r9_f32_serial(&surface).unwrap();
        assert_bit_exact(&simd.pixels, &serial.pixels);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar_full_sweep() {
        if !has_avx512() {
            return;
        }
        let words = sweep_words();
        let width = words.len() as u32;
        let surface = e5_surface(&words, width, 1, width * 4);
        let simd = unsafe { load_e5b9g9r9_f32_avx512(&surface).unwrap() };
        let serial = load_e5b9g9r9_f32_serial(&surface).unwrap();
        assert_bit_exact(&simd.pixels, &serial.pixels);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_full_sweep() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let words = sweep_words();
        let width = words.len() as u32;
        let surface = e5_surface(&words, width, 1, width * 4);
        let simd = unsafe { load_e5b9g9r9_f32_neon(&surface).unwrap() };
        let serial = load_e5b9g9r9_f32_serial(&surface).unwrap();
        assert_bit_exact(&simd.pixels, &serial.pixels);
    }

    /// Odd widths (1..=17) exercise every tail residue on each kernel.
    #[test]
    fn odd_widths_match_scalar() {
        let base = sweep_words();
        for width in 1..=17u32 {
            let words: Vec<u32> = base.iter().copied().take(width as usize).collect();
            let surface = e5_surface(&words, width, 1, width * 4);
            let serial = load_e5b9g9r9_f32_serial(&surface).unwrap();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse4.1") {
                    let simd = unsafe { load_e5b9g9r9_f32_sse4_1(&surface).unwrap() };
                    assert_bit_exact(&simd.pixels, &serial.pixels);
                }
                if is_x86_feature_detected!("avx2") {
                    let simd = unsafe { load_e5b9g9r9_f32_avx2(&surface).unwrap() };
                    assert_bit_exact(&simd.pixels, &serial.pixels);
                }
                if has_avx512() {
                    let simd = unsafe { load_e5b9g9r9_f32_avx512(&surface).unwrap() };
                    assert_bit_exact(&simd.pixels, &serial.pixels);
                }
            }
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                let simd = unsafe { load_e5b9g9r9_f32_neon(&surface).unwrap() };
                assert_bit_exact(&simd.pixels, &serial.pixels);
            }
        }
    }

    /// Multi-row surfaces with junk stride padding must decode identically to
    /// the scalar path (padding is skipped).
    #[test]
    fn stride_padding_is_skipped() {
        let width = 5u32;
        let height = 3u32;
        let stride = width * 4 + 12; // 12 bytes of padding per row
        let base = sweep_words();
        let words: Vec<u32> = base
            .iter()
            .copied()
            .take((width * height) as usize)
            .collect();

        let mut surface = e5_surface(&words, width, height, stride);
        // Poison the padding so a bug that reads it would diverge.
        for y in 0..height as usize {
            let pad = y * stride as usize + (width * 4) as usize;
            for b in &mut surface.data[pad..pad + 12] {
                *b = 0xAB;
            }
        }
        let serial = load_e5b9g9r9_f32_serial(&surface).unwrap();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
                let simd = unsafe { load_e5b9g9r9_f32_sse4_1(&surface).unwrap() };
                assert_bit_exact(&simd.pixels, &serial.pixels);
            }
            if is_x86_feature_detected!("avx2") {
                let simd = unsafe { load_e5b9g9r9_f32_avx2(&surface).unwrap() };
                assert_bit_exact(&simd.pixels, &serial.pixels);
            }
            if has_avx512() {
                let simd = unsafe { load_e5b9g9r9_f32_avx512(&surface).unwrap() };
                assert_bit_exact(&simd.pixels, &serial.pixels);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            let simd = unsafe { load_e5b9g9r9_f32_neon(&surface).unwrap() };
            assert_bit_exact(&simd.pixels, &serial.pixels);
        }
    }
}
