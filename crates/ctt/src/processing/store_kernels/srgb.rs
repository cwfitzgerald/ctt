//! sRGB-encoded u8 stores.
//!
//! Shared state:
//! * [`OETF_LUT`] — 4097-entry scalar lookup table over `[0, 1]` for the
//!   linearly-interpolated fast scalar path.
//! * [`SRGB_OETF_MINIMAX_A`]..[`SRGB_OETF_MINIMAX_D`] — piecewise minimax
//!   approximation used by the SIMD fast paths.

use std::sync::LazyLock;

use super::{Buffer, write_pixels};

const OETF_LUT_SIZE: usize = 4096;

/// sRGB OETF lookup table — 4097 entries over [0, 1] for linear interpolation.
static OETF_LUT: LazyLock<[f32; OETF_LUT_SIZE + 1]> = LazyLock::new(|| {
    let mut table = [0.0f32; OETF_LUT_SIZE + 1];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = i as f32 / OETF_LUT_SIZE as f32;
        *entry = srgb_oetf_precise(c);
    }
    table
});

fn srgb_oetf_precise(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[inline(always)]
fn srgb_oetf_fast(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    let scaled = c * OETF_LUT_SIZE as f32;
    let idx = scaled as usize;
    if idx >= OETF_LUT_SIZE {
        return OETF_LUT[OETF_LUT_SIZE];
    }
    let frac = scaled - idx as f32;
    OETF_LUT[idx] + frac * (OETF_LUT[idx + 1] - OETF_LUT[idx])
}

// Piecewise approximation of `1.055 * x^(1/2.4) - 0.055` over `[0.0031308, 1]`,
// with the curve branch designed to bit-exact round-trip against our load-side
// sRGB EOTF approximation in `load_kernels/srgb.rs`. Curve form, with `x` the
// clamped linear value:
//   quarter = sqrt(sqrt(x))                       (2× sqrtps, full precision)
//   diff    = quarter - A
//   r3      = rsqrt_refined(diff)                 (1× rsqrtps + 1 NR step)
//   inner   = r3 - B
//   cube    = inner * inner * inner
//   rcp     = rcp_refined(cube)                   (1× rcpps + 1 NR step)
//   curve   = rcp - C
// Constants tuned by PySR + differential evolution against the union of:
//   * byte-roundtrip points `(EOTF_approx(b/255), b/255)` for b in 11..=255,
//   * smooth OETF grid on `[0.0031308, 1]` against the exact `x^(1/2.4)`.
// With NR refinement, the worst-case adversarial error over ±1.5·2⁻¹² on each
// approximate op stays ~8e-4 — comfortably inside ±0.5/255. See
// `srgb-approx.py` / `srgb-opt.py` / `srgb-sim.py`.
#[cfg(target_arch = "x86_64")]
const SRGB_OETF_MINIMAX_A: f32 = 0.075_058_33;
#[cfg(target_arch = "x86_64")]
const SRGB_OETF_MINIMAX_B: f32 = 0.048_553_98;
#[cfg(target_arch = "x86_64")]
const SRGB_OETF_MINIMAX_C: f32 = 0.027_579_91;

// One Newton-Raphson iteration squares the initial ~1.5·2⁻¹² relative error of
// `rsqrtps`/`rcpps` to ~1.3e-7, below float32 ε — so the refined result is
// indistinguishable from a full-precision reciprocal / reciprocal-sqrt.

/// `1 / sqrt(x)` via `rsqrtps` + 1 NR step: `y' = y · (1.5 − 0.5·x·y²)`.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn rsqrt_refined_sse4_1(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;
    let y = _mm_rsqrt_ps(x);
    let y_sq = _mm_mul_ps(y, y);
    let half_x = _mm_mul_ps(_mm_set1_ps(0.5), x);
    let correction = _mm_sub_ps(_mm_set1_ps(1.5), _mm_mul_ps(half_x, y_sq));
    _mm_mul_ps(y, correction)
}

/// `1 / x` via `rcpps` + 1 NR step: `y' = y · (2 − x·y)`.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn rcp_refined_sse4_1(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;
    let y = _mm_rcp_ps(x);
    let correction = _mm_sub_ps(_mm_set1_ps(2.0), _mm_mul_ps(x, y));
    _mm_mul_ps(y, correction)
}

/// AVX2+FMA counterpart of [`rsqrt_refined_sse4_1`], using one `vfnmadd`.
///
/// # Safety
/// * AVX2 and FMA must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn rsqrt_refined_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let y = _mm256_rsqrt_ps(x);
    let y_sq = _mm256_mul_ps(y, y);
    let half_x = _mm256_mul_ps(_mm256_set1_ps(0.5), x);
    let correction = _mm256_fnmadd_ps(half_x, y_sq, _mm256_set1_ps(1.5));
    _mm256_mul_ps(y, correction)
}

/// AVX2+FMA counterpart of [`rcp_refined_sse4_1`], using one `vfnmadd`.
///
/// # Safety
/// * AVX2 and FMA must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn rcp_refined_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let y = _mm256_rcp_ps(x);
    let correction = _mm256_fnmadd_ps(x, y, _mm256_set1_ps(2.0));
    _mm256_mul_ps(y, correction)
}

pub fn store_srgb8_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32");

    #[cfg(target_arch = "x86_64")]
    {
        if channels == 4 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime check confirms avx2 + fma are available.
            return unsafe { store_srgb8_f32_avx2_fma::<false>(buf) };
        }
        if channels == 4 && is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { store_srgb8_f32_sse4_1::<false>(buf) };
        }
    }

    write_pixels(buf, channels, 1, |lanes, bytes| {
        for (c, (&lane, byte)) in lanes.iter().zip(bytes.iter_mut()).enumerate() {
            let encoded = if c < 3 {
                srgb_oetf_fast(lane)
            } else {
                lane.clamp(0.0, 1.0)
            };
            *byte = (encoded * 255.0).round() as u8;
        }
    })
}

pub fn store_bgra8_srgb_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgra8_srgb_f32");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime check confirms avx2 + fma are available.
            return unsafe { store_srgb8_f32_avx2_fma::<true>(buf) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { store_srgb8_f32_sse4_1::<true>(buf) };
        }
    }

    write_pixels(buf, 4, 1, |lanes, bytes| {
        let arr = <&mut [u8; 4]>::try_from(bytes).expect("4-byte pixel");
        arr[0] = (srgb_oetf_fast(lanes[2]) * 255.0).round() as u8;
        arr[1] = (srgb_oetf_fast(lanes[1]) * 255.0).round() as u8;
        arr[2] = (srgb_oetf_fast(lanes[0]) * 255.0).round() as u8;
        arr[3] = (lanes[3].clamp(0.0, 1.0) * 255.0).round() as u8;
    })
}

pub fn store_bgr8_srgb_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgr8_srgb_f32");
    write_pixels(buf, 3, 1, |lanes, bytes| {
        let arr = <&mut [u8; 3]>::try_from(bytes).expect("3-byte pixel");
        arr[0] = (srgb_oetf_fast(lanes[2]) * 255.0).round() as u8;
        arr[1] = (srgb_oetf_fast(lanes[1]) * 255.0).round() as u8;
        arr[2] = (srgb_oetf_fast(lanes[0]) * 255.0).round() as u8;
    })
}

/// Encode one `[R, G, B, A]` linear-f32 pixel into a packed `u32` of four
/// sRGB-u8 bytes, shared between the SSE4.1 main loop and the AVX2 fast
/// path's tail.
///
/// Byte order in the returned `u32`:
/// * `BGRA = false` → byte 0 = R, byte 3 = A (used by `R8G8B8A8_SRGB`).
/// * `BGRA = true`  → byte 0 = B, byte 3 = A (used by `B8G8R8A8_SRGB`).
///
/// The R↔B lane swap is a single `shufps` on the input vector; every
/// subsequent op is symmetric across the three color lanes, so the math path
/// is identical and the compiler monomorphizes two nearly-free copies.
///
/// Piecewise form with `x = clamp(lane, 0, 1)` on the color lanes:
/// * `x < 0.0031308`: `12.92 * x` (linear segment of the sRGB spec).
/// * `x >= 0.0031308`: NR-refined rsqrt/rcp approximation of
///   `1.055 * x^(1/2.4) - 0.055` (see `SRGB_OETF_MINIMAX_*`). Worst-case
///   adversarial error ~8e-4 — well inside the ±0.5/255 u8-roundtrip margin,
///   and tuned to bit-exact invert the load-side EOTF approximation.
///
/// The alpha lane bypasses the OETF and is written as a straight unorm.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn encode_srgb_pixel_sse4_1<const BGRA: bool>(lanes: std::arch::x86_64::__m128) -> u32 {
    use std::arch::x86_64::*;

    // Swap R↔B lanes so the BGRA output byte order falls out of the packus
    // chain below. Alpha stays at lane 3 so `alpha_lane_mask` still applies.
    // `0b11_00_01_10` picks lanes [2, 1, 0, 3] → `[B, G, R, A]`.
    let lanes = if BGRA {
        _mm_shuffle_ps::<0b11_00_01_10>(lanes, lanes)
    } else {
        lanes
    };

    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let x = _mm_max_ps(_mm_min_ps(lanes, one), zero);

    let coeff_a = _mm_set1_ps(SRGB_OETF_MINIMAX_A);
    let coeff_b = _mm_set1_ps(SRGB_OETF_MINIMAX_B);
    let coeff_c = _mm_set1_ps(SRGB_OETF_MINIMAX_C);
    let linear_scale = _mm_set1_ps(12.92);
    let threshold = _mm_set1_ps(0.003_130_8);
    let scale_255 = _mm_set1_ps(255.0);
    // Lane 3 is the alpha channel in the [R,G,B,A] layout.
    let alpha_lane_mask = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, -1));

    let quarter = _mm_sqrt_ps(_mm_sqrt_ps(x));
    let diff = _mm_sub_ps(quarter, coeff_a);
    // SAFETY: helpers require sse4.1, matched by the enclosing `target_feature`.
    let r3 = unsafe { rsqrt_refined_sse4_1(diff) };
    let inner = _mm_sub_ps(r3, coeff_b);
    let cube = _mm_mul_ps(_mm_mul_ps(inner, inner), inner);
    // SAFETY: as above.
    let rcp = unsafe { rcp_refined_sse4_1(cube) };
    let curve = _mm_sub_ps(rcp, coeff_c);
    let linear = _mm_mul_ps(x, linear_scale);

    // Select the linear segment for x < threshold, curve otherwise.
    let use_linear = _mm_cmplt_ps(x, threshold);
    let rgb = _mm_blendv_ps(curve, linear, use_linear);
    // Alpha lane bypasses the OETF.
    let encoded = _mm_blendv_ps(rgb, x, alpha_lane_mask);

    // Round-to-nearest-even + saturating pack i32 → u16 → u8.
    let scaled = _mm_mul_ps(encoded, scale_255);
    let i32s = _mm_cvtps_epi32(scaled);
    let u16s = _mm_packus_epi32(i32s, i32s);
    let u8s = _mm_packus_epi16(u16s, u16s);
    _mm_cvtsi128_si32(u8s) as u32
}

/// SSE4.1 path for 4-channel sRGB store, parameterized by output byte order:
/// `BGRA = false` → `R8G8B8A8_SRGB`; `BGRA = true` → `B8G8R8A8_SRGB`.
///
/// Processes one pixel (4 f32 → 4 u8 bytes) per iteration via
/// [`encode_srgb_pixel_sse4_1`]. See that helper for the piecewise form and
/// accuracy guarantees.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn store_srgb8_f32_sse4_1<const BGRA: bool>(buf: &Buffer<f32>) -> Vec<u8> {
    use std::arch::x86_64::*;

    profiling::scope!("store_srgb8_f32_sse4_1");

    let total_pixels = buf.pixels.len();
    let mut out = vec![0u8; total_pixels * 4];
    let src_base = buf.pixels.as_ptr() as *const f32;
    let dst_base = out.as_mut_ptr();

    // SAFETY: every intrinsic and pointer op below runs with sse4.1 enabled
    // (target_feature on the enclosing fn). `src_base` spans `total_pixels * 4`
    // f32 lanes by construction of `Buffer`, and `out` was sized for the same
    // `total_pixels * 4` u8 bytes.
    unsafe {
        for i in 0..total_pixels {
            let lanes = _mm_loadu_ps(src_base.add(i * 4));
            let packed = encode_srgb_pixel_sse4_1::<BGRA>(lanes);
            dst_base.add(i * 4).cast::<u32>().write_unaligned(packed);
        }
    }

    out
}

/// AVX2 + FMA path for 4-channel sRGB store, parameterized by output byte
/// order: `BGRA = false` → `R8G8B8A8_SRGB`; `BGRA = true` → `B8G8R8A8_SRGB`.
///
/// Processes two pixels (8 f32 → 8 u8 bytes) per iteration; any 1-pixel odd
/// tail is handled by [`encode_srgb_pixel_sse4_1`] so the tail stays
/// vectorized and consistent with the SSE4.1 fast path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn store_srgb8_f32_avx2_fma<const BGRA: bool>(buf: &Buffer<f32>) -> Vec<u8> {
    use std::arch::x86_64::*;

    profiling::scope!("store_srgb8_f32_avx2_fma");

    let total_pixels = buf.pixels.len();
    let mut out = vec![0u8; total_pixels * 4];
    let src_base = buf.pixels.as_ptr() as *const f32;
    let dst_base = out.as_mut_ptr();

    let pair_count = total_pixels / 2;
    let tail = total_pixels % 2;

    // SAFETY: every intrinsic and pointer op below runs with avx2+fma enabled
    // (target_feature on the enclosing fn). `src_base` spans `total_pixels * 4`
    // f32 lanes by construction of `Buffer`, and `out` was sized for the same
    // `total_pixels * 4` u8 bytes.
    unsafe {
        let coeff_a = _mm256_set1_ps(SRGB_OETF_MINIMAX_A);
        let coeff_b = _mm256_set1_ps(SRGB_OETF_MINIMAX_B);
        let coeff_c = _mm256_set1_ps(SRGB_OETF_MINIMAX_C);
        let linear_scale = _mm256_set1_ps(12.92);
        let threshold = _mm256_set1_ps(0.003_130_8);
        let scale_255 = _mm256_set1_ps(255.0);
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        // Lanes 3 and 7 are the alpha channel in the [R,G,B,A,R,G,B,A] layout.
        let alpha_lane_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, -1));

        for i in 0..pair_count {
            let lanes = _mm256_loadu_ps(src_base.add(i * 8));
            // Per-128-bit-lane shuffle swaps R↔B within each pixel; alpha stays
            // at lanes 3 and 7. See `encode_srgb_pixel_sse4_1` for the mask.
            let lanes = if BGRA {
                _mm256_shuffle_ps::<0b11_00_01_10>(lanes, lanes)
            } else {
                lanes
            };
            let x = _mm256_max_ps(_mm256_min_ps(lanes, one), zero);

            let quarter = _mm256_sqrt_ps(_mm256_sqrt_ps(x));
            let diff = _mm256_sub_ps(quarter, coeff_a);
            let r3 = rsqrt_refined_avx2(diff);
            let inner = _mm256_sub_ps(r3, coeff_b);
            let cube = _mm256_mul_ps(_mm256_mul_ps(inner, inner), inner);
            let rcp = rcp_refined_avx2(cube);
            let curve = _mm256_sub_ps(rcp, coeff_c);
            let linear = _mm256_mul_ps(x, linear_scale);

            let use_linear = _mm256_cmp_ps::<_CMP_LT_OQ>(x, threshold);
            let rgb = _mm256_blendv_ps(curve, linear, use_linear);
            let encoded = _mm256_blendv_ps(rgb, x, alpha_lane_mask);

            let scaled = _mm256_mul_ps(encoded, scale_255);
            let i32s = _mm256_cvtps_epi32(scaled);
            let lo = _mm256_castsi256_si128(i32s);
            let hi = _mm256_extracti128_si256::<1>(i32s);
            let u16s = _mm_packus_epi32(lo, hi);
            let u8s = _mm_packus_epi16(u16s, u16s);

            _mm_storel_epi64(dst_base.add(i * 8) as *mut __m128i, u8s);
        }

        if tail == 1 {
            let lanes = _mm_loadu_ps(src_base.add(pair_count * 8));
            let packed = encode_srgb_pixel_sse4_1::<BGRA>(lanes);
            dst_base
                .add(pair_count * 8)
                .cast::<u32>()
                .write_unaligned(packed);
        }
    }

    out
}

#[cfg(all(test, target_arch = "x86_64"))]
mod simd_tests {
    use super::*;

    /// Reference LUT-based encoding that the SIMD fast paths must agree with
    /// within ±0.5/255. `BGRA` selects output byte order.
    fn reference_bytes<const BGRA: bool>(pixels: &[[f32; 4]]) -> Vec<u8> {
        let mut out = vec![0u8; pixels.len() * 4];
        for (pixel, bytes) in pixels.iter().zip(out.chunks_exact_mut(4)) {
            let (r, g, b, a) = (pixel[0], pixel[1], pixel[2], pixel[3]);
            let r_u8 = (srgb_oetf_fast(r) * 255.0).round() as u8;
            let g_u8 = (srgb_oetf_fast(g) * 255.0).round() as u8;
            let b_u8 = (srgb_oetf_fast(b) * 255.0).round() as u8;
            let a_u8 = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
            if BGRA {
                bytes.copy_from_slice(&[b_u8, g_u8, r_u8, a_u8]);
            } else {
                bytes.copy_from_slice(&[r_u8, g_u8, b_u8, a_u8]);
            }
        }
        out
    }

    fn assert_within_u8_tolerance<const BGRA: bool>(actual: &[u8], pixels: &[[f32; 4]]) {
        let reference = reference_bytes::<BGRA>(pixels);
        assert_eq!(actual.len(), reference.len());
        for (i, (&got, &want)) in actual.iter().zip(&reference).enumerate() {
            let diff = got.abs_diff(want);
            assert!(
                diff <= 1,
                "pixel {} byte {} got={got} want={want} (BGRA={BGRA})",
                i / 4,
                i % 4,
            );
        }
    }

    /// Every u8 byte value, fed through the exact EOTF, must round-trip back to
    /// the original byte via the approximation.
    fn u8_roundtrip_pixels() -> Vec<[f32; 4]> {
        fn srgb_eotf_exact(c: f32) -> f32 {
            if c <= 0.040_45 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        (0..=255u8)
            .map(|b| {
                let lin = srgb_eotf_exact(b as f32 / 255.0);
                [lin, lin, lin, b as f32 / 255.0]
            })
            .collect()
    }

    fn assert_roundtrips(bytes: &[u8]) {
        // R=G=B=A for every pixel, so RGBA and BGRA layouts both land on the
        // same byte values and we can share this check across variants.
        for b in 0..=255u8 {
            let base = b as usize * 4;
            assert_eq!(bytes[base], b, "byte 0 roundtrip failed for value {b}");
            assert_eq!(bytes[base + 1], b, "byte 1 roundtrip failed for value {b}");
            assert_eq!(bytes[base + 2], b, "byte 2 roundtrip failed for value {b}");
            assert_eq!(bytes[base + 3], b, "byte 3 roundtrip failed for value {b}");
        }
    }

    /// Sweep linear values across a fine grid (same sample density as the
    /// minimax fit) to catch systematic approximation drift.
    fn fine_grid_pixels() -> Vec<[f32; 4]> {
        let n = 1024usize;
        (0..n)
            .map(|i| {
                let x = i as f32 / (n - 1) as f32;
                [x, (x * 0.5 + 0.2).clamp(0.0, 1.0), x * x, x]
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

    #[test]
    fn sse4_rgba_matches_lut_within_u8_tolerance() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_sse4_1::<false>(&buf) };
        assert_within_u8_tolerance::<false>(&got, &buf.pixels);
    }

    #[test]
    fn sse4_bgra_matches_lut_within_u8_tolerance() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_sse4_1::<true>(&buf) };
        assert_within_u8_tolerance::<true>(&got, &buf.pixels);
    }

    #[test]
    fn avx2_rgba_matches_lut_within_u8_tolerance() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_avx2_fma::<false>(&buf) };
        assert_within_u8_tolerance::<false>(&got, &buf.pixels);
    }

    #[test]
    fn avx2_bgra_matches_lut_within_u8_tolerance() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_avx2_fma::<true>(&buf) };
        assert_within_u8_tolerance::<true>(&got, &buf.pixels);
    }

    #[test]
    fn sse4_rgba_u8_roundtrip_is_exact() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_sse4_1::<false>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn sse4_bgra_u8_roundtrip_is_exact() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_sse4_1::<true>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn avx2_rgba_u8_roundtrip_is_exact() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_avx2_fma::<false>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn avx2_bgra_u8_roundtrip_is_exact() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_avx2_fma::<true>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn avx2_odd_count_tail_matches_sse4_path() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        // 3 pixels: one AVX2 iteration (2 px) + one SSE4.1 tail (1 px).
        let pixels = vec![
            [0.0, 0.1, 0.5, 1.0],
            [0.25, 0.75, 0.9, 0.5],
            [0.123, 0.456, 0.789, 0.321],
        ];
        let buf = buf_from(pixels.clone());

        for bgra in [false, true] {
            let (avx2, sse4_tail) = unsafe {
                let tail_buf = buf_from(vec![pixels[2]]);
                if bgra {
                    (
                        store_srgb8_f32_avx2_fma::<true>(&buf),
                        store_srgb8_f32_sse4_1::<true>(&tail_buf),
                    )
                } else {
                    (
                        store_srgb8_f32_avx2_fma::<false>(&buf),
                        store_srgb8_f32_sse4_1::<false>(&tail_buf),
                    )
                }
            };
            assert_eq!(&avx2[8..12], &sse4_tail[..], "bgra={bgra}");
        }
    }

    #[test]
    fn bgra_swaps_r_and_b_bytes() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        // One pixel with distinct RGB values so the lane swap is observable.
        // Pick linear inputs that each hit a unique sRGB byte.
        let pixels = vec![[1.0, 0.0, 0.25, 0.5]];
        let buf = buf_from(pixels);

        let rgba = unsafe { store_srgb8_f32_sse4_1::<false>(&buf) };
        let bgra = unsafe { store_srgb8_f32_sse4_1::<true>(&buf) };

        // Byte 0/2 swapped; byte 1 (G) and byte 3 (A) unchanged.
        assert_eq!(bgra[0], rgba[2], "byte 0: BGRA should hold former-B");
        assert_eq!(bgra[1], rgba[1], "byte 1: G unchanged");
        assert_eq!(bgra[2], rgba[0], "byte 2: BGRA should hold former-R");
        assert_eq!(bgra[3], rgba[3], "byte 3: A unchanged");
    }

    #[test]
    fn clamps_out_of_range_inputs() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        // Values outside [0, 1] must clamp before OETF — negatives → 0, > 1 → 1.
        let pixels = vec![[-0.5, 2.0, 0.5, -0.1], [1.5, -1.0, 0.0, 1.2]];
        let buf = buf_from(pixels);
        let got = unsafe { store_srgb8_f32_sse4_1::<false>(&buf) };
        assert_eq!(got[0], 0); // R = -0.5 → 0
        assert_eq!(got[1], 255); // G = 2.0 → 255
        assert_eq!(got[3], 0); // A = -0.1 → 0
        assert_eq!(got[4], 255); // R = 1.5 → 255
        assert_eq!(got[5], 0); // G = -1.0 → 0
        assert_eq!(got[7], 255); // A = 1.2 → 255
    }
}
