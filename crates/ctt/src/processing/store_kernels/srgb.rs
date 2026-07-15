//! sRGB-encoded u8 stores and the in-place OETF pre-pass.
//!
//! Shared state:
//! * [`OETF_INTERP_LUT`] — interpolated lookup table over `[0, 1]` for the
//!   fast scalar path.
//! * [`SRGB_OETF_MINIMAX_A`]..[`SRGB_OETF_MINIMAX_C`] — piecewise minimax
//!   approximation used by the SIMD fast paths.
//!
//! The per-ISA `srgb_oetf_*` helpers evaluate the piecewise OETF curve on a
//! full vector register; the u8 store kernels and the f32 in-place pre-pass
//! ([`srgb_oetf_in_place_f32`]) both build on them.

use std::sync::LazyLock;

use crate::processing::curve_pass::{
    INTERP_LUT_SIZE, build_interp_lut, in_place_curve_pass, sample_interp_lut,
};
use crate::processing::dispatch::dispatch_simd;
#[cfg(target_arch = "x86_64")]
use crate::processing::x86::ALPHA_LANES_512;

use super::{Buffer, write_pixels};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// sRGB OETF lookup table over [0, 1] for linear interpolation.
static OETF_INTERP_LUT: LazyLock<[f32; INTERP_LUT_SIZE + 1]> =
    LazyLock::new(|| build_interp_lut(srgb_oetf_precise));

fn srgb_oetf_precise(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[inline(always)]
fn srgb_oetf_fast(c: f32) -> f32 {
    sample_interp_lut(&OETF_INTERP_LUT, c)
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
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SRGB_OETF_MINIMAX_A: f32 = 0.075_058_33;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SRGB_OETF_MINIMAX_B: f32 = 0.048_553_98;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SRGB_OETF_MINIMAX_C: f32 = 0.027_579_91;

// One Newton-Raphson iteration squares the initial ~1.5·2⁻¹² relative error of
// `rsqrtps`/`rcpps` to ~1.3e-7, below float32 ε — so the refined result is
// indistinguishable from a full-precision reciprocal / reciprocal-sqrt.

/// `1 / sqrt(x)` via `rsqrtps` + 1 NR step: `y' = y · (1.5 − 0.5·x·y²)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn rsqrt_refined_sse4_1(x: __m128) -> __m128 {
    let y = _mm_rsqrt_ps(x);
    let y_sq = _mm_mul_ps(y, y);
    let half_x = _mm_mul_ps(_mm_set1_ps(0.5), x);
    let correction = _mm_sub_ps(_mm_set1_ps(1.5), _mm_mul_ps(half_x, y_sq));
    _mm_mul_ps(y, correction)
}

/// `1 / x` via `rcpps` + 1 NR step: `y' = y · (2 − x·y)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn rcp_refined_sse4_1(x: __m128) -> __m128 {
    let y = _mm_rcp_ps(x);
    let correction = _mm_sub_ps(_mm_set1_ps(2.0), _mm_mul_ps(x, y));
    _mm_mul_ps(y, correction)
}

/// AVX2+FMA counterpart of [`rsqrt_refined_sse4_1`], using one `vfnmadd`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn rsqrt_refined_avx2(x: __m256) -> __m256 {
    let y = _mm256_rsqrt_ps(x);
    let y_sq = _mm256_mul_ps(y, y);
    let half_x = _mm256_mul_ps(_mm256_set1_ps(0.5), x);
    let correction = _mm256_fnmadd_ps(half_x, y_sq, _mm256_set1_ps(1.5));
    _mm256_mul_ps(y, correction)
}

/// AVX2+FMA counterpart of [`rcp_refined_sse4_1`], using one `vfnmadd`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn rcp_refined_avx2(x: __m256) -> __m256 {
    let y = _mm256_rcp_ps(x);
    let correction = _mm256_fnmadd_ps(x, y, _mm256_set1_ps(2.0));
    _mm256_mul_ps(y, correction)
}

/// AVX-512 counterpart of [`rsqrt_refined_avx2`]. Uses `rsqrt14ps`
/// (~2⁻¹⁴ initial relative error); one NR step squares that to ~2⁻²⁸,
/// comfortably below f32 ε.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn rsqrt_refined_avx512(x: __m512) -> __m512 {
    let y = _mm512_rsqrt14_ps(x);
    let y_sq = _mm512_mul_ps(y, y);
    let half_x = _mm512_mul_ps(_mm512_set1_ps(0.5), x);
    let correction = _mm512_fnmadd_ps(half_x, y_sq, _mm512_set1_ps(1.5));
    _mm512_mul_ps(y, correction)
}

/// AVX-512 counterpart of [`rcp_refined_avx2`]. Uses `rcp14ps`; one NR
/// step squares the error to well below f32 ε.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn rcp_refined_avx512(x: __m512) -> __m512 {
    let y = _mm512_rcp14_ps(x);
    let correction = _mm512_fnmadd_ps(x, y, _mm512_set1_ps(2.0));
    _mm512_mul_ps(y, correction)
}

pub fn store_srgb8_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32");

    if channels == 4 {
        dispatch_simd! {
            x86_64: {
                avx512: store_srgb8_f32_avx512::<false>(buf),
                avx2_fma: store_srgb8_f32_avx2_fma::<false>(buf),
                sse4_1: store_srgb8_f32_sse4_1::<false>(buf),
            },
            aarch64: {
                neon: store_srgb8_f32_neon::<false>(buf),
            },
        }
    }

    store_srgb8_f32_serial(buf, channels)
}

/// Serial LUT path for sRGB8 stores.
///
/// **Not part of the public API.** Exposed so benchmarks can compare the
/// scalar implementation directly against each runtime-selectable SIMD mode.
#[doc(hidden)]
pub fn store_srgb8_f32_serial(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32_serial");

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

    dispatch_simd! {
        x86_64: {
            avx512: store_srgb8_f32_avx512::<true>(buf),
            avx2_fma: store_srgb8_f32_avx2_fma::<true>(buf),
            sse4_1: store_srgb8_f32_sse4_1::<true>(buf),
        },
        aarch64: {
            neon: store_srgb8_f32_neon::<true>(buf),
        },
    }

    store_bgra8_srgb_f32_serial(buf)
}

/// Serial LUT path for BGRA sRGB8 stores.
///
/// **Not part of the public API.** Exposed so benchmarks can compare the
/// scalar implementation directly against each runtime-selectable SIMD mode.
#[doc(hidden)]
pub fn store_bgra8_srgb_f32_serial(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgra8_srgb_f32_serial");

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

in_place_curve_pass! {
    /// Apply the sRGB OETF in place to the RGB lanes of every pixel, leaving
    /// the alpha lane untouched.
    ///
    /// Pre-pass for stores to FormatKinds with no sRGB kernel variant (16+
    /// bit formats) whose target color_space is nonetheless `Srgb`: the
    /// buffer is OETF-encoded here, then written through the format's linear
    /// kernel.
    ///
    /// RGB lanes are clamped to `[0, 1]` before encoding; alpha is preserved
    /// bit-exactly. The SIMD paths use the same piecewise minimax curve as
    /// the u8 store kernels (worst-case error ~8e-4, tuned to bit-exact
    /// invert the load-side EOTF approximation); the serial path uses the
    /// interpolated `OETF_INTERP_LUT`.
    ///
    /// NaN color lanes are not handled consistently — the serial and NEON
    /// paths propagate the NaN while the x86 paths encode it as 1.0 — so
    /// callers must not rely on either behavior.
    pub fn srgb_oetf_in_place_f32;
    serial: srgb_oetf_in_place_f32_serial(srgb_oetf_fast);
    sse4_1: srgb_oetf_in_place_f32_sse4_1(srgb_oetf_sse4_1);
    avx2_fma: srgb_oetf_in_place_f32_avx2_fma(srgb_oetf_avx2);
    avx512: srgb_oetf_in_place_f32_avx512(srgb_oetf_avx512);
    neon: srgb_oetf_in_place_f32_neon(srgb_oetf_neon);
}

/// Piecewise sRGB OETF over all four lanes of `x`.
///
/// `x` must already be clamped to `[0, 1]`. Every lane goes through the
/// curve — callers with an alpha lane blend it back themselves.
///
/// Piecewise form:
/// * `x < 0.0031308`: `12.92 * x` (linear segment of the sRGB spec).
/// * `x >= 0.0031308`: NR-refined rsqrt/rcp approximation of
///   `1.055 * x^(1/2.4) - 0.055` (see `SRGB_OETF_MINIMAX_*`). Worst-case
///   adversarial error ~8e-4 — well inside the ±0.5/255 u8-roundtrip margin,
///   and tuned to bit-exact invert the load-side EOTF approximation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn srgb_oetf_sse4_1(x: __m128) -> __m128 {
    let quarter = _mm_sqrt_ps(_mm_sqrt_ps(x));
    let diff = _mm_sub_ps(quarter, _mm_set1_ps(SRGB_OETF_MINIMAX_A));
    let r3 = rsqrt_refined_sse4_1(diff);
    let inner = _mm_sub_ps(r3, _mm_set1_ps(SRGB_OETF_MINIMAX_B));
    let cube = _mm_mul_ps(_mm_mul_ps(inner, inner), inner);
    let rcp = rcp_refined_sse4_1(cube);
    let curve = _mm_sub_ps(rcp, _mm_set1_ps(SRGB_OETF_MINIMAX_C));
    let linear = _mm_mul_ps(x, _mm_set1_ps(12.92));

    // Select the linear segment for x < threshold, curve otherwise.
    let use_linear = _mm_cmplt_ps(x, _mm_set1_ps(0.003_130_8));
    _mm_blendv_ps(curve, linear, use_linear)
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
/// Color lanes go through [`srgb_oetf_sse4_1`] (see it for the piecewise
/// form and accuracy guarantees); the alpha lane bypasses the OETF and is
/// written as a straight unorm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn encode_srgb_pixel_sse4_1<const BGRA: bool>(lanes: __m128) -> u32 {
    // Swap R↔B lanes so the BGRA output byte order falls out of the packus
    // chain below. Alpha stays at lane 3 so `alpha_lane_mask` still applies.
    // `0b11_00_01_10` picks lanes [2, 1, 0, 3] → `[B, G, R, A]`.
    let lanes = if BGRA {
        _mm_shuffle_ps::<0b11_00_01_10>(lanes, lanes)
    } else {
        lanes
    };

    let x = _mm_max_ps(_mm_min_ps(lanes, _mm_set1_ps(1.0)), _mm_setzero_ps());
    let rgb = srgb_oetf_sse4_1(x);
    // Lane 3 (alpha) bypasses the OETF in the [R,G,B,A] layout.
    let alpha_lane_mask = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, -1));
    let encoded = _mm_blendv_ps(rgb, x, alpha_lane_mask);

    // Round-to-nearest-even + saturating pack i32 → u16 → u8.
    let scaled = _mm_mul_ps(encoded, _mm_set1_ps(255.0));
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
///
/// **Not part of the public API.** Exposed as `pub` + `doc(hidden)` only so
/// `benches/` (a separate crate) can measure this kernel directly without
/// going through dispatch. No stability guarantees — may be renamed,
/// removed, or have its signature changed across patch releases. Real
/// callers should use [`store_srgb8_f32`] or [`store_bgra8_srgb_f32`],
/// which pick the best available kernel at runtime.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn store_srgb8_f32_sse4_1<const BGRA: bool>(buf: &Buffer<f32>) -> Vec<u8> {
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

/// AVX2+FMA counterpart of [`srgb_oetf_sse4_1`] — the same piecewise curve
/// widened to 8 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn srgb_oetf_avx2(x: __m256) -> __m256 {
    let quarter = _mm256_sqrt_ps(_mm256_sqrt_ps(x));
    let diff = _mm256_sub_ps(quarter, _mm256_set1_ps(SRGB_OETF_MINIMAX_A));
    let r3 = rsqrt_refined_avx2(diff);
    let inner = _mm256_sub_ps(r3, _mm256_set1_ps(SRGB_OETF_MINIMAX_B));
    let cube = _mm256_mul_ps(_mm256_mul_ps(inner, inner), inner);
    let rcp = rcp_refined_avx2(cube);
    let curve = _mm256_sub_ps(rcp, _mm256_set1_ps(SRGB_OETF_MINIMAX_C));
    let linear = _mm256_mul_ps(x, _mm256_set1_ps(12.92));

    let use_linear = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(0.003_130_8));
    _mm256_blendv_ps(curve, linear, use_linear)
}

/// Encode one `__m256` of two `[R, G, B, A]` linear-f32 pixels into a packed
/// `__m128i` of 8 sRGB-u16 lanes (one per output byte, awaiting the final
/// `packus_epi16` that the caller chains across two of these to produce 16
/// bytes — or against itself for an 8-byte tail store).
///
/// `BGRA = false` → byte order `R,G,B,A` per pixel (`R8G8B8A8_SRGB`);
/// `BGRA = true`  → byte order `B,G,R,A` per pixel (`B8G8R8A8_SRGB`).
///
/// Color lanes go through [`srgb_oetf_avx2`]; the alpha lanes bypass the
/// OETF and are written as straight unorms.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn encode_srgb_pixels_avx2<const BGRA: bool>(lanes: __m256) -> __m128i {
    // Per-128-bit-lane shuffle swaps R↔B within each pixel; alpha stays at
    // lanes 3 and 7 so the alpha-bypass blend still applies.
    let lanes = if BGRA {
        _mm256_shuffle_ps::<0b11_00_01_10>(lanes, lanes)
    } else {
        lanes
    };

    let x = _mm256_max_ps(
        _mm256_min_ps(lanes, _mm256_set1_ps(1.0)),
        _mm256_setzero_ps(),
    );
    let rgb = srgb_oetf_avx2(x);
    // Lanes 3 and 7 are the alpha channel in the [R,G,B,A,R,G,B,A] layout.
    let alpha_lane_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, -1));
    let encoded = _mm256_blendv_ps(rgb, x, alpha_lane_mask);

    let scaled = _mm256_mul_ps(encoded, _mm256_set1_ps(255.0));
    let i32s = _mm256_cvtps_epi32(scaled);
    let lo = _mm256_castsi256_si128(i32s);
    let hi = _mm256_extracti128_si256::<1>(i32s);
    _mm_packus_epi32(lo, hi)
}

/// AVX2 + FMA path for 4-channel sRGB store, parameterized by output byte
/// order: `BGRA = false` → `R8G8B8A8_SRGB`; `BGRA = true` → `B8G8R8A8_SRGB`.
///
/// Processes four pixels (16 f32 → 16 u8 bytes) per iteration via two
/// chained calls to [`encode_srgb_pixels_avx2`], whose 8-u16 outputs are
/// joined by a single `packus_epi16` for one 16-byte unaligned store.
/// Running two independent OETF chains per iteration exposes more ILP than
/// the previous 2-pixel layout while keeping the same per-pixel intrinsic
/// count.
///
/// A 0-3 pixel tail is handled by an optional 2-pixel AVX2 step (helper
/// packed against itself for an 8-byte store) plus an optional 1-pixel
/// [`encode_srgb_pixel_sse4_1`] step, so the tail stays vectorized.
///
/// **Not part of the public API.** See [`store_srgb8_f32_sse4_1`] for the
/// rationale; use [`store_srgb8_f32`] / [`store_bgra8_srgb_f32`] for the
/// stable, runtime-dispatched entry points.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn store_srgb8_f32_avx2_fma<const BGRA: bool>(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32_avx2_fma");

    let total_pixels = buf.pixels.len();
    let mut out = vec![0u8; total_pixels * 4];
    let src_base = buf.pixels.as_ptr() as *const f32;
    let dst_base = out.as_mut_ptr();

    let quad_count = total_pixels / 4;
    let tail_pixels = total_pixels % 4;

    // SAFETY: every intrinsic and pointer op below runs with avx2+fma enabled
    // (target_feature on the enclosing fn). `src_base` spans `total_pixels * 4`
    // f32 lanes by construction of `Buffer`, and `out` was sized for the same
    // `total_pixels * 4` u8 bytes.
    unsafe {
        // 4 pixels (16 f32 → 16 u8 bytes) per iteration.
        for i in 0..quad_count {
            let lanes_a = _mm256_loadu_ps(src_base.add(i * 16));
            let lanes_b = _mm256_loadu_ps(src_base.add(i * 16 + 8));
            let u16s_a = encode_srgb_pixels_avx2::<BGRA>(lanes_a);
            let u16s_b = encode_srgb_pixels_avx2::<BGRA>(lanes_b);
            let u8s = _mm_packus_epi16(u16s_a, u16s_b);
            _mm_storeu_si128(dst_base.add(i * 16) as *mut __m128i, u8s);
        }

        // 0-3 pixel tail: at most one 2-pixel AVX2 step + one 1-pixel SSE4.1
        // step. `encode_srgb_pixel_sse4_1` requires only sse4.1, which avx2
        // implies — the call site is already inside the enclosing avx2+fma
        // target_feature scope.
        let mut offset = quad_count * 16;
        if tail_pixels >= 2 {
            let lanes = _mm256_loadu_ps(src_base.add(offset));
            let u16s = encode_srgb_pixels_avx2::<BGRA>(lanes);
            let u8s = _mm_packus_epi16(u16s, u16s);
            _mm_storel_epi64(dst_base.add(offset) as *mut __m128i, u8s);
            offset += 8;
        }
        if tail_pixels % 2 == 1 {
            let lanes = _mm_loadu_ps(src_base.add(offset));
            let packed = encode_srgb_pixel_sse4_1::<BGRA>(lanes);
            dst_base.add(offset).cast::<u32>().write_unaligned(packed);
        }
    }

    out
}

/// AVX-512 counterpart of [`srgb_oetf_sse4_1`] — the same piecewise curve
/// widened to 16 lanes, using `rsqrt14`/`rcp14` (initial ~2⁻¹⁴ error,
/// NR-refined well below f32 ε).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn srgb_oetf_avx512(x: __m512) -> __m512 {
    let quarter = _mm512_sqrt_ps(_mm512_sqrt_ps(x));
    let diff = _mm512_sub_ps(quarter, _mm512_set1_ps(SRGB_OETF_MINIMAX_A));
    let r3 = rsqrt_refined_avx512(diff);
    let inner = _mm512_sub_ps(r3, _mm512_set1_ps(SRGB_OETF_MINIMAX_B));
    let cube = _mm512_mul_ps(_mm512_mul_ps(inner, inner), inner);
    let rcp = rcp_refined_avx512(cube);
    let curve = _mm512_sub_ps(rcp, _mm512_set1_ps(SRGB_OETF_MINIMAX_C));
    let linear = _mm512_mul_ps(x, _mm512_set1_ps(12.92));

    let use_linear = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_set1_ps(0.003_130_8));
    _mm512_mask_blend_ps(use_linear, curve, linear)
}

/// Encode one `__m512` of four `[R, G, B, A]` linear-f32 pixels into a
/// packed `__m128i` of 16 sRGB-u8 bytes, shared between the AVX-512 main
/// loop and its masked 1-3 pixel tail.
///
/// `BGRA = false` → byte order `R,G,B,A` per pixel (`R8G8B8A8_SRGB`);
/// `BGRA = true`  → byte order `B,G,R,A` per pixel (`B8G8R8A8_SRGB`).
///
/// Color lanes go through [`srgb_oetf_avx512`]; the alpha lanes bypass the
/// OETF and are written as straight unorms.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
fn encode_srgb_pixels_avx512<const BGRA: bool>(lanes: __m512) -> __m128i {
    // Per-128-bit-lane shuffle swaps R↔B within each pixel; alpha stays at
    // lanes 3/7/11/15 so the alpha-bypass blend still applies.
    let lanes = if BGRA {
        _mm512_shuffle_ps::<0b11_00_01_10>(lanes, lanes)
    } else {
        lanes
    };

    let x = _mm512_max_ps(
        _mm512_min_ps(lanes, _mm512_set1_ps(1.0)),
        _mm512_setzero_ps(),
    );
    let rgb = srgb_oetf_avx512(x);
    let encoded = _mm512_mask_blend_ps(ALPHA_LANES_512, rgb, x);

    // Round-to-nearest-even + unsigned-saturating pack i32 → u8. Values are
    // in [0, ~255.2] after clamp + scale, so unsigned saturation is correct.
    let scaled = _mm512_mul_ps(encoded, _mm512_set1_ps(255.0));
    let i32s = _mm512_cvtps_epi32(scaled);
    _mm512_cvtusepi32_epi8(i32s)
}

/// AVX-512 path for 4-channel sRGB store, parameterized by output byte
/// order: `BGRA = false` → `R8G8B8A8_SRGB`; `BGRA = true` → `B8G8R8A8_SRGB`.
///
/// Processes four pixels (16 f32 → 16 u8 bytes) per iteration via
/// [`encode_srgb_pixels_avx512`]. A 1-3 pixel masked AVX-512 tail handles
/// any remainder without dropping to a narrower width.
///
/// **Not part of the public API.** See [`store_srgb8_f32_sse4_1`] for the
/// rationale; use [`store_srgb8_f32`] / [`store_bgra8_srgb_f32`] for the
/// stable, runtime-dispatched entry points.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn store_srgb8_f32_avx512<const BGRA: bool>(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32_avx512");

    let total_pixels = buf.pixels.len();
    let mut out = vec![0u8; total_pixels * 4];
    let src_base = buf.pixels.as_ptr() as *const f32;
    let dst_base = out.as_mut_ptr();

    let quad_count = total_pixels / 4;
    let tail_pixels = total_pixels % 4;

    // SAFETY: every intrinsic and pointer op below runs with avx512f+bw+vl
    // enabled (target_feature on the enclosing fn). `src_base` spans
    // `total_pixels * 4` f32 lanes by construction of `Buffer`, and `out`
    // was sized for the same `total_pixels * 4` u8 bytes.
    unsafe {
        // 4 pixels (16 f32 → 16 u8 bytes) per iteration.
        for i in 0..quad_count {
            let lanes = _mm512_loadu_ps(src_base.add(i * 16));
            let u8s = encode_srgb_pixels_avx512::<BGRA>(lanes);
            _mm_storeu_si128(dst_base.add(i * 16) as *mut __m128i, u8s);
        }

        // 1-3 pixel tail in a single masked AVX-512 iteration. The mask
        // suppresses both the f32 load and the byte store for lanes beyond
        // the remaining pixel count, so the intervening math can run
        // unmasked on garbage without touching memory.
        if tail_pixels > 0 {
            let offset = quad_count * 16;
            let mask: __mmask16 = (1u16 << (tail_pixels * 4)) - 1;

            let lanes = _mm512_maskz_loadu_ps(mask, src_base.add(offset));
            let u8s = encode_srgb_pixels_avx512::<BGRA>(lanes);
            _mm_mask_storeu_epi8(dst_base.add(offset) as *mut i8, mask, u8s);
        }
    }

    out
}

/// NEON counterpart of [`srgb_oetf_sse4_1`] — the same piecewise minimax
/// curve, but with the reciprocal steps mapped to full-precision AArch64
/// vector division.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn srgb_oetf_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);

    let quarter = vsqrtq_f32(vsqrtq_f32(x));
    let diff = vsubq_f32(quarter, vdupq_n_f32(SRGB_OETF_MINIMAX_A));
    let r3 = vdivq_f32(one, vsqrtq_f32(diff));
    let inner = vsubq_f32(r3, vdupq_n_f32(SRGB_OETF_MINIMAX_B));
    let cube = vmulq_f32(vmulq_f32(inner, inner), inner);
    let rcp = vdivq_f32(one, cube);
    let curve = vsubq_f32(rcp, vdupq_n_f32(SRGB_OETF_MINIMAX_C));
    let linear = vmulq_f32(x, vdupq_n_f32(12.92));

    let use_linear = vcltq_f32(x, vdupq_n_f32(0.003_130_8));
    vbslq_f32(use_linear, linear, curve)
}

/// Encode one `[R, G, B, A]` linear-f32 pixel into four saturated u16 lanes
/// containing the final sRGB-u8 byte values.
///
/// `BGRA = false` → byte order `R,G,B,A`; `BGRA = true` → `B,G,R,A`.
///
/// Color lanes go through [`srgb_oetf_neon`]; the alpha lane bypasses the
/// OETF and is written as a straight unorm.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn encode_srgb_lanes_neon<const BGRA: bool>(lanes: float32x4_t) -> uint16x4_t {
    let lanes = if BGRA {
        let r = vgetq_lane_f32::<0>(lanes);
        let b = vgetq_lane_f32::<2>(lanes);
        let lanes = vsetq_lane_f32::<0>(b, lanes);
        vsetq_lane_f32::<2>(r, lanes)
    } else {
        lanes
    };

    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let x = vmaxq_f32(vminq_f32(lanes, one), zero);

    let rgb = srgb_oetf_neon(x);
    let alpha_lane_mask = vsetq_lane_u32::<3>(u32::MAX, vdupq_n_u32(0));
    let encoded = vbslq_f32(alpha_lane_mask, x, rgb);
    let scaled = vmulq_f32(encoded, vdupq_n_f32(255.0));
    vqmovun_s32(vcvtnq_s32_f32(scaled))
}

/// Encode one pixel into a packed little-endian RGBA/BGRA `u32`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn encode_srgb_pixel_neon<const BGRA: bool>(lanes: float32x4_t) -> u32 {
    let u16s = encode_srgb_lanes_neon::<BGRA>(lanes);
    let u8s = vqmovn_u16(vcombine_u16(u16s, u16s));
    vget_lane_u32::<0>(vreinterpret_u32_u8(u8s))
}

/// NEON path for 4-channel sRGB store, parameterized by output byte order:
/// `BGRA = false` → `R8G8B8A8_SRGB`; `BGRA = true` → `B8G8R8A8_SRGB`.
///
/// Processes four pixels (16 f32 → 16 u8 bytes) per iteration and handles
/// any 1-3 pixel tail with the same per-pixel NEON encoder.
///
/// **Not part of the public API.** See [`store_srgb8_f32_sse4_1`] for the
/// rationale; use [`store_srgb8_f32`] / [`store_bgra8_srgb_f32`] for the
/// stable, runtime-dispatched entry points.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn store_srgb8_f32_neon<const BGRA: bool>(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32_neon");

    let total_pixels = buf.pixels.len();
    let mut out = vec![0u8; total_pixels * 4];
    let src_base = buf.pixels.as_ptr() as *const f32;
    let dst_base = out.as_mut_ptr();

    let quad_count = total_pixels / 4;
    let tail_pixels = total_pixels % 4;

    // SAFETY: every intrinsic and pointer op below runs with NEON enabled
    // (target_feature on the enclosing fn). `src_base` spans
    // `total_pixels * 4` f32 lanes by construction of `Buffer`, and `out`
    // was sized for the same `total_pixels * 4` u8 bytes.
    unsafe {
        for i in 0..quad_count {
            let offset = i * 16;
            let p0 = vld1q_f32(src_base.add(offset));
            let p1 = vld1q_f32(src_base.add(offset + 4));
            let p2 = vld1q_f32(src_base.add(offset + 8));
            let p3 = vld1q_f32(src_base.add(offset + 12));

            let u16s0 = encode_srgb_lanes_neon::<BGRA>(p0);
            let u16s1 = encode_srgb_lanes_neon::<BGRA>(p1);
            let u16s2 = encode_srgb_lanes_neon::<BGRA>(p2);
            let u16s3 = encode_srgb_lanes_neon::<BGRA>(p3);

            let bytes01 = vqmovn_u16(vcombine_u16(u16s0, u16s1));
            let bytes23 = vqmovn_u16(vcombine_u16(u16s2, u16s3));
            let bytes = vcombine_u8(bytes01, bytes23);
            vst1q_u8(dst_base.add(offset), bytes);
        }

        let mut offset = quad_count * 16;
        for _ in 0..tail_pixels {
            let lanes = vld1q_f32(src_base.add(offset));
            let packed = encode_srgb_pixel_neon::<BGRA>(lanes);
            dst_base.add(offset).cast::<u32>().write_unaligned(packed);
            offset += 4;
        }
    }

    out
}

#[cfg(test)]
mod oetf_in_place_tests {
    use super::*;

    /// Exact clamped OETF the in-place pre-pass approximates.
    pub(super) fn srgb_oetf_exact(c: f32) -> f32 {
        srgb_oetf_precise(c.clamp(0.0, 1.0))
    }

    /// One pixel per u8 lattice point, with the exact-EOTF linear value on
    /// the color lanes and the raw unorm on alpha. Every value must
    /// round-trip back to the original byte via the OETF approximations.
    pub(super) fn u8_roundtrip_pixels() -> Vec<[f32; 4]> {
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

    /// Byte-level roundtrip check for the u8 store kernels over
    /// [`u8_roundtrip_pixels`] output.
    pub(super) fn assert_roundtrips(bytes: &[u8]) {
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

    /// Every u8 byte value, fed through the exact EOTF, must land back on
    /// the same byte after the in-place OETF + unorm quantization — the same
    /// roundtrip guarantee the u8 store kernels provide.
    pub(super) fn assert_in_place_roundtrips(pixels: &[[f32; 4]]) {
        for (b, p) in pixels.iter().enumerate() {
            for (c, &lane) in p.iter().take(3).enumerate() {
                assert_eq!(
                    (lane * 255.0).round() as u8,
                    b as u8,
                    "lane {c} roundtrip failed for value {b}"
                );
            }
        }
    }

    /// RGB lanes within the documented worst-case error of the exact OETF;
    /// alpha lanes preserved bit-exactly.
    pub(super) fn assert_encoded_close(got: &[[f32; 4]], orig: &[[f32; 4]]) {
        assert_eq!(got.len(), orig.len());
        for (i, (g, o)) in got.iter().zip(orig).enumerate() {
            for c in 0..3 {
                let want = srgb_oetf_exact(o[c]);
                assert!(
                    (g[c] - want).abs() <= 1.5e-3,
                    "pixel {i} lane {c}: got={} want={want}",
                    g[c],
                );
            }
            assert_eq!(
                g[3].to_bits(),
                o[3].to_bits(),
                "pixel {i}: alpha must be preserved bit-exactly"
            );
        }
    }

    /// Grid over [0, 1] plus out-of-range RGB values (which must clamp) and
    /// adversarial alpha values (negative, > 1, NaN) that the pass must
    /// carry through untouched.
    pub(super) fn oetf_test_pixels() -> Vec<[f32; 4]> {
        let n = 1024usize;
        let mut pixels: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let x = i as f32 / (n - 1) as f32;
                [x, (x * 0.5 + 0.2).clamp(0.0, 1.0), x * x, x]
            })
            .collect();
        pixels.push([-0.5, 2.0, 0.001, -1.0]);
        pixels.push([1.5, -1.0, 0.5, 2.0]);
        pixels.push([0.25, 0.75, 1.0, f32::NAN]);
        pixels
    }

    #[test]
    fn serial_matches_exact_and_preserves_alpha() {
        let orig = oetf_test_pixels();
        let mut got = orig.clone();
        srgb_oetf_in_place_f32_serial(&mut got);
        assert_encoded_close(&got, &orig);
    }

    #[test]
    fn serial_oetf_in_place_u8_roundtrip_is_exact() {
        let mut pixels = u8_roundtrip_pixels();
        srgb_oetf_in_place_f32_serial(&mut pixels);
        assert_in_place_roundtrips(&pixels);
    }

    /// The dispatched entry point must agree with the exact OETF for every
    /// pixel count that exercises the SIMD kernels' main loops and tails.
    #[test]
    fn dispatch_matches_exact_across_tail_lengths() {
        let base = oetf_test_pixels();
        for n in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16] {
            let orig = base[..n].to_vec();
            let mut got = orig.clone();
            srgb_oetf_in_place_f32(&mut got);
            assert_encoded_close(&got, &orig);
        }

        let mut got = base.clone();
        srgb_oetf_in_place_f32(&mut got);
        assert_encoded_close(&got, &base);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod simd_tests {
    use super::oetf_in_place_tests::{
        assert_encoded_close, assert_in_place_roundtrips, assert_roundtrips, oetf_test_pixels,
        u8_roundtrip_pixels,
    };
    use super::*;
    use crate::processing::x86::{has_avx2_fma, has_avx512};

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
        if !has_avx2_fma() {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_avx2_fma::<false>(&buf) };
        assert_within_u8_tolerance::<false>(&got, &buf.pixels);
    }

    #[test]
    fn avx2_bgra_matches_lut_within_u8_tolerance() {
        if !has_avx2_fma() {
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
        if !has_avx2_fma() {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_avx2_fma::<false>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn avx2_bgra_u8_roundtrip_is_exact() {
        if !has_avx2_fma() {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_avx2_fma::<true>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn avx2_odd_count_tail_matches_sse4_path() {
        if !has_avx2_fma() {
            return;
        }

        // 3 pixels: 2-pixel AVX2 tail step + 1-pixel SSE4.1 tail step.
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
    fn avx2_quad_plus_tail_counts_match_lut() {
        if !has_avx2_fma() {
            return;
        }

        // Exercise every tail residue (0, 1, 2, 3) on top of at least one
        // 4-pixel main-loop iteration so the destination offset has to
        // compound across main + 2-pixel AVX2 tail + 1-pixel SSE4.1 tail.
        let mut pixels = Vec::with_capacity(11);
        for i in 0..11 {
            let t = i as f32 / 10.0;
            pixels.push([t, (t * 0.7).clamp(0.0, 1.0), 1.0 - t, t * t]);
        }

        for n in [4usize, 5, 6, 7, 8, 9, 10, 11] {
            let buf = buf_from(pixels[..n].to_vec());
            let rgba = unsafe { store_srgb8_f32_avx2_fma::<false>(&buf) };
            assert_within_u8_tolerance::<false>(&rgba, &buf.pixels);
            let bgra = unsafe { store_srgb8_f32_avx2_fma::<true>(&buf) };
            assert_within_u8_tolerance::<true>(&bgra, &buf.pixels);
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

    #[test]
    fn avx512_rgba_matches_lut_within_u8_tolerance() {
        if !has_avx512() {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_avx512::<false>(&buf) };
        assert_within_u8_tolerance::<false>(&got, &buf.pixels);
    }

    #[test]
    fn avx512_bgra_matches_lut_within_u8_tolerance() {
        if !has_avx512() {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_avx512::<true>(&buf) };
        assert_within_u8_tolerance::<true>(&got, &buf.pixels);
    }

    #[test]
    fn avx512_rgba_u8_roundtrip_is_exact() {
        if !has_avx512() {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_avx512::<false>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn avx512_bgra_u8_roundtrip_is_exact() {
        if !has_avx512() {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_avx512::<true>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn avx512_tail_matches_lut_within_u8_tolerance() {
        if !has_avx512() {
            return;
        }

        // 7 pixels exercises one 4-pixel main-loop iteration plus a
        // 3-pixel masked tail.
        let pixels = vec![
            [0.0, 0.1, 0.5, 1.0],
            [0.25, 0.75, 0.9, 0.5],
            [0.123, 0.456, 0.789, 0.321],
            [0.2, 0.4, 0.6, 0.8],
            [0.01, 0.99, 0.33, 0.77],
            [0.02, 0.98, 0.66, 0.44],
            [0.05, 0.95, 0.5, 0.5],
        ];
        let buf = buf_from(pixels);

        let rgba = unsafe { store_srgb8_f32_avx512::<false>(&buf) };
        assert_within_u8_tolerance::<false>(&rgba, &buf.pixels);

        let bgra = unsafe { store_srgb8_f32_avx512::<true>(&buf) };
        assert_within_u8_tolerance::<true>(&bgra, &buf.pixels);
    }

    #[test]
    fn avx512_multiple_main_plus_tail_matches_lut() {
        if !has_avx512() {
            return;
        }

        // 15 pixels exercises multiple 4-pixel main iterations plus a
        // 3-pixel masked tail, so the destination offset has to compound
        // correctly across them.
        let mut pixels = Vec::with_capacity(15);
        for i in 0..15 {
            let t = i as f32 / 14.0;
            pixels.push([t, (t * 0.7).clamp(0.0, 1.0), 1.0 - t, t * t]);
        }
        let buf = buf_from(pixels);

        let rgba = unsafe { store_srgb8_f32_avx512::<false>(&buf) };
        assert_within_u8_tolerance::<false>(&rgba, &buf.pixels);

        let bgra = unsafe { store_srgb8_f32_avx512::<true>(&buf) };
        assert_within_u8_tolerance::<true>(&bgra, &buf.pixels);
    }

    #[test]
    fn sse4_oetf_in_place_matches_exact() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let orig = oetf_test_pixels();
        let mut got = orig.clone();
        unsafe { srgb_oetf_in_place_f32_sse4_1(&mut got) };
        assert_encoded_close(&got, &orig);
    }

    #[test]
    fn avx2_oetf_in_place_matches_exact_across_tails() {
        if !has_avx2_fma() {
            return;
        }
        let base = oetf_test_pixels();
        // 0-9 covers the 2-pixel main loop with and without the 1-pixel
        // SSE4.1 tail.
        for n in [0usize, 1, 2, 3, 5, 8, 9] {
            let orig = base[..n].to_vec();
            let mut got = orig.clone();
            unsafe { srgb_oetf_in_place_f32_avx2_fma(&mut got) };
            assert_encoded_close(&got, &orig);
        }

        let mut got = base.clone();
        unsafe { srgb_oetf_in_place_f32_avx2_fma(&mut got) };
        assert_encoded_close(&got, &base);
    }

    #[test]
    fn avx512_oetf_in_place_matches_exact_across_tails() {
        if !has_avx512() {
            return;
        }
        let base = oetf_test_pixels();
        // 0-9 covers the 4-pixel main loop plus every 1-3 pixel masked tail.
        for n in [0usize, 1, 2, 3, 4, 5, 6, 7, 9] {
            let orig = base[..n].to_vec();
            let mut got = orig.clone();
            unsafe { srgb_oetf_in_place_f32_avx512(&mut got) };
            assert_encoded_close(&got, &orig);
        }

        let mut got = base.clone();
        unsafe { srgb_oetf_in_place_f32_avx512(&mut got) };
        assert_encoded_close(&got, &base);
    }

    #[test]
    fn sse4_oetf_in_place_u8_roundtrip_is_exact() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut pixels = u8_roundtrip_pixels();
        unsafe { srgb_oetf_in_place_f32_sse4_1(&mut pixels) };
        assert_in_place_roundtrips(&pixels);
    }

    #[test]
    fn avx2_oetf_in_place_u8_roundtrip_is_exact() {
        if !has_avx2_fma() {
            return;
        }
        let mut pixels = u8_roundtrip_pixels();
        unsafe { srgb_oetf_in_place_f32_avx2_fma(&mut pixels) };
        assert_in_place_roundtrips(&pixels);
    }

    #[test]
    fn avx512_oetf_in_place_u8_roundtrip_is_exact() {
        if !has_avx512() {
            return;
        }
        let mut pixels = u8_roundtrip_pixels();
        unsafe { srgb_oetf_in_place_f32_avx512(&mut pixels) };
        assert_in_place_roundtrips(&pixels);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests {
    use super::oetf_in_place_tests::{
        assert_encoded_close, assert_in_place_roundtrips, assert_roundtrips, oetf_test_pixels,
        u8_roundtrip_pixels,
    };
    use super::*;

    fn has_neon() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

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
    fn neon_rgba_matches_lut_within_u8_tolerance() {
        if !has_neon() {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_neon::<false>(&buf) };
        assert_within_u8_tolerance::<false>(&got, &buf.pixels);
    }

    #[test]
    fn neon_bgra_matches_lut_within_u8_tolerance() {
        if !has_neon() {
            return;
        }
        let buf = buf_from(fine_grid_pixels());
        let got = unsafe { store_srgb8_f32_neon::<true>(&buf) };
        assert_within_u8_tolerance::<true>(&got, &buf.pixels);
    }

    #[test]
    fn neon_rgba_u8_roundtrip_is_exact() {
        if !has_neon() {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_neon::<false>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn neon_bgra_u8_roundtrip_is_exact() {
        if !has_neon() {
            return;
        }
        let buf = buf_from(u8_roundtrip_pixels());
        let got = unsafe { store_srgb8_f32_neon::<true>(&buf) };
        assert_roundtrips(&got);
    }

    #[test]
    fn neon_tail_matches_lut_within_u8_tolerance() {
        if !has_neon() {
            return;
        }

        let pixels = vec![
            [0.0, 0.1, 0.5, 1.0],
            [0.25, 0.75, 0.9, 0.5],
            [0.123, 0.456, 0.789, 0.321],
            [0.2, 0.4, 0.6, 0.8],
            [0.01, 0.99, 0.33, 0.77],
            [0.02, 0.98, 0.66, 0.44],
            [0.05, 0.95, 0.5, 0.5],
        ];
        let buf = buf_from(pixels);

        let rgba = unsafe { store_srgb8_f32_neon::<false>(&buf) };
        assert_within_u8_tolerance::<false>(&rgba, &buf.pixels);

        let bgra = unsafe { store_srgb8_f32_neon::<true>(&buf) };
        assert_within_u8_tolerance::<true>(&bgra, &buf.pixels);
    }

    #[test]
    fn neon_bgra_swaps_r_and_b_bytes() {
        if !has_neon() {
            return;
        }

        let buf = buf_from(vec![[1.0, 0.0, 0.25, 0.5]]);
        let rgba = unsafe { store_srgb8_f32_neon::<false>(&buf) };
        let bgra = unsafe { store_srgb8_f32_neon::<true>(&buf) };

        assert_eq!(bgra[0], rgba[2]);
        assert_eq!(bgra[1], rgba[1]);
        assert_eq!(bgra[2], rgba[0]);
        assert_eq!(bgra[3], rgba[3]);
    }

    #[test]
    fn neon_oetf_in_place_matches_exact() {
        if !has_neon() {
            return;
        }
        let orig = oetf_test_pixels();
        let mut got = orig.clone();
        unsafe { srgb_oetf_in_place_f32_neon(&mut got) };
        assert_encoded_close(&got, &orig);
    }

    #[test]
    fn neon_oetf_in_place_u8_roundtrip_is_exact() {
        if !has_neon() {
            return;
        }
        let mut pixels = u8_roundtrip_pixels();
        unsafe { srgb_oetf_in_place_f32_neon(&mut pixels) };
        assert_in_place_roundtrips(&pixels);
    }
}
