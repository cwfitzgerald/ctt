//! sRGB-encoded u8 loaders and the in-place EOTF post-pass.
//!
//! Shared state:
//! * [`EOTF_LUT`] — 256-entry scalar lookup table (exact sRGB EOTF).
//! * [`EOTF_INTERP_LUT`] — 4097-entry scalar lookup table over `[0, 1]` for
//!   the linearly-interpolated fast scalar path.
//! * [`SRGB_MINIMAX_A`]/[`SRGB_MINIMAX_B`]/[`SRGB_MINIMAX_C`] — piecewise
//!   minimax approximation used by the SIMD fast paths.
//!
//! The per-ISA `srgb_eotf_*` helpers evaluate the piecewise EOTF curve on a
//! full vector register; the u8 load kernels and the f32 in-place post-pass
//! ([`srgb_eotf_in_place_f32`]) both build on them.

use std::sync::LazyLock;

use crate::error::Result;
#[cfg(target_arch = "x86_64")]
use crate::processing::x86::has_avx512;
use crate::surface::Surface;

use super::{Buffer, read_pixels_f32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// sRGB EOTF lookup table — maps every u8 value (0–255) to its linear f32 equivalent.
static EOTF_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut table = [0.0f32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *entry = srgb_eotf(c);
    }
    table
});

/// Apply the sRGB EOTF (sRGB-encoded → linear) to a single value.
fn srgb_eotf(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

const EOTF_INTERP_LUT_SIZE: usize = 4096;

/// sRGB EOTF lookup table — 4097 entries over [0, 1] for linear interpolation.
static EOTF_INTERP_LUT: LazyLock<[f32; EOTF_INTERP_LUT_SIZE + 1]> = LazyLock::new(|| {
    let mut table = [0.0f32; EOTF_INTERP_LUT_SIZE + 1];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = i as f32 / EOTF_INTERP_LUT_SIZE as f32;
        *entry = srgb_eotf(c);
    }
    table
});

#[inline(always)]
fn srgb_eotf_fast(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    let scaled = c * EOTF_INTERP_LUT_SIZE as f32;
    let idx = scaled as usize;
    if idx >= EOTF_INTERP_LUT_SIZE {
        return EOTF_INTERP_LUT[EOTF_INTERP_LUT_SIZE];
    }
    let frac = scaled - idx as f32;
    EOTF_INTERP_LUT[idx] + frac * (EOTF_INTERP_LUT[idx + 1] - EOTF_INTERP_LUT[idx])
}

// Piecewise minimax approximation of the sRGB EOTF. The curve branch fits
// `((x + 0.055) / 1.055)^2.4` with max abs error ≈ 1.28e-4 on the u8 byte
// lattice (`x = byte / 255`) — well inside ±0.5/255, so u8 round-trip stays
// bit-exact versus the LUT path — and ≈ 1.5e-4 over the continuous `[0, 1]`
// domain the f32 in-place post-pass feeds it. See `srgb-opt.py`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SRGB_MINIMAX_A: f32 = -0.983_177_1;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SRGB_MINIMAX_B: f32 = -0.083_670_19;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SRGB_MINIMAX_C: f32 = -0.121_285_7;

pub fn load_srgb8_f32(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_srgb8_f32");

    #[cfg(target_arch = "x86_64")]
    {
        if channels == 4 && has_avx512() {
            // SAFETY: runtime check confirms avx512f + vl + bw are available.
            return unsafe { load_srgb8_rgba_f32_avx512(surface) };
        }
        if channels == 4 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime check confirms avx2 + fma are available.
            return unsafe { load_srgb8_rgba_f32_avx2_fma(surface) };
        }
        if channels == 4 && is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { load_srgb8_rgba_f32_sse4_1(surface) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if channels == 4 && std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime check confirms NEON is available.
            return unsafe { load_srgb8_rgba_f32_neon(surface) };
        }
    }

    load_srgb8_f32_serial(surface, channels)
}

/// Serial LUT path for sRGB8 loads.
///
/// **Not part of the public API.** Exposed so benchmarks can compare the
/// scalar implementation directly against each runtime-selectable SIMD mode.
#[doc(hidden)]
pub fn load_srgb8_f32_serial(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_srgb8_f32_serial");

    let lut = &*EOTF_LUT;
    read_pixels_f32(surface, channels, 1, |bytes, lanes| {
        // RGB lanes through the sRGB EOTF, alpha linear.
        for (c, (lane, &byte)) in lanes.iter_mut().zip(bytes).enumerate() {
            *lane = if c < 3 {
                lut[byte as usize]
            } else {
                byte as f32 / 255.0
            };
        }
    })
}

pub fn load_bgra8_srgb_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_bgra8_srgb_f32");
    let lut = &*EOTF_LUT;
    read_pixels_f32(surface, 4, 1, |bytes, lanes| {
        let &[b, g, r, a] = <&[u8; 4]>::try_from(bytes).expect("4-byte pixel");
        lanes[0] = lut[r as usize];
        lanes[1] = lut[g as usize];
        lanes[2] = lut[b as usize];
        lanes[3] = a as f32 / 255.0;
    })
}

pub fn load_bgr8_srgb_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_bgr8_srgb_f32");
    let lut = &*EOTF_LUT;
    read_pixels_f32(surface, 3, 1, |bytes, lanes| {
        let &[b, g, r] = <&[u8; 3]>::try_from(bytes).expect("3-byte pixel");
        lanes[0] = lut[r as usize];
        lanes[1] = lut[g as usize];
        lanes[2] = lut[b as usize];
    })
}

/// Piecewise sRGB EOTF over all four lanes of `x`.
///
/// `x` must already be in (or clamped to) `[0, 1]`. Every lane goes through
/// the curve — callers with an alpha lane blend it back themselves.
///
/// Piecewise form:
/// * `x <= 0.04045`: `x / 12.92` (linear segment of the sRGB spec).
/// * `x > 0.04045`: `(a·x + b)^2 * (c·x + sqrt(x))` — minimax fit of
///   `((x + 0.055) / 1.055)^2.4` (see `SRGB_MINIMAX_*`). Max abs error
///   ≈ 1.28e-4 on the u8 byte lattice and ≈ 1.5e-4 over the continuous
///   domain.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn srgb_eotf_sse4_1(x: __m128) -> __m128 {
    let t = _mm_sqrt_ps(x);
    let u = _mm_add_ps(
        _mm_mul_ps(x, _mm_set1_ps(SRGB_MINIMAX_A)),
        _mm_set1_ps(SRGB_MINIMAX_B),
    );
    let v = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(SRGB_MINIMAX_C)), t);
    let curve = _mm_mul_ps(_mm_mul_ps(u, u), v);
    let linear = _mm_mul_ps(x, _mm_set1_ps(1.0 / 12.92));

    // Select the linear segment for x <= threshold, curve otherwise.
    let use_linear = _mm_cmple_ps(x, _mm_set1_ps(0.040_45));
    _mm_blendv_ps(curve, linear, use_linear)
}

/// Decode one 4-byte sRGB RGBA pixel into `[R, G, B, A]` linear f32 lanes,
/// shared between the SSE4.1 main loop and the AVX2 fast path's tail.
///
/// Color lanes go through [`srgb_eotf_sse4_1`] (see it for the piecewise
/// form and accuracy guarantees); the alpha lane bypasses the EOTF and is
/// read as a straight unorm.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
/// * `bytes_ptr` must be valid for a 4-byte read.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn decode_srgb_pixel_sse4_1(bytes_ptr: *const u8) -> __m128 {
    // SAFETY: caller guarantees 4 valid bytes at bytes_ptr.
    let raw = unsafe { bytes_ptr.cast::<u32>().read_unaligned() };
    let packed = _mm_cvtsi32_si128(raw as i32);
    let as_i32 = _mm_cvtepu8_epi32(packed);
    let as_f32 = _mm_cvtepi32_ps(as_i32);

    let x_norm = _mm_mul_ps(as_f32, _mm_set1_ps(1.0 / 255.0));
    // SAFETY: requires sse4.1, matched by the enclosing `target_feature`.
    let rgb = unsafe { srgb_eotf_sse4_1(x_norm) };
    // Lane 3 is the alpha channel in the [R,G,B,A] layout.
    let alpha_lane_mask = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, -1));
    _mm_blendv_ps(rgb, x_norm, alpha_lane_mask)
}

/// SSE4.1 path for `R8G8B8A8_SRGB` (and equivalent 4-channel sRGB layouts).
///
/// Processes one pixel (4 bytes → 4 f32) per iteration via
/// [`decode_srgb_pixel_sse4_1`]. See that helper for the piecewise form and
/// accuracy guarantees.
///
/// **Not part of the public API.** Exposed as `pub` + `doc(hidden)` only so
/// `benches/` (a separate crate) can measure this kernel directly without
/// going through dispatch. No stability guarantees — may be renamed,
/// removed, or have its signature changed across patch releases. Real
/// callers should use [`load_srgb8_f32`], which picks the best available
/// kernel at runtime.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn load_srgb8_rgba_f32_sse4_1(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_srgb8_rgba_f32_sse4_1");
    super::validate_surface(surface, 4)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * 4;
    let total_pixels = w * h;

    let mut pixels: Vec<[f32; 4]> = Vec::with_capacity(total_pixels);
    let out_base = pixels.as_mut_ptr() as *mut f32;

    // SAFETY: every intrinsic and pointer op below runs with sse4.1 enabled
    // (target_feature on the enclosing fn) and within the capacity reserved
    // for `pixels`; validate_surface has already bounded the input slice.
    unsafe {
        let mut out_f32 = 0usize;

        for row_region in surface.data.chunks(stride).take(h) {
            let row = &row_region[..row_bytes];
            let mut x = 0usize;

            // 1 pixel (4 input bytes, 4 output f32s) per iteration.
            while x + 4 <= row_bytes {
                let result = decode_srgb_pixel_sse4_1(row.as_ptr().add(x));
                _mm_storeu_ps(out_base.add(out_f32), result);
                out_f32 += 4;
                x += 4;
            }
        }

        debug_assert_eq!(out_f32, total_pixels * 4);
        pixels.set_len(total_pixels);
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

/// AVX2+FMA counterpart of [`srgb_eotf_sse4_1`] — the same piecewise curve
/// widened to 8 lanes.
///
/// # Safety
/// * AVX2 and FMA must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn srgb_eotf_avx2(x: __m256) -> __m256 {
    let t = _mm256_sqrt_ps(x);
    let u = _mm256_fmadd_ps(
        x,
        _mm256_set1_ps(SRGB_MINIMAX_A),
        _mm256_set1_ps(SRGB_MINIMAX_B),
    );
    let v = _mm256_fmadd_ps(x, _mm256_set1_ps(SRGB_MINIMAX_C), t);
    let curve = _mm256_mul_ps(_mm256_mul_ps(u, u), v);
    let linear = _mm256_mul_ps(x, _mm256_set1_ps(1.0 / 12.92));

    let use_linear = _mm256_cmp_ps::<_CMP_LE_OQ>(x, _mm256_set1_ps(0.040_45));
    _mm256_blendv_ps(curve, linear, use_linear)
}

/// AVX2 + FMA path for `R8G8B8A8_SRGB` (and equivalent 4-channel sRGB layouts).
///
/// Processes two pixels (8 bytes → 8 f32) per iteration via
/// [`srgb_eotf_avx2`]. Any 1-pixel remainder is handled by the SSE4.1
/// [`decode_srgb_pixel_sse4_1`] helper, so the tail stays vectorized and
/// consistent with the SSE4.1 fast path.
///
/// **Not part of the public API.** See
/// [`load_srgb8_rgba_f32_sse4_1`] for the rationale; use
/// [`load_srgb8_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn load_srgb8_rgba_f32_avx2_fma(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_srgb8_rgba_f32_avx2_fma");
    super::validate_surface(surface, 4)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * 4;
    let total_pixels = w * h;

    let mut pixels: Vec<[f32; 4]> = Vec::with_capacity(total_pixels);
    let out_base = pixels.as_mut_ptr() as *mut f32;

    // SAFETY: every intrinsic and pointer op below runs with avx2+fma enabled
    // (target_feature on the enclosing fn) and within the capacity reserved
    // for `pixels`; validate_surface has already bounded the input slice.
    unsafe {
        let inv_255 = _mm256_set1_ps(1.0 / 255.0);
        // Lanes 3 and 7 are the alpha channel in the [R,G,B,A,R,G,B,A] layout.
        let alpha_lane_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, -1));

        let mut out_f32 = 0usize;

        for row_region in surface.data.chunks(stride).take(h) {
            let row = &row_region[..row_bytes];
            let mut x = 0usize;

            // 2 pixels (8 input bytes, 8 output f32s) per iteration.
            while x + 8 <= row_bytes {
                let bytes = _mm_loadl_epi64(row.as_ptr().add(x) as *const __m128i);
                let as_i32 = _mm256_cvtepu8_epi32(bytes);
                let as_f32 = _mm256_cvtepi32_ps(as_i32);

                let x_norm = _mm256_mul_ps(as_f32, inv_255);
                let rgb = srgb_eotf_avx2(x_norm);
                let result = _mm256_blendv_ps(rgb, x_norm, alpha_lane_mask);

                _mm256_storeu_ps(out_base.add(out_f32), result);
                out_f32 += 8;
                x += 8;
            }

            // 1-pixel tail: at most one 4-byte pixel left. Hand it to the
            // SSE4.1 helper so the tail stays vectorized and algorithmically
            // consistent with the SSE4.1 fast path.
            if x < row_bytes {
                let result = decode_srgb_pixel_sse4_1(row.as_ptr().add(x));
                _mm_storeu_ps(out_base.add(out_f32), result);
                out_f32 += 4;
            }
        }

        debug_assert_eq!(out_f32, total_pixels * 4);
        pixels.set_len(total_pixels);
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

/// Lanes 3, 7, 11, 15 — alpha in the four-pixel `[R,G,B,A] × 4` layout of a
/// `__m512`.
#[cfg(target_arch = "x86_64")]
const ALPHA_LANES_512: __mmask16 = 0b1000_1000_1000_1000;

/// AVX-512 counterpart of [`srgb_eotf_sse4_1`] — the same piecewise curve
/// widened to 16 lanes.
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn srgb_eotf_avx512(x: __m512) -> __m512 {
    let t = _mm512_sqrt_ps(x);
    let u = _mm512_fmadd_ps(
        x,
        _mm512_set1_ps(SRGB_MINIMAX_A),
        _mm512_set1_ps(SRGB_MINIMAX_B),
    );
    let v = _mm512_fmadd_ps(x, _mm512_set1_ps(SRGB_MINIMAX_C), t);
    let curve = _mm512_mul_ps(_mm512_mul_ps(u, u), v);
    let linear = _mm512_mul_ps(x, _mm512_set1_ps(1.0 / 12.92));

    let use_linear = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(x, _mm512_set1_ps(0.040_45));
    _mm512_mask_blend_ps(use_linear, curve, linear)
}

/// Decode one aligned block of 16 consecutive sRGB bytes (= 4 RGBA pixels)
/// into 16 linear-f32 lanes, shared between the AVX-512 main loop and its
/// masked 1-3 pixel tail.
///
/// Only the lower `n*4` bytes of `bytes` need be valid; higher bytes are
/// treated as zero (produces zero-decoded lanes, which the caller filters
/// out via a masked store).
///
/// Color lanes go through [`srgb_eotf_avx512`] (see [`srgb_eotf_sse4_1`]
/// for the piecewise form and accuracy guarantees); the alpha lanes bypass
/// the EOTF and are read as straight unorms.
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn decode_srgb_pixels_avx512(bytes: __m128i) -> __m512 {
    let as_i32 = _mm512_cvtepu8_epi32(bytes);
    let as_f32 = _mm512_cvtepi32_ps(as_i32);

    let x_norm = _mm512_mul_ps(as_f32, _mm512_set1_ps(1.0 / 255.0));
    // SAFETY: requires avx512f, matched by the enclosing `target_feature`.
    let rgb = unsafe { srgb_eotf_avx512(x_norm) };
    _mm512_mask_blend_ps(ALPHA_LANES_512, rgb, x_norm)
}

/// AVX-512 path for `R8G8B8A8_SRGB` (and equivalent 4-channel sRGB layouts).
///
/// Processes four pixels (16 bytes → 16 f32s) per iteration via
/// [`decode_srgb_pixels_avx512`]. A 1-3 pixel masked AVX-512 tail handles
/// any remainder without dropping to a narrower width.
///
/// **Not part of the public API.** See
/// [`load_srgb8_rgba_f32_sse4_1`] for the rationale; use
/// [`load_srgb8_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn load_srgb8_rgba_f32_avx512(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_srgb8_rgba_f32_avx512");
    super::validate_surface(surface, 4)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * 4;
    let total_pixels = w * h;

    let mut pixels: Vec<[f32; 4]> = Vec::with_capacity(total_pixels);
    let out_base = pixels.as_mut_ptr() as *mut f32;

    // SAFETY: every intrinsic and pointer op below runs with avx512f+bw+vl
    // enabled (target_feature on the enclosing fn) and within the capacity
    // reserved for `pixels`; validate_surface has already bounded the input
    // slice.
    unsafe {
        let mut out_f32 = 0usize;

        for row_region in surface.data.chunks(stride).take(h) {
            let row = &row_region[..row_bytes];
            let mut x = 0usize;

            // 4 pixels (16 input bytes, 16 output f32s) per iteration.
            while x + 16 <= row_bytes {
                let bytes = _mm_loadu_si128(row.as_ptr().add(x) as *const __m128i);
                let result = decode_srgb_pixels_avx512(bytes);
                _mm512_storeu_ps(out_base.add(out_f32), result);
                out_f32 += 16;
                x += 16;
            }

            // 1-3 pixel tail in a single masked AVX-512 iteration. The mask
            // suppresses both the byte load and the f32 store for lanes
            // beyond the remaining pixel count, so the intervening math can
            // run unmasked on garbage without touching memory.
            if x < row_bytes {
                let tail_pixels = (row_bytes - x) / 4;
                let mask: __mmask16 = (1u16 << (tail_pixels * 4)) - 1;

                let bytes = _mm_maskz_loadu_epi8(mask, row.as_ptr().add(x) as *const i8);
                let result = decode_srgb_pixels_avx512(bytes);
                _mm512_mask_storeu_ps(out_base.add(out_f32), mask, result);
                out_f32 += tail_pixels * 4;
            }
        }

        debug_assert_eq!(out_f32, total_pixels * 4);
        pixels.set_len(total_pixels);
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

/// NEON counterpart of [`srgb_eotf_sse4_1`] — the same piecewise curve
/// mapped to AArch64 NEON.
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn srgb_eotf_neon(x: float32x4_t) -> float32x4_t {
    let t = vsqrtq_f32(x);
    let u = vaddq_f32(
        vmulq_f32(x, vdupq_n_f32(SRGB_MINIMAX_A)),
        vdupq_n_f32(SRGB_MINIMAX_B),
    );
    let v = vaddq_f32(vmulq_f32(x, vdupq_n_f32(SRGB_MINIMAX_C)), t);
    let curve = vmulq_f32(vmulq_f32(u, u), v);
    let linear = vmulq_f32(x, vdupq_n_f32(1.0 / 12.92));

    let use_linear = vcleq_f32(x, vdupq_n_f32(0.040_45));
    vbslq_f32(use_linear, linear, curve)
}

/// Decode one 4-lane `[R, G, B, A]` u8 vector into linear f32 lanes.
///
/// Color lanes go through [`srgb_eotf_neon`] (see [`srgb_eotf_sse4_1`] for
/// the piecewise form and accuracy guarantees); the alpha lane bypasses the
/// EOTF and is read as a straight unorm.
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_srgb_lanes_neon(as_u32: uint32x4_t) -> float32x4_t {
    let as_f32 = vcvtq_f32_u32(as_u32);
    let x_norm = vmulq_f32(as_f32, vdupq_n_f32(1.0 / 255.0));
    // SAFETY: requires NEON, matched by the enclosing `target_feature`.
    let rgb = unsafe { srgb_eotf_neon(x_norm) };
    let alpha_lane_mask = vsetq_lane_u32::<3>(u32::MAX, vdupq_n_u32(0));
    vbslq_f32(alpha_lane_mask, x_norm, rgb)
}

/// Decode one 4-byte sRGB RGBA pixel into `[R, G, B, A]` linear f32 lanes.
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
/// * `bytes_ptr` must be valid for a 4-byte read.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_srgb_pixel_neon(bytes_ptr: *const u8) -> float32x4_t {
    let mut lanes = vdupq_n_u32(0);
    // SAFETY: caller guarantees 4 valid bytes at bytes_ptr.
    unsafe {
        lanes = vsetq_lane_u32::<0>(*bytes_ptr.add(0) as u32, lanes);
        lanes = vsetq_lane_u32::<1>(*bytes_ptr.add(1) as u32, lanes);
        lanes = vsetq_lane_u32::<2>(*bytes_ptr.add(2) as u32, lanes);
        lanes = vsetq_lane_u32::<3>(*bytes_ptr.add(3) as u32, lanes);
    }
    // SAFETY: same NEON target feature as this helper.
    unsafe { decode_srgb_lanes_neon(lanes) }
}

/// NEON path for `R8G8B8A8_SRGB` (and equivalent 4-channel sRGB layouts).
///
/// Processes four pixels (16 bytes → 16 f32s) per iteration. A 1-3 pixel
/// tail is handled by the same NEON per-pixel helper, so the whole path stays
/// vectorized on M1/aarch64.
///
/// **Not part of the public API.** See
/// [`load_srgb8_rgba_f32_sse4_1`] for the rationale; use
/// [`load_srgb8_f32`] for the stable, runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn load_srgb8_rgba_f32_neon(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_srgb8_rgba_f32_neon");
    super::validate_surface(surface, 4)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * 4;
    let total_pixels = w * h;

    let mut pixels: Vec<[f32; 4]> = Vec::with_capacity(total_pixels);
    let out_base = pixels.as_mut_ptr() as *mut f32;

    // SAFETY: every intrinsic and pointer op below runs with NEON enabled
    // (target_feature on the enclosing fn) and within the capacity reserved
    // for `pixels`; validate_surface has already bounded the input slice.
    unsafe {
        let mut out_f32 = 0usize;

        for row_region in surface.data.chunks(stride).take(h) {
            let row = &row_region[..row_bytes];
            let mut x = 0usize;

            // 4 pixels (16 input bytes, 16 output f32s) per iteration.
            while x + 16 <= row_bytes {
                let bytes = vld1q_u8(row.as_ptr().add(x));
                let lo16 = vmovl_u8(vget_low_u8(bytes));
                let hi16 = vmovl_u8(vget_high_u8(bytes));

                let rgba0 = vmovl_u16(vget_low_u16(lo16));
                let rgba1 = vmovl_u16(vget_high_u16(lo16));
                let rgba2 = vmovl_u16(vget_low_u16(hi16));
                let rgba3 = vmovl_u16(vget_high_u16(hi16));

                vst1q_f32(out_base.add(out_f32), decode_srgb_lanes_neon(rgba0));
                vst1q_f32(out_base.add(out_f32 + 4), decode_srgb_lanes_neon(rgba1));
                vst1q_f32(out_base.add(out_f32 + 8), decode_srgb_lanes_neon(rgba2));
                vst1q_f32(out_base.add(out_f32 + 12), decode_srgb_lanes_neon(rgba3));

                out_f32 += 16;
                x += 16;
            }

            while x < row_bytes {
                let result = decode_srgb_pixel_neon(row.as_ptr().add(x));
                vst1q_f32(out_base.add(out_f32), result);
                out_f32 += 4;
                x += 4;
            }
        }

        debug_assert_eq!(out_f32, total_pixels * 4);
        pixels.set_len(total_pixels);
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

/// Apply the sRGB EOTF in place to the RGB lanes of every pixel, leaving the
/// alpha lane untouched.
///
/// Post-pass for loads from FormatKinds with no sRGB kernel variant (16+ bit
/// formats) whose source color_space is nonetheless `Srgb`: the buffer is
/// read through the format's linear kernel, then EOTF-decoded here.
///
/// RGB lanes are clamped to `[0, 1]` before decoding; alpha is preserved
/// bit-exactly. The SIMD paths use the same piecewise minimax curve as the
/// u8 load kernels (max abs error ≈ 1.5e-4, bit-exact inverted by the
/// store-side OETF approximation); the serial path uses the interpolated
/// `EOTF_INTERP_LUT`.
pub fn srgb_eotf_in_place_f32(pixels: &mut [[f32; 4]]) {
    profiling::scope!("srgb_eotf_in_place_f32");

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: runtime check confirms avx512f + vl + bw are available.
            return unsafe { srgb_eotf_in_place_f32_avx512(pixels) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime check confirms avx2 + fma are available.
            return unsafe { srgb_eotf_in_place_f32_avx2_fma(pixels) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: runtime check confirms sse4.1 is available.
            return unsafe { srgb_eotf_in_place_f32_sse4_1(pixels) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime check confirms NEON is available.
            return unsafe { srgb_eotf_in_place_f32_neon(pixels) };
        }
    }

    srgb_eotf_in_place_f32_serial(pixels)
}

/// Serial LUT path for the in-place EOTF post-pass.
///
/// **Not part of the public API.** Exposed so benchmarks can compare the
/// scalar implementation directly against each runtime-selectable SIMD mode.
#[doc(hidden)]
pub fn srgb_eotf_in_place_f32_serial(pixels: &mut [[f32; 4]]) {
    profiling::scope!("srgb_eotf_in_place_f32_serial");
    for p in pixels {
        p[0] = srgb_eotf_fast(p[0]);
        p[1] = srgb_eotf_fast(p[1]);
        p[2] = srgb_eotf_fast(p[2]);
    }
}

/// EOTF-decode one pixel's four f32 lanes in place at `p`, preserving the
/// original (unclamped) alpha lane bit-exactly. Shared between the SSE4.1
/// main loop and the AVX2 path's odd-pixel tail.
///
/// # Safety
/// * The SSE4.1 feature must be available (enforced by `target_feature`).
/// * `p` must be valid for 4 f32 reads and writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn eotf_pixel_in_place_sse4_1(p: *mut f32) {
    // SAFETY: `p` spans 4 f32 lanes per the caller; sse4.1 is matched by the
    // enclosing `target_feature`.
    unsafe {
        let lanes = _mm_loadu_ps(p);
        let x = _mm_max_ps(_mm_min_ps(lanes, _mm_set1_ps(1.0)), _mm_setzero_ps());
        let rgb = srgb_eotf_sse4_1(x);
        // Bit 3 selects the original alpha lane.
        let out = _mm_blend_ps::<0b1000>(rgb, lanes);
        _mm_storeu_ps(p, out);
    }
}

/// SSE4.1 path for the in-place EOTF post-pass: one pixel (4 f32) per
/// iteration via [`eotf_pixel_in_place_sse4_1`].
///
/// **Not part of the public API.** See [`load_srgb8_rgba_f32_sse4_1`] for
/// the rationale; use [`srgb_eotf_in_place_f32`] for the stable,
/// runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn srgb_eotf_in_place_f32_sse4_1(pixels: &mut [[f32; 4]]) {
    profiling::scope!("srgb_eotf_in_place_f32_sse4_1");

    let total_pixels = pixels.len();
    let base = pixels.as_mut_ptr() as *mut f32;

    // SAFETY: `base` spans `total_pixels * 4` f32 lanes; sse4.1 is matched by
    // the enclosing `target_feature`.
    unsafe {
        for i in 0..total_pixels {
            eotf_pixel_in_place_sse4_1(base.add(i * 4));
        }
    }
}

/// AVX2 + FMA path for the in-place EOTF post-pass: two pixels (8 f32) per
/// iteration, with an optional 1-pixel SSE4.1 tail.
///
/// **Not part of the public API.** See [`load_srgb8_rgba_f32_sse4_1`] for
/// the rationale; use [`srgb_eotf_in_place_f32`] for the stable,
/// runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn srgb_eotf_in_place_f32_avx2_fma(pixels: &mut [[f32; 4]]) {
    profiling::scope!("srgb_eotf_in_place_f32_avx2_fma");

    let total_pixels = pixels.len();
    let base = pixels.as_mut_ptr() as *mut f32;

    let pair_count = total_pixels / 2;

    // SAFETY: `base` spans `total_pixels * 4` f32 lanes; avx2+fma (and the
    // implied sse4.1 for the tail helper) are matched by the enclosing
    // `target_feature`.
    unsafe {
        for i in 0..pair_count {
            let p = base.add(i * 8);
            let lanes = _mm256_loadu_ps(p);
            let x = _mm256_max_ps(
                _mm256_min_ps(lanes, _mm256_set1_ps(1.0)),
                _mm256_setzero_ps(),
            );
            let rgb = srgb_eotf_avx2(x);
            // Bits 3 and 7 select the original alpha lanes.
            let out = _mm256_blend_ps::<0b1000_1000>(rgb, lanes);
            _mm256_storeu_ps(p, out);
        }
        if total_pixels % 2 == 1 {
            eotf_pixel_in_place_sse4_1(base.add(pair_count * 8));
        }
    }
}

/// AVX-512 path for the in-place EOTF post-pass: four pixels (16 f32) per
/// iteration, with a 1-3 pixel masked tail.
///
/// **Not part of the public API.** See [`load_srgb8_rgba_f32_sse4_1`] for
/// the rationale; use [`srgb_eotf_in_place_f32`] for the stable,
/// runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub unsafe fn srgb_eotf_in_place_f32_avx512(pixels: &mut [[f32; 4]]) {
    profiling::scope!("srgb_eotf_in_place_f32_avx512");

    let total_pixels = pixels.len();
    let base = pixels.as_mut_ptr() as *mut f32;

    let quad_count = total_pixels / 4;
    let tail_pixels = total_pixels % 4;

    // SAFETY: `base` spans `total_pixels * 4` f32 lanes; avx512f+vl+bw are
    // matched by the enclosing `target_feature`.
    unsafe {
        for i in 0..quad_count {
            let p = base.add(i * 16);
            let lanes = _mm512_loadu_ps(p);
            let x = _mm512_max_ps(
                _mm512_min_ps(lanes, _mm512_set1_ps(1.0)),
                _mm512_setzero_ps(),
            );
            let rgb = srgb_eotf_avx512(x);
            let out = _mm512_mask_blend_ps(ALPHA_LANES_512, rgb, lanes);
            _mm512_storeu_ps(p, out);
        }

        // 1-3 pixel tail in a single masked AVX-512 iteration. The mask
        // suppresses both the load and the store for lanes beyond the
        // remaining pixel count, so the intervening math can run unmasked
        // on zeros without touching memory.
        if tail_pixels > 0 {
            let p = base.add(quad_count * 16);
            let mask: __mmask16 = (1u16 << (tail_pixels * 4)) - 1;

            let lanes = _mm512_maskz_loadu_ps(mask, p);
            let x = _mm512_max_ps(
                _mm512_min_ps(lanes, _mm512_set1_ps(1.0)),
                _mm512_setzero_ps(),
            );
            let rgb = srgb_eotf_avx512(x);
            let out = _mm512_mask_blend_ps(ALPHA_LANES_512, rgb, lanes);
            _mm512_mask_storeu_ps(p, mask, out);
        }
    }
}

/// NEON path for the in-place EOTF post-pass: one pixel (4 f32) per
/// iteration via [`srgb_eotf_neon`], with the original (unclamped) alpha
/// lane blended back so it is preserved bit-exactly.
///
/// **Not part of the public API.** See [`load_srgb8_rgba_f32_sse4_1`] for
/// the rationale; use [`srgb_eotf_in_place_f32`] for the stable,
/// runtime-dispatched entry point.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn srgb_eotf_in_place_f32_neon(pixels: &mut [[f32; 4]]) {
    profiling::scope!("srgb_eotf_in_place_f32_neon");

    let total_pixels = pixels.len();
    let base = pixels.as_mut_ptr() as *mut f32;

    // SAFETY: `base` spans `total_pixels * 4` f32 lanes; NEON is matched by
    // the enclosing `target_feature`.
    unsafe {
        for i in 0..total_pixels {
            let p = base.add(i * 4);
            let lanes = vld1q_f32(p);
            let x = vmaxq_f32(vminq_f32(lanes, vdupq_n_f32(1.0)), vdupq_n_f32(0.0));
            let rgb = srgb_eotf_neon(x);
            let alpha_lane_mask = vsetq_lane_u32::<3>(u32::MAX, vdupq_n_u32(0));
            let out = vbslq_f32(alpha_lane_mask, lanes, rgb);
            vst1q_f32(p, out);
        }
    }
}

#[cfg(test)]
mod eotf_in_place_tests {
    use super::*;

    /// Exact clamped EOTF the in-place post-pass approximates.
    pub(super) fn srgb_eotf_exact(c: f32) -> f32 {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// RGB lanes within the documented worst-case error of the exact EOTF;
    /// alpha lanes preserved bit-exactly.
    pub(super) fn assert_decoded_close(got: &[[f32; 4]], orig: &[[f32; 4]]) {
        assert_eq!(got.len(), orig.len());
        for (i, (g, o)) in got.iter().zip(orig).enumerate() {
            for c in 0..3 {
                let want = srgb_eotf_exact(o[c]);
                assert!(
                    (g[c] - want).abs() <= 5.0e-4,
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
    pub(super) fn eotf_test_pixels() -> Vec<[f32; 4]> {
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
        let orig = eotf_test_pixels();
        let mut got = orig.clone();
        srgb_eotf_in_place_f32_serial(&mut got);
        assert_decoded_close(&got, &orig);
    }

    /// The dispatched entry point must agree with the exact EOTF for every
    /// pixel count that exercises the SIMD kernels' main loops and tails.
    #[test]
    fn dispatch_matches_exact_across_tail_lengths() {
        let base = eotf_test_pixels();
        for n in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16] {
            let orig = base[..n].to_vec();
            let mut got = orig.clone();
            srgb_eotf_in_place_f32(&mut got);
            assert_decoded_close(&got, &orig);
        }

        let mut got = base.clone();
        srgb_eotf_in_place_f32(&mut got);
        assert_decoded_close(&got, &base);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod simd_tests {
    use super::eotf_in_place_tests::{assert_decoded_close, eotf_test_pixels};
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::{ColorSpace, Surface};

    fn srgb_surface(data: Vec<u8>, width: u32, height: u32, stride: u32) -> Surface {
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_SRGB,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Opaque,
        }
    }

    /// Build a 2-row × 256-pixel test surface that covers every u8 input byte
    /// on every RGBA channel.
    fn full_domain_surface() -> Surface {
        let w: u32 = 256;
        let h: u32 = 2;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for x in 0..w as usize {
            let row_a = x * 4;
            data[row_a] = x as u8;
            data[row_a + 1] = (255 - x) as u8;
            data[row_a + 2] = ((x * 7) & 0xff) as u8;
            data[row_a + 3] = x as u8;

            let row_b = (w as usize + x) * 4;
            data[row_b] = x as u8;
            data[row_b + 1] = x as u8;
            data[row_b + 2] = x as u8;
            data[row_b + 3] = 255;
        }
        srgb_surface(data, w, h, w * 4)
    }

    fn assert_within_u8_tolerance(pixels: &[[f32; 4]], source: &[u8]) {
        let lut = &*EOTF_LUT;
        let tol = 0.5 / 255.0;
        for (i, px) in pixels.iter().enumerate() {
            let base = i * 4;
            let rb = source[base];
            let gb = source[base + 1];
            let bb = source[base + 2];
            let ab = source[base + 3];
            assert!(
                (px[0] - lut[rb as usize]).abs() < tol,
                "R byte {rb}: {}",
                px[0]
            );
            assert!(
                (px[1] - lut[gb as usize]).abs() < tol,
                "G byte {gb}: {}",
                px[1]
            );
            assert!(
                (px[2] - lut[bb as usize]).abs() < tol,
                "B byte {bb}: {}",
                px[2]
            );
            assert!(
                (px[3] - ab as f32 / 255.0).abs() < 1e-6,
                "A byte {ab}: {}",
                px[3]
            );
        }
    }

    #[test]
    fn avx2_srgb_matches_lut_within_u8_tolerance() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let surface = full_domain_surface();
        let simd = unsafe { load_srgb8_rgba_f32_avx2_fma(&surface).unwrap() };
        assert_within_u8_tolerance(&simd.pixels, &surface.data);
    }

    #[test]
    fn sse4_srgb_matches_lut_within_u8_tolerance() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let surface = full_domain_surface();
        let simd = unsafe { load_srgb8_rgba_f32_sse4_1(&surface).unwrap() };
        assert_within_u8_tolerance(&simd.pixels, &surface.data);
    }

    #[test]
    fn avx2_srgb_odd_width_tail_matches_sse4_path() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        // 3 px wide exercises one AVX2 iteration (2 px) + one SSE4.1 tail (1 px).
        let data = vec![
            0u8, 10, 11, 255, //
            128, 200, 255, 64, //
            17, 42, 99, 200,
        ];
        let surface = srgb_surface(data.clone(), 3, 1, 12);

        let avx2 = unsafe { load_srgb8_rgba_f32_avx2_fma(&surface).unwrap() };

        // The 1-pixel tail runs through the same SSE4.1 helper as the SSE4 path,
        // so a dedicated 1×1 SSE4 decode of the same bytes must match bit-exactly.
        let tail_surface = srgb_surface(data[8..].to_vec(), 1, 1, 4);
        let sse4 = unsafe { load_srgb8_rgba_f32_sse4_1(&tail_surface).unwrap() };
        assert_eq!(avx2.pixels[2], sse4.pixels[0]);

        // And the tail still lands inside u8 tolerance of the LUT reference.
        assert_within_u8_tolerance(&avx2.pixels[2..], &data[8..]);
    }

    #[test]
    fn avx2_srgb_stride_padding_is_skipped() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        // 2×2 pixel image with 4 bytes of junk padding on every row.
        let w = 2u32;
        let h = 2u32;
        let stride = w * 4 + 4;
        let mut data = Vec::new();
        let rows = [
            [10u8, 20, 30, 40, 50, 60, 70, 80],
            [90, 100, 110, 120, 130, 140, 150, 160],
        ];
        for r in &rows {
            data.extend_from_slice(r);
            data.extend_from_slice(&[0xFE, 0xFE, 0xFE, 0xFE]);
        }

        let surface = srgb_surface(data, w, h, stride);
        let simd = unsafe { load_srgb8_rgba_f32_avx2_fma(&surface).unwrap() };

        assert_eq!(simd.pixels.len(), 4);
        // The 0xFE junk must not show up — alpha of pixel 1 is 40/255, not 254/255.
        assert!((simd.pixels[0][3] - 40.0 / 255.0).abs() < 1e-6);
        assert!((simd.pixels[3][3] - 160.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn sse4_srgb_stride_padding_is_skipped() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        let w = 2u32;
        let h = 2u32;
        let stride = w * 4 + 4;
        let mut data = Vec::new();
        let rows = [
            [10u8, 20, 30, 40, 50, 60, 70, 80],
            [90, 100, 110, 120, 130, 140, 150, 160],
        ];
        for r in &rows {
            data.extend_from_slice(r);
            data.extend_from_slice(&[0xFE, 0xFE, 0xFE, 0xFE]);
        }

        let surface = srgb_surface(data, w, h, stride);
        let simd = unsafe { load_srgb8_rgba_f32_sse4_1(&surface).unwrap() };

        assert_eq!(simd.pixels.len(), 4);
        assert!((simd.pixels[0][3] - 40.0 / 255.0).abs() < 1e-6);
        assert!((simd.pixels[3][3] - 160.0 / 255.0).abs() < 1e-6);
    }

    fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
    }

    #[test]
    fn avx512_srgb_matches_lut_within_u8_tolerance() {
        if !has_avx512() {
            return;
        }
        let surface = full_domain_surface();
        let simd = unsafe { load_srgb8_rgba_f32_avx512(&surface).unwrap() };
        assert_within_u8_tolerance(&simd.pixels, &surface.data);
    }

    #[test]
    fn avx512_srgb_tail_matches_lut_within_u8_tolerance() {
        if !has_avx512() {
            return;
        }

        // 7 px wide exercises one 4-pixel main-loop iteration plus a
        // 3-pixel masked tail.
        let data = vec![
            0u8, 10, 11, 255, //
            128, 200, 255, 64, //
            17, 42, 99, 200, //
            77, 88, 99, 111, //
            1, 2, 3, 4, //
            250, 240, 230, 220, //
            5, 100, 200, 255,
        ];
        let surface = srgb_surface(data.clone(), 7, 1, 7 * 4);
        let simd = unsafe { load_srgb8_rgba_f32_avx512(&surface).unwrap() };
        assert_within_u8_tolerance(&simd.pixels, &data);
    }

    #[test]
    fn avx512_srgb_multiple_main_plus_tail_matches_lut() {
        if !has_avx512() {
            return;
        }

        // 15 px wide exercises multiple 4-pixel main iterations plus a
        // 3-pixel masked tail, so the output offset has to compound
        // correctly across them.
        let w = 15u32;
        let mut data = Vec::with_capacity((w * 4) as usize);
        for i in 0..w as usize {
            data.push((i.wrapping_mul(37)) as u8);
            data.push((i.wrapping_mul(59).wrapping_add(11)) as u8);
            data.push((i.wrapping_mul(97).wrapping_add(3)) as u8);
            data.push((i.wrapping_mul(13)) as u8);
        }
        let surface = srgb_surface(data.clone(), w, 1, w * 4);
        let simd = unsafe { load_srgb8_rgba_f32_avx512(&surface).unwrap() };
        assert_within_u8_tolerance(&simd.pixels, &data);
    }

    #[test]
    fn avx512_srgb_stride_padding_is_skipped() {
        if !has_avx512() {
            return;
        }

        // 2×2 pixel image with 4 bytes of junk padding on every row. The
        // AVX-512 tail mask must cover exactly 2 pixels (8 bytes) and leave
        // the junk past the row untouched.
        let w = 2u32;
        let h = 2u32;
        let stride = w * 4 + 4;
        let mut data = Vec::new();
        let rows = [
            [10u8, 20, 30, 40, 50, 60, 70, 80],
            [90, 100, 110, 120, 130, 140, 150, 160],
        ];
        for r in &rows {
            data.extend_from_slice(r);
            data.extend_from_slice(&[0xFE, 0xFE, 0xFE, 0xFE]);
        }

        let surface = srgb_surface(data, w, h, stride);
        let simd = unsafe { load_srgb8_rgba_f32_avx512(&surface).unwrap() };

        assert_eq!(simd.pixels.len(), 4);
        assert!((simd.pixels[0][3] - 40.0 / 255.0).abs() < 1e-6);
        assert!((simd.pixels[3][3] - 160.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn sse4_eotf_in_place_matches_exact() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let orig = eotf_test_pixels();
        let mut got = orig.clone();
        unsafe { srgb_eotf_in_place_f32_sse4_1(&mut got) };
        assert_decoded_close(&got, &orig);
    }

    #[test]
    fn avx2_eotf_in_place_matches_exact_across_tails() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let base = eotf_test_pixels();
        // 0-9 covers the 2-pixel main loop with and without the 1-pixel
        // SSE4.1 tail.
        for n in [0usize, 1, 2, 3, 5, 8, 9] {
            let orig = base[..n].to_vec();
            let mut got = orig.clone();
            unsafe { srgb_eotf_in_place_f32_avx2_fma(&mut got) };
            assert_decoded_close(&got, &orig);
        }

        let mut got = base.clone();
        unsafe { srgb_eotf_in_place_f32_avx2_fma(&mut got) };
        assert_decoded_close(&got, &base);
    }

    #[test]
    fn avx512_eotf_in_place_matches_exact_across_tails() {
        if !has_avx512() {
            return;
        }
        let base = eotf_test_pixels();
        // 0-9 covers the 4-pixel main loop plus every 1-3 pixel masked tail.
        for n in [0usize, 1, 2, 3, 4, 5, 6, 7, 9] {
            let orig = base[..n].to_vec();
            let mut got = orig.clone();
            unsafe { srgb_eotf_in_place_f32_avx512(&mut got) };
            assert_decoded_close(&got, &orig);
        }

        let mut got = base.clone();
        unsafe { srgb_eotf_in_place_f32_avx512(&mut got) };
        assert_decoded_close(&got, &base);
    }

    /// One `[c, c, c, c]` pixel per u8 lattice point.
    fn u8_lattice_pixels() -> Vec<[f32; 4]> {
        (0..=255u8)
            .map(|b| {
                let c = b as f32 / 255.0;
                [c, c, c, c]
            })
            .collect()
    }

    /// Every u8 byte value, decoded by the in-place EOTF, must land back on
    /// the same byte when re-encoded through the store-side in-place OETF —
    /// the bit-exact roundtrip the two approximations were tuned for.
    fn assert_in_place_roundtrips(pixels: &[[f32; 4]]) {
        for (b, p) in pixels.iter().enumerate() {
            for (c, &lane) in p.iter().enumerate() {
                assert_eq!(
                    (lane * 255.0).round() as u8,
                    b as u8,
                    "lane {c} roundtrip failed for value {b}"
                );
            }
        }
    }

    #[test]
    fn sse4_eotf_in_place_u8_roundtrip_is_exact() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut pixels = u8_lattice_pixels();
        unsafe { srgb_eotf_in_place_f32_sse4_1(&mut pixels) };
        unsafe {
            crate::processing::store_kernels::srgb::srgb_oetf_in_place_f32_sse4_1(&mut pixels)
        };
        assert_in_place_roundtrips(&pixels);
    }

    #[test]
    fn avx2_eotf_in_place_u8_roundtrip_is_exact() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut pixels = u8_lattice_pixels();
        unsafe { srgb_eotf_in_place_f32_avx2_fma(&mut pixels) };
        unsafe {
            crate::processing::store_kernels::srgb::srgb_oetf_in_place_f32_avx2_fma(&mut pixels)
        };
        assert_in_place_roundtrips(&pixels);
    }

    #[test]
    fn avx512_eotf_in_place_u8_roundtrip_is_exact() {
        if !has_avx512() {
            return;
        }
        let mut pixels = u8_lattice_pixels();
        unsafe { srgb_eotf_in_place_f32_avx512(&mut pixels) };
        unsafe {
            crate::processing::store_kernels::srgb::srgb_oetf_in_place_f32_avx512(&mut pixels)
        };
        assert_in_place_roundtrips(&pixels);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests {
    use super::eotf_in_place_tests::{assert_decoded_close, eotf_test_pixels};
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::{ColorSpace, Surface};

    fn has_neon() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    fn srgb_surface(data: Vec<u8>, width: u32, height: u32, stride: u32) -> Surface {
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_SRGB,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Opaque,
        }
    }

    fn full_domain_surface() -> Surface {
        let w: u32 = 256;
        let h: u32 = 2;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for x in 0..w as usize {
            let row_a = x * 4;
            data[row_a] = x as u8;
            data[row_a + 1] = (255 - x) as u8;
            data[row_a + 2] = ((x * 7) & 0xff) as u8;
            data[row_a + 3] = x as u8;

            let row_b = (w as usize + x) * 4;
            data[row_b] = x as u8;
            data[row_b + 1] = x as u8;
            data[row_b + 2] = x as u8;
            data[row_b + 3] = 255;
        }
        srgb_surface(data, w, h, w * 4)
    }

    fn assert_within_u8_tolerance(pixels: &[[f32; 4]], source: &[u8]) {
        let lut = &*EOTF_LUT;
        let tol = 0.5 / 255.0;
        for (i, px) in pixels.iter().enumerate() {
            let base = i * 4;
            let rb = source[base];
            let gb = source[base + 1];
            let bb = source[base + 2];
            let ab = source[base + 3];
            assert!((px[0] - lut[rb as usize]).abs() < tol);
            assert!((px[1] - lut[gb as usize]).abs() < tol);
            assert!((px[2] - lut[bb as usize]).abs() < tol);
            assert!((px[3] - ab as f32 / 255.0).abs() < 1e-6);
        }
    }

    #[test]
    fn neon_srgb_matches_lut_within_u8_tolerance() {
        if !has_neon() {
            return;
        }
        let surface = full_domain_surface();
        let simd = unsafe { load_srgb8_rgba_f32_neon(&surface).unwrap() };
        assert_within_u8_tolerance(&simd.pixels, &surface.data);
    }

    #[test]
    fn neon_srgb_tail_matches_lut_within_u8_tolerance() {
        if !has_neon() {
            return;
        }

        let data = vec![
            0u8, 10, 11, 255, //
            128, 200, 255, 64, //
            17, 42, 99, 200, //
            77, 88, 99, 111, //
            1, 2, 3, 4, //
            250, 240, 230, 220, //
            5, 100, 200, 255,
        ];
        let surface = srgb_surface(data.clone(), 7, 1, 7 * 4);
        let simd = unsafe { load_srgb8_rgba_f32_neon(&surface).unwrap() };
        assert_within_u8_tolerance(&simd.pixels, &data);
    }

    #[test]
    fn neon_srgb_stride_padding_is_skipped() {
        if !has_neon() {
            return;
        }

        let w = 2u32;
        let h = 2u32;
        let stride = w * 4 + 4;
        let mut data = Vec::new();
        let rows = [
            [10u8, 20, 30, 40, 50, 60, 70, 80],
            [90, 100, 110, 120, 130, 140, 150, 160],
        ];
        for r in &rows {
            data.extend_from_slice(r);
            data.extend_from_slice(&[0xFE, 0xFE, 0xFE, 0xFE]);
        }

        let surface = srgb_surface(data, w, h, stride);
        let simd = unsafe { load_srgb8_rgba_f32_neon(&surface).unwrap() };

        assert_eq!(simd.pixels.len(), 4);
        assert!((simd.pixels[0][3] - 40.0 / 255.0).abs() < 1e-6);
        assert!((simd.pixels[3][3] - 160.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn neon_eotf_in_place_matches_exact() {
        if !has_neon() {
            return;
        }
        let orig = eotf_test_pixels();
        let mut got = orig.clone();
        unsafe { srgb_eotf_in_place_f32_neon(&mut got) };
        assert_decoded_close(&got, &orig);
    }

    /// Every u8 byte value, decoded by the in-place EOTF, must land back on
    /// the same byte when re-encoded through the store-side in-place OETF —
    /// the bit-exact roundtrip the two approximations were tuned for.
    #[test]
    fn neon_eotf_in_place_u8_roundtrip_is_exact() {
        if !has_neon() {
            return;
        }
        let mut pixels: Vec<[f32; 4]> = (0..=255u8)
            .map(|b| {
                let c = b as f32 / 255.0;
                [c, c, c, c]
            })
            .collect();
        unsafe { srgb_eotf_in_place_f32_neon(&mut pixels) };
        unsafe { crate::processing::store_kernels::srgb::srgb_oetf_in_place_f32_neon(&mut pixels) };
        for (b, p) in pixels.iter().enumerate() {
            for (c, &lane) in p.iter().enumerate() {
                assert_eq!(
                    (lane * 255.0).round() as u8,
                    b as u8,
                    "lane {c} roundtrip failed for value {b}"
                );
            }
        }
    }
}
