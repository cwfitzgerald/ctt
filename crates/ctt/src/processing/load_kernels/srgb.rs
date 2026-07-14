//! sRGB-encoded u8 loaders.
//!
//! Shared state:
//! * [`EOTF_LUT`] — 256-entry scalar lookup table (exact sRGB EOTF).
//! * [`SRGB_MINIMAX_A`]/[`SRGB_MINIMAX_B`]/[`SRGB_MINIMAX_C`] — piecewise
//!   minimax approximation used by the SIMD fast paths.

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

// Piecewise minimax approximation of the sRGB EOTF over the u8 byte domain.
// With `x = byte / 255`, the curve branch fits `((x + 0.055) / 1.055)^2.4`
// with max abs error ≈ 1.28e-4 — well inside ±0.5/255, so u8 round-trip
// stays bit-exact versus the LUT path. See `srgb-opt.py`.
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

/// Decode one 4-byte sRGB RGBA pixel into `[R, G, B, A]` linear f32 lanes,
/// shared between the SSE4.1 main loop and the AVX2 fast path's tail.
///
/// Piecewise form for RGB, with `x = byte / 255`:
/// * `byte <= 10`: `x / 12.92` (linear segment of the sRGB spec)
/// * `byte >= 11`: `(a·x + b)^2 * (c·x + sqrt(x))` — minimax fit of
///   `((x + 0.055) / 1.055)^2.4` with max abs error ≈ 1.28e-4.
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

    let coeff_a = _mm_set1_ps(SRGB_MINIMAX_A);
    let coeff_b = _mm_set1_ps(SRGB_MINIMAX_B);
    let coeff_c = _mm_set1_ps(SRGB_MINIMAX_C);
    let inv_255 = _mm_set1_ps(1.0 / 255.0);
    let inv_255_12_92 = _mm_set1_ps(1.0 / (255.0 * 12.92));
    // Lane 3 is the alpha channel in the [R,G,B,A] layout.
    let alpha_lane_mask = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, -1));
    let curve_threshold = _mm_set1_epi32(10);

    let x_norm = _mm_mul_ps(as_f32, inv_255);
    let linear = _mm_mul_ps(as_f32, inv_255_12_92);
    let t = _mm_sqrt_ps(x_norm);
    let u = _mm_add_ps(_mm_mul_ps(x_norm, coeff_a), coeff_b);
    let v = _mm_add_ps(_mm_mul_ps(x_norm, coeff_c), t);
    let curve = _mm_mul_ps(_mm_mul_ps(u, u), v);
    let use_curve = _mm_castsi128_ps(_mm_cmpgt_epi32(as_i32, curve_threshold));
    let rgb = _mm_blendv_ps(linear, curve, use_curve);
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

/// AVX2 + FMA path for `R8G8B8A8_SRGB` (and equivalent 4-channel sRGB layouts).
///
/// Processes two pixels (8 bytes → 8 f32) per iteration. Any 1-pixel
/// remainder is handled by the SSE4.1 [`decode_srgb_pixel_sse4_1`] helper,
/// so the tail stays vectorized and consistent with the SSE4.1 fast path.
/// See that helper for the piecewise form and accuracy guarantees.
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
        let coeff_a = _mm256_set1_ps(SRGB_MINIMAX_A);
        let coeff_b = _mm256_set1_ps(SRGB_MINIMAX_B);
        let coeff_c = _mm256_set1_ps(SRGB_MINIMAX_C);
        let inv_255 = _mm256_set1_ps(1.0 / 255.0);
        let inv_255_12_92 = _mm256_set1_ps(1.0 / (255.0 * 12.92));
        // Lanes 3 and 7 are the alpha channel in the [R,G,B,A,R,G,B,A] layout.
        let alpha_lane_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, -1));
        let curve_threshold = _mm256_set1_epi32(10);

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
                let linear = _mm256_mul_ps(as_f32, inv_255_12_92);

                let t = _mm256_sqrt_ps(x_norm);
                let u = _mm256_fmadd_ps(x_norm, coeff_a, coeff_b);
                let v = _mm256_fmadd_ps(x_norm, coeff_c, t);
                let curve = _mm256_mul_ps(_mm256_mul_ps(u, u), v);

                let use_curve = _mm256_castsi256_ps(_mm256_cmpgt_epi32(as_i32, curve_threshold));
                let rgb = _mm256_blendv_ps(linear, curve, use_curve);
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

/// Decode one aligned block of 16 consecutive sRGB bytes (= 4 RGBA pixels)
/// into 16 linear-f32 lanes, shared between the AVX-512 main loop and its
/// masked 1-3 pixel tail.
///
/// Only the lower `n*4` bytes of `bytes` need be valid; higher bytes are
/// treated as zero (produces zero-decoded lanes, which the caller filters
/// out via a masked store).
///
/// See [`decode_srgb_pixel_sse4_1`] for the piecewise form and accuracy
/// guarantees — the math is identical, just widened to 16 lanes.
///
/// # Safety
/// * AVX-512 F must be available (enforced by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
#[inline]
unsafe fn decode_srgb_pixels_avx512(bytes: __m128i) -> __m512 {
    let coeff_a = _mm512_set1_ps(SRGB_MINIMAX_A);
    let coeff_b = _mm512_set1_ps(SRGB_MINIMAX_B);
    let coeff_c = _mm512_set1_ps(SRGB_MINIMAX_C);
    let inv_255 = _mm512_set1_ps(1.0 / 255.0);
    let inv_255_12_92 = _mm512_set1_ps(1.0 / (255.0 * 12.92));
    // Lanes 3, 7, 11, 15 are the alpha channel in the four-pixel
    // [R,G,B,A] × 4 layout.
    let alpha_lane_mask: __mmask16 = 0b1000_1000_1000_1000;
    let curve_threshold = _mm512_set1_epi32(10);

    let as_i32 = _mm512_cvtepu8_epi32(bytes);
    let as_f32 = _mm512_cvtepi32_ps(as_i32);

    let x_norm = _mm512_mul_ps(as_f32, inv_255);
    let linear = _mm512_mul_ps(as_f32, inv_255_12_92);

    let t = _mm512_sqrt_ps(x_norm);
    let u = _mm512_fmadd_ps(x_norm, coeff_a, coeff_b);
    let v = _mm512_fmadd_ps(x_norm, coeff_c, t);
    let curve = _mm512_mul_ps(_mm512_mul_ps(u, u), v);

    let use_curve = _mm512_cmpgt_epi32_mask(as_i32, curve_threshold);
    let rgb = _mm512_mask_blend_ps(use_curve, linear, curve);
    _mm512_mask_blend_ps(alpha_lane_mask, rgb, x_norm)
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

/// Decode one 4-lane `[R, G, B, A]` u8 vector into linear f32 lanes.
///
/// See [`decode_srgb_pixel_sse4_1`] for the piecewise form and accuracy
/// guarantees — the math is identical, just mapped to AArch64 NEON.
///
/// # Safety
/// * The NEON feature must be available (enforced by `target_feature`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_srgb_lanes_neon(as_u32: uint32x4_t) -> float32x4_t {
    let as_f32 = vcvtq_f32_u32(as_u32);

    let coeff_a = vdupq_n_f32(SRGB_MINIMAX_A);
    let coeff_b = vdupq_n_f32(SRGB_MINIMAX_B);
    let coeff_c = vdupq_n_f32(SRGB_MINIMAX_C);
    let inv_255 = vdupq_n_f32(1.0 / 255.0);
    let inv_255_12_92 = vdupq_n_f32(1.0 / (255.0 * 12.92));
    let alpha_lane_mask = vsetq_lane_u32::<3>(u32::MAX, vdupq_n_u32(0));
    let curve_threshold = vdupq_n_u32(10);

    let x_norm = vmulq_f32(as_f32, inv_255);
    let linear = vmulq_f32(as_f32, inv_255_12_92);
    let t = vsqrtq_f32(x_norm);
    let u = vaddq_f32(vmulq_f32(x_norm, coeff_a), coeff_b);
    let v = vaddq_f32(vmulq_f32(x_norm, coeff_c), t);
    let curve = vmulq_f32(vmulq_f32(u, u), v);
    let use_curve = vcgtq_u32(as_u32, curve_threshold);
    let rgb = vbslq_f32(use_curve, curve, linear);
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

#[cfg(all(test, target_arch = "x86_64"))]
mod simd_tests {
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
}

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests {
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
}
