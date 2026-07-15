//! Shared machinery for the in-place sRGB transfer-curve passes.
//!
//! Both directions (the load-side EOTF post-pass and the store-side OETF
//! pre-pass) are the same loop skeletons over `[[f32; 4]]` pixels with only
//! the curve evaluation swapped out; [`in_place_curve_pass`] generates the
//! whole kernel family from the per-ISA curve helpers, and the interp-LUT
//! helpers back the serial paths.

/// Interval count of the interpolated curve LUTs (tables hold one extra
/// entry so index `INTERP_LUT_SIZE` is a valid right endpoint).
pub(crate) const INTERP_LUT_SIZE: usize = 4096;

/// Sample `curve` over `[0, 1]` into an `INTERP_LUT_SIZE + 1`-entry table
/// for [`sample_interp_lut`].
pub(crate) fn build_interp_lut(curve: impl Fn(f32) -> f32) -> [f32; INTERP_LUT_SIZE + 1] {
    let mut table = [0.0f32; INTERP_LUT_SIZE + 1];
    for (i, entry) in table.iter_mut().enumerate() {
        *entry = curve(i as f32 / INTERP_LUT_SIZE as f32);
    }
    table
}

/// Linearly-interpolated lookup of `c`, clamped to `[0, 1]`.
#[inline(always)]
pub(crate) fn sample_interp_lut(lut: &[f32; INTERP_LUT_SIZE + 1], c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    let scaled = c * INTERP_LUT_SIZE as f32;
    let idx = scaled as usize;
    if idx >= INTERP_LUT_SIZE {
        return lut[INTERP_LUT_SIZE];
    }
    let frac = scaled - idx as f32;
    lut[idx] + frac * (lut[idx + 1] - lut[idx])
}

/// Generates an in-place transfer-curve pass family over `[[f32; 4]]`
/// pixels: a runtime-dispatched entry point plus serial and per-ISA kernels.
///
/// Every generated kernel applies the given curve to the RGB lanes of each
/// pixel — clamped to `[0, 1]` first on the SIMD paths; the scalar curve is
/// expected to clamp itself — and preserves the alpha lane bit-exactly.
///
/// Each curve helper must evaluate the curve on every lane of one full
/// vector register of the matching width; the per-ISA kernels blend the
/// original alpha lanes back over the curve output.
macro_rules! in_place_curve_pass {
    (
        $(#[$dispatch_doc:meta])*
        pub fn $dispatch:ident;
        serial: $serial:ident($scalar_curve:path);
        sse4_1: $sse4_1:ident($curve_sse4_1:path);
        avx2_fma: $avx2_fma:ident($curve_avx2:path);
        avx512: $avx512:ident($curve_avx512:path);
        neon: $neon:ident($curve_neon:path);
    ) => {
        $(#[$dispatch_doc])*
        pub fn $dispatch(pixels: &mut [[f32; 4]]) {
            profiling::scope!(stringify!($dispatch));
            $crate::processing::dispatch::dispatch_simd! {
                x86_64: {
                    avx512: $avx512(pixels),
                    avx2_fma: $avx2_fma(pixels),
                    sse4_1: $sse4_1(pixels),
                },
                aarch64: {
                    neon: $neon(pixels),
                },
            }
            $serial(pixels)
        }

        #[doc = concat!("Serial interpolated-LUT path for [`", stringify!($dispatch), "`].")]
        ///
        /// **Not part of the public API.** Exposed so benchmarks can compare
        /// the scalar implementation directly against each runtime-selectable
        /// SIMD mode.
        #[doc(hidden)]
        pub fn $serial(pixels: &mut [[f32; 4]]) {
            profiling::scope!(stringify!($serial));
            for p in pixels {
                p[0] = $scalar_curve(p[0]);
                p[1] = $scalar_curve(p[1]);
                p[2] = $scalar_curve(p[2]);
            }
        }

        #[doc = concat!("SSE4.1 path for [`", stringify!($dispatch), "`]: one pixel (4 f32) per iteration.")]
        ///
        /// **Not part of the public API.** Exposed for benchmarks; real
        #[doc = concat!("callers use the runtime-dispatched [`", stringify!($dispatch), "`].")]
        ///
        /// # Safety
        /// The SSE4.1 feature must be available.
        #[doc(hidden)]
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        pub unsafe fn $sse4_1(pixels: &mut [[f32; 4]]) {
            use std::arch::x86_64::*;
            profiling::scope!(stringify!($sse4_1));

            for px in pixels {
                let p = px.as_mut_ptr();
                // SAFETY: `px` spans exactly 4 f32 lanes.
                unsafe {
                    let lanes = _mm_loadu_ps(p);
                    let x = _mm_max_ps(_mm_min_ps(lanes, _mm_set1_ps(1.0)), _mm_setzero_ps());
                    let rgb = $curve_sse4_1(x);
                    // Bit 3 selects the original (unclamped) alpha lane.
                    let out = _mm_blend_ps::<0b1000>(rgb, lanes);
                    _mm_storeu_ps(p, out);
                }
            }
        }

        #[doc = concat!("AVX2+FMA path for [`", stringify!($dispatch), "`]: two pixels (8 f32) per iteration, with an optional 1-pixel SSE4.1 tail.")]
        ///
        /// **Not part of the public API.** Exposed for benchmarks; real
        #[doc = concat!("callers use the runtime-dispatched [`", stringify!($dispatch), "`].")]
        ///
        /// # Safety
        /// The AVX2 and FMA features must be available.
        #[doc(hidden)]
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn $avx2_fma(pixels: &mut [[f32; 4]]) {
            use std::arch::x86_64::*;
            profiling::scope!(stringify!($avx2_fma));

            let mut pairs = pixels.chunks_exact_mut(2);
            for pair in pairs.by_ref() {
                let p = pair.as_mut_ptr() as *mut f32;
                // SAFETY: `pair` spans exactly 8 f32 lanes.
                unsafe {
                    let lanes = _mm256_loadu_ps(p);
                    let x = _mm256_max_ps(
                        _mm256_min_ps(lanes, _mm256_set1_ps(1.0)),
                        _mm256_setzero_ps(),
                    );
                    let rgb = $curve_avx2(x);
                    // Bits 3 and 7 select the original (unclamped) alpha lanes.
                    let out = _mm256_blend_ps::<0b1000_1000>(rgb, lanes);
                    _mm256_storeu_ps(p, out);
                }
            }

            // Odd-pixel tail at 128-bit width (avx2 implies sse4.1).
            for px in pairs.into_remainder() {
                let p = px.as_mut_ptr();
                // SAFETY: `px` spans exactly 4 f32 lanes.
                unsafe {
                    let lanes = _mm_loadu_ps(p);
                    let x = _mm_max_ps(_mm_min_ps(lanes, _mm_set1_ps(1.0)), _mm_setzero_ps());
                    let rgb = $curve_sse4_1(x);
                    let out = _mm_blend_ps::<0b1000>(rgb, lanes);
                    _mm_storeu_ps(p, out);
                }
            }
        }

        #[doc = concat!("AVX-512 path for [`", stringify!($dispatch), "`]: four pixels (16 f32) per iteration, with a 1-3 pixel masked tail.")]
        ///
        /// **Not part of the public API.** Exposed for benchmarks; real
        #[doc = concat!("callers use the runtime-dispatched [`", stringify!($dispatch), "`].")]
        ///
        /// # Safety
        /// The AVX-512 F, VL, and BW features must be available.
        #[doc(hidden)]
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
        pub unsafe fn $avx512(pixels: &mut [[f32; 4]]) {
            use std::arch::x86_64::*;

            use $crate::processing::x86::ALPHA_LANES_512;

            profiling::scope!(stringify!($avx512));

            let mut quads = pixels.chunks_exact_mut(4);
            for quad in quads.by_ref() {
                let p = quad.as_mut_ptr() as *mut f32;
                // SAFETY: `quad` spans exactly 16 f32 lanes.
                unsafe {
                    let lanes = _mm512_loadu_ps(p);
                    let x = _mm512_max_ps(
                        _mm512_min_ps(lanes, _mm512_set1_ps(1.0)),
                        _mm512_setzero_ps(),
                    );
                    let rgb = $curve_avx512(x);
                    let out = _mm512_mask_blend_ps(ALPHA_LANES_512, rgb, lanes);
                    _mm512_storeu_ps(p, out);
                }
            }

            // 1-3 pixel tail in a single masked iteration. The mask
            // suppresses both the load and the store for lanes beyond the
            // remaining pixel count, so the intervening math can run
            // unmasked on zeros without touching memory.
            let tail = quads.into_remainder();
            if !tail.is_empty() {
                let p = tail.as_mut_ptr() as *mut f32;
                let mask: __mmask16 = (1u16 << (tail.len() * 4)) - 1;
                // SAFETY: the masked load/store only touch the
                // `tail.len() * 4` f32 lanes `tail` spans.
                unsafe {
                    let lanes = _mm512_maskz_loadu_ps(mask, p);
                    let x = _mm512_max_ps(
                        _mm512_min_ps(lanes, _mm512_set1_ps(1.0)),
                        _mm512_setzero_ps(),
                    );
                    let rgb = $curve_avx512(x);
                    let out = _mm512_mask_blend_ps(ALPHA_LANES_512, rgb, lanes);
                    _mm512_mask_storeu_ps(p, mask, out);
                }
            }
        }

        #[doc = concat!("NEON path for [`", stringify!($dispatch), "`]: one pixel (4 f32) per iteration.")]
        ///
        /// **Not part of the public API.** Exposed for benchmarks; real
        #[doc = concat!("callers use the runtime-dispatched [`", stringify!($dispatch), "`].")]
        ///
        /// # Safety
        /// The NEON feature must be available.
        #[doc(hidden)]
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        pub unsafe fn $neon(pixels: &mut [[f32; 4]]) {
            use std::arch::aarch64::*;
            profiling::scope!(stringify!($neon));

            // Lane 3 selects the original (unclamped) alpha value in the
            // blend below.
            let alpha_lane_mask = vsetq_lane_u32::<3>(u32::MAX, vdupq_n_u32(0));

            for px in pixels {
                let p = px.as_mut_ptr();
                // SAFETY: `px` spans exactly 4 f32 lanes.
                unsafe {
                    let lanes = vld1q_f32(p);
                    let x = vmaxq_f32(vminq_f32(lanes, vdupq_n_f32(1.0)), vdupq_n_f32(0.0));
                    let rgb = $curve_neon(x);
                    let out = vbslq_f32(alpha_lane_mask, lanes, rgb);
                    vst1q_f32(p, out);
                }
            }
        }
    };
}
pub(crate) use in_place_curve_pass;
