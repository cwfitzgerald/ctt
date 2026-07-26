//! sRGB-encoded u8 loaders and storers, and the in-place EOTF/OETF curve
//! passes.
//!
//! # Load side
//!
//! Two evaluations of the sRGB EOTF (sRGB-encoded → linear) live here:
//! * [`EOTF_LUT`], the 256-entry exact-curve u8 table, used by the per-pixel
//!   scalar path that serves the 1/2/3-channel sRGB formats (which have no
//!   packed-word SIMD kernel);
//! * the dispatched `fearless_simd` path — the 4-channel sRGB8 → linear-f32
//!   RGBA/BGRA load and the f32 in-place EOTF post-pass
//!   ([`srgb_eotf_in_place_f32`]).
//!
//! The SIMD kernels evaluate the EOTF through a branchless minimax fit,
//! `(a·x + b)² · (c·x + √x)` on the curve segment and a linear ramp below
//! [`EOTF_LINEAR_MAX`], staying within ±0.5/255 of the exact curve so a decoded
//! color lane lands on the same u8 as its [`EOTF_LUT`] entry. Color lanes go
//! through the curve; alpha is a straight unorm on the u8 load and preserved
//! bit-exactly on the in-place pass. `BGRA` selects the surface byte order.
//!
//! # Store side
//!
//! Two evaluations of the sRGB OETF (linear → sRGB-encoded) mirror them:
//! * [`srgb_oetf_scalar`], the per-pixel minimax curve used by the 1/2/3-channel
//!   sRGB formats (which have no packed-word SIMD kernel);
//! * the dispatched `fearless_simd` path — the 4-channel linear-f32 → sRGB8
//!   RGBA/BGRA store and the f32 in-place OETF pre-pass
//!   ([`srgb_oetf_in_place_f32`]).
//!
//! Both evaluate the same minimax curve with the same op sequence
//! ([`srgb_oetf`] and its scalar twin [`srgb_oetf_scalar`]) and quantize by
//! `round_ties_even`, so a channel encodes to the same byte whichever path it
//! takes. The `srgb_oetf_minimax_within_half_lsb` test pins the curve's absolute
//! accuracy within ±0.5/255 of the exact sRGB OETF.
//!
//! The reciprocal is a refined `rcp14` estimate on the 512-bit path (where the
//! divider is the throughput bottleneck) and an exact divide on narrower
//! widths. The reciprocal-sqrt has no portable estimate, so the generic path
//! computes `recip(sqrt(x))`; an AVX-512 escape hatch swaps in a direct
//! `rsqrt14` + Newton-Raphson via a `kernel!` block, avoiding the extra `sqrt`.

use std::sync::LazyLock;

#[cfg(target_arch = "x86_64")]
use fearless_simd::Avx512;
use fearless_simd::{Level, Simd, dispatch, prelude::*};

use crate::error::Result;
use crate::processing::Buffer;
use crate::processing::load_kernels::read_pixels;
use crate::processing::store_kernels::write_pixels;
use crate::surface::Surface;

use super::curve_pass::{CurveKernel, curve_in_place_with_token};
use super::driver::{self, clamp01};

// Piecewise minimax EOTF constants for the fit documented at the module level.
const EOTF_A: f32 = -0.983_177_1;
const EOTF_B: f32 = -0.083_670_19;
const EOTF_C: f32 = -0.121_285_7;
/// Boundary between the linear segment and the curve segment of the sRGB EOTF.
const EOTF_LINEAR_MAX: f32 = 0.040_45;
/// Reciprocal slope of the sRGB EOTF linear segment.
const EOTF_INV_LINEAR_SLOPE: f32 = 1.0 / 12.92;

/// sRGB EOTF lookup table — maps every u8 value (0–255) to its linear f32 equivalent.
static EOTF_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut table = [0.0f32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *entry = srgb_eotf_precise(c);
    }
    table
});

/// Apply the exact sRGB EOTF (sRGB-encoded → linear) to a single value.
fn srgb_eotf_precise(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn load_srgb8_f32(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_srgb8_f32");

    if channels == 4 {
        return load_srgb8_f32_at::<false>(Level::new(), surface);
    }

    load_srgb8_f32_scalar(surface, channels)
}

/// Per-pixel exact-LUT path for the 1/2/3-channel sRGB formats, whose pixels
/// are not one packed 32-bit word and so have no SIMD kernel.
fn load_srgb8_f32_scalar(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_srgb8_f32_scalar");

    let lut = &*EOTF_LUT;
    read_pixels(
        surface,
        channels,
        1,
        [0.0, 0.0, 0.0, 1.0],
        |bytes, lanes| {
            // RGB lanes through the sRGB EOTF, alpha linear.
            for (c, (lane, &byte)) in lanes.iter_mut().zip(bytes).enumerate() {
                *lane = if c < 3 {
                    lut[byte as usize]
                } else {
                    byte as f32 / 255.0
                };
            }
        },
    )
}

pub fn load_bgra8_srgb_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_bgra8_srgb_f32");
    load_srgb8_f32_at::<true>(Level::new(), surface)
}

pub fn load_bgr8_srgb_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_bgr8_srgb_f32");
    let lut = &*EOTF_LUT;
    read_pixels(surface, 3, 1, [0.0, 0.0, 0.0, 1.0], |bytes, lanes| {
        let &[b, g, r] = <&[u8; 3]>::try_from(bytes).expect("3-byte pixel");
        lanes[0] = lut[r as usize];
        lanes[1] = lut[g as usize];
        lanes[2] = lut[b as usize];
    })
}

// ---- sRGB8 → linear-f32 RGBA/BGRA SIMD load ----

/// SIMD kernel behind [`load_srgb8_f32`] and [`load_bgra8_srgb_f32`], at a
/// caller-chosen [`Level`] so benches can force each backend. `BGRA` selects
/// the surface byte order.
#[doc(hidden)]
pub fn load_srgb8_f32_at<const BGRA: bool>(level: Level, surface: &Surface) -> Result<Buffer<f32>> {
    dispatch!(level, simd => driver::load_packed32(
        simd,
        surface,
        #[expect(
            clippy::redundant_closure,
            reason = "the closure literal is the force-inline boundary carrying the token's features"
        )]
        #[inline(always)]
        |s, codes, dst| load_row::<_, BGRA>(s, codes, dst),
    ))
}

/// Decode `codes` (one packed word per pixel) into `dst` as `[R, G, B, A]`.
#[inline(always)]
fn load_row<S: Simd, const BGRA: bool>(simd: S, codes: &[u32], dst: &mut [f32]) {
    driver::for_each_block::<_, _, 1, 4>(
        S::u32s::N,
        codes.len(),
        codes,
        dst,
        #[inline(always)]
        |cin, cout| {
            let words = S::u32s::from_slice(simd, cin);
            let (r, g, b, a) = decode_words::<S, BGRA>(simd, words);
            driver::interleave_rgba_f32::<S>(r, g, b, a, cout);
        },
    );
}

/// Split packed little-endian words into channel bytes, normalize to `[0, 1]`,
/// and EOTF-decode the color channels; alpha stays a straight unorm. `BGRA`
/// swaps the R and B byte positions (G at bits 8–15, A at 24–31 are fixed).
#[inline(always)]
fn decode_words<S: Simd, const BGRA: bool>(
    simd: S,
    words: S::u32s,
) -> (S::f32s, S::f32s, S::f32s, S::f32s) {
    let inv = 1.0 / 255.0;
    let (rf, bf): (S::f32s, S::f32s) = if BGRA {
        (((words >> 16) & 0xff).to_float(), (words & 0xff).to_float())
    } else {
        ((words & 0xff).to_float(), ((words >> 16) & 0xff).to_float())
    };
    let gf: S::f32s = ((words >> 8) & 0xff).to_float();
    let af: S::f32s = (words >> 24).to_float();
    (
        srgb_eotf(simd, rf * inv),
        srgb_eotf(simd, gf * inv),
        srgb_eotf(simd, bf * inv),
        af * inv,
    )
}

/// Piecewise sRGB EOTF over `x ∈ [0, 1]`, using the minimax fit
/// `(a·x + b)^2 · (c·x + sqrt(x))`.
#[inline(always)]
fn srgb_eotf<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let t = x.sqrt();
    let u = x.mul_add(S::f32s::splat(simd, EOTF_A), S::f32s::splat(simd, EOTF_B));
    let v = x.mul_add(S::f32s::splat(simd, EOTF_C), t);
    let curve = u * u * v;
    let linear = x * EOTF_INV_LINEAR_SLOPE;

    x.simd_le(S::f32s::splat(simd, EOTF_LINEAR_MAX))
        .select(linear, curve)
}

// ---- in-place f32 EOTF post-pass ----

/// Apply the sRGB EOTF in place to the RGB lanes of every pixel, leaving the
/// alpha lane untouched.
///
/// Post-pass for loads from FormatKinds with no sRGB kernel variant (16+ bit
/// formats) whose source color_space is nonetheless `Srgb`: the buffer is read
/// through the format's linear kernel, then EOTF-decoded here.
///
/// RGB lanes are clamped to `[0, 1]` before decoding; alpha is preserved
/// bit-exactly. Max abs error ≈ 1.5e-4, bit-exact inverted by the store-side
/// OETF approximation. NaN color lanes have no guaranteed behavior.
pub fn srgb_eotf_in_place_f32(pixels: &mut [[f32; 4]]) {
    profiling::scope!("srgb_eotf_in_place_f32");
    srgb_eotf_in_place_f32_at(Level::new(), pixels);
}

/// sRGB EOTF (encoded → linear) for the load-side post-pass.
struct Eotf;
impl CurveKernel for Eotf {
    #[inline(always)]
    fn curve<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
        srgb_eotf(simd, x)
    }
}

/// Run the in-place EOTF at a caller-chosen [`Level`].
#[doc(hidden)]
pub fn srgb_eotf_in_place_f32_at(level: Level, pixels: &mut [[f32; 4]]) {
    dispatch!(level, simd => curve_in_place_with_token::<_, Eotf>(simd, pixels))
}

// Piecewise minimax OETF constants for the fit documented at the module level.
const OETF_A: f32 = 0.075_058_33;
const OETF_B: f32 = 0.048_553_98;
const OETF_C: f32 = 0.027_579_91;
/// Boundary between the linear segment and the curve segment of the sRGB OETF.
const OETF_LINEAR_MAX: f32 = 0.003_130_8;
/// Slope of the sRGB OETF linear segment.
const OETF_LINEAR_SLOPE: f32 = 12.92;

pub fn store_srgb8_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32");

    if channels == 4 {
        return store_srgb8_f32_at::<false>(Level::new(), buf);
    }

    store_srgb8_f32_scalar(buf, channels)
}

/// Per-pixel path for the 1/2/3-channel sRGB formats, whose pixels are not one
/// packed 32-bit word and so have no SIMD kernel.
fn store_srgb8_f32_scalar(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32_scalar");

    write_pixels(buf, channels, 1, |lanes, bytes| {
        for (c, (&lane, byte)) in lanes.iter().zip(bytes.iter_mut()).enumerate() {
            let encoded = if c < 3 {
                srgb_oetf_scalar(lane)
            } else {
                lane.clamp(0.0, 1.0)
            };
            *byte = (encoded * 255.0).round_ties_even() as u8;
        }
    })
}

pub fn store_bgra8_srgb_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgra8_srgb_f32");
    store_srgb8_f32_at::<true>(Level::new(), buf)
}

pub fn store_bgr8_srgb_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgr8_srgb_f32");
    write_pixels(buf, 3, 1, |lanes, bytes| {
        let arr = <&mut [u8; 3]>::try_from(bytes).expect("3-byte pixel");
        arr[0] = (srgb_oetf_scalar(lanes[2]) * 255.0).round_ties_even() as u8;
        arr[1] = (srgb_oetf_scalar(lanes[1]) * 255.0).round_ties_even() as u8;
        arr[2] = (srgb_oetf_scalar(lanes[0]) * 255.0).round_ties_even() as u8;
    })
}

// ---- linear-f32 → sRGB8 RGBA/BGRA SIMD store ----

/// SIMD kernel behind [`store_srgb8_f32`] and [`store_bgra8_srgb_f32`], at a
/// caller-chosen [`Level`] so benches can force each backend. AVX-512 routes
/// through the intrinsic rsqrt escape; every other level runs the generic
/// kernel. `BGRA` selects the byte order.
#[doc(hidden)]
pub fn store_srgb8_f32_at<const BGRA: bool>(level: Level, buf: &Buffer<f32>) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if let Some(avx512) = level.as_avx512() {
        return store_srgb8_f32_avx512_escape::<BGRA>(avx512, buf);
    }
    store_srgb8_f32_generic_at::<BGRA>(level, buf)
}

/// The store kernel with the pure-generic transcendentals (no AVX-512 escape) at
/// a caller-chosen [`Level`], so benches can measure the generic rsqrt path even
/// where AVX-512 is available.
#[doc(hidden)]
pub fn store_srgb8_f32_generic_at<const BGRA: bool>(level: Level, buf: &Buffer<f32>) -> Vec<u8> {
    dispatch!(level, simd => store_with_token::<_, BGRA, RsqrtGeneric>(simd, buf))
}

/// AVX-512 escape configuration: the reciprocal-sqrt routes through a direct
/// `rsqrt14` + Newton-Raphson; every other op stays generic. Runs at a
/// caller-chosen backend for benches.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub fn store_srgb8_f32_avx512_escape<const BGRA: bool>(
    avx512: Avx512,
    buf: &Buffer<f32>,
) -> Vec<u8> {
    avx512.vectorize(
        #[inline(always)]
        || store_with_token::<Avx512, BGRA, RsqrtAvx512Escape>(avx512, buf),
    )
}

/// The whole store loop lives behind this `#[inline(always)]` boundary. The
/// reciprocal-sqrt is a zero-sized `R: RsqrtKernel<S>` type parameter (not a
/// bare `impl Fn` value) so the AVX-512 escape's substitution inlines instead
/// of routing through an out-of-line `Fn::call`.
#[inline(always)]
fn store_with_token<S: Simd, const BGRA: bool, R: RsqrtKernel<S>>(
    simd: S,
    buf: &Buffer<f32>,
) -> Vec<u8> {
    let src: &[f32] = bytemuck::cast_slice(&buf.pixels);
    driver::store_packed32(
        simd,
        src,
        #[inline(always)]
        |s, src, words| store_rows::<_, BGRA, R>(s, src, words),
    )
}

/// Encode every pixel of `src` (4 f32 each) into one packed `u32` word each.
#[inline(always)]
fn store_rows<S: Simd, const BGRA: bool, R: RsqrtKernel<S>>(
    simd: S,
    src: &[f32],
    words: &mut [u32],
) {
    driver::for_each_block::<_, _, 4, 1>(
        S::f32s::N,
        words.len(),
        src,
        words,
        #[inline(always)]
        |cin, cout| {
            let block = encode_block::<S, BGRA, R>(simd, cin);
            block.store_slice(cout);
        },
    );
}

/// Encode `n` interleaved `[R, G, B, A]` pixels into `n` packed little-endian
/// words. `BGRA = false` → byte order `R, G, B, A`; `BGRA = true` → `B, G, R, A`.
#[inline(always)]
fn encode_block<S: Simd, const BGRA: bool, R: RsqrtKernel<S>>(simd: S, src: &[f32]) -> S::u32s {
    let (r, g, b, a) = driver::split_rgba_f32(simd, src);

    let r8 = to_byte::<S>(srgb_oetf::<S, R>(simd, clamp01(simd, r)));
    let g8 = to_byte::<S>(srgb_oetf::<S, R>(simd, clamp01(simd, g)));
    let b8 = to_byte::<S>(srgb_oetf::<S, R>(simd, clamp01(simd, b)));
    let a8 = to_byte::<S>(clamp01(simd, a));

    let (byte0, byte2) = if BGRA { (b8, r8) } else { (r8, b8) };
    byte0 | (g8 << 8) | (byte2 << 16) | (a8 << 24)
}

/// `1 / x`, refined to within f32 ε.
///
/// A refined hardware reciprocal estimate only beats an exact divide once the
/// divider is the throughput bottleneck, which on this target is the 512-bit
/// path. Narrower widths divide exactly — measurably faster there, since the
/// Newton-Raphson multiplies otherwise just add work (and on the fallback
/// `approximate_recip` is itself an exact divide). The lane count is a
/// monomorphization constant, so only one arm survives per backend.
#[inline(always)]
fn recip<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    if S::f32s::N >= 16 {
        // NR step `y' = y·(2 − x·y)` over the `rcp14` estimate, squaring its
        // ~2⁻¹⁴ error below f32 ε.
        let y = x.approximate_recip();
        y * (S::f32s::splat(simd, 2.0) - x * y)
    } else {
        S::f32s::splat(simd, 1.0) / x
    }
}

/// Portable reciprocal-sqrt `1 / sqrt(x)`: exact `sqrt` fed through [`recip`].
/// The 512-bit AVX-512 escape substitutes a direct `rsqrt14` estimate instead.
#[inline(always)]
fn rsqrt_generic<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    recip(simd, x.sqrt())
}

/// Reciprocal-sqrt evaluated on a full native vector. A zero-sized dispatch
/// type rather than a bare `impl Fn` parameter, so the estimate/escape choice
/// inlines into the hot loop instead of going out-of-line through `Fn::call`.
trait RsqrtKernel<S: Simd> {
    fn rsqrt(simd: S, x: S::f32s) -> S::f32s;
}

/// Portable reciprocal-sqrt: exact `sqrt` fed through [`recip`]. Used by every
/// generic (non-escape) entry.
struct RsqrtGeneric;
impl<S: Simd> RsqrtKernel<S> for RsqrtGeneric {
    #[inline(always)]
    fn rsqrt(simd: S, x: S::f32s) -> S::f32s {
        rsqrt_generic(simd, x)
    }
}

/// AVX-512 escape: a direct `rsqrt14` + Newton-Raphson, avoiding the generic
/// path's extra `sqrt`. Used only by the concrete AVX-512 escape entry.
#[cfg(target_arch = "x86_64")]
struct RsqrtAvx512Escape;
#[cfg(target_arch = "x86_64")]
impl RsqrtKernel<Avx512> for RsqrtAvx512Escape {
    #[inline(always)]
    fn rsqrt(simd: Avx512, x: <Avx512 as Simd>::f32s) -> <Avx512 as Simd>::f32s {
        escape::rsqrt_escape(simd, x)
    }
}

/// Piecewise sRGB OETF over `x ∈ [0, 1]`. The reciprocal-sqrt is a type
/// parameter so the AVX-512 escape can substitute an intrinsic `rsqrt14` + NR
/// estimate.
#[inline(always)]
fn srgb_oetf<S: Simd, R: RsqrtKernel<S>>(simd: S, x: S::f32s) -> S::f32s {
    let quarter = x.sqrt().sqrt();
    let diff = quarter - OETF_A;
    let r3 = R::rsqrt(simd, diff);
    let inner = r3 - OETF_B;
    let cube = inner * inner * inner;
    let curve = recip(simd, cube) - OETF_C;
    let linear = x * OETF_LINEAR_SLOPE;

    x.simd_le(S::f32s::splat(simd, OETF_LINEAR_MAX))
        .select(linear, curve)
}

/// Scale `[0, 1]` to `[0, 255]`, round to nearest even, and narrow to a u32
/// byte lane. Callers pre-clamp to `[0, 1]`, so the scaled value never exceeds
/// 255 and the truncating narrow is exact.
#[inline(always)]
fn to_byte<S: Simd>(x: S::f32s) -> S::u32s {
    (x * 255.0).round_ties_even().to_int::<S::u32s>()
}

/// Scalar counterpart of [`srgb_oetf`], replicating its op sequence exactly so
/// the exact-divide backends encode bit-for-bit against it: clamp to `[0, 1]`,
/// then the piecewise minimax curve with plain f32 mul/div (no FMA, no `powf`).
/// Callers quantize the result with `round_ties_even`, as [`to_byte`] does.
fn srgb_oetf_scalar(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= OETF_LINEAR_MAX {
        return x * OETF_LINEAR_SLOPE;
    }
    let r3 = 1.0 / (x.sqrt().sqrt() - OETF_A).sqrt();
    let inner = r3 - OETF_B;
    1.0 / (inner * inner * inner) - OETF_C
}

// ---- AVX-512 escape hatch: intrinsic reciprocal-sqrt ----

#[cfg(target_arch = "x86_64")]
mod escape {
    use fearless_simd::{Avx512, f32x16, prelude::*};
    use std::arch::x86_64::*;

    // `rsqrt14ps` (~2⁻¹⁴ initial error) refined by one Newton-Raphson step
    // `y' = y·(1.5 − 0.5·x·y²)`, squaring the error below f32 ε.
    fearless_simd::kernel!(
        #[inline(always)]
        fn rsqrt_raw(token: Avx512, x: __m512) -> __m512 {
            let y = _mm512_rsqrt14_ps(x);
            let y_sq = _mm512_mul_ps(y, y);
            let half_x = _mm512_mul_ps(_mm512_set1_ps(0.5), x);
            let correction = _mm512_fnmadd_ps(half_x, y_sq, _mm512_set1_ps(1.5));
            _mm512_mul_ps(y, correction)
        }
    );

    #[inline(always)]
    pub fn rsqrt_escape(simd: Avx512, x: f32x16<Avx512>) -> f32x16<Avx512> {
        rsqrt_raw(simd, x.into()).simd_into(simd)
    }
}

// ---- in-place f32 OETF pre-pass ----

/// Apply the sRGB OETF in place to the RGB lanes of every pixel, leaving the
/// alpha lane untouched.
///
/// Pre-pass for stores to FormatKinds with no sRGB kernel variant (16+ bit
/// formats) whose target color_space is nonetheless `Srgb`: the buffer is
/// OETF-encoded here, then written through the format's linear kernel.
///
/// RGB lanes are clamped to `[0, 1]` before encoding; alpha is preserved
/// bit-exactly. Worst-case error ~8e-4, tuned to bit-exact invert the
/// load-side EOTF approximation. NaN color lanes have no guaranteed behavior.
pub fn srgb_oetf_in_place_f32(pixels: &mut [[f32; 4]]) {
    profiling::scope!("srgb_oetf_in_place_f32");
    srgb_oetf_in_place_f32_at(Level::new(), pixels);
}

/// sRGB OETF (linear → encoded) for the store-side pre-pass. Uses the portable
/// reciprocal-sqrt; the in-place passes feed non-u8 formats and are not on the
/// benchmarked u8 hot path.
struct Oetf;
impl CurveKernel for Oetf {
    #[inline(always)]
    fn curve<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
        srgb_oetf::<S, RsqrtGeneric>(simd, x)
    }
}

/// Run the in-place OETF at a caller-chosen [`Level`].
#[doc(hidden)]
pub fn srgb_oetf_in_place_f32_at(level: Level, pixels: &mut [[f32; 4]]) {
    dispatch!(level, simd => curve_in_place_with_token::<_, Oetf>(simd, pixels))
}

#[cfg(test)]
mod load_tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::processing::kernels::constructible_levels;
    use crate::processing::srgb_test_support::{
        assert_curve_close, curve_test_pixels, eotf_exact, eotf_exact_clamped, in_place_pixels,
    };
    use crate::surface::ColorSpace;

    // ---- in-place EOTF ----

    /// RGB lanes within the documented worst-case error of the exact EOTF;
    /// alpha lanes preserved bit-exactly.
    fn assert_decoded_close(got: &[[f32; 4]], orig: &[[f32; 4]]) {
        assert_curve_close(got, orig, eotf_exact_clamped, 5.0e-4, "dispatch");
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

    /// Every u8 byte value, decoded by the in-place EOTF and re-encoded by the
    /// store-side in-place OETF, must land back on the same byte — the bit-exact
    /// roundtrip the two approximations were tuned for. Runs at every level.
    #[test]
    fn simd_eotf_oetf_in_place_u8_roundtrip_is_exact() {
        for (label, level) in constructible_levels() {
            let mut pixels = u8_lattice_pixels();
            srgb_eotf_in_place_f32_at(level, &mut pixels);
            srgb_oetf_in_place_f32_at(level, &mut pixels);
            for (b, p) in pixels.iter().enumerate() {
                for (c, &lane) in p.iter().enumerate() {
                    assert_eq!(
                        (lane * 255.0).round() as u8,
                        b as u8,
                        "{label}: lane {c} roundtrip failed for value {b}"
                    );
                }
            }
        }
    }

    /// The dispatched entry point must agree with the exact EOTF for every
    /// pixel count that exercises the SIMD kernels' main loops and tails.
    #[test]
    fn dispatch_matches_exact_across_tail_lengths() {
        let base = curve_test_pixels();
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

    // ---- SIMD load ----

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

    /// Per-pixel oracle: the exact sRGB EOTF on the three color lanes (in the
    /// requested byte order) and a straight unorm on alpha.
    fn oracle_pixel<const BGRA: bool>(bytes: [u8; 4]) -> [f32; 4] {
        let (r, g, b, a) = if BGRA {
            (bytes[2], bytes[1], bytes[0], bytes[3])
        } else {
            (bytes[0], bytes[1], bytes[2], bytes[3])
        };
        [
            eotf_exact(r as f32 / 255.0),
            eotf_exact(g as f32 / 255.0),
            eotf_exact(b as f32 / 255.0),
            a as f32 / 255.0,
        ]
    }

    /// Oracle for a whole surface: walk the rows at the surface stride, ignoring
    /// any inter-row padding.
    fn oracle_pixels<const BGRA: bool>(surface: &Surface) -> Vec<[f32; 4]> {
        let w = surface.width as usize;
        let stride = surface.stride as usize;
        let mut out = Vec::with_capacity(w * surface.height as usize);
        for y in 0..surface.height as usize {
            let row = &surface.data[y * stride..y * stride + w * 4];
            for px in row.chunks_exact(4) {
                out.push(oracle_pixel::<BGRA>(px.try_into().unwrap()));
            }
        }
        out
    }

    /// Hand-computed lattice points, pinning the exact-EOTF oracle: 0 and 255
    /// are the fixed points, byte 10 (`0.0392 ≤ 0.04045`) is still on the linear
    /// ramp at `c/12.92`, byte 11 has crossed onto the power segment, and 128 is
    /// the familiar mid-gray `≈ 0.2159`.
    #[test]
    fn oracle_matches_known_values() {
        let approx = |got: f32, want: f32, what: &str| {
            assert!((got - want).abs() <= 1e-6, "{what}: got {got} want {want}");
        };
        approx(eotf_exact(0.0), 0.0, "byte 0");
        approx(eotf_exact(1.0), 1.0, "byte 255");
        approx(eotf_exact(10.0 / 255.0), (10.0 / 255.0) / 12.92, "byte 10");
        approx(
            eotf_exact(11.0 / 255.0),
            ((11.0 / 255.0 + 0.055) / 1.055f32).powf(2.4),
            "byte 11",
        );
        approx(eotf_exact(127.0 / 255.0), 0.212_230_8, "byte 127");
        approx(eotf_exact(128.0 / 255.0), 0.215_860_5, "byte 128");
    }

    /// Color lanes within the ±0.5/255 u8 tolerance the minimax curve was fit
    /// to; alpha is a straight unorm.
    fn assert_load_close(got: &[[f32; 4]], want: &[[f32; 4]], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: pixel count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            for c in 0..3 {
                assert!(
                    (g[c] - w[c]).abs() < 0.5 / 255.0,
                    "{label}: pixel {i} lane {c}: got {} want {}",
                    g[c],
                    w[c],
                );
            }
            assert!(
                (g[3] - w[3]).abs() < 1e-6,
                "{label}: pixel {i} alpha: got {} want {}",
                g[3],
                w[3],
            );
        }
    }

    /// 256×2 surface covering every u8 value on every channel.
    fn full_domain_surface() -> Surface {
        let w: u32 = 256;
        let h: u32 = 2;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for x in 0..w as usize {
            let a = x * 4;
            data[a] = x as u8;
            data[a + 1] = (255 - x) as u8;
            data[a + 2] = ((x * 7) & 0xff) as u8;
            data[a + 3] = x as u8;
            let b = (w as usize + x) * 4;
            data[b] = x as u8;
            data[b + 1] = x as u8;
            data[b + 2] = x as u8;
            data[b + 3] = 255;
        }
        srgb_surface(data, w, h, w * 4)
    }

    /// Odd width (exercises every tail residue) with junk stride padding.
    fn tail_stride_surface(width: u32) -> Surface {
        let h = 3u32;
        let row_bytes = width * 4;
        let stride = row_bytes + 12;
        let mut data = vec![0u8; (stride * h) as usize];
        for y in 0..h {
            for xp in 0..width {
                let seed = y.wrapping_mul(9973).wrapping_add(xp.wrapping_mul(7));
                let off = (y * stride + xp * 4) as usize;
                data[off] = seed as u8;
                data[off + 1] = seed.wrapping_mul(3) as u8;
                data[off + 2] = seed.wrapping_mul(5) as u8;
                data[off + 3] = seed.wrapping_mul(7) as u8;
            }
            let pad = (y * stride + row_bytes) as usize;
            for byte in &mut data[pad..pad + 12] {
                *byte = 0xAB;
            }
        }
        srgb_surface(data, width, h, stride)
    }

    #[test]
    fn simd_load_full_domain_matches_oracle() {
        let s = full_domain_surface();
        let want = oracle_pixels::<false>(&s);
        for (label, level) in constructible_levels() {
            let got = load_srgb8_f32_at::<false>(level, &s).unwrap();
            assert_load_close(&got.pixels, &want, label);
        }
    }

    #[test]
    fn simd_load_tails_match_oracle() {
        for (label, level) in constructible_levels() {
            for width in 1..=49u32 {
                let s = tail_stride_surface(width);
                let want = oracle_pixels::<false>(&s);
                let got = load_srgb8_f32_at::<false>(level, &s).unwrap();
                assert_load_close(&got.pixels, &want, &format!("{label} w={width}"));
            }
        }
    }

    /// BGRA load: the same byte data read as `B, G, R, A` must match the exact
    /// oracle in that order, to the same tolerance as the RGBA path.
    #[test]
    fn simd_bgra_load_full_domain_matches_oracle() {
        let s = full_domain_surface();
        let want = oracle_pixels::<true>(&s);
        for (label, level) in constructible_levels() {
            let got = load_srgb8_f32_at::<true>(level, &s).unwrap();
            assert_load_close(&got.pixels, &want, &format!("{label} bgra"));
        }
    }

    #[test]
    fn simd_bgra_load_tails_match_oracle() {
        for (label, level) in constructible_levels() {
            for width in 1..=49u32 {
                let s = tail_stride_surface(width);
                let want = oracle_pixels::<true>(&s);
                let got = load_srgb8_f32_at::<true>(level, &s).unwrap();
                assert_load_close(&got.pixels, &want, &format!("{label} bgra w={width}"));
            }
        }
    }

    /// The 1/2/3-channel scalar production path (no SIMD kernel exists for those
    /// pixel widths) decodes through the exact LUT, so its color lanes must equal
    /// the oracle exactly; absent channels keep their lane defaults.
    #[test]
    fn scalar_sub_four_channel_load_matches_oracle() {
        for channels in 1..=3usize {
            let w = 256u32;
            let mut data = vec![0u8; w as usize * channels];
            for x in 0..w as usize {
                for c in 0..channels {
                    data[x * channels + c] = (x * (c + 1)) as u8;
                }
            }
            let mut s = srgb_surface(data, w, 1, w * channels as u32);
            s.format = match channels {
                1 => ktx2::Format::R8_SRGB,
                2 => ktx2::Format::R8G8_SRGB,
                _ => ktx2::Format::R8G8B8_SRGB,
            };
            let got = load_srgb8_f32(&s, channels).unwrap();
            for (x, px) in got.pixels.iter().enumerate() {
                for (c, &lane) in px.iter().enumerate() {
                    let want = if c < channels {
                        eotf_exact(((x * (c + 1)) as u8) as f32 / 255.0)
                    } else if c == 3 {
                        1.0
                    } else {
                        0.0
                    };
                    assert_eq!(
                        lane, want,
                        "channels={channels} pixel {x} lane {c}: got {lane} want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn simd_eotf_in_place_matches_exact() {
        for (label, level) in constructible_levels() {
            for n in 1..=49usize {
                let orig = in_place_pixels(n);
                let mut got = orig.clone();
                srgb_eotf_in_place_f32_at(level, &mut got);
                assert_curve_close(
                    &got,
                    &orig,
                    eotf_exact_clamped,
                    5.0e-4,
                    &format!("{label} n={n}"),
                );
            }
        }
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use crate::processing::kernels::constructible_levels;
    use crate::processing::srgb_test_support::{
        assert_curve_close, curve_test_pixels, in_place_pixels, oetf_exact, oetf_exact_clamped,
        u8_roundtrip_pixels,
    };

    fn buf_from(pixels: Vec<[f32; 4]>) -> Buffer<f32> {
        let width = pixels.len().max(1) as u32;
        Buffer {
            pixels,
            width,
            height: 1,
        }
    }

    /// Grid over `[0, 1]` plus out-of-range values that must clamp.
    fn grid_pixels() -> Vec<[f32; 4]> {
        let n = 1024usize;
        let mut pixels: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let x = i as f32 / (n - 1) as f32;
                [x, (x * 0.5 + 0.2).clamp(0.0, 1.0), x * x, x]
            })
            .collect();
        pixels.push([-0.5, 2.0, 0.5, -0.1]);
        pixels.push([1.5, -1.0, 0.0, 1.2]);
        pixels
    }

    /// Per-pixel oracle: the exact sRGB OETF on the color lanes and a straight
    /// unorm on alpha, quantized with ties-to-even, in the requested byte order.
    fn oracle_bytes<const BGRA: bool>(pixels: &[[f32; 4]]) -> Vec<u8> {
        let quantize = |x: f32| (x * 255.0).round_ties_even() as u8;
        pixels
            .iter()
            .flat_map(|p| {
                let (c0, c2) = if BGRA { (p[2], p[0]) } else { (p[0], p[2]) };
                [
                    quantize(oetf_exact_clamped(c0)),
                    quantize(oetf_exact_clamped(p[1])),
                    quantize(oetf_exact_clamped(c2)),
                    quantize(p[3].clamp(0.0, 1.0)),
                ]
            })
            .collect()
    }

    /// Store bytes against the exact-curve oracle. The kernel evaluates a
    /// minimax fit within ±0.5/255 of the exact OETF
    /// (`srgb_oetf_minimax_within_half_lsb`), so a color byte can land at most
    /// one LSB off the exactly-encoded value; alpha takes no curve and must be
    /// bit-exact.
    fn assert_bytes_close(actual: &[u8], reference: &[u8], label: &str) {
        assert_eq!(actual.len(), reference.len(), "{label}: length");
        for (i, (&got, &want)) in actual.iter().zip(reference).enumerate() {
            let tol = if i % 4 == 3 { 0 } else { 1 };
            assert!(
                got.abs_diff(want) <= tol,
                "{label}: byte {i} (pixel {}, lane {}) got={got} want={want} (tol {tol})",
                i / 4,
                i % 4,
            );
        }
    }

    /// Two production configurations of the same kernel must agree byte-for-byte.
    /// Only the AVX-512 generic/escape pair (x86_64) has two configurations.
    #[cfg(target_arch = "x86_64")]
    fn assert_bytes_eq(actual: &[u8], reference: &[u8], label: &str) {
        assert_eq!(actual.len(), reference.len(), "{label}: length");
        for (i, (&got, &want)) in actual.iter().zip(reference).enumerate() {
            assert_eq!(
                got,
                want,
                "{label}: byte {i} (pixel {}, lane {}) got={got} want={want}",
                i / 4,
                i % 4,
            );
        }
    }

    /// The minimax OETF the SIMD store evaluates stays within ±0.5/255 of the
    /// exact curve over a dense `[0, 1]` grid, which is what bounds the store
    /// bytes to within ±1 of the exactly-encoded value. Runs the generic
    /// (exact-sqrt/divide) path via the fallback backend, representative of the
    /// narrow-width kernels.
    #[test]
    fn srgb_oetf_minimax_within_half_lsb() {
        let simd = fearless_simd::Fallback::new();
        let n = <fearless_simd::Fallback as Simd>::f32s::N;
        let steps = 1_000_000u32;
        let mut max_err = 0.0f64;
        let mut worst = 0.0f32;
        for i in 0..=steps {
            let x = i as f32 / steps as f32;
            let v = <fearless_simd::Fallback as Simd>::f32s::splat(simd, x);
            let out = srgb_oetf::<_, RsqrtGeneric>(simd, clamp01(simd, v));
            let mut buf = [0.0f32; driver::MAX_LANES];
            out.store_slice(&mut buf[..n]);
            let err = (buf[0] - oetf_exact(x)).abs() as f64;
            if err > max_err {
                max_err = err;
                worst = x;
            }
        }
        eprintln!(
            "minimax OETF max |err| = {max_err:.3e} ({:.4} LSB) at x={worst}",
            max_err * 255.0
        );
        assert!(
            max_err < 0.5 / 255.0,
            "minimax OETF exceeds 0.5/255: {max_err:.3e} at x={worst}"
        );
    }

    /// Hand-computed bytes, pinning the exact-curve oracle independently: the
    /// endpoints are fixed, the linear ramp holds below `0.0031308` (`x·12.92`),
    /// mid-gray `0.2159` encodes to 128, and out-of-range inputs clamp.
    #[test]
    fn oracle_matches_known_values() {
        let one = |x: f32| oracle_bytes::<false>(&[[x, x, x, x]])[0];
        assert_eq!(one(0.0), 0);
        assert_eq!(one(1.0), 255);
        assert_eq!(one(-1.0), 0, "negatives clamp to 0");
        assert_eq!(one(2.0), 255, "above 1 clamps to 255");
        // Linear segment: 0.001 · 12.92 · 255 = 3.29 → 3.
        assert_eq!(one(0.001), 3);
        // The load side's decode of byte 128 must encode back to 128.
        assert_eq!(one(0.215_860_5), 128);
        assert_eq!(one(0.212_230_8), 127);
        // Alpha takes no curve: 0.5 · 255 = 127.5, ties to even → 128.
        assert_eq!(oracle_bytes::<false>(&[[0.0, 0.0, 0.0, 0.5]])[3], 128);
    }

    #[test]
    fn simd_matches_exact_oracle() {
        let buf = buf_from(grid_pixels());
        let rgba_ref = oracle_bytes::<false>(&buf.pixels);
        let bgra_ref = oracle_bytes::<true>(&buf.pixels);
        for (label, level) in constructible_levels() {
            let rgba = store_srgb8_f32_at::<false>(level, &buf);
            assert_bytes_close(&rgba, &rgba_ref, &format!("{label} rgba"));
            let bgra = store_srgb8_f32_at::<true>(level, &buf);
            assert_bytes_close(&bgra, &bgra_ref, &format!("{label} bgra"));
        }
    }

    #[test]
    fn simd_tails_match_exact_oracle() {
        for width in 1..=49usize {
            let pixels: Vec<[f32; 4]> = (0..width)
                .map(|i| {
                    let t = i as f32 / width as f32;
                    [t, (t * 0.7).clamp(0.0, 1.0), 1.0 - t, t * t]
                })
                .collect();
            let buf = buf_from(pixels);
            let rgba_ref = oracle_bytes::<false>(&buf.pixels);
            let bgra_ref = oracle_bytes::<true>(&buf.pixels);
            for (label, level) in constructible_levels() {
                let rgba = store_srgb8_f32_at::<false>(level, &buf);
                assert_bytes_close(&rgba, &rgba_ref, &format!("{label} rgba w={width}"));
                let bgra = store_srgb8_f32_at::<true>(level, &buf);
                assert_bytes_close(&bgra, &bgra_ref, &format!("{label} bgra w={width}"));
            }
        }
    }

    /// The 1/2/3-channel scalar production path evaluates the same minimax curve
    /// as the SIMD kernel, so a channel value must encode to the same byte
    /// whichever pixel width it arrives in — a 3-channel and a 4-channel store of
    /// the same image agree exactly.
    #[test]
    fn scalar_sub_four_channel_store_agrees_with_simd() {
        let buf = buf_from(grid_pixels());
        let rgba = store_srgb8_f32_at::<false>(Level::new(), &buf);
        for channels in 1..=3usize {
            let got = store_srgb8_f32(&buf, channels);
            for (i, &byte) in got.iter().enumerate() {
                let (pixel, lane) = (i / channels, i % channels);
                assert_eq!(
                    byte,
                    rgba[pixel * 4 + lane],
                    "channels={channels} pixel {pixel} lane {lane}"
                );
            }
        }
    }

    #[test]
    fn simd_u8_roundtrip_is_exact() {
        let buf = buf_from(u8_roundtrip_pixels());
        for (label, level) in constructible_levels() {
            for bytes in [
                store_srgb8_f32_at::<false>(level, &buf),
                store_srgb8_f32_at::<true>(level, &buf),
            ] {
                for b in 0..=255u8 {
                    let base = b as usize * 4;
                    for (lane, &byte) in bytes[base..base + 4].iter().enumerate() {
                        assert_eq!(byte, b, "{label}: lane {lane} roundtrip for value {b}");
                    }
                }
            }
        }
    }

    /// The `constructible_levels()` loop above routes AVX-512 through the escape
    /// via `_at`; this pins the pure-generic AVX-512 path (which that routing
    /// skips) and re-checks the escape directly. Both 512-bit configurations are
    /// checked against the exact-curve oracle, and against each other
    /// byte-for-byte — the `rsqrt14` + Newton-Raphson escape must not move a
    /// single byte (its ~1 ULP intermediate drift only reaches a byte within
    /// ~1e-7 of a quantization midpoint).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_avx512_generic_and_escape_match_oracle() {
        let Some(avx512) = Level::new().as_avx512() else {
            eprintln!("skipping: no avx512");
            return;
        };
        let level = avx512.level();
        let buf = buf_from(grid_pixels());
        let rgba_ref = oracle_bytes::<false>(&buf.pixels);
        let bgra_ref = oracle_bytes::<true>(&buf.pixels);

        let generic_rgba = store_srgb8_f32_generic_at::<false>(level, &buf);
        let generic_bgra = store_srgb8_f32_generic_at::<true>(level, &buf);
        let escape_rgba = store_srgb8_f32_avx512_escape::<false>(avx512, &buf);
        let escape_bgra = store_srgb8_f32_avx512_escape::<true>(avx512, &buf);

        for (label, got, reference) in [
            ("avx512 generic rgba", &generic_rgba, &rgba_ref),
            ("avx512 generic bgra", &generic_bgra, &bgra_ref),
            ("avx512 escape rgba", &escape_rgba, &rgba_ref),
            ("avx512 escape bgra", &escape_bgra, &bgra_ref),
        ] {
            assert_bytes_close(got, reference, label);
        }
        assert_bytes_eq(&escape_rgba, &generic_rgba, "avx512 escape vs generic rgba");
        assert_bytes_eq(&escape_bgra, &generic_bgra, "avx512 escape vs generic bgra");
    }

    /// Exact u8 round-trip for both 512-bit configurations.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_avx512_generic_and_escape_roundtrip_exact() {
        let Some(avx512) = Level::new().as_avx512() else {
            eprintln!("skipping: no avx512");
            return;
        };
        let level = avx512.level();
        let buf = buf_from(u8_roundtrip_pixels());
        for bytes in [
            store_srgb8_f32_generic_at::<false>(level, &buf),
            store_srgb8_f32_generic_at::<true>(level, &buf),
            store_srgb8_f32_avx512_escape::<false>(avx512, &buf),
            store_srgb8_f32_avx512_escape::<true>(avx512, &buf),
        ] {
            for b in 0..=255u8 {
                let base = b as usize * 4;
                for (lane, &byte) in bytes[base..base + 4].iter().enumerate() {
                    assert_eq!(byte, b, "lane {lane} roundtrip for value {b}");
                }
            }
        }
    }

    // ---- in-place OETF ----

    /// RGB lanes within the documented worst-case error of the exact OETF;
    /// alpha lanes preserved bit-exactly.
    fn assert_encoded_close(got: &[[f32; 4]], orig: &[[f32; 4]]) {
        assert_curve_close(got, orig, oetf_exact_clamped, 1.5e-3, "dispatch");
    }

    /// Every u8 byte value, decoded by the exact EOTF, must land back on the same
    /// byte after the in-place OETF + unorm quantization — the same roundtrip
    /// guarantee the u8 store kernels provide, on every backend.
    #[test]
    fn simd_oetf_in_place_u8_roundtrip_is_exact() {
        for (label, level) in constructible_levels() {
            let mut pixels = u8_roundtrip_pixels();
            srgb_oetf_in_place_f32_at(level, &mut pixels);
            for (b, p) in pixels.iter().enumerate() {
                for (c, &lane) in p.iter().take(3).enumerate() {
                    assert_eq!(
                        (lane * 255.0).round() as u8,
                        b as u8,
                        "{label}: lane {c} roundtrip failed for value {b}"
                    );
                }
            }
        }
    }

    /// The dispatched entry point must agree with the exact OETF for every
    /// pixel count that exercises the SIMD kernels' main loops and tails.
    #[test]
    fn dispatch_matches_exact_across_tail_lengths() {
        let base = curve_test_pixels();
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

    #[test]
    fn simd_oetf_in_place_matches_exact() {
        for (label, level) in constructible_levels() {
            for n in 1..=49usize {
                let orig = in_place_pixels(n);
                let mut got = orig.clone();
                srgb_oetf_in_place_f32_at(level, &mut got);
                assert_curve_close(
                    &got,
                    &orig,
                    oetf_exact_clamped,
                    1.5e-3,
                    &format!("{label} n={n}"),
                );
            }
        }
    }
}
