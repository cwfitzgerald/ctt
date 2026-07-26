//! Packed 32-bit `A2B10G10R10` / `A2R10G10B10` loaders and storers.
//!
//! One little-endian u32 per pixel: three 10-bit channels at bits `[0,10)`,
//! `[10,20)`, `[20,30)` plus a 2-bit alpha at `[30,32)`. G always sits in the
//! middle slot; the channel *order* only decides whether R or B occupies the
//! low slot — `A2B10G10R10` puts R in the low bits, `A2R10G10B10` puts R in
//! the high 10-bit slot. Alpha is always the top 2 bits.
//!
//! UNORM/SNORM use the f32 pipeline; UINT/SINT the u32 pipeline.
//!
//! # Load side
//!
//! The SIMD kernels extract the four bit-fields into planar (SoA) lane vectors
//! using only compile-time-constant shifts against the *fixed* slot positions
//! (0/10/20/30), then choose which slot feeds R vs. B based on `R_SHIFT`. This
//! keeps a single body generic over both channel orders — and, via the
//! `SNORM`/`SINT` const params, over all four numeric types per channel order
//! — without needing generic const shift expressions. Field normalization
//! multiplies by the f32 reciprocal of each divisor (`1/1023`, `1/511`,
//! `1/3`).
//!
//! The UINT/SINT integer paths extract and sign-extend exactly; the
//! UNORM/SNORM loads normalize by a reciprocal multiply (`field · (1/1023)`
//! etc.), which lands within 1 ULP of an exact divide.
//!
//! # Store side
//!
//! The SIMD kernels de-interleave 4/8/16 pixels into planar lane vectors,
//! encode each channel, then pack lane-wise into words at the *fixed* slot
//! positions (0/10/20/30), choosing which slot receives R vs. B from
//! `R_SHIFT`. UNORM/SNORM inputs are clamped to `[0, 1]` / `[-1, 1]`, scaled by
//! the field maximum, and rounded half away from zero via
//! `trunc(x + copysign(0.5, x))`; UINT/SINT inputs are clamped to the field
//! range. NaN input is undefined: the SIMD stores clamp with `min`/`max`, which
//! does not preserve it.

use fearless_simd::{Level, Simd, dispatch, prelude::*};

use crate::error::Result;
use crate::processing::Buffer;
use crate::surface::Surface;

use super::driver;

/// R-channel bit offset for `A2B10G10R10` (R in the low 10-bit slot).
pub const A2B_R_SHIFT: u32 = 0;
/// R-channel bit offset for `A2R10G10B10` (R in the high 10-bit slot).
pub const A2R_R_SHIFT: u32 = 20;

// =====================================================================
// SIMD: f32 pipeline (UNORM / SNORM)
// =====================================================================

/// SIMD kernel behind the UNORM/SNORM loaders, at a caller-chosen [`Level`] so
/// benches can force each backend.
#[doc(hidden)]
pub fn load_a2_f32_at<const R_SHIFT: u32, const SNORM: bool>(
    level: Level,
    surface: &Surface,
) -> Result<Buffer<f32>> {
    dispatch!(level, simd => driver::load_packed32(
        simd,
        surface,
        #[expect(
            clippy::redundant_closure,
            reason = "the closure literal is the force-inline boundary carrying the token's features"
        )]
        #[inline(always)]
        |s, codes, dst| load_row_f32::<_, R_SHIFT, SNORM>(s, codes, dst),
    ))
}

/// Decode `codes` (one packed word per pixel) into `dst` as `[R, G, B, A]`.
#[inline(always)]
fn load_row_f32<S: Simd, const R_SHIFT: u32, const SNORM: bool>(
    simd: S,
    codes: &[u32],
    dst: &mut [f32],
) {
    driver::for_each_block::<_, _, 1, 4>(
        S::f32s::N,
        codes.len(),
        codes,
        dst,
        #[inline(always)]
        |cin, cout| {
            let words = S::u32s::from_slice(simd, cin);
            let (r, g, b, a) = decode_f32::<S, R_SHIFT, SNORM>(words);
            driver::interleave_rgba_f32::<S>(r, g, b, a, cout);
        },
    );
}

/// Split packed words into `(R, G, B, A)` planes and normalize each.
#[inline(always)]
fn decode_f32<S: Simd, const R_SHIFT: u32, const SNORM: bool>(
    words: S::u32s,
) -> (S::f32s, S::f32s, S::f32s, S::f32s) {
    let (r, g, b, a) = extract_slots::<S, R_SHIFT>(words);
    if SNORM {
        (
            decode_snorm10::<S>(r),
            decode_snorm10::<S>(g),
            decode_snorm10::<S>(b),
            decode_snorm2::<S>(a),
        )
    } else {
        (
            decode_unorm::<S>(r, 1.0 / 1023.0),
            decode_unorm::<S>(g, 1.0 / 1023.0),
            decode_unorm::<S>(b, 1.0 / 1023.0),
            decode_unorm::<S>(a, 1.0 / 3.0),
        )
    }
}

/// `field * inv` — normalizes by a reciprocal multiply, ≤1 ULP off an exact
/// divide but far cheaper than a per-lane divide.
#[inline(always)]
fn decode_unorm<S: Simd>(field: S::u32s, inv: f32) -> S::f32s {
    let f: S::f32s = field.to_float();
    f * inv
}

/// Sign-extend a 10-bit field, then `max(-1, s * (1/511))`.
#[inline(always)]
fn decode_snorm10<S: Simd>(field: S::u32s) -> S::f32s {
    let s: S::i32s = sext_bits::<S>(field, 22);
    let f: S::f32s = s.to_float();
    (f * (1.0 / 511.0)).max(-1.0)
}

/// Sign-extend a 2-bit field, then `max(-1, s)`.
#[inline(always)]
fn decode_snorm2<S: Simd>(field: S::u32s) -> S::f32s {
    let s: S::i32s = sext_bits::<S>(field, 30);
    let f: S::f32s = s.to_float();
    f.max(-1.0)
}

// =====================================================================
// SIMD: u32 pipeline (UINT / SINT)
// =====================================================================

/// SIMD kernel behind the UINT/SINT loaders, at a caller-chosen [`Level`] so
/// benches can force each backend.
#[doc(hidden)]
pub fn load_a2_u32_at<const R_SHIFT: u32, const SINT: bool>(
    level: Level,
    surface: &Surface,
) -> Result<Buffer<u32>> {
    dispatch!(level, simd => driver::load_packed32(
        simd,
        surface,
        #[expect(
            clippy::redundant_closure,
            reason = "the closure literal is the force-inline boundary carrying the token's features"
        )]
        #[inline(always)]
        |s, codes, dst| load_row_u32::<_, R_SHIFT, SINT>(s, codes, dst),
    ))
}

/// Decode `codes` (one packed word per pixel) into `dst` as `[R, G, B, A]`.
#[inline(always)]
fn load_row_u32<S: Simd, const R_SHIFT: u32, const SINT: bool>(
    simd: S,
    codes: &[u32],
    dst: &mut [u32],
) {
    driver::for_each_block::<_, _, 1, 4>(
        S::u32s::N,
        codes.len(),
        codes,
        dst,
        #[inline(always)]
        |cin, cout| {
            let words = S::u32s::from_slice(simd, cin);
            let (r, g, b, a) = decode_u32::<S, R_SHIFT, SINT>(words);
            driver::interleave_rgba_u32::<S>(r, g, b, a, cout);
        },
    );
}

/// Split packed words into `(R, G, B, A)` planes, sign-extending for SINT.
#[inline(always)]
fn decode_u32<S: Simd, const R_SHIFT: u32, const SINT: bool>(
    words: S::u32s,
) -> (S::u32s, S::u32s, S::u32s, S::u32s) {
    let (r, g, b, a) = extract_slots::<S, R_SHIFT>(words);
    if SINT {
        (
            sext_bits::<S>(r, 22).bitcast(),
            sext_bits::<S>(g, 22).bitcast(),
            sext_bits::<S>(b, 22).bitcast(),
            sext_bits::<S>(a, 30).bitcast(),
        )
    } else {
        (r, g, b, a)
    }
}

// ---- SIMD shared helpers ----

/// Split a word into channel-ordered `(r, g, b, a)`. G is the middle 10-bit
/// slot; `R_SHIFT` selects which outer 10-bit slot is R, the other is B. The
/// inverse of the store side's field-packing step.
#[inline(always)]
fn extract_slots<S: Simd, const R_SHIFT: u32>(
    words: S::u32s,
) -> (S::u32s, S::u32s, S::u32s, S::u32s) {
    let slot0 = words & 0x3ff;
    let g = (words >> 10) & 0x3ff;
    let slot2 = (words >> 20) & 0x3ff;
    let a = words >> 30;
    let (r, b) = if R_SHIFT == 0 {
        (slot0, slot2)
    } else {
        (slot2, slot0)
    };
    (r, g, b, a)
}

/// Sign-extend the low `32 - shift` bits of each lane (`shift = 22` for 10-bit
/// fields, `30` for the 2-bit alpha), returning the signed result.
#[inline(always)]
fn sext_bits<S: Simd>(field: S::u32s, shift: u32) -> S::i32s {
    let i: S::i32s = field.bitcast();
    (i << shift) >> shift
}

// ---- UNORM (f32 pipeline) ----

pub fn load_a2b10g10r10_unorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2b10g10r10_unorm_f32");
    load_a2_f32_at::<A2B_R_SHIFT, false>(Level::new(), surface)
}

pub fn load_a2r10g10b10_unorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2r10g10b10_unorm_f32");
    load_a2_f32_at::<A2R_R_SHIFT, false>(Level::new(), surface)
}

// ---- SNORM (f32 pipeline) ----

pub fn load_a2b10g10r10_snorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2b10g10r10_snorm_f32");
    load_a2_f32_at::<A2B_R_SHIFT, true>(Level::new(), surface)
}

pub fn load_a2r10g10b10_snorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_a2r10g10b10_snorm_f32");
    load_a2_f32_at::<A2R_R_SHIFT, true>(Level::new(), surface)
}

// ---- UINT (u32 pipeline) ----

pub fn load_a2b10g10r10_uint_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2b10g10r10_uint_u32");
    load_a2_u32_at::<A2B_R_SHIFT, false>(Level::new(), surface)
}

pub fn load_a2r10g10b10_uint_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2r10g10b10_uint_u32");
    load_a2_u32_at::<A2R_R_SHIFT, false>(Level::new(), surface)
}

// ---- SINT (u32 pipeline) ----

pub fn load_a2b10g10r10_sint_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2b10g10r10_sint_u32");
    load_a2_u32_at::<A2B_R_SHIFT, true>(Level::new(), surface)
}

pub fn load_a2r10g10b10_sint_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_a2r10g10b10_sint_u32");
    load_a2_u32_at::<A2R_R_SHIFT, true>(Level::new(), surface)
}

// =====================================================================
// SIMD store: f32 pipeline (UNORM / SNORM)
// =====================================================================

/// SIMD kernel behind the UNORM/SNORM storers, at a caller-chosen [`Level`] so
/// benches can force each backend.
#[doc(hidden)]
pub fn store_a2_f32_at<const R_SHIFT: u32, const SNORM: bool>(
    level: Level,
    buf: &Buffer<f32>,
) -> Vec<u8> {
    let src: &[f32] = bytemuck::cast_slice(&buf.pixels);
    dispatch!(level, simd => driver::store_packed32(
        simd,
        src,
        #[expect(
            clippy::redundant_closure,
            reason = "the closure literal is the force-inline boundary carrying the token's features"
        )]
        #[inline(always)]
        |s, src, words| store_rows_f32::<_, R_SHIFT, SNORM>(s, src, words),
    ))
}

/// Encode every pixel of `src` (4 f32 lanes each) into one packed word.
#[inline(always)]
fn store_rows_f32<S: Simd, const R_SHIFT: u32, const SNORM: bool>(
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
            let block = encode_block_f32::<S, R_SHIFT, SNORM>(simd, cin);
            block.store_slice(cout);
        },
    );
}

#[inline(always)]
fn encode_block_f32<S: Simd, const R_SHIFT: u32, const SNORM: bool>(
    simd: S,
    src: &[f32],
) -> S::u32s {
    let (r, g, b, a) = driver::split_rgba_f32(simd, src);

    let (rgb_scale, a_scale) = if SNORM { (511.0, 1.0) } else { (1023.0, 3.0) };
    let ri = encode_f32_field::<S, SNORM>(simd, r, rgb_scale, 0x3ff);
    let gi = encode_f32_field::<S, SNORM>(simd, g, rgb_scale, 0x3ff);
    let bi = encode_f32_field::<S, SNORM>(simd, b, rgb_scale, 0x3ff);
    let ai = encode_f32_field::<S, SNORM>(simd, a, a_scale, 0x3);
    pack_lanes::<S, R_SHIFT>(ri, gi, bi, ai)
}

/// Clamp to `[-1, 1]` (SNORM) or `[0, 1]` (UNORM), scale, round half away
/// from zero, and mask into a packed field. The scale and `±0.5` bias stay an
/// unfused mul-then-add, so the rounded value is exactly `trunc(x·scale ± 0.5)`
/// — fusing them would round the product first and shift ties.
#[inline(always)]
fn encode_f32_field<S: Simd, const SNORM: bool>(
    simd: S,
    x: S::f32s,
    scale: f32,
    mask: u32,
) -> S::u32s {
    let clamped = if SNORM {
        x.max(S::f32s::splat(simd, -1.0))
            .min(S::f32s::splat(simd, 1.0))
    } else {
        driver::clamp01(simd, x)
    };
    let rounded: S::i32s = if SNORM {
        // Round half away from zero: bias by ±0.5 matching the sign.
        let scaled = clamped * scale;
        let bias = S::f32s::splat(simd, 0.5).copysign(scaled);
        (scaled + bias).to_int()
    } else {
        // Clamped is non-negative, so the bias is a plain `+0.5`.
        (clamped * scale + 0.5).to_int()
    };
    let bits: S::u32s = rounded.bitcast();
    bits & mask
}

// =====================================================================
// SIMD store: u32 pipeline (UINT / SINT)
// =====================================================================

/// SIMD kernel behind the UINT/SINT storers, at a caller-chosen [`Level`] so
/// benches can force each backend.
#[doc(hidden)]
pub fn store_a2_u32_at<const R_SHIFT: u32, const SINT: bool>(
    level: Level,
    buf: &Buffer<u32>,
) -> Vec<u8> {
    let src: &[u32] = bytemuck::cast_slice(&buf.pixels);
    dispatch!(level, simd => driver::store_packed32(
        simd,
        src,
        #[expect(
            clippy::redundant_closure,
            reason = "the closure literal is the force-inline boundary carrying the token's features"
        )]
        #[inline(always)]
        |s, src, words| store_rows_u32::<_, R_SHIFT, SINT>(s, src, words),
    ))
}

/// Encode every pixel of `src` (4 u32 lanes each) into one packed word.
#[inline(always)]
fn store_rows_u32<S: Simd, const R_SHIFT: u32, const SINT: bool>(
    simd: S,
    src: &[u32],
    words: &mut [u32],
) {
    driver::for_each_block::<_, _, 4, 1>(
        S::u32s::N,
        words.len(),
        src,
        words,
        #[inline(always)]
        |cin, cout| {
            let block = encode_block_u32::<S, R_SHIFT, SINT>(simd, cin);
            block.store_slice(cout);
        },
    );
}

#[inline(always)]
fn encode_block_u32<S: Simd, const R_SHIFT: u32, const SINT: bool>(
    simd: S,
    src: &[u32],
) -> S::u32s {
    let (r, g, b, a) = driver::split_rgba_u32(simd, src);

    let (ri, gi, bi, ai) = if SINT {
        (
            clamp_sint::<S>(r, -512, 511, 0x3ff),
            clamp_sint::<S>(g, -512, 511, 0x3ff),
            clamp_sint::<S>(b, -512, 511, 0x3ff),
            clamp_sint::<S>(a, -2, 1, 0x3),
        )
    } else {
        (
            r.min(0x3ffu32),
            g.min(0x3ffu32),
            b.min(0x3ffu32),
            a.min(0x3u32),
        )
    };
    pack_lanes::<S, R_SHIFT>(ri, gi, bi, ai)
}

/// Signed-clamp a raw u32 plane to `[lo, hi]` and mask into a packed field.
#[inline(always)]
fn clamp_sint<S: Simd>(x: S::u32s, lo: i32, hi: i32, mask: u32) -> S::u32s {
    let xi: S::i32s = x.bitcast();
    let clamped = xi.min(hi).max(lo);
    let bits: S::u32s = clamped.bitcast();
    bits & mask
}

// ---- SIMD store shared helpers ----

/// Pack four masked planar field vectors into one word per pixel. The inverse
/// of the load side's slot-extraction step.
#[inline(always)]
fn pack_lanes<S: Simd, const R_SHIFT: u32>(
    r: S::u32s,
    g: S::u32s,
    b: S::u32s,
    a: S::u32s,
) -> S::u32s {
    let (low, high) = if R_SHIFT == 0 { (r, b) } else { (b, r) };
    (a << 30) | (high << 20) | (g << 10) | low
}

// ---- UNORM stores (f32 pipeline) ----

pub fn store_a2b10g10r10_unorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_a2b10g10r10_unorm_f32");
    store_a2_f32_at::<A2B_R_SHIFT, false>(Level::new(), buf)
}

pub fn store_a2r10g10b10_unorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_a2r10g10b10_unorm_f32");
    store_a2_f32_at::<A2R_R_SHIFT, false>(Level::new(), buf)
}

// ---- SNORM stores (f32 pipeline) ----

pub fn store_a2b10g10r10_snorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_a2b10g10r10_snorm_f32");
    store_a2_f32_at::<A2B_R_SHIFT, true>(Level::new(), buf)
}

pub fn store_a2r10g10b10_snorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_a2r10g10b10_snorm_f32");
    store_a2_f32_at::<A2R_R_SHIFT, true>(Level::new(), buf)
}

// ---- UINT stores (u32 pipeline) ----

pub fn store_a2b10g10r10_uint_u32(buf: &Buffer<u32>) -> Vec<u8> {
    profiling::scope!("store_a2b10g10r10_uint_u32");
    store_a2_u32_at::<A2B_R_SHIFT, false>(Level::new(), buf)
}

pub fn store_a2r10g10b10_uint_u32(buf: &Buffer<u32>) -> Vec<u8> {
    profiling::scope!("store_a2r10g10b10_uint_u32");
    store_a2_u32_at::<A2R_R_SHIFT, false>(Level::new(), buf)
}

// ---- SINT stores (u32 pipeline) ----

pub fn store_a2b10g10r10_sint_u32(buf: &Buffer<u32>) -> Vec<u8> {
    profiling::scope!("store_a2b10g10r10_sint_u32");
    store_a2_u32_at::<A2B_R_SHIFT, true>(Level::new(), buf)
}

pub fn store_a2r10g10b10_sint_u32(buf: &Buffer<u32>) -> Vec<u8> {
    profiling::scope!("store_a2r10g10b10_sint_u32");
    store_a2_u32_at::<A2R_R_SHIFT, true>(Level::new(), buf)
}

#[cfg(test)]
mod load_tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::processing::kernels::constructible_levels;
    use crate::surface::ColorSpace;

    fn a2_surface(data: Vec<u8>, width: u32, height: u32, stride: u32) -> Surface {
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::A2B10G10R10_UNORM_PACK32,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        }
    }

    // ---- Per-pixel oracles: closed-form field extraction and normalization ----

    /// Split a packed word into `(r, g, b, a)` raw fields for the given R
    /// offset. The B channel takes whichever 10-bit slot R does not.
    fn extract<const R_SHIFT: u32>(word: u32) -> (u32, u32, u32, u32) {
        let r = (word >> R_SHIFT) & 0x3ff;
        let g = (word >> 10) & 0x3ff;
        let b = (word >> (20 - R_SHIFT)) & 0x3ff;
        let a = (word >> 30) & 0x3;
        (r, g, b, a)
    }

    /// Sign-extend a 10-bit two's-complement field, then `max(x / 511, -1)`.
    fn snorm10(v: u32) -> f32 {
        let s = ((v as i32) << 22) >> 22;
        (s as f32 / 511.0).max(-1.0)
    }

    /// Sign-extend a 2-bit two's-complement field, then `max(x, -1)`.
    fn snorm2(v: u32) -> f32 {
        let s = ((v as i32) << 30) >> 30;
        (s as f32).max(-1.0)
    }

    /// Sign-extend a 10-bit two's-complement field into the low bits of a u32.
    fn sext10(v: u32) -> u32 {
        (((v as i32) << 22) >> 22) as u32
    }

    /// Sign-extend a 2-bit two's-complement field into the low bits of a u32.
    fn sext2(v: u32) -> u32 {
        (((v as i32) << 30) >> 30) as u32
    }

    /// UNORM oracle: exact divide by each field's maximum.
    fn oracle_unorm<const R_SHIFT: u32>(word: u32) -> [f32; 4] {
        let (r, g, b, a) = extract::<R_SHIFT>(word);
        [
            r as f32 / 1023.0,
            g as f32 / 1023.0,
            b as f32 / 1023.0,
            a as f32 / 3.0,
        ]
    }

    /// SNORM oracle: sign-extend, exact divide, clamp the `-max` code to -1.
    fn oracle_snorm<const R_SHIFT: u32>(word: u32) -> [f32; 4] {
        let (r, g, b, a) = extract::<R_SHIFT>(word);
        [snorm10(r), snorm10(g), snorm10(b), snorm2(a)]
    }

    /// UINT oracle: the raw fields.
    fn oracle_uint<const R_SHIFT: u32>(word: u32) -> [u32; 4] {
        let (r, g, b, a) = extract::<R_SHIFT>(word);
        [r, g, b, a]
    }

    /// SINT oracle: the sign-extended fields.
    fn oracle_sint<const R_SHIFT: u32>(word: u32) -> [u32; 4] {
        let (r, g, b, a) = extract::<R_SHIFT>(word);
        [sext10(r), sext10(g), sext10(b), sext2(a)]
    }

    /// Walk a surface's rows at its stride, decoding each word with `f` and
    /// ignoring any inter-row padding.
    fn oracle_pixels<T, F: Fn(u32) -> [T; 4]>(surface: &Surface, f: F) -> Vec<[T; 4]> {
        let w = surface.width as usize;
        let stride = surface.stride as usize;
        let mut out = Vec::with_capacity(w * surface.height as usize);
        for y in 0..surface.height as usize {
            let row = &surface.data[y * stride..y * stride + w * 4];
            for chunk in row.chunks_exact(4) {
                out.push(f(u32::from_le_bytes(chunk.try_into().unwrap())));
            }
        }
        out
    }

    /// Hand-computed field values, pinning the oracles independently of the
    /// bit-twiddling they share: UNORM spans `0 → 0.0` to `1023 → 1.0` with the
    /// midpoint code at `512/1023`, alpha spans `0..=3 → 0, 1/3, 2/3, 1`; SNORM
    /// reads `511 → 1.0`, the `-512` code clamps to `-1.0`, and the 2-bit alpha
    /// `0b10` (= -2) clamps to `-1.0`; SINT sign-extends `0x3ff` to `-1`.
    #[test]
    fn oracles_match_known_values() {
        // A2B10G10R10: R low, then G, then B, alpha on top.
        let word = |r: u32, g: u32, b: u32, a: u32| (a << 30) | (b << 20) | (g << 10) | r;

        assert_eq!(
            oracle_unorm::<A2B_R_SHIFT>(word(0, 1023, 512, 1)),
            [0.0, 1.0, 512.0 / 1023.0, 1.0 / 3.0]
        );
        assert_eq!(
            oracle_unorm::<A2B_R_SHIFT>(word(1023, 0, 1, 3)),
            [1.0, 0.0, 1.0 / 1023.0, 1.0]
        );
        assert_eq!(
            oracle_snorm::<A2B_R_SHIFT>(word(511, 512, 1023, 1)),
            [1.0, -1.0, -1.0 / 511.0, 1.0]
        );
        assert_eq!(
            oracle_snorm::<A2B_R_SHIFT>(word(0, 256, 768, 2)),
            [0.0, 256.0 / 511.0, -256.0 / 511.0, -1.0]
        );
        assert_eq!(
            oracle_uint::<A2B_R_SHIFT>(word(0, 1023, 512, 3)),
            [0, 1023, 512, 3]
        );
        assert_eq!(
            oracle_sint::<A2B_R_SHIFT>(word(1023, 512, 511, 3)),
            [(-1i32) as u32, (-512i32) as u32, 511, (-1i32) as u32]
        );

        // A2R10G10B10 puts R in the high 10-bit slot and B in the low one.
        let swapped = word(0, 1023, 512, 0); // low slot 0, mid 1023, high 512
        assert_eq!(
            oracle_unorm::<A2R_R_SHIFT>(swapped),
            [512.0 / 1023.0, 1.0, 0.0, 0.0]
        );
    }

    fn pack_word<const R_SHIFT: u32>(r: u32, g: u32, b: u32, a: u32) -> [u8; 4] {
        let word = (a << 30) | (b << (20 - R_SHIFT)) | (g << 10) | (r << R_SHIFT);
        word.to_le_bytes()
    }

    /// Every 10-bit value in the R slot and (separately) the B slot, crossed
    /// with all four alpha values, plus a varying mid slot.
    fn full_domain_words<const R_SHIFT: u32>() -> Vec<u8> {
        let mut data = Vec::new();
        for field in 0u32..1024 {
            for a in 0u32..4 {
                data.extend_from_slice(&pack_word::<R_SHIFT>(
                    field,
                    (field ^ 0x155) & 0x3ff,
                    512,
                    a,
                ));
                data.extend_from_slice(&pack_word::<R_SHIFT>(
                    511,
                    (field ^ 0x2aa) & 0x3ff,
                    field,
                    a,
                ));
            }
        }
        data
    }

    /// Odd width (exercises every tail residue) with junk stride padding.
    fn tail_stride_surface<const R_SHIFT: u32>(width: u32) -> Surface {
        let h = 3u32;
        let row_bytes = width * 4;
        let stride = row_bytes + 12;
        let mut data = vec![0u8; (stride * h) as usize];
        for y in 0..h {
            for xp in 0..width {
                let seed = y.wrapping_mul(9973).wrapping_add(xp.wrapping_mul(7));
                let r = seed & 0x3ff;
                let g = seed.wrapping_mul(3) & 0x3ff;
                let b = seed.wrapping_mul(5) & 0x3ff;
                let a = seed & 0x3;
                let off = (y * stride + xp * 4) as usize;
                data[off..off + 4].copy_from_slice(&pack_word::<R_SHIFT>(r, g, b, a));
            }
            let pad_off = (y * stride + row_bytes) as usize;
            for byte in &mut data[pad_off..pad_off + 12] {
                *byte = 0xAB;
            }
        }
        a2_surface(data, width, h, stride)
    }

    /// Signed ULP distance between two finite f32s (0 for equal, `±0.0` equal).
    fn ulp_diff(a: f32, b: f32) -> u64 {
        let ordered = |f: f32| -> i64 {
            let mag = (f.to_bits() & 0x7fff_ffff) as i64;
            if f.is_sign_negative() { -mag } else { mag }
        };
        (ordered(a) - ordered(b)).unsigned_abs()
    }

    /// The reciprocal-multiply normalization drifts at most 1 ULP off the
    /// oracle's exact divide, so the decoded floats are compared to ≤1 ULP
    /// rather than bit-exact.
    fn check_load_f32<const R_SHIFT: u32, const SNORM: bool>(surface: &Surface, label: &str) {
        let want = if SNORM {
            oracle_pixels(surface, oracle_snorm::<R_SHIFT>)
        } else {
            oracle_pixels(surface, oracle_unorm::<R_SHIFT>)
        };
        for (name, level) in constructible_levels() {
            let got = load_a2_f32_at::<R_SHIFT, SNORM>(level, surface).unwrap();
            assert_eq!(got.pixels.len(), want.len(), "{label} {name} count");
            for (i, (g, w)) in got.pixels.iter().zip(&want).enumerate() {
                for c in 0..4 {
                    assert!(
                        ulp_diff(g[c], w[c]) <= 1,
                        "{label} {name}: pixel {i} lane {c}: got {} want {}",
                        g[c],
                        w[c],
                    );
                }
            }
        }
    }

    fn check_load_u32<const R_SHIFT: u32, const SINT: bool>(surface: &Surface, label: &str) {
        let want = if SINT {
            oracle_pixels(surface, oracle_sint::<R_SHIFT>)
        } else {
            oracle_pixels(surface, oracle_uint::<R_SHIFT>)
        };
        for (name, level) in constructible_levels() {
            let got = load_a2_u32_at::<R_SHIFT, SINT>(level, surface).unwrap();
            assert_eq!(got.pixels, want, "{label} {name}");
        }
    }

    #[test]
    fn load_full_domain_matches_oracle() {
        let a2b = full_domain_words::<A2B_R_SHIFT>();
        let count = (a2b.len() / 4) as u32;
        let surf = a2_surface(a2b, count, 1, count * 4);
        check_load_f32::<A2B_R_SHIFT, false>(&surf, "a2b unorm");
        check_load_f32::<A2B_R_SHIFT, true>(&surf, "a2b snorm");
        check_load_u32::<A2B_R_SHIFT, false>(&surf, "a2b uint");
        check_load_u32::<A2B_R_SHIFT, true>(&surf, "a2b sint");

        let a2r = full_domain_words::<A2R_R_SHIFT>();
        let surf = a2_surface(a2r, count, 1, count * 4);
        check_load_f32::<A2R_R_SHIFT, false>(&surf, "a2r unorm");
        check_load_f32::<A2R_R_SHIFT, true>(&surf, "a2r snorm");
        check_load_u32::<A2R_R_SHIFT, false>(&surf, "a2r uint");
        check_load_u32::<A2R_R_SHIFT, true>(&surf, "a2r sint");
    }

    #[test]
    fn load_tails_matches_oracle() {
        for width in 1..=49u32 {
            let surf = tail_stride_surface::<A2B_R_SHIFT>(width);
            check_load_f32::<A2B_R_SHIFT, false>(&surf, "a2b unorm");
            check_load_f32::<A2B_R_SHIFT, true>(&surf, "a2b snorm");
            check_load_u32::<A2B_R_SHIFT, false>(&surf, "a2b uint");
            check_load_u32::<A2B_R_SHIFT, true>(&surf, "a2b sint");

            let surf = tail_stride_surface::<A2R_R_SHIFT>(width);
            check_load_f32::<A2R_R_SHIFT, false>(&surf, "a2r unorm");
            check_load_f32::<A2R_R_SHIFT, true>(&surf, "a2r snorm");
            check_load_u32::<A2R_R_SHIFT, false>(&surf, "a2r uint");
            check_load_u32::<A2R_R_SHIFT, true>(&surf, "a2r sint");
        }
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use crate::processing::kernels::constructible_levels;

    // ---- Per-pixel oracles: clamp, scale, round, pack ----

    /// Pack raw `(r, g, b, a)` fields (already masked to their bit widths) into
    /// a word for the given R offset.
    fn pack<const R_SHIFT: u32>(r: u32, g: u32, b: u32, a: u32) -> u32 {
        (a << 30) | (b << (20 - R_SHIFT)) | (g << 10) | (r << R_SHIFT)
    }

    /// UNORM oracle: clamp to `[0, 1]`, scale by the field maximum, round half
    /// away from zero (`f32::round`), mask.
    fn oracle_unorm_word<const R_SHIFT: u32>(lanes: &[f32; 4]) -> u32 {
        let field = |v: f32| (v.clamp(0.0, 1.0) * 1023.0).round() as u32 & 0x3ff;
        pack::<R_SHIFT>(
            field(lanes[0]),
            field(lanes[1]),
            field(lanes[2]),
            (lanes[3].clamp(0.0, 1.0) * 3.0).round() as u32 & 0x3,
        )
    }

    /// SNORM oracle: clamp to `[-1, 1]`, scale, round half away from zero, mask
    /// the two's-complement result. The 2-bit alpha's scale factor is 1.
    fn oracle_snorm_word<const R_SHIFT: u32>(lanes: &[f32; 4]) -> u32 {
        let field = |v: f32| ((v.clamp(-1.0, 1.0) * 511.0).round() as i32) as u32 & 0x3ff;
        pack::<R_SHIFT>(
            field(lanes[0]),
            field(lanes[1]),
            field(lanes[2]),
            (lanes[3].clamp(-1.0, 1.0).round() as i32) as u32 & 0x3,
        )
    }

    /// UINT oracle: unsigned clamp to each field maximum.
    fn oracle_uint_word<const R_SHIFT: u32>(lanes: &[u32; 4]) -> u32 {
        pack::<R_SHIFT>(
            lanes[0].min(0x3ff),
            lanes[1].min(0x3ff),
            lanes[2].min(0x3ff),
            lanes[3].min(0x3),
        )
    }

    /// SINT oracle: signed clamp to each field range, then mask.
    fn oracle_sint_word<const R_SHIFT: u32>(lanes: &[u32; 4]) -> u32 {
        let field = |v: u32| ((v as i32).clamp(-512, 511)) as u32 & 0x3ff;
        pack::<R_SHIFT>(
            field(lanes[0]),
            field(lanes[1]),
            field(lanes[2]),
            ((lanes[3] as i32).clamp(-2, 1)) as u32 & 0x3,
        )
    }

    fn oracle_bytes<T, F: Fn(&[T; 4]) -> u32>(pixels: &[[T; 4]], f: F) -> Vec<u8> {
        pixels.iter().flat_map(|p| f(p).to_le_bytes()).collect()
    }

    /// Hand-computed field encodings, pinning the oracles independently of the
    /// packing they share. Halfway inputs round away from zero (`0.5 · 1023 =
    /// 511.5 → 512`), out-of-range inputs saturate, and negative SNORM fields
    /// come back as 10-bit two's complement (`-511 → 0x201`, `-256 → 0x300`).
    #[test]
    fn oracles_match_known_values() {
        let fields = |word: u32| {
            (
                word & 0x3ff,
                (word >> 10) & 0x3ff,
                (word >> 20) & 0x3ff,
                word >> 30,
            )
        };

        assert_eq!(
            fields(oracle_unorm_word::<A2B_R_SHIFT>(&[0.0, 1.0, 0.5, 1.0])),
            (0, 1023, 512, 3)
        );
        assert_eq!(
            fields(oracle_unorm_word::<A2B_R_SHIFT>(&[
                -1.0,
                2.0,
                1.0 / 1023.0,
                0.5
            ])),
            (0, 1023, 1, 2) // alpha 0.5·3 = 1.5 → 2
        );
        assert_eq!(
            fields(oracle_snorm_word::<A2B_R_SHIFT>(&[1.0, -1.0, 0.5, 1.0])),
            (511, 0x201, 256, 1)
        );
        assert_eq!(
            fields(oracle_snorm_word::<A2B_R_SHIFT>(&[-0.5, -2.0, 2.0, -1.0])),
            (0x300, 0x201, 511, 0x3)
        );
        assert_eq!(
            fields(oracle_uint_word::<A2B_R_SHIFT>(&[0, 1023, 1024, 4])),
            (0, 1023, 1023, 3)
        );
        assert_eq!(
            fields(oracle_sint_word::<A2B_R_SHIFT>(&[
                600,
                (-600i32) as u32,
                (-1i32) as u32,
                (-3i32) as u32
            ])),
            (511, 0x200, 0x3ff, 0x2)
        );

        // A2R10G10B10 swaps which outer slot receives R.
        let word = oracle_unorm_word::<A2R_R_SHIFT>(&[1.0, 0.0, 0.5, 0.0]);
        assert_eq!(fields(word), (512, 0, 1023, 0));
    }

    fn buf_f32(pixels: Vec<[f32; 4]>) -> Buffer<f32> {
        let width = pixels.len().max(1) as u32;
        Buffer {
            pixels,
            width,
            height: 1,
        }
    }

    fn buf_u32(pixels: Vec<[u32; 4]>) -> Buffer<u32> {
        let width = pixels.len().max(1) as u32;
        Buffer {
            pixels,
            width,
            height: 1,
        }
    }

    /// Halfway points for both scales, out-of-range clamps, and the ±Inf / min /
    /// max edge inputs, which the `min`/`max` clamp saturates.
    fn f32_store_pixels() -> Vec<[f32; 4]> {
        let mut pixels = Vec::new();
        for k in 0..1023u32 {
            let v = (k as f32 + 0.5) / 1023.0;
            pixels.push([v, v, v, ((k % 3) as f32 + 0.5) / 3.0]);
        }
        for k in 0..511u32 {
            let v = (k as f32 + 0.5) / 511.0;
            pixels.push([v, -v, v, 0.5]);
            pixels.push([-v, v, -v, -0.5]);
        }
        for &v in &[
            -2.0f32,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            f32::MIN,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            pixels.push([v, v, v, v]);
        }
        pixels
    }

    fn u32_store_pixels() -> Vec<[u32; 4]> {
        let mut pixels = Vec::new();
        for field in 0u32..1024 {
            for a in 0u32..4 {
                pixels.push([field, (field ^ 0x155) & 0xffff, field.wrapping_add(700), a]);
            }
        }
        for &v in &[1023u32, 1024, 2000, 0xffff_ffff, 0x8000_0000, 512, 511] {
            pixels.push([v, v, v, v & 0xf]);
        }
        // Exact sint clamp boundaries as i32 bit patterns: 10-bit RGB at the
        // in-range extremes (±512, 511) and their just-out-of-range neighbors
        // (-513, 512); 2-bit alpha at ±2, 1 and their neighbors (-3, 2).
        for &(rgb, a) in &[
            ((-512i32) as u32, (-2i32) as u32),
            ((-513i32) as u32, (-3i32) as u32),
            (511u32, 1u32),
            (512u32, 2u32),
        ] {
            pixels.push([rgb, rgb, rgb, a]);
        }
        pixels
    }

    /// The store keeps the scale and round bias unfused, so it is bit-exact with
    /// the oracle's round for finite inputs on every tier.
    fn check_store_f32<const R_SHIFT: u32, const SNORM: bool>(buf: &Buffer<f32>, label: &str) {
        let want = if SNORM {
            oracle_bytes(&buf.pixels, oracle_snorm_word::<R_SHIFT>)
        } else {
            oracle_bytes(&buf.pixels, oracle_unorm_word::<R_SHIFT>)
        };
        for (name, level) in constructible_levels() {
            let got = store_a2_f32_at::<R_SHIFT, SNORM>(level, buf);
            assert_eq!(got, want, "{label} {name}");
        }
    }

    fn check_store_u32<const R_SHIFT: u32, const SINT: bool>(buf: &Buffer<u32>, label: &str) {
        let want = if SINT {
            oracle_bytes(&buf.pixels, oracle_sint_word::<R_SHIFT>)
        } else {
            oracle_bytes(&buf.pixels, oracle_uint_word::<R_SHIFT>)
        };
        for (name, level) in constructible_levels() {
            let got = store_a2_u32_at::<R_SHIFT, SINT>(level, buf);
            assert_eq!(got, want, "{label} {name}");
        }
    }

    #[test]
    fn store_sweep_matches_oracle() {
        let fbuf = buf_f32(f32_store_pixels());
        check_store_f32::<A2B_R_SHIFT, false>(&fbuf, "a2b unorm");
        check_store_f32::<A2B_R_SHIFT, true>(&fbuf, "a2b snorm");
        check_store_f32::<A2R_R_SHIFT, false>(&fbuf, "a2r unorm");
        check_store_f32::<A2R_R_SHIFT, true>(&fbuf, "a2r snorm");

        let ubuf = buf_u32(u32_store_pixels());
        check_store_u32::<A2B_R_SHIFT, false>(&ubuf, "a2b uint");
        check_store_u32::<A2B_R_SHIFT, true>(&ubuf, "a2b sint");
        check_store_u32::<A2R_R_SHIFT, false>(&ubuf, "a2r uint");
        check_store_u32::<A2R_R_SHIFT, true>(&ubuf, "a2r sint");
    }

    #[test]
    fn store_tails_matches_oracle() {
        for count in 1..=49usize {
            let f: Vec<[f32; 4]> = (0..count)
                .map(|i| {
                    let x = i as f32;
                    [x * 0.013, 1.0 - x * 0.017, x * 0.031, (x * 0.007) % 1.0]
                })
                .collect();
            let fbuf = buf_f32(f);
            check_store_f32::<A2B_R_SHIFT, false>(&fbuf, "a2b unorm");
            check_store_f32::<A2B_R_SHIFT, true>(&fbuf, "a2b snorm");
            check_store_f32::<A2R_R_SHIFT, false>(&fbuf, "a2r unorm");
            check_store_f32::<A2R_R_SHIFT, true>(&fbuf, "a2r snorm");

            let u: Vec<[u32; 4]> = (0..count as u32)
                .map(|i| [i * 37, i * 101 + 7, i * 211 + 3, i])
                .collect();
            let ubuf = buf_u32(u);
            check_store_u32::<A2B_R_SHIFT, false>(&ubuf, "a2b uint");
            check_store_u32::<A2B_R_SHIFT, true>(&ubuf, "a2b sint");
            check_store_u32::<A2R_R_SHIFT, false>(&ubuf, "a2r uint");
            check_store_u32::<A2R_R_SHIFT, true>(&ubuf, "a2r sint");
        }
    }
}
