//! `E5B9G9R9_UFLOAT_PACK32` loader and storer (shared-exponent RGB).
//!
//! One little-endian u32 per pixel: three 9-bit mantissas (R at bits `[0,9)`,
//! G at `[9,18)`, B at `[18,27)`) sharing a single 5-bit exponent at `[27,32)`.
//! There is no alpha channel and no sign bit — every channel is
//! `mantissa * 2^(exp - 24)` (bias 15, 9 mantissa bits). Alpha loads as 1.0 and
//! is dropped on store.
//!
//! # Load side
//!
//! The SIMD kernel is width-generic at the backend's native vector width
//! (128-bit on Fallback/SSE4.2, 256-bit on AVX2, 512-bit on AVX-512) and is
//! exact at every width: `2^(exp - 24)` is the power of two it represents
//! exactly, so scaling each mantissa by it rounds the same real value the
//! closed-form `mantissa * 2^(exp - 24)` does, even into subnormal results.
//!
//! # Store side
//!
//! The store follows the Khronos / OpenGL shared-exponent algorithm: clamp to
//! the representable range, pick the shared exponent from the largest channel,
//! then round each channel's mantissa against that exponent.
//!
//! It too is width-generic and exact at every width: `inv_denom` is the exact
//! reciprocal of `denom` (both normal f32s, `inv_denom` in `[2^-8, 2^24]`), so
//! `channel * inv_denom` equals `channel / denom` bit-for-bit, and the mantissa
//! rounds by `+ 0.5` then truncation.

use fearless_simd::{Level, Simd, dispatch, prelude::*};

use crate::error::Result;
use crate::processing::Buffer;
use crate::surface::Surface;

use super::driver;

pub fn load_e5b9g9r9_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_e5b9g9r9_f32");
    load_e5b9g9r9_f32_at(Level::new(), surface)
}

/// SIMD kernel behind [`load_e5b9g9r9_f32`], at a caller-chosen [`Level`] so
/// benches can force each backend.
#[doc(hidden)]
pub fn load_e5b9g9r9_f32_at(level: Level, surface: &Surface) -> Result<Buffer<f32>> {
    dispatch!(level, simd => driver::load_packed32(
        simd,
        surface,
        #[expect(
            clippy::redundant_closure,
            reason = "the closure literal is the force-inline boundary carrying the token's features"
        )]
        #[inline(always)]
        |s, codes, dst| load_row(s, codes, dst),
    ))
}

/// Decode `codes` (one packed word per pixel) into `dst` as `[R, G, B, 1.0]`.
#[inline(always)]
fn load_row<S: Simd>(simd: S, codes: &[u32], dst: &mut [f32]) {
    driver::for_each_block::<_, _, 1, 4>(
        S::u32s::N,
        codes.len(),
        codes,
        dst,
        #[inline(always)]
        |cin, cout| {
            let words = S::u32s::from_slice(simd, cin);
            let (r, g, b) = decode_words::<S>(words);
            let one = S::f32s::splat(simd, 1.0);
            driver::interleave_rgba_f32::<S>(r, g, b, one, cout);
        },
    );
}

/// Split packed words into the three 9-bit mantissas and scale each by the
/// shared `2^(exp - 24)`. The scale is an exact power of two, so the product is
/// exact.
#[inline(always)]
fn decode_words<S: Simd>(words: S::u32s) -> (S::f32s, S::f32s, S::f32s) {
    let r = words & 0x1ff;
    let g = (words >> 9) & 0x1ff;
    let b = (words >> 18) & 0x1ff;
    let exp = words >> 27;
    // scale = 2^(exp - 24): IEEE exponent field exp + 103.
    let scale: S::f32s = ((exp + 103) << 23).bitcast();
    let rf: S::f32s = r.to_float();
    let gf: S::f32s = g.to_float();
    let bf: S::f32s = b.to_float();
    (rf * scale, gf * scale, bf * scale)
}

// ---- linear-f32 → shared-exponent SIMD store ----

const MANTISSA_BITS: i32 = 9;
const EXP_BIAS: i32 = 15;
const MAX_EXP: i32 = 31;

/// Largest finite value representable: `(2^9 - 1) / 2^9 * 2^(31 - 15)`.
const SHARED_EXP_MAX: f32 = ((1i32 << MANTISSA_BITS) - 1) as f32 / (1i32 << MANTISSA_BITS) as f32
    * (1u32 << (MAX_EXP - EXP_BIAS)) as f32;

pub fn store_e5b9g9r9_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_e5b9g9r9_f32");
    store_e5b9g9r9_f32_at(Level::new(), buf)
}

/// SIMD kernel behind [`store_e5b9g9r9_f32`], at a caller-chosen [`Level`] so
/// benches can force each backend.
#[doc(hidden)]
pub fn store_e5b9g9r9_f32_at(level: Level, buf: &Buffer<f32>) -> Vec<u8> {
    let src: &[f32] = bytemuck::cast_slice(&buf.pixels);
    dispatch!(level, simd => driver::store_packed32(
        simd,
        src,
        #[expect(
            clippy::redundant_closure,
            reason = "the closure literal is the force-inline boundary carrying the token's features"
        )]
        #[inline(always)]
        |s, src, words| store_rows(s, src, words),
    ))
}

/// Encode every pixel of `src` (4 f32 each) into one packed `u32` word each.
#[inline(always)]
fn store_rows<S: Simd>(simd: S, src: &[f32], words: &mut [u32]) {
    driver::for_each_block::<_, _, 4, 1>(
        S::f32s::N,
        words.len(),
        src,
        words,
        #[inline(always)]
        |cin, cout| {
            let block = encode_block(simd, cin);
            block.store_slice(cout);
        },
    );
}

/// Encode `n` interleaved `[R, G, B, A]` pixels into `n` packed words. Alpha is
/// dropped. `inv_denom` is the exact reciprocal of the spec's `denom` and both
/// are normal f32s, so `channel * inv_denom` equals `channel / denom`
/// bit-for-bit, and the mantissa rounds by `+ 0.5` then truncation.
#[inline(always)]
fn encode_block<S: Simd>(simd: S, src: &[f32]) -> S::u32s {
    let (r, g, b, _a) = driver::split_rgba_f32(simd, src);

    let rc = clamp_channel_simd(simd, r);
    let gc = clamp_channel_simd(simd, g);
    let bc = clamp_channel_simd(simd, b);
    let max_c = rc.max(gc).max(bc);

    // Preliminary shared exponent per lane. With `e = exp_field - 127`, the
    // scalar `max(e, -16) + 16` equals `max(exp_field, 111) - 111` (both operands
    // non-negative, so an unsigned max suffices).
    let exp_field_bits = (max_c.bitcast::<S::u32s>() >> 23) & 0xff;
    let esp = exp_field_bits.max(S::u32s::splat(simd, 111)) - 111;

    // inv_denom_p = 2^(24 - esp): IEEE exponent field 151 - esp.
    // The rounded products are non-negative and well under 2^31, so a signed
    // truncating convert is exact and avoids the unsigned-convert range fixup
    // (a compare + branch + second convert per channel on SSE4.2/AVX2).
    // `inv_denom` is a power of two, so each `channel * inv_denom` product is
    // exact and the FMA's single rounding falls only on the `+ 0.5` — the same
    // result an unfused multiply-then-add gives, while still fusing on FMA tiers.
    let half = S::f32s::splat(simd, 0.5);
    let inv_denom_p: S::f32s = ((S::u32s::splat(simd, 151) - esp) << 23).bitcast();
    let max_s: S::u32s = max_c
        .mul_add(inv_denom_p, half)
        .to_int::<S::i32s>()
        .bitcast();
    // Rounding the max channel into the next binade bumps the shared exponent.
    let exp_shared = max_s.simd_eq(1u32 << MANTISSA_BITS).select(esp + 1, esp);

    let inv_denom: S::f32s = ((S::u32s::splat(simd, 151) - exp_shared) << 23).bitcast();
    let rs = rc
        .mul_add(inv_denom, half)
        .to_int::<S::i32s>()
        .bitcast::<S::u32s>()
        & 0x1ff;
    let gs = gc
        .mul_add(inv_denom, half)
        .to_int::<S::i32s>()
        .bitcast::<S::u32s>()
        & 0x1ff;
    let bs = bc
        .mul_add(inv_denom, half)
        .to_int::<S::i32s>()
        .bitcast::<S::u32s>()
        & 0x1ff;

    let exp = (exp_shared & 0x1f) << 27;
    exp | (bs << 18) | (gs << 9) | rs
}

/// Clamp a channel into `[0, SHARED_EXP_MAX]`, mapping negatives to 0 and
/// `+Inf` / overflow to the max. NaN input is undefined: `max`/`min` NaN
/// semantics are backend-defined (x86 `maxps` returns the second operand,
/// scrubbing NaN to 0; NEON `fmax` propagates it into the shared-exponent
/// extraction, poisoning the whole pixel's word), and per the kernel NaN
/// policy no instructions are spent scrubbing it.
#[inline(always)]
fn clamp_channel_simd<S: Simd>(simd: S, c: S::f32s) -> S::f32s {
    c.max(S::f32s::splat(simd, 0.0))
        .min(S::f32s::splat(simd, SHARED_EXP_MAX))
}

#[cfg(test)]
mod load_tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::processing::kernels::constructible_levels;
    use crate::surface::ColorSpace;

    /// Per-pixel oracle: the shared exponent has no implicit leading one and no
    /// denormal special case, so each channel is simply `mantissa · 2^(exp−24)`.
    /// The scale is built directly as an IEEE-754 f32: `exp` spans `0..=31`, so
    /// `exp - 24 + 127` (i.e. `exp + 103`) lands in `103..=134`, always a valid
    /// normal exponent.
    fn oracle_pixel(word: u32) -> [f32; 4] {
        let scale = f32::from_bits(((word >> 27) + 103) << 23);
        [
            (word & 0x1ff) as f32 * scale,
            ((word >> 9) & 0x1ff) as f32 * scale,
            ((word >> 18) & 0x1ff) as f32 * scale,
            1.0,
        ]
    }

    /// Oracle for a whole surface: walk the rows at the surface stride and
    /// decode each word, ignoring any inter-row padding.
    fn oracle_pixels(surface: &Surface) -> Vec<[f32; 4]> {
        let w = surface.width as usize;
        let stride = surface.stride as usize;
        let mut out = Vec::with_capacity(w * surface.height as usize);
        for y in 0..surface.height as usize {
            let row = &surface.data[y * stride..y * stride + w * 4];
            for chunk in row.chunks_exact(4) {
                out.push(oracle_pixel(u32::from_le_bytes(chunk.try_into().unwrap())));
            }
        }
        out
    }

    /// Hand-computed shared-exponent cases, pinning the oracle independently:
    /// `exp = 24` makes the scale exactly 1 so the mantissa reads as an integer,
    /// each exponent step halves or doubles, `exp = 31` with a full mantissa is
    /// the format's largest value (`511 · 2^7 = 65408`), and `exp = 0` with
    /// mantissa 1 is the smallest (`2^-24`). All three channels share one
    /// exponent, so a single word pins their relative scaling too.
    #[test]
    fn oracle_matches_known_values() {
        let word = |exp: u32, r: u32, g: u32, b: u32| (exp << 27) | (b << 18) | (g << 9) | r;
        for &(w, want) in &[
            (0u32, [0.0f32, 0.0, 0.0]),
            (word(24, 1, 2, 511), [1.0, 2.0, 511.0]),
            (word(23, 1, 256, 511), [0.5, 128.0, 255.5]),
            (word(25, 1, 2, 4), [2.0, 4.0, 8.0]),
            (word(31, 511, 256, 0), [65408.0, 32768.0, 0.0]),
            (
                word(0, 1, 511, 0),
                [2f32.powi(-24), 511.0 * 2f32.powi(-24), 0.0],
            ),
            (
                word(15, 511, 0, 1),
                [511.0 * 2f32.powi(-9), 0.0, 2f32.powi(-9)],
            ),
        ] {
            let got = oracle_pixel(w);
            assert_eq!([got[0], got[1], got[2]], want, "word {w:#010x}");
            assert_eq!(got[3], 1.0, "alpha of {w:#010x}");
        }
    }

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
        let mut state = 0x1234_5678u32;
        for _ in 0..8000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            words.push(state);
        }
        words
    }

    fn assert_load_bit_exact(simd: &[[f32; 4]], surface: &Surface, label: &str) {
        let reference = oracle_pixels(surface);
        assert_eq!(simd.len(), reference.len(), "{label}: pixel count");
        for (i, (got, want)) in simd.iter().zip(&reference).enumerate() {
            for c in 0..4 {
                assert_eq!(
                    got[c].to_bits(),
                    want[c].to_bits(),
                    "{label}: pixel {i} lane {c}: got {} want {}",
                    got[c],
                    want[c],
                );
            }
        }
    }

    #[test]
    fn simd_load_sweep_matches_oracle() {
        let words = sweep_words();
        let width = words.len() as u32;
        let s = e5_surface(&words, width, 1, width * 4);
        for (label, level) in constructible_levels() {
            let simd = load_e5b9g9r9_f32_at(level, &s).unwrap();
            assert_load_bit_exact(&simd.pixels, &s, label);
        }
    }

    #[test]
    fn simd_load_tails_match_oracle() {
        let base = sweep_words();
        for width in 1..=49u32 {
            // Junk stride padding so a tail bug that reads it would diverge.
            let height = 3u32;
            let stride = width * 4 + 12;
            let words: Vec<u32> = base
                .iter()
                .copied()
                .take((width * height) as usize)
                .collect();
            let mut s = e5_surface(&words, width, height, stride);
            for y in 0..height as usize {
                let pad = y * stride as usize + (width * 4) as usize;
                for b in &mut s.data[pad..pad + 12] {
                    *b = 0xAB;
                }
            }
            for (label, level) in constructible_levels() {
                let simd = load_e5b9g9r9_f32_at(level, &s).unwrap();
                assert_load_bit_exact(&simd.pixels, &s, &format!("{label} w={width}"));
            }
        }
    }

    /// A stride that isn't a multiple of 4 puts most row starts off a 4-byte
    /// boundary relative to the surface's byte buffer (whose base the
    /// allocator aligns to at least 4), exercising the loader's
    /// `try_cast_slice` failure arm (the reused word-scratch fallback).
    #[test]
    fn simd_load_misaligned_stride_matches_oracle() {
        let width = 3u32;
        let height = 5u32;
        let row_bytes = width * 4;
        let stride = row_bytes + 1; // not a multiple of 4
        let mut data = vec![0u8; (stride * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let seed = y.wrapping_mul(97).wrapping_add(x.wrapping_mul(13));
                let word = (seed << 27)
                    | ((seed.wrapping_mul(5) & 0x1ff) << 18)
                    | ((seed.wrapping_mul(3) & 0x1ff) << 9)
                    | (seed & 0x1ff);
                let off = (y * stride + x * 4) as usize;
                data[off..off + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let s = Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::E5B9G9R9_UFLOAT_PACK32,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        };
        for (label, level) in constructible_levels() {
            let got = load_e5b9g9r9_f32_at(level, &s).unwrap();
            assert_load_bit_exact(&got.pixels, &s, label);
        }
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use crate::processing::kernels::constructible_levels;

    /// Per-pixel oracle: the Khronos / OpenGL shared-exponent encode, written
    /// straight from the spec with exact divides.
    fn oracle_word(rgb: [f32; 3]) -> u32 {
        /// `2^n` as f32 over the exponent range this encoder produces (always a
        /// valid normal exponent, so a direct bit construction is exact).
        fn exp2i(n: i32) -> f32 {
            f32::from_bits(((n + 127) as u32) << 23)
        }

        /// Clamp into `[0, SHARED_EXP_MAX]`, mapping negatives to 0 and `+Inf`
        /// to the max. NaN is excluded from every test input: the kernel's NaN
        /// output is backend-defined (see `clamp_channel_simd`), so nothing is
        /// asserted about it.
        fn clamp_channel(c: f32) -> f32 {
            c.clamp(0.0, SHARED_EXP_MAX)
        }

        let rc = clamp_channel(rgb[0]);
        let gc = clamp_channel(rgb[1]);
        let bc = clamp_channel(rgb[2]);
        let max_c = rc.max(gc).max(bc);

        // Preliminary shared exponent from the largest channel's magnitude.
        let exp_shared_p = if max_c > 0.0 {
            // floor(log2(max_c)) == unbiased IEEE exponent for normal floats;
            // any denormal/tiny value lands below -EXP_BIAS-1 and clamps.
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

        (((exp_shared as u32) & 0x1f) << 27)
            | ((bs & 0x1ff) << 18)
            | ((gs & 0x1ff) << 9)
            | (rs & 0x1ff)
    }

    fn oracle_bytes(buf: &Buffer<f32>) -> Vec<u8> {
        buf.pixels
            .iter()
            .flat_map(|p| oracle_word([p[0], p[1], p[2]]).to_le_bytes())
            .collect()
    }

    /// Hand-computed encodings, pinning the oracle independently. Reading the
    /// packed word back as `mantissa · 2^(exp − 24)` reproduces the input, and
    /// the exponent is the smallest one whose 9-bit mantissa holds the largest
    /// channel: `1.0` → `256 · 2^-8`, `0.5` → `256 · 2^-9`, the format max
    /// `65408` → `511 · 2^7`, and zero → the all-zero word.
    #[test]
    fn oracle_matches_known_values() {
        let want = |exp: u32, r: u32, g: u32, b: u32| (exp << 27) | (b << 18) | (g << 9) | r;
        for &(rgb, expected) in &[
            ([0.0f32, 0.0, 0.0], 0u32),
            ([1.0, 0.5, 0.0], want(16, 256, 128, 0)),
            ([0.5, 0.25, 0.0], want(15, 256, 128, 0)),
            ([2.0, 1.0, 0.0], want(17, 256, 128, 0)),
            ([65408.0, 32704.0, 0.0], want(31, 511, 256, 0)),
            // Clamped inputs: negatives (including -0.0) → 0, +Inf → the max.
            ([-1.0, -0.0, f32::INFINITY], want(31, 0, 0, 511)),
            // `1.9993 · 2^8 = 511.8`, so the max channel rounds to mantissa 512
            // and bumps the shared exponent from 16 to 17.
            ([1.9993, 0.0, 0.0], want(17, 256, 0, 0)),
        ] {
            assert_eq!(
                oracle_word(rgb),
                expected,
                "encode of {rgb:?}: got {:#010x} want {expected:#010x}",
                oracle_word(rgb),
            );
        }
    }

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
            [65407.0, 65409.0, 65408.0, 1.0], // just below/above the max
            [70000.0, 1.0, 0.001, 1.0],       // above max → clamp/overflow
            [1e30, 2e30, 3e30, 1.0],          // far above max
            [-1.0, -0.5, -0.001, 1.0],        // negatives → 0
            [-5.0, 2.0, 0.0, 1.0],            // mixed sign
            [-0.0, 1.0, 2.0, 1.0],            // negative zero
            // NaN is deliberately absent: the kernel's output for NaN input is
            // backend-defined (see `clamp_channel_simd`).
            [f32::INFINITY, 0.5, 0.25, 1.0], // +Inf → max
            [f32::NEG_INFINITY, 0.5, 2.0, 1.0],
            [1.0, -0.0, f32::INFINITY, 1.0],
            [f32::MIN_POSITIVE, 1e-40, 5e-41, 1.0], // smallest normal + denormals
            [511.0, 512.0, 513.0, 1.0],             // near-binade rounding
            [0.9999, 1.0, 1.0001, 1.0],             // rounding across 1.0
            // Mantissa rounds up to 512, bumping the shared exponent (16 → 17);
            // see `simd_store_mantissa_bump_exponent` for a direct assertion
            // on the packed exponent field.
            [1.9993, 0.0, 0.0, 1.0],
            [65280.0, 65408.0, 100.0, 1.0], // large mixed magnitudes, no bump
        ];
        // A pseudo-random sweep across a wide dynamic range.
        let mut state = 0x9e37_79b9u32;
        for _ in 0..6000 {
            let mut nxt = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state
            };
            let mk = |bits: u32| {
                let exp = ((bits >> 3) % 60) as i32 - 40;
                let mant = 1.0 + ((bits & 0x7ff) as f32) / 2048.0;
                let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
                sign * mant * 2f32.powi(exp)
            };
            v.push([mk(nxt()), mk(nxt()), mk(nxt()), 1.0]);
        }
        v
    }

    fn assert_store_bit_exact(got: &[u8], want: &[u8], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
            let gw = u32::from_le_bytes(g.try_into().unwrap());
            let ww = u32::from_le_bytes(w.try_into().unwrap());
            assert_eq!(gw, ww, "{label}: pixel {i}: got {gw:#010x} want {ww:#010x}");
        }
    }

    /// `1.9993` at the preliminary shared exponent (field 16) rounds to
    /// mantissa `round(1.9993 * 2^8) = 512`, overflowing the 9-bit mantissa and
    /// bumping the shared exponent to 17. Asserts the packed exponent field of
    /// the kernel's own output directly.
    #[test]
    fn simd_store_mantissa_bump_exponent() {
        let buf = buf_from(vec![[1.9993, 0.0, 0.0, 1.0]]);
        for (label, level) in constructible_levels() {
            let got = store_e5b9g9r9_f32_at(level, &buf);
            let word = u32::from_le_bytes(got[0..4].try_into().unwrap());
            assert_eq!(
                (word >> 27) & 0x1f,
                17,
                "{label}: mantissa overflow should bump the shared exponent to 17"
            );
            assert_eq!(word & 0x1ff, 256, "{label}: rescaled mantissa");
        }
    }

    #[test]
    fn simd_store_edge_cases_match_oracle() {
        let buf = buf_from(edge_case_pixels());
        let want = oracle_bytes(&buf);
        for (label, level) in constructible_levels() {
            let got = store_e5b9g9r9_f32_at(level, &buf);
            assert_store_bit_exact(&got, &want, label);
        }
    }

    /// The storer's `try_cast_slice_mut` failure arm (the u32-scratch fallback
    /// that copies out via `cast_slice`) is only taken when the freshly
    /// allocated output `Vec<u8>` happens to land off a 4-byte boundary, which
    /// real allocators essentially never do — so this calls `store_rows` (the
    /// code that arm runs) directly against a standalone scratch buffer,
    /// exercising the same copy-out path deterministically.
    #[test]
    fn simd_store_scratch_fallback_matches_oracle() {
        let buf = buf_from(edge_case_pixels());
        let want = oracle_bytes(&buf);
        let total = buf.pixels.len();
        let src: &[f32] = bytemuck::cast_slice(&buf.pixels);
        for (label, level) in constructible_levels() {
            let mut scratch = vec![0u32; total];
            dispatch!(level, simd => store_rows(simd, src, &mut scratch));
            let got: Vec<u8> = bytemuck::cast_slice(&scratch).to_vec();
            assert_store_bit_exact(&got, &want, &format!("{label} scratch-fallback"));
        }
    }

    #[test]
    fn simd_store_tails_match_oracle() {
        let all = edge_case_pixels();
        for n in 1..=49usize {
            let buf = buf_from(all[..n].to_vec());
            let want = oracle_bytes(&buf);
            for (label, level) in constructible_levels() {
                let got = store_e5b9g9r9_f32_at(level, &buf);
                assert_store_bit_exact(&got, &want, &format!("{label} n={n}"));
            }
        }
    }
}
