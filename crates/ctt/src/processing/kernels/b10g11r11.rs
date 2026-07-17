//! `B10G11R11_UFLOAT_PACK32` loader and storer (packed unsigned floats).
//!
//! One little-endian u32 per pixel: R is an 11-bit float at bits `[0,11)`, G
//! an 11-bit float at `[11,22)`, B a 10-bit float at `[22,32)`. All are
//! unsigned (no sign bit) with a 5-bit exponent (bias 15); the 11-bit forms
//! carry 6 mantissa bits, the 10-bit form 5. Alpha loads as 1.0 and is dropped
//! on store.
//!
//! # Load side
//!
//! The SIMD kernels below are width-generic, running at the backend's native
//! vector width (128-bit on Fallback/SSE4.2, 256-bit on AVX2, 512-bit on
//! AVX-512), and the decode is exact at every width: the reconstructed value
//! `(mant_max + mant)·2^(exp−M−15)` carries at most `M + 1` (≤7) significant
//! mantissa bits, so the product is exact in f32 regardless of FMA fusion.
//!
//! # Store side
//!
//! The per-channel packing is the standard round-to-nearest-even f32 → small
//! unsigned float conversion (as used by D3D's `R11G11B10_FLOAT`): negatives
//! flush to 0, finite overflow and +Inf map to Inf. NaN input is undefined.
//!
//! The store kernels are width-generic in the same way. Every lane performs the
//! same integer bit manipulation, so the packed words are exactly the
//! round-to-nearest-even conversion described above at every width.

use fearless_simd::{Level, Simd, dispatch, prelude::*};

use crate::error::Result;
use crate::processing::Buffer;
use crate::surface::Surface;

use super::driver;

/// Bit pattern of `+Inf`.
const F32_INF_BITS: u32 = 0x7f80_0000;
/// Bit pattern of a quiet NaN (matches `f32::NAN`).
const F32_NAN_BITS: u32 = 0x7fc0_0000;
/// Bit pattern of `2^-14`, the smallest positive normal small-float value
/// (IEEE exponent field `127 - 14 = 113`).
const F32_2POW_M14_BITS: u32 = 0x3880_0000;

/// Closed-form decode of an unsigned float with `MANT_BITS` mantissa bits and a
/// 5-bit, bias-15 exponent — the exact IEEE small-float rules `B10G11R11` uses
/// (exp 0 is zero/denormal, exp 31 is Inf/NaN), written per-channel and
/// branchful.
///
/// Test-only: the oracle the SIMD decode is validated against, and the source
/// of the store round-trip test's inputs.
#[cfg(test)]
fn decode_small_float<const MANT_BITS: u32>(v: u32) -> f32 {
    let mant_max = 1u32 << MANT_BITS;
    let exp = (v >> MANT_BITS) & 0x1f;
    let mant = v & (mant_max - 1);
    if exp == 0 {
        // Zero (mant == 0) or denormal: 2^-14 * (mant / 2^MANT_BITS).
        2f32.powi(-14) * (mant as f32 / mant_max as f32)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        // Normal: 2^(exp - 15) * (1 + mant / 2^MANT_BITS).
        let scale = f32::from_bits((exp + 112) << 23); // 2^(exp - 15)
        scale * (1.0 + mant as f32 / mant_max as f32)
    }
}

pub fn load_b10g11r11_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_b10g11r11_f32");
    load_b10g11r11_f32_at(Level::new(), surface)
}

/// SIMD kernel behind [`load_b10g11r11_f32`], at a caller-chosen [`Level`] so
/// benches can force each backend.
#[doc(hidden)]
pub fn load_b10g11r11_f32_at(level: Level, surface: &Surface) -> Result<Buffer<f32>> {
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
            let (r, g, b) = decode_words(simd, words);
            let one = S::f32s::splat(simd, 1.0);
            driver::interleave_rgba_f32::<S>(r, g, b, one, cout);
        },
    );
}

/// Split packed words into right-aligned R/G/B channel codes and decode each.
#[inline(always)]
fn decode_words<S: Simd>(simd: S, words: S::u32s) -> (S::f32s, S::f32s, S::f32s) {
    let r = words & 0x7ff;
    let g = (words >> 11) & 0x7ff;
    let b = (words >> 22) & 0x3ff;
    (
        decode_codes::<S, 6>(simd, r),
        decode_codes::<S, 6>(simd, g),
        decode_codes::<S, 5>(simd, b),
    )
}

/// Branchless small-float decode for uniform `M` mantissa bits, 5-bit exponent.
#[inline(always)]
fn decode_codes<S: Simd, const M: u32>(simd: S, codes: S::u32s) -> S::f32s {
    let mant_max = 1u32 << M;
    let mant = codes & (mant_max - 1);
    let exp = (codes >> M) & 0x1f;

    let mant_f: S::f32s = mant.to_float();
    let frac = mant_f * (1.0 / mant_max as f32);
    let denorm = frac * f32::from_bits(F32_2POW_M14_BITS);
    // scale = 2^(exp-15): write the IEEE exponent field directly.
    let scale: S::f32s = ((exp + 112) << 23).bitcast();
    // scale * (1 + frac) fused as scale * frac + scale.
    let normal = scale.mul_add(frac, scale);

    let val = exp.simd_eq(0u32).select(denorm, normal);

    let inf = splat_bits(simd, F32_INF_BITS);
    let nan = splat_bits(simd, F32_NAN_BITS);
    let special = mant.simd_gt(0u32).select(nan, inf);
    exp.simd_eq(0x1f).select(special, val)
}

/// Splat a raw bit pattern across a native `f32` vector.
#[inline(always)]
fn splat_bits<S: Simd>(simd: S, bits: u32) -> S::f32s {
    let v: S::u32s = bits.simd_into(simd);
    v.bitcast()
}

// ---- linear-f32 → packed small-float SIMD store ----

pub fn store_b10g11r11_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_b10g11r11_f32");
    store_b10g11r11_f32_at(Level::new(), buf)
}

/// SIMD kernel behind [`store_b10g11r11_f32`], at a caller-chosen [`Level`] so
/// benches can force each backend.
#[doc(hidden)]
pub fn store_b10g11r11_f32_at(level: Level, buf: &Buffer<f32>) -> Vec<u8> {
    // The pixel bits feed the encode as raw words; the AoS→SoA shuffle is a
    // lane permute, so working in `u32` space is exact and skips any bitcast.
    let src: &[u32] = bytemuck::cast_slice(&buf.pixels);
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

/// Encode every pixel of `src` (4 raw f32-bit words each) into one packed word.
#[inline(always)]
fn store_rows<S: Simd>(simd: S, src: &[u32], words: &mut [u32]) {
    driver::for_each_block::<_, _, 4, 1>(
        S::u32s::N,
        words.len(),
        src,
        words,
        #[inline(always)]
        |cin, cout| {
            let (r, g, b, _a) = driver::split_rgba_u32(simd, cin);
            let ri = encode_codes::<S, 6>(simd, r);
            let gi = encode_codes::<S, 6>(simd, g);
            let bi = encode_codes::<S, 5>(simd, b);
            let block = ri | (gi << 11) | (bi << 22);
            block.store_slice(cout);
        },
    );
}

/// Branchless f32 → `5 + M`-bit unsigned-float encode: rebias the exponent (or
/// build a subnormal below `2^-14`), round to nearest even, and mask to the
/// target width, with negatives flushed to 0 and overflow mapped to Inf.
#[inline(always)]
fn encode_codes<S: Simd, const M: u32>(simd: S, bits: S::u32s) -> S::u32s {
    let drop = 23 - M;
    let mant_bits_mask = (1u32 << M) - 1;
    let result_mask = (1u32 << (5 + M)) - 1;
    let round = (1u32 << (drop - 1)) - 1;
    let inf = 0x1fu32 << M;
    let max_bits = (142u32 << 23) | (mant_bits_mask << drop);

    let zero = S::u32s::splat(simd, 0);
    let mag = bits & 0x7fff_ffffu32;

    let is_over = mag.simd_gt(max_bits);
    let is_sub = mag.simd_lt(0x3880_0000u32);

    // Subnormal path: `(0x0080_0000 | mantissa) >> (113 - exp)`, zeroed at 32+.
    let numer = (mag & 0x007f_ffffu32) | 0x0080_0000u32;
    let shift = S::u32s::splat(simd, 113) - (mag >> 23);
    let sub_i = if S::u32s::N >= 8 {
        // AVX2/AVX-512 `vpsrlvd` yields 0 for counts ≥ 32, matching the guard.
        numer >> shift
    } else {
        // Scalar/SSE backends shift with Rust's `>>` (panics in debug for counts
        // ≥ 32) or lack a native variable shift, so clamp the count and mask the
        // ≥ 32 lanes explicitly.
        let sub_raw = numer >> shift.min(31u32);
        shift.simd_gt(31u32).select(zero, sub_raw)
    };

    // Normal path: rebias the exponent from bias 127 to bias 15.
    let norm_i = mag + 0xC800_0000u32;
    let i = is_sub.select(sub_i, norm_i);

    // Round-to-nearest-even, then mask to the result width.
    let odd = (i >> drop) & 1u32;
    let t = i + round + odd;
    let rounded = (t >> drop) & result_mask;

    // Overflow region: negative → 0, else → Inf. (NaN input is undefined; it
    // lands here via its all-ones exponent and takes the sign-based branch.)
    let sign = (bits & 0x8000_0000u32).simd_eq(0x8000_0000u32);

    let inf_vec = S::u32s::splat(simd, inf);
    let over_res = sign.select(zero, inf_vec);
    let nonover_res = sign.select(zero, rounded);
    is_over.select(over_res, nonover_res)
}

#[cfg(test)]
mod load_tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::processing::kernels::constructible_levels;
    use crate::surface::ColorSpace;

    /// Per-pixel oracle: decode one packed word into `[R, G, B, 1.0]` with
    /// [`decode_small_float`].
    fn oracle_pixel(word: u32) -> [f32; 4] {
        [
            decode_small_float::<6>(word & 0x7ff),
            decode_small_float::<6>((word >> 11) & 0x7ff),
            decode_small_float::<5>((word >> 22) & 0x3ff),
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

    /// Hand-computed small-float codes, pinning the oracle independently of the
    /// arithmetic it shares with nothing else: zero, the smallest denormal, the
    /// exponent-15 unit value, one mantissa step above it, half, the largest
    /// finite code, and the Inf/NaN exponent.
    #[test]
    fn oracle_matches_known_values() {
        // (code, expected) for the 11-bit form (6 mantissa bits, unit = 15 << 6).
        for &(code, want) in &[
            (0u32, 0.0f32),
            (1, 2f32.powi(-14) / 64.0),         // denormal, mant = 1
            (63, 2f32.powi(-14) * 63.0 / 64.0), // largest denormal
            (64, 2f32.powi(-14)),               // exp 1, mant 0 → min normal
            (15 << 6, 1.0),                     // exp 15 → 2^0
            ((15 << 6) + 1, 1.0 + 1.0 / 64.0),  // one mantissa step above 1.0
            (14 << 6, 0.5),
            (16 << 6, 2.0),
            ((30 << 6) | 63, 65024.0), // largest finite: 2^15 · (1 + 63/64)
        ] {
            assert_eq!(decode_small_float::<6>(code), want, "11-bit code {code:#x}");
        }
        assert!(decode_small_float::<6>(31 << 6).is_infinite());
        assert!(decode_small_float::<6>((31 << 6) | 1).is_nan());

        // (code, expected) for the 10-bit form (5 mantissa bits, unit = 15 << 5).
        for &(code, want) in &[
            (0u32, 0.0f32),
            (1, 2f32.powi(-14) / 32.0),
            (32, 2f32.powi(-14)),
            (15 << 5, 1.0),
            ((15 << 5) + 1, 1.0 + 1.0 / 32.0),
            (14 << 5, 0.5),
            ((30 << 5) | 31, 64512.0), // largest finite: 2^15 · (1 + 31/32)
        ] {
            assert_eq!(decode_small_float::<5>(code), want, "10-bit code {code:#x}");
        }
        assert!(decode_small_float::<5>(31 << 5).is_infinite());
        assert!(decode_small_float::<5>((31 << 5) | 1).is_nan());
    }

    fn b10_surface(data: Vec<u8>, width: u32, height: u32, stride: u32) -> Surface {
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::B10G11R11_UFLOAT_PACK32,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    /// The decode is exact on every backend, so every lane must match the
    /// oracle to the bit.
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

    /// Sweeps all 2^11 R/G codes and all 2^10 B codes in their real bit slots.
    fn sweep_surface() -> Surface {
        let n = 2048u32;
        let mut data = Vec::with_capacity((n * 4) as usize);
        for i in 0..n {
            let word = (i & 0x7ff) | ((i & 0x7ff) << 11) | ((i & 0x3ff) << 22);
            data.extend_from_slice(&word.to_le_bytes());
        }
        b10_surface(data, n, 1, n * 4)
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
                let r = seed & 0x7ff;
                let g = seed.wrapping_mul(3) & 0x7ff;
                let b = seed.wrapping_mul(5) & 0x3ff;
                let word = r | (g << 11) | (b << 22);
                let off = (y * stride + xp * 4) as usize;
                data[off..off + 4].copy_from_slice(&word.to_le_bytes());
            }
            let pad_off = (y * stride + row_bytes) as usize;
            for byte in &mut data[pad_off..pad_off + 12] {
                *byte = 0xAB;
            }
        }
        b10_surface(data, width, h, stride)
    }

    #[test]
    fn simd_sweep_matches_oracle() {
        let s = sweep_surface();
        for (label, level) in constructible_levels() {
            let simd = load_b10g11r11_f32_at(level, &s).unwrap();
            assert_load_bit_exact(&simd.pixels, &s, label);
        }
    }

    #[test]
    fn simd_tails_match_oracle() {
        for (label, level) in constructible_levels() {
            for width in 1..=49u32 {
                let s = tail_stride_surface(width);
                let simd = load_b10g11r11_f32_at(level, &s).unwrap();
                assert_load_bit_exact(&simd.pixels, &s, label);
            }
        }
    }

    /// A stride that isn't a multiple of 4 puts most row starts off a 4-byte
    /// boundary relative to the surface's byte buffer (whose base the allocator
    /// aligns to at least 4), exercising the loader's `try_cast_slice` failure
    /// arm (the reused word-scratch fallback).
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
                let r = seed & 0x7ff;
                let g = seed.wrapping_mul(3) & 0x7ff;
                let b = seed.wrapping_mul(5) & 0x3ff;
                let word = r | (g << 11) | (b << 22);
                let off = (y * stride + x * 4) as usize;
                data[off..off + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let s = b10_surface(data, width, height, stride);
        for (label, level) in constructible_levels() {
            let got = load_b10g11r11_f32_at(level, &s).unwrap();
            assert_load_bit_exact(&got.pixels, &s, label);
        }
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use crate::processing::kernels::constructible_levels;

    /// Per-channel oracle: the branchful closed form of the f32 →
    /// `5 + MANT_BITS`-bit unsigned-float conversion (D3D `R11G11B10_FLOAT`
    /// rules) — negatives flush to 0, finite overflow and `+Inf` map to Inf,
    /// values below `2^-14` become subnormals, and the mantissa rounds to
    /// nearest even.
    fn encode_small_float<const MANT_BITS: u32>(x: f32) -> u32 {
        let result_mask = (1u32 << (5 + MANT_BITS)) - 1;
        let drop = 23 - MANT_BITS;
        let inf = 0x1fu32 << MANT_BITS; // exponent all ones, zero mantissa
        // Largest finite f32 whose rounded value still fits: exponent 2^15 with
        // a full target mantissa.
        let max_bits = (142u32 << 23) | (((1u32 << MANT_BITS) - 1) << drop);

        let bits = x.to_bits();
        let mag = bits & 0x7fff_ffff;

        if mag > max_bits {
            // Overflow (finite too large or +Inf) → Inf; anything with the sign
            // bit set (incl. -Inf) → 0. NaN input is undefined and falls
            // through here.
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
            i = i.wrapping_add(0xC800_0000); // rebias bias 127 → bias 15
        }

        let round = (1u32 << (drop - 1)) - 1;
        ((i + round + ((i >> drop) & 1)) >> drop) & result_mask
    }

    /// Per-pixel oracle: `[R, G, B]` into one packed `B10G11R11` word.
    fn oracle_word(lanes: &[f32; 4]) -> u32 {
        encode_small_float::<6>(lanes[0])
            | (encode_small_float::<6>(lanes[1]) << 11)
            | (encode_small_float::<5>(lanes[2]) << 22)
    }

    fn oracle_bytes(buf: &Buffer<f32>) -> Vec<u8> {
        buf.pixels
            .iter()
            .flat_map(|p| oracle_word(p).to_le_bytes())
            .collect()
    }

    fn store_buf(pixels: Vec<[f32; 4]>) -> Buffer<f32> {
        let width = pixels.len().max(1) as u32;
        Buffer {
            pixels,
            width,
            height: 1,
        }
    }

    /// Packed words must be bit-identical to the oracle.
    fn assert_store_matches(actual: &[u8], buf: &Buffer<f32>, label: &str) {
        let reference = oracle_bytes(buf);
        assert_eq!(actual.len(), reference.len(), "{label}: byte count");
        for (i, (&got, &want)) in actual.iter().zip(&reference).enumerate() {
            assert_eq!(
                got,
                want,
                "{label}: pixel {} byte {}: got {got:#04x} want {want:#04x}",
                i / 4,
                i % 4
            );
        }
    }

    /// Hand-computed encodings, pinning the oracle independently: the unit value
    /// lands on exponent 15 (mantissa 0), halving/doubling steps the exponent,
    /// one mantissa step is `1 + 1/2^M`, the smallest normal is `2^-14`, the
    /// largest finite value keeps a full mantissa, anything larger saturates to
    /// Inf, and negatives flush to zero.
    #[test]
    fn oracle_matches_known_values() {
        for &(x, want) in &[
            (0.0f32, 0u32),
            (-0.0, 0),
            (-1.0, 0),
            (f32::NEG_INFINITY, 0),
            (1.0, 15 << 6),
            (0.5, 14 << 6),
            (2.0, 16 << 6),
            (1.0 + 1.0 / 64.0, (15 << 6) | 1),
            (2f32.powi(-14), 1 << 6),   // smallest normal, exp field 1
            (2f32.powi(-14) / 64.0, 1), // smallest subnormal
            (65024.0, (30 << 6) | 63),  // largest finite 11-bit value
            (65280.0, 31 << 6),         // beyond it → Inf
            (f32::INFINITY, 31 << 6),
        ] {
            assert_eq!(encode_small_float::<6>(x), want, "11-bit encode of {x}");
        }
        for &(x, want) in &[
            (0.0f32, 0u32),
            (1.0, 15 << 5),
            (0.5, 14 << 5),
            (1.0 + 1.0 / 32.0, (15 << 5) | 1),
            (2f32.powi(-14), 1 << 5),
            (64512.0, (30 << 5) | 31), // largest finite 10-bit value
            (65024.0, 31 << 5),        // beyond it → Inf
            (f32::INFINITY, 31 << 5),
        ] {
            assert_eq!(encode_small_float::<5>(x), want, "10-bit encode of {x}");
        }
    }

    /// Round-trip: every code that decodes to a non-NaN value (0 through the Inf
    /// code inclusive), re-encoded, must land back on itself at every level.
    /// Codes above the Inf code decode to NaN, whose encoding is undefined.
    #[test]
    fn store_roundtrips_representable() {
        /// Highest 11-bit code that is not NaN (the Inf code).
        const MAX_11: u32 = 31 << 6;
        /// Highest 10-bit code that is not NaN.
        const MAX_10: u32 = 31 << 5;

        // R/G sweep the 11-bit codes; B sweeps the 10-bit codes, both padded to
        // the longer of the two so one buffer covers everything.
        let pixels: Vec<[f32; 4]> = (0..=MAX_11)
            .map(|code| {
                let rg = decode_small_float::<6>(code);
                let b = decode_small_float::<5>(code.min(MAX_10));
                [rg, rg, b, 1.0]
            })
            .collect();
        let buf = store_buf(pixels);
        for (label, level) in constructible_levels() {
            let got = store_b10g11r11_f32_at(level, &buf);
            for (i, word) in got.chunks_exact(4).enumerate() {
                let word = u32::from_le_bytes(word.try_into().unwrap());
                let code = i as u32;
                assert_eq!(word & 0x7ff, code, "{label}: R code {code}");
                assert_eq!((word >> 11) & 0x7ff, code, "{label}: G code {code}");
                assert_eq!(
                    (word >> 22) & 0x3ff,
                    code.min(MAX_10),
                    "{label}: B code {code}"
                );
            }
            assert_store_matches(&got, &buf, label);
        }
    }

    /// Per-channel f32 inputs spanning every defined encode branch: signed
    /// zeros, subnormals, values straddling the target min-normal and rounding
    /// boundaries, overflow-to-Inf, +Inf, and negatives, plus a pseudo-random
    /// magnitude sweep across both signs. (NaN input is undefined by policy, so
    /// none is included.)
    fn store_values() -> Vec<f32> {
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
            -1.0,
            -0.5,
            -100.0,
            -70000.0,
        ];
        let mut state = 0x1234_5678u32;
        for _ in 0..2000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let f = f32::from_bits(state);
            if f.is_finite() {
                v.push(f % 131072.0);
            }
            v.push((state as f32 / u32::MAX as f32) * 4.0 - 1.0);
        }
        v
    }

    /// Pixels whose channels each march through `store_values` at a per-channel
    /// offset so R, G, B rarely share the same encode branch on a given pixel.
    fn store_test_pixels() -> Vec<[f32; 4]> {
        let values = store_values();
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

    #[test]
    fn simd_store_matches_oracle() {
        let buf = store_buf(store_test_pixels());
        for (label, level) in constructible_levels() {
            let got = store_b10g11r11_f32_at(level, &buf);
            assert_store_matches(&got, &buf, label);
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
        let buf = store_buf(store_test_pixels());
        let total = buf.pixels.len();
        let src: &[u32] = bytemuck::cast_slice(&buf.pixels);
        for (label, level) in constructible_levels() {
            let mut scratch = vec![0u32; total];
            dispatch!(level, simd => store_rows(simd, src, &mut scratch));
            let got: Vec<u8> = bytemuck::cast_slice(&scratch).to_vec();
            assert_store_matches(&got, &buf, &format!("{label} scratch-fallback"));
        }
    }

    #[test]
    fn simd_store_tails_match_oracle() {
        let pixels = store_test_pixels();
        for (label, level) in constructible_levels() {
            for width in 1..=49usize {
                let buf = store_buf(pixels[..width].to_vec());
                let got = store_b10g11r11_f32_at(level, &buf);
                assert_store_matches(&got, &buf, &format!("{label} w={width}"));
            }
        }
    }
}
