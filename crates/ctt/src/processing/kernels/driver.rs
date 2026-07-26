//! Format-independent scaffolding for the `fearless_simd` kernels: the packed
//! one-word-per-pixel surface load/store drivers, the block+tail loop, and
//! the AoS↔SoA transposes.
//!
//! Everything here is `#[inline(always)]` so it inlines into `dispatch!`'s
//! target-feature scope. A non-`#[inline(always)]` boundary would strip the
//! enabled features, forcing every wide intrinsic out-of-line and collapsing
//! AVX2/AVX-512 throughput. Kernel modules reference this note rather than
//! restating it.
//!
//! That extends to the behavior passed *into* these drivers: the row and block
//! handles are `impl FnMut` parameters, and LLVM declines to inline a
//! kernel-sized closure body through one on its own. Every call site must write
//! the handle as an `#[inline(always)]` closure literal — a bare function item
//! goes out-of-line through a `Fn::call` shim. Measured on Zen 5, omitting the
//! attribute costs 4× (b10g11r11 load) to 7× (a2 unorm store) at AVX-512, and
//! puts the SIMD tiers behind `Fallback`.

use bytemuck::Pod;
use fearless_simd::{Simd, prelude::*};

use crate::error::Result;
use crate::processing::Buffer;
use crate::surface::Surface;

/// Widest native vector across the `fearless_simd` levels is 512-bit: 16 lanes.
pub(crate) const MAX_LANES: usize = 16;

/// Clamp every lane to `[0, 1]`.
#[inline(always)]
pub(crate) fn clamp01<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    x.max(S::f32s::splat(simd, 0.0))
        .min(S::f32s::splat(simd, 1.0))
}

/// Load a packed one-word-per-pixel surface into a `Buffer<T>`.
///
/// `load_row(simd, codes, dst)` must initialize all `4 * codes.len()` lanes
/// of `dst`, and must be an `#[inline(always)]` closure literal (see the module
/// note).
#[inline(always)]
pub(crate) fn load_packed32<S: Simd, T: Pod>(
    simd: S,
    surface: &Surface,
    mut load_row: impl FnMut(S, &[u32], &mut [T]),
) -> Result<Buffer<T>> {
    crate::processing::load_kernels::validate_surface(surface, 4)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * 4;
    let row_lanes = w * 4;
    let total = w * h;

    let mut pixels: Vec<[T; 4]> = Vec::with_capacity(total);
    let out_base = pixels.as_mut_ptr() as *mut T;

    let mut scratch: Vec<u32> = Vec::new();
    let mut out_i = 0usize;
    for row_region in surface.data.chunks(stride).take(h) {
        let row = &row_region[..row_bytes];
        // Rows at an aligned offset (the common case) cast in place; an odd
        // stride that lands a row at a non-4-byte offset falls back to a
        // reused word buffer.
        let codes: &[u32] = match bytemuck::try_cast_slice::<u8, u32>(row) {
            Ok(c) => c,
            Err(_) => {
                scratch.clear();
                scratch.extend(
                    row.chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                );
                &scratch
            }
        };
        // SAFETY: `out_base` is valid for `total * 4` lanes of `T`; `out_i`
        // stays within that reservation, and `load_row` initializes all
        // `row_lanes` lanes.
        let dst = unsafe { core::slice::from_raw_parts_mut(out_base.add(out_i), row_lanes) };
        load_row(simd, codes, dst);
        out_i += row_lanes;
    }
    // SAFETY: every one of the `total * 4` lanes was initialized above.
    unsafe { pixels.set_len(total) };

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

/// Store 4-lanes-per-pixel data as one packed 32-bit word per pixel.
///
/// `src` holds 4 elements per pixel (typically a bit-view of the buffer's
/// pixels); `store_rows(simd, src, words)` must encode every word and must be
/// an `#[inline(always)]` closure literal (see the module note). Word casts
/// assume a little-endian target: the packed words land in the output bytes in
/// `to_le_bytes` order.
#[inline(always)]
pub(crate) fn store_packed32<S: Simd, T: Pod>(
    simd: S,
    src: &[T],
    mut store_rows: impl FnMut(S, &[T], &mut [u32]),
) -> Vec<u8> {
    let total = src.len() / 4;
    let mut out = vec![0u8; total * 4];
    // A fresh byte allocation is 4-aligned in the common case, so the packed
    // words store straight into it; an unaligned allocation falls back to a
    // word scratch that is copied out (correctness-only, cold).
    match bytemuck::try_cast_slice_mut::<u8, u32>(&mut out) {
        Ok(words) => store_rows(simd, src, words),
        Err(_) => {
            let mut scratch = vec![0u32; total];
            store_rows(simd, src, &mut scratch);
            out.copy_from_slice(bytemuck::cast_slice(&scratch));
        }
    }
    out
}

/// Run `block` over native-width pixel blocks, zero-padding the tail.
///
/// `input` holds `IN_PER` elements per pixel and `output` `OUT_PER` elements
/// per pixel. `block(cin, cout)` always sees full `n`-pixel slices; on the
/// tail the input is zero-padded and only the valid lanes are copied back
/// out. `block` must be an `#[inline(always)]` closure literal (see the module
/// note).
#[inline(always)]
pub(crate) fn for_each_block<I: Pod, O: Pod, const IN_PER: usize, const OUT_PER: usize>(
    n: usize,
    pixels: usize,
    input: &[I],
    output: &mut [O],
    mut block: impl FnMut(&[I], &mut [O]),
) {
    debug_assert!(
        n <= MAX_LANES && IN_PER <= 4 && OUT_PER <= 4,
        "block exceeds the tail scratch"
    );
    let mut x = 0;
    while x + n <= pixels {
        block(
            &input[x * IN_PER..(x + n) * IN_PER],
            &mut output[x * OUT_PER..(x + n) * OUT_PER],
        );
        x += n;
    }
    if x < pixels {
        let rem = pixels - x;
        let mut cin = [I::zeroed(); MAX_LANES * 4];
        cin[..rem * IN_PER].copy_from_slice(&input[x * IN_PER..]);
        let mut cout = [O::zeroed(); MAX_LANES * 4];
        block(&cin[..n * IN_PER], &mut cout[..n * OUT_PER]);
        output[x * OUT_PER..].copy_from_slice(&cout[..rem * OUT_PER]);
    }
}

/// Interleave four channel planes into `[R, G, B, A]` pixels.
///
/// Two element-granularity interleaves compose into a 4-channel transpose at
/// any native width: `interleave` alternates lanes across the whole vector,
/// so `(r∶b)` then `(rb∶ga)` lands `[r, g, b, a]` per pixel.
#[inline(always)]
pub(crate) fn interleave_rgba_f32<S: Simd>(
    r: S::f32s,
    g: S::f32s,
    b: S::f32s,
    a: S::f32s,
    dst: &mut [f32],
) {
    let n = S::f32s::N;
    let (rb_lo, rb_hi) = r.interleave(b);
    let (ga_lo, ga_hi) = g.interleave(a);
    let (p0, p1) = rb_lo.interleave(ga_lo);
    let (p2, p3) = rb_hi.interleave(ga_hi);
    p0.store_slice(&mut dst[0..n]);
    p1.store_slice(&mut dst[n..2 * n]);
    p2.store_slice(&mut dst[2 * n..3 * n]);
    p3.store_slice(&mut dst[3 * n..4 * n]);
}

/// [`interleave_rgba_f32`] over `u32` lanes.
#[inline(always)]
pub(crate) fn interleave_rgba_u32<S: Simd>(
    r: S::u32s,
    g: S::u32s,
    b: S::u32s,
    a: S::u32s,
    dst: &mut [u32],
) {
    let n = S::u32s::N;
    let (rb_lo, rb_hi) = r.interleave(b);
    let (ga_lo, ga_hi) = g.interleave(a);
    let (p0, p1) = rb_lo.interleave(ga_lo);
    let (p2, p3) = rb_hi.interleave(ga_hi);
    p0.store_slice(&mut dst[0..n]);
    p1.store_slice(&mut dst[n..2 * n]);
    p2.store_slice(&mut dst[2 * n..3 * n]);
    p3.store_slice(&mut dst[3 * n..4 * n]);
}

/// Deinterleave `[R, G, B, A]` pixels into four channel planes — the inverse
/// of [`interleave_rgba_f32`].
#[inline(always)]
pub(crate) fn split_rgba_f32<S: Simd>(
    simd: S,
    src: &[f32],
) -> (S::f32s, S::f32s, S::f32s, S::f32s) {
    let n = S::f32s::N;
    let p0 = S::f32s::from_slice(simd, &src[0..n]);
    let p1 = S::f32s::from_slice(simd, &src[n..2 * n]);
    let p2 = S::f32s::from_slice(simd, &src[2 * n..3 * n]);
    let p3 = S::f32s::from_slice(simd, &src[3 * n..4 * n]);
    let (rb_lo, ga_lo) = p0.deinterleave(p1);
    let (rb_hi, ga_hi) = p2.deinterleave(p3);
    let (r, b) = rb_lo.deinterleave(rb_hi);
    let (g, a) = ga_lo.deinterleave(ga_hi);
    (r, g, b, a)
}

/// [`split_rgba_f32`] over `u32` lanes.
#[inline(always)]
pub(crate) fn split_rgba_u32<S: Simd>(
    simd: S,
    src: &[u32],
) -> (S::u32s, S::u32s, S::u32s, S::u32s) {
    let n = S::u32s::N;
    let p0 = S::u32s::from_slice(simd, &src[0..n]);
    let p1 = S::u32s::from_slice(simd, &src[n..2 * n]);
    let p2 = S::u32s::from_slice(simd, &src[2 * n..3 * n]);
    let p3 = S::u32s::from_slice(simd, &src[3 * n..4 * n]);
    let (rb_lo, ga_lo) = p0.deinterleave(p1);
    let (rb_hi, ga_hi) = p2.deinterleave(p3);
    let (r, b) = rb_lo.deinterleave(rb_hi);
    let (g, a) = ga_lo.deinterleave(ga_hi);
    (r, g, b, a)
}
