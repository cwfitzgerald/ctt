//! Shared machinery for the in-place sRGB transfer-curve passes.
//!
//! Two directions share this module: the load-side EOTF post-pass and the
//! store-side OETF pre-pass. Both run [`curve_in_place_with_token`], which
//! applies a [`CurveKernel`] to the RGB lanes of every pixel over native-width
//! blocks.

use fearless_simd::{Simd, prelude::*};

use super::driver::{self, clamp01};

/// A transfer curve evaluated on a full native vector. A zero-sized dispatch
/// type (not a bare `impl Fn` value) so each direction's curve inlines into
/// the hot loop instead of routing through an out-of-line `Fn::call`.
pub(crate) trait CurveKernel {
    fn curve<S: Simd>(simd: S, x: S::f32s) -> S::f32s;
}

/// Apply `C` to the RGB lanes of every pixel (clamped to `[0, 1]` first),
/// preserving each alpha lane bit-exactly. Pixels are a flat `[f32]`; a lane is
/// an alpha lane iff its index mod 4 is 3, which holds within every block
/// because both the width and the tail remainder are multiples of 4. The pass
/// is in-place (input aliases output) so it runs its own block loop rather than
/// [`driver::for_each_block`].
#[inline(always)]
pub(crate) fn curve_in_place_with_token<S: Simd, C: CurveKernel>(simd: S, pixels: &mut [[f32; 4]]) {
    let buf: &mut [f32] = bytemuck::cast_slice_mut(pixels);
    let n = S::f32s::N;
    debug_assert!(n <= driver::MAX_LANES);
    let total = buf.len();

    let mut ids = [0u32; driver::MAX_LANES];
    for (i, id) in ids.iter_mut().enumerate() {
        *id = i as u32;
    }
    let lane_ids = S::u32s::from_slice(simd, &ids[..n]);

    let mut x = 0;
    while x + n <= total {
        curve_block::<S, C>(simd, lane_ids, &mut buf[x..x + n]);
        x += n;
    }

    // Tail: zero-pad a full native block, process, copy out the valid lanes.
    if x < total {
        let rem = total - x;
        let mut scratch = [0.0f32; driver::MAX_LANES];
        scratch[..rem].copy_from_slice(&buf[x..]);
        curve_block::<S, C>(simd, lane_ids, &mut scratch[..n]);
        buf[x..].copy_from_slice(&scratch[..rem]);
    }
}

/// Clamp + curve one native block, blending the original alpha lanes back.
#[inline(always)]
fn curve_block<S: Simd, C: CurveKernel>(simd: S, lane_ids: S::u32s, buf: &mut [f32]) {
    let orig = S::f32s::from_slice(simd, buf);
    let curved = C::curve(simd, clamp01(simd, orig));
    let is_alpha = (lane_ids & 3).simd_eq(3u32);
    is_alpha.select(orig, curved).store_slice(buf);
}
