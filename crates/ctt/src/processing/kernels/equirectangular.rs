//! Equirectangular → cubemap projection kernels.
//!
//! One width-generic coordinate phase (polynomial `atan2`/`log2`, singular-value
//! footprint, anisotropic tap setup) runs at the backend's native width on
//! `S::f32s`; the narrow trilinear fetch runs on the fixed 128-bit `f32x4<S>`,
//! valid at every level. The derivations behind the coordinate phase are in
//! [`coords`].

use std::f32::consts::{FRAC_PI_2, PI};

use fearless_simd::{Level, Simd, dispatch, f32x4, prelude::*};

use crate::processing::Buffer;
use crate::processing::equirectangular::{
    ATAN_COEFFS, DIR_NORM_GUARD, EquirectangularOrientation, EquirectangularPyramid, FaceBasis,
    GRAD_MAX, LOG2_COEFFS, MAX_ANISO_LOG2, ProjectParams, TexelCmd, project_with,
};

/// One prefiltered level, unpacked for the kernels' narrow phase.
struct LevelInfo<'a> {
    px: &'a [[f32; 4]],
    w: i32,
    h: i32,
    wf: f32,
    hf: f32,
}

fn level_table(pyr: &EquirectangularPyramid) -> Vec<LevelInfo<'_>> {
    pyr.levels()
        .iter()
        .map(|l| LevelInfo {
            px: &l.pixels,
            w: l.width as i32,
            h: l.height as i32,
            wf: l.width as f32,
            hf: l.height as f32,
        })
        .collect()
}

/// Wide-phase output for one SIMD lane bundle, spilled to the stack for the
/// narrow phase.
struct LaneCmds<const N: usize> {
    u: [f32; N],
    v: [f32; N],
    lod: [f32; N],
    step_u: [f32; N],
    step_v: [f32; N],
    taps_log2: [i32; N],
}

impl<const N: usize> LaneCmds<N> {
    #[inline]
    fn zeroed() -> Self {
        Self {
            u: [0.0; N],
            v: [0.0; N],
            lod: [0.0; N],
            step_u: [0.0; N],
            step_v: [0.0; N],
            taps_log2: [0; N],
        }
    }

    /// Reassemble lane `i` as a scalar [`TexelCmd`] for the narrow phase.
    #[inline]
    fn get(&self, i: usize) -> TexelCmd {
        TexelCmd {
            u: self.u[i],
            v: self.v[i],
            lod: self.lod[i],
            step_u: self.step_u[i],
            step_v: self.step_v[i],
            taps_log2: self.taps_log2[i] as u32,
        }
    }
}

/// Project at the best available level: one [`Level::new()`] for the whole run
/// (it is expensive), then [`project_f32_at`]. The geometry-side entry point
/// [`project_f32`](crate::processing::equirectangular::project_f32) delegates
/// here.
pub(crate) fn project_f32(
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> [Buffer<f32>; 6] {
    project_f32_at(Level::new(), pyr, face_size, orientation)
}

/// SIMD kernel behind [`project_f32`](crate::processing::equirectangular::project_f32),
/// at a caller-chosen [`Level`] so benches can force each backend.
#[doc(hidden)]
pub fn project_f32_at(
    level: Level,
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> [Buffer<f32>; 6] {
    dispatch!(level, simd => run_generic(simd, pyr, face_size, orientation))
}

/// Set up the six-face gather and run every band's coordinate + fetch phase
/// with the generic transcendentals. Each band re-enters a `vectorize` scope so
/// the wide ops stay inlined and feature-enabled on whichever `rayon` worker
/// executes it.
#[inline(always)]
fn run_generic<S: Simd>(
    simd: S,
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> [Buffer<f32>; 6] {
    // Built once for the whole projection and borrowed by every band (each
    // `LevelInfo` holds a shared `&[f32]`, so the table is `Sync` for rayon).
    let levels = level_table(pyr);
    project_with(
        pyr,
        face_size,
        orientation,
        move |pyr, basis, n, y_start, out| {
            // The closure must be `#[inline(always)]` (as `dispatch!` does) so
            // `band`'s wide ops inline into `vectorize`'s target-feature scope on
            // each rayon worker; otherwise every wide op degrades to an
            // out-of-line intrinsic call and the tier collapses below scalar.
            simd.vectorize(
                #[inline(always)]
                || band(simd, pyr, basis, &levels, n, y_start, out),
            );
        },
    )
}

/// One row band of one face: the wide coordinate phase produces a lane bundle of
/// [`TexelCmd`]s, then the narrow phase fetches each texel.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn band<S: Simd>(
    simd: S,
    pyr: &EquirectangularPyramid,
    basis: &FaceBasis,
    levels: &[LevelInfo<'_>],
    face_size: u32,
    y_start: u32,
    out: &mut [[f32; 4]],
) {
    let params = ProjectParams::new(pyr, face_size);
    let nf = face_size as usize;
    let inv_n = 1.0 / face_size as f32;
    let nl = S::f32s::N;

    // Reused across every lane bundle: `coords` overwrites lanes `[..nl]` and
    // `get(i)` reads only `i < count ≤ nl`, so stale tail lanes never leak.
    let mut cmds = LaneCmds::<16>::zeroed();
    for (r, row) in out.chunks_exact_mut(nf).enumerate() {
        let b = (2 * (y_start + r as u32) + 1) as f32 * inv_n - 1.0;
        // Row-constant part of the direction: axis + b·v.
        let rb = [
            basis.axis[0] + b * basis.v[0],
            basis.axis[1] + b * basis.v[1],
            basis.axis[2] + b * basis.v[2],
        ];
        let mut x = 0usize;
        while x < nf {
            let count = (nf - x).min(nl);
            coords(simd, basis, &params, rb, x as f32, inv_n, &mut cmds);
            for i in 0..count {
                row[x + i] = fetch_aniso(simd, levels, &cmds.get(i));
            }
            x += count;
        }
    }
}

/// Coordinate phase for `S::f32s::N` consecutive texels of one face row. Results
/// spill into the first `N` lanes of `cmds`.
///
/// Per lane: the face coordinate `a` and the row-constant `axis + b·v` give the
/// sampling direction; longitude `φ = atan2(x, z)` and colatitude
/// `θ = atan2(hyp, y)` map it to `(u, v)`; the analytic gradients of `(u, v)` in
/// mip-0 source texels per output texel form a 2×2 Jacobian whose singular
/// values are the footprint ellipse's axes, which set the LOD, the tap count,
/// and the tap step. The derivations for each step are inline below.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn coords<S: Simd>(
    simd: S,
    basis: &FaceBasis,
    p: &ProjectParams,
    rb: [f32; 3],
    x0: f32,
    inv_n: f32,
    cmds: &mut LaneCmds<16>,
) {
    let nl = S::f32s::N;
    // a = (2·(x0+lane) + 1)/n − 1
    let ramp = S::f32s::from_fn(simd, |i| i as f32);
    let a = ramp.mul_add(2.0 * inv_n, (2.0 * x0 + 1.0) * inv_n - 1.0);

    let [ux, uy, uz] = basis.u;
    let [vx, vy, vz] = basis.v;
    let dx = a.mul_add(ux, rb[0]);
    let dy = a.mul_add(uy, rb[1]);
    let dz = a.mul_add(uz, rb[2]);

    let xz2 = dx.mul_add(dx, dz * dz);
    let hyp = xz2.sqrt();
    let len2 = dy.mul_add(dy, xz2);

    // Longitude φ ∈ (-π, π], colatitude θ ∈ [0, π]. Both are scale-invariant in
    // the direction, so it is never normalized.
    let phi = atan2_generic(simd, dx, dz);
    let theta = atan2_generic(simd, hyp, dy);
    let u = phi.mul_add(0.5 / PI, 0.5);
    let v = theta * (1.0 / PI);

    // Analytic gradients of the source coordinate (in mip-0 texels) per output
    // texel, for both face axes. `xz2 → 0` only at the exact ±Y pole, where the
    // footprint is genuinely unbounded; the guard keeps the math finite and the
    // LOD clamp does the rest.
    let one = S::f32s::splat(simd, 1.0);
    let inv_xz2 = one / xz2.max(f32::MIN_POSITIVE);
    let inv_len2 = one / len2;
    let inv_hyp = hyp * inv_xz2; // hyp/xz² = 1/hyp, reusing the guard

    let dphi_da = dz.mul_sub(ux, dx * uz) * inv_xz2;
    let proj_u = dx.mul_add(ux, dz * uz);
    let dtheta_da = (dy * proj_u).mul_sub(inv_hyp, hyp * uy) * inv_len2;
    let dphi_db = dz.mul_sub(vx, dx * vz) * inv_xz2;
    let proj_v = dx.mul_add(vx, dz * vz);
    let dtheta_db = (dy * proj_v).mul_sub(inv_hyp, hyp * vy) * inv_len2;

    let clamp_g = |g: S::f32s| g.min(GRAD_MAX).max(-GRAD_MAX);
    let gua = clamp_g(dphi_da * p.ku);
    let gva = clamp_g(dtheta_da * p.kv);
    let gub = clamp_g(dphi_db * p.ku);
    let gvb = clamp_g(dtheta_db * p.kv);

    // The footprint ellipse axes are the singular values of the Jacobian
    // J = [[gua, gub], [gva, gvb]] — the face-axis gradient vectors themselves
    // are *not* the principal axes (near the poles both are huge and nearly
    // parallel, which would fake isotropy). For the 2×2 case the singular values
    // and the major eigenvector of J·Jᵀ have a closed form.
    let am = gua.mul_add(gua, gub * gub);
    let bm = gva.mul_add(gva, gvb * gvb);
    let cm = gua.mul_add(gva, gub * gvb);
    let half_diff = (am - bm) * 0.5;
    let disc = half_diff.mul_add(half_diff, cm * cm).sqrt();
    let mean = (am + bm) * 0.5;
    let smax2 = mean + disc;
    let smin2 = (mean - disc).max(f32::MIN_POSITIVE);
    // Major-axis direction: eigenvector of [[am, cm], [cm, bm]] for the larger
    // eigenvalue, picking the numerically larger of the two equivalent forms.
    // Degenerate (isotropic) footprints give a ~zero vector, which the guarded
    // normalization below turns into a zero step.
    let a_ge_b = am.simd_ge(bm);
    let dir_u = a_ge_b.select(disc + half_diff, cm);
    let dir_v = a_ge_b.select(cm, disc - half_diff);
    let norm2 = dir_u.mul_add(dir_u, dir_v * dir_v);
    let norm2 = norm2.max(smax2 * DIR_NORM_GUARD).max(f32::MIN_POSITIVE);
    let scale = (smax2 / norm2).sqrt();
    let gu = dir_u * scale;
    let gv = dir_v * scale;

    // Power-of-two tap count covering the anisotropy ratio, capped; the
    // remainder is folded back into the LOD.
    let lod_minor = log2_generic(simd, smin2) * 0.5;
    let lod_major = log2_generic(simd, smax2.max(f32::MIN_POSITIVE)) * 0.5;
    let delta = (lod_major - lod_minor).max(0.0);
    let k = delta.ceil().min(MAX_ANISO_LOG2 as f32);
    let ki = k.to_int::<S::i32s>();
    let bias = (delta - k).max(0.0);
    let lod = (lod_minor + bias).max(0.0).min(p.max_lod);

    // Tap step along the major gradient, converted to normalized source units
    // and divided among the taps. 2^-k for k ∈ {0..MAX_ANISO_LOG2}: write the
    // exponent field directly. One normalized unit already spans the whole
    // source, so the clamp only tames the unbounded gradients at the exact ±Y
    // poles (where every longitude is equivalent anyway).
    let inv_taps: S::f32s = ((S::i32s::splat(simd, 127) - ki) << 23u32).bitcast();
    let step_u = ((gu * p.inv_w0) * inv_taps).min(1.0).max(-1.0);
    let step_v = ((gv * p.inv_h0) * inv_taps).min(1.0).max(-1.0);

    u.store_slice(&mut cmds.u[..nl]);
    v.store_slice(&mut cmds.v[..nl]);
    lod.store_slice(&mut cmds.lod[..nl]);
    step_u.store_slice(&mut cmds.step_u[..nl]);
    step_v.store_slice(&mut cmds.step_v[..nl]);
    ki.store_slice(&mut cmds.taps_log2[..nl]);
}

/// Run the coordinate phase for a single face texel at `(a, b)` and return
/// lane 0's command, so tests can inspect the production mapping one texel at a
/// time at a chosen [`Level`].
///
/// [`coords`] derives lane 0's face coordinate as `(2·x0 + 1)·inv_n − 1`, so
/// `inv_n = 1`, `x0 = a/2` evaluates the requested `a`.
#[cfg(test)]
pub(crate) fn map_texel_at(
    level: Level,
    basis: &FaceBasis,
    a: f32,
    b: f32,
    p: &ProjectParams,
) -> TexelCmd {
    let rb = [
        basis.axis[0] + b * basis.v[0],
        basis.axis[1] + b * basis.v[1],
        basis.axis[2] + b * basis.v[2],
    ];
    let mut cmds = LaneCmds::<16>::zeroed();
    dispatch!(level, simd => coords(simd, basis, p, rb, a * 0.5, 1.0, &mut cmds));
    cmds.get(0)
}

/// Branchless four-quadrant `atan2(y, x)`, ~1.4e-7 rad max error.
///
/// `#[inline(always)]` is load-bearing: the polynomial must fold fully into the
/// large `band` body's wide phase. Do not reach it through a bare `impl Fn`
/// parameter either — the function item gets invoked through an out-of-line
/// `Fn::call` shim that carries no target features, degrading every vector op
/// inside it to a stack-crossing wrapper call (measured 1.3–1.6× on this
/// kernel). An `#[inline(always)]` closure wrapper is codegen-identical to the
/// direct call but nothing enforces the annotation, so call directly.
#[inline(always)]
fn atan2_generic<S: Simd>(simd: S, y: S::f32s, x: S::f32s) -> S::f32s {
    let ay = y.abs();
    let ax = x.abs();
    let mn = ay.min(ax);
    let mx = ay.max(ax);
    let t = mn / mx.max(f32::MIN_POSITIVE);
    let s = t * t;

    let mut p = S::f32s::splat(simd, ATAN_COEFFS[7]);
    for c in ATAN_COEFFS[..7].iter().rev() {
        p = p.mul_add(s, *c);
    }
    let mut r = p * t;

    // Swap when |y| > |x|; mirror when x's sign bit is set (so −0.0 mirrors);
    // then transfer y's sign. copysign suffices since r ≥ 0 here.
    let swap = ay.simd_gt(ax);
    r = swap.select(S::f32s::splat(simd, FRAC_PI_2) - r, r);
    let xneg = x.bitcast::<S::i32s>().simd_lt(0);
    r = xneg.select(S::f32s::splat(simd, PI) - r, r);
    r.copysign(y)
}

/// `log2(x)` for normal positive `x`, ~2e-4 max error.
#[inline(always)]
fn log2_generic<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let bits = x.bitcast::<S::u32s>();
    let e = (bits >> 23u32).to_float::<S::f32s>() - 127.0;
    let m: S::f32s = ((bits & 0x007f_ffff) | 0x3f80_0000).bitcast();
    let f = m - 1.0;
    let mut p = S::f32s::splat(simd, LOG2_COEFFS[4]);
    for c in LOG2_COEFFS[..4].iter().rev() {
        p = p.mul_add(f, *c);
    }
    e + p
}

// ---- Narrow phase (fixed 128-bit `f32x4<S>`) ----

/// Walk the anisotropic tap line for one texel and average the trilinear taps.
#[inline(always)]
fn fetch_aniso<S: Simd>(simd: S, levels: &[LevelInfo<'_>], cmd: &TexelCmd) -> [f32; 4] {
    let taps = 1u32 << cmd.taps_log2;
    let center = 0.5 * (taps - 1) as f32;
    let mut acc = f32x4::splat(simd, 0.0);
    for k in 0..taps {
        let t = k as f32 - center;
        acc += trilinear(
            simd,
            levels,
            cmd.u + t * cmd.step_u,
            cmd.v + t * cmd.step_v,
            cmd.lod,
        );
    }
    (acc * (1.0 / taps as f32)).into()
}

#[inline(always)]
fn trilinear<S: Simd>(simd: S, levels: &[LevelInfo<'_>], u: f32, v: f32, lod: f32) -> f32x4<S> {
    // lod is pre-clamped to [0, levels.len() - 1].
    let l0 = lod as usize;
    let frac = lod - l0 as f32;
    // SAFETY: lod is clamped to [0, levels.len() - 1], so l0 is in range and
    // l1 = min(l0 + 1, len - 1) is too, so both `get_unchecked` calls are sound.
    let c0 = bilinear(simd, unsafe { levels.get_unchecked(l0) }, u, v);
    if frac <= 0.0 {
        return c0;
    }
    let l1 = (l0 + 1).min(levels.len() - 1);
    let c1 = bilinear(simd, unsafe { levels.get_unchecked(l1) }, u, v);
    (c1 - c0).mul_add(frac, c0)
}

/// Bilinear tap with horizontal wrap and vertical clamp, as one whole-RGBA lerp.
///
/// `u` may lie outside `[0, 1]` (anisotropic tap lines cross the seam), so it is
/// pre-wrapped by `u − floor(u)` into `[0, 1)` and the two x indices get a
/// branch fixup. `v` is left unclamped and the y *indices* are clamped instead,
/// which is what makes the poles replicate the edge row rather than wrap.
#[inline(always)]
fn bilinear<S: Simd>(simd: S, lv: &LevelInfo<'_>, u: f32, v: f32) -> f32x4<S> {
    let uw = u - u.floor();
    let up = uw * lv.wf - 0.5;
    let vp = v * lv.hf - 0.5;
    let x0f = up.floor();
    let y0f = vp.floor();
    let fx = up - x0f;
    let fy = vp - y0f;

    let x0i = x0f as i32;
    let x0 = if x0i < 0 { lv.w - 1 } else { x0i } as usize;
    let x1i = x0i + 1;
    let x1 = if x1i >= lv.w { 0 } else { x1i } as usize;
    let y0i = y0f as i32;
    let y0 = y0i.clamp(0, lv.h - 1) as usize;
    let y1 = (y0i + 1).clamp(0, lv.h - 1) as usize;

    let w = lv.w as usize;
    let row0 = y0 * w;
    let row1 = y1 * w;
    // Load each texel straight from the slice by reference (`load_array_ref_f32x4`
    // transmute-copies in place; `simd_into` on the `[f32; 4]` place would copy it
    // out by value first).
    // `get_unchecked` drops the four bounds-check branches that otherwise serialize
    // the loads — the wrap/clamp above keeps every index in `[0, w)`×`[0, h)`, so
    // `row0/row1 + x0/x1` is always in bounds of the level's pixel slice.
    // SAFETY: x0/x1 < w and y0/y1 < h by the wrap/clamp above.
    let (p00, p01, p10, p11) = unsafe {
        (
            simd.load_array_ref_f32x4(lv.px.get_unchecked(row0 + x0)),
            simd.load_array_ref_f32x4(lv.px.get_unchecked(row0 + x1)),
            simd.load_array_ref_f32x4(lv.px.get_unchecked(row1 + x0)),
            simd.load_array_ref_f32x4(lv.px.get_unchecked(row1 + x1)),
        )
    };
    let top = (p01 - p00).mul_add(fx, p00);
    let bot = (p11 - p10).mul_add(fx, p10);
    (bot - top).mul_add(fy, top)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::kernels::constructible_levels;

    // ---- SIMD transcendental helpers ----

    #[inline(always)]
    fn atan2_block<S: Simd>(simd: S, y: &[f32], x: &[f32], out: &mut [f32]) {
        let n = S::f32s::N;
        for ((yc, xc), oc) in y
            .chunks_exact(n)
            .zip(x.chunks_exact(n))
            .zip(out.chunks_exact_mut(n))
        {
            let yv = S::f32s::from_slice(simd, yc);
            let xv = S::f32s::from_slice(simd, xc);
            atan2_generic(simd, yv, xv).store_slice(oc);
        }
    }

    #[inline(always)]
    fn log2_block<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
        let n = S::f32s::N;
        for (xc, oc) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            let xv = S::f32s::from_slice(simd, xc);
            log2_generic(simd, xv).store_slice(oc);
        }
    }

    /// Grid of interesting (y, x) pairs: all quadrants, axes, signed zero, and
    /// magnitude spreads. Length padded to a multiple of 16 (widest lane count).
    fn atan2_cases() -> (Vec<f32>, Vec<f32>) {
        let mut vals = vec![0.0f32, -0.0];
        for e in [-3i32, -1, 0, 1, 4] {
            let m = 2f32.powi(e);
            for f in [1.0f32, 1.37, 1.93] {
                vals.push(m * f);
                vals.push(-m * f);
            }
        }
        let (mut ys, mut xs) = (Vec::new(), Vec::new());
        for &y in &vals {
            for &x in &vals {
                ys.push(y);
                xs.push(x);
            }
        }
        while !ys.len().is_multiple_of(16) {
            ys.push(1.0);
            xs.push(1.0);
        }
        (ys, xs)
    }

    fn log2_cases() -> Vec<f32> {
        let mut cases = Vec::new();
        for e in -30..30 {
            for f in [1.0f32, 1.111, 1.5, 1.999] {
                cases.push(2f32.powi(e) * f);
            }
        }
        while !cases.len().is_multiple_of(16) {
            cases.push(1.0);
        }
        cases
    }

    const ATAN2_TOL: f32 = 3e-7;
    const LOG2_TOL: f32 = 3e-4;

    #[test]
    fn atan2_generic_matches_std() {
        let (ys, xs) = atan2_cases();
        for (label, level) in constructible_levels() {
            let mut got = vec![0.0f32; ys.len()];
            dispatch!(level, simd => atan2_block(simd, &ys, &xs, &mut got));
            for i in 0..ys.len() {
                let want = ys[i].atan2(xs[i]);
                assert!(
                    (got[i] - want).abs() <= ATAN2_TOL,
                    "{label}: atan2({}, {}) = {} want {want}",
                    ys[i],
                    xs[i],
                    got[i],
                );
            }
        }
    }

    #[test]
    fn log2_generic_matches_std() {
        let xs = log2_cases();
        for (label, level) in constructible_levels() {
            let mut got = vec![0.0f32; xs.len()];
            dispatch!(level, simd => log2_block(simd, &xs, &mut got));
            for i in 0..xs.len() {
                let want = xs[i].log2();
                assert!(
                    (got[i] - want).abs() <= LOG2_TOL,
                    "{label}: log2({}) = {} want {want}",
                    xs[i],
                    got[i],
                );
            }
        }
    }
}
