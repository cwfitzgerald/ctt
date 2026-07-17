//! NEON kernel for the equirectangular → cubemap projection.
//!
//! A 4-wide port of the x86 structure: a wide coordinate phase (polynomial
//! `atan2`, singular-value footprint analysis, `log2` LOD) spilled to stack
//! arrays, then a shared narrow phase that walks the anisotropic tap line
//! with whole-RGBA `float32x4_t` lerps. See `x86.rs` for the algorithm
//! commentary; only the instruction selection differs.

use std::arch::aarch64::*;
use std::f32::consts::{FRAC_PI_2, PI};

use super::super::buffer::Buffer;
use super::{
    ATAN_COEFFS, DIR_NORM_GUARD, EquirectangularOrientation, EquirectangularPyramid, FaceBasis,
    GRAD_MAX, LOG2_COEFFS, LaneCmds, LevelInfo, MAX_ANISO_LOG2, ProjectParams, TexelCmd,
    level_table, project_with,
};

/// Full projection on the NEON kernel.
///
/// **Not part of the public API.** Exposed for benchmarks; real callers use
/// the runtime-dispatched [`super::project_f32`].
///
/// # Safety
/// The NEON feature must be available.
#[doc(hidden)]
pub unsafe fn project_f32_neon(
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> [Buffer<f32>; 6] {
    profiling::scope!("equirectangular::project_f32_neon");
    project_with(
        pyr,
        face_size,
        orientation,
        |pyr, basis, n, y_start, out| {
            // SAFETY: the caller guarantees NEON.
            unsafe { band_neon(pyr, basis, n, y_start, out) }
        },
    )
}

#[target_feature(enable = "neon")]
unsafe fn band_neon(
    pyr: &EquirectangularPyramid,
    basis: &FaceBasis,
    face_size: u32,
    y_start: u32,
    out: &mut [[f32; 4]],
) {
    let params = ProjectParams::new(pyr, face_size);
    let levels = level_table(pyr);
    let n = face_size as usize;
    let inv_n = 1.0 / face_size as f32;

    for (r, row) in out.chunks_exact_mut(n).enumerate() {
        let b = (2 * (y_start + r as u32) + 1) as f32 * inv_n - 1.0;
        let rb = [
            basis.axis[0] + b * basis.v[0],
            basis.axis[1] + b * basis.v[1],
            basis.axis[2] + b * basis.v[2],
        ];
        let mut x = 0usize;
        while x < n {
            let count = (n - x).min(4);
            let cmds = coords4(basis, &params, rb, x as f32, inv_n);
            for i in 0..count {
                // SAFETY: propagated NEON.
                row[x + i] = unsafe { fetch_aniso(&levels, &cmds.get(i)) };
            }
            x += count;
        }
    }
}

/// Coordinate phase for 4 consecutive texels of one face row.
#[target_feature(enable = "neon")]
#[inline]
fn coords4(
    basis: &FaceBasis,
    params: &ProjectParams,
    rb: [f32; 3],
    x0: f32,
    inv_n: f32,
) -> LaneCmds<4> {
    let ramp = unsafe { vld1q_f32([0.0f32, 1.0, 2.0, 3.0].as_ptr()) };
    let a = vfmaq_f32(
        vdupq_n_f32((2.0 * x0 + 1.0) * inv_n - 1.0),
        ramp,
        vdupq_n_f32(2.0 * inv_n),
    );

    let (ux, uy, uz) = (basis.u[0], basis.u[1], basis.u[2]);
    let (vx, vy, vz) = (basis.v[0], basis.v[1], basis.v[2]);
    let dx = vfmaq_f32(vdupq_n_f32(rb[0]), a, vdupq_n_f32(ux));
    let dy = vfmaq_f32(vdupq_n_f32(rb[1]), a, vdupq_n_f32(uy));
    let dz = vfmaq_f32(vdupq_n_f32(rb[2]), a, vdupq_n_f32(uz));

    let xz2 = vfmaq_f32(vmulq_f32(dz, dz), dx, dx);
    let hyp = vsqrtq_f32(xz2);
    let len2 = vfmaq_f32(xz2, dy, dy);

    let phi = atan2_neon(dx, dz);
    let theta = atan2_neon(hyp, dy);
    let u = vfmaq_f32(vdupq_n_f32(0.5), phi, vdupq_n_f32(0.5 / PI));
    let v = vmulq_f32(theta, vdupq_n_f32(1.0 / PI));

    let one = vdupq_n_f32(1.0);
    let min_pos = vdupq_n_f32(f32::MIN_POSITIVE);
    let inv_xz2 = vdivq_f32(one, vmaxq_f32(xz2, min_pos));
    let inv_len2 = vdivq_f32(one, len2);
    let inv_hyp = vmulq_f32(hyp, inv_xz2);

    // dphi/dtheta along both face axes; see the serial reference.
    let dphi_da = vmulq_f32(
        vfmsq_f32(vmulq_f32(dz, vdupq_n_f32(ux)), dx, vdupq_n_f32(uz)),
        inv_xz2,
    );
    let proj_u = vfmaq_f32(vmulq_f32(dz, vdupq_n_f32(uz)), dx, vdupq_n_f32(ux));
    let dtheta_da = vmulq_f32(
        vfmsq_f32(
            vmulq_f32(vmulq_f32(dy, proj_u), inv_hyp),
            hyp,
            vdupq_n_f32(uy),
        ),
        inv_len2,
    );
    let dphi_db = vmulq_f32(
        vfmsq_f32(vmulq_f32(dz, vdupq_n_f32(vx)), dx, vdupq_n_f32(vz)),
        inv_xz2,
    );
    let proj_v = vfmaq_f32(vmulq_f32(dz, vdupq_n_f32(vz)), dx, vdupq_n_f32(vx));
    let dtheta_db = vmulq_f32(
        vfmsq_f32(
            vmulq_f32(vmulq_f32(dy, proj_v), inv_hyp),
            hyp,
            vdupq_n_f32(vy),
        ),
        inv_len2,
    );

    let grad_max = vdupq_n_f32(GRAD_MAX);
    let grad_min = vdupq_n_f32(-GRAD_MAX);
    let clamp_g = |g: float32x4_t| vmaxq_f32(vminq_f32(g, grad_max), grad_min);
    let gua = clamp_g(vmulq_f32(dphi_da, vdupq_n_f32(params.ku)));
    let gva = clamp_g(vmulq_f32(dtheta_da, vdupq_n_f32(params.kv)));
    let gub = clamp_g(vmulq_f32(dphi_db, vdupq_n_f32(params.ku)));
    let gvb = clamp_g(vmulq_f32(dtheta_db, vdupq_n_f32(params.kv)));

    // Footprint singular values and major axis: see the serial reference.
    let am = vfmaq_f32(vmulq_f32(gub, gub), gua, gua);
    let bm = vfmaq_f32(vmulq_f32(gvb, gvb), gva, gva);
    let cm = vfmaq_f32(vmulq_f32(gub, gvb), gua, gva);
    let half = vdupq_n_f32(0.5);
    let half_diff = vmulq_f32(vsubq_f32(am, bm), half);
    let disc = vsqrtq_f32(vfmaq_f32(vmulq_f32(cm, cm), half_diff, half_diff));
    let mean = vmulq_f32(vaddq_f32(am, bm), half);
    let smax2 = vaddq_f32(mean, disc);
    let smin2 = vmaxq_f32(vsubq_f32(mean, disc), min_pos);
    let a_ge_b = vcgeq_f32(am, bm);
    let dir_u = vbslq_f32(a_ge_b, vaddq_f32(disc, half_diff), cm);
    let dir_v = vbslq_f32(a_ge_b, cm, vsubq_f32(disc, half_diff));
    let norm2 = vfmaq_f32(vmulq_f32(dir_v, dir_v), dir_u, dir_u);
    let norm2 = vmaxq_f32(
        vmaxq_f32(norm2, vmulq_f32(smax2, vdupq_n_f32(DIR_NORM_GUARD))),
        min_pos,
    );
    let scale = vsqrtq_f32(vdivq_f32(smax2, norm2));
    let gu = vmulq_f32(dir_u, scale);
    let gv = vmulq_f32(dir_v, scale);

    let lod_minor = vmulq_f32(log2_neon(smin2), half);
    let lod_major = vmulq_f32(log2_neon(vmaxq_f32(smax2, min_pos)), half);
    let zero = vdupq_n_f32(0.0);
    let delta = vmaxq_f32(vsubq_f32(lod_major, lod_minor), zero);
    let k = vminq_f32(vrndpq_f32(delta), vdupq_n_f32(MAX_ANISO_LOG2 as f32));
    let ki = vcvtnq_s32_f32(k);
    let bias = vmaxq_f32(vsubq_f32(delta, k), zero);
    let lod = vminq_f32(
        vmaxq_f32(vaddq_f32(lod_minor, bias), zero),
        vdupq_n_f32(params.max_lod),
    );

    // 2^-k for small integer k: build the exponent field directly.
    let inv_taps = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vsubq_s32(vdupq_n_s32(127), ki)));
    let neg_one = vdupq_n_f32(-1.0);
    let step_u = vmaxq_f32(
        vminq_f32(
            vmulq_f32(vmulq_f32(gu, vdupq_n_f32(params.inv_w0)), inv_taps),
            one,
        ),
        neg_one,
    );
    let step_v = vmaxq_f32(
        vminq_f32(
            vmulq_f32(vmulq_f32(gv, vdupq_n_f32(params.inv_h0)), inv_taps),
            one,
        ),
        neg_one,
    );

    let mut cmds = LaneCmds::<4>::zeroed();
    // SAFETY: each array holds exactly 4 lanes.
    unsafe {
        vst1q_f32(cmds.u.as_mut_ptr(), u);
        vst1q_f32(cmds.v.as_mut_ptr(), v);
        vst1q_f32(cmds.lod.as_mut_ptr(), lod);
        vst1q_f32(cmds.step_u.as_mut_ptr(), step_u);
        vst1q_f32(cmds.step_v.as_mut_ptr(), step_v);
        vst1q_s32(cmds.taps_log2.as_mut_ptr(), ki);
    }
    cmds
}

/// Branchless four-quadrant `atan2(y, x)`, ~1.4e-7 rad max error.
#[target_feature(enable = "neon")]
#[inline]
fn atan2_neon(y: float32x4_t, x: float32x4_t) -> float32x4_t {
    let ay = vabsq_f32(y);
    let ax = vabsq_f32(x);
    let mn = vminq_f32(ay, ax);
    let mx = vmaxq_f32(ay, ax);
    let t = vdivq_f32(mn, vmaxq_f32(mx, vdupq_n_f32(f32::MIN_POSITIVE)));
    let s = vmulq_f32(t, t);

    let mut p = vdupq_n_f32(ATAN_COEFFS[7]);
    for c in ATAN_COEFFS[..7].iter().rev() {
        p = vfmaq_f32(vdupq_n_f32(*c), p, s);
    }
    let mut r = vmulq_f32(p, t);

    let swap = vcgtq_f32(ay, ax);
    r = vbslq_f32(swap, vsubq_f32(vdupq_n_f32(FRAC_PI_2), r), r);
    // Mirror where x's sign bit is set (including -0.0): arithmetic shift
    // of the sign bit builds the mask.
    let xneg = vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_f32(x)));
    r = vbslq_f32(xneg, vsubq_f32(vdupq_n_f32(PI), r), r);
    // Transfer y's sign.
    let sign = vandq_u32(vreinterpretq_u32_f32(y), vdupq_n_u32(0x8000_0000));
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(r), sign))
}

/// `log2(x)` for normal positive `x`, ~2e-4 max error.
#[target_feature(enable = "neon")]
#[inline]
fn log2_neon(x: float32x4_t) -> float32x4_t {
    let bits = vreinterpretq_u32_f32(x);
    let e = vcvtq_f32_s32(vsubq_s32(
        vreinterpretq_s32_u32(vshrq_n_u32::<23>(bits)),
        vdupq_n_s32(127),
    ));
    let m = vreinterpretq_f32_u32(vorrq_u32(
        vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
        vdupq_n_u32(0x3f80_0000),
    ));
    let f = vsubq_f32(m, vdupq_n_f32(1.0));
    let mut p = vdupq_n_f32(LOG2_COEFFS[4]);
    for c in LOG2_COEFFS[..4].iter().rev() {
        p = vfmaq_f32(vdupq_n_f32(*c), p, f);
    }
    vaddq_f32(e, p)
}

// ---- Narrow phase ----

/// # Safety
/// The NEON feature must be available.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fetch_aniso(levels: &[LevelInfo<'_>], cmd: &TexelCmd) -> [f32; 4] {
    let taps = 1u32 << cmd.taps_log2;
    let center = 0.5 * (taps - 1) as f32;
    let mut acc = vdupq_n_f32(0.0);
    for k in 0..taps {
        let t = k as f32 - center;
        // SAFETY: propagated from the caller.
        let s = unsafe {
            trilinear(
                levels,
                cmd.u + t * cmd.step_u,
                cmd.v + t * cmd.step_v,
                cmd.lod,
            )
        };
        acc = vaddq_f32(acc, s);
    }
    let avg = vmulq_f32(acc, vdupq_n_f32(1.0 / taps as f32));
    let mut out = [0.0f32; 4];
    // SAFETY: the array holds exactly 4 lanes.
    unsafe { vst1q_f32(out.as_mut_ptr(), avg) };
    out
}

/// # Safety
/// The NEON feature must be available.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn trilinear(levels: &[LevelInfo<'_>], u: f32, v: f32, lod: f32) -> float32x4_t {
    let l0 = lod as usize;
    let frac = lod - l0 as f32;
    // SAFETY: propagated; l0 is in range by the LOD clamp.
    let c0 = unsafe { bilinear(levels.get_unchecked(l0), u, v) };
    if frac <= 0.0 {
        return c0;
    }
    let l1 = (l0 + 1).min(levels.len() - 1);
    // SAFETY: as above.
    let c1 = unsafe { bilinear(levels.get_unchecked(l1), u, v) };
    vfmaq_f32(c0, vsubq_f32(c1, c0), vdupq_n_f32(frac))
}

/// # Safety
/// The NEON feature must be available.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn bilinear(lv: &LevelInfo<'_>, u: f32, v: f32) -> float32x4_t {
    let uw = u - u.floor();
    let up = uw * lv.wf - 0.5;
    let vp = v.clamp(0.0, 1.0) * lv.hf - 0.5;
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
    // SAFETY: x0/x1 < w and y0/y1 < h by the wrap/clamp above.
    let (p00, p01, p10, p11) = unsafe {
        (
            vld1q_f32(lv.px.get_unchecked(row0 + x0).as_ptr()),
            vld1q_f32(lv.px.get_unchecked(row0 + x1).as_ptr()),
            vld1q_f32(lv.px.get_unchecked(row1 + x0).as_ptr()),
            vld1q_f32(lv.px.get_unchecked(row1 + x1).as_ptr()),
        )
    };
    let fxv = vdupq_n_f32(fx);
    let top = vfmaq_f32(p00, vsubq_f32(p01, p00), fxv);
    let bot = vfmaq_f32(p10, vsubq_f32(p11, p10), fxv);
    vfmaq_f32(top, vsubq_f32(bot, top), vdupq_n_f32(fy))
}
