//! x86 SIMD kernels for the equirectangular → cubemap projection.
//!
//! Each kernel splits per-texel work into two phases:
//!
//! - **Wide phase** (8 lanes on AVX2, 16 on AVX-512): the transcendental
//!   coordinate math — direction, polynomial `atan2`, gradients, `log2`
//!   for the LOD, anisotropic tap setup — producing a [`TexelCmd`]-shaped
//!   lane bundle spilled to stack arrays.
//! - **Narrow phase** (shared 128-bit code): per texel, walk the tap line
//!   and evaluate trilinear taps as whole-RGBA `__m128` lerps. The four
//!   loads per bilinear tap are independent across taps and texels, which
//!   is what hides the gather latency.
//!
//! The AVX-512 path uses instructions with no AVX2 equivalent even though
//! the narrow phase is shared: `vrangeps` folds the abs/min/max of the
//! `atan2` octant reduction into two ops, `vgetexpps`/`vgetmantps` replace
//! the integer exponent-extraction dance in `log2`, `vscalefps` builds
//! `2^-k` directly, and `vpternlogd` fuses the sign transfer.

use std::arch::x86_64::*;
use std::f32::consts::{FRAC_PI_2, PI};

use super::super::buffer::Buffer;
use super::{
    ATAN_COEFFS, EquirectangularOrientation, EquirectangularPyramid, FaceBasis, LOG2_COEFFS,
    LaneCmds, LevelInfo, MAX_ANISO_LOG2, ProjectParams, TexelCmd, level_table, project_with,
};

/// Full projection on the AVX2+FMA kernel.
///
/// **Not part of the public API.** Exposed for benchmarks; real callers use
/// the runtime-dispatched [`super::project_f32`].
///
/// # Safety
/// The AVX2 and FMA features must be available.
#[doc(hidden)]
pub unsafe fn project_f32_avx2_fma(
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> [Buffer<f32>; 6] {
    profiling::scope!("equirectangular::project_f32_avx2_fma");
    project_with(
        pyr,
        face_size,
        orientation,
        |pyr, basis, n, y_start, out| {
            // SAFETY: the caller guarantees avx2+fma.
            unsafe { band_avx2_fma(pyr, basis, n, y_start, out) }
        },
    )
}

/// Full projection on the AVX-512 kernel.
///
/// **Not part of the public API.** Exposed for benchmarks; real callers use
/// the runtime-dispatched [`super::project_f32`].
///
/// # Safety
/// The AVX-512 F/VL/BW/DQ features must be available.
#[doc(hidden)]
pub unsafe fn project_f32_avx512(
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> [Buffer<f32>; 6] {
    profiling::scope!("equirectangular::project_f32_avx512");
    project_with(
        pyr,
        face_size,
        orientation,
        |pyr, basis, n, y_start, out| {
            // SAFETY: the caller guarantees the AVX-512 feature set.
            unsafe { band_avx512(pyr, basis, n, y_start, out) }
        },
    )
}

// ---- AVX2+FMA ----

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn band_avx2_fma(
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
        // Row-constant part of the direction: axis + b·v.
        let rb = [
            basis.axis[0] + b * basis.v[0],
            basis.axis[1] + b * basis.v[1],
            basis.axis[2] + b * basis.v[2],
        ];
        let mut x = 0usize;
        while x < n {
            let count = (n - x).min(8);
            let cmds = coords8(basis, &params, rb, x as f32, inv_n);
            for i in 0..count {
                // SAFETY: avx2+fma implies the helper's sse4.1+fma.
                row[x + i] = unsafe { fetch_aniso(&levels, &cmds.get(i)) };
            }
            x += count;
        }
    }
}

/// Coordinate phase for 8 consecutive texels of one face row.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
fn coords8(basis: &FaceBasis, p: &ProjectParams, rb: [f32; 3], x0: f32, inv_n: f32) -> LaneCmds<8> {
    // a = (2·(x0+lane) + 1)/n − 1
    let ramp = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    let a = _mm256_fmadd_ps(
        ramp,
        _mm256_set1_ps(2.0 * inv_n),
        _mm256_set1_ps((2.0 * x0 + 1.0) * inv_n - 1.0),
    );

    let dx = _mm256_fmadd_ps(a, _mm256_set1_ps(basis.u[0]), _mm256_set1_ps(rb[0]));
    let dy = _mm256_fmadd_ps(a, _mm256_set1_ps(basis.u[1]), _mm256_set1_ps(rb[1]));
    let dz = _mm256_fmadd_ps(a, _mm256_set1_ps(basis.u[2]), _mm256_set1_ps(rb[2]));

    let xz2 = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dz, dz));
    let hyp = _mm256_sqrt_ps(xz2);
    let len2 = _mm256_fmadd_ps(dy, dy, xz2);

    let phi = atan2_avx2(dx, dz);
    let theta = atan2_avx2(hyp, dy);
    let u = _mm256_fmadd_ps(phi, _mm256_set1_ps(0.5 / PI), _mm256_set1_ps(0.5));
    let v = _mm256_mul_ps(theta, _mm256_set1_ps(1.0 / PI));

    // Gradients of (φ, θ) along both face axes, scaled to mip-0 texels per
    // output texel.
    let one = _mm256_set1_ps(1.0);
    let inv_xz2 = _mm256_div_ps(one, _mm256_max_ps(xz2, _mm256_set1_ps(f32::MIN_POSITIVE)));
    let inv_len2 = _mm256_div_ps(one, len2);
    let inv_hyp = _mm256_mul_ps(hyp, inv_xz2);

    let (ux, uy, uz) = (
        _mm256_set1_ps(basis.u[0]),
        _mm256_set1_ps(basis.u[1]),
        _mm256_set1_ps(basis.u[2]),
    );
    let dphi_da = _mm256_mul_ps(_mm256_fmsub_ps(dz, ux, _mm256_mul_ps(dx, uz)), inv_xz2);
    let proj_u = _mm256_fmadd_ps(dx, ux, _mm256_mul_ps(dz, uz));
    let dtheta_da = _mm256_mul_ps(
        _mm256_fmsub_ps(_mm256_mul_ps(dy, proj_u), inv_hyp, _mm256_mul_ps(hyp, uy)),
        inv_len2,
    );

    let (vx, vy, vz) = (
        _mm256_set1_ps(basis.v[0]),
        _mm256_set1_ps(basis.v[1]),
        _mm256_set1_ps(basis.v[2]),
    );
    let dphi_db = _mm256_mul_ps(_mm256_fmsub_ps(dz, vx, _mm256_mul_ps(dx, vz)), inv_xz2);
    let proj_v = _mm256_fmadd_ps(dx, vx, _mm256_mul_ps(dz, vz));
    let dtheta_db = _mm256_mul_ps(
        _mm256_fmsub_ps(_mm256_mul_ps(dy, proj_v), inv_hyp, _mm256_mul_ps(hyp, vy)),
        inv_len2,
    );

    let ku = _mm256_set1_ps(p.ku);
    let kv = _mm256_set1_ps(p.kv);
    let grad_max = _mm256_set1_ps(super::GRAD_MAX);
    let grad_min = _mm256_set1_ps(-super::GRAD_MAX);
    let clamp_g = |g: __m256| _mm256_max_ps(_mm256_min_ps(g, grad_max), grad_min);
    let gua = clamp_g(_mm256_mul_ps(dphi_da, ku));
    let gva = clamp_g(_mm256_mul_ps(dtheta_da, kv));
    let gub = clamp_g(_mm256_mul_ps(dphi_db, ku));
    let gvb = clamp_g(_mm256_mul_ps(dtheta_db, kv));

    // Footprint singular values and major axis: see the serial reference.
    let am = _mm256_fmadd_ps(gua, gua, _mm256_mul_ps(gub, gub));
    let bm = _mm256_fmadd_ps(gva, gva, _mm256_mul_ps(gvb, gvb));
    let cm = _mm256_fmadd_ps(gua, gva, _mm256_mul_ps(gub, gvb));
    let half = _mm256_set1_ps(0.5);
    let half_diff = _mm256_mul_ps(_mm256_sub_ps(am, bm), half);
    let disc = _mm256_sqrt_ps(_mm256_fmadd_ps(half_diff, half_diff, _mm256_mul_ps(cm, cm)));
    let mean = _mm256_mul_ps(_mm256_add_ps(am, bm), half);
    let min_pos = _mm256_set1_ps(f32::MIN_POSITIVE);
    let smax2 = _mm256_add_ps(mean, disc);
    let smin2 = _mm256_max_ps(_mm256_sub_ps(mean, disc), min_pos);
    let a_ge_b = _mm256_cmp_ps::<_CMP_GE_OQ>(am, bm);
    let dir_u = _mm256_blendv_ps(cm, _mm256_add_ps(disc, half_diff), a_ge_b);
    let dir_v = _mm256_blendv_ps(_mm256_sub_ps(disc, half_diff), cm, a_ge_b);
    let norm2 = _mm256_fmadd_ps(dir_u, dir_u, _mm256_mul_ps(dir_v, dir_v));
    let norm2 = _mm256_max_ps(
        _mm256_max_ps(
            norm2,
            _mm256_mul_ps(smax2, _mm256_set1_ps(super::DIR_NORM_GUARD)),
        ),
        min_pos,
    );
    let scale = _mm256_sqrt_ps(_mm256_div_ps(smax2, norm2));
    let gu = _mm256_mul_ps(dir_u, scale);
    let gv = _mm256_mul_ps(dir_v, scale);

    let lod_minor = _mm256_mul_ps(log2_avx2(smin2), half);
    let lod_major = _mm256_mul_ps(log2_avx2(_mm256_max_ps(smax2, min_pos)), half);
    let zero = _mm256_setzero_ps();
    let delta = _mm256_max_ps(_mm256_sub_ps(lod_major, lod_minor), zero);
    let k = _mm256_min_ps(_mm256_ceil_ps(delta), _mm256_set1_ps(MAX_ANISO_LOG2 as f32));
    let ki = _mm256_cvtps_epi32(k);
    let bias = _mm256_max_ps(_mm256_sub_ps(delta, k), zero);
    let lod = _mm256_min_ps(
        _mm256_max_ps(_mm256_add_ps(lod_minor, bias), zero),
        _mm256_set1_ps(p.max_lod),
    );

    // 2^-k for k ∈ {0..MAX_ANISO_LOG2}: build the exponent field directly.
    let inv_taps = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_sub_epi32(
        _mm256_set1_epi32(127),
        ki,
    )));
    // Step clamp: see the serial reference — tames pole-singular gradients.
    let neg_one = _mm256_set1_ps(-1.0);
    let step_u = _mm256_max_ps(
        _mm256_min_ps(
            _mm256_mul_ps(_mm256_mul_ps(gu, _mm256_set1_ps(p.inv_w0)), inv_taps),
            one,
        ),
        neg_one,
    );
    let step_v = _mm256_max_ps(
        _mm256_min_ps(
            _mm256_mul_ps(_mm256_mul_ps(gv, _mm256_set1_ps(p.inv_h0)), inv_taps),
            one,
        ),
        neg_one,
    );

    let mut cmds = LaneCmds::<8>::zeroed();
    // SAFETY: each array holds exactly 8 lanes.
    unsafe {
        _mm256_storeu_ps(cmds.u.as_mut_ptr(), u);
        _mm256_storeu_ps(cmds.v.as_mut_ptr(), v);
        _mm256_storeu_ps(cmds.lod.as_mut_ptr(), lod);
        _mm256_storeu_ps(cmds.step_u.as_mut_ptr(), step_u);
        _mm256_storeu_ps(cmds.step_v.as_mut_ptr(), step_v);
        _mm256_storeu_si256(cmds.taps_log2.as_mut_ptr().cast(), ki);
    }
    cmds
}

/// Branchless four-quadrant `atan2(y, x)`, ~1.4e-7 rad max error.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
fn atan2_avx2(y: __m256, x: __m256) -> __m256 {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let sign_mask = _mm256_set1_ps(-0.0);
    let ay = _mm256_and_ps(y, abs_mask);
    let ax = _mm256_and_ps(x, abs_mask);
    let mn = _mm256_min_ps(ay, ax);
    let mx = _mm256_max_ps(ay, ax);
    let t = _mm256_div_ps(mn, _mm256_max_ps(mx, _mm256_set1_ps(f32::MIN_POSITIVE)));
    let s = _mm256_mul_ps(t, t);

    let mut p = _mm256_set1_ps(ATAN_COEFFS[7]);
    for c in ATAN_COEFFS[..7].iter().rev() {
        p = _mm256_fmadd_ps(p, s, _mm256_set1_ps(*c));
    }
    let mut r = _mm256_mul_ps(p, t);

    // Octant / quadrant fixups: swap when |y| > |x|, mirror when x < 0,
    // then transfer y's sign. blendv keys off the mask's sign bit, so `x`
    // itself drives the mirror select.
    let swap = _mm256_cmp_ps::<_CMP_GT_OQ>(ay, ax);
    r = _mm256_blendv_ps(r, _mm256_sub_ps(_mm256_set1_ps(FRAC_PI_2), r), swap);
    r = _mm256_blendv_ps(r, _mm256_sub_ps(_mm256_set1_ps(PI), r), x);
    _mm256_xor_ps(r, _mm256_and_ps(y, sign_mask))
}

/// `log2(x)` for normal positive `x`, ~2e-4 max error.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
fn log2_avx2(x: __m256) -> __m256 {
    let bits = _mm256_castps_si256(x);
    let e = _mm256_cvtepi32_ps(_mm256_sub_epi32(
        _mm256_srli_epi32::<23>(bits),
        _mm256_set1_epi32(127),
    ));
    let m = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f80_0000),
    ));
    let f = _mm256_sub_ps(m, _mm256_set1_ps(1.0));
    let mut p = _mm256_set1_ps(LOG2_COEFFS[4]);
    for c in LOG2_COEFFS[..4].iter().rev() {
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(*c));
    }
    _mm256_add_ps(e, p)
}

// ---- AVX-512 ----

#[target_feature(
    enable = "avx512f",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "avx512dq"
)]
unsafe fn band_avx512(
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
            let count = (n - x).min(16);
            let cmds = coords16(basis, &params, rb, x as f32, inv_n);
            for i in 0..count {
                // SAFETY: avx512 implies the helper's sse4.1+fma.
                row[x + i] = unsafe { fetch_aniso(&levels, &cmds.get(i)) };
            }
            x += count;
        }
    }
}

/// Coordinate phase for 16 consecutive texels of one face row.
#[target_feature(
    enable = "avx512f",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "avx512dq"
)]
#[inline]
fn coords16(
    basis: &FaceBasis,
    p: &ProjectParams,
    rb: [f32; 3],
    x0: f32,
    inv_n: f32,
) -> LaneCmds<16> {
    let ramp = _mm512_setr_ps(
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    );
    let a = _mm512_fmadd_ps(
        ramp,
        _mm512_set1_ps(2.0 * inv_n),
        _mm512_set1_ps((2.0 * x0 + 1.0) * inv_n - 1.0),
    );

    let dx = _mm512_fmadd_ps(a, _mm512_set1_ps(basis.u[0]), _mm512_set1_ps(rb[0]));
    let dy = _mm512_fmadd_ps(a, _mm512_set1_ps(basis.u[1]), _mm512_set1_ps(rb[1]));
    let dz = _mm512_fmadd_ps(a, _mm512_set1_ps(basis.u[2]), _mm512_set1_ps(rb[2]));

    let xz2 = _mm512_fmadd_ps(dx, dx, _mm512_mul_ps(dz, dz));
    let hyp = _mm512_sqrt_ps(xz2);
    let len2 = _mm512_fmadd_ps(dy, dy, xz2);

    let phi = atan2_avx512(dx, dz);
    let theta = atan2_avx512(hyp, dy);
    let u = _mm512_fmadd_ps(phi, _mm512_set1_ps(0.5 / PI), _mm512_set1_ps(0.5));
    let v = _mm512_mul_ps(theta, _mm512_set1_ps(1.0 / PI));

    let one = _mm512_set1_ps(1.0);
    let inv_xz2 = _mm512_div_ps(one, _mm512_max_ps(xz2, _mm512_set1_ps(f32::MIN_POSITIVE)));
    let inv_len2 = _mm512_div_ps(one, len2);
    let inv_hyp = _mm512_mul_ps(hyp, inv_xz2);

    let (ux, uy, uz) = (
        _mm512_set1_ps(basis.u[0]),
        _mm512_set1_ps(basis.u[1]),
        _mm512_set1_ps(basis.u[2]),
    );
    let dphi_da = _mm512_mul_ps(_mm512_fmsub_ps(dz, ux, _mm512_mul_ps(dx, uz)), inv_xz2);
    let proj_u = _mm512_fmadd_ps(dx, ux, _mm512_mul_ps(dz, uz));
    let dtheta_da = _mm512_mul_ps(
        _mm512_fmsub_ps(_mm512_mul_ps(dy, proj_u), inv_hyp, _mm512_mul_ps(hyp, uy)),
        inv_len2,
    );

    let (vx, vy, vz) = (
        _mm512_set1_ps(basis.v[0]),
        _mm512_set1_ps(basis.v[1]),
        _mm512_set1_ps(basis.v[2]),
    );
    let dphi_db = _mm512_mul_ps(_mm512_fmsub_ps(dz, vx, _mm512_mul_ps(dx, vz)), inv_xz2);
    let proj_v = _mm512_fmadd_ps(dx, vx, _mm512_mul_ps(dz, vz));
    let dtheta_db = _mm512_mul_ps(
        _mm512_fmsub_ps(_mm512_mul_ps(dy, proj_v), inv_hyp, _mm512_mul_ps(hyp, vy)),
        inv_len2,
    );

    let ku = _mm512_set1_ps(p.ku);
    let kv = _mm512_set1_ps(p.kv);
    // Gradient clamp via vrangeps: imm 0b0010 = min magnitude with src1's
    // sign — a symmetric magnitude clamp in a single instruction.
    let grad_max = _mm512_set1_ps(super::GRAD_MAX);
    let clamp_g = |g: __m512| _mm512_range_ps::<0b0010>(g, grad_max);
    let gua = clamp_g(_mm512_mul_ps(dphi_da, ku));
    let gva = clamp_g(_mm512_mul_ps(dtheta_da, kv));
    let gub = clamp_g(_mm512_mul_ps(dphi_db, ku));
    let gvb = clamp_g(_mm512_mul_ps(dtheta_db, kv));

    // Footprint singular values and major axis: see the serial reference.
    let am = _mm512_fmadd_ps(gua, gua, _mm512_mul_ps(gub, gub));
    let bm = _mm512_fmadd_ps(gva, gva, _mm512_mul_ps(gvb, gvb));
    let cm = _mm512_fmadd_ps(gua, gva, _mm512_mul_ps(gub, gvb));
    let half = _mm512_set1_ps(0.5);
    let half_diff = _mm512_mul_ps(_mm512_sub_ps(am, bm), half);
    let disc = _mm512_sqrt_ps(_mm512_fmadd_ps(half_diff, half_diff, _mm512_mul_ps(cm, cm)));
    let mean = _mm512_mul_ps(_mm512_add_ps(am, bm), half);
    let min_pos = _mm512_set1_ps(f32::MIN_POSITIVE);
    let smax2 = _mm512_add_ps(mean, disc);
    let smin2 = _mm512_max_ps(_mm512_sub_ps(mean, disc), min_pos);
    let a_ge_b = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(am, bm);
    let dir_u = _mm512_mask_blend_ps(a_ge_b, cm, _mm512_add_ps(disc, half_diff));
    let dir_v = _mm512_mask_blend_ps(a_ge_b, _mm512_sub_ps(disc, half_diff), cm);
    let norm2 = _mm512_fmadd_ps(dir_u, dir_u, _mm512_mul_ps(dir_v, dir_v));
    let norm2 = _mm512_max_ps(
        _mm512_max_ps(
            norm2,
            _mm512_mul_ps(smax2, _mm512_set1_ps(super::DIR_NORM_GUARD)),
        ),
        min_pos,
    );
    let scale = _mm512_sqrt_ps(_mm512_div_ps(smax2, norm2));
    let gu = _mm512_mul_ps(dir_u, scale);
    let gv = _mm512_mul_ps(dir_v, scale);

    let lod_minor = _mm512_mul_ps(log2_avx512(smin2), half);
    let lod_major = _mm512_mul_ps(log2_avx512(_mm512_max_ps(smax2, min_pos)), half);
    let zero = _mm512_setzero_ps();
    let delta = _mm512_max_ps(_mm512_sub_ps(lod_major, lod_minor), zero);
    // Ceil via roundscale (toward +inf, exceptions suppressed).
    let k = _mm512_min_ps(
        _mm512_roundscale_ps::<0x0A>(delta),
        _mm512_set1_ps(MAX_ANISO_LOG2 as f32),
    );
    let ki = _mm512_cvtps_epi32(k);
    let bias = _mm512_max_ps(_mm512_sub_ps(delta, k), zero);
    let lod = _mm512_min_ps(
        _mm512_max_ps(_mm512_add_ps(lod_minor, bias), zero),
        _mm512_set1_ps(p.max_lod),
    );

    // 2^-k directly via vscalefps.
    let inv_taps = _mm512_scalef_ps(one, _mm512_sub_ps(zero, k));
    // Step clamp: see the serial reference — tames pole-singular gradients.
    let neg_one = _mm512_set1_ps(-1.0);
    let step_u = _mm512_max_ps(
        _mm512_min_ps(
            _mm512_mul_ps(_mm512_mul_ps(gu, _mm512_set1_ps(p.inv_w0)), inv_taps),
            one,
        ),
        neg_one,
    );
    let step_v = _mm512_max_ps(
        _mm512_min_ps(
            _mm512_mul_ps(_mm512_mul_ps(gv, _mm512_set1_ps(p.inv_h0)), inv_taps),
            one,
        ),
        neg_one,
    );

    let mut cmds = LaneCmds::<16>::zeroed();
    // SAFETY: each array holds exactly 16 lanes.
    unsafe {
        _mm512_storeu_ps(cmds.u.as_mut_ptr(), u);
        _mm512_storeu_ps(cmds.v.as_mut_ptr(), v);
        _mm512_storeu_ps(cmds.lod.as_mut_ptr(), lod);
        _mm512_storeu_ps(cmds.step_u.as_mut_ptr(), step_u);
        _mm512_storeu_ps(cmds.step_v.as_mut_ptr(), step_v);
        _mm512_storeu_si512(cmds.taps_log2.as_mut_ptr().cast(), ki);
    }
    cmds
}

/// Branchless four-quadrant `atan2(y, x)` using `vrangeps` for the octant
/// reduction and `vpternlogd` for the sign transfer.
#[target_feature(
    enable = "avx512f",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "avx512dq"
)]
#[inline]
fn atan2_avx512(y: __m512, x: __m512) -> __m512 {
    // range imm 0b1010 = min(|a|,|b|) forced positive; 0b1011 = max(|a|,|b|).
    let mn = _mm512_range_ps::<0b1010>(y, x);
    let mx = _mm512_range_ps::<0b1011>(y, x);
    let ay = _mm512_abs_ps(y);
    let t = _mm512_div_ps(mn, _mm512_max_ps(mx, _mm512_set1_ps(f32::MIN_POSITIVE)));
    let s = _mm512_mul_ps(t, t);

    let mut p = _mm512_set1_ps(ATAN_COEFFS[7]);
    for c in ATAN_COEFFS[..7].iter().rev() {
        p = _mm512_fmadd_ps(p, s, _mm512_set1_ps(*c));
    }
    let mut r = _mm512_mul_ps(p, t);

    let swap = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(ay, _mm512_abs_ps(x));
    let swapped = _mm512_sub_ps(_mm512_set1_ps(FRAC_PI_2), r);
    r = _mm512_mask_blend_ps(swap, r, swapped);
    // Mirror where x's sign bit is set (vpmovd2m reads sign bits).
    let xneg = _mm512_movepi32_mask(_mm512_castps_si512(x));
    let mirrored = _mm512_sub_ps(_mm512_set1_ps(PI), r);
    r = _mm512_mask_blend_ps(xneg, r, mirrored);
    // r ^ (y & signmask) in one ternary-logic op (imm 0x78 = A ^ (B & C)).
    _mm512_castsi512_ps(_mm512_ternarylogic_epi32::<0x78>(
        _mm512_castps_si512(r),
        _mm512_castps_si512(y),
        _mm512_set1_epi32(i32::MIN),
    ))
}

/// `log2(x)` for positive `x` via `vgetexpps`/`vgetmantps`.
#[target_feature(
    enable = "avx512f",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "avx512dq"
)]
#[inline]
fn log2_avx512(x: __m512) -> __m512 {
    let e = _mm512_getexp_ps(x);
    let m = _mm512_getmant_ps::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_ZERO>(x);
    let f = _mm512_sub_ps(m, _mm512_set1_ps(1.0));
    let mut p = _mm512_set1_ps(LOG2_COEFFS[4]);
    for c in LOG2_COEFFS[..4].iter().rev() {
        p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(*c));
    }
    _mm512_add_ps(e, p)
}

// ---- Shared narrow phase (128-bit) ----

/// Walk the anisotropic tap line for one texel and average the trilinear
/// taps. Inlined into both wide kernels.
///
/// # Safety
/// The SSE4.1 and FMA features must be available (implied by both callers).
#[target_feature(enable = "sse4.1", enable = "fma")]
#[inline]
unsafe fn fetch_aniso(levels: &[LevelInfo<'_>], cmd: &TexelCmd) -> [f32; 4] {
    let taps = 1u32 << cmd.taps_log2;
    let center = 0.5 * (taps - 1) as f32;
    let mut acc = _mm_setzero_ps();
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
        acc = _mm_add_ps(acc, s);
    }
    let avg = _mm_mul_ps(acc, _mm_set1_ps(1.0 / taps as f32));
    let mut out = [0.0f32; 4];
    // SAFETY: the array holds exactly 4 lanes.
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), avg) };
    out
}

/// # Safety
/// The SSE4.1 and FMA features must be available.
#[target_feature(enable = "sse4.1", enable = "fma")]
#[inline]
unsafe fn trilinear(levels: &[LevelInfo<'_>], u: f32, v: f32, lod: f32) -> __m128 {
    // lod is pre-clamped to [0, levels.len() - 1].
    let l0 = lod as usize;
    let frac = lod - l0 as f32;
    // SAFETY: propagated from the caller; l0 is in range by the clamp.
    let c0 = unsafe { bilinear(levels.get_unchecked(l0), u, v) };
    if frac <= 0.0 {
        return c0;
    }
    let l1 = (l0 + 1).min(levels.len() - 1);
    // SAFETY: as above.
    let c1 = unsafe { bilinear(levels.get_unchecked(l1), u, v) };
    _mm_fmadd_ps(_mm_sub_ps(c1, c0), _mm_set1_ps(frac), c0)
}

/// Bilinear tap with horizontal wrap and vertical clamp, evaluated as one
/// whole-RGBA 128-bit lerp tree.
///
/// # Safety
/// The SSE4.1 and FMA features must be available.
#[target_feature(enable = "sse4.1", enable = "fma")]
#[inline]
unsafe fn bilinear(lv: &LevelInfo<'_>, u: f32, v: f32) -> __m128 {
    // Wrap u into [0, 1) first so tap lines that cross the seam (or loop
    // the sphere near the poles) land on valid columns.
    let uw = u - u.floor();
    let up = uw * lv.wf - 0.5;
    let vp = v.clamp(0.0, 1.0) * lv.hf - 0.5;
    let x0f = up.floor();
    let y0f = vp.floor();
    let fx = up - x0f;
    let fy = vp - y0f;

    // x0 ∈ [-1, w-1] after the wrap above; select-based wrap on both edges.
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
    // SAFETY: x0/x1 < w and y0/y1 < h by the wrap/clamp above, so every
    // index is in bounds of the level's pixel slice.
    let (p00, p01, p10, p11) = unsafe {
        (
            _mm_loadu_ps(lv.px.get_unchecked(row0 + x0).as_ptr()),
            _mm_loadu_ps(lv.px.get_unchecked(row0 + x1).as_ptr()),
            _mm_loadu_ps(lv.px.get_unchecked(row1 + x0).as_ptr()),
            _mm_loadu_ps(lv.px.get_unchecked(row1 + x1).as_ptr()),
        )
    };
    let fxv = _mm_set1_ps(fx);
    let top = _mm_fmadd_ps(_mm_sub_ps(p01, p00), fxv, p00);
    let bot = _mm_fmadd_ps(_mm_sub_ps(p11, p10), fxv, p10);
    _mm_fmadd_ps(_mm_sub_ps(bot, top), _mm_set1_ps(fy), top)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::x86::{has_avx2_fma, has_avx512};

    #[target_feature(enable = "avx2", enable = "fma")]
    fn atan2_avx2_arr(y: [f32; 8], x: [f32; 8]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        // SAFETY: the arrays hold exactly 8 lanes.
        unsafe {
            let r = atan2_avx2(_mm256_loadu_ps(y.as_ptr()), _mm256_loadu_ps(x.as_ptr()));
            _mm256_storeu_ps(out.as_mut_ptr(), r);
        }
        out
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn log2_avx2_arr(x: [f32; 8]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        // SAFETY: the arrays hold exactly 8 lanes.
        unsafe {
            let r = log2_avx2(_mm256_loadu_ps(x.as_ptr()));
            _mm256_storeu_ps(out.as_mut_ptr(), r);
        }
        out
    }

    #[target_feature(
        enable = "avx512f",
        enable = "avx512vl",
        enable = "avx512bw",
        enable = "avx512dq"
    )]
    fn atan2_avx512_arr(y: [f32; 16], x: [f32; 16]) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        // SAFETY: the arrays hold exactly 16 lanes.
        unsafe {
            let r = atan2_avx512(_mm512_loadu_ps(y.as_ptr()), _mm512_loadu_ps(x.as_ptr()));
            _mm512_storeu_ps(out.as_mut_ptr(), r);
        }
        out
    }

    #[target_feature(
        enable = "avx512f",
        enable = "avx512vl",
        enable = "avx512bw",
        enable = "avx512dq"
    )]
    fn log2_avx512_arr(x: [f32; 16]) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        // SAFETY: the arrays hold exactly 16 lanes.
        unsafe {
            let r = log2_avx512(_mm512_loadu_ps(x.as_ptr()));
            _mm512_storeu_ps(out.as_mut_ptr(), r);
        }
        out
    }

    /// Grid of interesting (y, x) pairs: all quadrants, axes, signed zero,
    /// and magnitude spreads.
    fn atan2_cases() -> Vec<(f32, f32)> {
        let mut vals = vec![0.0f32, -0.0];
        for e in [-3i32, -1, 0, 1, 4] {
            let m = 2f32.powi(e);
            for f in [1.0f32, 1.37, 1.93] {
                vals.push(m * f);
                vals.push(-m * f);
            }
        }
        let mut cases = Vec::new();
        for &y in &vals {
            for &x in &vals {
                cases.push((y, x));
            }
        }
        cases
    }

    const ATAN2_TOL: f32 = 3e-7;

    fn check_atan2(chunk: &[(f32, f32)], got: &[f32]) {
        for (i, &(y, x)) in chunk.iter().enumerate() {
            let want = y.atan2(x);
            assert!(
                (got[i] - want).abs() <= ATAN2_TOL,
                "atan2({y}, {x}): got {} want {want}",
                got[i],
            );
        }
    }

    #[test]
    fn atan2_avx2_matches_std() {
        if !has_avx2_fma() {
            eprintln!("skipping: no avx2+fma");
            return;
        }
        for chunk in atan2_cases().chunks(8) {
            let mut y = [1.0f32; 8];
            let mut x = [1.0f32; 8];
            for (i, &(cy, cx)) in chunk.iter().enumerate() {
                y[i] = cy;
                x[i] = cx;
            }
            // SAFETY: checked above.
            let got = unsafe { atan2_avx2_arr(y, x) };
            check_atan2(chunk, &got);
        }
    }

    #[test]
    fn atan2_avx512_matches_std() {
        if !has_avx512() {
            eprintln!("skipping: no avx512");
            return;
        }
        for chunk in atan2_cases().chunks(16) {
            let mut y = [1.0f32; 16];
            let mut x = [1.0f32; 16];
            for (i, &(cy, cx)) in chunk.iter().enumerate() {
                y[i] = cy;
                x[i] = cx;
            }
            // SAFETY: checked above.
            let got = unsafe { atan2_avx512_arr(y, x) };
            check_atan2(chunk, &got);
        }
    }

    fn log2_cases() -> Vec<f32> {
        let mut cases = Vec::new();
        for e in -30..30 {
            for f in [1.0f32, 1.111, 1.5, 1.999] {
                cases.push(2f32.powi(e) * f);
            }
        }
        cases
    }

    const LOG2_TOL: f32 = 3e-4;

    fn check_log2(chunk: &[f32], got: &[f32]) {
        for (i, &x) in chunk.iter().enumerate() {
            let want = x.log2();
            assert!(
                (got[i] - want).abs() <= LOG2_TOL,
                "log2({x}): got {} want {want}",
                got[i],
            );
        }
    }

    #[test]
    fn log2_avx2_matches_std() {
        if !has_avx2_fma() {
            eprintln!("skipping: no avx2+fma");
            return;
        }
        for chunk in log2_cases().chunks(8) {
            let mut x = [1.0f32; 8];
            x[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: checked above.
            let got = unsafe { log2_avx2_arr(x) };
            check_log2(chunk, &got);
        }
    }

    #[test]
    fn log2_avx512_matches_std() {
        if !has_avx512() {
            eprintln!("skipping: no avx512");
            return;
        }
        for chunk in log2_cases().chunks(16) {
            let mut x = [1.0f32; 16];
            x[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: checked above.
            let got = unsafe { log2_avx512_arr(x) };
            check_log2(chunk, &got);
        }
    }
}
