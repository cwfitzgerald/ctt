use std::f32::consts::PI;

use super::super::buffer::Buffer;
use super::*;
use crate::processing::kernels::constructible_levels;
use crate::processing::kernels::equirectangular::{map_texel_at, project_f32_at};

/// Direction for normalized equirectangular coordinates — the inverse of the
/// kernel's dir→(u, v) mapping, used to synthesize test sources.
fn equirectangular_dir(u: f32, v: f32) -> [f32; 3] {
    let phi = (u - 0.5) * 2.0 * PI;
    let theta = v * PI;
    [
        theta.sin() * phi.sin(),
        theta.cos(),
        theta.sin() * phi.cos(),
    ]
}

/// Build an equirectangular source by evaluating `f` at every texel center.
fn synth(w: u32, h: u32, f: impl Fn(f32, f32) -> [f32; 4]) -> Buffer<f32> {
    let mut pixels = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            pixels.push(f(u, v));
        }
    }
    Buffer {
        pixels,
        width: w,
        height: h,
    }
}

fn face_texel(face: &Buffer<f32>, x: u32, y: u32) -> [f32; 4] {
    face.pixels[(y * face.width + x) as usize]
}

const FACE_COLORS: [[f32; 4]; 6] = [
    [1.0, 0.0, 0.0, 1.0], // +X
    [0.0, 1.0, 1.0, 1.0], // -X
    [0.0, 1.0, 0.0, 1.0], // +Y
    [1.0, 0.0, 1.0, 1.0], // -Y
    [0.0, 0.0, 1.0, 1.0], // +Z
    [1.0, 1.0, 0.0, 1.0], // -Z
];

/// Source colored by the dominant axis of each texel's direction.
fn axis_colored_equirectangular(w: u32, h: u32) -> Buffer<f32> {
    synth(w, h, |u, v| {
        let [x, y, z] = equirectangular_dir(u, v);
        let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
        let face = if ax >= ay && ax >= az {
            if x >= 0.0 { 0 } else { 1 }
        } else if ay >= ax && ay >= az {
            if y >= 0.0 { 2 } else { 3 }
        } else if z >= 0.0 {
            4
        } else {
            5
        };
        FACE_COLORS[face]
    })
}

#[test]
fn solid_color_projects_to_solid_faces() {
    let src = synth(64, 32, |_, _| [3.0, 5.0, 0.25, 11.0]);
    let pyr = EquirectangularPyramid::new(src).unwrap();
    let faces = project_f32(&pyr, 16, EquirectangularOrientation::default()).unwrap();
    for (fi, face) in faces.iter().enumerate() {
        assert_eq!(face.width, 16);
        assert_eq!(face.height, 16);
        for (i, px) in face.pixels.iter().enumerate() {
            for c in 0..4 {
                let want = [3.0, 5.0, 0.25, 11.0][c];
                assert!(
                    (px[c] - want).abs() < 1e-4,
                    "face {fi} texel {i} chan {c}: {} vs {want}",
                    px[c],
                );
            }
        }
    }
}

#[test]
fn face_centers_match_vulkan_orientation() {
    let src = axis_colored_equirectangular(128, 64);
    let pyr = EquirectangularPyramid::new(src).unwrap();
    let n = 17; // odd: the center texel direction is exactly the face axis
    for (label, level) in constructible_levels() {
        let faces = project_f32_at(level, &pyr, n, EquirectangularOrientation::default());
        for (fi, face) in faces.iter().enumerate() {
            let center = face_texel(face, n / 2, n / 2);
            for c in 0..4 {
                assert!(
                    (center[c] - FACE_COLORS[fi][c]).abs() < 0.08,
                    "{label}: face {fi} center chan {c}: {} vs {}",
                    center[c],
                    FACE_COLORS[fi][c],
                );
            }
        }
    }
}

/// The source stores its own (u, v); face centers must read back the
/// coordinates the mapping predicts.
#[test]
fn face_centers_sample_expected_coordinates() {
    let src = synth(128, 64, |u, v| [u, v, 0.0, 1.0]);
    let pyr = EquirectangularPyramid::new(src).unwrap();
    let n = 17;
    // (face, expected u, expected v); u is None where the face center sits
    // on the wrap seam and the ramp image is discontinuous.
    let cases: [(usize, Option<f32>, f32); 6] = [
        (0, Some(0.75), 0.5), // +X
        (1, Some(0.25), 0.5), // -X
        (2, None, 0.05),      // +Y: near the top pole
        (3, None, 0.95),      // -Y: near the bottom pole
        (4, Some(0.5), 0.5),  // +Z
        (5, None, 0.5),       // -Z: u seam
    ];
    for (label, level) in constructible_levels() {
        let faces = project_f32_at(level, &pyr, n, EquirectangularOrientation::default());
        for (fi, want_u, want_v) in cases {
            let got = face_texel(&faces[fi], n / 2, n / 2);
            if let Some(wu) = want_u {
                assert!(
                    (got[0] - wu).abs() < 0.02,
                    "{label}: face {fi} u: {} vs {wu}",
                    got[0],
                );
            }
            let v_tol = if want_v == 0.5 { 0.02 } else { 0.1 };
            assert!(
                (got[1] - want_v).abs() < v_tol,
                "{label}: face {fi} v: {} vs {want_v}",
                got[1],
            );
        }
    }
}

/// A smooth longitude-only pattern must cross the ±π seam (center of the
/// -Z face) without a jump, on every backend.
#[test]
fn seam_is_continuous_on_neg_z_face() {
    let pyr = EquirectangularPyramid::new(test_support::seam_source()).unwrap();
    for (label, level) in constructible_levels() {
        let faces = project_f32_at(
            level,
            &pyr,
            test_support::SEAM_FACE_SIZE,
            EquirectangularOrientation::default(),
        );
        test_support::assert_neg_z_seam_continuous(&faces, test_support::SEAM_FACE_SIZE, label);
    }
}

/// Alternating columns (the worst horizontal-aliasing case) must converge
/// to their mean near the poles instead of sparkling, on every backend.
#[test]
fn anisotropic_filter_averages_pole_region() {
    let pyr = EquirectangularPyramid::new(test_support::stripe_source()).unwrap();
    for (label, level) in constructible_levels() {
        let faces = project_f32_at(
            level,
            &pyr,
            test_support::POLE_FACE_SIZE,
            EquirectangularOrientation::default(),
        );
        test_support::assert_pole_region_averaged(&faces, test_support::POLE_FACE_SIZE, label);
    }
}

#[test]
fn map_texel_equator_is_single_tap() {
    // 2:1 source, face = width/4: sampling rates match at the equator, so
    // the footprint is nearly isotropic there.
    let pyr = EquirectangularPyramid::new(synth(128, 64, |_, _| [0.0; 4])).unwrap();
    let n = 32;
    let params = ProjectParams::new(&pyr, n);
    for (label, level) in constructible_levels() {
        let cmd = map_texel_at(level, &FACE_BASES[4], 0.0, 0.0, &params); // +Z center
        assert_eq!(cmd.taps_log2, 0, "{label}: equator should be isotropic");
        assert!((cmd.u - 0.5).abs() < 1e-6, "{label}: u = {}", cmd.u);
        assert!((cmd.v - 0.5).abs() < 1e-6, "{label}: v = {}", cmd.v);
        assert!(
            (0.0..1.0).contains(&cmd.lod),
            "{label}: equator lod should be mild, got {}",
            cmd.lod,
        );
    }
}

#[test]
fn map_texel_pole_hits_aniso_cap() {
    let pyr = EquirectangularPyramid::new(synth(128, 64, |_, _| [0.0; 4])).unwrap();
    let params = ProjectParams::new(&pyr, 32);
    for (label, level) in constructible_levels() {
        // Just off the +Y face center: extreme anisotropy.
        let cmd = map_texel_at(level, &FACE_BASES[2], 1e-3, 1e-3, &params);
        assert_eq!(
            cmd.taps_log2, MAX_ANISO_LOG2,
            "{label}: pole should cap taps"
        );
        assert!(cmd.step_u.is_finite() && cmd.step_v.is_finite(), "{label}");
        assert!(cmd.lod.is_finite() && cmd.lod >= 0.0, "{label}");
    }
}

#[test]
fn rejects_bad_inputs() {
    assert!(
        EquirectangularPyramid::new(Buffer {
            pixels: vec![],
            width: 0,
            height: 1,
        })
        .is_err()
    );
    assert!(
        EquirectangularPyramid::new(Buffer {
            pixels: vec![[0.0; 4]; 3],
            width: 2,
            height: 2,
        })
        .is_err()
    );
    let pyr = EquirectangularPyramid::new(synth(4, 2, |_, _| [0.0; 4])).unwrap();
    assert!(project_f32(&pyr, 0, EquirectangularOrientation::default()).is_err());
    assert!(project_f32(&pyr, 1 << 20, EquirectangularOrientation::default()).is_err());
}

#[test]
fn default_face_size_is_quarter_width() {
    let pyr = EquirectangularPyramid::new(synth(128, 64, |_, _| [0.0; 4])).unwrap();
    assert_eq!(pyr.default_face_size(), 32);
    let tiny = EquirectangularPyramid::new(synth(2, 1, |_, _| [0.0; 4])).unwrap();
    assert_eq!(tiny.default_face_size(), 1);
}

/// Every orientation places the pano center (`u = 0.5`) on its front
/// face, and `u = 0.75` (a quarter turn along the longitude direction)
/// on the expected neighbor.
#[test]
fn orientation_conventions_place_expected_faces() {
    let src = synth(128, 64, |u, v| [u, v, 0.0, 1.0]);
    let pyr = EquirectangularPyramid::new(src).unwrap();
    let n = 17;
    // (front, mirror, face index at u=0.5, face index at u=0.75)
    let cases = [
        (EquirectangularFront::PosZ, false, 4, 0), // +Z front, quarter turn → +X
        (EquirectangularFront::PosZ, true, 4, 1),  // mirrored → -X
        (EquirectangularFront::NegZ, false, 5, 1), // -Z front → -X
        (EquirectangularFront::PosX, false, 0, 5), // +X front → -Z
        (EquirectangularFront::PosX, true, 0, 4),  // three.js / glTF IBL: +X → +Z
        (EquirectangularFront::NegX, false, 1, 4), // -X front → +Z
    ];
    for (label, level) in constructible_levels() {
        for (front, mirror, front_face, quarter_face) in cases {
            let orientation = EquirectangularOrientation { front, mirror };
            let faces = project_f32_at(level, &pyr, n, orientation);
            let center = face_texel(&faces[front_face], n / 2, n / 2);
            assert!(
                (center[0] - 0.5).abs() < 0.02,
                "{label} {orientation:?}: front face {front_face} reads u = {}",
                center[0],
            );
            let quarter = face_texel(&faces[quarter_face], n / 2, n / 2);
            assert!(
                (quarter[0] - 0.75).abs() < 0.02,
                "{label} {orientation:?}: quarter-turn face {quarter_face} reads u = {}",
                quarter[0],
            );
        }
    }
}

/// `PosX` + mirror must reproduce the three.js / Khronos glTF-IBL-Sampler
/// mapping `u = 0.5 + atan2(d.z, d.x)/2π` exactly, for every face.
#[test]
fn pos_x_mirror_matches_threejs_formula() {
    let pyr = EquirectangularPyramid::new(synth(128, 64, |_, _| [0.0; 4])).unwrap();
    let params = ProjectParams::new(&pyr, 32);
    let bases = oriented_bases(EquirectangularOrientation {
        front: EquirectangularFront::PosX,
        mirror: true,
    });
    for (label, level) in constructible_levels() {
        for face in 0..6 {
            for (a, b) in [(0.3f32, -0.4), (-0.7, 0.2), (0.0, 0.9)] {
                let cmd = map_texel_at(level, &bases[face], a, b, &params);
                // World-space sampling direction from the untransformed basis.
                let w = &FACE_BASES[face];
                let d: [f32; 3] = std::array::from_fn(|i| w.axis[i] + a * w.u[i] + b * w.v[i]);
                let want = 0.5 + d[2].atan2(d[0]) / (2.0 * PI);
                assert!(
                    (cmd.u - want).abs() < 1e-6,
                    "{label}: face {face} ({a}, {b}): u = {} want {want}",
                    cmd.u,
                );
            }
        }
    }
}

// ---- SIMD full projection ----

fn smooth_equirect(w: u32, h: u32) -> Buffer<f32> {
    let mut pixels = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            let phi = (u - 0.5) * 2.0 * PI;
            let theta = v * PI;
            pixels.push([
                phi.sin() * 0.5 + 0.5,
                theta.cos() * 0.5 + 0.5,
                (phi.cos() * theta.sin()) * 0.5 + 0.5,
                1.0,
            ]);
        }
    }
    Buffer {
        pixels,
        width: w,
        height: h,
    }
}

fn assert_faces_close(a: &[Buffer<f32>; 6], b: &[Buffer<f32>; 6], tol: f32, what: &str) {
    let mut max_diff = 0.0f32;
    for (fa, fb) in a.iter().zip(b) {
        for (pa, pb) in fa.pixels.iter().zip(&fb.pixels) {
            for c in 0..4 {
                max_diff = max_diff.max((pa[c] - pb[c]).abs());
            }
        }
    }
    assert!(max_diff < tol, "{what}: max diff {max_diff} >= {tol}");
}

/// Cross-level agreement bound. The floor is set by tap-count quantization, not
/// raw polynomial error: near an anisotropy boundary a different `log2` rounding
/// can nudge `delta.ceil()` so two backends pick different tap counts, and
/// averaging a different number of taps moves the result by more than the
/// transcendental error alone.
const CROSS_LEVEL_TOL: f32 = 5e-3;

/// Every backend must agree with the lowest constructible one. Not tautological:
/// the tiers run different vector widths, FMA availability, and (on the store
/// side of other kernels) different instruction sequences, and the tail handling
/// differs per width. The base is taken from `constructible_levels` rather than
/// a hand-built `Fallback` token, which `dispatch!` would normalize to the
/// baseline backend on targets like aarch64.
#[test]
fn levels_agree_across_backends() {
    let pyr = EquirectangularPyramid::new(smooth_equirect(128, 64)).unwrap();
    let mirrored = EquirectangularOrientation {
        front: EquirectangularFront::PosX,
        mirror: true,
    };
    let levels = constructible_levels();
    let (base_label, base_level) = levels[0];
    for orientation in [EquirectangularOrientation::default(), mirrored] {
        let base = project_f32_at(base_level, &pyr, 33, orientation);
        for &(label, level) in &levels[1..] {
            let got = project_f32_at(level, &pyr, 33, orientation);
            assert_faces_close(
                &base,
                &got,
                CROSS_LEVEL_TOL,
                &format!("{label} vs {base_label} {orientation:?}"),
            );
        }
    }
}

/// A solid-color source must project exactly (within trilinear rounding) on
/// every level. Face size 19 leaves a tail at every native width; 16 covers
/// the tail-free path.
#[test]
fn simd_solid_color_is_exact() {
    let pyr = EquirectangularPyramid::new(smooth_solid()).unwrap();
    let check = |faces: &[Buffer<f32>; 6], what: &str| {
        for face in faces {
            for px in &face.pixels {
                for c in 0..4 {
                    let want = [0.125, 2.0, -4.5, 1.0][c];
                    assert!((px[c] - want).abs() < 1e-4, "{what}: {} vs {want}", px[c]);
                }
            }
        }
    };
    for (label, level) in constructible_levels() {
        for face_size in [16u32, 19] {
            let faces = project_f32_at(
                level,
                &pyr,
                face_size,
                EquirectangularOrientation::default(),
            );
            check(&faces, &format!("{label} n={face_size}"));
        }
    }
}

fn smooth_solid() -> Buffer<f32> {
    Buffer {
        pixels: vec![[0.125, 2.0, -4.5, 1.0]; 64 * 32],
        width: 64,
        height: 32,
    }
}
