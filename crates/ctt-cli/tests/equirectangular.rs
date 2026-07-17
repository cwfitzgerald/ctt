//! Equirectangular panorama → cubemap projection tests.

use std::f32::consts::PI;

use ctt::{AlphaMode, ColorSpace, Format, TextureKind};

use crate::common::synth::{make_image, write_ktx2};
use crate::common::{TestFixture, assert, read, run_cli};

/// Colors for the +X, -X, +Y, -Y, +Z, -Z axis regions of the sphere.
const AXIS_COLORS: [[f32; 4]; 6] = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
];

/// RGBA32F equirectangular where every texel is colored by the dominant axis of
/// its direction (image center faces +Z, top row is +Y).
fn axis_colored_equirectangular_rgba32f(w: u32, h: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((w * h * 16) as usize);
    for yi in 0..h {
        for xi in 0..w {
            let u = (xi as f32 + 0.5) / w as f32;
            let v = (yi as f32 + 0.5) / h as f32;
            let phi = (u - 0.5) * 2.0 * PI;
            let theta = v * PI;
            let (x, y, z) = (
                theta.sin() * phi.sin(),
                theta.cos(),
                theta.sin() * phi.cos(),
            );
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
            for c in AXIS_COLORS[face] {
                data.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    data
}

fn face_pixel_f32(data: &[u8], width: u32, x: u32, y: u32) -> [f32; 4] {
    let off = ((y * width + x) * 16) as usize;
    std::array::from_fn(|c| {
        f32::from_le_bytes(data[off + c * 4..off + c * 4 + 4].try_into().unwrap())
    })
}

/// Equirectangular KTX2 + `--cubemap --cubemap-layout equirectangular` → KTX2 cubemap
/// whose face centers carry the matching axis color in Vulkan face order.
#[test]
fn equirectangular_projects_into_cubemap() {
    let f = TestFixture::new();
    let input = f.output_file("pano.ktx2");
    write_ktx2(
        make_image(
            axis_colored_equirectangular_rgba32f(128, 64),
            128,
            64,
            Format::R32G32B32A32_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );
    let output = f.output_file("cube.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--cubemap",
        "--cubemap-layout",
        "equirectangular",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.face_count, 6, "must be a cubemap");
    assert_eq!(info.width, 32, "default face size is width / 4");
    assert_eq!(info.height, 32);
    assert_eq!(info.format, Some(ktx2::Format::R32G32B32A32_SFLOAT));

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.kind, TextureKind::Cubemap);
    assert_eq!(decoded.surfaces.len(), 6);
    for (i, want) in AXIS_COLORS.iter().enumerate() {
        let face = &decoded.surfaces[i][0];
        let center = face_pixel_f32(&face.data, face.width, 16, 16);
        for c in 0..4 {
            assert!(
                (center[c] - want[c]).abs() < 0.1,
                "face {i} center chan {c}: {} vs {}",
                center[c],
                want[c],
            );
        }
    }
}

/// `--cubemap-face-size` overrides the default face size.
#[test]
fn equirectangular_face_size_override() {
    let f = TestFixture::new();
    let input = f.output_file("pano.ktx2");
    write_ktx2(
        make_image(
            axis_colored_equirectangular_rgba32f(64, 32),
            64,
            32,
            Format::R32G32B32A32_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );
    let output = f.output_file("cube.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--cubemap",
        "--cubemap-layout",
        "equirectangular",
        "--cubemap-face-size",
        "24",
    ])
    .expect("run succeeded");

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(info.face_count, 6);
    assert_eq!(info.width, 24);
    assert_eq!(info.height, 24);
}

/// `--equirectangular-front pos-x --equirectangular-mirror` (the three.js / glTF IBL
/// convention) puts the pano center on the +X face instead of +Z. Uses the
/// short `equirect` aliases to keep them covered.
#[test]
fn equirectangular_orientation_flags_change_convention() {
    let f = TestFixture::new();
    let input = f.output_file("pano.ktx2");
    // Pano storing its own u coordinate in the red channel.
    let (w, h) = (128u32, 64u32);
    let mut data = Vec::with_capacity((w * h * 16) as usize);
    for _ in 0..h {
        for xi in 0..w {
            let u = (xi as f32 + 0.5) / w as f32;
            for c in [u, 0.0, 0.0, 1.0] {
                data.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    write_ktx2(
        make_image(
            data,
            w,
            h,
            Format::R32G32B32A32_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );
    let output = f.output_file("cube.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--cubemap",
        "--cubemap-layout",
        "equirect",
        "--cubemap-face-size",
        "17",
        "--equirect-front",
        "pos-x",
        "--equirect-mirror",
    ])
    .expect("run succeeded");

    let decoded = assert::decode(&read(&output));
    // +X face center reads the pano center (u = 0.5); +Z face center is a
    // quarter turn along the mirrored longitude (u = 0.75).
    let pos_x = &decoded.surfaces[0][0];
    let center = face_pixel_f32(&pos_x.data, pos_x.width, 8, 8);
    assert!(
        (center[0] - 0.5).abs() < 0.02,
        "+X center u = {}",
        center[0]
    );
    let pos_z = &decoded.surfaces[4][0];
    let quarter = face_pixel_f32(&pos_z.data, pos_z.width, 8, 8);
    assert!(
        (quarter[0] - 0.75).abs() < 0.02,
        "+Z center u = {}",
        quarter[0],
    );
}

/// Orientation flags without the equirectangular layout are rejected.
#[test]
fn orientation_flags_without_equirectangular_layout_error() {
    let f = TestFixture::new();
    let input = f.data_file("cross_palette_64x48.ktx2");
    let output = f.output_file("cube.ktx2");

    for flags in [
        vec!["--equirectangular-front", "pos-x"],
        vec!["--equirectangular-mirror"],
    ] {
        let mut argv = vec![
            "ctt",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--cubemap",
        ];
        argv.extend(flags.iter());
        let err = run_cli(argv).expect_err("must fail");
        assert!(
            err.to_string().contains("--cubemap-layout equirectangular"),
            "unexpected error: {err}"
        );
    }
}

/// `--cubemap-face-size` without the equirectangular layout is rejected.
#[test]
fn face_size_without_equirectangular_layout_errors() {
    let f = TestFixture::new();
    let input = f.data_file("cross_palette_64x48.ktx2");
    let output = f.output_file("cube.ktx2");

    let err = run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--cubemap",
        "--cubemap-layout",
        "cross",
        "--cubemap-face-size",
        "16",
    ])
    .expect_err("must fail");
    assert!(
        err.to_string().contains("--cubemap-face-size"),
        "unexpected error: {err}"
    );
}
