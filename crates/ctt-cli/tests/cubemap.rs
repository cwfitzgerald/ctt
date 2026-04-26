//! Cubemap construction tests.

use crate::common::synth::CUBEMAP_FACE_COLORS;
use crate::common::{TestFixture, assert, read, run_cli};

/// 6 distinct-color KTX2 inputs + `--cubemap` → KTX2 cubemap; the
/// face-order convention (+X/-X/+Y/-Y/+Z/-Z) is preserved.
#[test]
fn six_inputs_assemble_into_cubemap() {
    let f = TestFixture::new();
    let inputs = [
        f.data_file("cube_face_pos_x.ktx2"),
        f.data_file("cube_face_neg_x.ktx2"),
        f.data_file("cube_face_pos_y.ktx2"),
        f.data_file("cube_face_neg_y.ktx2"),
        f.data_file("cube_face_pos_z.ktx2"),
        f.data_file("cube_face_neg_z.ktx2"),
    ];
    let output = f.output_file("cube.ktx2");

    let mut argv: Vec<String> = vec!["ctt".to_string()];
    for p in &inputs {
        argv.push(p.to_str().unwrap().to_string());
    }
    argv.extend([
        "-o".to_string(),
        output.to_str().unwrap().to_string(),
        "--cubemap".to_string(),
    ]);

    run_cli(argv).expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.face_count, 6, "must be a cubemap");
    assert_eq!(info.width, 16);
    assert_eq!(info.height, 16);

    let decoded = assert::decode(&bytes);
    assert!(decoded.is_cubemap);
    assert_eq!(decoded.surfaces.len(), 6);
    for (i, color) in CUBEMAP_FACE_COLORS.iter().enumerate() {
        let face = &decoded.surfaces[i][0];
        let chunks: Vec<&[u8]> = face.data.chunks_exact(4).collect();
        let mismatched = chunks.iter().filter(|p| p[..] != color[..]).count();
        assert_eq!(
            mismatched, 0,
            "face {i} expected color {color:?}, found {mismatched} mismatched pixels"
        );
    }
}

/// Cross-layout RGBA8 input + `--cubemap --cubemap-layout cross` →
/// KTX2 cubemap with each face filled with its palette color.
#[test]
fn cross_layout_splits_into_cubemap() {
    let f = TestFixture::new();
    let input = f.data_file("cross_palette_64x48.ktx2");
    let output = f.output_file("cube.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--cubemap",
        "--cubemap-layout",
        "cross",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.face_count, 6);
    assert_eq!(info.width, 16, "face dimensions are 64/4 = 16");
    assert_eq!(info.height, 16, "face dimensions are 48/3 = 16");

    let decoded = assert::decode(&bytes);
    for (i, color) in CUBEMAP_FACE_COLORS.iter().enumerate() {
        let face = &decoded.surfaces[i][0];
        let first = &face.data[..4];
        assert_eq!(
            first, color,
            "face {i} first pixel: expected {color:?}, got {first:?}"
        );
    }
}
