//! Cubemap construction tests.

use ctt::Format;

use crate::common::synth::{CUBEMAP_FACE_COLORS, synth_compressed, write_ktx2};
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

/// 6 BC7 KTX2 inputs + `--cubemap` produce a BC7 cubemap with all 6 faces
/// preserved at their compressed byte representation.
fn assert_compressed_cubemap_preserved(format: Format, ktx2_format: ktx2::Format) {
    let f = TestFixture::new();
    let face_paths: Vec<_> = (0..6)
        .map(|i| f.output_file(&format!("face_{i}.ktx2")))
        .collect();
    for path in &face_paths {
        write_ktx2(synth_compressed(format, 8, 8), path);
    }
    let output = f.output_file("cube.ktx2");

    let mut argv: Vec<String> = vec!["ctt".to_string()];
    for p in &face_paths {
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
    assert_eq!(info.face_count, 6, "{format:?}: must be a cubemap");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert_eq!(
        info.format,
        Some(ktx2_format),
        "{format:?}: format must be preserved"
    );

    let decoded = assert::decode(&bytes);
    assert!(decoded.is_cubemap);
    assert_eq!(decoded.surfaces.len(), 6);
    let expected = assert::decode(&read(&face_paths[0])).surfaces[0][0]
        .data
        .clone();
    for (i, layer) in decoded.surfaces.iter().enumerate() {
        assert_eq!(
            layer[0].data, expected,
            "face {i}: compressed bytes must match the source face"
        );
    }
}

#[test]
fn six_bc7_inputs_assemble_into_bc7_cubemap() {
    assert_compressed_cubemap_preserved(Format::BC7_UNORM_BLOCK, ktx2::Format::BC7_UNORM_BLOCK);
}

#[test]
fn six_bc6h_inputs_assemble_into_bc6h_cubemap() {
    assert_compressed_cubemap_preserved(Format::BC6H_UFLOAT_BLOCK, ktx2::Format::BC6H_UFLOAT_BLOCK);
}

/// Already-cubemap KTX2 input + `--cubemap` is a no-op assembly: the existing
/// cubemap is forwarded through, face count and per-face colors preserved.
#[test]
fn already_cubemap_input_with_cubemap_flag_is_passthrough() {
    let f = TestFixture::new();
    let input = f.data_file("cube_palette_16.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--cubemap",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.face_count, 6);
    assert_eq!(info.width, 16);
    assert_eq!(info.height, 16);

    let decoded = assert::decode(&bytes);
    assert!(decoded.is_cubemap);
    for (i, color) in CUBEMAP_FACE_COLORS.iter().enumerate() {
        let face = &decoded.surfaces[i][0];
        let first = &face.data[..4];
        assert_eq!(first, color, "face {i}: expected {color:?}, got {first:?}");
    }
}
