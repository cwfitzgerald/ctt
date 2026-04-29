//! Cubemap-array construction and passthrough tests.
//!
//! A cubemap array has `surfaces.len() = 6 * cube_count` and `kind =
//! Cubemap`. KTX2 stores it with `face_count = 6` and `layer_count = N`
//! where `N` is the number of cubes.

use ctt::{Format, TextureKind};

use crate::common::synth::{make_cubemap_array_rgba8, write_ktx2, write_solid_rgba8_png};
use crate::common::{TestFixture, assert, read, run_cli};

/// 2-cube KTX2 cubemap array → KTX2 byte-identical passthrough. Exercises
/// the writer's `face_count = 6, layer_count = 2` path.
#[test]
fn rgba8_cubemap_array_ktx2_passthrough_byte_equal() {
    let f = TestFixture::new();
    let input = f.output_file("cube_array_in.ktx2");
    let output = f.output_file("cube_array_out.ktx2");
    write_ktx2(make_cubemap_array_rgba8(2, 8), &input);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    assert_eq!(
        read(&input),
        read(&output),
        "cubemap-array KTX2→KTX2 passthrough must be byte-identical"
    );

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(info.face_count, 6, "cubemap array must keep face_count = 6");
    assert_eq!(
        info.layer_count, 2,
        "2-cube array must report layer_count = 2"
    );
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
}

/// 12 PNG face inputs + `--cubemap` → 2-cube KTX2 cubemap array. The CLI
/// groups inputs in groups of 6 and assembles a cubemap array, preserving
/// argv order.
#[test]
fn twelve_pngs_assemble_into_cubemap_array() {
    let f = TestFixture::new();
    // Two cubes; each face gets a unique color so we can verify ordering.
    let face_colors: [[u8; 4]; 12] = [
        // cube 0
        [255, 0, 0, 255],
        [128, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 128, 0, 255],
        [0, 0, 255, 255],
        [0, 0, 128, 255],
        // cube 1
        [255, 255, 0, 255],
        [128, 128, 0, 255],
        [0, 255, 255, 255],
        [0, 128, 128, 255],
        [255, 0, 255, 255],
        [128, 0, 128, 255],
    ];
    let inputs: Vec<_> = face_colors
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let path = f.output_file(&format!("face_{i:02}.png"));
            write_solid_rgba8_png(&path, 4, 4, *color);
            path
        })
        .collect();
    let output = f.output_file("cube_array.ktx2");

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
    assert_eq!(info.face_count, 6);
    assert_eq!(info.layer_count, 2, "12 inputs → 2 cubes");
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.kind, TextureKind::Cubemap);
    assert_eq!(decoded.surfaces.len(), 12);
    for (i, expected) in face_colors.iter().enumerate() {
        let face = &decoded.surfaces[i][0];
        let first = &face.data[..4];
        assert_eq!(
            first, expected,
            "face slot {i}: expected {expected:?}, got {first:?}"
        );
    }
}

/// 2 already-cubemap KTX2 inputs + `--cubemap` → cubemap array with 12 face
/// surfaces in input-cube order.
#[test]
fn two_cubemap_ktx2s_assemble_into_cubemap_array() {
    let f = TestFixture::new();
    let cube0_path = f.output_file("cube0.ktx2");
    let cube1_path = f.output_file("cube1.ktx2");
    write_ktx2(make_cubemap_array_rgba8(1, 8), &cube0_path);
    // Build cube 1 separately so its palette differs.
    let cube1 = {
        let mut img = make_cubemap_array_rgba8(1, 8);
        // Rotate the per-face colors so cube 1 is visually distinct.
        for (face_idx, surface) in img.surfaces.iter_mut().enumerate() {
            let rotated_color = crate::common::synth::CUBEMAP_FACE_COLORS[(face_idx + 3) % 6];
            surface[0].data = rotated_color.repeat(8 * 8);
        }
        img
    };
    write_ktx2(cube1, &cube1_path);

    let output = f.output_file("cube_array.ktx2");
    run_cli([
        "ctt",
        cube0_path.to_str().unwrap(),
        cube1_path.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--cubemap",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.face_count, 6);
    assert_eq!(info.layer_count, 2);

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.kind, TextureKind::Cubemap);
    assert_eq!(decoded.surfaces.len(), 12);
}

/// Cubemap array round-trips through DDS too. DDS represents this as
/// `array_layers = 6 * cube_count` with the cubemap caps2 bit set.
#[test]
fn rgba8_cubemap_array_ktx2_to_dds_preserves_faces() {
    let f = TestFixture::new();
    let input = f.output_file("cube_array.ktx2");
    let output = f.output_file("cube_array.dds");
    let original = make_cubemap_array_rgba8(2, 8);
    write_ktx2(original.clone(), &input);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let dds_bytes = read(&output);
    let decoded = assert::decode(&dds_bytes);
    assert_eq!(decoded.kind, TextureKind::Cubemap);
    assert_eq!(decoded.surfaces.len(), 12, "2 cubes × 6 faces");
    for (i, layer) in decoded.surfaces.iter().enumerate() {
        assert_eq!(
            layer[0].data, original.surfaces[i][0].data,
            "DDS face slot {i} payload must match the source",
        );
    }
}

/// 12 BC7 face inputs `--cubemap` → 2-cube BC7 cubemap array. Verifies the
/// compressed-array path works for cubemap arrays too.
#[test]
fn twelve_bc7_inputs_assemble_into_cubemap_array() {
    use crate::common::synth::{synth_compressed, write_ktx2 as wktx2};

    let f = TestFixture::new();
    let face_paths: Vec<_> = (0..12)
        .map(|i| f.output_file(&format!("bc7_face_{i:02}.ktx2")))
        .collect();
    for path in &face_paths {
        wktx2(synth_compressed(Format::BC7_UNORM_BLOCK, 8, 8), path);
    }
    let output = f.output_file("bc7_cube_array.ktx2");

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
    assert_eq!(info.face_count, 6);
    assert_eq!(info.layer_count, 2);
    assert_eq!(info.format, Some(ktx2::Format::BC7_UNORM_BLOCK));

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.kind, TextureKind::Cubemap);
    assert_eq!(decoded.surfaces.len(), 12);
}
