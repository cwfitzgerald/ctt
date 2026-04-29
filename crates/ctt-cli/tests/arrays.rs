//! Array texture assembly tests.
//!
//! Multiple non-cubemap inputs collapse into a single array texture with one
//! layer per input.

use ctt::{AlphaMode, ColorSpace, Format, TextureKind};

use crate::common::synth::{
    make_array_image, make_image, synth_compressed_array, write_ktx2, write_solid_rgba8_png,
};
use crate::common::{TestFixture, assert, read, run_cli};

/// Three single-layer RGBA8 inputs (red, green, blue) → 3-layer KTX2 array.
/// Per-layer payloads must match the source colors.
#[test]
fn three_inputs_assemble_into_array() {
    let f = TestFixture::new();
    let colors: [[u8; 4]; 3] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    let inputs: Vec<_> = colors
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let path = f.output_file(&format!("layer_{i}.ktx2"));
            let pixels = color.repeat(4 * 4);
            let img = make_image(
                pixels,
                4,
                4,
                Format::R8G8B8A8_UNORM,
                ColorSpace::Linear,
                AlphaMode::Opaque,
            );
            write_ktx2(img, &path);
            path
        })
        .collect();
    let output = f.output_file("array.ktx2");

    let mut argv: Vec<String> = vec!["ctt".to_string()];
    for p in &inputs {
        argv.push(p.to_str().unwrap().to_string());
    }
    argv.extend(["-o".to_string(), output.to_str().unwrap().to_string()]);

    run_cli(argv).expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.layer_count, 3, "must encode 3 layers");
    assert_eq!(info.face_count, 1, "non-cubemap");
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.kind, TextureKind::Texture2D);
    assert_eq!(decoded.surfaces.len(), 3);
    for (i, color) in colors.iter().enumerate() {
        let layer = &decoded.surfaces[i][0];
        for (px_idx, px) in layer.data.chunks_exact(4).enumerate() {
            assert_eq!(
                px, color,
                "layer {i} pixel {px_idx}: expected {color:?}, got {px:?}"
            );
        }
    }
}

/// Pre-built 3-layer KTX2 array → KTX2 out (no `-f`): byte-equal passthrough.
/// One existing array file in, one out, no transformation.
#[test]
fn rgba8_array_ktx2_passthrough_byte_equal() {
    let f = TestFixture::new();
    let layers: Vec<Vec<u8>> = (0..3)
        .map(|i| {
            let color = [
                (i * 80) as u8,
                ((i + 1) * 60) as u8,
                ((i + 2) * 40) as u8,
                255,
            ];
            color.repeat(8 * 8)
        })
        .collect();
    let input = f.output_file("array_in.ktx2");
    let output = f.output_file("array_out.ktx2");
    write_ktx2(
        make_array_image(layers, 8, 8, ColorSpace::Linear, AlphaMode::Opaque),
        &input,
    );

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
        "2D array KTX2→KTX2 passthrough must be byte-identical"
    );
}

/// 4-layer BC7 array → KTX2 passthrough: each compressed layer's bytes
/// survive intact, and the encoded file declares `layer_count=4`.
#[test]
fn bc7_array_ktx2_passthrough_byte_equal() {
    let f = TestFixture::new();
    let input = f.output_file("bc7_array_in.ktx2");
    let output = f.output_file("bc7_array_out.ktx2");
    write_ktx2(
        synth_compressed_array(Format::BC7_UNORM_BLOCK, 8, 8, 4),
        &input,
    );

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
        "compressed 2D array KTX2→KTX2 passthrough must be byte-identical"
    );

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(info.layer_count, 4, "must encode 4 layers");
    assert_eq!(info.face_count, 1);
    assert_eq!(info.format, Some(ktx2::Format::BC7_UNORM_BLOCK));
}

/// 3 RGBA8 PNG inputs (red/green/blue) → KTX2 array. Verifies the slice
/// inputs are stacked in argv order and per-layer data matches.
#[test]
fn three_pngs_assemble_into_array() {
    let f = TestFixture::new();
    let colors: [[u8; 4]; 3] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    let inputs: Vec<_> = colors
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let path = f.output_file(&format!("layer_{i}.png"));
            write_solid_rgba8_png(&path, 4, 4, *color);
            path
        })
        .collect();
    let output = f.output_file("array.ktx2");

    let mut argv: Vec<String> = vec!["ctt".to_string()];
    for p in &inputs {
        argv.push(p.to_str().unwrap().to_string());
    }
    argv.extend(["-o".to_string(), output.to_str().unwrap().to_string()]);

    run_cli(argv).expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.layer_count, 3, "must encode 3 layers");
    assert_eq!(info.face_count, 1, "non-cubemap");
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.kind, TextureKind::Texture2D);
    assert_eq!(decoded.surfaces.len(), 3);
    for (i, color) in colors.iter().enumerate() {
        let layer = &decoded.surfaces[i][0];
        for (px_idx, px) in layer.data.chunks_exact(4).enumerate() {
            assert_eq!(
                px, color,
                "layer {i} pixel {px_idx}: expected {color:?}, got {px:?}"
            );
        }
    }
}

/// 3-layer KTX2 array in → DDS array out: payloads survive across the
/// container conversion. DDS uses array_layers; per-layer payload bytes
/// must match.
#[test]
fn rgba8_array_ktx2_to_dds_preserves_layers() {
    let f = TestFixture::new();
    let layers: Vec<Vec<u8>> = (0..3)
        .map(|i| {
            let color = [255u8, (i * 80) as u8, 0, 255];
            color.repeat(8 * 8)
        })
        .collect();
    let input = f.output_file("array.ktx2");
    let output = f.output_file("array.dds");
    write_ktx2(
        make_array_image(
            layers.clone(),
            8,
            8,
            ColorSpace::Linear,
            AlphaMode::Straight,
        ),
        &input,
    );

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let dds_bytes = read(&output);
    let info = assert::parse_dds(&dds_bytes);
    assert_eq!(info.array_layers, 3, "DDS must record 3 array layers");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);

    let decoded = assert::decode(&dds_bytes);
    assert_eq!(decoded.kind, TextureKind::Texture2D);
    assert_eq!(decoded.surfaces.len(), 3);
    for (i, expected) in layers.iter().enumerate() {
        assert_eq!(
            &decoded.surfaces[i][0].data, expected,
            "DDS layer {i} payload must match the source",
        );
    }
}
