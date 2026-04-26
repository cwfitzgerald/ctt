//! Array texture assembly tests.
//!
//! Multiple non-cubemap inputs collapse into a single array texture with one
//! layer per input.

use ctt::{AlphaMode, ColorSpace, Format};

use crate::common::synth::{make_image, write_ktx2};
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
    assert!(!decoded.is_cubemap);
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
