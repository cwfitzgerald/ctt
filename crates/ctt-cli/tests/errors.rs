//! Error-case tests.

use ctt::{AlphaMode, ColorSpace, Format};

use crate::common::synth::{synth_uncompressed, to_ktx2_with_mips, write_ktx2};
use crate::common::{TestFixture, run_cli, try_parse_args};

/// DDS does not support ETC formats — the CLI must reject this combination.
#[test]
fn dds_etc_rejected() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.dds");

    let result = run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        "etc2_rgba",
    ]);
    assert!(result.is_err(), "DDS + ETC should be rejected");
}

/// `--cubemap` requires either 1 or 6 inputs. Five inputs must error.
#[test]
fn cubemap_with_five_inputs_rejected() {
    let f = TestFixture::new();
    let inputs = [
        f.data_file("cube_face_pos_x.ktx2"),
        f.data_file("cube_face_neg_x.ktx2"),
        f.data_file("cube_face_pos_y.ktx2"),
        f.data_file("cube_face_neg_y.ktx2"),
        f.data_file("cube_face_pos_z.ktx2"),
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

    let result = run_cli(argv);
    assert!(result.is_err(), "5 inputs + --cubemap should be rejected");
}

/// `--swizzle` must be exactly 4 characters.
#[test]
fn swizzle_three_chars_rejected() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    let argv = [
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--swizzle",
        "bgr",
    ];

    if try_parse_args(argv).is_err() {
        return;
    }

    let result = run_cli(argv);
    assert!(result.is_err(), "3-character swizzle must be rejected");
}

/// Array assembly requires every input to have the same mip count.
/// A 2-mip input combined with a 1-mip input must error.
#[test]
fn array_mismatched_mip_counts_rejected() {
    let f = TestFixture::new();
    let multi_mip = f.output_file("multi.ktx2");
    let single_mip = f.output_file("single.ktx2");
    let output = f.output_file("array.ktx2");

    let multi_image = synth_uncompressed(
        Format::R8G8B8A8_UNORM,
        4,
        4,
        ColorSpace::Linear,
        AlphaMode::Opaque,
    );
    std::fs::write(&multi_mip, to_ktx2_with_mips(multi_image, 2)).expect("write multi");

    write_ktx2(
        synth_uncompressed(
            Format::R8G8B8A8_UNORM,
            4,
            4,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &single_mip,
    );

    let result = run_cli([
        "ctt",
        multi_mip.to_str().unwrap(),
        single_mip.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ]);
    assert!(
        result.is_err(),
        "array assembly with mismatched mip counts must be rejected"
    );
}

/// Integer (UINT) input + compression target — the float pipeline can't accept
/// integer inputs, so this must error.
#[test]
fn integer_input_to_compressed_rejected() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("out.ktx2");

    write_ktx2(
        synth_uncompressed(
            Format::R8G8B8A8_UINT,
            4,
            4,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );

    let result = run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        "bc7",
        "--quality",
        "ultra-fast",
    ]);
    assert!(result.is_err(), "integer input cannot be block-compressed");
}

/// Integer pipeline does not support `--mipmap`.
#[test]
fn integer_pipeline_with_mipmap_rejected() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("out.ktx2");

    write_ktx2(
        synth_uncompressed(
            Format::R32G32B32A32_UINT,
            4,
            4,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );

    let result = run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--mipmap",
    ]);
    assert!(
        result.is_err(),
        "integer pipeline must reject mipmap generation"
    );
}

/// Integer pipeline does not support `--output-color-space` changes.
#[test]
fn integer_pipeline_with_output_color_space_rejected() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("out.ktx2");

    write_ktx2(
        synth_uncompressed(
            Format::R32G32B32A32_UINT,
            4,
            4,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );

    let result = run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--output-color-space",
        "srgb",
    ]);
    assert!(
        result.is_err(),
        "integer pipeline must reject output color space changes"
    );
}

/// Family mismatch: UNORM input cannot be converted to UINT target.
#[test]
fn family_mismatch_unorm_to_uint_rejected() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("out.ktx2");

    write_ktx2(
        synth_uncompressed(
            Format::R8G8B8A8_UNORM,
            4,
            4,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );

    let result = run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        "r8g8b8a8_uint",
    ]);
    assert!(
        result.is_err(),
        "UNORM → UINT must be rejected as a family mismatch"
    );
}

/// Supercompression is a KTX2-only feature; combining `--zstd` with a DDS
/// output must error.
#[test]
fn supercompression_on_dds_rejected() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.dds");

    let result = run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--zstd",
    ]);
    assert!(result.is_err(), "--zstd with DDS output must be rejected");
}
