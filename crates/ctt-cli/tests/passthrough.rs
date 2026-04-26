//! Passthrough fast path: format-preserving operations stay byte-equal
//! (or payload-equal across container conversion).

use ctt::{AlphaMode, ColorSpace, Format};

use crate::common::synth::{synth_compressed, synth_uncompressed, write_ktx2};
use crate::common::{TestFixture, assert, read, run_cli};

/// Synthesize a KTX2 file of the requested compressed `format`, then run
/// the CLI in passthrough mode (no `-f`) and assert byte-identical output.
fn assert_compressed_ktx2_passthrough(format: Format) {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    write_ktx2(synth_compressed(format, 8, 8), &input);

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
        "compressed KTX2→KTX2 passthrough must be byte-identical for {format:?}"
    );
}

/// Synthesize a KTX2 file of the requested uncompressed `format`, then run
/// the CLI in passthrough mode (no `-f`) and assert byte-identical output.
fn assert_uncompressed_ktx2_passthrough(format: Format, color_space: ColorSpace) {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    write_ktx2(
        synth_uncompressed(format, 8, 8, color_space, AlphaMode::Straight),
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
        "uncompressed KTX2→KTX2 passthrough must be byte-identical for {format:?}/{color_space:?}"
    );
}

/// BC7 KTX2 in → KTX2 out (no `-f`): output bytes must match input bytes
/// exactly. Container metadata (linear color space) flows through without
/// the user needing to pass `--input-color-space`.
#[test]
fn bc7_ktx2_to_ktx2_byte_equal() {
    let f = TestFixture::new();
    let input = f.data_file("bc7_4x4.ktx2");
    let output = f.output_file("out.ktx2");

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
        "passthrough must be byte-identical"
    );
}

/// RGBA8 sRGB KTX2 in → KTX2 out: byte-identical when the container
/// metadata is honored (the default).
#[test]
fn rgba8_srgb_ktx2_to_ktx2_byte_equal() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_srgb.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    assert_eq!(read(&input), read(&output));
}

/// KTX2 cubemap input → KTX2 out (no `--cubemap`): the cubemap nature
/// of the input is preserved by passthrough.
#[test]
fn cubemap_ktx2_passthrough_preserves_face_count() {
    let f = TestFixture::new();
    let input = f.data_file("cube_palette_16.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(info.face_count, 6, "cubemap face count must be preserved");
    assert_eq!(info.width, 16);
    assert_eq!(info.height, 16);
}

/// Explicit `--input-color-space` still overrides container metadata.
/// Linear input + `--input-color-space srgb` re-tags the surface and
/// changes the DFD transfer function — output bytes must differ.
#[test]
fn explicit_input_color_space_overrides_container_metadata() {
    let f = TestFixture::new();
    let input = f.data_file("bc7_4x4.ktx2"); // synthesized as linear
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--input-color-space",
        "srgb",
    ])
    .expect("run succeeded");

    let out_bytes = read(&output);
    assert_ne!(
        read(&input),
        out_bytes,
        "explicit override should re-tag the surface"
    );

    let img = assert::decode(&out_bytes);
    assert_eq!(img.surfaces[0][0].color_space, ctt::ColorSpace::Srgb);
}

// One representative test per supported compressed format. Each builds a
// synthetic 8×8 KTX2 input and asserts the CLI passthrough is byte-identical.
// Only one block-size per family is exercised; ASTC block-size variants and
// quality presets are passthrough metadata.

#[test]
fn bc1_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::BC1_RGBA_UNORM_BLOCK);
}

#[test]
fn bc3_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::BC3_UNORM_BLOCK);
}

#[test]
fn bc4_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::BC4_UNORM_BLOCK);
}

#[test]
fn bc5_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::BC5_UNORM_BLOCK);
}

#[test]
fn bc6h_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::BC6H_UFLOAT_BLOCK);
}

#[test]
fn bc7_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::BC7_UNORM_BLOCK);
}

#[test]
fn etc2_rgba_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::ETC2_R8G8B8A8_UNORM_BLOCK);
}

#[test]
fn eac_r_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::EAC_R11_UNORM_BLOCK);
}

#[test]
fn eac_rg_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::EAC_R11G11_UNORM_BLOCK);
}

#[test]
fn astc_4x4_passthrough_byte_equal() {
    assert_compressed_ktx2_passthrough(Format::ASTC_4x4_UNORM_BLOCK);
}

// One representative per uncompressed format family. Synthesized KTX2
// inputs round-trip byte-identical through CLI passthrough.

#[test]
fn r8g8b8a8_unorm_identity() {
    assert_uncompressed_ktx2_passthrough(Format::R8G8B8A8_UNORM, ColorSpace::Linear);
}

#[test]
fn r8g8b8a8_srgb_identity() {
    assert_uncompressed_ktx2_passthrough(Format::R8G8B8A8_UNORM, ColorSpace::Srgb);
}

#[test]
fn r8g8b8a8_snorm_identity() {
    assert_uncompressed_ktx2_passthrough(Format::R8G8B8A8_SNORM, ColorSpace::Linear);
}

#[test]
fn bgra8_unorm_identity() {
    assert_uncompressed_ktx2_passthrough(Format::B8G8R8A8_UNORM, ColorSpace::Linear);
}

#[test]
fn r16g16b16a16_unorm_identity() {
    assert_uncompressed_ktx2_passthrough(Format::R16G16B16A16_UNORM, ColorSpace::Linear);
}

#[test]
fn r32g32b32a32_sfloat_identity() {
    assert_uncompressed_ktx2_passthrough(Format::R32G32B32A32_SFLOAT, ColorSpace::Linear);
}

#[test]
fn r32g32b32a32_uint_identity() {
    assert_uncompressed_ktx2_passthrough(Format::R32G32B32A32_UINT, ColorSpace::Linear);
}
