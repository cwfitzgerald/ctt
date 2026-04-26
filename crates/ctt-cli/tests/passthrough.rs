//! Passthrough fast path: format-preserving operations stay byte-equal
//! (or payload-equal across container conversion).

use crate::common::{TestFixture, assert, read, run_cli};

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
