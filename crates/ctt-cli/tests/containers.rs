//! Container conversion tests (KTX2 ↔ DDS).

use ctt::Format;

use crate::common::synth::{synth_compressed, write_dds, write_ktx2};
use crate::common::{TestFixture, assert, read, run_cli};

/// Synthesize a KTX2 input of `format` and convert it to a DDS output via
/// the CLI; verify DDS magic and per-surface payload equality.
fn assert_ktx2_to_dds_roundtrip(format: Format) {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.dds");

    write_ktx2(synth_compressed(format, 8, 8), &input);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let out_bytes = read(&output);
    assert!(
        out_bytes.starts_with(assert::DDS_MAGIC),
        "{format:?}: output must start with DDS magic"
    );
    assert::assert_payload_eq(&read(&input), &out_bytes);
}

/// Synthesize a DDS input of `format` and convert it to a KTX2 output via
/// the CLI; verify KTX2 magic, format preserved, and payload equality.
fn assert_dds_to_ktx2_roundtrip(format: Format) {
    let f = TestFixture::new();
    let input = f.output_file("input.dds");
    let output = f.output_file("output.ktx2");

    write_dds(synth_compressed(format, 8, 8), &input);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let out_bytes = read(&output);
    assert!(
        out_bytes.starts_with(assert::KTX2_MAGIC),
        "{format:?}: output must start with KTX2 magic"
    );
    let info = assert::parse_ktx2(&out_bytes);
    assert_eq!(
        info.format,
        Some(format),
        "{format:?}: KTX2 header format mismatch"
    );
    assert::assert_payload_eq(&read(&input), &out_bytes);
}

/// BC7 KTX2 in → DDS out: payload bytes survive through the container
/// conversion and the DDS file has a valid header.
#[test]
fn bc7_ktx2_to_dds_payload_preserved() {
    let f = TestFixture::new();
    let input = f.data_file("bc7_4x4.ktx2");
    let output = f.output_file("out.dds");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let out_bytes = read(&output);
    assert!(
        out_bytes.starts_with(assert::DDS_MAGIC),
        "output must start with DDS magic"
    );

    let dds_info = assert::parse_dds(&out_bytes);
    assert_eq!(dds_info.width, 4);
    assert_eq!(dds_info.height, 4);

    assert::assert_payload_eq(&read(&input), &out_bytes);
}

// ── Container conversion sweep (KTX2 ↔ DDS) ──────────────────────────────
//
// Each non-ETC compressed format from the Phase 1 list, both directions.
// DDS does not support ETC; that error case is covered in errors.rs.

#[test]
fn bc1_ktx2_to_dds() {
    assert_ktx2_to_dds_roundtrip(Format::BC1_RGBA_UNORM_BLOCK);
}

#[test]
fn bc1_dds_to_ktx2() {
    assert_dds_to_ktx2_roundtrip(Format::BC1_RGBA_UNORM_BLOCK);
}

#[test]
fn bc3_ktx2_to_dds() {
    assert_ktx2_to_dds_roundtrip(Format::BC3_UNORM_BLOCK);
}

#[test]
fn bc3_dds_to_ktx2() {
    assert_dds_to_ktx2_roundtrip(Format::BC3_UNORM_BLOCK);
}

#[test]
fn bc4_ktx2_to_dds() {
    assert_ktx2_to_dds_roundtrip(Format::BC4_UNORM_BLOCK);
}

#[test]
fn bc4_dds_to_ktx2() {
    assert_dds_to_ktx2_roundtrip(Format::BC4_UNORM_BLOCK);
}

#[test]
fn bc5_ktx2_to_dds() {
    assert_ktx2_to_dds_roundtrip(Format::BC5_UNORM_BLOCK);
}

#[test]
fn bc5_dds_to_ktx2() {
    assert_dds_to_ktx2_roundtrip(Format::BC5_UNORM_BLOCK);
}

#[test]
fn bc6h_ktx2_to_dds() {
    assert_ktx2_to_dds_roundtrip(Format::BC6H_UFLOAT_BLOCK);
}

#[test]
fn bc6h_dds_to_ktx2() {
    assert_dds_to_ktx2_roundtrip(Format::BC6H_UFLOAT_BLOCK);
}

#[test]
fn bc7_ktx2_to_dds() {
    assert_ktx2_to_dds_roundtrip(Format::BC7_UNORM_BLOCK);
}

#[test]
fn bc7_dds_to_ktx2() {
    assert_dds_to_ktx2_roundtrip(Format::BC7_UNORM_BLOCK);
}

#[test]
fn astc_4x4_ktx2_to_dds() {
    assert_ktx2_to_dds_roundtrip(Format::ASTC_4x4_UNORM_BLOCK);
}

#[test]
fn astc_4x4_dds_to_ktx2() {
    assert_dds_to_ktx2_roundtrip(Format::ASTC_4x4_UNORM_BLOCK);
}
