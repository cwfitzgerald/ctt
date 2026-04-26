//! KTX2 supercompression tests.
//!
//! Each test runs the CLI with a supercompression flag, checks that the
//! KTX2 header records the scheme, and round-trips the output through
//! `decode_container` to confirm the level data decompresses cleanly back
//! to the source bytes.

use crate::common::{TestFixture, assert, read, run_cli};

/// `--zstd` enables Zstandard supercompression. The output header records
/// `Zstandard`, and decoding round-trips the source pixel data.
#[test]
fn zstd_supercompression_roundtrip() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--zstd",
    ])
    .expect("run succeeded");

    let out_bytes = read(&output);
    let info = assert::parse_ktx2(&out_bytes);
    assert_eq!(
        info.supercompression,
        Some(ktx2::SupercompressionScheme::Zstandard),
        "header must record Zstandard supercompression"
    );

    assert::assert_payload_eq(&read(&input), &out_bytes);
}

/// `--zlib` enables ZLIB supercompression. The output header records
/// `ZLIB`, and decoding round-trips the source pixel data.
#[test]
fn zlib_supercompression_roundtrip() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--zlib",
    ])
    .expect("run succeeded");

    let out_bytes = read(&output);
    let info = assert::parse_ktx2(&out_bytes);
    assert_eq!(
        info.supercompression,
        Some(ktx2::SupercompressionScheme::ZLIB),
        "header must record ZLIB supercompression"
    );

    assert::assert_payload_eq(&read(&input), &out_bytes);
}
