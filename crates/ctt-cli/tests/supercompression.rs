//! KTX2 supercompression tests.
//!
//! Each test runs the CLI with a supercompression flag, checks that the
//! KTX2 header records the scheme, and round-trips the output through
//! `decode_container` to confirm the level data decompresses cleanly back
//! to the source bytes.

use crate::common::synth::write_solid_rgba8_png;
use crate::common::{TestFixture, assert, read, run_cli, try_parse_args};

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

/// `--zstd=5` sets the level via the required `=` syntax and produces a valid
/// Zstandard-supercompressed KTX2.
#[test]
fn zstd_equals_level_works() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--zstd=5",
    ])
    .expect("--zstd=5 run succeeded");

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(
        info.supercompression,
        Some(ktx2::SupercompressionScheme::Zstandard)
    );
}

/// With `require_equals`, a bare `--zstd` no longer swallows the following
/// positional: `ctt --zstd in.png -o out.ktx2` treats `in.png` as the input
/// (and applies the default zstd level).
#[test]
fn bare_zstd_does_not_swallow_input() {
    let f = TestFixture::new();
    let input = f.output_file("in.png");
    write_solid_rgba8_png(&input, 8, 8, [10, 20, 30, 255]);
    let output = f.output_file("out.ktx2");

    // Note the arg order: `--zstd` immediately precedes the positional input.
    let args = try_parse_args([
        "ctt",
        "--zstd",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("parse succeeded");
    assert_eq!(args.input.len(), 1, "in.png must be parsed as the input");
    assert_eq!(args.zstd, Some(0), "bare --zstd uses the default level");

    run_cli([
        "ctt",
        "--zstd",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(
        info.supercompression,
        Some(ktx2::SupercompressionScheme::Zstandard)
    );
}

/// Out-of-range compression levels are rejected at parse time with a clear
/// clap error.
#[test]
fn out_of_range_levels_rejected() {
    // zstd max is 22.
    assert!(
        try_parse_args(["ctt", "in.ktx2", "-o", "o.ktx2", "--zstd=999"]).is_err(),
        "zstd level above the library max must be rejected"
    );
    // zlib documented range is 1..=10.
    assert!(
        try_parse_args(["ctt", "in.ktx2", "-o", "o.ktx2", "--zlib=0"]).is_err(),
        "zlib level 0 (below the documented minimum) must be rejected"
    );
    assert!(
        try_parse_args(["ctt", "in.ktx2", "-o", "o.ktx2", "--zlib=11"]).is_err(),
        "zlib level above 10 must be rejected"
    );
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
