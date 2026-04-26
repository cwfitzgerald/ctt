//! Edge-case dimension tests.
//!
//! Boundary inputs (single pixel, single-row, sub-block-size) exercise the
//! pipeline's padding and degenerate-dimension handling.

use ctt::{AlphaMode, ColorSpace, Format};

use crate::common::synth::{synth_uncompressed, write_ktx2};
use crate::common::{TestFixture, assert, read, run_cli};

fn write_rgba8_input(path: &std::path::Path, w: u32, h: u32) {
    write_ktx2(
        synth_uncompressed(
            Format::R8G8B8A8_UNORM,
            w,
            h,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        path,
    );
}

/// 1×1 RGBA8 passthrough → byte-identical KTX2.
#[test]
fn one_by_one_passthrough_byte_equal() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");
    write_rgba8_input(&input, 1, 1);

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
        "1×1 passthrough must be byte-identical"
    );
}

/// 1×1 RGBA8 + `-f bc7`: encoder pads to a single block, output Surface keeps
/// the original 1×1 logical dimensions.
#[test]
fn one_by_one_to_bc7_produces_single_block() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");
    write_rgba8_input(&input, 1, 1);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        "bc7",
        "--quality",
        "ultra-fast",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.width, 1);
    assert_eq!(info.height, 1);
    assert_eq!(info.format, Some(ktx2::Format::BC7_UNORM_BLOCK));

    let decoded = assert::decode(&bytes);
    let surface = &decoded.surfaces[0][0];
    assert_eq!(surface.format, Format::BC7_UNORM_BLOCK);
    assert_eq!(surface.width, 1);
    assert_eq!(surface.height, 1);
    assert_eq!(surface.data.len(), 16, "one BC7 block = 16 bytes");
}

/// 1×1 RGBA8 + `--mipmap`: a 1-pixel base produces a 1-level chain.
#[test]
fn one_by_one_mipmap_chain_is_one_level() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");
    write_rgba8_input(&input, 1, 1);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--mipmap",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.level_count, 1, "1×1 has only one mip level");
    assert_eq!(info.width, 1);
    assert_eq!(info.height, 1);
}

/// 1024×1 single-row passthrough: degenerate aspect ratio survives container
/// roundtrip with bytes intact.
#[test]
fn long_thin_1024x1_passthrough_byte_equal() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");
    write_rgba8_input(&input, 1024, 1);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    assert_eq!(read(&input), read(&output));
    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(info.width, 1024);
    assert_eq!(info.height, 1);
}

/// Sub-block-size source (3×3 RGBA8) → BC7: encoder pads to fill one full
/// 4×4 block. The output retains the 3×3 logical dimensions but holds 16
/// compressed bytes (one block).
#[test]
fn sub_block_size_3x3_to_bc7_padded_to_one_block() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");
    write_rgba8_input(&input, 3, 3);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        "bc7",
        "--quality",
        "ultra-fast",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.width, 3);
    assert_eq!(info.height, 3);
    assert_eq!(info.format, Some(ktx2::Format::BC7_UNORM_BLOCK));

    let decoded = assert::decode(&bytes);
    let surface = &decoded.surfaces[0][0];
    assert_eq!(surface.width, 3);
    assert_eq!(surface.height, 3);
    assert_eq!(
        surface.data.len(),
        16,
        "3×3 padded to one 4×4 BC7 block = 16 bytes"
    );
}
