//! CLI encode smoke matrix.
//!
//! Encodes a synthetic RGBA8 PNG to every compressed format the default build
//! supports (ctt-cli always compiles in all five encoder backends), and checks
//! the resulting KTX2 parses with the expected `vkFormat`. Also covers the
//! uncompressed `-f` name spellings.
//!
//! Inputs are loaded with `--ic linear` so the output stays on the UNORM/base
//! format instead of being denormalized to an sRGB variant.

use ktx2::Format as F;

use crate::common::synth::rgba8_gradient;
use crate::common::synth::write_image_bytes;
use crate::common::{TestFixture, assert, read, run_cli};

const W: u32 = 16;
const H: u32 = 16;

/// Write a 16×16 RGBA8 gradient PNG to the fixture and return its path.
fn gradient_png(f: &TestFixture) -> std::path::PathBuf {
    let path = f.output_file("input.png");
    write_image_bytes(
        &path,
        W,
        H,
        image::ExtendedColorType::Rgba8,
        &rgba8_gradient(W, H),
    );
    path
}

/// Encode `input` to KTX2 with target `-f fmt` (ultra-fast) and assert the
/// output header reports `expected`.
fn assert_encode(fmt: &str, expected: F) {
    let f = TestFixture::new();
    let input = gradient_png(&f);
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        fmt,
        "--ic",
        "linear",
        "--quality",
        "ultra-fast",
    ])
    .unwrap_or_else(|e| panic!("{fmt}: run failed: {e}"));

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.width, W, "{fmt}: width");
    assert_eq!(info.height, H, "{fmt}: height");
    assert_eq!(info.format, Some(expected), "{fmt}: KTX2 vkFormat");
}

#[test]
fn encode_bc1() {
    assert_encode("bc1", F::BC1_RGBA_UNORM_BLOCK);
}

#[test]
fn encode_bc2() {
    assert_encode("bc2", F::BC2_UNORM_BLOCK);
}

#[test]
fn encode_bc3() {
    assert_encode("bc3", F::BC3_UNORM_BLOCK);
}

#[test]
fn encode_bc4() {
    assert_encode("bc4", F::BC4_UNORM_BLOCK);
}

#[test]
fn encode_bc4s() {
    assert_encode("bc4s", F::BC4_SNORM_BLOCK);
}

#[test]
fn encode_bc5() {
    assert_encode("bc5", F::BC5_UNORM_BLOCK);
}

#[test]
fn encode_bc5s() {
    assert_encode("bc5s", F::BC5_SNORM_BLOCK);
}

#[test]
fn encode_bc6h() {
    assert_encode("bc6h", F::BC6H_UFLOAT_BLOCK);
}

#[test]
fn encode_bc6hsf() {
    assert_encode("bc6hsf", F::BC6H_SFLOAT_BLOCK);
}

#[test]
fn encode_bc7() {
    assert_encode("bc7", F::BC7_UNORM_BLOCK);
}

#[test]
fn encode_etc1() {
    assert_encode("etc1", F::ETC2_R8G8B8_UNORM_BLOCK);
}

#[test]
fn encode_etc2_rgba() {
    assert_encode("etc2_rgba", F::ETC2_R8G8B8A8_UNORM_BLOCK);
}

#[test]
fn encode_eac_r() {
    assert_encode("eac_r", F::EAC_R11_UNORM_BLOCK);
}

#[test]
fn encode_eac_rg() {
    assert_encode("eac_rg", F::EAC_R11G11_UNORM_BLOCK);
}

#[test]
fn encode_astc_4x4() {
    assert_encode("astc_4x4", F::ASTC_4x4_UNORM_BLOCK);
}

// --- Uncompressed `-f` name spellings (test f) ---

#[test]
fn uncompressed_webgpu_name() {
    assert_encode("rgba8unorm", F::R8G8B8A8_UNORM);
}

#[test]
fn uncompressed_vulkan_name() {
    assert_encode("r8g8b8a8_unorm", F::R8G8B8A8_UNORM);
}
