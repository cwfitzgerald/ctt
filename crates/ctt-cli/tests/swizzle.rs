//! Channel swizzle tests.
//!
//! Each test feeds a known RGBA pixel through `--swizzle <pattern>` and
//! asserts the per-channel mapping. Inputs are tagged [`AlphaMode::Opaque`]
//! so the float pipeline skips premultiplication/unpremultiplication and
//! the swizzle is a pure channel rearrange.

use ctt::{AlphaMode, ColorSpace, Format};

use crate::common::synth::{make_image, write_ktx2};
use crate::common::{TestFixture, assert, read, run_cli};

const SRC: [u8; 4] = [10, 20, 30, 40];

/// Build a 4×4 RGBA8 KTX2 input where every pixel is [`SRC`], run the CLI
/// with `--swizzle <pattern>`, and assert every pixel of the decoded
/// output matches `expected`.
fn assert_swizzle(pattern: &str, expected: [u8; 4]) {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    let pixels = SRC.repeat(4 * 4);
    let img = make_image(
        pixels,
        4,
        4,
        Format::R8G8B8A8_UNORM,
        ColorSpace::Linear,
        AlphaMode::Opaque,
    );
    write_ktx2(img, &input);

    // KTX2 doesn't distinguish Opaque from Straight on disk; without this
    // override the float pipeline would premultiply on load and unpremul on
    // store, mangling channels whenever the swizzle changes alpha.
    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--swizzle",
        pattern,
        "--input-alpha",
        "opaque",
        "--output-alpha",
        "opaque",
    ])
    .expect("run succeeded");

    let decoded = assert::decode(&read(&output));
    let surface = &decoded.surfaces[0][0];
    assert_eq!(surface.format, Format::R8G8B8A8_UNORM);
    for (i, px) in surface.data.chunks_exact(4).enumerate() {
        assert_eq!(
            px, &expected,
            "pixel {i}: swizzle '{pattern}' on {SRC:?} → expected {expected:?}, got {px:?}"
        );
    }
}

#[test]
fn swizzle_bgra_swaps_red_and_blue() {
    assert_swizzle("bgra", [SRC[2], SRC[1], SRC[0], SRC[3]]);
}

#[test]
fn swizzle_rgb1_forces_opaque_alpha() {
    assert_swizzle("rgb1", [SRC[0], SRC[1], SRC[2], 255]);
}

#[test]
fn swizzle_r000_isolates_red_channel() {
    assert_swizzle("r000", [SRC[0], 0, 0, 0]);
}

#[test]
fn swizzle_0r0g_packs_normal_channels() {
    assert_swizzle("0r0g", [0, SRC[0], 0, SRC[1]]);
}
