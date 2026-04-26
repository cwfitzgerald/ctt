//! Color-space and alpha-mode tests.

use ctt::{AlphaMode, ColorSpace, Format};

use crate::common::synth::{make_image, write_ktx2};
use crate::common::{TestFixture, assert, read, run_cli};

/// Build a 4×4 RGBA8 KTX2 file where every pixel is `pixel` in the
/// requested color space + alpha mode.
fn write_solid_rgba8_ktx2(
    path: &std::path::Path,
    pixel: [u8; 4],
    color_space: ColorSpace,
    alpha: AlphaMode,
) {
    let pixels = pixel.repeat(4 * 4);
    let img = make_image(pixels, 4, 4, Format::R8G8B8A8_UNORM, color_space, alpha);
    write_ktx2(img, path);
}

/// Decode a KTX2 file and return the first surface's pixel data and color space.
fn first_surface(bytes: &[u8]) -> (Vec<u8>, ColorSpace, AlphaMode, Format) {
    let img = assert::decode(bytes);
    let s = &img.surfaces[0][0];
    (s.data.clone(), s.color_space, s.alpha, s.format)
}

/// sRGB(128) decodes to linear ≈ 0.2158 ≈ u8 55. Convert mid-grey sRGB
/// → linear and verify channels land in [54, 56] (LUT precision tolerance).
#[test]
fn srgb_to_linear_changes_mid_grey() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    write_solid_rgba8_ktx2(
        &input,
        [128, 128, 128, 255],
        ColorSpace::Srgb,
        AlphaMode::Opaque,
    );

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--output-color-space",
        "linear",
    ])
    .expect("run succeeded");

    let (data, cs, _, _) = first_surface(&read(&output));
    assert_eq!(
        cs,
        ColorSpace::Linear,
        "output color space tag must be linear"
    );
    for px in data.chunks_exact(4) {
        assert!(
            (54..=56).contains(&px[0]),
            "R: expected ~55 (sRGB EOTF of 128), got {}",
            px[0]
        );
        assert_eq!(px[1], px[0], "channels must match");
        assert_eq!(px[2], px[0], "channels must match");
        assert_eq!(px[3], 255, "alpha unchanged");
    }
}

/// Linear(128) encodes to sRGB ≈ 0.7349 ≈ u8 187. Convert mid-grey linear
/// → sRGB and verify channels land in [186, 188] (LUT precision tolerance).
#[test]
fn linear_to_srgb_changes_mid_grey() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    write_solid_rgba8_ktx2(
        &input,
        [128, 128, 128, 255],
        ColorSpace::Linear,
        AlphaMode::Opaque,
    );

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--output-color-space",
        "srgb",
    ])
    .expect("run succeeded");

    let (data, cs, _, _) = first_surface(&read(&output));
    assert_eq!(cs, ColorSpace::Srgb, "output color space tag must be sRGB");
    for px in data.chunks_exact(4) {
        assert!(
            (186..=188).contains(&px[0]),
            "R: expected ~187 (sRGB OETF of 128), got {}",
            px[0]
        );
        assert_eq!(px[1], px[0], "channels must match");
        assert_eq!(px[2], px[0], "channels must match");
        assert_eq!(px[3], 255, "alpha unchanged");
    }
}

/// `--output-alpha premultiplied` actually multiplies RGB by alpha.
/// (200, 100, 50, 128) Straight → (100, 50, 25, 128) Premultiplied via
/// `rgb * (a/255)` with .round() store rounding.
#[test]
fn output_alpha_premultiplied_multiplies_rgb_by_alpha() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    write_solid_rgba8_ktx2(
        &input,
        [200, 100, 50, 128],
        ColorSpace::Linear,
        AlphaMode::Straight,
    );

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--output-alpha",
        "premultiplied",
    ])
    .expect("run succeeded");

    let (data, _, alpha, _) = first_surface(&read(&output));
    assert_eq!(
        alpha,
        AlphaMode::Premultiplied,
        "alpha tag must be premultiplied"
    );
    for px in data.chunks_exact(4) {
        assert_eq!(px, &[100, 50, 25, 128], "premultiplied RGBA mismatch");
    }
}

/// `--output-alpha opaque` only re-tags the surface; with input alpha=255
/// the channels round-trip exactly through the float pipeline (no
/// premul/unpremul math performed because alpha=1.0).
#[test]
fn output_alpha_opaque_only_retags() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    let pixel = [200, 100, 50, 255];
    write_solid_rgba8_ktx2(&input, pixel, ColorSpace::Linear, AlphaMode::Straight);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--output-alpha",
        "opaque",
    ])
    .expect("run succeeded");

    let (data, _, alpha, _) = first_surface(&read(&output));
    // KTX2 collapses Opaque to "not premultiplied" on disk, which decodes
    // back as Straight. The DFD bit that distinguishes Opaque is not in
    // the container, so we can't assert tag == Opaque after a roundtrip.
    // Per the plan's intent, the data must be unchanged.
    assert!(
        matches!(alpha, AlphaMode::Straight | AlphaMode::Opaque),
        "alpha tag should be Straight or Opaque, got {alpha:?}"
    );
    for px in data.chunks_exact(4) {
        assert_eq!(px, &pixel, "RGBA bytes must be unchanged");
    }
}

/// `-f bc7 --output-color-space srgb` picks the BC7_SRGB_BLOCK on-disk
/// format (denormalized from BC7_UNORM_BLOCK + Srgb).
#[test]
fn srgb_output_picks_srgb_block_variant() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_srgb.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        "bc7",
        "--output-color-space",
        "srgb",
        "--quality",
        "ultra-fast",
    ])
    .expect("run succeeded");

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(
        info.format,
        Some(ktx2::Format::BC7_SRGB_BLOCK),
        "sRGB output must pick BC7_SRGB_BLOCK on disk"
    );
}
