//! Non-RGBA8 standard-image load paths.
//!
//! Exercises each `load_standard_image` branch by writing a PNG (or EXR for
//! the float types) of the corresponding color type, converting it to KTX2
//! with no `-f` (uncompressed passthrough) and `--ic linear` so the loaded
//! format flows through to the output header unchanged.

use ktx2::Format as F;

use crate::common::synth::write_image_bytes;
use crate::common::{TestFixture, assert, read, run_cli};

const W: u32 = 8;
const H: u32 = 8;

/// Write `bytes` as `color_type` to `name` in the fixture, convert to KTX2,
/// and assert the output header carries `expected` at W×H.
fn assert_loads_as(name: &str, color_type: image::ExtendedColorType, bytes: &[u8], expected: F) {
    let f = TestFixture::new();
    let input = f.output_file(name);
    write_image_bytes(&input, W, H, color_type, bytes);
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--ic",
        "linear",
    ])
    .unwrap_or_else(|e| panic!("{name}: run failed: {e}"));

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(info.width, W, "{name}: width");
    assert_eq!(info.height, H, "{name}: height");
    assert_eq!(info.format, Some(expected), "{name}: KTX2 vkFormat");
}

fn u16_bytes(samples: usize) -> Vec<u8> {
    let v: Vec<u16> = (0..samples).map(|i| (i as u16).wrapping_mul(257)).collect();
    bytemuck::cast_slice(&v).to_vec()
}

fn f32_bytes(samples: usize) -> Vec<u8> {
    let v: Vec<f32> = (0..samples)
        .map(|i| (i as f32) / (samples as f32))
        .collect();
    bytemuck::cast_slice(&v).to_vec()
}

#[test]
fn luma8_loads_as_r8_unorm() {
    let px = (W * H) as usize;
    assert_loads_as(
        "gray8.png",
        image::ExtendedColorType::L8,
        &(0..px).map(|i| i as u8).collect::<Vec<u8>>(),
        F::R8_UNORM,
    );
}

#[test]
fn luma_alpha8_loads_as_r8g8_unorm() {
    let n = (W * H * 2) as usize;
    assert_loads_as(
        "gray_alpha8.png",
        image::ExtendedColorType::La8,
        &(0..n).map(|i| i as u8).collect::<Vec<u8>>(),
        F::R8G8_UNORM,
    );
}

#[test]
fn rgb8_loads_as_r8g8b8_unorm() {
    let n = (W * H * 3) as usize;
    assert_loads_as(
        "rgb8.png",
        image::ExtendedColorType::Rgb8,
        &(0..n).map(|i| i as u8).collect::<Vec<u8>>(),
        F::R8G8B8_UNORM,
    );
}

#[test]
fn luma16_loads_as_r16_unorm() {
    assert_loads_as(
        "gray16.png",
        image::ExtendedColorType::L16,
        &u16_bytes((W * H) as usize),
        F::R16_UNORM,
    );
}

#[test]
fn luma_alpha16_loads_as_r16g16_unorm() {
    assert_loads_as(
        "gray_alpha16.png",
        image::ExtendedColorType::La16,
        &u16_bytes((W * H * 2) as usize),
        F::R16G16_UNORM,
    );
}

#[test]
fn rgb16_loads_as_r16g16b16_unorm() {
    assert_loads_as(
        "rgb16.png",
        image::ExtendedColorType::Rgb16,
        &u16_bytes((W * H * 3) as usize),
        F::R16G16B16_UNORM,
    );
}

#[test]
fn rgba16_loads_as_r16g16b16a16_unorm() {
    assert_loads_as(
        "rgba16.png",
        image::ExtendedColorType::Rgba16,
        &u16_bytes((W * H * 4) as usize),
        F::R16G16B16A16_UNORM,
    );
}

/// EXR input without `--ic` must default to linear: HDR values above 1.0
/// survive a passthrough conversion untouched (the sRGB EOTF would clamp
/// and bend them).
#[test]
fn exr_defaults_to_linear_and_preserves_hdr_values() {
    let f = TestFixture::new();
    let input = f.output_file("hdr.exr");
    let values: Vec<f32> = (0..(W * H * 4) as usize)
        .map(|i| if i % 4 == 3 { 1.0 } else { i as f32 * 0.5 })
        .collect();
    write_image_bytes(
        &input,
        W,
        H,
        image::ExtendedColorType::Rgba32F,
        bytemuck::cast_slice(&values),
    );
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let decoded = assert::decode(&read(&output));
    let s = &decoded.surfaces[0][0];
    assert_eq!(s.format, F::R32G32B32A32_SFLOAT);
    assert_eq!(s.color_space, ctt::ColorSpace::Linear);
    let got: &[f32] = bytemuck::cast_slice(&s.data);
    assert_eq!(got, values.as_slice(), "HDR values must pass through");
}

#[test]
fn rgb32f_loads_as_r32g32b32_sfloat() {
    assert_loads_as(
        "rgb32f.exr",
        image::ExtendedColorType::Rgb32F,
        &f32_bytes((W * H * 3) as usize),
        F::R32G32B32_SFLOAT,
    );
}

#[test]
fn rgba32f_loads_as_r32g32b32a32_sfloat() {
    assert_loads_as(
        "rgba32f.exr",
        image::ExtendedColorType::Rgba32F,
        &f32_bytes((W * H * 4) as usize),
        F::R32G32B32A32_SFLOAT,
    );
}
