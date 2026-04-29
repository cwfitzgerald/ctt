//! Mipmap generation tests.

use ctt::{AlphaMode, ColorSpace, Format, TextureKind};

use crate::common::synth::{
    CUBEMAP_FACE_COLORS, make_cubemap_rgba8, rgba8_solid, synth_uncompressed, write_ktx2,
};
use crate::common::{TestFixture, assert, read, run_cli};

/// 16×16 RGBA8 + `--mipmap` → KTX2 with a 5-level chain (16, 8, 4, 2, 1)
/// and the per-level dimensions match.
#[test]
fn full_mip_chain_16x16() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

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
    assert_eq!(info.width, 16);
    assert_eq!(info.height, 16);
    assert_eq!(info.level_count, 5, "16×16 produces a 5-level chain");

    let decoded = assert::decode(&bytes);
    let expected_dims = [(16u32, 16u32), (8, 8), (4, 4), (2, 2), (1, 1)];
    for (i, (w, h)) in expected_dims.iter().enumerate() {
        let s = &decoded.surfaces[0][i];
        assert_eq!(s.width, *w, "mip {i} width");
        assert_eq!(s.height, *h, "mip {i} height");
    }
}

/// `--mipmap-count 3` truncates the chain at 3 levels even though a full
/// chain from 16×16 would have 5.
#[test]
fn truncated_mip_chain_count_3() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--mipmap",
        "--mipmap-count",
        "3",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.level_count, 3, "truncated chain must have 3 levels");

    let decoded = assert::decode(&bytes);
    let expected_dims = [(16u32, 16u32), (8, 8), (4, 4)];
    for (i, (w, h)) in expected_dims.iter().enumerate() {
        let s = &decoded.surfaces[0][i];
        assert_eq!(s.width, *w, "mip {i} width");
        assert_eq!(s.height, *h, "mip {i} height");
    }
}

/// NPOT 5×3 input + `--mipmap` → mip dimensions round down: 5×3, 2×1, 1×1.
/// `full_mip_count(5, 3) = floor(log2(5)) + 1 = 3`.
#[test]
fn npot_mipmap_5x3_rounds_down() {
    let f = TestFixture::new();
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    write_ktx2(
        synth_uncompressed(
            Format::R8G8B8A8_UNORM,
            5,
            3,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );

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
    assert_eq!(info.width, 5);
    assert_eq!(info.height, 3);
    assert_eq!(info.level_count, 3, "5×3 produces a 3-level chain");

    let decoded = assert::decode(&bytes);
    let expected_dims = [(5u32, 3u32), (2, 1), (1, 1)];
    for (i, (w, h)) in expected_dims.iter().enumerate() {
        let s = &decoded.surfaces[0][i];
        assert_eq!(s.width, *w, "mip {i} width");
        assert_eq!(s.height, *h, "mip {i} height");
    }
}

/// `--mipmap` combined with `-f bc7`: every mip level is BC7-compressed.
#[test]
fn mipmap_plus_bc7_compresses_every_level() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--mipmap",
        "-f",
        "bc7",
        "--quality",
        "ultra-fast",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.level_count, 5, "16×16 produces a 5-level chain");
    assert_eq!(info.format, Some(ktx2::Format::BC7_UNORM_BLOCK));

    let decoded = assert::decode(&bytes);
    let expected_dims = [(16u32, 16u32), (8, 8), (4, 4), (2, 2), (1, 1)];
    for (i, (w, h)) in expected_dims.iter().enumerate() {
        let s = &decoded.surfaces[0][i];
        assert_eq!(s.width, *w, "mip {i} width");
        assert_eq!(s.height, *h, "mip {i} height");
        assert_eq!(s.format, Format::BC7_UNORM_BLOCK, "mip {i} format");
    }
}

/// `--mipmap` combined with `--cubemap`: each face has its own mip chain
/// with per-level dimensions matching the standard halving rule.
#[test]
fn mipmap_plus_cubemap_per_face_mip_chain() {
    let f = TestFixture::new();
    let face_size = 8u32;
    let faces: [Vec<u8>; 6] =
        std::array::from_fn(|i| rgba8_solid(face_size, face_size, CUBEMAP_FACE_COLORS[i]));
    let input = f.output_file("input.ktx2");
    let output = f.output_file("output.ktx2");

    write_ktx2(
        make_cubemap_rgba8(
            face_size,
            face_size,
            faces,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        ),
        &input,
    );

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--cubemap",
        "--mipmap",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.face_count, 6, "must be a cubemap");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert_eq!(info.level_count, 4, "8×8 produces a 4-level chain");

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.kind, TextureKind::Cubemap);
    assert_eq!(decoded.surfaces.len(), 6);
    let expected_dims = [(8u32, 8u32), (4, 4), (2, 2), (1, 1)];
    for face_idx in 0..6 {
        for (i, (w, h)) in expected_dims.iter().enumerate() {
            let s = &decoded.surfaces[face_idx][i];
            assert_eq!(s.width, *w, "face {face_idx} mip {i} width");
            assert_eq!(s.height, *h, "face {face_idx} mip {i} height");
        }
    }
}
