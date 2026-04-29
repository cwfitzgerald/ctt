//! 3D (volume) texture construction and passthrough tests.
//!
//! 3D textures are represented as `kind = Texture3D` with a single Surface
//! whose `depth > 1` and whose `data` packs `depth` Z slices contiguously.
//! KTX2 stores them via `pixel_depth`; DDS via the `Texture3D` resource
//! dimension.

use ctt::{Format, TextureKind};

use crate::common::synth::{make_volume_rgba8, write_ktx2, write_solid_rgba8_png};
use crate::common::{TestFixture, assert, read, run_cli, try_parse_args};

/// 4-slice RGBA8 KTX2 volume → KTX2 byte-identical passthrough. Verifies
/// the encoder writes `pixel_depth` correctly and the decoder reads it back.
#[test]
fn rgba8_volume_ktx2_passthrough_byte_equal() {
    let f = TestFixture::new();
    let slice_colors = vec![
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    let input = f.output_file("volume_in.ktx2");
    let output = f.output_file("volume_out.ktx2");
    write_ktx2(make_volume_rgba8(8, 8, slice_colors), &input);

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
        "3D KTX2→KTX2 passthrough must be byte-identical"
    );

    let info = assert::parse_ktx2(&read(&output));
    assert_eq!(info.layer_count, 0, "3D textures use depth, not layers");
    assert_eq!(info.face_count, 1, "3D is not a cubemap");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
}

/// 4-slice volume KTX2 → DDS: payload survives across the container
/// conversion. DDS records `D3D10ResourceDimension::Texture3D` with
/// `depth = 4`.
#[test]
fn rgba8_volume_ktx2_to_dds_preserves_depth() {
    let f = TestFixture::new();
    let slice_colors = vec![
        [255u8, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    let input = f.output_file("volume.ktx2");
    let output = f.output_file("volume.dds");
    let original = make_volume_rgba8(8, 8, slice_colors.clone());
    write_ktx2(original.clone(), &input);

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let dds_bytes = read(&output);
    let decoded = assert::decode(&dds_bytes);
    assert_eq!(decoded.kind, TextureKind::Texture3D);
    assert_eq!(decoded.surfaces.len(), 1);
    let s = &decoded.surfaces[0][0];
    assert_eq!(s.width, 8);
    assert_eq!(s.height, 8);
    assert_eq!(s.depth, 4);
    assert_eq!(s.data, original.surfaces[0][0].data);
}

/// 4 RGBA8 PNGs (red/green/blue/yellow) + `--volume` → 4-slice 3D KTX2.
/// Slices are stacked in argv order; per-slice payload matches the source
/// PNG.
#[test]
fn four_pngs_assemble_into_volume() {
    let f = TestFixture::new();
    let slice_colors: [[u8; 4]; 4] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    let inputs: Vec<_> = slice_colors
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let path = f.output_file(&format!("slice_{i}.png"));
            write_solid_rgba8_png(&path, 4, 4, *color);
            path
        })
        .collect();
    let output = f.output_file("volume.ktx2");

    let mut argv: Vec<String> = vec!["ctt".to_string()];
    for p in &inputs {
        argv.push(p.to_str().unwrap().to_string());
    }
    argv.extend([
        "-o".to_string(),
        output.to_str().unwrap().to_string(),
        "--volume".to_string(),
    ]);

    run_cli(argv).expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.layer_count, 0);
    assert_eq!(info.face_count, 1);
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.kind, TextureKind::Texture3D);
    let s = &decoded.surfaces[0][0];
    assert_eq!(s.depth, 4);

    // Each Z slice should be a solid color block of 4×4 pixels.
    let bytes_per_slice = (s.stride * s.height) as usize;
    for (i, color) in slice_colors.iter().enumerate() {
        let start = i * bytes_per_slice;
        let slice_data = &s.data[start..start + bytes_per_slice];
        for px in slice_data.chunks_exact(4) {
            assert_eq!(px, color, "slice {i}: expected {color:?}, got {px:?}");
        }
    }
}

/// 4 PNG slices + `--volume -o vol.dds` → 4-slice DDS volume. The DDS
/// container uses Texture3D resource dim with `depth = 4`.
#[test]
fn four_pngs_assemble_into_volume_dds() {
    let f = TestFixture::new();
    let slice_colors: [[u8; 4]; 4] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    let inputs: Vec<_> = slice_colors
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let path = f.output_file(&format!("slice_{i}.png"));
            write_solid_rgba8_png(&path, 4, 4, *color);
            path
        })
        .collect();
    let output = f.output_file("volume.dds");

    let mut argv: Vec<String> = vec!["ctt".to_string()];
    for p in &inputs {
        argv.push(p.to_str().unwrap().to_string());
    }
    argv.extend([
        "-o".to_string(),
        output.to_str().unwrap().to_string(),
        "--volume".to_string(),
    ]);

    run_cli(argv).expect("run succeeded");

    let dds_bytes = read(&output);
    let decoded = assert::decode(&dds_bytes);
    assert_eq!(decoded.kind, TextureKind::Texture3D);
    let s = &decoded.surfaces[0][0];
    assert_eq!(s.depth, 4);
    assert_eq!(s.width, 4);
    assert_eq!(s.height, 4);
}

/// `--volume --mipmap` is rejected at argument parse time. 3D mip
/// generation is intentionally out of scope.
#[test]
fn volume_with_mipmap_rejected() {
    let err = try_parse_args(["ctt", "in.png", "-o", "out.ktx2", "--volume", "--mipmap"])
        .expect_err("--volume + --mipmap must conflict");
    let msg = err.to_string();
    assert!(
        msg.contains("--volume") || msg.contains("--mipmap"),
        "expected conflict mention, got: {msg}"
    );
}

/// `--volume --cubemap` is rejected at argument parse time.
#[test]
fn volume_with_cubemap_rejected() {
    let err = try_parse_args(["ctt", "a.png", "-o", "out.ktx2", "--volume", "--cubemap"])
        .expect_err("--volume + --cubemap must conflict");
    let msg = err.to_string();
    assert!(
        msg.contains("--volume") || msg.contains("--cubemap"),
        "expected conflict mention, got: {msg}"
    );
}

/// Mismatched slice dimensions error out clearly.
#[test]
fn volume_slice_size_mismatch_rejected() {
    let f = TestFixture::new();
    let a = f.output_file("a.png");
    let b = f.output_file("b.png");
    write_solid_rgba8_png(&a, 4, 4, [255, 0, 0, 255]);
    write_solid_rgba8_png(&b, 8, 8, [0, 255, 0, 255]);
    let output = f.output_file("vol.ktx2");

    let result = run_cli([
        "ctt",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--volume",
    ]);
    let err = result.expect_err("mismatched slice sizes must error");
    assert!(
        err.to_string().contains("dimensions"),
        "expected dimensions-mismatch error, got: {err}"
    );
}

/// Compressed BC7 KTX2 with depth>1 round-trips through CLI passthrough
/// byte-equal.
#[test]
fn bc7_volume_ktx2_passthrough_byte_equal() {
    use ctt::{AlphaMode, ColorSpace, Image, Surface};

    let f = TestFixture::new();
    // Single-mip BC7 volume of 8×8×4. BC7 = 16 bytes per 4×4 block; 8×8 = 4
    // blocks per slice → 64 bytes per slice → 256 bytes total for depth=4.
    let stride = 2 * 16; // 2 blocks across × 16 bytes/block
    let blocks_per_slice = 2 * 2; // 2x2 blocks
    let slice_stride = blocks_per_slice * 16;
    let depth = 4u32;
    let total = slice_stride * depth as usize;
    let data: Vec<u8> = (0..total).map(|i| (i & 0xff) as u8).collect();

    let original = Image {
        surfaces: vec![vec![Surface {
            data,
            width: 8,
            height: 8,
            depth,
            stride: stride as u32,
            slice_stride: slice_stride as u32,
            format: Format::BC7_UNORM_BLOCK,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }]],
        kind: TextureKind::Texture3D,
    };

    let input = f.output_file("bc7_volume_in.ktx2");
    let output = f.output_file("bc7_volume_out.ktx2");
    write_ktx2(original, &input);

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
        "BC7 3D KTX2→KTX2 passthrough must be byte-identical"
    );
}
