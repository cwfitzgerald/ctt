//! Synthesis helpers: build `Image`s and serialize them to KTX2/DDS bytes.

use std::path::Path;

use ctt::{
    AlphaMode, ColorSpace, Container, ConvertSettings, Format, FormatExt, Image, PipelineOutput,
    Surface, TextureKind,
};

/// Solid color RGBA8 image data.
pub fn rgba8_solid(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..w * h {
        buf.extend_from_slice(&color);
    }
    buf
}

/// Distinguishable per-pixel RGBA8 data: r=x%256, g=y%256, b=(x^y)%256, a=255.
pub fn rgba8_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            buf.extend_from_slice(&[
                (x & 0xff) as u8,
                (y & 0xff) as u8,
                ((x ^ y) & 0xff) as u8,
                255,
            ]);
        }
    }
    buf
}

/// Six visually distinct colors in +X, -X, +Y, -Y, +Z, -Z order.
pub const CUBEMAP_FACE_COLORS: [[u8; 4]; 6] = [
    [255, 0, 0, 255],   // +X — red
    [0, 255, 255, 255], // -X — cyan
    [0, 255, 0, 255],   // +Y — green
    [255, 0, 255, 255], // -Y — magenta
    [0, 0, 255, 255],   // +Z — blue
    [255, 255, 0, 255], // -Z — yellow
];

/// Build a single-layer single-mip [`Image`] from raw uncompressed pixel data.
pub fn make_image(
    data: Vec<u8>,
    width: u32,
    height: u32,
    format: Format,
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Image {
    let bpp = format.bytes_per_pixel().expect("uncompressed format") as u32;
    Image {
        surfaces: vec![vec![Surface {
            data,
            width,
            height,
            depth: 1,
            stride: width * bpp,
            slice_stride: 0,
            format,
            color_space,
            alpha,
        }]],
        kind: TextureKind::Texture2D,
    }
}

/// Build a cubemap [`Image`] from six per-face RGBA8 surfaces.
pub fn make_cubemap_rgba8(
    face_w: u32,
    face_h: u32,
    faces: [Vec<u8>; 6],
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Image {
    let surfaces: Vec<Vec<Surface>> = faces
        .into_iter()
        .map(|data| {
            vec![Surface {
                data,
                width: face_w,
                height: face_h,
                depth: 1,
                stride: face_w * 4,
                slice_stride: 0,
                format: Format::R8G8B8A8_UNORM,
                color_space,
                alpha,
            }]
        })
        .collect();
    Image {
        surfaces,
        kind: TextureKind::Cubemap,
    }
}

/// Build a single image holding raw block bytes for a compressed format.
///
/// `data` must be a single mip level's worth of compressed bytes.
pub fn make_compressed_image(
    data: Vec<u8>,
    width: u32,
    height: u32,
    format: Format,
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Image {
    let (block_w, block_h) = format.block_size().expect("compressed format");
    let bytes_per_block = format.bytes_per_block().expect("compressed format");
    let blocks_x = width.div_ceil(block_w as u32);
    let stride = blocks_x * bytes_per_block as u32;
    assert_eq!(
        data.len(),
        (blocks_x * height.div_ceil(block_h as u32)) as usize * bytes_per_block,
        "compressed payload size mismatch"
    );
    Image {
        surfaces: vec![vec![Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format,
            color_space,
            alpha,
        }]],
        kind: TextureKind::Texture2D,
    }
}

/// Deterministic per-byte pattern: each byte is its index modulo 256.
fn pattern_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i & 0xff) as u8).collect()
}

/// Synthesize a compressed [`Image`] of `width`×`height` filled with a
/// deterministic byte pattern. The contents are not "valid" compressed
/// blocks for any particular decoder, but they round-trip exactly through
/// container encode/decode, which is what passthrough tests exercise.
pub fn synth_compressed(format: Format, width: u32, height: u32) -> Image {
    let (block_w, block_h) = format.block_size().expect("compressed format");
    let bytes_per_block = format.bytes_per_block().expect("compressed format");
    let blocks_x = width.div_ceil(block_w as u32);
    let blocks_y = height.div_ceil(block_h as u32);
    let total = (blocks_x * blocks_y) as usize * bytes_per_block;
    make_compressed_image(
        pattern_bytes(total),
        width,
        height,
        format,
        ColorSpace::Linear,
        AlphaMode::Opaque,
    )
}

/// Synthesize an uncompressed [`Image`] of `width`×`height` filled with a
/// deterministic byte pattern in the requested format.
pub fn synth_uncompressed(
    format: Format,
    width: u32,
    height: u32,
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Image {
    let bpp = format.bytes_per_pixel().expect("uncompressed format");
    let total = (width * height) as usize * bpp;
    make_image(
        pattern_bytes(total),
        width,
        height,
        format,
        color_space,
        alpha,
    )
}

/// Encode an `Image` as KTX2 bytes.
pub fn to_ktx2(image: Image) -> Vec<u8> {
    encode(image, Container::Ktx2(None))
}

/// Encode an `Image` as DDS bytes.
pub fn to_dds(image: Image) -> Vec<u8> {
    encode(image, Container::Dds)
}

/// Encode an `Image` into a container, returning the encoded bytes.
fn encode(image: Image, container: Container) -> Vec<u8> {
    match ctt::convert(
        image,
        ConvertSettings {
            format: None,
            container,
            ..Default::default()
        },
    )
    .expect("ctt::convert succeeded")
    {
        PipelineOutput::Encoded(bytes) => bytes,
        PipelineOutput::Raw(_) => panic!("expected encoded output"),
    }
}

/// Encode an `Image` with a generated `mip_count`-deep mip chain into KTX2.
pub fn to_ktx2_with_mips(image: Image, mip_count: usize) -> Vec<u8> {
    match ctt::convert(
        image,
        ConvertSettings {
            format: None,
            container: Container::Ktx2(None),
            mipmap: true,
            mipmap_count: Some(mip_count),
            ..Default::default()
        },
    )
    .expect("ctt::convert succeeded")
    {
        PipelineOutput::Encoded(bytes) => bytes,
        PipelineOutput::Raw(_) => panic!("expected encoded output"),
    }
}

/// Write a KTX2-encoded image to `path`.
pub fn write_ktx2(image: Image, path: &Path) {
    std::fs::write(path, to_ktx2(image)).expect("write ktx2");
}

/// Write a DDS-encoded image to `path`.
pub fn write_dds(image: Image, path: &Path) {
    std::fs::write(path, to_dds(image)).expect("write dds");
}

/// Write a single-layer RGBA8 PNG of `width`×`height` filled with `color`.
pub fn write_solid_rgba8_png(path: &Path, width: u32, height: u32, color: [u8; 4]) {
    let pixels: Vec<u8> = color.repeat((width * height) as usize);
    image::save_buffer(
        path,
        &pixels,
        width,
        height,
        image::ExtendedColorType::Rgba8,
    )
    .expect("write png");
}

/// Build a multi-layer Image from N tightly-packed RGBA8 layers (each
/// `width`×`height`). All layers share color space and alpha mode.
pub fn make_array_image(
    layers: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Image {
    let surfaces: Vec<Vec<Surface>> = layers
        .into_iter()
        .map(|data| {
            vec![Surface {
                data,
                width,
                height,
                depth: 1,
                stride: width * 4,
                slice_stride: 0,
                format: Format::R8G8B8A8_UNORM,
                color_space,
                alpha,
            }]
        })
        .collect();
    Image {
        surfaces,
        kind: ctt::TextureKind::Texture2D,
    }
}

/// Build a cubemap-array Image with `cubes` cubes, each containing 6 distinct
/// per-face colors (palette palette is the standard +X/-X/+Y/-Y/+Z/-Z palette,
/// rotated by `cube_idx` so each cube is visually distinct).
pub fn make_cubemap_array_rgba8(cubes: usize, face: u32) -> Image {
    let mut surfaces: Vec<Vec<Surface>> = Vec::with_capacity(cubes * 6);
    for cube_idx in 0..cubes {
        for face_idx in 0..6 {
            let palette = CUBEMAP_FACE_COLORS[(face_idx + cube_idx) % 6];
            let data = palette.repeat((face * face) as usize);
            surfaces.push(vec![Surface {
                data,
                width: face,
                height: face,
                depth: 1,
                stride: face * 4,
                slice_stride: 0,
                format: Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Opaque,
            }]);
        }
    }
    Image {
        surfaces,
        kind: ctt::TextureKind::Cubemap,
    }
}

/// Build a 3D (volume) Image with `depth` Z-slices of width×height RGBA8.
/// Each slice is a solid color from `slice_colors[z]`.
pub fn make_volume_rgba8(width: u32, height: u32, slice_colors: Vec<[u8; 4]>) -> Image {
    let depth = slice_colors.len() as u32;
    let stride = width * 4;
    let slice_stride = stride * height;
    let mut data = Vec::with_capacity((slice_stride * depth) as usize);
    for color in &slice_colors {
        data.extend(color.repeat((width * height) as usize));
    }
    Image {
        surfaces: vec![vec![Surface {
            data,
            width,
            height,
            depth,
            stride,
            slice_stride,
            format: Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }]],
        kind: ctt::TextureKind::Texture3D,
    }
}

/// Build a multi-layer compressed Image where each layer carries a
/// deterministic byte pattern offset by its layer index. All layers share
/// dimensions and format.
pub fn synth_compressed_array(format: Format, width: u32, height: u32, layers: usize) -> Image {
    let (block_w, block_h) = format.block_size().expect("compressed format");
    let bytes_per_block = format.bytes_per_block().expect("compressed format");
    let blocks_x = width.div_ceil(block_w as u32);
    let blocks_y = height.div_ceil(block_h as u32);
    let stride = blocks_x * bytes_per_block as u32;
    let total = (blocks_x * blocks_y) as usize * bytes_per_block;

    let surfaces: Vec<Vec<Surface>> = (0..layers)
        .map(|layer_idx| {
            let data: Vec<u8> = (0..total)
                .map(|i| ((i + layer_idx * 17) & 0xff) as u8)
                .collect();
            vec![Surface {
                data,
                width,
                height,
                depth: 1,
                stride,
                slice_stride: 0,
                format,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Opaque,
            }]
        })
        .collect();
    Image {
        surfaces,
        kind: ctt::TextureKind::Texture2D,
    }
}

/// Build a 4×3-tile cross-layout image whose 6 face tiles are each filled
/// with a distinct color in the +X/-X/+Y/-Y/+Z/-Z palette.
///
/// Layout:
/// ```text
///     [+Y]
/// [-X][+Z][+X][-Z]
///     [-Y]
/// ```
/// Returns RGBA8 bytes of size (face*4) × (face*3).
pub fn cross_layout_rgba8(face: u32) -> Vec<u8> {
    let w = face * 4;
    let h = face * 3;
    let mut buf = vec![0u8; (w * h * 4) as usize];

    // (col, row) of each face in face-tile units, matching split_cross.
    let positions = [
        (2u32, 1u32, CUBEMAP_FACE_COLORS[0]), // +X
        (0, 1, CUBEMAP_FACE_COLORS[1]),       // -X
        (1, 0, CUBEMAP_FACE_COLORS[2]),       // +Y
        (1, 2, CUBEMAP_FACE_COLORS[3]),       // -Y
        (1, 1, CUBEMAP_FACE_COLORS[4]),       // +Z
        (3, 1, CUBEMAP_FACE_COLORS[5]),       // -Z
    ];

    for (col, row, color) in positions {
        let x0 = col * face;
        let y0 = row * face;
        for py in 0..face {
            for px in 0..face {
                let off = (((y0 + py) * w + (x0 + px)) * 4) as usize;
                buf[off..off + 4].copy_from_slice(&color);
            }
        }
    }
    buf
}
