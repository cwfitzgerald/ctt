//! Regenerate the golden test inputs under `crates/ctt-cli/tests/data/`.
//!
//! These files are checked in. CI does not run this command; developers
//! invoke it whenever the synthesis rules change.
//!
//! Run with `cargo run -p regen-test-data`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ctt::{
    AlphaMode, ColorSpace, Container, ConvertSettings, Format, FormatExt, Image, PipelineOutput,
    Quality, Surface, TargetFormat,
};

const FACE_COLORS: [[u8; 4]; 6] = [
    [255, 0, 0, 255],
    [0, 255, 255, 255],
    [0, 255, 0, 255],
    [255, 0, 255, 255],
    [0, 0, 255, 255],
    [255, 255, 0, 255],
];

fn main() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("regen-test-data is a workspace member");
    let dir = workspace
        .join("crates")
        .join("ctt-cli")
        .join("tests")
        .join("data");
    fs::create_dir_all(&dir)?;

    write_rgba8_gradient_ktx2(&dir, 16, 16, ColorSpace::Linear, "rgba8_16x16_linear.ktx2")?;
    write_rgba8_gradient_ktx2(&dir, 16, 16, ColorSpace::Srgb, "rgba8_16x16_srgb.ktx2")?;

    // BC7 4x4 single-block: encode a solid mid-grey 4x4 to BC7.
    write_bc7_solid(
        &dir,
        4,
        4,
        [128, 128, 128, 255],
        Container::Ktx2(None),
        "bc7_4x4.ktx2",
    )?;
    write_bc7_solid(
        &dir,
        4,
        4,
        [128, 128, 128, 255],
        Container::Dds,
        "bc7_4x4.dds",
    )?;

    // Cross-layout RGBA8 palette: 16-pixel face → 64×48 image.
    write_cross_palette_ktx2(&dir, 16, "cross_palette_64x48.ktx2")?;

    // Six per-face KTX2 files for the 6-input cubemap test.
    for (i, name) in ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"]
        .iter()
        .enumerate()
    {
        write_solid_ktx2(
            &dir,
            16,
            16,
            FACE_COLORS[i],
            ColorSpace::Linear,
            &format!("cube_face_{name}.ktx2"),
        )?;
    }

    // An already-cubemap KTX2 with each face filled with the palette color.
    write_cubemap_palette_ktx2(&dir, 16, "cube_palette_16.ktx2")?;

    println!("Wrote test data to {}", dir.display());
    Ok(())
}

fn write_rgba8_gradient_ktx2(dir: &Path, w: u32, h: u32, cs: ColorSpace, name: &str) -> Result<()> {
    let data = rgba8_gradient(w, h);
    let image = make_rgba8_image(data, w, h, cs, AlphaMode::Opaque);
    write_container(dir.join(name), image, Container::Ktx2(None))
}

fn write_solid_ktx2(
    dir: &Path,
    w: u32,
    h: u32,
    color: [u8; 4],
    cs: ColorSpace,
    name: &str,
) -> Result<()> {
    let data = rgba8_solid(w, h, color);
    let image = make_rgba8_image(data, w, h, cs, AlphaMode::Opaque);
    write_container(dir.join(name), image, Container::Ktx2(None))
}

fn write_cross_palette_ktx2(dir: &Path, face: u32, name: &str) -> Result<()> {
    let data = cross_layout_rgba8(face);
    let image = make_rgba8_image(
        data,
        face * 4,
        face * 3,
        ColorSpace::Linear,
        AlphaMode::Opaque,
    );
    write_container(dir.join(name), image, Container::Ktx2(None))
}

fn write_cubemap_palette_ktx2(dir: &Path, face: u32, name: &str) -> Result<()> {
    let surfaces: Vec<Vec<Surface>> = FACE_COLORS
        .iter()
        .map(|&color| {
            vec![Surface {
                data: rgba8_solid(face, face, color),
                width: face,
                height: face,
                stride: face * 4,
                format: Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Opaque,
            }]
        })
        .collect();
    let image = Image {
        surfaces,
        is_cubemap: true,
    };
    write_container(dir.join(name), image, Container::Ktx2(None))
}

fn write_bc7_solid(
    dir: &Path,
    w: u32,
    h: u32,
    color: [u8; 4],
    container: Container,
    name: &str,
) -> Result<()> {
    let data = rgba8_solid(w, h, color);
    let image = make_rgba8_image(data, w, h, ColorSpace::Linear, AlphaMode::Opaque);
    let bytes = match ctt::convert(
        image,
        ConvertSettings {
            format: Some(TargetFormat::Compressed {
                // Pin to bc7e so the golden bytes stay stable even if more
                // BC7-capable encoders get added to the workspace later.
                encoder_name: Some("bc7e".to_string()),
                format: Format::BC7_UNORM_BLOCK,
            }),
            container,
            quality: Quality::UltraFast,
            ..Default::default()
        },
    )? {
        PipelineOutput::Encoded(b) => b,
        PipelineOutput::Raw(_) => unreachable!(),
    };
    fs::write(dir.join(name), bytes)?;
    Ok(())
}

fn write_container(path: PathBuf, image: Image, container: Container) -> Result<()> {
    let bytes = match ctt::convert(
        image,
        ConvertSettings {
            format: None,
            container,
            ..Default::default()
        },
    )? {
        PipelineOutput::Encoded(b) => b,
        PipelineOutput::Raw(_) => unreachable!(),
    };
    fs::write(path, bytes)?;
    Ok(())
}

fn make_rgba8_image(
    data: Vec<u8>,
    w: u32,
    h: u32,
    color_space: ColorSpace,
    alpha: AlphaMode,
) -> Image {
    let bpp = Format::R8G8B8A8_UNORM.bytes_per_pixel().unwrap() as u32;
    Image {
        surfaces: vec![vec![Surface {
            data,
            width: w,
            height: h,
            stride: w * bpp,
            format: Format::R8G8B8A8_UNORM,
            color_space,
            alpha,
        }]],
        is_cubemap: false,
    }
}

fn rgba8_solid(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..w * h {
        buf.extend_from_slice(&color);
    }
    buf
}

fn rgba8_gradient(w: u32, h: u32) -> Vec<u8> {
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

fn cross_layout_rgba8(face: u32) -> Vec<u8> {
    let w = face * 4;
    let h = face * 3;
    let mut buf = vec![0u8; (w * h * 4) as usize];

    let positions = [
        (2u32, 1u32, FACE_COLORS[0]),
        (0, 1, FACE_COLORS[1]),
        (1, 0, FACE_COLORS[2]),
        (1, 2, FACE_COLORS[3]),
        (1, 1, FACE_COLORS[4]),
        (3, 1, FACE_COLORS[5]),
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
