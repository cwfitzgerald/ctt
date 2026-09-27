use crate::error::{Error, Result};
use crate::processing::equirectangular::{
    self, EquirectangularOrientation, EquirectangularPyramid,
};
use crate::processing::{load, store};
use crate::surface::{ColorSpace, FormatDesc, Image, Surface, SurfaceRef, TextureKind};
use crate::vk_format::FormatExt;

/// Input for cubemap face extraction: one source surface (an atlas or a
/// panorama) and its format.
///
/// To build a cubemap from six separate faces, make an [`Image`] with
/// [`TextureKind::Cubemap`] directly.
pub enum CubemapInput<'a> {
    /// A cross layout — horizontal (4:3) or vertical (3:4); the orientation is
    /// detected from the aspect ratio. See [`split_cubemap`].
    Cross {
        surface: SurfaceRef<'a>,
        desc: FormatDesc,
    },
    /// A horizontal strip of 6 faces side by side.
    Strip {
        surface: SurfaceRef<'a>,
        desc: FormatDesc,
    },
    /// An equirectangular (lat-long) panorama, projected onto six faces
    /// with anisotropic filtering. Faces follow the Vulkan/KTX2 cube map
    /// orientation; the panorama convention (which axis the image center
    /// faces, longitude direction) is set by `orientation`. Faces come
    /// out as `R32G32B32A32_SFLOAT` in linear space.
    Equirectangular {
        surface: SurfaceRef<'a>,
        desc: FormatDesc,
        /// Face edge length. Defaults to a quarter of the source width,
        /// which matches sampling rates at the equator.
        face_size: Option<u32>,
        /// Panorama orientation convention; see [`EquirectangularOrientation`].
        orientation: EquirectangularOrientation,
    },
}

impl CubemapInput<'_> {
    /// Verify that the source is an uncompressed 2D surface with enough data
    /// for its declared size and stride. This keeps the slicing in
    /// `extract_region` from panicking on malformed or short input.
    fn validate(&self) -> Result<()> {
        let (surface, desc) = match self {
            Self::Cross { surface, desc }
            | Self::Strip { surface, desc }
            | Self::Equirectangular { surface, desc, .. } => (*surface, *desc),
        };
        desc.validate()?;
        if desc.format.bytes_per_pixel().is_none() {
            return Err(Error::InvalidImage(format!(
                "cubemap requires an uncompressed format, got {:?}",
                desc.format,
            )));
        }
        if surface.depth != 1 {
            return Err(Error::InvalidImage(format!(
                "cubemap source must be 2D, got depth {}",
                surface.depth,
            )));
        }
        surface
            .validate_layout(desc.format)
            .map_err(|msg| Error::InvalidImage(format!("cubemap source: {msg}")))
    }
}

/// Split a cubemap atlas into a cubemap [`Image`] with one mip level per face.
pub fn split_cubemap(input: CubemapInput<'_>) -> Result<Image> {
    input.validate()?;
    let (faces, desc) = match input {
        CubemapInput::Cross { surface, desc } => {
            log::debug!("Splitting cubemap: cross input");
            log::debug!("Cross source: {}x{}", surface.width, surface.height);
            (split_cross(surface, desc.format)?, desc)
        }
        CubemapInput::Strip { surface, desc } => {
            log::debug!("Splitting cubemap: strip input");
            log::debug!("Strip source: {}x{}", surface.width, surface.height);
            (split_strip(surface, desc.format)?, desc)
        }
        CubemapInput::Equirectangular {
            surface,
            desc,
            face_size,
            orientation,
        } => {
            log::debug!("Splitting cubemap: equirectangular input ({orientation:?})");
            log::debug!(
                "Equirectangular source: {}x{}",
                surface.width,
                surface.height
            );
            project_equirectangular(surface, desc, face_size, orientation)?
        }
    };
    Ok(Image {
        surfaces: faces.into_iter().map(|face| vec![face]).collect(),
        kind: TextureKind::Cubemap,
        desc,
    })
}

/// Project an equirectangular panorama onto six cube faces.
///
/// The projection runs on the linear f32 pipeline: sRGB sources are
/// linearized and straight alpha is premultiplied for filtering, then both
/// are undone on the way out. Faces are stored as `R32G32B32A32_SFLOAT`
/// tagged linear, so no precision is lost after the filter itself. Returns
/// the faces and their format.
fn project_equirectangular(
    surface: SurfaceRef<'_>,
    desc: FormatDesc,
    face_size: Option<u32>,
    orientation: EquirectangularOrientation,
) -> Result<(Vec<Surface>, FormatDesc)> {
    profiling::scope!("project_equirectangular");
    let buf = load::load_f32(surface, desc)?;
    let pyramid = EquirectangularPyramid::new(buf)?;
    let n = face_size.unwrap_or_else(|| pyramid.default_face_size());
    log::debug!(
        "Equirectangular {}x{} → 6 × {n}x{n} faces",
        pyramid.width(),
        pyramid.height(),
    );
    let faces = equirectangular::project_f32(&pyramid, n, orientation)?;
    drop(pyramid);

    let face_desc = FormatDesc {
        format: ktx2::Format::R32G32B32A32_SFLOAT,
        color_space: ColorSpace::Linear,
        alpha: desc.alpha,
    };
    let faces: Vec<Surface> = faces
        .into_iter()
        .map(|face| store::store_f32(face, face_desc))
        .collect::<Result<_>>()?;
    Ok((faces, face_desc))
}

/// Extract faces from a cross layout, detecting orientation from the aspect
/// ratio: wider-than-tall is a horizontal (4:3) cross, taller-than-wide is a
/// vertical (3:4) cross. See [`split_cross_horizontal`] and
/// [`split_cross_vertical`] for the exact face arrangements.
fn split_cross(surface: SurfaceRef<'_>, format: ktx2::Format) -> Result<Vec<Surface>> {
    profiling::scope!("split_cross");
    if surface.width > surface.height {
        split_cross_horizontal(surface, format)
    } else if surface.height > surface.width {
        split_cross_vertical(surface, format)
    } else {
        Err(Error::InvalidImage(format!(
            "cross layout must be 4:3 (horizontal) or 3:4 (vertical); \
             got square {}x{}",
            surface.width, surface.height,
        )))
    }
}

/// Extract faces from a horizontal cross layout.
///
/// Layout (4 wide x 3 tall grid of face-sized tiles):
/// ```text
///     [+Y]
/// [-X][+Z][+X][-Z]
///     [-Y]
/// ```
/// Grid positions: +X=(2,1), -X=(0,1), +Y=(1,0), -Y=(1,2), +Z=(1,1), -Z=(3,1)
fn split_cross_horizontal(surface: SurfaceRef<'_>, format: ktx2::Format) -> Result<Vec<Surface>> {
    if !surface.width.is_multiple_of(4) || !surface.height.is_multiple_of(3) {
        return Err(Error::InvalidImage(format!(
            "horizontal cross requires width divisible by 4 and height by 3, got {}x{}",
            surface.width, surface.height,
        )));
    }
    let face_w = surface.width / 4;
    let face_h = surface.height / 3;
    if face_w != face_h {
        return Err(Error::InvalidImage(format!(
            "horizontal cross faces must be square, got {face_w}x{face_h}",
        )));
    }

    // +X, -X, +Y, -Y, +Z, -Z grid positions (col, row)
    let positions = [
        (2, 1), // +X
        (0, 1), // -X
        (1, 0), // +Y
        (1, 2), // -Y
        (1, 1), // +Z
        (3, 1), // -Z
    ];

    let faces: Vec<Surface> = positions
        .iter()
        .map(|&(col, row)| {
            extract_region(surface, format, col * face_w, row * face_h, face_w, face_h)
        })
        .collect();

    Ok(faces)
}

/// Extract faces from a vertical cross layout.
///
/// Layout (3 wide x 4 tall grid of face-sized tiles):
/// ```text
///     [+Y]
/// [-X][+Z][+X]
///     [-Y]
///     [-Z]
/// ```
/// Grid positions: +X=(2,1), -X=(0,1), +Y=(1,0), -Y=(1,2), +Z=(1,1), -Z=(1,3).
///
/// This follows the conventional vertical cross: the bottom face (-Z) is
/// stored rotated 180° so that folding the cross into a cube yields the same
/// orientation as the horizontal cross. The other five faces are unrotated.
fn split_cross_vertical(surface: SurfaceRef<'_>, format: ktx2::Format) -> Result<Vec<Surface>> {
    if !surface.width.is_multiple_of(3) || !surface.height.is_multiple_of(4) {
        return Err(Error::InvalidImage(format!(
            "vertical cross requires width divisible by 3 and height by 4, got {}x{}",
            surface.width, surface.height,
        )));
    }
    let face_w = surface.width / 3;
    let face_h = surface.height / 4;
    if face_w != face_h {
        return Err(Error::InvalidImage(format!(
            "vertical cross faces must be square, got {face_w}x{face_h}",
        )));
    }

    // +X, -X, +Y, -Y, +Z, -Z grid positions (col, row)
    let positions = [
        (2, 1), // +X
        (0, 1), // -X
        (1, 0), // +Y
        (1, 2), // -Y
        (1, 1), // +Z
        (1, 3), // -Z (rotated 180° below)
    ];

    let mut faces: Vec<Surface> = positions
        .iter()
        .map(|&(col, row)| {
            extract_region(surface, format, col * face_w, row * face_h, face_w, face_h)
        })
        .collect();

    // Conventional vertical cross stores -Z upside-down.
    rotate_180(&mut faces[5], format);

    Ok(faces)
}

/// Extract faces from a horizontal strip (6 faces side by side).
fn split_strip(surface: SurfaceRef<'_>, format: ktx2::Format) -> Result<Vec<Surface>> {
    profiling::scope!("split_strip");
    if !surface.width.is_multiple_of(6) {
        return Err(Error::InvalidImage(format!(
            "strip layout requires width divisible by 6, got {}",
            surface.width,
        )));
    }
    let face_w = surface.width / 6;
    let face_h = surface.height;
    if face_w != face_h {
        return Err(Error::InvalidImage(format!(
            "strip faces must be square, got {face_w}x{face_h}",
        )));
    }

    let faces: Vec<Surface> = (0..6)
        .map(|i| extract_region(surface, format, i * face_w, 0, face_w, face_h))
        .collect();

    Ok(faces)
}

/// Rotate a tightly-packed face 180° in place (both axes flipped).
///
/// `extract_region` always produces a tight surface (`stride == width * bpp`),
/// so a 180° rotation is just a reversal of the pixel sequence.
fn rotate_180(face: &mut Surface, format: ktx2::Format) {
    let bpp = format
        .bytes_per_pixel()
        .expect("cubemap requires uncompressed format");
    let w = face.width as usize;
    let h = face.height as usize;
    let mut rotated = vec![0u8; face.data.len()];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * bpp;
            let dst = ((h - 1 - y) * w + (w - 1 - x)) * bpp;
            rotated[dst..dst + bpp].copy_from_slice(&face.data[src..src + bpp]);
        }
    }
    face.data = rotated;
}

fn extract_region(
    src: SurfaceRef<'_>,
    format: ktx2::Format,
    src_x: u32,
    src_y: u32,
    width: u32,
    height: u32,
) -> Surface {
    profiling::scope!("extract_region");
    let bpp = format
        .bytes_per_pixel()
        .expect("cubemap requires uncompressed format");
    let new_stride = width * bpp as u32;
    let mut data = Vec::with_capacity((new_stride * height) as usize);

    for row in 0..height {
        let src_offset = ((src_y + row) * src.stride + src_x * bpp as u32) as usize;
        let row_bytes = &src.data[src_offset..src_offset + new_stride as usize];
        data.extend_from_slice(row_bytes);
    }

    Surface {
        data,
        width,
        height,
        depth: 1,
        stride: new_stride,
        slice_stride: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;

    const RGBA8: FormatDesc = FormatDesc {
        format: ktx2::Format::R8G8B8A8_UNORM,
        color_space: ColorSpace::Linear,
        alpha: AlphaMode::Straight,
    };

    /// Split `input` and return the six faces in order.
    fn split_faces(input: CubemapInput<'_>) -> Vec<Surface> {
        let cube = split_cubemap(input).unwrap();
        cube.surfaces
            .into_iter()
            .map(|mut layer| layer.remove(0))
            .collect()
    }

    /// Build a `(cols*n) x (rows*n)` RGBA8 atlas where the pixel at global
    /// `(gx, gy)` encodes its grid tile and local offset as
    /// `[col*10 + row, local_x, local_y, 255]`. Lets a split test verify both
    /// which tile a face came from and whether it was rotated.
    fn make_atlas(cols: u32, rows: u32, n: u32) -> Surface {
        let w = cols * n;
        let h = rows * n;
        let stride = w * 4;
        let mut data = vec![0u8; (stride * h) as usize];
        for gy in 0..h {
            for gx in 0..w {
                let col = gx / n;
                let row = gy / n;
                let off = (gy * stride + gx * 4) as usize;
                data[off] = (col * 10 + row) as u8;
                data[off + 1] = (gx % n) as u8;
                data[off + 2] = (gy % n) as u8;
                data[off + 3] = 255;
            }
        }
        Surface {
            data,
            width: w,
            height: h,
            depth: 1,
            stride,
            slice_stride: 0,
        }
    }

    fn face_pixel(face: &Surface, x: u32, y: u32) -> [u8; 4] {
        let off = (y * face.stride + x * 4) as usize;
        face.data[off..off + 4].try_into().unwrap()
    }

    /// Assert `face` (size `n`) was taken from grid tile `(col, row)`, applying
    /// a 180° rotation when `rotated`.
    fn assert_face(face: &Surface, n: u32, col: u32, row: u32, rotated: bool) {
        assert_eq!(face.width, n);
        assert_eq!(face.height, n);
        for ly in 0..n {
            for lx in 0..n {
                let (slx, sly) = if rotated {
                    (n - 1 - lx, n - 1 - ly)
                } else {
                    (lx, ly)
                };
                let want = [(col * 10 + row) as u8, slx as u8, sly as u8, 255];
                assert_eq!(
                    face_pixel(face, lx, ly),
                    want,
                    "tile ({col},{row}) rotated={rotated} at local ({lx},{ly})",
                );
            }
        }
    }

    #[test]
    fn horizontal_cross_splits_into_six_faces() {
        let n = 4;
        let atlas = make_atlas(4, 3, n); // 4:3
        let faces = split_faces(CubemapInput::Cross {
            surface: atlas.as_ref(),
            desc: RGBA8,
        });
        // Emit order +X,-X,+Y,-Y,+Z,-Z; no rotation in the horizontal cross.
        assert_face(&faces[0], n, 2, 1, false); // +X
        assert_face(&faces[1], n, 0, 1, false); // -X
        assert_face(&faces[2], n, 1, 0, false); // +Y
        assert_face(&faces[3], n, 1, 2, false); // -Y
        assert_face(&faces[4], n, 1, 1, false); // +Z
        assert_face(&faces[5], n, 3, 1, false); // -Z
    }

    #[test]
    fn vertical_cross_splits_into_six_faces() {
        let n = 4;
        let atlas = make_atlas(3, 4, n); // 3:4
        let faces = split_faces(CubemapInput::Cross {
            surface: atlas.as_ref(),
            desc: RGBA8,
        });
        // Emit order +X,-X,+Y,-Y,+Z,-Z; -Z is rotated 180°.
        assert_face(&faces[0], n, 2, 1, false); // +X
        assert_face(&faces[1], n, 0, 1, false); // -X
        assert_face(&faces[2], n, 1, 0, false); // +Y
        assert_face(&faces[3], n, 1, 2, false); // -Y
        assert_face(&faces[4], n, 1, 1, false); // +Z
        assert_face(&faces[5], n, 1, 3, true); // -Z rotated 180°
    }

    #[test]
    fn cross_short_data_errors_no_panic() {
        // Valid 4:3 aspect but truncated data must error, not panic.
        let mut atlas = make_atlas(4, 3, 4);
        atlas.data.truncate(10);
        let err = split_cubemap(CubemapInput::Cross {
            surface: atlas.as_ref(),
            desc: RGBA8,
        })
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidImage(_)),
            "expected InvalidImage, got {err:?}",
        );
    }

    #[test]
    fn cross_non_divisible_dims_rejected() {
        // 10x9 is wider-than-tall (horizontal) but 10 % 4 != 0.
        let mut atlas = make_atlas(4, 3, 4);
        atlas.width = 10;
        atlas.height = 9;
        atlas.stride = 10 * 4;
        atlas.data = vec![0u8; (atlas.stride * atlas.height) as usize];
        let err = split_cubemap(CubemapInput::Cross {
            surface: atlas.as_ref(),
            desc: RGBA8,
        })
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidImage(_)),
            "expected InvalidImage, got {err:?}",
        );
    }

    #[test]
    fn cross_square_rejected() {
        let atlas = make_atlas(4, 4, 4); // square → not a cross
        let err = split_cubemap(CubemapInput::Cross {
            surface: atlas.as_ref(),
            desc: RGBA8,
        })
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidImage(_)),
            "expected InvalidImage, got {err:?}",
        );
    }

    #[test]
    fn strip_non_divisible_rejected() {
        // width 20 is not divisible by 6.
        let mut atlas = make_atlas(6, 1, 4);
        atlas.width = 20;
        atlas.stride = 20 * 4;
        atlas.data = vec![0u8; (atlas.stride * atlas.height) as usize];
        let err = split_cubemap(CubemapInput::Strip {
            surface: atlas.as_ref(),
            desc: RGBA8,
        })
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidImage(_)),
            "expected InvalidImage, got {err:?}",
        );
    }

    #[test]
    fn strip_splits_into_six_square_faces() {
        let n = 4;
        let atlas = make_atlas(6, 1, n);
        let faces = split_faces(CubemapInput::Strip {
            surface: atlas.as_ref(),
            desc: RGBA8,
        });
        for (i, face) in faces.iter().enumerate() {
            assert_face(face, n, i as u32, 0, false);
        }
    }

    #[test]
    fn split_returns_cubemap_image() {
        let atlas = make_atlas(6, 1, 4);
        let cube = split_cubemap(CubemapInput::Strip {
            surface: atlas.as_ref(),
            desc: RGBA8,
        })
        .unwrap();
        assert_eq!(cube.kind, TextureKind::Cubemap);
        assert_eq!(cube.desc, RGBA8);
        assert_eq!(cube.surfaces.len(), 6);
        assert!(cube.surfaces.iter().all(|layer| layer.len() == 1));
        cube.to_ref().validate().unwrap();
    }

    #[test]
    fn compressed_source_rejected() {
        let atlas = make_atlas(6, 1, 4);
        let err = split_cubemap(CubemapInput::Strip {
            surface: atlas.as_ref(),
            desc: FormatDesc {
                format: ktx2::Format::BC7_UNORM_BLOCK,
                ..RGBA8
            },
        })
        .unwrap_err();
        assert!(err.to_string().contains("uncompressed"), "got: {err}");
    }

    #[test]
    fn volume_source_rejected() {
        let mut atlas = make_atlas(6, 1, 4);
        atlas.depth = 2;
        let err = split_cubemap(CubemapInput::Strip {
            surface: atlas.as_ref(),
            desc: RGBA8,
        })
        .unwrap_err();
        assert!(err.to_string().contains("must be 2D"), "got: {err}");
    }
}
