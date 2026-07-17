use crate::error::{Error, Result};
use crate::processing::equirectangular::{
    self, EquirectangularOrientation, EquirectangularPyramid,
};
use crate::processing::{load, store};
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::FormatExt;

/// Input for cubemap face extraction.
pub enum CubemapInput {
    /// Six separate face images in order: +X, -X, +Y, -Y, +Z, -Z.
    SeparateFaces(Box<[Surface; 6]>),
    /// A cross layout — horizontal (4:3) or vertical (3:4); the orientation is
    /// detected from the aspect ratio. See [`split_cubemap`].
    Cross(Surface),
    /// A horizontal strip of 6 faces side by side.
    Strip(Surface),
    /// An equirectangular (lat-long) panorama, projected onto six faces
    /// with anisotropic filtering. Faces follow the Vulkan/KTX2 cube map
    /// orientation; the panorama convention (which axis the image center
    /// faces, longitude direction) is set by `orientation`. Faces come
    /// out as `R32G32B32A32_SFLOAT` in linear space.
    Equirectangular {
        surface: Surface,
        /// Face edge length. Defaults to a quarter of the source width,
        /// which matches sampling rates at the equator.
        face_size: Option<u32>,
        /// Panorama orientation convention; see [`EquirectangularOrientation`].
        orientation: EquirectangularOrientation,
    },
}

/// Split a cubemap input into its 6 individual faces.
pub fn split_cubemap(input: CubemapInput) -> Result<[Surface; 6]> {
    match input {
        CubemapInput::SeparateFaces(faces) => {
            log::debug!("Splitting cubemap: separate faces input");
            validate_uniform_faces(&faces).map(|()| *faces)
        }
        CubemapInput::Cross(surface) => {
            log::debug!("Splitting cubemap: cross input");
            log::debug!("Cross source: {}x{}", surface.width, surface.height);
            split_cross(&surface)
        }
        CubemapInput::Strip(surface) => {
            log::debug!("Splitting cubemap: strip input");
            log::debug!("Strip source: {}x{}", surface.width, surface.height);
            split_strip(&surface)
        }
        CubemapInput::Equirectangular {
            surface,
            face_size,
            orientation,
        } => {
            log::debug!("Splitting cubemap: equirectangular input ({orientation:?})");
            log::debug!(
                "Equirectangular source: {}x{}",
                surface.width,
                surface.height
            );
            project_equirectangular(surface, face_size, orientation)
        }
    }
}

/// Project an equirectangular panorama onto six cube faces.
///
/// The projection runs on the linear f32 pipeline: sRGB sources are
/// linearized and straight alpha is premultiplied for filtering, then both
/// are undone on the way out. Faces are stored as `R32G32B32A32_SFLOAT`
/// tagged linear, so no precision is lost after the filter itself.
fn project_equirectangular(
    surface: Surface,
    face_size: Option<u32>,
    orientation: EquirectangularOrientation,
) -> Result<[Surface; 6]> {
    profiling::scope!("project_equirectangular");
    validate_source(&surface)?;
    let alpha = surface.alpha;
    let buf = load::load_f32(&surface)?;
    drop(surface);
    let pyramid = EquirectangularPyramid::new(buf)?;
    let n = face_size.unwrap_or_else(|| pyramid.default_face_size());
    log::debug!(
        "Equirectangular {}x{} → 6 × {n}x{n} faces",
        pyramid.width(),
        pyramid.height(),
    );
    let faces = equirectangular::project_f32(&pyramid, n, orientation)?;
    drop(pyramid);

    let faces: Vec<Surface> = faces
        .into_iter()
        .map(|face| {
            store::store_f32(
                face,
                ktx2::Format::R32G32B32A32_SFLOAT,
                ColorSpace::Linear,
                alpha,
            )
        })
        .collect::<Result<_>>()?;
    Ok(faces.try_into().unwrap_or_else(|_| unreachable!()))
}

fn validate_uniform_faces(faces: &[Surface; 6]) -> Result<()> {
    for face in faces.iter() {
        validate_face(face)?;
    }
    let (w, h) = (faces[0].width, faces[0].height);
    for face in &faces[1..] {
        if face.width != w || face.height != h {
            return Err(Error::CubemapNonUniformFaces);
        }
    }
    Ok(())
}

/// Validate a face that will be passed through without pixel extraction.
/// Unlike cross and strip sources, separate faces may already be block
/// compressed.
fn validate_face(s: &Surface) -> Result<()> {
    if s.width == 0 || s.height == 0 {
        return Err(Error::InvalidImage(format!(
            "cubemap face has a zero dimension: {}x{}",
            s.width, s.height,
        )));
    }
    let tight_row = s.tight_row_bytes().ok_or_else(|| {
        Error::InvalidImage(format!(
            "cubemap face has unsupported format {:?}",
            s.format,
        ))
    })?;
    if s.stride < tight_row {
        return Err(Error::InvalidImage(format!(
            "cubemap face stride {} is below the tight minimum {tight_row}",
            s.stride,
        )));
    }
    let rows = if let Some((_, block_height)) = s.format.block_size() {
        s.height.div_ceil(block_height as u32)
    } else {
        s.height
    };
    let required = (rows as usize - 1)
        .checked_mul(s.stride as usize)
        .and_then(|prefix| prefix.checked_add(tight_row as usize))
        .ok_or_else(|| Error::InvalidImage("cubemap face size overflows usize".into()))?;
    if s.data.len() < required {
        return Err(Error::InvalidImage(format!(
            "cubemap face data is {} bytes, need at least {required}",
            s.data.len(),
        )));
    }
    Ok(())
}

/// Validate that a source surface is uncompressed and carries enough data for
/// its declared dimensions and stride. This is what keeps `extract_region`'s
/// slicing (and its `bytes_per_pixel` unwrap) from panicking on malformed or
/// short input.
fn validate_source(s: &Surface) -> Result<()> {
    let Some(bpp) = s.format.bytes_per_pixel() else {
        return Err(Error::InvalidImage(format!(
            "cubemap requires an uncompressed format, got {:?}",
            s.format,
        )));
    };
    if s.width == 0 || s.height == 0 {
        return Err(Error::InvalidImage(format!(
            "cubemap source has a zero dimension: {}x{}",
            s.width, s.height,
        )));
    }
    // Widen before multiplying — face atlases can be large.
    let tight_row = s.width as usize * bpp;
    if (s.stride as usize) < tight_row {
        return Err(Error::InvalidImage(format!(
            "cubemap source stride {} is below the tight minimum {tight_row}",
            s.stride,
        )));
    }
    let required = (s.height as usize - 1) * s.stride as usize + tight_row;
    if s.data.len() < required {
        return Err(Error::InvalidImage(format!(
            "cubemap source data is {} bytes, need at least {required} for \
             {}x{} at stride {}",
            s.data.len(),
            s.width,
            s.height,
            s.stride,
        )));
    }
    Ok(())
}

/// Extract faces from a cross layout, detecting orientation from the aspect
/// ratio: wider-than-tall is a horizontal (4:3) cross, taller-than-wide is a
/// vertical (3:4) cross. See [`split_cross_horizontal`] and
/// [`split_cross_vertical`] for the exact face arrangements.
fn split_cross(surface: &Surface) -> Result<[Surface; 6]> {
    profiling::scope!("split_cross");
    validate_source(surface)?;
    if surface.width > surface.height {
        split_cross_horizontal(surface)
    } else if surface.height > surface.width {
        split_cross_vertical(surface)
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
fn split_cross_horizontal(surface: &Surface) -> Result<[Surface; 6]> {
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
        .map(|&(col, row)| extract_region(surface, col * face_w, row * face_h, face_w, face_h))
        .collect();

    Ok(std::array::from_fn(|i| faces[i].clone()))
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
fn split_cross_vertical(surface: &Surface) -> Result<[Surface; 6]> {
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
        .map(|&(col, row)| extract_region(surface, col * face_w, row * face_h, face_w, face_h))
        .collect();

    // Conventional vertical cross stores -Z upside-down.
    rotate_180(&mut faces[5]);

    Ok(std::array::from_fn(|i| faces[i].clone()))
}

/// Extract faces from a horizontal strip (6 faces side by side).
fn split_strip(surface: &Surface) -> Result<[Surface; 6]> {
    profiling::scope!("split_strip");
    validate_source(surface)?;
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
        .map(|i| extract_region(surface, i * face_w, 0, face_w, face_h))
        .collect();

    Ok(std::array::from_fn(|i| faces[i].clone()))
}

/// Rotate a tightly-packed face 180° in place (both axes flipped).
///
/// `extract_region` always produces a tight surface (`stride == width * bpp`),
/// so a 180° rotation is just a reversal of the pixel sequence.
fn rotate_180(face: &mut Surface) {
    let bpp = face
        .format
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

fn extract_region(src: &Surface, src_x: u32, src_y: u32, width: u32, height: u32) -> Surface {
    profiling::scope!("extract_region");
    let bpp = src
        .format
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
        format: src.format,
        color_space: src.color_space,
        alpha: src.alpha,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    fn make_face(width: u32, height: u32, fill: u8) -> Surface {
        let stride = width * 4;
        Surface {
            data: vec![fill; (stride * height) as usize],
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Straight,
        }
    }

    #[test]
    fn separate_faces_passthrough() {
        let faces = std::array::from_fn(|i| make_face(64, 64, i as u8));
        let result = split_cubemap(CubemapInput::SeparateFaces(Box::new(faces))).unwrap();
        for (i, face) in result.iter().enumerate() {
            assert_eq!(face.width, 64);
            assert_eq!(face.height, 64);
            assert_eq!(face.data[0], i as u8);
        }
    }

    #[test]
    fn compressed_separate_faces_passthrough() {
        let faces = std::array::from_fn(|i| Surface {
            data: vec![i as u8; 16 * 4],
            width: 8,
            height: 8,
            depth: 1,
            stride: 32,
            slice_stride: 0,
            format: ktx2::Format::BC7_UNORM_BLOCK,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        });

        let split = split_cubemap(CubemapInput::SeparateFaces(Box::new(faces))).unwrap();
        for (i, face) in split.iter().enumerate() {
            assert_eq!(face.format, ktx2::Format::BC7_UNORM_BLOCK);
            assert_eq!(face.data, vec![i as u8; 16 * 4]);
        }
    }

    #[test]
    fn non_uniform_faces_error() {
        let mut faces = std::array::from_fn(|_| make_face(64, 64, 0));
        faces[3] = make_face(32, 32, 0);
        let result = split_cubemap(CubemapInput::SeparateFaces(Box::new(faces)));
        assert!(result.is_err());
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
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Straight,
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
        let faces = split_cubemap(CubemapInput::Cross(atlas)).unwrap();
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
        let faces = split_cubemap(CubemapInput::Cross(atlas)).unwrap();
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
        let err = split_cubemap(CubemapInput::Cross(atlas)).unwrap_err();
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
        let err = split_cubemap(CubemapInput::Cross(atlas)).unwrap_err();
        assert!(
            matches!(err, Error::InvalidImage(_)),
            "expected InvalidImage, got {err:?}",
        );
    }

    #[test]
    fn cross_square_rejected() {
        let atlas = make_atlas(4, 4, 4); // square → not a cross
        let err = split_cubemap(CubemapInput::Cross(atlas)).unwrap_err();
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
        let err = split_cubemap(CubemapInput::Strip(atlas)).unwrap_err();
        assert!(
            matches!(err, Error::InvalidImage(_)),
            "expected InvalidImage, got {err:?}",
        );
    }

    #[test]
    fn strip_splits_into_six_square_faces() {
        let n = 4;
        let atlas = make_atlas(6, 1, n);
        let faces = split_cubemap(CubemapInput::Strip(atlas)).unwrap();
        for (i, face) in faces.iter().enumerate() {
            assert_face(face, n, i as u32, 0, false);
        }
    }
}
