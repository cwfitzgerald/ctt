use crate::error::{Error, Result};
use crate::surface::Surface;
use crate::vk_format::FormatExt;

/// Input for cubemap face extraction.
pub enum CubemapInput {
    /// Six separate face images in order: +X, -X, +Y, -Y, +Z, -Z.
    SeparateFaces(Box<[Surface; 6]>),
    /// A horizontal or vertical cross layout.
    Cross(Surface),
    /// A horizontal strip of 6 faces side by side.
    Strip(Surface),
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
    }
}

fn validate_uniform_faces(faces: &[Surface; 6]) -> Result<()> {
    let (w, h) = (faces[0].width, faces[0].height);
    for face in &faces[1..] {
        if face.width != w || face.height != h {
            return Err(Error::CubemapNonUniformFaces);
        }
    }
    Ok(())
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
fn split_cross(surface: &Surface) -> Result<[Surface; 6]> {
    profiling::scope!("split_cross");
    let face_w = surface.width / 4;
    let face_h = surface.height / 3;
    if face_w == 0 || face_h == 0 {
        return Err(Error::InvalidDimensions(
            "cross layout image too small".into(),
        ));
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

/// Extract faces from a horizontal strip (6 faces side by side).
fn split_strip(surface: &Surface) -> Result<[Surface; 6]> {
    profiling::scope!("split_strip");
    let face_w = surface.width / 6;
    let face_h = surface.height;
    if face_w == 0 {
        return Err(Error::InvalidDimensions(
            "strip layout image too small".into(),
        ));
    }

    let faces: Vec<Surface> = (0..6)
        .map(|i| extract_region(surface, i * face_w, 0, face_w, face_h))
        .collect();

    Ok(std::array::from_fn(|i| faces[i].clone()))
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
    fn non_uniform_faces_error() {
        let mut faces = std::array::from_fn(|_| make_face(64, 64, 0));
        faces[3] = make_face(32, 32, 0);
        let result = split_cubemap(CubemapInput::SeparateFaces(Box::new(faces)));
        assert!(result.is_err());
    }
}
