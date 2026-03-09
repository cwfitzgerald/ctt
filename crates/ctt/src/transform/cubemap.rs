use crate::error::{Error, Result};
use crate::image::RawImage;

/// Input for cubemap face extraction.
pub enum CubemapInput {
    /// Six separate face images in order: +X, -X, +Y, -Y, +Z, -Z.
    SeparateFaces([RawImage; 6]),
    /// A horizontal or vertical cross layout.
    Cross(RawImage),
    /// A horizontal strip of 6 faces side by side.
    Strip(RawImage),
}

/// Split a cubemap input into its 6 individual faces.
pub fn split_cubemap(input: CubemapInput) -> Result<[RawImage; 6]> {
    match input {
        CubemapInput::SeparateFaces(faces) => validate_uniform_faces(&faces).map(|()| faces),
        CubemapInput::Cross(image) => split_cross(&image),
        CubemapInput::Strip(image) => split_strip(&image),
    }
}

fn validate_uniform_faces(faces: &[RawImage; 6]) -> Result<()> {
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
fn split_cross(image: &RawImage) -> Result<[RawImage; 6]> {
    let face_w = image.width / 4;
    let face_h = image.height / 3;
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

    let faces: Vec<RawImage> = positions
        .iter()
        .map(|&(col, row)| extract_region(image, col * face_w, row * face_h, face_w, face_h))
        .collect();

    Ok(std::array::from_fn(|i| faces[i].clone()))
}

/// Extract faces from a horizontal strip (6 faces side by side).
fn split_strip(image: &RawImage) -> Result<[RawImage; 6]> {
    let face_w = image.width / 6;
    let face_h = image.height;
    if face_w == 0 {
        return Err(Error::InvalidDimensions(
            "strip layout image too small".into(),
        ));
    }

    let faces: Vec<RawImage> = (0..6)
        .map(|i| extract_region(image, i * face_w, 0, face_w, face_h))
        .collect();

    Ok(std::array::from_fn(|i| faces[i].clone()))
}

fn extract_region(
    src: &RawImage,
    src_x: u32,
    src_y: u32,
    width: u32,
    height: u32,
) -> RawImage {
    let bpp = src.pixel_format.components.channel_count();
    let new_stride = width * bpp as u32;
    let mut data = Vec::with_capacity((new_stride * height) as usize);

    for row in 0..height {
        let src_offset = ((src_y + row) * src.stride + src_x * bpp as u32) as usize;
        let row_bytes = &src.data[src_offset..src_offset + new_stride as usize];
        data.extend_from_slice(row_bytes);
    }

    RawImage {
        data,
        width,
        height,
        stride: new_stride,
        pixel_format: src.pixel_format,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ColorSpace, PixelComponents, PixelFormat};

    fn make_face(width: u32, height: u32, fill: u8) -> RawImage {
        let stride = width * 4;
        RawImage {
            data: vec![fill; (stride * height) as usize],
            width,
            height,
            stride,
            pixel_format: PixelFormat {
                components: PixelComponents::Rgba,
                color_space: ColorSpace::Srgb,
            },
        }
    }

    #[test]
    fn separate_faces_passthrough() {
        let faces = std::array::from_fn(|i| make_face(64, 64, i as u8));
        let result = split_cubemap(CubemapInput::SeparateFaces(faces)).unwrap();
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
        let result = split_cubemap(CubemapInput::SeparateFaces(faces));
        assert!(result.is_err());
    }
}
