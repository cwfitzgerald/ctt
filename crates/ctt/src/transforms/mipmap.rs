use image::{
    DynamicImage, GrayAlphaImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, Rgb32FImage, RgbImage,
    Rgba, Rgba32FImage, RgbaImage,
};

use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::{Error, Result};
use crate::surface::{ColorSpace, Image, Surface};
use crate::transforms::Transform;
use crate::vk_format::FormatExt;

/// Supported filter types for mipmap downsampling.
///
/// Mirrors [`image::imageops::FilterType`] so library consumers don't need
/// to depend on the `image` crate directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MipmapFilter {
    Nearest,
    Triangle,
    #[default]
    CatmullRom,
    Gaussian,
    Lanczos3,
}

impl MipmapFilter {
    fn to_image_filter(self) -> image::imageops::FilterType {
        match self {
            Self::Nearest => image::imageops::FilterType::Nearest,
            Self::Triangle => image::imageops::FilterType::Triangle,
            Self::CatmullRom => image::imageops::FilterType::CatmullRom,
            Self::Gaussian => image::imageops::FilterType::Gaussian,
            Self::Lanczos3 => image::imageops::FilterType::Lanczos3,
        }
    }
}

/// All ktx2 formats that map to an `image::DynamicImage` variant.
const SUPPORTED_FORMATS: &[ktx2::Format] = &[
    ktx2::Format::R8_UNORM,
    ktx2::Format::R8G8_UNORM,
    ktx2::Format::R8G8B8_UNORM,
    ktx2::Format::R8G8B8A8_UNORM,
    ktx2::Format::R16_UNORM,
    ktx2::Format::R16G16_UNORM,
    ktx2::Format::R16G16B16_UNORM,
    ktx2::Format::R16G16B16A16_UNORM,
    ktx2::Format::R32G32B32_SFLOAT,
    ktx2::Format::R32G32B32A32_SFLOAT,
];

/// Compute the full mip chain length for the given dimensions (including the base level).
fn full_mip_count(width: u32, height: u32) -> usize {
    (width.max(height).max(1) as f64).log2().floor() as usize + 1
}

/// Convert a [`Surface`] into a [`DynamicImage`].
///
/// The surface format must be one of [`SUPPORTED_FORMATS`].
fn surface_to_dynamic(surface: &Surface) -> Result<DynamicImage> {
    let w = surface.width;
    let h = surface.height;
    let stride = surface.stride as usize;
    let bpp = surface
        .format
        .bytes_per_pixel()
        .ok_or_else(|| Error::UnsupportedFormat(format!("unknown bpp for {:?}", surface.format)))?;
    let row_bytes = w as usize * bpp;

    // Strip row padding if stride > tight row width.
    let tight: Vec<u8> = if stride == row_bytes {
        surface.data.clone()
    } else {
        (0..h as usize)
            .flat_map(|y| {
                let start = y * stride;
                surface.data[start..start + row_bytes].iter().copied()
            })
            .collect()
    };

    let img = match surface.format {
        ktx2::Format::R8_UNORM => {
            let buf = GrayImage::from_raw(w, h, tight)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageLuma8(buf)
        }
        ktx2::Format::R8G8_UNORM => {
            let buf = GrayAlphaImage::from_raw(w, h, tight)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageLumaA8(buf)
        }
        ktx2::Format::R8G8B8_UNORM => {
            let buf = RgbImage::from_raw(w, h, tight)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageRgb8(buf)
        }
        ktx2::Format::R8G8B8A8_UNORM => {
            let buf = RgbaImage::from_raw(w, h, tight)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageRgba8(buf)
        }
        ktx2::Format::R16_UNORM => {
            let pixels: Vec<u16> = bytemuck::cast_slice(&tight).to_vec();
            let buf = ImageBuffer::<Luma<u16>, _>::from_raw(w, h, pixels)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageLuma16(buf)
        }
        ktx2::Format::R16G16_UNORM => {
            let pixels: Vec<u16> = bytemuck::cast_slice(&tight).to_vec();
            let buf = ImageBuffer::<LumaA<u16>, _>::from_raw(w, h, pixels)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageLumaA16(buf)
        }
        ktx2::Format::R16G16B16_UNORM => {
            let pixels: Vec<u16> = bytemuck::cast_slice(&tight).to_vec();
            let buf = ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, pixels)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageRgb16(buf)
        }
        ktx2::Format::R16G16B16A16_UNORM => {
            let pixels: Vec<u16> = bytemuck::cast_slice(&tight).to_vec();
            let buf = ImageBuffer::<Rgba<u16>, _>::from_raw(w, h, pixels)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageRgba16(buf)
        }
        ktx2::Format::R32G32B32_SFLOAT => {
            let pixels: Vec<f32> = bytemuck::cast_slice(&tight).to_vec();
            let buf = Rgb32FImage::from_raw(w, h, pixels)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageRgb32F(buf)
        }
        ktx2::Format::R32G32B32A32_SFLOAT => {
            let pixels: Vec<f32> = bytemuck::cast_slice(&tight).to_vec();
            let buf = Rgba32FImage::from_raw(w, h, pixels)
                .ok_or_else(|| Error::UnsupportedFormat("buffer size mismatch".into()))?;
            DynamicImage::ImageRgba32F(buf)
        }
        other => {
            return Err(Error::UnsupportedFormat(format!(
                "mipmap: unsupported format {other:?}"
            )));
        }
    };
    Ok(img)
}

/// Convert a [`DynamicImage`] back into a [`Surface`], inheriting format metadata from `template`.
///
/// The `DynamicImage` variant must already match the template format (i.e. it came from
/// `surface_to_dynamic` on the same format). This function will error if the variant doesn't match.
fn dynamic_to_surface(img: &DynamicImage, template: &Surface) -> Result<Surface> {
    let (w, h) = (img.width(), img.height());
    let bpp = template.format.bytes_per_pixel().ok_or_else(|| {
        Error::UnsupportedFormat(format!("unknown bpp for {:?}", template.format))
    })?;
    let stride = w * bpp as u32;

    let mismatch = || {
        Error::UnsupportedFormat(format!(
            "mipmap: DynamicImage variant does not match expected format {:?}",
            template.format,
        ))
    };

    let data: Vec<u8> = match (template.format, img) {
        (ktx2::Format::R8_UNORM, DynamicImage::ImageLuma8(buf)) => buf.as_raw().clone(),
        (ktx2::Format::R8G8_UNORM, DynamicImage::ImageLumaA8(buf)) => buf.as_raw().clone(),
        (ktx2::Format::R8G8B8_UNORM, DynamicImage::ImageRgb8(buf)) => buf.as_raw().clone(),
        (ktx2::Format::R8G8B8A8_UNORM, DynamicImage::ImageRgba8(buf)) => buf.as_raw().clone(),
        (ktx2::Format::R16_UNORM, DynamicImage::ImageLuma16(buf)) => {
            bytemuck::cast_slice(buf.as_raw().as_slice()).to_vec()
        }
        (ktx2::Format::R16G16_UNORM, DynamicImage::ImageLumaA16(buf)) => {
            bytemuck::cast_slice(buf.as_raw().as_slice()).to_vec()
        }
        (ktx2::Format::R16G16B16_UNORM, DynamicImage::ImageRgb16(buf)) => {
            bytemuck::cast_slice(buf.as_raw().as_slice()).to_vec()
        }
        (ktx2::Format::R16G16B16A16_UNORM, DynamicImage::ImageRgba16(buf)) => {
            bytemuck::cast_slice(buf.as_raw().as_slice()).to_vec()
        }
        (ktx2::Format::R32G32B32_SFLOAT, DynamicImage::ImageRgb32F(buf)) => {
            bytemuck::cast_slice(buf.as_raw().as_slice()).to_vec()
        }
        (ktx2::Format::R32G32B32A32_SFLOAT, DynamicImage::ImageRgba32F(buf)) => {
            bytemuck::cast_slice(buf.as_raw().as_slice()).to_vec()
        }
        _ => return Err(mismatch()),
    };

    Ok(Surface {
        data,
        width: w,
        height: h,
        stride,
        format: template.format,
        color_space: template.color_space,
        alpha: template.alpha,
    })
}

/// A transform that generates a mip chain by cascading `DynamicImage::resize_exact`.
pub struct MipmapTransform {
    /// Total number of mip levels (including the base). `None` = full chain down to 1×1.
    mip_count: Option<usize>,
    filter: MipmapFilter,
}

impl MipmapTransform {
    pub fn new(mip_count: Option<usize>, filter: MipmapFilter) -> Self {
        Self { mip_count, filter }
    }
}

impl Transform for MipmapTransform {
    fn name(&self) -> &str {
        "mipmap"
    }

    fn constraint(&self) -> FormatConstraint {
        FormatConstraint {
            formats: Some(SUPPORTED_FORMATS.to_vec()),
            color_spaces: Some(vec![ColorSpace::Linear]),
            alpha_modes: Some(vec![AlphaMode::Premultiplied]),
        }
    }

    fn output_format(
        &self,
        input: ktx2::Format,
        cs: ColorSpace,
        alpha: AlphaMode,
    ) -> (ktx2::Format, ColorSpace, AlphaMode) {
        (input, cs, alpha)
    }

    fn execute(&self, mut image: Image) -> Result<Image> {
        let filter = self.filter.to_image_filter();

        for layer in &mut image.surfaces {
            if layer.is_empty() {
                continue;
            }

            let base = &layer[0];
            let target_count = self
                .mip_count
                .unwrap_or_else(|| full_mip_count(base.width, base.height));

            // Trim excess mips.
            if layer.len() > target_count {
                layer.truncate(target_count);
                continue;
            }

            // Skip if already complete.
            if layer.len() == target_count {
                continue;
            }

            log::info!(
                "Generating mip chain: {}x{}, {} -> {} levels, filter {:?}",
                base.width,
                base.height,
                layer.len(),
                target_count,
                self.filter,
            );

            // Start cascade from the last existing mip.
            let mut prev = surface_to_dynamic(layer.last().unwrap())?;

            while layer.len() < target_count {
                let new_w = (prev.width() / 2).max(1);
                let new_h = (prev.height() / 2).max(1);

                let resized = prev.resize_exact(new_w, new_h, filter);
                let surface = dynamic_to_surface(&resized, &layer[0])?;
                layer.push(surface);

                prev = resized;
            }
        }

        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_surface(width: u32, height: u32) -> Surface {
        let stride = width * 4;
        Surface {
            data: vec![128u8; (stride * height) as usize],
            width,
            height,
            stride,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Premultiplied,
        }
    }

    #[test]
    fn full_mip_count_powers_of_two() {
        assert_eq!(full_mip_count(256, 256), 9); // 256→1 = 9 levels
        assert_eq!(full_mip_count(1, 1), 1);
        assert_eq!(full_mip_count(2, 2), 2);
        assert_eq!(full_mip_count(4, 4), 3);
    }

    #[test]
    fn full_mip_count_non_square() {
        assert_eq!(full_mip_count(256, 1), 9);
        assert_eq!(full_mip_count(1, 128), 8);
    }

    #[test]
    fn generate_all_mips() {
        let image = Image {
            surfaces: vec![vec![make_surface(16, 16)]],
            is_cubemap: false,
        };

        let transform = MipmapTransform::new(None, MipmapFilter::CatmullRom);
        let result = transform.execute(image).unwrap();

        assert_eq!(result.surfaces[0].len(), 5); // 16, 8, 4, 2, 1
        assert_eq!(result.surfaces[0][1].width, 8);
        assert_eq!(result.surfaces[0][1].height, 8);
        assert_eq!(result.surfaces[0][4].width, 1);
        assert_eq!(result.surfaces[0][4].height, 1);
    }

    #[test]
    fn generate_limited_mips() {
        let image = Image {
            surfaces: vec![vec![make_surface(64, 64)]],
            is_cubemap: false,
        };

        let transform = MipmapTransform::new(Some(3), MipmapFilter::CatmullRom);
        let result = transform.execute(image).unwrap();

        assert_eq!(result.surfaces[0].len(), 3); // 64, 32, 16
    }

    #[test]
    fn skip_existing_mips() {
        let image = Image {
            surfaces: vec![vec![
                make_surface(16, 16),
                make_surface(8, 8),
                make_surface(4, 4),
                make_surface(2, 2),
                make_surface(1, 1),
            ]],
            is_cubemap: false,
        };

        let transform = MipmapTransform::new(None, MipmapFilter::CatmullRom);
        let result = transform.execute(image).unwrap();

        // Already complete, should be unchanged.
        assert_eq!(result.surfaces[0].len(), 5);
    }

    #[test]
    fn trim_excess_mips() {
        let image = Image {
            surfaces: vec![vec![
                make_surface(16, 16),
                make_surface(8, 8),
                make_surface(4, 4),
                make_surface(2, 2),
                make_surface(1, 1),
            ]],
            is_cubemap: false,
        };

        let transform = MipmapTransform::new(Some(3), MipmapFilter::CatmullRom);
        let result = transform.execute(image).unwrap();

        assert_eq!(result.surfaces[0].len(), 3);
    }

    #[test]
    fn fill_missing_mips() {
        // Has base + first mip, needs the rest.
        let image = Image {
            surfaces: vec![vec![make_surface(16, 16), make_surface(8, 8)]],
            is_cubemap: false,
        };

        let transform = MipmapTransform::new(None, MipmapFilter::CatmullRom);
        let result = transform.execute(image).unwrap();

        assert_eq!(result.surfaces[0].len(), 5);
        // Cascade from 8x8: next should be 4x4
        assert_eq!(result.surfaces[0][2].width, 4);
        assert_eq!(result.surfaces[0][2].height, 4);
    }

    #[test]
    fn non_square_mips() {
        let image = Image {
            surfaces: vec![vec![make_surface(32, 8)]],
            is_cubemap: false,
        };

        let transform = MipmapTransform::new(None, MipmapFilter::CatmullRom);
        let result = transform.execute(image).unwrap();

        // max(32,8) = 32 → 6 levels
        assert_eq!(result.surfaces[0].len(), 6);
        assert_eq!(result.surfaces[0][1].width, 16);
        assert_eq!(result.surfaces[0][1].height, 4);
        assert_eq!(result.surfaces[0][2].width, 8);
        assert_eq!(result.surfaces[0][2].height, 2);
        assert_eq!(result.surfaces[0][3].width, 4);
        assert_eq!(result.surfaces[0][3].height, 1);
        assert_eq!(result.surfaces[0][4].width, 2);
        assert_eq!(result.surfaces[0][4].height, 1);
        assert_eq!(result.surfaces[0][5].width, 1);
        assert_eq!(result.surfaces[0][5].height, 1);
    }
}
