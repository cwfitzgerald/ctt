//! Mipmap generation on f32 pipeline buffers.
//!
//! Backed by `fast_image_resize` for SIMD-accelerated filtered downsampling
//! of RGBA32F data. Restricted to the f32 pipeline for now; widening to f64
//! or uint can slot in later.

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::pixels::PixelType;
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};

use crate::error::{Error, Result};

use super::buffer::Buffer;

/// Filter types for mipmap downsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MipmapFilter {
    Nearest,
    #[default]
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

impl MipmapFilter {
    fn to_resize_alg(self) -> ResizeAlg {
        match self {
            Self::Nearest => ResizeAlg::Nearest,
            Self::Triangle => ResizeAlg::Convolution(FilterType::Bilinear),
            Self::CatmullRom => ResizeAlg::Convolution(FilterType::CatmullRom),
            Self::Gaussian => ResizeAlg::Convolution(FilterType::Gaussian),
            Self::Lanczos3 => ResizeAlg::Convolution(FilterType::Lanczos3),
        }
    }
}

/// Compute the full mip chain length for the given dimensions (including the base level).
pub fn full_mip_count(width: u32, height: u32) -> usize {
    (width.max(height).max(1) as f64).log2().floor() as usize + 1
}

/// Generate a mip chain starting from `base` (which becomes mip 0).
///
/// Produces `count` levels; `count` must be at least 1.
pub fn generate(
    base: Buffer<f32>,
    filter: MipmapFilter,
    count: Option<usize>,
) -> Result<Vec<Buffer<f32>>> {
    profiling::scope!("mipmap::generate");
    let target = count.unwrap_or_else(|| full_mip_count(base.width, base.height));
    if target == 0 {
        return Err(Error::UnsupportedFormat("mipmap count must be >= 1".into()));
    }

    let options = ResizeOptions::new()
        .resize_alg(filter.to_resize_alg())
        .use_alpha(false);
    let mut resizer = Resizer::new();
    let mut out = Vec::with_capacity(target);
    out.push(base);

    while out.len() < target {
        profiling::scope!("mip_level");
        let next = {
            let prev = out.last().unwrap();
            let new_w = (prev.width / 2).max(1);
            let new_h = (prev.height / 2).max(1);
            let src = ImageRef::new(
                prev.width,
                prev.height,
                bytemuck::cast_slice(&prev.pixels),
                PixelType::F32x4,
            )
            .expect("buffer dimensions match pixel count");
            let mut dst_pixels = vec![[0.0f32; 4]; (new_w * new_h) as usize];
            {
                let mut dst = Image::from_slice_u8(
                    new_w,
                    new_h,
                    bytemuck::cast_slice_mut(&mut dst_pixels),
                    PixelType::F32x4,
                )
                .expect("destination dimensions match pixel count");
                resizer
                    .resize(&src, &mut dst, &options)
                    .map_err(|e| Error::UnsupportedFormat(format!("resize failed: {e}")))?;
            }
            Buffer {
                pixels: dst_pixels,
                width: new_w,
                height: new_h,
            }
        };
        out.push(next);
    }

    Ok(out)
}
