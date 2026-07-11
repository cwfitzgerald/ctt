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
    /// Nearest-neighbor: no filtering, fastest, lowest quality.
    Nearest,
    /// Triangle (bilinear) filter — the default.
    #[default]
    Triangle,
    /// Catmull-Rom cubic filter: sharper than triangle.
    CatmullRom,
    /// Gaussian filter: smooth, blurry downsampling.
    Gaussian,
    /// Lanczos filter with a support of 3: high quality, slowest.
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
    width.max(height).max(1).ilog2() as usize + 1
}

/// Generate a mip chain starting from `base` (which becomes mip 0).
///
/// Produces up to `count` levels; `count` must be at least 1. A requested
/// `count` larger than the true full chain length is clamped down to it, so
/// the result never contains duplicate 1×1 levels.
pub fn generate(
    base: Buffer<f32>,
    filter: MipmapFilter,
    count: Option<usize>,
) -> Result<Vec<Buffer<f32>>> {
    profiling::scope!("mipmap::generate");
    // Clamp any requested count to the true full chain length: an oversized
    // count (up to usize::MAX) must neither blow up `Vec::with_capacity` nor
    // push duplicate 1×1 levels past the real chain.
    let full = full_mip_count(base.width, base.height);
    let target = count.map_or(full, |c| c.min(full));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32) -> Buffer<f32> {
        Buffer {
            pixels: vec![[0.5, 0.5, 0.5, 1.0]; (width * height) as usize],
            width,
            height,
        }
    }

    #[test]
    fn full_mip_count_ilog2() {
        assert_eq!(full_mip_count(1, 1), 1);
        assert_eq!(full_mip_count(2, 2), 2);
        assert_eq!(full_mip_count(3, 3), 2);
        assert_eq!(full_mip_count(8, 8), 4);
        assert_eq!(full_mip_count(1024, 1024), 11);
    }

    #[test]
    fn generate_clamps_usize_max_count() {
        // usize::MAX must not blow up Vec::with_capacity; it clamps to the
        // true full chain (8×8 → 8,4,2,1 = 4 levels).
        let chain = generate(solid(8, 8), MipmapFilter::Triangle, Some(usize::MAX)).unwrap();
        assert_eq!(chain.len(), 4);
        assert_eq!((chain[3].width, chain[3].height), (1, 1));
    }

    #[test]
    fn generate_clamps_oversized_count_no_duplicate_1x1() {
        // 20 requested on 8×8 clamps to 4; no duplicate 1×1 levels tacked on.
        let chain = generate(solid(8, 8), MipmapFilter::Triangle, Some(20)).unwrap();
        assert_eq!(chain.len(), 4);
        let ones = chain
            .iter()
            .filter(|b| b.width == 1 && b.height == 1)
            .count();
        assert_eq!(ones, 1, "exactly one 1×1 level");
    }
}
