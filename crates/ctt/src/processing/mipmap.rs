//! Mipmap generation on f32 pipeline buffers.
//!
//! We route through [`image::Rgba32FImage`] so we can use `image::imageops`
//! for filtered downsampling. This restricts mipmap to the f32 pipeline for
//! now; widening to f64 or uint can slot in later.

use image::{Rgba32FImage, imageops};

use crate::error::{Error, Result};

use super::buffer::Buffer;

/// Filter types for mipmap downsampling. Mirrors [`imageops::FilterType`].
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
    fn to_image_filter(self) -> imageops::FilterType {
        match self {
            Self::Nearest => imageops::FilterType::Nearest,
            Self::Triangle => imageops::FilterType::Triangle,
            Self::CatmullRom => imageops::FilterType::CatmullRom,
            Self::Gaussian => imageops::FilterType::Gaussian,
            Self::Lanczos3 => imageops::FilterType::Lanczos3,
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

    if target == 1 {
        return Ok(vec![base]);
    }

    let imf = filter.to_image_filter();
    let mut prev = buffer_to_image(&base);

    let mut out = Vec::with_capacity(target);
    out.push(base);

    while out.len() < target {
        profiling::scope!("mip_level");
        let new_w = (prev.width() / 2).max(1);
        let new_h = (prev.height() / 2).max(1);
        let resized = imageops::resize(&prev, new_w, new_h, imf);
        out.push(image_to_buffer(&resized));
        prev = resized;
    }

    Ok(out)
}

fn buffer_to_image(buf: &Buffer<f32>) -> Rgba32FImage {
    profiling::scope!("buffer_to_image");
    Rgba32FImage::from_vec(
        buf.width,
        buf.height,
        bytemuck::pod_collect_to_vec(&buf.pixels),
    )
    .unwrap()
}

fn image_to_buffer(img: &Rgba32FImage) -> Buffer<f32> {
    let raw = img.as_raw();
    let pixels = raw
        .chunks_exact(4)
        .map(|ch| [ch[0], ch[1], ch[2], ch[3]])
        .collect();
    Buffer {
        pixels,
        width: img.width(),
        height: img.height(),
    }
}
