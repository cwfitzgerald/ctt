//! Per-[`FormatKind`](crate::format_kind::FormatKind) decoders that read a
//! [`Surface`] into a `Buffer<T>`.
//!
//! Loaders land in *linear*, *straight alpha* space — premultiplication is
//! handled separately in [`super::alpha`]. sRGB decoding is applied here
//! (RGB channels only; alpha rides through as linear).

pub(crate) mod srgb;
pub use srgb::{load_bgr8_srgb_f32, load_bgra8_srgb_f32, load_srgb8_f32};

use half::f16;

use crate::error::{Error, Result};
use crate::surface::Surface;

use super::buffer::Buffer;

/// Read `channels` bytes per pixel, producing one `[f32; 4]` with lane 3
/// defaulted to 1.0 (and intermediate lanes defaulted to 0.0).
pub fn load_u8_unorm_f32(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_u8_unorm_f32");
    read_pixels_f32(surface, channels, 1, |bytes, lanes| {
        for (lane, &byte) in lanes.iter_mut().zip(bytes) {
            *lane = byte as f32 / 255.0;
        }
    })
}

pub fn load_i8_snorm_f32(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_i8_snorm_f32");
    read_pixels_f32(surface, channels, 1, |bytes, lanes| {
        for (lane, &byte) in lanes.iter_mut().zip(bytes) {
            *lane = ((byte as i8) as f32 / 127.0).max(-1.0);
        }
    })
}

pub fn load_bgra8_unorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_bgra8_unorm_f32");
    read_pixels_f32(surface, 4, 1, |bytes, lanes| {
        let &[b, g, r, a] = <&[u8; 4]>::try_from(bytes).expect("4-byte pixel");
        lanes[0] = r as f32 / 255.0;
        lanes[1] = g as f32 / 255.0;
        lanes[2] = b as f32 / 255.0;
        lanes[3] = a as f32 / 255.0;
    })
}

pub fn load_bgr8_unorm_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_bgr8_unorm_f32");
    read_pixels_f32(surface, 3, 1, |bytes, lanes| {
        let &[b, g, r] = <&[u8; 3]>::try_from(bytes).expect("3-byte pixel");
        lanes[0] = r as f32 / 255.0;
        lanes[1] = g as f32 / 255.0;
        lanes[2] = b as f32 / 255.0;
    })
}

pub fn load_u16_unorm_f32(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_u16_unorm_f32");
    read_pixels_f32(surface, channels, 2, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<2>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = u16::from_le_bytes(chunk) as f32 / 65535.0;
        }
    })
}

pub fn load_i16_snorm_f32(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_i16_snorm_f32");
    read_pixels_f32(surface, channels, 2, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<2>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = (i16::from_le_bytes(chunk) as f32 / 32767.0).max(-1.0);
        }
    })
}

pub fn load_f16_f32(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_f16_f32");

    // On little-endian (every realistic target), the file's f16 bytes match
    // the native f16 in-memory representation, so we can cast and dispatch
    // through `half`'s bulk SIMD-accelerated converter. On big-endian we'd be
    // misinterpreting the bytes — fall back to the scalar `from_le_bytes`
    // path that the rest of the codebase uses.
    #[cfg(target_endian = "little")]
    {
        load_f16_f32_bulk(surface, channels)
    }

    #[cfg(target_endian = "big")]
    {
        read_pixels_f32(surface, channels, 2, |bytes, lanes| {
            let (chunks, _) = bytes.as_chunks::<2>();
            for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
                *lane = f16::from_bits(u16::from_le_bytes(chunk)).to_f32();
            }
        })
    }
}

#[cfg(target_endian = "little")]
fn load_f16_f32_bulk(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    use half::slice::HalfFloatSliceExt;

    let pixel_bytes = channels * 2;
    validate_surface(surface, pixel_bytes)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * pixel_bytes;

    // Pre-fill default lanes (alpha=1.0) for sub-4-channel inputs.
    let mut pixels = vec![[0.0f32, 0.0, 0.0, 1.0]; w * h];

    if channels == 4 {
        // Each pixel is 4×f16 mapping 1:1 onto 4×f32 — bulk-convert each row
        // straight into the destination lanes.
        for (row_idx, row_region) in surface.data.chunks(stride).take(h).enumerate() {
            let src: &[f16] = bytemuck::cast_slice(&row_region[..row_bytes]);
            let dst_pixels = &mut pixels[row_idx * w..(row_idx + 1) * w];
            let dst: &mut [f32] = bytemuck::cast_slice_mut(dst_pixels);
            src.convert_to_f32_slice(dst);
        }
    } else {
        // 1–3 channels: bulk-convert each row into a packed temp buffer, then
        // scatter into the leading lanes. The default alpha=1.0 stays put.
        let mut row_f32 = vec![0f32; w * channels];
        for (row_idx, row_region) in surface.data.chunks(stride).take(h).enumerate() {
            let src: &[f16] = bytemuck::cast_slice(&row_region[..row_bytes]);
            src.convert_to_f32_slice(&mut row_f32);
            let dst_pixels = &mut pixels[row_idx * w..(row_idx + 1) * w];
            for (pixel, chunk) in dst_pixels.iter_mut().zip(row_f32.chunks_exact(channels)) {
                pixel[..channels].copy_from_slice(chunk);
            }
        }
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

pub fn load_f32_f32(surface: &Surface, channels: usize) -> Result<Buffer<f32>> {
    profiling::scope!("load_f32_f32");
    read_pixels_f32(surface, channels, 4, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<4>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = f32::from_le_bytes(chunk);
        }
    })
}

// ---- f64 pipeline ----

pub fn load_f32_f64(surface: &Surface, channels: usize) -> Result<Buffer<f64>> {
    profiling::scope!("load_f32_f64");
    read_pixels_f64(surface, channels, 4, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<4>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = f32::from_le_bytes(chunk) as f64;
        }
    })
}

pub fn load_f64_f64(surface: &Surface, channels: usize) -> Result<Buffer<f64>> {
    profiling::scope!("load_f64_f64");
    read_pixels_f64(surface, channels, 8, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<8>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = f64::from_le_bytes(chunk);
        }
    })
}

// ---- Integer (u32) pipeline ----

/// Load 8-bit unsigned integers into u32 lanes. Alpha lane defaults to u32::MAX.
pub fn load_u8_uint_u32(surface: &Surface, channels: usize) -> Result<Buffer<u32>> {
    profiling::scope!("load_u8_uint_u32");
    read_pixels_u32(surface, channels, 1, |bytes, lanes| {
        for (lane, &byte) in lanes.iter_mut().zip(bytes) {
            *lane = byte as u32;
        }
    })
}

/// Load 8-bit signed integers (sign-extended) into u32 lanes via bit-cast.
pub fn load_i8_sint_u32(surface: &Surface, channels: usize) -> Result<Buffer<u32>> {
    profiling::scope!("load_i8_sint_u32");
    read_pixels_u32(surface, channels, 1, |bytes, lanes| {
        for (lane, &byte) in lanes.iter_mut().zip(bytes) {
            *lane = ((byte as i8) as i32) as u32;
        }
    })
}

pub fn load_u16_uint_u32(surface: &Surface, channels: usize) -> Result<Buffer<u32>> {
    profiling::scope!("load_u16_uint_u32");
    read_pixels_u32(surface, channels, 2, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<2>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = u16::from_le_bytes(chunk) as u32;
        }
    })
}

pub fn load_i16_sint_u32(surface: &Surface, channels: usize) -> Result<Buffer<u32>> {
    profiling::scope!("load_i16_sint_u32");
    read_pixels_u32(surface, channels, 2, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<2>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = (i16::from_le_bytes(chunk) as i32) as u32;
        }
    })
}

pub fn load_u32_uint_u32(surface: &Surface, channels: usize) -> Result<Buffer<u32>> {
    profiling::scope!("load_u32_uint_u32");
    read_pixels_u32(surface, channels, 4, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<4>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = u32::from_le_bytes(chunk);
        }
    })
}

pub fn load_i32_sint_u32(surface: &Surface, channels: usize) -> Result<Buffer<u32>> {
    profiling::scope!("load_i32_sint_u32");
    read_pixels_u32(surface, channels, 4, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<4>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = u32::from_le_bytes(chunk); // bit-cast of i32 → u32
        }
    })
}

// ---- Integer (u64) pipeline ----

pub fn load_u64_uint_u64(surface: &Surface, channels: usize) -> Result<Buffer<u64>> {
    profiling::scope!("load_u64_uint_u64");
    read_pixels_u64(surface, channels, 8, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<8>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = u64::from_le_bytes(chunk);
        }
    })
}

pub fn load_i64_sint_u64(surface: &Surface, channels: usize) -> Result<Buffer<u64>> {
    profiling::scope!("load_i64_sint_u64");
    read_pixels_u64(surface, channels, 8, |bytes, lanes| {
        let (chunks, _) = bytes.as_chunks::<8>();
        for (lane, &chunk) in lanes.iter_mut().zip(chunks) {
            *lane = u64::from_le_bytes(chunk); // bit-cast i64 → u64
        }
    })
}

// ---- Helpers ----

fn validate_surface(surface: &Surface, pixel_bytes: usize) -> Result<()> {
    let w = surface.width as usize;
    let h = surface.height as usize;
    let row_bytes = w * pixel_bytes;
    let stride = surface.stride as usize;
    if stride < row_bytes {
        return Err(Error::DataLengthMismatch {
            expected: row_bytes,
            actual: stride,
        });
    }
    let required = stride * h.saturating_sub(1) + row_bytes;
    if surface.data.len() < required {
        return Err(Error::DataLengthMismatch {
            expected: required,
            actual: surface.data.len(),
        });
    }
    Ok(())
}

fn read_pixels_f32(
    surface: &Surface,
    channels: usize,
    channel_bytes: usize,
    mut decode: impl FnMut(&[u8], &mut [f32; 4]),
) -> Result<Buffer<f32>> {
    let pixel_bytes = channels * channel_bytes;
    validate_surface(surface, pixel_bytes)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * pixel_bytes;

    let mut pixels = Vec::with_capacity(w * h);
    for row_region in surface.data.chunks(stride).take(h) {
        let row = &row_region[..row_bytes];
        pixels.extend(row.chunks_exact(pixel_bytes).map(|pixel| {
            let mut lanes = [0.0f32, 0.0, 0.0, 1.0];
            decode(pixel, &mut lanes);
            lanes
        }));
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

fn read_pixels_f64(
    surface: &Surface,
    channels: usize,
    channel_bytes: usize,
    mut decode: impl FnMut(&[u8], &mut [f64; 4]),
) -> Result<Buffer<f64>> {
    let pixel_bytes = channels * channel_bytes;
    validate_surface(surface, pixel_bytes)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * pixel_bytes;

    let mut pixels = Vec::with_capacity(w * h);
    for row_region in surface.data.chunks(stride).take(h) {
        let row = &row_region[..row_bytes];
        pixels.extend(row.chunks_exact(pixel_bytes).map(|pixel| {
            let mut lanes = [0.0f64, 0.0, 0.0, 1.0];
            decode(pixel, &mut lanes);
            lanes
        }));
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

fn read_pixels_u32(
    surface: &Surface,
    channels: usize,
    channel_bytes: usize,
    mut decode: impl FnMut(&[u8], &mut [u32; 4]),
) -> Result<Buffer<u32>> {
    let pixel_bytes = channels * channel_bytes;
    validate_surface(surface, pixel_bytes)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * pixel_bytes;

    let mut pixels = Vec::with_capacity(w * h);
    for row_region in surface.data.chunks(stride).take(h) {
        let row = &row_region[..row_bytes];
        pixels.extend(row.chunks_exact(pixel_bytes).map(|pixel| {
            let mut lanes = [0u32, 0, 0, u32::MAX];
            decode(pixel, &mut lanes);
            lanes
        }));
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

fn read_pixels_u64(
    surface: &Surface,
    channels: usize,
    channel_bytes: usize,
    mut decode: impl FnMut(&[u8], &mut [u64; 4]),
) -> Result<Buffer<u64>> {
    let pixel_bytes = channels * channel_bytes;
    validate_surface(surface, pixel_bytes)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * pixel_bytes;

    let mut pixels = Vec::with_capacity(w * h);
    for row_region in surface.data.chunks(stride).take(h) {
        let row = &row_region[..row_bytes];
        pixels.extend(row.chunks_exact(pixel_bytes).map(|pixel| {
            let mut lanes = [0u64, 0, 0, u64::MAX];
            decode(pixel, &mut lanes);
            lanes
        }));
    }

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}
