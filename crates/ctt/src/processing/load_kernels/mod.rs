//! Per-[`FormatKind`](crate::format_kind::FormatKind) decoders that read a
//! [`Surface`] into a `Buffer<T>`.
//!
//! Loaders land in *linear*, *straight alpha* space — premultiplication is
//! handled separately in [`super::alpha`]. sRGB decoding is applied here
//! (RGB channels only; alpha rides through as linear).

pub(crate) mod a2_10_10_10;
pub(crate) mod b10g11r11;
pub(crate) mod e5b9g9r9;
pub(crate) mod srgb;

pub use a2_10_10_10::{
    load_a2b10g10r10_sint_u32, load_a2b10g10r10_snorm_f32, load_a2b10g10r10_uint_u32,
    load_a2b10g10r10_unorm_f32, load_a2r10g10b10_sint_u32, load_a2r10g10b10_snorm_f32,
    load_a2r10g10b10_uint_u32, load_a2r10g10b10_unorm_f32,
};
pub use b10g11r11::load_b10g11r11_f32;
pub use e5b9g9r9::load_e5b9g9r9_f32;
pub use srgb::{load_bgr8_srgb_f32, load_bgra8_srgb_f32, load_srgb8_f32, srgb_eotf_in_place_f32};

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

    // Rows start at `row_idx * stride`; an odd stride yields odd byte offsets
    // that `bytemuck::cast_slice::<u8, f16>` (align 2) would reject with a
    // panic. `aligned_f16` casts in place when the row happens to be aligned
    // and otherwise decodes it into a reused scratch buffer.
    let mut scratch: Vec<f16> = Vec::new();
    if channels == 4 {
        // Each pixel is 4×f16 mapping 1:1 onto 4×f32 — bulk-convert each row
        // straight into the destination lanes.
        for (row_idx, row_region) in surface.data.chunks(stride).take(h).enumerate() {
            let src = aligned_f16(&row_region[..row_bytes], &mut scratch);
            let dst_pixels = &mut pixels[row_idx * w..(row_idx + 1) * w];
            let dst: &mut [f32] = bytemuck::cast_slice_mut(dst_pixels);
            src.convert_to_f32_slice(dst);
        }
    } else {
        // 1–3 channels: bulk-convert each row into a packed temp buffer, then
        // scatter into the leading lanes. The default alpha=1.0 stays put.
        let mut row_f32 = vec![0f32; w * channels];
        for (row_idx, row_region) in surface.data.chunks(stride).take(h).enumerate() {
            let src = aligned_f16(&row_region[..row_bytes], &mut scratch);
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

/// Reinterpret a row of bytes as `f16` lanes without ever misaligning.
///
/// Casts in place when `row` is already 2-byte aligned (the common case);
/// otherwise — e.g. an odd `stride` places a row at an odd byte offset —
/// decodes the row into `scratch` (reused across rows) via `from_le_bytes`
/// and returns that. `row.len()` is always a multiple of 2 here.
#[cfg(target_endian = "little")]
fn aligned_f16<'a>(row: &'a [u8], scratch: &'a mut Vec<f16>) -> &'a [f16] {
    match bytemuck::try_cast_slice::<u8, f16>(row) {
        Ok(src) => src,
        Err(_) => {
            scratch.resize(row.len() / 2, f16::from_bits(0));
            let scratch_bytes: &mut [u8] = bytemuck::cast_slice_mut(scratch.as_mut_slice());
            scratch_bytes.copy_from_slice(row);
            scratch.as_slice()
        }
    }
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

/// Drive a SIMD packed-32-bit loader: run `row_fn` over each row of a
/// 4-byte-per-pixel surface, decoding into 4 lanes per pixel. Because 4 input
/// bytes become 4 output lanes, byte offsets double as lane offsets throughout
/// the row helpers.
///
/// # Safety
/// `row_fn` must write `row.len()` lanes starting at the pointer it is given.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
unsafe fn load_packed_rows<T: Copy + bytemuck::Pod>(
    surface: &Surface,
    mut row_fn: impl FnMut(&[u8], *mut T),
) -> Result<Buffer<T>> {
    validate_surface(surface, 4)?;

    let w = surface.width as usize;
    let h = surface.height as usize;
    let stride = surface.stride as usize;
    let row_bytes = w * 4;
    let total_pixels = w * h;

    let mut pixels: Vec<[T; 4]> = Vec::with_capacity(total_pixels);
    let out_base = pixels.as_mut_ptr() as *mut T;

    let mut out_i = 0usize;
    for row_region in surface.data.chunks(stride).take(h) {
        let row = &row_region[..row_bytes];
        // SAFETY: `out_i` stays within the reserved capacity; validate_surface
        // bounded the input slice.
        row_fn(row, unsafe { out_base.add(out_i) });
        out_i += row_bytes;
    }
    debug_assert_eq!(out_i, total_pixels * 4);
    // SAFETY: `row_fn` initialized all `total_pixels * 4` lanes (caller contract).
    unsafe { pixels.set_len(total_pixels) };

    Ok(Buffer {
        pixels,
        width: surface.width,
        height: surface.height,
    })
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    /// R16G16B16A16_SFLOAT with an *odd* row stride must load without panicking
    /// — rows land at odd byte offsets that a plain `cast_slice::<u8, f16>`
    /// would reject on its 2-byte alignment requirement.
    #[test]
    fn load_f16_odd_stride_no_panic() {
        use half::f16;
        let width = 2u32;
        let height = 2u32;
        let pixel_bytes = 4 * 2; // RGBA f16
        let row_bytes = width as usize * pixel_bytes;
        // One extra byte of padding per row makes the stride odd.
        let stride = row_bytes + 1;

        let mut data = vec![0u8; stride * height as usize];
        let mut expected = Vec::new();
        for y in 0..height as usize {
            for x in 0..width as usize {
                for c in 0..4usize {
                    let v = (y * 8 + x * 4 + c) as f32 * 0.125 - 0.5;
                    expected.push(v);
                    let off = y * stride + (x * 4 + c) * 2;
                    data[off..off + 2].copy_from_slice(&f16::from_f32(v).to_le_bytes());
                }
            }
        }

        let surface = Surface {
            data,
            width,
            height,
            depth: 1,
            stride: stride as u32,
            slice_stride: 0,
            format: ktx2::Format::R16G16B16A16_SFLOAT,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        };

        let buf = load_f16_f32(&surface, 4).unwrap();
        assert_eq!(buf.width, width);
        assert_eq!(buf.height, height);
        // Values round-trip through f16 exactly (they were built from f16).
        let flat: Vec<f32> = buf.pixels.iter().flat_map(|p| p.iter().copied()).collect();
        for (got, want) in flat.iter().zip(&expected) {
            assert_eq!(got, want);
        }
    }
}
