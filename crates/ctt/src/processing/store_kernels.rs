//! Per-[`FormatKind`](crate::format_kind::FormatKind) encoders that write a
//! `Buffer<T>` back into a [`Surface`]'s byte vector.
//!
//! Input lanes are in *linear*, *premultiplied* space (for float pipelines);
//! unpremultiplication and sRGB re-encoding happen at the callers / here.

use std::sync::LazyLock;

use half::f16;

use super::buffer::Buffer;

const OETF_LUT_SIZE: usize = 4096;

/// sRGB OETF lookup table — 4097 entries over [0, 1] for linear interpolation.
static OETF_LUT: LazyLock<[f32; OETF_LUT_SIZE + 1]> = LazyLock::new(|| {
    let mut table = [0.0f32; OETF_LUT_SIZE + 1];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = i as f32 / OETF_LUT_SIZE as f32;
        *entry = srgb_oetf_precise(c);
    }
    table
});

fn srgb_oetf_precise(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[inline(always)]
fn srgb_oetf_fast(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    let scaled = c * OETF_LUT_SIZE as f32;
    let idx = scaled as usize;
    if idx >= OETF_LUT_SIZE {
        return OETF_LUT[OETF_LUT_SIZE];
    }
    let frac = scaled - idx as f32;
    OETF_LUT[idx] + frac * (OETF_LUT[idx + 1] - OETF_LUT[idx])
}

// ---- f32 pipeline stores ----

pub fn store_u8_unorm_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u8_unorm_f32");
    write_pixels(buf, channels, 1, |lanes, bytes| {
        for (&lane, byte) in lanes.iter().zip(bytes.iter_mut()) {
            *byte = (lane.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    })
}

pub fn store_i8_snorm_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i8_snorm_f32");
    write_pixels(buf, channels, 1, |lanes, bytes| {
        for (&lane, byte) in lanes.iter().zip(bytes.iter_mut()) {
            let v = (lane.clamp(-1.0, 1.0) * 127.0).round() as i8;
            *byte = v as u8;
        }
    })
}

pub fn store_srgb8_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_srgb8_f32");
    write_pixels(buf, channels, 1, |lanes, bytes| {
        for (c, (&lane, byte)) in lanes.iter().zip(bytes.iter_mut()).enumerate() {
            let encoded = if c < 3 {
                srgb_oetf_fast(lane)
            } else {
                lane.clamp(0.0, 1.0)
            };
            *byte = (encoded * 255.0).round() as u8;
        }
    })
}

pub fn store_bgra8_unorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgra8_unorm_f32");
    write_pixels(buf, 4, 1, |lanes, bytes| {
        let arr = <&mut [u8; 4]>::try_from(bytes).expect("4-byte pixel");
        arr[0] = (lanes[2].clamp(0.0, 1.0) * 255.0).round() as u8; // B
        arr[1] = (lanes[1].clamp(0.0, 1.0) * 255.0).round() as u8; // G
        arr[2] = (lanes[0].clamp(0.0, 1.0) * 255.0).round() as u8; // R
        arr[3] = (lanes[3].clamp(0.0, 1.0) * 255.0).round() as u8; // A
    })
}

pub fn store_bgra8_srgb_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgra8_srgb_f32");
    write_pixels(buf, 4, 1, |lanes, bytes| {
        let arr = <&mut [u8; 4]>::try_from(bytes).expect("4-byte pixel");
        arr[0] = (srgb_oetf_fast(lanes[2]) * 255.0).round() as u8;
        arr[1] = (srgb_oetf_fast(lanes[1]) * 255.0).round() as u8;
        arr[2] = (srgb_oetf_fast(lanes[0]) * 255.0).round() as u8;
        arr[3] = (lanes[3].clamp(0.0, 1.0) * 255.0).round() as u8;
    })
}

pub fn store_bgr8_unorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgr8_unorm_f32");
    write_pixels(buf, 3, 1, |lanes, bytes| {
        let arr = <&mut [u8; 3]>::try_from(bytes).expect("3-byte pixel");
        arr[0] = (lanes[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        arr[1] = (lanes[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        arr[2] = (lanes[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    })
}

pub fn store_bgr8_srgb_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgr8_srgb_f32");
    write_pixels(buf, 3, 1, |lanes, bytes| {
        let arr = <&mut [u8; 3]>::try_from(bytes).expect("3-byte pixel");
        arr[0] = (srgb_oetf_fast(lanes[2]) * 255.0).round() as u8;
        arr[1] = (srgb_oetf_fast(lanes[1]) * 255.0).round() as u8;
        arr[2] = (srgb_oetf_fast(lanes[0]) * 255.0).round() as u8;
    })
}

pub fn store_u16_unorm_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u16_unorm_f32");
    write_pixels(buf, channels, 2, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<2>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            let v = (lane.clamp(0.0, 1.0) * 65535.0).round() as u16;
            *chunk = v.to_le_bytes();
        }
    })
}

pub fn store_i16_snorm_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i16_snorm_f32");
    write_pixels(buf, channels, 2, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<2>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            let v = (lane.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            *chunk = v.to_le_bytes();
        }
    })
}

pub fn store_f16_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_f16_f32");
    write_pixels(buf, channels, 2, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<2>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = f16::from_f32(lane).to_le_bytes();
        }
    })
}

pub fn store_f32_f32(buf: &Buffer<f32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_f32_f32");
    write_pixels(buf, channels, 4, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

// ---- f64 pipeline stores ----

pub fn store_f32_f64(buf: &Buffer<f64>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_f32_f64");
    write_pixels_f64(buf, channels, 4, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = (lane as f32).to_le_bytes();
        }
    })
}

pub fn store_f64_f64(buf: &Buffer<f64>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_f64_f64");
    write_pixels_f64(buf, channels, 8, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<8>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

// ---- u32 pipeline stores ----

pub fn store_u8_uint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u8_uint_u32");
    write_pixels_u32(buf, channels, 1, |lanes, bytes| {
        for (&lane, byte) in lanes.iter().zip(bytes.iter_mut()) {
            *byte = lane.min(u8::MAX as u32) as u8;
        }
    })
}

pub fn store_i8_sint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i8_sint_u32");
    write_pixels_u32(buf, channels, 1, |lanes, bytes| {
        for (&lane, byte) in lanes.iter().zip(bytes.iter_mut()) {
            let v = (lane as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            *byte = v as u8;
        }
    })
}

pub fn store_u16_uint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u16_uint_u32");
    write_pixels_u32(buf, channels, 2, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<2>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            let v = lane.min(u16::MAX as u32) as u16;
            *chunk = v.to_le_bytes();
        }
    })
}

pub fn store_i16_sint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i16_sint_u32");
    write_pixels_u32(buf, channels, 2, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<2>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            let v = (lane as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            *chunk = v.to_le_bytes();
        }
    })
}

pub fn store_u32_uint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u32_uint_u32");
    write_pixels_u32(buf, channels, 4, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

pub fn store_i32_sint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i32_sint_u32");
    write_pixels_u32(buf, channels, 4, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            // bit-cast u32 → i32 → le_bytes (same bytes either way).
            *chunk = lane.to_le_bytes();
        }
    })
}

// ---- u64 pipeline stores ----

pub fn store_u64_uint_u64(buf: &Buffer<u64>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u64_uint_u64");
    write_pixels_u64(buf, channels, 8, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<8>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

pub fn store_i64_sint_u64(buf: &Buffer<u64>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i64_sint_u64");
    write_pixels_u64(buf, channels, 8, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<8>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

// ---- Helpers ----

fn write_pixels(
    buf: &Buffer<f32>,
    channels: usize,
    channel_bytes: usize,
    mut encode: impl FnMut(&[f32; 4], &mut [u8]),
) -> Vec<u8> {
    let pixel_bytes = channels * channel_bytes;
    let mut out = vec![0u8; buf.pixels.len() * pixel_bytes];
    for (pixel, bytes) in buf.pixels.iter().zip(out.chunks_exact_mut(pixel_bytes)) {
        encode(pixel, bytes);
    }
    out
}

fn write_pixels_f64(
    buf: &Buffer<f64>,
    channels: usize,
    channel_bytes: usize,
    mut encode: impl FnMut(&[f64; 4], &mut [u8]),
) -> Vec<u8> {
    let pixel_bytes = channels * channel_bytes;
    let mut out = vec![0u8; buf.pixels.len() * pixel_bytes];
    for (pixel, bytes) in buf.pixels.iter().zip(out.chunks_exact_mut(pixel_bytes)) {
        encode(pixel, bytes);
    }
    out
}

fn write_pixels_u32(
    buf: &Buffer<u32>,
    channels: usize,
    channel_bytes: usize,
    mut encode: impl FnMut(&[u32; 4], &mut [u8]),
) -> Vec<u8> {
    let pixel_bytes = channels * channel_bytes;
    let mut out = vec![0u8; buf.pixels.len() * pixel_bytes];
    for (pixel, bytes) in buf.pixels.iter().zip(out.chunks_exact_mut(pixel_bytes)) {
        encode(pixel, bytes);
    }
    out
}

fn write_pixels_u64(
    buf: &Buffer<u64>,
    channels: usize,
    channel_bytes: usize,
    mut encode: impl FnMut(&[u64; 4], &mut [u8]),
) -> Vec<u8> {
    let pixel_bytes = channels * channel_bytes;
    let mut out = vec![0u8; buf.pixels.len() * pixel_bytes];
    for (pixel, bytes) in buf.pixels.iter().zip(out.chunks_exact_mut(pixel_bytes)) {
        encode(pixel, bytes);
    }
    out
}
