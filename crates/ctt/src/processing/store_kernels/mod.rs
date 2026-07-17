//! Per-[`FormatKind`](crate::format_kind::FormatKind) encoders that write a
//! `Buffer<T>` back into a [`Surface`]'s byte vector.
//!
//! Input lanes are in *linear*, *premultiplied* space (for float pipelines);
//! unpremultiplication and sRGB re-encoding happen at the callers / here.
//!
//! The plain per-channel codecs live in this module; the formats with an
//! explicitly vectorized storer (sRGB8 and the packed 32-bit formats) live in
//! [`super::kernels`] and are re-exported below, so this module stays the
//! format-facing surface for [`super::store`].

pub use crate::processing::kernels::a2_10_10_10::{
    store_a2b10g10r10_sint_u32, store_a2b10g10r10_snorm_f32, store_a2b10g10r10_uint_u32,
    store_a2b10g10r10_unorm_f32, store_a2r10g10b10_sint_u32, store_a2r10g10b10_snorm_f32,
    store_a2r10g10b10_uint_u32, store_a2r10g10b10_unorm_f32,
};
pub use crate::processing::kernels::b10g11r11::store_b10g11r11_f32;
pub use crate::processing::kernels::e5b9g9r9::store_e5b9g9r9_f32;
pub use crate::processing::kernels::srgb::{
    srgb_oetf_in_place_f32, store_bgr8_srgb_f32, store_bgra8_srgb_f32, store_srgb8_f32,
};

use bytemuck::Pod;
use half::f16;

use super::buffer::Buffer;

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

pub fn store_bgr8_unorm_f32(buf: &Buffer<f32>) -> Vec<u8> {
    profiling::scope!("store_bgr8_unorm_f32");
    write_pixels(buf, 3, 1, |lanes, bytes| {
        let arr = <&mut [u8; 3]>::try_from(bytes).expect("3-byte pixel");
        arr[0] = (lanes[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        arr[1] = (lanes[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        arr[2] = (lanes[0].clamp(0.0, 1.0) * 255.0).round() as u8;
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

    // 4-channel stores map 1:1 onto the bulk converter and beat the scalar
    // loop. Sub-4-channel stores need a gather pass, which (per benchmarking)
    // costs more than it saves vs. plain `f16::from_f32` per lane — keep
    // those on the scalar path.
    if channels == 4 {
        return store_f16_f32_bulk_rgba(buf);
    }

    write_pixels(buf, channels, 2, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<2>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = f16::from_f32(lane).to_le_bytes();
        }
    })
}

fn store_f16_f32_bulk_rgba(buf: &Buffer<f32>) -> Vec<u8> {
    use half::slice::HalfFloatSliceExt;

    let mut out = vec![0u8; buf.pixels.len() * 8];
    let src: &[f32] = bytemuck::cast_slice(&buf.pixels);
    let dst: &mut [f16] = bytemuck::cast_slice_mut(&mut out);
    dst.convert_from_f32_slice(src);
    out
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
    write_pixels(buf, channels, 4, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = (lane as f32).to_le_bytes();
        }
    })
}

pub fn store_f64_f64(buf: &Buffer<f64>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_f64_f64");
    write_pixels(buf, channels, 8, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<8>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

// ---- u32 pipeline stores ----

pub fn store_u8_uint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u8_uint_u32");
    write_pixels(buf, channels, 1, |lanes, bytes| {
        for (&lane, byte) in lanes.iter().zip(bytes.iter_mut()) {
            *byte = lane.min(u8::MAX as u32) as u8;
        }
    })
}

pub fn store_i8_sint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i8_sint_u32");
    write_pixels(buf, channels, 1, |lanes, bytes| {
        for (&lane, byte) in lanes.iter().zip(bytes.iter_mut()) {
            let v = (lane as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            *byte = v as u8;
        }
    })
}

pub fn store_u16_uint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u16_uint_u32");
    write_pixels(buf, channels, 2, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<2>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            let v = lane.min(u16::MAX as u32) as u16;
            *chunk = v.to_le_bytes();
        }
    })
}

pub fn store_i16_sint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i16_sint_u32");
    write_pixels(buf, channels, 2, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<2>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            let v = (lane as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            *chunk = v.to_le_bytes();
        }
    })
}

pub fn store_u32_uint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_u32_uint_u32");
    write_pixels(buf, channels, 4, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

pub fn store_i32_sint_u32(buf: &Buffer<u32>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i32_sint_u32");
    write_pixels(buf, channels, 4, |lanes, bytes| {
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
    write_pixels(buf, channels, 8, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<8>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

pub fn store_i64_sint_u64(buf: &Buffer<u64>, channels: usize) -> Vec<u8> {
    profiling::scope!("store_i64_sint_u64");
    write_pixels(buf, channels, 8, |lanes, bytes| {
        let (chunks, _) = bytes.as_chunks_mut::<8>();
        for (&lane, chunk) in lanes.iter().zip(chunks) {
            *chunk = lane.to_le_bytes();
        }
    })
}

// ---- Helpers ----

/// Shared by the plain codecs above and by the scalar production paths in
/// [`super::kernels`].
pub(crate) fn write_pixels<T: Copy + Pod>(
    buf: &Buffer<T>,
    channels: usize,
    channel_bytes: usize,
    mut encode: impl FnMut(&[T; 4], &mut [u8]),
) -> Vec<u8> {
    let pixel_bytes = channels * channel_bytes;
    let mut out = vec![0u8; buf.pixels.len() * pixel_bytes];
    for (pixel, bytes) in buf.pixels.iter().zip(out.chunks_exact_mut(pixel_bytes)) {
        encode(pixel, bytes);
    }
    out
}
