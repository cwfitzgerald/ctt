mod default;
pub mod graph;
mod premultiplication;
mod srgb;

pub use default::build_default_graph;
pub use graph::{ConversionGraph, FormatState, SurfaceConverter, check_lossless};

use crate::error::Result;
use crate::surface::Surface;
use crate::vk_format::{ChannelKind, FormatExt};

/// Convert a surface to a different uncompressed format.
///
/// Supports channel extraction (RGBA->R, RGBA->RG), channel expansion
/// (R->RGBA, RG->RGBA, RGB->RGBA), and bit-depth conversion between
/// U8, U16, F16, and F32.
pub fn convert_surface(surface: &Surface, target: ktx2::Format) -> Result<Surface> {
    if surface.format == target {
        return Ok(surface.clone());
    }

    let src_cc = surface
        .format
        .channel_count()
        .expect("unknown src channel count");
    let src_ck = surface
        .format
        .channel_kind()
        .expect("unknown src channel kind");
    let src_cs = src_ck.byte_size();
    let src_bpp = src_cc * src_cs;

    let dst_cc = target.channel_count().expect("unknown dst channel count");
    let dst_ck = target.channel_kind().expect("unknown dst channel kind");
    let dst_cs = dst_ck.byte_size();
    let dst_bpp = dst_cc * dst_cs;

    let width = surface.width as usize;
    let height = surface.height as usize;
    let src_stride = surface.stride as usize;
    let dst_stride = width * dst_bpp;

    let mut out = vec![0u8; dst_stride * height];

    for y in 0..height {
        for x in 0..width {
            let src_off = y * src_stride + x * src_bpp;
            let dst_off = y * dst_stride + x * dst_bpp;

            for dst_ch in 0..dst_cc {
                let val = if dst_ch < src_cc {
                    let ch_off = src_off + dst_ch * src_cs;
                    read_channel(&surface.data, ch_off, src_ck)
                } else {
                    // Expansion: alpha defaults to max, others to 0.
                    if dst_ch == 3 { 1.0 } else { 0.0 }
                };

                let ch_off = dst_off + dst_ch * dst_cs;
                write_channel(&mut out, ch_off, dst_ck, val);
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: dst_stride as u32,
        format: target,
        color_space: surface.color_space,
        alpha: surface.alpha,
    })
}

/// Read a single channel value as f64, normalized to [0, 1] for integer types.
fn read_channel(data: &[u8], offset: usize, ck: ChannelKind) -> f64 {
    match ck {
        ChannelKind::U8 => data[offset] as f64 / 255.0,
        ChannelKind::U16 => {
            let v = u16::from_le_bytes([data[offset], data[offset + 1]]);
            v as f64 / 65535.0
        }
        ChannelKind::F16 => {
            let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
            half::f16::from_bits(bits).to_f64()
        }
        ChannelKind::F32 => {
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            f32::from_le_bytes(bytes) as f64
        }
        ChannelKind::U32 => {
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            u32::from_le_bytes(bytes) as f64 / u32::MAX as f64
        }
    }
}

/// Write a single channel value (f64, normalized [0,1] for integer types).
fn write_channel(data: &mut [u8], offset: usize, ck: ChannelKind, val: f64) {
    match ck {
        ChannelKind::U8 => {
            data[offset] = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        ChannelKind::U16 => {
            let v = (val.clamp(0.0, 1.0) * 65535.0).round() as u16;
            data[offset..offset + 2].copy_from_slice(&v.to_le_bytes());
        }
        ChannelKind::F16 => {
            let h = half::f16::from_f64(val);
            data[offset..offset + 2].copy_from_slice(&h.to_le_bytes());
        }
        ChannelKind::F32 => {
            let v = val as f32;
            data[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        }
        ChannelKind::U32 => {
            let v = (val.clamp(0.0, 1.0) * u32::MAX as f64).round() as u32;
            data[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    #[test]
    fn convert_surface_no_op_when_same_format() {
        let surface = Surface {
            data: vec![10, 20, 30, 255],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let result = convert_surface(&surface, ktx2::Format::R8G8B8A8_UNORM).unwrap();
        assert_eq!(result.data, surface.data);
    }

    #[test]
    fn convert_surface_rgba8_to_r8() {
        let surface = Surface {
            data: vec![100, 150, 200, 255],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let result = convert_surface(&surface, ktx2::Format::R8_UNORM).unwrap();
        assert_eq!(result.data, vec![100]);
    }

    #[test]
    fn convert_surface_r8_to_rgba8() {
        let surface = Surface {
            data: vec![100],
            width: 1,
            height: 1,
            stride: 1,
            format: ktx2::Format::R8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let result = convert_surface(&surface, ktx2::Format::R8G8B8A8_UNORM).unwrap();
        // R=100, G=0, B=0, A=255
        assert_eq!(result.data, vec![100, 0, 0, 255]);
    }

    #[test]
    fn convert_surface_u8_to_u16_roundtrip() {
        let surface = Surface {
            data: vec![128, 0, 0, 255],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let u16_surface = convert_surface(&surface, ktx2::Format::R16G16B16A16_UNORM).unwrap();
        assert_eq!(u16_surface.data.len(), 8);

        let back = convert_surface(&u16_surface, ktx2::Format::R8G8B8A8_UNORM).unwrap();
        assert_eq!(back.data, surface.data);
    }
}
