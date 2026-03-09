use crate::error::Result;
use crate::format::{ChannelType, PixelFormat};
use crate::image::RawImage;

/// Convert a raw image to the target pixel format.
///
/// If the image already matches `target`, returns a clone.
/// Supports channel extraction (RGBA→R, RGBA→RG), channel expansion
/// (R→RGBA, RG→RGBA, RGB→RGBA), and bit-depth conversion between
/// U8, U16, F16, and F32.
pub fn convert_image(image: &RawImage, target: PixelFormat) -> Result<RawImage> {
    if image.pixel_format == target {
        return Ok(image.clone());
    }

    let src_fmt = image.pixel_format;
    let width = image.width as usize;
    let height = image.height as usize;
    let src_stride = image.stride as usize;
    let src_cc = src_fmt.components.channel_count();
    let src_cs = src_fmt.channel_type.byte_size();
    let src_bpp = src_cc * src_cs;

    let dst_cc = target.components.channel_count();
    let dst_cs = target.channel_type.byte_size();
    let dst_bpp = dst_cc * dst_cs;
    let dst_stride = width * dst_bpp;

    let mut out = vec![0u8; dst_stride * height];

    for y in 0..height {
        for x in 0..width {
            let src_off = y * src_stride + x * src_bpp;
            let dst_off = y * dst_stride + x * dst_bpp;

            for dst_ch in 0..dst_cc {
                let val = if dst_ch < src_cc {
                    // Read source channel as f64 for precision during conversion.
                    let ch_off = src_off + dst_ch * src_cs;
                    read_channel(&image.data, ch_off, src_fmt.channel_type)
                } else {
                    // Expansion: fill missing channels.
                    // Alpha channel (index 3) defaults to max, others to 0.
                    if dst_ch == 3 { 1.0 } else { 0.0 }
                };

                let ch_off = dst_off + dst_ch * dst_cs;
                write_channel(&mut out, ch_off, target.channel_type, val);
            }
        }
    }

    Ok(RawImage {
        data: out,
        width: image.width,
        height: image.height,
        stride: dst_stride as u32,
        pixel_format: target,
    })
}

/// Read a single channel value as f64, normalized to [0, 1] for integer types.
fn read_channel(data: &[u8], offset: usize, ct: ChannelType) -> f64 {
    match ct {
        ChannelType::U8 => data[offset] as f64 / 255.0,
        ChannelType::U16 => {
            let v = u16::from_le_bytes([data[offset], data[offset + 1]]);
            v as f64 / 65535.0
        }
        ChannelType::F16 => {
            let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
            half::f16::from_bits(bits).to_f64()
        }
        ChannelType::F32 => {
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            f32::from_le_bytes(bytes) as f64
        }
    }
}

/// Write a single channel value (f64, normalized [0,1] for integer types).
fn write_channel(data: &mut [u8], offset: usize, ct: ChannelType, val: f64) {
    match ct {
        ChannelType::U8 => {
            data[offset] = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        ChannelType::U16 => {
            let v = (val.clamp(0.0, 1.0) * 65535.0).round() as u16;
            data[offset..offset + 2].copy_from_slice(&v.to_le_bytes());
        }
        ChannelType::F16 => {
            let h = half::f16::from_f64(val);
            data[offset..offset + 2].copy_from_slice(&h.to_le_bytes());
        }
        ChannelType::F32 => {
            let v = val as f32;
            data[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ColorSpace, PixelComponents};

    fn rgba8_format() -> PixelFormat {
        PixelFormat {
            components: PixelComponents::Rgba,
            channel_type: ChannelType::U8,
            color_space: ColorSpace::Linear,
        }
    }

    #[test]
    fn no_op_when_same_format() {
        let fmt = rgba8_format();
        let image = RawImage {
            data: vec![10, 20, 30, 255],
            width: 1,
            height: 1,
            stride: 4,
            pixel_format: fmt,
        };
        let result = convert_image(&image, fmt).unwrap();
        assert_eq!(result.data, image.data);
    }

    #[test]
    fn rgba8_to_r8() {
        let src = RawImage {
            data: vec![100, 150, 200, 255],
            width: 1,
            height: 1,
            stride: 4,
            pixel_format: rgba8_format(),
        };
        let target = PixelFormat {
            components: PixelComponents::R,
            channel_type: ChannelType::U8,
            color_space: ColorSpace::Linear,
        };
        let result = convert_image(&src, target).unwrap();
        assert_eq!(result.data, vec![100]);
    }

    #[test]
    fn rgba8_to_rg8() {
        let src = RawImage {
            data: vec![100, 150, 200, 255],
            width: 1,
            height: 1,
            stride: 4,
            pixel_format: rgba8_format(),
        };
        let target = PixelFormat {
            components: PixelComponents::Rg,
            channel_type: ChannelType::U8,
            color_space: ColorSpace::Linear,
        };
        let result = convert_image(&src, target).unwrap();
        assert_eq!(result.data, vec![100, 150]);
    }

    #[test]
    fn r8_to_rgba8() {
        let src = RawImage {
            data: vec![100],
            width: 1,
            height: 1,
            stride: 1,
            pixel_format: PixelFormat {
                components: PixelComponents::R,
                channel_type: ChannelType::U8,
                color_space: ColorSpace::Linear,
            },
        };
        let result = convert_image(&src, rgba8_format()).unwrap();
        // R=100, G=0, B=0, A=255
        assert_eq!(result.data, vec![100, 0, 0, 255]);
    }

    #[test]
    fn u8_to_u16_roundtrip() {
        let src = RawImage {
            data: vec![128, 0, 0, 255],
            width: 1,
            height: 1,
            stride: 4,
            pixel_format: rgba8_format(),
        };
        let u16_fmt = PixelFormat {
            components: PixelComponents::Rgba,
            channel_type: ChannelType::U16,
            color_space: ColorSpace::Linear,
        };
        let u16_img = convert_image(&src, u16_fmt).unwrap();
        assert_eq!(u16_img.data.len(), 8); // 4 channels * 2 bytes

        // Convert back to U8
        let back = convert_image(&u16_img, rgba8_format()).unwrap();
        assert_eq!(back.data, src.data);
    }

    #[test]
    fn f32_to_u16_hdr() {
        // Simulate an HDR pixel: R=0.5 in F32
        let r_bytes = 0.5f32.to_le_bytes();
        let zero = 0.0f32.to_le_bytes();
        let one = 1.0f32.to_le_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&r_bytes);
        data.extend_from_slice(&zero);
        data.extend_from_slice(&zero);
        data.extend_from_slice(&one);

        let src = RawImage {
            data,
            width: 1,
            height: 1,
            stride: 16,
            pixel_format: PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::F32,
                color_space: ColorSpace::Linear,
            },
        };
        let u16_fmt = PixelFormat {
            components: PixelComponents::Rgba,
            channel_type: ChannelType::U16,
            color_space: ColorSpace::Linear,
        };
        let result = convert_image(&src, u16_fmt).unwrap();
        // R should be ~32768 (0.5 * 65535 = 32767.5 → 32768)
        let r = u16::from_le_bytes([result.data[0], result.data[1]]);
        assert!((r as i32 - 32768).unsigned_abs() <= 1);
    }
}
