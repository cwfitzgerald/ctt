use crate::error::Result;
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::FormatExt;

use super::{read_channel, write_channel};

/// Apply the sRGB EOTF (electro-optical transfer function) to convert a single channel from
/// sRGB-encoded to linear.
pub(crate) fn srgb_eotf(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Apply the inverse sRGB EOTF (OETF) to convert a single channel from linear to sRGB-encoded.
pub(crate) fn srgb_oetf(c: f64) -> f64 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert a surface from sRGB to linear.
///
/// Reads from any format and writes to the given target format.
/// RGB channels get the sRGB EOTF applied; alpha (if present) is treated as already linear.
pub(crate) fn srgb_to_linear(
    surface: &Surface,
    target: ktx2::Format,
    has_alpha: bool,
) -> Result<Surface> {
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

            for ch in 0..dst_cc {
                let val = if ch < src_cc {
                    let raw = read_channel(&surface.data, src_off + ch * src_cs, src_ck);
                    if has_alpha && ch == 3 {
                        raw // alpha is linear
                    } else {
                        srgb_eotf(raw)
                    }
                } else if ch == 3 {
                    1.0 // alpha default
                } else {
                    0.0
                };

                write_channel(&mut out, dst_off + ch * dst_cs, dst_ck, val);
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: dst_stride as u32,
        format: target,
        color_space: ColorSpace::Linear,
        alpha: surface.alpha,
    })
}

/// Convert a surface from linear to sRGB.
///
/// Reads from any format and writes to the given target format.
/// RGB channels get the inverse sRGB EOTF applied; alpha (if present) is treated as linear.
pub(crate) fn linear_to_srgb(
    surface: &Surface,
    target: ktx2::Format,
    has_alpha: bool,
) -> Result<Surface> {
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

            for ch in 0..dst_cc {
                let linear = if ch < src_cc {
                    read_channel(&surface.data, src_off + ch * src_cs, src_ck)
                } else if ch == 3 {
                    1.0
                } else {
                    0.0
                };

                let encoded = if has_alpha && ch == 3 {
                    linear // alpha stays linear
                } else {
                    srgb_oetf(linear)
                };

                write_channel(&mut out, dst_off + ch * dst_cs, dst_ck, encoded);
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: dst_stride as u32,
        format: target,
        color_space: ColorSpace::Srgb,
        alpha: surface.alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;

    #[test]
    fn srgb_roundtrip_surface() {
        // 1x1 pixel: sRGB(128, 64, 32, 200)
        let surface = Surface {
            data: vec![128, 64, 32, 200],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Straight,
        };

        let linear = srgb_to_linear(&surface, ktx2::Format::R32G32B32A32_SFLOAT, true).unwrap();
        assert_eq!(linear.color_space, ColorSpace::Linear);
        assert_eq!(linear.format, ktx2::Format::R32G32B32A32_SFLOAT);

        // Alpha should pass through linearly: 200/255
        let alpha_bytes = &linear.data[12..16];
        let alpha = f32::from_le_bytes(alpha_bytes.try_into().unwrap());
        assert!((alpha - 200.0 / 255.0).abs() < 1e-5);

        let back = linear_to_srgb(&linear, ktx2::Format::R8G8B8A8_UNORM, true).unwrap();
        assert_eq!(back.color_space, ColorSpace::Srgb);
        // Should round-trip within +-1 due to u8 quantization.
        for i in 0..4 {
            assert!(
                (back.data[i] as i16 - surface.data[i] as i16).unsigned_abs() <= 1,
                "channel {i}: {} vs {}",
                back.data[i],
                surface.data[i],
            );
        }
    }

    #[test]
    fn srgb_roundtrip_f16_surface() {
        // 1x1 pixel: sRGB values stored as F16
        let r = half::f16::from_f64(128.0 / 255.0);
        let g = half::f16::from_f64(64.0 / 255.0);
        let b = half::f16::from_f64(32.0 / 255.0);
        let a = half::f16::from_f64(200.0 / 255.0);

        let mut data = vec![0u8; 8];
        data[0..2].copy_from_slice(&r.to_le_bytes());
        data[2..4].copy_from_slice(&g.to_le_bytes());
        data[4..6].copy_from_slice(&b.to_le_bytes());
        data[6..8].copy_from_slice(&a.to_le_bytes());

        let surface = Surface {
            data,
            width: 1,
            height: 1,
            stride: 8,
            format: ktx2::Format::R16G16B16A16_SFLOAT,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Straight,
        };

        let linear = srgb_to_linear(&surface, ktx2::Format::R32G32B32A32_SFLOAT, true).unwrap();
        assert_eq!(linear.color_space, ColorSpace::Linear);
        assert_eq!(linear.format, ktx2::Format::R32G32B32A32_SFLOAT);

        let back = linear_to_srgb(&linear, ktx2::Format::R16G16B16A16_SFLOAT, true).unwrap();
        assert_eq!(back.color_space, ColorSpace::Srgb);
        assert_eq!(back.format, ktx2::Format::R16G16B16A16_SFLOAT);

        // Should round-trip within F16 precision.
        for i in 0..4 {
            let orig = half::f16::from_le_bytes([surface.data[i * 2], surface.data[i * 2 + 1]]);
            let result = half::f16::from_le_bytes([back.data[i * 2], back.data[i * 2 + 1]]);
            assert!(
                (orig.to_f64() - result.to_f64()).abs() < 1e-3,
                "channel {i}: {} vs {}",
                orig,
                result,
            );
        }
    }
}
