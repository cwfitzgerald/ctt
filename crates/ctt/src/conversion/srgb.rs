use std::any::TypeId;
use std::mem::size_of;
use std::sync::LazyLock;

use bytemuck::cast_slice;
use bytemuck::cast_slice_mut;

use crate::error::Result;
use crate::sample::{Sample, dispatch_sample3};
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::FormatExt;

/// sRGB EOTF lookup table — maps every u8 value (0–255) to its linear f32 equivalent.
static EOTF_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut table = [0.0f32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *entry = srgb_eotf(c);
    }
    table
});

/// sRGB OETF lookup table — 4097 entries (inclusive endpoints) for linear interpolation
/// over [0, 1].
const OETF_LUT_SIZE: usize = 4096;

static OETF_LUT: LazyLock<[f32; OETF_LUT_SIZE + 1]> = LazyLock::new(|| {
    let mut table = [0.0f32; OETF_LUT_SIZE + 1];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = i as f32 / OETF_LUT_SIZE as f32;
        *entry = srgb_oetf_precise(c);
    }
    table
});

/// Apply the sRGB EOTF (electro-optical transfer function) to convert a single channel
/// from sRGB-encoded to linear.
fn srgb_eotf(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Precise sRGB OETF (inverse EOTF) to convert a single channel from linear to
/// sRGB-encoded.
fn srgb_oetf_precise(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Fast sRGB OETF using LUT with linear interpolation.
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

/// Convert a surface from sRGB to linear.
///
/// Reads from any format and writes to the given target format.
/// RGB channels get the sRGB EOTF applied; alpha (if present) is treated as already linear.
pub(crate) fn srgb_to_linear(
    surface: &Surface,
    target: ktx2::Format,
    has_alpha: bool,
) -> Result<Surface> {
    dispatch_sample3!(
        surface.format,
        target,
        srgb_to_linear_inner(surface, target, has_alpha)
    )
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
    dispatch_sample3!(
        surface.format,
        target,
        linear_to_srgb_inner(surface, target, has_alpha)
    )
}

fn srgb_to_linear_inner<S: Sample, D: Sample, const N: usize>(
    surface: &Surface,
    target: ktx2::Format,
    has_alpha: bool,
) -> Result<Surface> {
    let width = surface.width as usize;
    let height = surface.height as usize;
    let src_stride = surface.stride as usize;
    let src_row_bytes = width * size_of::<[S; N]>();
    let dst_stride = width * size_of::<[D; N]>();

    let use_lut = TypeId::of::<S>() == TypeId::of::<u8>();

    let mut out = vec![0u8; dst_stride * height];

    for y in 0..height {
        let src_row: &[[S; N]] = cast_slice(&surface.data[y * src_stride..][..src_row_bytes]);
        let dst_row: &mut [[D; N]] = cast_slice_mut(&mut out[y * dst_stride..][..dst_stride]);

        for (src_pixel, dst_pixel) in src_row.iter().zip(dst_row.iter_mut()) {
            for (channel, (s, d)) in src_pixel.iter().zip(dst_pixel.iter_mut()).enumerate() {
                let val = if has_alpha && channel == 3 {
                    s.to_f32()
                } else if use_lut {
                    EOTF_LUT[bytemuck::bytes_of(s)[0] as usize]
                } else {
                    srgb_eotf(s.to_f32())
                };
                *d = D::from_f32(val);
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

fn linear_to_srgb_inner<S: Sample, D: Sample, const N: usize>(
    surface: &Surface,
    target: ktx2::Format,
    has_alpha: bool,
) -> Result<Surface> {
    let width = surface.width as usize;
    let height = surface.height as usize;
    let src_stride = surface.stride as usize;
    let src_row_bytes = width * size_of::<[S; N]>();
    let dst_stride = width * size_of::<[D; N]>();

    let mut out = vec![0u8; dst_stride * height];

    for y in 0..height {
        let src_row: &[[S; N]] = cast_slice(&surface.data[y * src_stride..][..src_row_bytes]);
        let dst_row: &mut [[D; N]] = cast_slice_mut(&mut out[y * dst_stride..][..dst_stride]);

        for (src_pixel, dst_pixel) in src_row.iter().zip(dst_row.iter_mut()) {
            for (channel, (s, d)) in src_pixel.iter().zip(dst_pixel.iter_mut()).enumerate() {
                let val = if has_alpha && channel == 3 {
                    s.to_f32()
                } else {
                    srgb_oetf_fast(s.to_f32())
                };
                *d = D::from_f32(val);
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
    fn eotf_lut_matches_powf() {
        for i in 0..256 {
            let c = i as f32 / 255.0;
            let expected = srgb_eotf(c);
            assert!(
                (EOTF_LUT[i] - expected).abs() < f32::EPSILON,
                "EOTF LUT mismatch at {i}: lut={} expected={}",
                EOTF_LUT[i],
                expected,
            );
        }
    }

    #[test]
    fn oetf_lut_roundtrip_u8_precision() {
        for i in 0..=255u8 {
            let linear = EOTF_LUT[i as usize];
            let encoded = srgb_oetf_fast(linear);
            let back = (encoded.clamp(0.0, 1.0) * 255.0).round() as u8;
            assert_eq!(
                back, i,
                "OETF LUT roundtrip failed at {i}: linear={linear}, encoded={encoded}, back={back}",
            );
        }
    }

    #[test]
    fn oetf_lut_max_error_vs_precise() {
        let mut max_err: f32 = 0.0;
        let steps = 100_000;
        for i in 0..=steps {
            let c = i as f32 / steps as f32;
            let fast = srgb_oetf_fast(c);
            let precise = srgb_oetf_precise(c);
            let err = (fast - precise).abs();
            if err > max_err {
                max_err = err;
            }
        }
        let half_lsb_8bit = 0.5 / 255.0;
        assert!(
            max_err < half_lsb_8bit,
            "OETF LUT max error {max_err:.2e} exceeds 0.5 LSB at 8-bit ({half_lsb_8bit:.2e})",
        );
    }

    #[test]
    fn srgb_roundtrip_surface() {
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

        let alpha_bytes = &linear.data[12..16];
        let alpha = f32::from_le_bytes(alpha_bytes.try_into().unwrap());
        assert!((alpha - 200.0 / 255.0).abs() < 1e-5);

        let back = linear_to_srgb(&linear, ktx2::Format::R8G8B8A8_UNORM, true).unwrap();
        assert_eq!(back.color_space, ColorSpace::Srgb);
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
        let r = half::f16::from_f32(128.0 / 255.0);
        let g = half::f16::from_f32(64.0 / 255.0);
        let b = half::f16::from_f32(32.0 / 255.0);
        let a = half::f16::from_f32(200.0 / 255.0);

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

        for i in 0..4 {
            let orig = half::f16::from_le_bytes([surface.data[i * 2], surface.data[i * 2 + 1]]);
            let result = half::f16::from_le_bytes([back.data[i * 2], back.data[i * 2 + 1]]);
            assert!(
                (orig.to_f32() - result.to_f32()).abs() < 1e-3,
                "channel {i}: {} vs {}",
                orig,
                result,
            );
        }
    }

    #[test]
    fn srgb_roundtrip_all_u8_values() {
        for val in 0..=255u8 {
            let surface = Surface {
                data: vec![val, val, val, 255],
                width: 1,
                height: 1,
                stride: 4,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Srgb,
                alpha: AlphaMode::Straight,
            };

            let linear = srgb_to_linear(&surface, ktx2::Format::R32G32B32A32_SFLOAT, true).unwrap();
            let back = linear_to_srgb(&linear, ktx2::Format::R8G8B8A8_UNORM, true).unwrap();

            for channel in 0..3 {
                assert_eq!(
                    back.data[channel], val,
                    "u8 value {val} did not roundtrip on channel {channel}: got {}",
                    back.data[channel],
                );
            }
        }
    }
}
