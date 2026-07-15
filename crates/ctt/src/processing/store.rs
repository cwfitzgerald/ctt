//! Top-level store entry points.
//!
//! Writes a pipeline `Buffer<T>` back to a [`Surface`] in the chosen target
//! format. For float pipelines, handles the linear → sRGB and premul →
//! straight transitions based on target settings.

use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::format_kind::{FormatFamily, FormatKind, classify};
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::FormatExt;

use super::alpha;
use super::buffer::Buffer;
use super::store_kernels as k;

/// Store a f32 buffer to a surface of `target_format`.
///
/// The buffer is assumed to be in linear + premultiplied form (as loaders
/// produce it). `target_alpha` drives unpremultiplication before write.
pub fn store_f32(
    mut buf: Buffer<f32>,
    target_format: ktx2::Format,
    target_color_space: ColorSpace,
    target_alpha: AlphaMode,
) -> Result<Surface> {
    profiling::scope!("store_f32");
    let info = classify(target_format, target_color_space).ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "float pipeline: unsupported target format {target_format:?}"
        ))
    })?;

    if info.family.is_integer() {
        return Err(Error::UnsupportedConversion(format!(
            "cannot store from float pipeline to integer format {target_format:?}"
        )));
    }

    if target_alpha == AlphaMode::Straight {
        alpha::unpremultiply_f32(&mut buf);
    }

    // Classify how the target FormatKind handles sRGB.
    let kind_is_srgb_native = matches!(
        info.kind,
        FormatKind::Srgb8 | FormatKind::Bgra8Srgb | FormatKind::Bgr8Srgb,
    );
    let kind_has_srgb_variant = matches!(
        info.kind,
        FormatKind::U8 | FormatKind::Bgra8 | FormatKind::Bgr8,
    );

    // For 16+ bit FormatKinds with target_color_space=Srgb we apply OETF in
    // a pre-pass and then write the linear kernel, since they have no sRGB
    // kernel variant.
    let need_oetf_pass =
        target_color_space == ColorSpace::Srgb && !kind_is_srgb_native && !kind_has_srgb_variant;

    if need_oetf_pass {
        k::srgb_oetf_in_place_f32(&mut buf.pixels);
    }

    let write_as_srgb =
        kind_is_srgb_native || (target_color_space == ColorSpace::Srgb && kind_has_srgb_variant);

    let data = match (info.kind, write_as_srgb) {
        (FormatKind::U8, true) => k::store_srgb8_f32(&buf, info.channels),
        (FormatKind::U8, false) => k::store_u8_unorm_f32(&buf, info.channels),
        (FormatKind::I8, _) => k::store_i8_snorm_f32(&buf, info.channels),
        (FormatKind::Srgb8, _) => k::store_srgb8_f32(&buf, info.channels),
        (FormatKind::Bgra8, true) => k::store_bgra8_srgb_f32(&buf),
        (FormatKind::Bgra8, false) => k::store_bgra8_unorm_f32(&buf),
        (FormatKind::Bgra8Srgb, _) => k::store_bgra8_srgb_f32(&buf),
        (FormatKind::Bgr8, true) => k::store_bgr8_srgb_f32(&buf),
        (FormatKind::Bgr8, false) => k::store_bgr8_unorm_f32(&buf),
        (FormatKind::Bgr8Srgb, _) => k::store_bgr8_srgb_f32(&buf),
        (FormatKind::U16, _) => k::store_u16_unorm_f32(&buf, info.channels),
        (FormatKind::I16, _) => k::store_i16_snorm_f32(&buf, info.channels),
        (FormatKind::F16, _) => k::store_f16_f32(&buf, info.channels),
        (FormatKind::F32, _) => k::store_f32_f32(&buf, info.channels),
        (FormatKind::E5b9g9r9Ufloat, _) => k::store_e5b9g9r9_f32(&buf),
        (FormatKind::B10g11r11Ufloat, _) => k::store_b10g11r11_f32(&buf),
        (FormatKind::A2b10g10r10Unorm, _) => k::store_a2b10g10r10_unorm_f32(&buf),
        (FormatKind::A2b10g10r10Snorm, _) => k::store_a2b10g10r10_snorm_f32(&buf),
        (FormatKind::A2r10g10b10Unorm, _) => k::store_a2r10g10b10_unorm_f32(&buf),
        (FormatKind::A2r10g10b10Snorm, _) => k::store_a2r10g10b10_snorm_f32(&buf),
        (other, _) => {
            return Err(Error::UnsupportedFormat(format!(
                "float pipeline: unsupported target kind {other:?}"
            )));
        }
    };

    let bpp = target_format.bytes_per_pixel().ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "cannot determine bytes_per_pixel for {target_format:?}"
        ))
    })?;
    let stride = buf.width * bpp as u32;

    Ok(Surface {
        data,
        width: buf.width,
        height: buf.height,
        depth: 1,
        stride,
        slice_stride: 0,
        format: target_format,
        color_space: target_color_space,
        alpha: target_alpha,
    })
}

/// Store a f64 buffer to a surface of `target_format` (float family only).
pub fn store_f64(
    mut buf: Buffer<f64>,
    target_format: ktx2::Format,
    target_color_space: ColorSpace,
    target_alpha: AlphaMode,
) -> Result<Surface> {
    profiling::scope!("store_f64");
    let info = classify(target_format, target_color_space).ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "f64 pipeline: unsupported target format {target_format:?}"
        ))
    })?;

    if !matches!(info.family, FormatFamily::Float) {
        return Err(Error::UnsupportedConversion(format!(
            "f64 pipeline can only store to float targets, got {target_format:?}"
        )));
    }

    if target_alpha == AlphaMode::Straight {
        alpha::unpremultiply_f64(&mut buf);
    }

    let data = match info.kind {
        FormatKind::F32 => k::store_f32_f64(&buf, info.channels),
        FormatKind::F64 => k::store_f64_f64(&buf, info.channels),
        other => {
            return Err(Error::UnsupportedFormat(format!(
                "f64 pipeline: unsupported target kind {other:?}"
            )));
        }
    };

    let bpp = target_format.bytes_per_pixel().ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "cannot determine bytes_per_pixel for {target_format:?}"
        ))
    })?;
    let stride = buf.width * bpp as u32;

    Ok(Surface {
        data,
        width: buf.width,
        height: buf.height,
        depth: 1,
        stride,
        slice_stride: 0,
        format: target_format,
        color_space: target_color_space,
        alpha: target_alpha,
    })
}

/// Store a u32 buffer to an integer-family target format.
pub fn store_u32(
    buf: Buffer<u32>,
    target_format: ktx2::Format,
    target_alpha: AlphaMode,
) -> Result<Surface> {
    profiling::scope!("store_u32");
    // Integer targets never carry sRGB — color_space is irrelevant here.
    let info = classify(target_format, ColorSpace::Linear).ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "u32 pipeline: unsupported target format {target_format:?}"
        ))
    })?;

    if !info.family.is_integer() {
        return Err(Error::UnsupportedConversion(format!(
            "u32 pipeline can only store to integer targets, got {target_format:?}"
        )));
    }

    let data = match info.kind {
        FormatKind::U8 => k::store_u8_uint_u32(&buf, info.channels),
        FormatKind::I8 => k::store_i8_sint_u32(&buf, info.channels),
        FormatKind::U16 => k::store_u16_uint_u32(&buf, info.channels),
        FormatKind::I16 => k::store_i16_sint_u32(&buf, info.channels),
        FormatKind::U32 => k::store_u32_uint_u32(&buf, info.channels),
        FormatKind::I32 => k::store_i32_sint_u32(&buf, info.channels),
        FormatKind::A2b10g10r10Uint => k::store_a2b10g10r10_uint_u32(&buf),
        FormatKind::A2b10g10r10Sint => k::store_a2b10g10r10_sint_u32(&buf),
        FormatKind::A2r10g10b10Uint => k::store_a2r10g10b10_uint_u32(&buf),
        FormatKind::A2r10g10b10Sint => k::store_a2r10g10b10_sint_u32(&buf),
        other => {
            return Err(Error::UnsupportedFormat(format!(
                "u32 pipeline: unsupported target kind {other:?}"
            )));
        }
    };

    let bpp = target_format.bytes_per_pixel().ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "cannot determine bytes_per_pixel for {target_format:?}"
        ))
    })?;
    let stride = buf.width * bpp as u32;

    Ok(Surface {
        data,
        width: buf.width,
        height: buf.height,
        depth: 1,
        stride,
        slice_stride: 0,
        format: target_format,
        color_space: ColorSpace::Linear,
        alpha: target_alpha,
    })
}

pub fn store_u64(
    buf: Buffer<u64>,
    target_format: ktx2::Format,
    target_alpha: AlphaMode,
) -> Result<Surface> {
    profiling::scope!("store_u64");
    let info = classify(target_format, ColorSpace::Linear).ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "u64 pipeline: unsupported target format {target_format:?}"
        ))
    })?;

    if !info.family.is_integer() {
        return Err(Error::UnsupportedConversion(format!(
            "u64 pipeline can only store to integer targets, got {target_format:?}"
        )));
    }

    let data = match info.kind {
        FormatKind::U64 => k::store_u64_uint_u64(&buf, info.channels),
        FormatKind::I64 => k::store_i64_sint_u64(&buf, info.channels),
        other => {
            return Err(Error::UnsupportedFormat(format!(
                "u64 pipeline: unsupported target kind {other:?}"
            )));
        }
    };

    let bpp = target_format.bytes_per_pixel().ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "cannot determine bytes_per_pixel for {target_format:?}"
        ))
    })?;
    let stride = buf.width * bpp as u32;

    Ok(Surface {
        data,
        width: buf.width,
        height: buf.height,
        depth: 1,
        stride,
        slice_stride: 0,
        format: target_format,
        color_space: ColorSpace::Linear,
        alpha: target_alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-bit targets with target_color_space=Srgb have no sRGB kernel
    /// variant, so they must go through the in-place OETF pre-pass before
    /// the linear u16 kernel.
    #[test]
    fn u16_srgb_target_runs_oetf_pre_pass() {
        fn srgb_oetf_exact(c: f32) -> f32 {
            if c < 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        }

        // 3 pixels also exercises a SIMD kernel tail behind the dispatch.
        let lanes = [0.5f32, 0.25, 0.001, 0.5];
        let buf = Buffer {
            pixels: vec![lanes; 3],
            width: 3,
            height: 1,
        };
        let surface = store_f32(
            buf,
            ktx2::Format::R16G16B16A16_UNORM,
            ColorSpace::Srgb,
            AlphaMode::Opaque,
        )
        .unwrap();

        let words: Vec<u16> = surface
            .data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(words.len(), 12);

        for pixel in words.chunks_exact(4) {
            for c in 0..3 {
                let want = (srgb_oetf_exact(lanes[c]) * 65535.0).round();
                let got = pixel[c] as f32;
                // Worst-case curve error is ~8e-4 ≈ 53 u16 steps.
                assert!(
                    (got - want).abs() <= 64.0,
                    "lane {c}: got={got} want={want}"
                );
            }
            // Alpha bypasses the OETF entirely.
            assert_eq!(pixel[3], 32768, "alpha must be stored as straight unorm");
        }
    }
}
