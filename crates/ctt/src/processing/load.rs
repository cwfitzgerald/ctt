//! Top-level load entry points for each pipeline variant.
//!
//! Each loader:
//! 1. Classifies the surface's format.
//! 2. Dispatches to the matching kernel in [`super::load_kernels`].
//! 3. Premultiplies alpha on float pipelines if the source is
//!    [`AlphaMode::Straight`].

use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::format_kind::{FormatFamily, FormatKind, classify};
use crate::surface::{ColorSpace, Surface};

use super::alpha;
use super::buffer::Buffer;
use super::load_kernels as k;

/// Load a surface into the f32 pipeline (linear + premultiplied).
pub fn load_f32(surface: &Surface) -> Result<Buffer<f32>> {
    profiling::scope!("load_f32");
    let info = classify(surface.format, surface.color_space).ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "float pipeline: unsupported input format {:?}",
            surface.format
        ))
    })?;

    if info.family.is_integer() {
        return Err(Error::UnsupportedConversion(format!(
            "cannot load integer format {:?} into float pipeline",
            surface.format,
        )));
    }

    let (mut buf, srgb_decoded_by_kernel) = match info.kind {
        FormatKind::U8 => (k::load_u8_unorm_f32(surface, info.channels)?, false),
        FormatKind::I8 => (k::load_i8_snorm_f32(surface, info.channels)?, false),
        FormatKind::Srgb8 => (k::load_srgb8_f32(surface, info.channels)?, true),
        FormatKind::Bgra8 => (k::load_bgra8_unorm_f32(surface)?, false),
        FormatKind::Bgra8Srgb => (k::load_bgra8_srgb_f32(surface)?, true),
        FormatKind::Bgr8 => (k::load_bgr8_unorm_f32(surface)?, false),
        FormatKind::Bgr8Srgb => (k::load_bgr8_srgb_f32(surface)?, true),
        FormatKind::U16 => (k::load_u16_unorm_f32(surface, info.channels)?, false),
        FormatKind::I16 => (k::load_i16_snorm_f32(surface, info.channels)?, false),
        FormatKind::F16 => (k::load_f16_f32(surface, info.channels)?, false),
        FormatKind::F32 => (k::load_f32_f32(surface, info.channels)?, false),
        FormatKind::E5b9g9r9Ufloat => (k::load_e5b9g9r9_f32(surface)?, false),
        FormatKind::B10g11r11Ufloat => (k::load_b10g11r11_f32(surface)?, false),
        FormatKind::A2b10g10r10Unorm => (k::load_a2b10g10r10_unorm_f32(surface)?, false),
        FormatKind::A2b10g10r10Snorm => (k::load_a2b10g10r10_snorm_f32(surface)?, false),
        FormatKind::A2r10g10b10Unorm => (k::load_a2r10g10b10_unorm_f32(surface)?, false),
        FormatKind::A2r10g10b10Snorm => (k::load_a2r10g10b10_snorm_f32(surface)?, false),
        other => {
            return Err(Error::UnsupportedFormat(format!(
                "float pipeline: unsupported format kind {other:?}"
            )));
        }
    };

    // Respect Surface::color_space for FormatKinds that don't encode sRGB in
    // the format itself (e.g. R16G16B16A16_UNORM + ColorSpace::Srgb). 8-bit
    // kinds are promoted to their sRGB variants by `classify` and decoded
    // through the LUT inside the kernel.
    if !srgb_decoded_by_kernel && surface.color_space == ColorSpace::Srgb {
        k::srgb_eotf_in_place_f32(&mut buf.pixels);
    }

    if surface.alpha == AlphaMode::Straight {
        alpha::premultiply_f32(&mut buf);
    }

    Ok(buf)
}

/// Load a surface into the f64 pipeline (float only, no sRGB / no integer).
pub fn load_f64(surface: &Surface) -> Result<Buffer<f64>> {
    profiling::scope!("load_f64");
    let info = classify(surface.format, surface.color_space).ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "f64 pipeline: unsupported input format {:?}",
            surface.format
        ))
    })?;

    if !matches!(info.family, FormatFamily::Float) {
        return Err(Error::UnsupportedConversion(format!(
            "f64 pipeline requires a float-family source, got {:?}",
            surface.format,
        )));
    }

    let mut buf = match info.kind {
        FormatKind::F32 => k::load_f32_f64(surface, info.channels)?,
        FormatKind::F64 => k::load_f64_f64(surface, info.channels)?,
        other => {
            return Err(Error::UnsupportedFormat(format!(
                "f64 pipeline: unsupported format kind {other:?}"
            )));
        }
    };

    if surface.alpha == AlphaMode::Straight {
        alpha::premultiply_f64(&mut buf);
    }

    Ok(buf)
}

/// Load a surface into the u32 pipeline (UINT/SINT only, sign-extended bit-cast).
pub fn load_u32(surface: &Surface) -> Result<Buffer<u32>> {
    profiling::scope!("load_u32");
    let info = classify(surface.format, surface.color_space).ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "u32 pipeline: unsupported input format {:?}",
            surface.format
        ))
    })?;

    if !info.family.is_integer() {
        return Err(Error::UnsupportedConversion(format!(
            "u32 pipeline requires an integer-family source, got {:?}",
            surface.format,
        )));
    }

    match info.kind {
        FormatKind::U8 => k::load_u8_uint_u32(surface, info.channels),
        FormatKind::I8 => k::load_i8_sint_u32(surface, info.channels),
        FormatKind::U16 => k::load_u16_uint_u32(surface, info.channels),
        FormatKind::I16 => k::load_i16_sint_u32(surface, info.channels),
        FormatKind::U32 => k::load_u32_uint_u32(surface, info.channels),
        FormatKind::I32 => k::load_i32_sint_u32(surface, info.channels),
        FormatKind::A2b10g10r10Uint => k::load_a2b10g10r10_uint_u32(surface),
        FormatKind::A2b10g10r10Sint => k::load_a2b10g10r10_sint_u32(surface),
        FormatKind::A2r10g10b10Uint => k::load_a2r10g10b10_uint_u32(surface),
        FormatKind::A2r10g10b10Sint => k::load_a2r10g10b10_sint_u32(surface),
        other => Err(Error::UnsupportedFormat(format!(
            "u32 pipeline: unsupported format kind {other:?}"
        ))),
    }
}

/// Load a surface into the u64 pipeline (R64_UINT / R64_SINT only).
pub fn load_u64(surface: &Surface) -> Result<Buffer<u64>> {
    profiling::scope!("load_u64");
    let info = classify(surface.format, surface.color_space).ok_or_else(|| {
        Error::UnsupportedFormat(format!(
            "u64 pipeline: unsupported input format {:?}",
            surface.format
        ))
    })?;

    if !info.family.is_integer() {
        return Err(Error::UnsupportedConversion(format!(
            "u64 pipeline requires an integer-family source, got {:?}",
            surface.format,
        )));
    }

    match info.kind {
        FormatKind::U64 => k::load_u64_uint_u64(surface, info.channels),
        FormatKind::I64 => k::load_i64_sint_u64(surface, info.channels),
        other => Err(Error::UnsupportedFormat(format!(
            "u64 pipeline: unsupported format kind {other:?}"
        ))),
    }
}
