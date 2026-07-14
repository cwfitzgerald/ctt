//! Format classification for the processing pipeline.
//!
//! Drives load/store kernel dispatch and variant (float/integer, precision) selection.
//!
//! ctt normalizes ktx2 formats at the boundary so that sRGB-ness lives on
//! [`Surface::color_space`](crate::surface::ColorSpace) rather than as a
//! format variant (`R8G8B8A8_SRGB` etc.). Classification therefore derives
//! the sRGB decision from the supplied `color_space`, not the format's own
//! sRGB-ness.

/// Byte-level layout of a format's channels.
///
/// Captures the per-lane storage width and any channel-order quirks (BGR/BGRA,
/// sRGB byte decoding). The [`FormatFamily`] reports the semantic interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatKind {
    /// 8-bit unsigned integer lanes, RGBA channel order.
    U8,
    /// 8-bit signed integer lanes, RGBA channel order.
    I8,
    /// 16-bit unsigned integer lanes, RGBA channel order.
    U16,
    /// 16-bit signed integer lanes, RGBA channel order.
    I16,
    /// 16-bit half-precision float lanes, RGBA channel order.
    F16,
    /// 32-bit unsigned integer lanes, RGBA channel order.
    U32,
    /// 32-bit signed integer lanes, RGBA channel order.
    I32,
    /// 32-bit single-precision float lanes, RGBA channel order.
    F32,
    /// 64-bit unsigned integer lanes, RGBA channel order.
    U64,
    /// 64-bit signed integer lanes, RGBA channel order.
    I64,
    /// 64-bit double-precision float lanes, RGBA channel order.
    F64,
    /// 8-bit sRGB-encoded lanes, RGBA channel order.
    Srgb8,
    /// 8-bit unsigned integer lanes, BGRA channel order.
    Bgra8,
    /// 8-bit sRGB-encoded lanes, BGRA channel order.
    Bgra8Srgb,
    /// 8-bit unsigned integer lanes, BGR channel order.
    Bgr8,
    /// 8-bit sRGB-encoded lanes, BGR channel order.
    Bgr8Srgb,

    // --- Packed 32-bit formats (single u32 word, bit-packed channels). ---
    /// Shared 5-bit exponent + 9-bit mantissas, `E5B9G9R9` layout (RGB, no alpha).
    E5b9g9r9Ufloat,
    /// Unsigned 11/11/10-bit floats, `B10G11R11` layout (RGB, no alpha).
    B10g11r11Ufloat,
    /// 2-bit alpha + three 10-bit unorm lanes, `A2B10G10R10` layout.
    A2b10g10r10Unorm,
    /// 2-bit alpha + three 10-bit snorm lanes, `A2B10G10R10` layout.
    A2b10g10r10Snorm,
    /// 2-bit alpha + three 10-bit uint lanes, `A2B10G10R10` layout.
    A2b10g10r10Uint,
    /// 2-bit alpha + three 10-bit sint lanes, `A2B10G10R10` layout.
    A2b10g10r10Sint,
    /// 2-bit alpha + three 10-bit unorm lanes, `A2R10G10B10` layout.
    A2r10g10b10Unorm,
    /// 2-bit alpha + three 10-bit snorm lanes, `A2R10G10B10` layout.
    A2r10g10b10Snorm,
    /// 2-bit alpha + three 10-bit uint lanes, `A2R10G10B10` layout.
    A2r10g10b10Uint,
    /// 2-bit alpha + three 10-bit sint lanes, `A2R10G10B10` layout.
    A2r10g10b10Sint,
}

/// Semantic interpretation of a format's values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatFamily {
    /// Unsigned normalized to `[0, 1]`.
    Unorm,
    /// Signed normalized to `[-1, 1]`.
    Snorm,
    /// Unsigned integer.
    Uint,
    /// Signed integer.
    Sint,
    /// Floating point.
    Float,
    /// sRGB-encoded unorm (gamma-compressed).
    Srgb,
}

impl FormatFamily {
    /// Is this an integer family (UINT/SINT) — requires the integer pipeline.
    pub fn is_integer(self) -> bool {
        matches!(self, FormatFamily::Uint | FormatFamily::Sint)
    }

    /// Is this a float-side family (UNORM/SNORM/FLOAT/SRGB) — routed through the float pipeline.
    pub fn is_float_side(self) -> bool {
        !self.is_integer()
    }
}

/// Complete classification of a format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FormatInfo {
    pub kind: FormatKind,
    pub family: FormatFamily,
    pub channels: usize,
}

/// Classify an uncompressed format, or return `None` if unsupported by the
/// new processing pipeline.
///
/// The format is normalized internally — `color_space` is authoritative for
/// the sRGB-vs-linear decision. Callers that only need the family/channel
/// structure (e.g. integer-vs-float routing) may pass
/// [`ColorSpace::Linear`](crate::surface::ColorSpace::Linear).
pub fn classify(fmt: ktx2::Format, color_space: crate::surface::ColorSpace) -> Option<FormatInfo> {
    use crate::surface::ColorSpace;
    use crate::vk_format::FormatExt;
    use FormatFamily::*;
    use FormatKind::*;
    use ktx2::Format as F;

    // sRGB-ness lives on the Surface, not the format; strip any sRGB variant
    // so the match below only deals with UNORM/SNORM/UINT/SINT/SFLOAT bases.
    let (base, _) = fmt.normalize();

    let (mut kind, mut family, channels) = match base {
        // --- R8 ---
        F::R8_UNORM => (U8, Unorm, 1),
        F::R8_SNORM => (I8, Snorm, 1),
        F::R8_UINT => (U8, Uint, 1),
        F::R8_SINT => (I8, Sint, 1),

        // --- R8G8 ---
        F::R8G8_UNORM => (U8, Unorm, 2),
        F::R8G8_SNORM => (I8, Snorm, 2),
        F::R8G8_UINT => (U8, Uint, 2),
        F::R8G8_SINT => (I8, Sint, 2),

        // --- R8G8B8 ---
        F::R8G8B8_UNORM => (U8, Unorm, 3),
        F::R8G8B8_SNORM => (I8, Snorm, 3),
        F::R8G8B8_UINT => (U8, Uint, 3),
        F::R8G8B8_SINT => (I8, Sint, 3),

        // --- B8G8R8 ---
        F::B8G8R8_UNORM => (Bgr8, Unorm, 3),

        // --- R8G8B8A8 ---
        F::R8G8B8A8_UNORM => (U8, Unorm, 4),
        F::R8G8B8A8_SNORM => (I8, Snorm, 4),
        F::R8G8B8A8_UINT => (U8, Uint, 4),
        F::R8G8B8A8_SINT => (I8, Sint, 4),

        // --- B8G8R8A8 ---
        F::B8G8R8A8_UNORM => (Bgra8, Unorm, 4),

        // --- R16 ---
        F::R16_UNORM => (U16, Unorm, 1),
        F::R16_SNORM => (I16, Snorm, 1),
        F::R16_UINT => (U16, Uint, 1),
        F::R16_SINT => (I16, Sint, 1),
        F::R16_SFLOAT => (F16, Float, 1),

        // --- R16G16 ---
        F::R16G16_UNORM => (U16, Unorm, 2),
        F::R16G16_SNORM => (I16, Snorm, 2),
        F::R16G16_UINT => (U16, Uint, 2),
        F::R16G16_SINT => (I16, Sint, 2),
        F::R16G16_SFLOAT => (F16, Float, 2),

        // --- R16G16B16 ---
        F::R16G16B16_UNORM => (U16, Unorm, 3),
        F::R16G16B16_SNORM => (I16, Snorm, 3),
        F::R16G16B16_UINT => (U16, Uint, 3),
        F::R16G16B16_SINT => (I16, Sint, 3),
        F::R16G16B16_SFLOAT => (F16, Float, 3),

        // --- R16G16B16A16 ---
        F::R16G16B16A16_UNORM => (U16, Unorm, 4),
        F::R16G16B16A16_SNORM => (I16, Snorm, 4),
        F::R16G16B16A16_UINT => (U16, Uint, 4),
        F::R16G16B16A16_SINT => (I16, Sint, 4),
        F::R16G16B16A16_SFLOAT => (F16, Float, 4),

        // --- R32 ---
        F::R32_UINT => (U32, Uint, 1),
        F::R32_SINT => (I32, Sint, 1),
        F::R32_SFLOAT => (F32, Float, 1),
        F::R32G32_UINT => (U32, Uint, 2),
        F::R32G32_SINT => (I32, Sint, 2),
        F::R32G32_SFLOAT => (F32, Float, 2),
        F::R32G32B32_UINT => (U32, Uint, 3),
        F::R32G32B32_SINT => (I32, Sint, 3),
        F::R32G32B32_SFLOAT => (F32, Float, 3),
        F::R32G32B32A32_UINT => (U32, Uint, 4),
        F::R32G32B32A32_SINT => (I32, Sint, 4),
        F::R32G32B32A32_SFLOAT => (F32, Float, 4),

        // --- R64 ---
        F::R64_UINT => (U64, Uint, 1),
        F::R64_SINT => (I64, Sint, 1),
        F::R64_SFLOAT => (F64, Float, 1),
        F::R64G64_UINT => (U64, Uint, 2),
        F::R64G64_SINT => (I64, Sint, 2),
        F::R64G64_SFLOAT => (F64, Float, 2),
        F::R64G64B64_UINT => (U64, Uint, 3),
        F::R64G64B64_SINT => (I64, Sint, 3),
        F::R64G64B64_SFLOAT => (F64, Float, 3),
        F::R64G64B64A64_UINT => (U64, Uint, 4),
        F::R64G64B64A64_SINT => (I64, Sint, 4),
        F::R64G64B64A64_SFLOAT => (F64, Float, 4),

        // --- Packed 32-bit ---
        F::E5B9G9R9_UFLOAT_PACK32 => (E5b9g9r9Ufloat, Float, 3),
        F::B10G11R11_UFLOAT_PACK32 => (B10g11r11Ufloat, Float, 3),

        F::A2B10G10R10_UNORM_PACK32 => (A2b10g10r10Unorm, Unorm, 4),
        F::A2B10G10R10_SNORM_PACK32 => (A2b10g10r10Snorm, Snorm, 4),
        F::A2B10G10R10_UINT_PACK32 => (A2b10g10r10Uint, Uint, 4),
        F::A2B10G10R10_SINT_PACK32 => (A2b10g10r10Sint, Sint, 4),

        F::A2R10G10B10_UNORM_PACK32 => (A2r10g10b10Unorm, Unorm, 4),
        F::A2R10G10B10_SNORM_PACK32 => (A2r10g10b10Snorm, Snorm, 4),
        F::A2R10G10B10_UINT_PACK32 => (A2r10g10b10Uint, Uint, 4),
        F::A2R10G10B10_SINT_PACK32 => (A2r10g10b10Sint, Sint, 4),

        _ => return None,
    };

    // Promote 8-bit UNORM kinds to their sRGB-native counterparts when the
    // surface's color_space says the data is sRGB-encoded. Higher-bit formats
    // have no sRGB-native kernel — callers apply a scalar EOTF/OETF post-pass.
    if color_space == ColorSpace::Srgb {
        match kind {
            U8 => {
                kind = Srgb8;
                family = Srgb;
            }
            Bgra8 => {
                kind = Bgra8Srgb;
                family = Srgb;
            }
            Bgr8 => {
                kind = Bgr8Srgb;
                family = Srgb;
            }
            _ => {}
        }
    }

    Some(FormatInfo {
        kind,
        family,
        channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::ColorSpace;
    use ktx2::Format as F;

    #[test]
    fn classify_rgba8_unorm() {
        let info = classify(F::R8G8B8A8_UNORM, ColorSpace::Linear).unwrap();
        assert_eq!(info.kind, FormatKind::U8);
        assert_eq!(info.family, FormatFamily::Unorm);
        assert_eq!(info.channels, 4);
    }

    #[test]
    fn classify_r8_uint_vs_unorm() {
        let uint = classify(F::R8_UINT, ColorSpace::Linear).unwrap();
        let unorm = classify(F::R8_UNORM, ColorSpace::Linear).unwrap();
        assert_eq!(uint.kind, unorm.kind); // same layout
        assert_ne!(uint.family, unorm.family); // different semantics
    }

    #[test]
    fn classify_color_space_drives_srgb() {
        // sRGB-ness comes from color_space; the format is normalized first.
        let linear = classify(F::R8G8B8A8_UNORM, ColorSpace::Linear).unwrap();
        assert_eq!(linear.kind, FormatKind::U8);
        assert_eq!(linear.family, FormatFamily::Unorm);

        let srgb = classify(F::R8G8B8A8_UNORM, ColorSpace::Srgb).unwrap();
        assert_eq!(srgb.kind, FormatKind::Srgb8);
        assert_eq!(srgb.family, FormatFamily::Srgb);
    }

    #[test]
    fn classify_srgb_format_is_normalized() {
        // A non-normalized sRGB format behaves identically to its UNORM base
        // with color_space=Srgb.
        let from_srgb_format = classify(F::B8G8R8A8_SRGB, ColorSpace::Srgb).unwrap();
        let from_normalized = classify(F::B8G8R8A8_UNORM, ColorSpace::Srgb).unwrap();
        assert_eq!(from_srgb_format, from_normalized);
        assert_eq!(from_normalized.kind, FormatKind::Bgra8Srgb);
        assert_eq!(from_normalized.family, FormatFamily::Srgb);
    }

    #[test]
    fn classify_srgb_color_space_no_effect_on_16bit() {
        // Higher-bit formats have no sRGB-native kernel — color_space is
        // applied via a scalar post-pass, not a kind promotion.
        let info = classify(F::R16G16B16A16_UNORM, ColorSpace::Srgb).unwrap();
        assert_eq!(info.kind, FormatKind::U16);
        assert_eq!(info.family, FormatFamily::Unorm);
    }

    #[test]
    fn classify_rgba32_sfloat() {
        let info = classify(F::R32G32B32A32_SFLOAT, ColorSpace::Linear).unwrap();
        assert_eq!(info.kind, FormatKind::F32);
        assert_eq!(info.family, FormatFamily::Float);
    }

    #[test]
    fn classify_r64_uint() {
        let info = classify(F::R64_UINT, ColorSpace::Linear).unwrap();
        assert_eq!(info.kind, FormatKind::U64);
        assert_eq!(info.family, FormatFamily::Uint);
    }

    #[test]
    fn classify_unsupported_packed() {
        assert!(classify(F::R5G6B5_UNORM_PACK16, ColorSpace::Linear).is_none());
    }

    #[test]
    fn classify_unsupported_compressed() {
        assert!(classify(F::BC7_UNORM_BLOCK, ColorSpace::Linear).is_none());
    }
}
