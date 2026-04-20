use crate::encoders::EncoderRegistry;
use crate::error::Error;

/// The target format for a conversion operation.
///
/// Use [`parse_format`] to construct this from a format string, or build
/// it directly when you already know the target [`Format`](ktx2::Format).
pub enum TargetFormat {
    /// A block-compressed format, optionally targeting a specific encoder.
    Compressed {
        /// Specific encoder backend to use (e.g. `"intel"`, `"bc7e"`).
        /// If `None`, the highest-priority encoder that supports the format is used.
        encoder_name: Option<String>,
        /// The target compressed format (e.g. `Format::BC7_UNORM_BLOCK`).
        format: ktx2::Format,
    },
    /// An uncompressed pixel format (e.g. `Format::R8G8B8A8_UNORM`).
    Uncompressed(ktx2::Format),
}

/// Parse a format string that may be compressed or uncompressed.
///
/// Compressed formats may have an encoder prefix (e.g. `intel_bc7`).
/// Uncompressed formats accept both WebGPU-style (`rgba8unorm`) and Vulkan-style
/// (`r8g8b8a8_unorm`) names.
pub fn parse_format(s: &str, registry: &EncoderRegistry) -> Result<TargetFormat, Error> {
    let lower = s.to_lowercase();

    // Try uncompressed first — these names never collide with encoder prefixes.
    if let Some(format) = parse_uncompressed(&lower) {
        return Ok(TargetFormat::Uncompressed(format));
    }

    // Try encoder-prefixed compressed formats.
    let mut prefixes: Vec<String> = registry
        .encoders()
        .iter()
        .map(|e| e.name().to_string())
        .collect();
    prefixes.sort_by_key(|p| std::cmp::Reverse(p.len()));

    for prefix in &prefixes {
        if let Some(rest) = lower
            .strip_prefix(prefix.as_str())
            .and_then(|r| r.strip_prefix('_'))
        {
            let format = parse_compressed(rest, s)?;
            return Ok(TargetFormat::Compressed {
                encoder_name: Some(prefix.to_string()),
                format,
            });
        }
    }

    // Bare compressed format.
    let format = parse_compressed(&lower, s)?;
    Ok(TargetFormat::Compressed {
        encoder_name: None,
        format,
    })
}

/// Short display name for a compressed format (used in encoder listings).
pub fn format_short_name(format: ktx2::Format) -> String {
    use ktx2::Format as F;
    match format {
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGB_UNORM_BLOCK => "bc1".into(),
        F::BC2_UNORM_BLOCK => "bc2".into(),
        F::BC3_UNORM_BLOCK => "bc3".into(),
        F::BC4_UNORM_BLOCK => "bc4".into(),
        F::BC4_SNORM_BLOCK => "bc4s".into(),
        F::BC5_UNORM_BLOCK => "bc5".into(),
        F::BC5_SNORM_BLOCK => "bc5s".into(),
        F::BC6H_UFLOAT_BLOCK => "bc6h".into(),
        F::BC6H_SFLOAT_BLOCK => "bc6hsf".into(),
        F::BC7_UNORM_BLOCK => "bc7".into(),
        F::ETC2_R8G8B8_UNORM_BLOCK => "etc1".into(),
        F::ETC2_R8G8B8A1_UNORM_BLOCK => "etc2_rgba1".into(),
        F::ETC2_R8G8B8A8_UNORM_BLOCK => "etc2_rgba".into(),
        F::EAC_R11_UNORM_BLOCK => "eac_r".into(),
        F::EAC_R11G11_UNORM_BLOCK => "eac_rg".into(),
        _ => format!("{format:?}").to_lowercase(),
    }
}

// ---------------------------------------------------------------------------
// Compressed format parsing
// ---------------------------------------------------------------------------

fn parse_compressed(lower: &str, original: &str) -> Result<ktx2::Format, Error> {
    use ktx2::Format as F;
    match lower {
        "bc1" => Ok(F::BC1_RGBA_UNORM_BLOCK),
        "bc2" => Ok(F::BC2_UNORM_BLOCK),
        "bc3" => Ok(F::BC3_UNORM_BLOCK),
        "bc4" => Ok(F::BC4_UNORM_BLOCK),
        "bc4s" => Ok(F::BC4_SNORM_BLOCK),
        "bc5" => Ok(F::BC5_UNORM_BLOCK),
        "bc5s" => Ok(F::BC5_SNORM_BLOCK),
        "bc6h" => Ok(F::BC6H_UFLOAT_BLOCK),
        "bc6hsf" | "bc6h_sf" => Ok(F::BC6H_SFLOAT_BLOCK),
        "bc7" => Ok(F::BC7_UNORM_BLOCK),
        "etc1" | "etc2_rgb" => Ok(F::ETC2_R8G8B8_UNORM_BLOCK),
        "etc2_rgba1" | "etc2_rgb_a1" => Ok(F::ETC2_R8G8B8A1_UNORM_BLOCK),
        "etc2" | "etc2_rgba" => Ok(F::ETC2_R8G8B8A8_UNORM_BLOCK),
        "eac_r" | "etc2_r" | "etc2_r11" => Ok(F::EAC_R11_UNORM_BLOCK),
        "eac_rg" | "etc2_rg" | "etc2_rg11" => Ok(F::EAC_R11G11_UNORM_BLOCK),
        other => {
            if let Some(rest) = other.strip_prefix("astc_") {
                let (w, h) = rest
                    .split_once('x')
                    .ok_or_else(|| Error::UnsupportedFormat(original.into()))?;
                let block_width: u8 = w
                    .parse()
                    .map_err(|_| Error::UnsupportedFormat(original.into()))?;
                let block_height: u8 = h
                    .parse()
                    .map_err(|_| Error::UnsupportedFormat(original.into()))?;
                astc_format(block_width, block_height)
                    .ok_or_else(|| Error::UnsupportedFormat(original.into()))
            } else {
                Err(Error::UnsupportedFormat(original.into()))
            }
        }
    }
}

fn astc_format(block_width: u8, block_height: u8) -> Option<ktx2::Format> {
    use ktx2::Format as F;
    Some(match (block_width, block_height) {
        (4, 4) => F::ASTC_4x4_UNORM_BLOCK,
        (5, 4) => F::ASTC_5x4_UNORM_BLOCK,
        (5, 5) => F::ASTC_5x5_UNORM_BLOCK,
        (6, 5) => F::ASTC_6x5_UNORM_BLOCK,
        (6, 6) => F::ASTC_6x6_UNORM_BLOCK,
        (8, 5) => F::ASTC_8x5_UNORM_BLOCK,
        (8, 6) => F::ASTC_8x6_UNORM_BLOCK,
        (8, 8) => F::ASTC_8x8_UNORM_BLOCK,
        (10, 5) => F::ASTC_10x5_UNORM_BLOCK,
        (10, 6) => F::ASTC_10x6_UNORM_BLOCK,
        (10, 8) => F::ASTC_10x8_UNORM_BLOCK,
        (10, 10) => F::ASTC_10x10_UNORM_BLOCK,
        (12, 10) => F::ASTC_12x10_UNORM_BLOCK,
        (12, 12) => F::ASTC_12x12_UNORM_BLOCK,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Uncompressed format parsing
// ---------------------------------------------------------------------------

/// Try to parse an uncompressed format name.
///
/// Each format accepts both WebGPU-style (`rgba8unorm`) and Vulkan-style (`r8g8b8a8_unorm`)
/// names. Float suffixes (`float`, `sfloat`) map to SFLOAT. Integer type suffixes
/// (`unorm`, `snorm`, `uint`, `sint`) must be specified explicitly.
#[rustfmt::skip]
fn parse_uncompressed(s: &str) -> Option<ktx2::Format> {
    use ktx2::Format as F;

    Some(match s {
        // --- R8 ---
        "r8unorm" | "r8_unorm" => F::R8_UNORM,
        "r8snorm" | "r8_snorm" => F::R8_SNORM,
        "r8uint" | "r8_uint" => F::R8_UINT,
        "r8sint" | "r8_sint" => F::R8_SINT,

        // --- RG8 ---
        "rg8unorm" | "r8g8_unorm" => F::R8G8_UNORM,
        "rg8snorm" | "r8g8_snorm" => F::R8G8_SNORM,
        "rg8uint" | "r8g8_uint" => F::R8G8_UINT,
        "rg8sint" | "r8g8_sint" => F::R8G8_SINT,

        // --- RGBA8 ---
        "rgba8unorm" | "r8g8b8a8_unorm" => F::R8G8B8A8_UNORM,
        "rgba8snorm" | "r8g8b8a8_snorm" => F::R8G8B8A8_SNORM,
        "rgba8uint" | "r8g8b8a8_uint" => F::R8G8B8A8_UINT,
        "rgba8sint" | "r8g8b8a8_sint" => F::R8G8B8A8_SINT,

        // --- BGRA8 ---
        "bgra8unorm" | "b8g8r8a8_unorm" => F::B8G8R8A8_UNORM,
        "bgra8snorm" | "b8g8r8a8_snorm" => F::B8G8R8A8_SNORM,
        "bgra8uint" | "b8g8r8a8_uint" => F::B8G8R8A8_UINT,
        "bgra8sint" | "b8g8r8a8_sint" => F::B8G8R8A8_SINT,

        // --- R16 ---
        "r16unorm" | "r16_unorm" => F::R16_UNORM,
        "r16snorm" | "r16_snorm" => F::R16_SNORM,
        "r16uint" | "r16_uint" => F::R16_UINT,
        "r16sint" | "r16_sint" => F::R16_SINT,
        "r16float" | "r16sfloat" | "r16_float" | "r16_sfloat" => F::R16_SFLOAT,

        // --- RG16 ---
        "rg16unorm" | "r16g16_unorm" => F::R16G16_UNORM,
        "rg16snorm" | "r16g16_snorm" => F::R16G16_SNORM,
        "rg16uint" | "r16g16_uint" => F::R16G16_UINT,
        "rg16sint" | "r16g16_sint" => F::R16G16_SINT,
        "rg16float" | "rg16sfloat" | "r16g16_float" | "r16g16_sfloat" => F::R16G16_SFLOAT,

        // --- RGBA16 ---
        "rgba16unorm" | "r16g16b16a16_unorm" => F::R16G16B16A16_UNORM,
        "rgba16snorm" | "r16g16b16a16_snorm" => F::R16G16B16A16_SNORM,
        "rgba16uint" | "r16g16b16a16_uint" => F::R16G16B16A16_UINT,
        "rgba16sint" | "r16g16b16a16_sint" => F::R16G16B16A16_SINT,
        "rgba16float" | "rgba16sfloat" | "r16g16b16a16_float" | "r16g16b16a16_sfloat" => F::R16G16B16A16_SFLOAT,

        // --- R32 ---
        "r32uint" | "r32_uint" => F::R32_UINT,
        "r32sint" | "r32_sint" => F::R32_SINT,
        "r32float" | "r32sfloat" | "r32_float" | "r32_sfloat" => F::R32_SFLOAT,

        // --- RG32 ---
        "rg32uint" | "r32g32_uint" => F::R32G32_UINT,
        "rg32sint" | "r32g32_sint" => F::R32G32_SINT,
        "rg32float" | "rg32sfloat" | "r32g32_float" | "r32g32_sfloat" => F::R32G32_SFLOAT,

        // --- RGBA32 ---
        "rgba32uint" | "r32g32b32a32_uint" => F::R32G32B32A32_UINT,
        "rgba32sint" | "r32g32b32a32_sint" => F::R32G32B32A32_SINT,
        "rgba32float" | "rgba32sfloat" | "r32g32b32a32_float" | "r32g32b32a32_sfloat" => F::R32G32B32A32_SFLOAT,

        // --- Packed ---
        "rg11b10ufloat" | "rg11b10float" | "b10g11r11_ufloat" | "b10g11r11_ufloat_pack32" => F::B10G11R11_UFLOAT_PACK32,

        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktx2::Format as F;

    #[test]
    fn webgpu_rgba8unorm() {
        assert_eq!(parse_uncompressed("rgba8unorm"), Some(F::R8G8B8A8_UNORM));
    }

    #[test]
    fn vulkan_r8g8b8a8_unorm() {
        assert_eq!(
            parse_uncompressed("r8g8b8a8_unorm"),
            Some(F::R8G8B8A8_UNORM)
        );
    }

    #[test]
    fn webgpu_float_defaults_to_sfloat() {
        assert_eq!(
            parse_uncompressed("rgba16float"),
            Some(F::R16G16B16A16_SFLOAT)
        );
        assert_eq!(parse_uncompressed("r32float"), Some(F::R32_SFLOAT));
    }

    #[test]
    fn vulkan_float_defaults_to_sfloat() {
        assert_eq!(
            parse_uncompressed("r16g16b16a16_float"),
            Some(F::R16G16B16A16_SFLOAT)
        );
        assert_eq!(parse_uncompressed("r32_float"), Some(F::R32_SFLOAT));
    }

    #[test]
    fn webgpu_bgra() {
        assert_eq!(parse_uncompressed("bgra8unorm"), Some(F::B8G8R8A8_UNORM));
    }

    #[test]
    fn vulkan_bgra() {
        assert_eq!(
            parse_uncompressed("b8g8r8a8_unorm"),
            Some(F::B8G8R8A8_UNORM)
        );
    }

    #[test]
    fn packed_rg11b10() {
        assert_eq!(
            parse_uncompressed("rg11b10ufloat"),
            Some(F::B10G11R11_UFLOAT_PACK32)
        );
        assert_eq!(
            parse_uncompressed("b10g11r11_ufloat"),
            Some(F::B10G11R11_UFLOAT_PACK32)
        );
    }

    #[test]
    fn bare_name_without_type_is_not_uncompressed() {
        assert_eq!(parse_uncompressed("rgba8"), None);
        assert_eq!(parse_uncompressed("r16"), None);
    }

    #[test]
    fn compressed_not_parsed_as_uncompressed() {
        assert_eq!(parse_uncompressed("bc7"), None);
        assert_eq!(parse_uncompressed("astc_4x4"), None);
    }

    #[test]
    fn parse_format_uncompressed() {
        let registry = EncoderRegistry::default_registry();
        match parse_format("rgba8unorm", &registry).unwrap() {
            TargetFormat::Uncompressed(f) => assert_eq!(f, F::R8G8B8A8_UNORM),
            _ => panic!("expected uncompressed"),
        }
    }

    #[test]
    fn parse_format_compressed() {
        let registry = EncoderRegistry::default_registry();
        match parse_format("bc7", &registry).unwrap() {
            TargetFormat::Compressed {
                encoder_name,
                format,
            } => {
                assert_eq!(encoder_name, None);
                assert_eq!(format, F::BC7_UNORM_BLOCK);
            }
            _ => panic!("expected compressed"),
        }
    }
}
