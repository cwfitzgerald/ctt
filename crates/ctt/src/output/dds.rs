use ddsfile::{AlphaMode, Caps2, D3D10ResourceDimension, Dds, DxgiFormat, NewDxgiParams};

use crate::error::{Error, Result};
use crate::surface::{ColorSpace, TextureKind};
use crate::vk_format::FormatExt as _;

/// Map a `ktx2::Format` + `ColorSpace` to a DXGI format.
pub fn vk_format_to_dxgi(format: ktx2::Format, color_space: ColorSpace) -> Result<DxgiFormat> {
    use ktx2::Format as F;

    let full_format = format.denormalize(color_space);
    match full_format {
        // ── Uncompressed ────────────────────────────────────────────────

        // R8
        F::R8_UNORM => Ok(DxgiFormat::R8_UNorm),
        F::R8_SNORM => Ok(DxgiFormat::R8_SNorm),
        F::R8_UINT => Ok(DxgiFormat::R8_UInt),
        F::R8_SINT => Ok(DxgiFormat::R8_SInt),

        // R8G8
        F::R8G8_UNORM => Ok(DxgiFormat::R8G8_UNorm),
        F::R8G8_SNORM => Ok(DxgiFormat::R8G8_SNorm),
        F::R8G8_UINT => Ok(DxgiFormat::R8G8_UInt),
        F::R8G8_SINT => Ok(DxgiFormat::R8G8_SInt),

        // R8G8B8A8
        F::R8G8B8A8_UNORM => Ok(DxgiFormat::R8G8B8A8_UNorm),
        F::R8G8B8A8_SRGB => Ok(DxgiFormat::R8G8B8A8_UNorm_sRGB),
        F::R8G8B8A8_SNORM => Ok(DxgiFormat::R8G8B8A8_SNorm),
        F::R8G8B8A8_UINT => Ok(DxgiFormat::R8G8B8A8_UInt),
        F::R8G8B8A8_SINT => Ok(DxgiFormat::R8G8B8A8_SInt),

        // B8G8R8A8
        F::B8G8R8A8_UNORM => Ok(DxgiFormat::B8G8R8A8_UNorm),
        F::B8G8R8A8_SRGB => Ok(DxgiFormat::B8G8R8A8_UNorm_sRGB),

        // A2B10G10R10 (VK) -> R10G10B10A2 (DXGI) — same bit layout
        F::A2B10G10R10_UNORM_PACK32 => Ok(DxgiFormat::R10G10B10A2_UNorm),
        F::A2B10G10R10_UINT_PACK32 => Ok(DxgiFormat::R10G10B10A2_UInt),

        // R16
        F::R16_UNORM => Ok(DxgiFormat::R16_UNorm),
        F::R16_SNORM => Ok(DxgiFormat::R16_SNorm),
        F::R16_UINT => Ok(DxgiFormat::R16_UInt),
        F::R16_SINT => Ok(DxgiFormat::R16_SInt),
        F::R16_SFLOAT => Ok(DxgiFormat::R16_Float),

        // R16G16
        F::R16G16_UNORM => Ok(DxgiFormat::R16G16_UNorm),
        F::R16G16_SNORM => Ok(DxgiFormat::R16G16_SNorm),
        F::R16G16_UINT => Ok(DxgiFormat::R16G16_UInt),
        F::R16G16_SINT => Ok(DxgiFormat::R16G16_SInt),
        F::R16G16_SFLOAT => Ok(DxgiFormat::R16G16_Float),

        // R16G16B16A16
        F::R16G16B16A16_UNORM => Ok(DxgiFormat::R16G16B16A16_UNorm),
        F::R16G16B16A16_SNORM => Ok(DxgiFormat::R16G16B16A16_SNorm),
        F::R16G16B16A16_UINT => Ok(DxgiFormat::R16G16B16A16_UInt),
        F::R16G16B16A16_SINT => Ok(DxgiFormat::R16G16B16A16_SInt),
        F::R16G16B16A16_SFLOAT => Ok(DxgiFormat::R16G16B16A16_Float),

        // R32
        F::R32_SFLOAT => Ok(DxgiFormat::R32_Float),
        F::R32_UINT => Ok(DxgiFormat::R32_UInt),
        F::R32_SINT => Ok(DxgiFormat::R32_SInt),

        // R32G32
        F::R32G32_SFLOAT => Ok(DxgiFormat::R32G32_Float),
        F::R32G32_UINT => Ok(DxgiFormat::R32G32_UInt),
        F::R32G32_SINT => Ok(DxgiFormat::R32G32_SInt),

        // R32G32B32
        F::R32G32B32_SFLOAT => Ok(DxgiFormat::R32G32B32_Float),
        F::R32G32B32_UINT => Ok(DxgiFormat::R32G32B32_UInt),
        F::R32G32B32_SINT => Ok(DxgiFormat::R32G32B32_SInt),

        // R32G32B32A32
        F::R32G32B32A32_SFLOAT => Ok(DxgiFormat::R32G32B32A32_Float),
        F::R32G32B32A32_UINT => Ok(DxgiFormat::R32G32B32A32_UInt),
        F::R32G32B32A32_SINT => Ok(DxgiFormat::R32G32B32A32_SInt),

        // Packed
        F::B10G11R11_UFLOAT_PACK32 => Ok(DxgiFormat::R11G11B10_Float),
        F::E5B9G9R9_UFLOAT_PACK32 => Ok(DxgiFormat::R9G9B9E5_SharedExp),
        F::B5G6R5_UNORM_PACK16 => Ok(DxgiFormat::B5G6R5_UNorm),
        F::B5G5R5A1_UNORM_PACK16 => Ok(DxgiFormat::B5G5R5A1_UNorm),
        F::B4G4R4A4_UNORM_PACK16 => Ok(DxgiFormat::B4G4R4A4_UNorm),

        // ── BC compressed ───────────────────────────────────────────────
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGB_UNORM_BLOCK => Ok(DxgiFormat::BC1_UNorm),
        F::BC1_RGBA_SRGB_BLOCK | F::BC1_RGB_SRGB_BLOCK => Ok(DxgiFormat::BC1_UNorm_sRGB),
        F::BC2_UNORM_BLOCK => Ok(DxgiFormat::BC2_UNorm),
        F::BC2_SRGB_BLOCK => Ok(DxgiFormat::BC2_UNorm_sRGB),
        F::BC3_UNORM_BLOCK => Ok(DxgiFormat::BC3_UNorm),
        F::BC3_SRGB_BLOCK => Ok(DxgiFormat::BC3_UNorm_sRGB),
        F::BC4_UNORM_BLOCK => Ok(DxgiFormat::BC4_UNorm),
        F::BC4_SNORM_BLOCK => Ok(DxgiFormat::BC4_SNorm),
        F::BC5_UNORM_BLOCK => Ok(DxgiFormat::BC5_UNorm),
        F::BC5_SNORM_BLOCK => Ok(DxgiFormat::BC5_SNorm),
        F::BC6H_UFLOAT_BLOCK => Ok(DxgiFormat::BC6H_UF16),
        F::BC6H_SFLOAT_BLOCK => Ok(DxgiFormat::BC6H_SF16),
        F::BC7_UNORM_BLOCK => Ok(DxgiFormat::BC7_UNorm),
        F::BC7_SRGB_BLOCK => Ok(DxgiFormat::BC7_UNorm_sRGB),

        // ── ASTC compressed (NVIDIA DXGI extension) ─────────────────────
        F::ASTC_4x4_UNORM_BLOCK => Ok(DxgiFormat::ASTC_4x4_UNorm),
        F::ASTC_4x4_SRGB_BLOCK => Ok(DxgiFormat::ASTC_4x4_UNorm_sRGB),
        F::ASTC_5x4_UNORM_BLOCK => Ok(DxgiFormat::ASTC_5x4_UNorm),
        F::ASTC_5x4_SRGB_BLOCK => Ok(DxgiFormat::ASTC_5x4_UNorm_sRGB),
        F::ASTC_5x5_UNORM_BLOCK => Ok(DxgiFormat::ASTC_5x5_UNorm),
        F::ASTC_5x5_SRGB_BLOCK => Ok(DxgiFormat::ASTC_5x5_UNorm_sRGB),
        F::ASTC_6x5_UNORM_BLOCK => Ok(DxgiFormat::ASTC_6x5_UNorm),
        F::ASTC_6x5_SRGB_BLOCK => Ok(DxgiFormat::ASTC_6x5_UNorm_sRGB),
        F::ASTC_6x6_UNORM_BLOCK => Ok(DxgiFormat::ASTC_6x6_UNorm),
        F::ASTC_6x6_SRGB_BLOCK => Ok(DxgiFormat::ASTC_6x6_UNorm_sRGB),
        F::ASTC_8x5_UNORM_BLOCK => Ok(DxgiFormat::ASTC_8x5_UNorm),
        F::ASTC_8x5_SRGB_BLOCK => Ok(DxgiFormat::ASTC_8x5_UNorm_sRGB),
        F::ASTC_8x6_UNORM_BLOCK => Ok(DxgiFormat::ASTC_8x6_UNorm),
        F::ASTC_8x6_SRGB_BLOCK => Ok(DxgiFormat::ASTC_8x6_UNorm_sRGB),
        F::ASTC_8x8_UNORM_BLOCK => Ok(DxgiFormat::ASTC_8x8_UNorm),
        F::ASTC_8x8_SRGB_BLOCK => Ok(DxgiFormat::ASTC_8x8_UNorm_sRGB),
        F::ASTC_10x5_UNORM_BLOCK => Ok(DxgiFormat::ASTC_10x5_UNorm),
        F::ASTC_10x5_SRGB_BLOCK => Ok(DxgiFormat::ASTC_10x5_UNorm_sRGB),
        F::ASTC_10x6_UNORM_BLOCK => Ok(DxgiFormat::ASTC_10x6_UNorm),
        F::ASTC_10x6_SRGB_BLOCK => Ok(DxgiFormat::ASTC_10x6_UNorm_sRGB),
        F::ASTC_10x8_UNORM_BLOCK => Ok(DxgiFormat::ASTC_10x8_UNorm),
        F::ASTC_10x8_SRGB_BLOCK => Ok(DxgiFormat::ASTC_10x8_UNorm_sRGB),
        F::ASTC_10x10_UNORM_BLOCK => Ok(DxgiFormat::ASTC_10x10_UNorm),
        F::ASTC_10x10_SRGB_BLOCK => Ok(DxgiFormat::ASTC_10x10_UNorm_sRGB),
        F::ASTC_12x10_UNORM_BLOCK => Ok(DxgiFormat::ASTC_12x10_UNorm),
        F::ASTC_12x10_SRGB_BLOCK => Ok(DxgiFormat::ASTC_12x10_UNorm_sRGB),
        F::ASTC_12x12_UNORM_BLOCK => Ok(DxgiFormat::ASTC_12x12_UNorm),
        F::ASTC_12x12_SRGB_BLOCK => Ok(DxgiFormat::ASTC_12x12_UNorm_sRGB),

        _ => Err(Error::UnsupportedFormat(format!(
            "{full_format:?} is not supported in DDS"
        ))),
    }
}

/// Encode an [`Image`](crate::surface::Image) as a DDS file.
pub fn encode_dds_image(image: &crate::surface::Image) -> Result<Vec<u8>> {
    let first = &image.surfaces[0][0];
    let dxgi_format = vk_format_to_dxgi(first.format, first.color_space)?;

    // Image::validate has already enforced kind invariants by the time we get
    // here (Cubemap multiple-of-6, Texture3D single surface).
    let is_cubemap = matches!(image.kind, TextureKind::Cubemap);
    let is_3d = matches!(image.kind, TextureKind::Texture3D);

    let (depth_param, array_layers, resource_dimension) = if is_3d {
        (
            Some(first.depth.max(1)),
            None,
            D3D10ResourceDimension::Texture3D,
        )
    } else {
        (
            None,
            Some(image.surfaces.len() as u32),
            D3D10ResourceDimension::Texture2D,
        )
    };

    log::debug!(
        "DDS: {:?}, {} surfaces, {} mips, kind={:?}",
        dxgi_format,
        image.surfaces.len(),
        image.surfaces[0].len(),
        image.kind,
    );

    let mut dds = Dds::new_dxgi(NewDxgiParams {
        height: first.height,
        width: first.width,
        depth: depth_param,
        format: dxgi_format,
        mipmap_levels: Some(image.surfaces[0].len() as u32),
        array_layers,
        caps2: if is_cubemap {
            Some(Caps2::CUBEMAP | Caps2::CUBEMAP_ALLFACES)
        } else {
            None
        },
        is_cubemap,
        resource_dimension,
        alpha_mode: AlphaMode::Unknown,
    })
    .map_err(|e| Error::OutputEncoding(format!("DDS creation failed: {e}")))?;

    // tight_data strips per-row and per-slice padding so the DDS payload
    // matches D3D's expected packing.
    let mut data = Vec::new();
    for layer in &image.surfaces {
        for mip in layer {
            data.extend_from_slice(&mip.tight_data());
        }
    }
    dds.data = data;

    let mut output = Vec::new();
    dds.write(&mut output)
        .map_err(|e| Error::OutputEncoding(format!("DDS write failed: {e}")))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktx2::Format as F;

    #[test]
    fn bc1_srgb_maps_correctly() {
        assert_eq!(
            vk_format_to_dxgi(F::BC1_RGBA_UNORM_BLOCK, ColorSpace::Srgb).unwrap(),
            DxgiFormat::BC1_UNorm_sRGB
        );
    }

    #[test]
    fn bc7_linear_maps_correctly() {
        assert_eq!(
            vk_format_to_dxgi(F::BC7_UNORM_BLOCK, ColorSpace::Linear).unwrap(),
            DxgiFormat::BC7_UNorm
        );
    }

    #[test]
    fn etc1_dds_unsupported() {
        assert!(vk_format_to_dxgi(F::ETC2_R8G8B8_UNORM_BLOCK, ColorSpace::Srgb).is_err());
    }

    #[test]
    fn astc_4x4_srgb_maps_correctly() {
        assert_eq!(
            vk_format_to_dxgi(F::ASTC_4x4_UNORM_BLOCK, ColorSpace::Srgb).unwrap(),
            DxgiFormat::ASTC_4x4_UNorm_sRGB
        );
    }

    #[test]
    fn astc_12x12_linear_maps_correctly() {
        assert_eq!(
            vk_format_to_dxgi(F::ASTC_12x12_UNORM_BLOCK, ColorSpace::Linear).unwrap(),
            DxgiFormat::ASTC_12x12_UNorm
        );
    }

    #[test]
    fn bc4_ignores_color_space() {
        assert_eq!(
            vk_format_to_dxgi(F::BC4_UNORM_BLOCK, ColorSpace::Srgb).unwrap(),
            DxgiFormat::BC4_UNorm
        );
        assert_eq!(
            vk_format_to_dxgi(F::BC4_UNORM_BLOCK, ColorSpace::Linear).unwrap(),
            DxgiFormat::BC4_UNorm
        );
    }

    #[test]
    fn r10g10b10a2_maps_correctly() {
        assert_eq!(
            vk_format_to_dxgi(F::A2B10G10R10_UNORM_PACK32, ColorSpace::Linear).unwrap(),
            DxgiFormat::R10G10B10A2_UNorm
        );
    }

    #[test]
    fn shared_exponent_maps_correctly() {
        assert_eq!(
            vk_format_to_dxgi(F::E5B9G9R9_UFLOAT_PACK32, ColorSpace::Linear).unwrap(),
            DxgiFormat::R9G9B9E5_SharedExp
        );
    }
}
