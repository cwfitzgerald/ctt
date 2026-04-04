use ddsfile::{AlphaMode, Caps2, D3D10ResourceDimension, Dds, DxgiFormat, NewDxgiParams};

use crate::error::{Error, Result};
use crate::format::ColorSpace;
use crate::vk_format::FormatExt as _;

/// Map a `ktx2::Format` + `ColorSpace` to a DXGI format.
pub fn vk_format_to_dxgi(format: ktx2::Format, color_space: ColorSpace) -> Result<DxgiFormat> {
    use ktx2::Format as F;

    let full_format = format.denormalize(color_space);
    match full_format {
        // Uncompressed
        F::R8G8B8A8_UNORM => Ok(DxgiFormat::R8G8B8A8_UNorm),
        F::R8G8B8A8_SRGB => Ok(DxgiFormat::R8G8B8A8_UNorm_sRGB),
        F::B8G8R8A8_UNORM => Ok(DxgiFormat::B8G8R8A8_UNorm),
        F::B8G8R8A8_SRGB => Ok(DxgiFormat::B8G8R8A8_UNorm_sRGB),
        F::R16G16B16A16_SFLOAT => Ok(DxgiFormat::R16G16B16A16_Float),
        F::R32G32B32A32_SFLOAT => Ok(DxgiFormat::R32G32B32A32_Float),
        F::R8_UNORM => Ok(DxgiFormat::R8_UNorm),
        F::R8G8_UNORM => Ok(DxgiFormat::R8G8_UNorm),
        F::R16_UNORM => Ok(DxgiFormat::R16_UNorm),
        F::R16_SFLOAT => Ok(DxgiFormat::R16_Float),
        F::R32_SFLOAT => Ok(DxgiFormat::R32_Float),
        F::B10G11R11_UFLOAT_PACK32 => Ok(DxgiFormat::R11G11B10_Float),

        // BC compressed
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

        _ => Err(Error::UnsupportedFormat(format!(
            "{full_format:?} is not supported in DDS"
        ))),
    }
}

/// Encode an [`Image`] as a DDS file.
pub fn encode_dds_image(image: &crate::surface::Image) -> Result<Vec<u8>> {
    let first = &image.surfaces[0][0];
    let dxgi_format = vk_format_to_dxgi(first.format, first.color_space)?;
    log::debug!(
        "DDS: {:?}, {} layers, {} mips",
        dxgi_format,
        image.surfaces.len(),
        image.surfaces[0].len()
    );

    let mut dds = Dds::new_dxgi(NewDxgiParams {
        height: first.height,
        width: first.width,
        depth: None,
        format: dxgi_format,
        mipmap_levels: Some(image.surfaces[0].len() as u32),
        array_layers: Some(image.surfaces.len() as u32),
        caps2: if image.is_cubemap {
            Some(Caps2::CUBEMAP | Caps2::CUBEMAP_ALLFACES)
        } else {
            None
        },
        is_cubemap: image.is_cubemap,
        resource_dimension: D3D10ResourceDimension::Texture2D,
        alpha_mode: AlphaMode::Unknown,
    })
    .map_err(|e| Error::OutputEncoding(format!("DDS creation failed: {e}")))?;

    let mut data = Vec::new();
    for layer in &image.surfaces {
        for mip in layer {
            data.extend_from_slice(&mip.data);
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
    fn astc_dds_unsupported() {
        assert!(vk_format_to_dxgi(F::ASTC_4x4_UNORM_BLOCK, ColorSpace::Srgb).is_err());
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
}
