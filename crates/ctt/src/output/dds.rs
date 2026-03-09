use ddsfile::{AlphaMode, Caps2, D3D10ResourceDimension, Dds, DxgiFormat, NewDxgiParams};

use crate::compress::CompressedTexture;
use crate::error::{Error, Result};
use crate::format::{ColorSpace, CompressedFormat};

/// Map a compressed format + color space to a DXGI format.
pub fn to_dxgi_format(format: CompressedFormat, color_space: ColorSpace) -> Result<DxgiFormat> {
    let srgb = color_space == ColorSpace::Srgb;
    match format {
        CompressedFormat::Bc1 => Ok(if srgb {
            DxgiFormat::BC1_UNorm_sRGB
        } else {
            DxgiFormat::BC1_UNorm
        }),
        CompressedFormat::Bc3 => Ok(if srgb {
            DxgiFormat::BC3_UNorm_sRGB
        } else {
            DxgiFormat::BC3_UNorm
        }),
        CompressedFormat::Bc4 => Ok(DxgiFormat::BC4_UNorm),
        CompressedFormat::Bc5 => Ok(DxgiFormat::BC5_UNorm),
        CompressedFormat::Bc6h => Ok(DxgiFormat::BC6H_UF16),
        CompressedFormat::Bc7 => Ok(if srgb {
            DxgiFormat::BC7_UNorm_sRGB
        } else {
            DxgiFormat::BC7_UNorm
        }),
        CompressedFormat::Etc1 => Err(Error::UnsupportedFormat(
            "ETC1 is not supported in DDS".into(),
        )),
        CompressedFormat::Astc { .. } => Err(Error::UnsupportedFormat(
            "ASTC is not supported in DDS".into(),
        )),
    }
}

/// Encode a compressed texture as a DDS file.
pub fn encode_dds(texture: &CompressedTexture) -> Result<Vec<u8>> {
    let first = &texture.layers[0][0];
    let dxgi_format = to_dxgi_format(first.format, texture.color_space)?;

    let mut dds = Dds::new_dxgi(NewDxgiParams {
        height: first.height,
        width: first.width,
        depth: None,
        format: dxgi_format,
        mipmap_levels: Some(texture.layers[0].len() as u32),
        array_layers: Some(texture.layers.len() as u32),
        caps2: if texture.is_cubemap {
            Some(Caps2::CUBEMAP | Caps2::CUBEMAP_ALLFACES)
        } else {
            None
        },
        is_cubemap: texture.is_cubemap,
        resource_dimension: D3D10ResourceDimension::Texture2D,
        alpha_mode: AlphaMode::Unknown,
    })
    .map_err(|e| Error::OutputEncoding(format!("DDS creation failed: {e}")))?;

    // Concatenate all layer/mip data into the DDS data buffer.
    let mut data = Vec::new();
    for layer in &texture.layers {
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

    #[test]
    fn bc1_srgb_maps_correctly() {
        assert_eq!(
            to_dxgi_format(CompressedFormat::Bc1, ColorSpace::Srgb).unwrap(),
            DxgiFormat::BC1_UNorm_sRGB
        );
    }

    #[test]
    fn bc7_linear_maps_correctly() {
        assert_eq!(
            to_dxgi_format(CompressedFormat::Bc7, ColorSpace::Linear).unwrap(),
            DxgiFormat::BC7_UNorm
        );
    }

    #[test]
    fn etc1_dds_unsupported() {
        assert!(to_dxgi_format(CompressedFormat::Etc1, ColorSpace::Srgb).is_err());
    }

    #[test]
    fn astc_dds_unsupported() {
        assert!(to_dxgi_format(
            CompressedFormat::Astc {
                block_width: 4,
                block_height: 4
            },
            ColorSpace::Srgb
        )
        .is_err());
    }

    #[test]
    fn bc4_ignores_color_space() {
        assert_eq!(
            to_dxgi_format(CompressedFormat::Bc4, ColorSpace::Srgb).unwrap(),
            DxgiFormat::BC4_UNorm
        );
        assert_eq!(
            to_dxgi_format(CompressedFormat::Bc4, ColorSpace::Linear).unwrap(),
            DxgiFormat::BC4_UNorm
        );
    }
}
