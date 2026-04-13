use ddsfile::{Caps2, D3DFormat, Dds, DxgiFormat};

use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::surface::{ColorSpace, Image, Surface};
use crate::vk_format::FormatExt as _;

/// Map a DXGI format to a `ktx2::Format` + `ColorSpace`.
///
/// This is the inverse of [`crate::output::dds::vk_format_to_dxgi`].
fn dxgi_to_vk_format(dxgi: DxgiFormat) -> Result<(ktx2::Format, ColorSpace)> {
    use ktx2::Format as F;

    let (format, cs) = match dxgi {
        // ── Uncompressed ────────────────────────────────────────────────

        // R8
        DxgiFormat::R8_UNorm => (F::R8_UNORM, ColorSpace::Linear),
        DxgiFormat::R8_SNorm => (F::R8_SNORM, ColorSpace::Linear),
        DxgiFormat::R8_UInt => (F::R8_UINT, ColorSpace::Linear),
        DxgiFormat::R8_SInt => (F::R8_SINT, ColorSpace::Linear),

        // R8G8
        DxgiFormat::R8G8_UNorm => (F::R8G8_UNORM, ColorSpace::Linear),
        DxgiFormat::R8G8_SNorm => (F::R8G8_SNORM, ColorSpace::Linear),
        DxgiFormat::R8G8_UInt => (F::R8G8_UINT, ColorSpace::Linear),
        DxgiFormat::R8G8_SInt => (F::R8G8_SINT, ColorSpace::Linear),

        // R8G8B8A8
        DxgiFormat::R8G8B8A8_UNorm => (F::R8G8B8A8_UNORM, ColorSpace::Linear),
        DxgiFormat::R8G8B8A8_UNorm_sRGB => (F::R8G8B8A8_UNORM, ColorSpace::Srgb),
        DxgiFormat::R8G8B8A8_SNorm => (F::R8G8B8A8_SNORM, ColorSpace::Linear),
        DxgiFormat::R8G8B8A8_UInt => (F::R8G8B8A8_UINT, ColorSpace::Linear),
        DxgiFormat::R8G8B8A8_SInt => (F::R8G8B8A8_SINT, ColorSpace::Linear),

        // B8G8R8A8
        DxgiFormat::B8G8R8A8_UNorm => (F::B8G8R8A8_UNORM, ColorSpace::Linear),
        DxgiFormat::B8G8R8A8_UNorm_sRGB => (F::B8G8R8A8_UNORM, ColorSpace::Srgb),

        // R10G10B10A2 -> A2B10G10R10 (same bit layout)
        DxgiFormat::R10G10B10A2_UNorm => (F::A2B10G10R10_UNORM_PACK32, ColorSpace::Linear),
        DxgiFormat::R10G10B10A2_UInt => (F::A2B10G10R10_UINT_PACK32, ColorSpace::Linear),

        // R16
        DxgiFormat::R16_UNorm => (F::R16_UNORM, ColorSpace::Linear),
        DxgiFormat::R16_SNorm => (F::R16_SNORM, ColorSpace::Linear),
        DxgiFormat::R16_UInt => (F::R16_UINT, ColorSpace::Linear),
        DxgiFormat::R16_SInt => (F::R16_SINT, ColorSpace::Linear),
        DxgiFormat::R16_Float => (F::R16_SFLOAT, ColorSpace::Linear),

        // R16G16
        DxgiFormat::R16G16_UNorm => (F::R16G16_UNORM, ColorSpace::Linear),
        DxgiFormat::R16G16_SNorm => (F::R16G16_SNORM, ColorSpace::Linear),
        DxgiFormat::R16G16_UInt => (F::R16G16_UINT, ColorSpace::Linear),
        DxgiFormat::R16G16_SInt => (F::R16G16_SINT, ColorSpace::Linear),
        DxgiFormat::R16G16_Float => (F::R16G16_SFLOAT, ColorSpace::Linear),

        // R16G16B16A16
        DxgiFormat::R16G16B16A16_UNorm => (F::R16G16B16A16_UNORM, ColorSpace::Linear),
        DxgiFormat::R16G16B16A16_SNorm => (F::R16G16B16A16_SNORM, ColorSpace::Linear),
        DxgiFormat::R16G16B16A16_UInt => (F::R16G16B16A16_UINT, ColorSpace::Linear),
        DxgiFormat::R16G16B16A16_SInt => (F::R16G16B16A16_SINT, ColorSpace::Linear),
        DxgiFormat::R16G16B16A16_Float => (F::R16G16B16A16_SFLOAT, ColorSpace::Linear),

        // R32
        DxgiFormat::R32_Float => (F::R32_SFLOAT, ColorSpace::Linear),
        DxgiFormat::R32_UInt => (F::R32_UINT, ColorSpace::Linear),
        DxgiFormat::R32_SInt => (F::R32_SINT, ColorSpace::Linear),

        // R32G32
        DxgiFormat::R32G32_Float => (F::R32G32_SFLOAT, ColorSpace::Linear),
        DxgiFormat::R32G32_UInt => (F::R32G32_UINT, ColorSpace::Linear),
        DxgiFormat::R32G32_SInt => (F::R32G32_SINT, ColorSpace::Linear),

        // R32G32B32
        DxgiFormat::R32G32B32_Float => (F::R32G32B32_SFLOAT, ColorSpace::Linear),
        DxgiFormat::R32G32B32_UInt => (F::R32G32B32_UINT, ColorSpace::Linear),
        DxgiFormat::R32G32B32_SInt => (F::R32G32B32_SINT, ColorSpace::Linear),

        // R32G32B32A32
        DxgiFormat::R32G32B32A32_Float => (F::R32G32B32A32_SFLOAT, ColorSpace::Linear),
        DxgiFormat::R32G32B32A32_UInt => (F::R32G32B32A32_UINT, ColorSpace::Linear),
        DxgiFormat::R32G32B32A32_SInt => (F::R32G32B32A32_SINT, ColorSpace::Linear),

        // Packed
        DxgiFormat::R11G11B10_Float => (F::B10G11R11_UFLOAT_PACK32, ColorSpace::Linear),
        DxgiFormat::R9G9B9E5_SharedExp => (F::E5B9G9R9_UFLOAT_PACK32, ColorSpace::Linear),
        DxgiFormat::B5G6R5_UNorm => (F::B5G6R5_UNORM_PACK16, ColorSpace::Linear),
        DxgiFormat::B5G5R5A1_UNorm => (F::B5G5R5A1_UNORM_PACK16, ColorSpace::Linear),
        DxgiFormat::B4G4R4A4_UNorm => (F::B4G4R4A4_UNORM_PACK16, ColorSpace::Linear),

        // ── BC compressed ───────────────────────────────────────────────
        DxgiFormat::BC1_UNorm => (F::BC1_RGBA_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC1_UNorm_sRGB => (F::BC1_RGBA_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::BC2_UNorm => (F::BC2_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC2_UNorm_sRGB => (F::BC2_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::BC3_UNorm => (F::BC3_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC3_UNorm_sRGB => (F::BC3_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::BC4_UNorm => (F::BC4_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC4_SNorm => (F::BC4_SNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC5_UNorm => (F::BC5_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC5_SNorm => (F::BC5_SNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC6H_UF16 => (F::BC6H_UFLOAT_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC6H_SF16 => (F::BC6H_SFLOAT_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC7_UNorm => (F::BC7_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::BC7_UNorm_sRGB => (F::BC7_UNORM_BLOCK, ColorSpace::Srgb),

        // ── ASTC compressed (NVIDIA DXGI extension) ─────────────────────
        DxgiFormat::ASTC_4x4_UNorm => (F::ASTC_4x4_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_4x4_UNorm_sRGB => (F::ASTC_4x4_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_5x4_UNorm => (F::ASTC_5x4_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_5x4_UNorm_sRGB => (F::ASTC_5x4_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_5x5_UNorm => (F::ASTC_5x5_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_5x5_UNorm_sRGB => (F::ASTC_5x5_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_6x5_UNorm => (F::ASTC_6x5_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_6x5_UNorm_sRGB => (F::ASTC_6x5_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_6x6_UNorm => (F::ASTC_6x6_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_6x6_UNorm_sRGB => (F::ASTC_6x6_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_8x5_UNorm => (F::ASTC_8x5_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_8x5_UNorm_sRGB => (F::ASTC_8x5_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_8x6_UNorm => (F::ASTC_8x6_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_8x6_UNorm_sRGB => (F::ASTC_8x6_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_8x8_UNorm => (F::ASTC_8x8_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_8x8_UNorm_sRGB => (F::ASTC_8x8_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_10x5_UNorm => (F::ASTC_10x5_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_10x5_UNorm_sRGB => (F::ASTC_10x5_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_10x6_UNorm => (F::ASTC_10x6_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_10x6_UNorm_sRGB => (F::ASTC_10x6_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_10x8_UNorm => (F::ASTC_10x8_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_10x8_UNorm_sRGB => (F::ASTC_10x8_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_10x10_UNorm => (F::ASTC_10x10_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_10x10_UNorm_sRGB => (F::ASTC_10x10_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_12x10_UNorm => (F::ASTC_12x10_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_12x10_UNorm_sRGB => (F::ASTC_12x10_UNORM_BLOCK, ColorSpace::Srgb),
        DxgiFormat::ASTC_12x12_UNorm => (F::ASTC_12x12_UNORM_BLOCK, ColorSpace::Linear),
        DxgiFormat::ASTC_12x12_UNorm_sRGB => (F::ASTC_12x12_UNORM_BLOCK, ColorSpace::Srgb),

        other => {
            return Err(Error::InputDecoding(format!(
                "DXGI format {other:?} is not supported"
            )));
        }
    };

    Ok((format, cs))
}

/// Map a legacy D3D format to a `ktx2::Format` + `ColorSpace`.
fn d3d_to_vk_format(d3d: D3DFormat) -> Result<(ktx2::Format, ColorSpace)> {
    use ktx2::Format as F;

    let (format, cs) = match d3d {
        // DXT compressed → BC
        D3DFormat::DXT1 => (F::BC1_RGBA_UNORM_BLOCK, ColorSpace::Srgb),
        D3DFormat::DXT2 | D3DFormat::DXT3 => (F::BC2_UNORM_BLOCK, ColorSpace::Srgb),
        D3DFormat::DXT4 | D3DFormat::DXT5 => (F::BC3_UNORM_BLOCK, ColorSpace::Srgb),

        // Uncompressed
        D3DFormat::A8B8G8R8 => (F::R8G8B8A8_UNORM, ColorSpace::Srgb),
        D3DFormat::A8R8G8B8 => (F::B8G8R8A8_UNORM, ColorSpace::Srgb),
        D3DFormat::X8R8G8B8 => (F::B8G8R8A8_UNORM, ColorSpace::Srgb),
        D3DFormat::X8B8G8R8 => (F::R8G8B8A8_UNORM, ColorSpace::Srgb),
        D3DFormat::A2B10G10R10 => (F::A2B10G10R10_UNORM_PACK32, ColorSpace::Linear),
        D3DFormat::R5G6B5 => (F::B5G6R5_UNORM_PACK16, ColorSpace::Srgb),
        D3DFormat::A1R5G5B5 => (F::B5G5R5A1_UNORM_PACK16, ColorSpace::Srgb),
        D3DFormat::A4R4G4B4 => (F::B4G4R4A4_UNORM_PACK16, ColorSpace::Srgb),
        D3DFormat::A8 => (F::R8_UNORM, ColorSpace::Linear),
        D3DFormat::L8 => (F::R8_UNORM, ColorSpace::Linear),
        D3DFormat::L16 => (F::R16_UNORM, ColorSpace::Linear),
        D3DFormat::A8L8 => (F::R8G8_UNORM, ColorSpace::Linear),
        D3DFormat::G16R16 => (F::R16G16_UNORM, ColorSpace::Linear),
        D3DFormat::A16B16G16R16 => (F::R16G16B16A16_UNORM, ColorSpace::Linear),
        D3DFormat::Q16W16V16U16 => (F::R16G16B16A16_SNORM, ColorSpace::Linear),
        D3DFormat::R16F => (F::R16_SFLOAT, ColorSpace::Linear),
        D3DFormat::G16R16F => (F::R16G16_SFLOAT, ColorSpace::Linear),
        D3DFormat::A16B16G16R16F => (F::R16G16B16A16_SFLOAT, ColorSpace::Linear),
        D3DFormat::R32F => (F::R32_SFLOAT, ColorSpace::Linear),
        D3DFormat::G32R32F => (F::R32G32_SFLOAT, ColorSpace::Linear),
        D3DFormat::A32B32G32R32F => (F::R32G32B32A32_SFLOAT, ColorSpace::Linear),

        other => {
            return Err(Error::InputDecoding(format!(
                "D3D format {other:?} is not supported"
            )));
        }
    };

    Ok((format, cs))
}

/// Decode a DDS file from raw bytes into an [`Image`].
///
/// Supports both DXGI (DX10 header) and legacy D3D format DDS files.
/// Reads layers, mip levels, and cubemap face data.
pub fn decode_dds_image(data: &[u8]) -> Result<Image> {
    let dds = Dds::read(&mut std::io::Cursor::new(data))
        .map_err(|e| Error::InputDecoding(format!("DDS parse failed: {e}")))?;

    let (format, color_space) = if let Some(dxgi) = dds.get_dxgi_format() {
        dxgi_to_vk_format(dxgi)?
    } else if let Some(d3d) = dds.get_d3d_format() {
        d3d_to_vk_format(d3d)?
    } else {
        return Err(Error::InputDecoding(
            "DDS file has no recognizable format".into(),
        ));
    };

    let width = dds.get_width();
    let height = dds.get_height();
    let mip_count = dds.get_num_mipmap_levels() as usize;
    let array_layers = dds.get_num_array_layers() as usize;

    let is_cubemap = dds.header.caps2.contains(Caps2::CUBEMAP);

    log::debug!(
        "DDS input: {:?}, {}x{}, {} layers, {} mips, cubemap={}",
        format,
        width,
        height,
        array_layers,
        mip_count,
        is_cubemap,
    );

    // Build surfaces[layer][mip].
    let mut surfaces: Vec<Vec<Surface>> = Vec::with_capacity(array_layers);

    for layer_idx in 0..array_layers {
        let layer_data = dds
            .get_data(layer_idx as u32)
            .map_err(|e| Error::InputDecoding(format!("DDS layer {layer_idx} data: {e}")))?;

        let mut mips = Vec::with_capacity(mip_count);
        let mut offset = 0usize;

        for mip_idx in 0..mip_count {
            let mip_w = (width >> mip_idx as u32).max(1);
            let mip_h = (height >> mip_idx as u32).max(1);

            let (mip_size, stride) = compute_mip_size_and_stride(mip_w, mip_h, format)?;

            if offset + mip_size > layer_data.len() {
                return Err(Error::InputDecoding(format!(
                    "DDS layer {layer_idx} mip {mip_idx}: expected {mip_size} bytes \
                     at offset {offset}, but layer data is only {} bytes",
                    layer_data.len(),
                )));
            }

            mips.push(Surface {
                data: layer_data[offset..offset + mip_size].to_vec(),
                width: mip_w,
                height: mip_h,
                stride,
                format,
                color_space,
                alpha: AlphaMode::Straight,
            });

            offset += mip_size;
        }

        surfaces.push(mips);
    }

    Ok(Image {
        surfaces,
        is_cubemap,
    })
}

/// Compute the byte size and stride for a single mip level.
fn compute_mip_size_and_stride(
    width: u32,
    height: u32,
    format: ktx2::Format,
) -> Result<(usize, u32)> {
    if format.is_compressed() {
        let (bw, bh) = format.block_size().unwrap();
        let bpb = format.bytes_per_block().unwrap();
        let blocks_x = width.div_ceil(bw as u32);
        let blocks_y = height.div_ceil(bh as u32);
        let stride = blocks_x * bpb as u32;
        let size = blocks_x as usize * blocks_y as usize * bpb;
        Ok((size, stride))
    } else {
        let bpp = format
            .bytes_per_pixel()
            .ok_or_else(|| Error::InputDecoding(format!("unknown bpp for {format:?}")))?;
        let stride = width * bpp as u32;
        let size = stride as usize * height as usize;
        Ok((size, stride))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::dds::encode_dds_image;

    /// Round-trip: encode an Image to DDS bytes, then decode it back.
    #[test]
    fn roundtrip_rgba8() {
        let original = Image {
            surfaces: vec![vec![Surface {
                data: vec![42u8; 4 * 4 * 4],
                width: 4,
                height: 4,
                stride: 16,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Srgb,
                alpha: AlphaMode::Straight,
            }]],
            is_cubemap: false,
        };

        let encoded = encode_dds_image(&original).unwrap();
        let decoded = decode_dds_image(&encoded).unwrap();

        assert_eq!(decoded.surfaces.len(), 1);
        assert_eq!(decoded.surfaces[0].len(), 1);
        let s = &decoded.surfaces[0][0];
        assert_eq!(s.width, 4);
        assert_eq!(s.height, 4);
        assert_eq!(s.format, ktx2::Format::R8G8B8A8_UNORM);
        assert_eq!(s.color_space, ColorSpace::Srgb);
        assert_eq!(s.data, vec![42u8; 64]);
    }

    /// Round-trip BC7 compressed.
    #[test]
    fn roundtrip_bc7() {
        let original = Image {
            surfaces: vec![vec![Surface {
                data: vec![0xFF; 16],
                width: 4,
                height: 4,
                stride: 16,
                format: ktx2::Format::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            is_cubemap: false,
        };

        let encoded = encode_dds_image(&original).unwrap();
        let decoded = decode_dds_image(&encoded).unwrap();

        assert_eq!(decoded.surfaces[0][0].format, ktx2::Format::BC7_UNORM_BLOCK);
        assert_eq!(decoded.surfaces[0][0].color_space, ColorSpace::Linear);
        assert_eq!(decoded.surfaces[0][0].data, vec![0xFF; 16]);
    }

    /// Round-trip with mip levels.
    #[test]
    fn roundtrip_mips() {
        let original = Image {
            surfaces: vec![vec![
                Surface {
                    data: vec![0xAA; 4 * 4 * 4],
                    width: 4,
                    height: 4,
                    stride: 16,
                    format: ktx2::Format::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
                Surface {
                    data: vec![0xBB; 2 * 2 * 4],
                    width: 2,
                    height: 2,
                    stride: 8,
                    format: ktx2::Format::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
                Surface {
                    data: vec![0xCC; 4],
                    width: 1,
                    height: 1,
                    stride: 4,
                    format: ktx2::Format::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
            ]],
            is_cubemap: false,
        };

        let encoded = encode_dds_image(&original).unwrap();
        let decoded = decode_dds_image(&encoded).unwrap();

        assert_eq!(decoded.surfaces.len(), 1);
        assert_eq!(decoded.surfaces[0].len(), 3);
        assert_eq!(decoded.surfaces[0][0].data, vec![0xAA; 64]);
        assert_eq!(decoded.surfaces[0][1].data, vec![0xBB; 16]);
        assert_eq!(decoded.surfaces[0][2].data, vec![0xCC; 4]);
    }

    /// Round-trip a cubemap.
    #[test]
    fn roundtrip_cubemap() {
        let faces: Vec<Vec<Surface>> = (0..6)
            .map(|i| {
                vec![Surface {
                    data: vec![i as u8; 4 * 4 * 4],
                    width: 4,
                    height: 4,
                    stride: 16,
                    format: ktx2::Format::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                }]
            })
            .collect();

        let original = Image {
            surfaces: faces,
            is_cubemap: true,
        };

        let encoded = encode_dds_image(&original).unwrap();
        let decoded = decode_dds_image(&encoded).unwrap();

        assert!(decoded.is_cubemap);
        assert_eq!(decoded.surfaces.len(), 6);
        for i in 0..6 {
            assert_eq!(decoded.surfaces[i][0].data, vec![i as u8; 64]);
        }
    }

    /// DXGI ↔ VK format inverse relationship matches output module.
    #[test]
    fn dxgi_vk_roundtrip_bc7_srgb() {
        let (fmt, cs) = dxgi_to_vk_format(DxgiFormat::BC7_UNorm_sRGB).unwrap();
        assert_eq!(fmt, ktx2::Format::BC7_UNORM_BLOCK);
        assert_eq!(cs, ColorSpace::Srgb);
    }

    #[test]
    fn dxgi_vk_roundtrip_bc1_linear() {
        let (fmt, cs) = dxgi_to_vk_format(DxgiFormat::BC1_UNorm).unwrap();
        assert_eq!(fmt, ktx2::Format::BC1_RGBA_UNORM_BLOCK);
        assert_eq!(cs, ColorSpace::Linear);
    }

    #[test]
    fn d3d_dxt1_maps_to_bc1() {
        let (fmt, _cs) = d3d_to_vk_format(D3DFormat::DXT1).unwrap();
        assert_eq!(fmt, ktx2::Format::BC1_RGBA_UNORM_BLOCK);
    }

    #[test]
    fn d3d_dxt5_maps_to_bc3() {
        let (fmt, _cs) = d3d_to_vk_format(D3DFormat::DXT5).unwrap();
        assert_eq!(fmt, ktx2::Format::BC3_UNORM_BLOCK);
    }
}
