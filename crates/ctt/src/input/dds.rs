use ddsfile::{Caps2, D3D10ResourceDimension, D3DFormat, Dds, DxgiFormat, MiscFlag};

use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::surface::{ColorSpace, Image, Surface, TextureKind};
use crate::vk_format::FormatExt as _;

const MAX_SURFACE_COUNT: usize = 1 << 20;

#[derive(Debug, Clone, Copy)]
struct MipLayout {
    width: u32,
    height: u32,
    depth: u32,
    total_size: usize,
    stride: u32,
    slice_stride: u32,
}

fn checked_array_layers(dds: &Dds, is_cubemap: bool) -> Result<usize> {
    let layers = if let Some(header10) = &dds.header10 {
        if header10.array_size == 0 {
            return Err(Error::InputDecoding(
                "DDS DX10 array size must be at least 1".into(),
            ));
        }
        if is_cubemap {
            header10.array_size.checked_mul(6).ok_or_else(|| {
                Error::InputDecoding("DDS cubemap array layer count overflow".into())
            })?
        } else {
            header10.array_size
        }
    } else if is_cubemap {
        6
    } else {
        1
    };
    let layers = usize::try_from(layers).map_err(|_| {
        Error::InputDecoding("DDS array layer count does not fit this platform".into())
    })?;
    if layers > MAX_SURFACE_COUNT {
        return Err(Error::InputDecoding(format!(
            "DDS declares {layers} surfaces; the supported maximum is {MAX_SURFACE_COUNT}"
        )));
    }
    Ok(layers)
}

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
        // DXGI packed names are LSB-first, VK `_PACKnn` names are MSB-first, so
        // the component order reverses. DXGI B5G6R5 has red in bits 15:11 and
        // blue in 4:0 — the same word layout as VK R5G6B5_PACK16.
        DxgiFormat::B5G6R5_UNorm => (F::R5G6B5_UNORM_PACK16, ColorSpace::Linear),
        // DXGI B5G5R5A1 has alpha in bit 15 down to blue in 4:0 — VK A1R5G5B5_PACK16.
        DxgiFormat::B5G5R5A1_UNorm => (F::A1R5G5B5_UNORM_PACK16, ColorSpace::Linear),
        // DXGI B4G4R4A4 has alpha in bits 15:12 down to blue in 3:0 — VK A4R4G4B4_PACK16.
        DxgiFormat::B4G4R4A4_UNorm => (F::A4R4G4B4_UNORM_PACK16, ColorSpace::Linear),

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
        // Legacy D3D9 names are MSB-first (like VK `_PACKnn`), so they map
        // directly without reversal: D3DFMT_R5G6B5 == VK R5G6B5_PACK16,
        // D3DFMT_A1R5G5B5 == VK A1R5G5B5_PACK16, etc.
        D3DFormat::R5G6B5 => (F::R5G6B5_UNORM_PACK16, ColorSpace::Srgb),
        D3DFormat::A1R5G5B5 => (F::A1R5G5B5_UNORM_PACK16, ColorSpace::Srgb),
        D3DFormat::A4R4G4B4 => (F::A4R4G4B4_UNORM_PACK16, ColorSpace::Srgb),
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
    let mip_count = dds.get_num_mipmap_levels().max(1) as usize;
    let depth = dds.get_depth().max(1);

    if width == 0 || height == 0 {
        return Err(Error::InputDecoding(
            "DDS width and height must both be at least 1".into(),
        ));
    }

    // Reject a mip count larger than the full chain the dimensions can produce.
    // This both rejects malformed files and keeps the `>> mip_idx` shifts below
    // within range (a count >= 32 would otherwise shift-overflow-panic in debug).
    let max_dim = width.max(height).max(depth).max(1);
    let max_levels = (32 - max_dim.leading_zeros()) as usize;
    if mip_count > max_levels {
        return Err(Error::InputDecoding(format!(
            "DDS declares {mip_count} mip levels but dimensions {width}x{height}x{depth} \
             allow at most {max_levels}"
        )));
    }

    // A DX10 cubemap is flagged via the header10 misc flag; ddsfile does not
    // back-fill the legacy Caps2::CUBEMAP bit for it, so check both.
    let is_cubemap = dds.header.caps2.contains(Caps2::CUBEMAP)
        || dds
            .header10
            .as_ref()
            .is_some_and(|h| h.misc_flag.contains(MiscFlag::TEXTURECUBE));
    let is_3d = matches!(
        dds.header10.as_ref().map(|h| h.resource_dimension),
        Some(D3D10ResourceDimension::Texture3D),
    ) || depth > 1;
    let array_layers = checked_array_layers(&dds, is_cubemap)?;

    if is_3d && (is_cubemap || array_layers > 1) {
        return Err(Error::InputDecoding(
            "DDS 3D textures cannot be combined with cubemap faces or array layers".into(),
        ));
    }

    let kind = if is_cubemap {
        TextureKind::Cubemap
    } else if is_3d {
        TextureKind::Texture3D
    } else {
        TextureKind::Texture2D
    };

    let mut layouts = Vec::new();
    layouts.try_reserve_exact(mip_count).map_err(|e| {
        Error::InputDecoding(format!("DDS mip count {mip_count} is too large: {e}"))
    })?;
    let mut layer_size = 0usize;
    for mip_idx in 0..mip_count {
        let mip_w = (width >> mip_idx as u32).max(1);
        let mip_h = (height >> mip_idx as u32).max(1);
        let mip_d = (depth >> mip_idx as u32).max(1);
        let (slice_size, stride) = compute_mip_size_and_stride(mip_w, mip_h, format)?;
        let total_size = slice_size.checked_mul(mip_d as usize).ok_or_else(|| {
            Error::InputDecoding(format!("DDS mip {mip_idx}: 3D byte size overflow"))
        })?;
        let slice_stride = if is_3d {
            u32::try_from(slice_size).map_err(|_| {
                Error::InputDecoding(format!(
                    "DDS mip {mip_idx}: slice size {slice_size} does not fit u32"
                ))
            })?
        } else {
            0
        };
        layer_size = layer_size
            .checked_add(total_size)
            .ok_or_else(|| Error::InputDecoding("DDS full mip-chain byte size overflow".into()))?;
        layouts.push(MipLayout {
            width: mip_w,
            height: mip_h,
            depth: mip_d,
            total_size,
            stride,
            slice_stride,
        });
    }
    let required_data = layer_size
        .checked_mul(array_layers)
        .ok_or_else(|| Error::InputDecoding("DDS total array byte size overflow".into()))?;
    if required_data > dds.data.len() {
        return Err(Error::InputDecoding(format!(
            "DDS declares {array_layers} layers requiring at least {required_data} bytes, but \
             contains only {} bytes",
            dds.data.len()
        )));
    }

    log::debug!(
        "DDS input: {:?}, {}x{}x{}, {} layers, {} mips, kind={:?}",
        format,
        width,
        height,
        depth,
        array_layers,
        mip_count,
        kind,
    );

    // Build surfaces[layer][mip].
    let mut surfaces: Vec<Vec<Surface>> = Vec::new();
    surfaces.try_reserve_exact(array_layers).map_err(|e| {
        Error::InputDecoding(format!(
            "DDS layer count {array_layers} is too large to allocate: {e}"
        ))
    })?;

    for layer_idx in 0..array_layers {
        let layer_offset = layer_idx
            .checked_mul(layer_size)
            .ok_or_else(|| Error::InputDecoding("DDS layer offset overflow".into()))?;
        let layer_end = layer_offset
            .checked_add(layer_size)
            .ok_or_else(|| Error::InputDecoding("DDS layer end offset overflow".into()))?;
        let layer_data = &dds.data[layer_offset..layer_end];

        let mut mips = Vec::new();
        mips.try_reserve_exact(mip_count).map_err(|e| {
            Error::InputDecoding(format!("DDS mip count {mip_count} is too large: {e}"))
        })?;
        let mut offset = 0usize;

        for layout in &layouts {
            let end = offset
                .checked_add(layout.total_size)
                .ok_or_else(|| Error::InputDecoding("DDS mip end offset overflow".into()))?;

            mips.push(Surface {
                data: layer_data[offset..end].to_vec(),
                width: layout.width,
                height: layout.height,
                depth: layout.depth,
                stride: layout.stride,
                slice_stride: layout.slice_stride,
                format,
                color_space,
                alpha: AlphaMode::Straight,
            });

            offset = end;
        }

        surfaces.push(mips);
    }

    Ok(Image { surfaces, kind })
}

/// Compute the byte size and stride for a single mip level.
///
/// Uses checked arithmetic throughout: attacker-controlled dimensions could
/// otherwise overflow `u32` (panic in debug, silent wrap in release before the
/// caller's bounds checks).
fn compute_mip_size_and_stride(
    width: u32,
    height: u32,
    format: ktx2::Format,
) -> Result<(usize, u32)> {
    let overflow = || Error::InputDecoding(format!("DDS mip size overflow for {format:?}"));

    if format.is_compressed() {
        let (bw, bh) = format.block_size().unwrap();
        let bpb = format.bytes_per_block().unwrap();
        let blocks_x = width.div_ceil(bw as u32);
        let blocks_y = height.div_ceil(bh as u32);
        let stride = blocks_x.checked_mul(bpb as u32).ok_or_else(overflow)?;
        let size = (blocks_x as usize)
            .checked_mul(blocks_y as usize)
            .and_then(|v| v.checked_mul(bpb))
            .ok_or_else(overflow)?;
        Ok((size, stride))
    } else {
        let bpp = format
            .bytes_per_pixel()
            .ok_or_else(|| Error::InputDecoding(format!("unknown bpp for {format:?}")))?;
        let stride = width.checked_mul(bpp as u32).ok_or_else(overflow)?;
        let size = (stride as usize)
            .checked_mul(height as usize)
            .ok_or_else(overflow)?;
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
                depth: 1,
                stride: 16,
                slice_stride: 0,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Srgb,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture2D,
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
                depth: 1,
                stride: 16,
                slice_stride: 0,
                format: ktx2::Format::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture2D,
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
                    depth: 1,
                    stride: 16,
                    slice_stride: 0,
                    format: ktx2::Format::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
                Surface {
                    data: vec![0xBB; 2 * 2 * 4],
                    width: 2,
                    height: 2,
                    depth: 1,
                    stride: 8,
                    slice_stride: 0,
                    format: ktx2::Format::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
                Surface {
                    data: vec![0xCC; 4],
                    width: 1,
                    height: 1,
                    depth: 1,
                    stride: 4,
                    slice_stride: 0,
                    format: ktx2::Format::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
            ]],
            kind: TextureKind::Texture2D,
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
                    depth: 1,
                    stride: 16,
                    slice_stride: 0,
                    format: ktx2::Format::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                }]
            })
            .collect();

        let original = Image {
            surfaces: faces,
            kind: TextureKind::Cubemap,
        };

        let encoded = encode_dds_image(&original).unwrap();
        let decoded = decode_dds_image(&encoded).unwrap();

        assert_eq!(decoded.kind, TextureKind::Cubemap);
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

    /// DXGI 16-bit packed formats must reverse component order to VK `_PACK16`.
    #[test]
    fn dxgi_packed_formats_component_order() {
        assert_eq!(
            dxgi_to_vk_format(DxgiFormat::B5G6R5_UNorm).unwrap().0,
            ktx2::Format::R5G6B5_UNORM_PACK16
        );
        assert_eq!(
            dxgi_to_vk_format(DxgiFormat::B5G5R5A1_UNorm).unwrap().0,
            ktx2::Format::A1R5G5B5_UNORM_PACK16
        );
        assert_eq!(
            dxgi_to_vk_format(DxgiFormat::B4G4R4A4_UNorm).unwrap().0,
            ktx2::Format::A4R4G4B4_UNORM_PACK16
        );
    }

    /// Legacy D3D9 16-bit packed names are MSB-first and map directly to VK.
    #[test]
    fn d3d_packed_formats_component_order() {
        assert_eq!(
            d3d_to_vk_format(D3DFormat::R5G6B5).unwrap().0,
            ktx2::Format::R5G6B5_UNORM_PACK16
        );
        assert_eq!(
            d3d_to_vk_format(D3DFormat::A1R5G5B5).unwrap().0,
            ktx2::Format::A1R5G5B5_UNORM_PACK16
        );
        assert_eq!(
            d3d_to_vk_format(D3DFormat::A4R4G4B4).unwrap().0,
            ktx2::Format::A4R4G4B4_UNORM_PACK16
        );
    }

    /// Read (DXGI→VK) then write (VK→DXGI) reproduces the original DXGI format.
    #[test]
    fn packed_format_dxgi_roundtrip() {
        for dxgi in [
            DxgiFormat::B5G6R5_UNorm,
            DxgiFormat::B5G5R5A1_UNorm,
            DxgiFormat::B4G4R4A4_UNorm,
        ] {
            let (vk, cs) = dxgi_to_vk_format(dxgi).unwrap();
            let back = crate::output::dds::vk_format_to_dxgi(vk, cs).unwrap();
            assert_eq!(back, dxgi, "roundtrip failed for {dxgi:?}");
        }
    }

    /// A DX10 cubemap flagged only via the header10 misc flag (no legacy
    /// Caps2::CUBEMAP bit) must still be classified as a cubemap.
    #[test]
    fn dx10_cubemap_without_caps2_is_cubemap() {
        let dds = Dds::new_dxgi(ddsfile::NewDxgiParams {
            height: 4,
            width: 4,
            depth: None,
            format: DxgiFormat::R8G8B8A8_UNorm,
            mipmap_levels: Some(1),
            array_layers: Some(6),
            caps2: None,      // deliberately omit the legacy cubemap bit
            is_cubemap: true, // sets header10.misc_flag = TEXTURECUBE
            resource_dimension: D3D10ResourceDimension::Texture2D,
            alpha_mode: ddsfile::AlphaMode::Unknown,
        })
        .unwrap();

        // Precondition: the legacy bit is absent, but the DX10 flag is present.
        assert!(!dds.header.caps2.contains(Caps2::CUBEMAP));
        assert!(
            dds.header10
                .as_ref()
                .unwrap()
                .misc_flag
                .contains(MiscFlag::TEXTURECUBE)
        );

        let mut bytes = Vec::new();
        dds.write(&mut bytes).unwrap();

        let img = decode_dds_image(&bytes).unwrap();
        assert_eq!(img.kind, TextureKind::Cubemap);
        assert_eq!(img.surfaces.len(), 6);
    }

    /// A mip count larger than the dimensions allow must error, not shift-panic.
    #[test]
    fn absurd_mip_count_rejected() {
        // 4x4 supports at most 3 mip levels; claim 40. Building with 40 sets the
        // MIPMAPCOUNT header flag so the count survives the write/read round-trip.
        let dds = Dds::new_dxgi(ddsfile::NewDxgiParams {
            height: 4,
            width: 4,
            depth: None,
            format: DxgiFormat::R8G8B8A8_UNorm,
            mipmap_levels: Some(40),
            array_layers: Some(1),
            caps2: None,
            is_cubemap: false,
            resource_dimension: D3D10ResourceDimension::Texture2D,
            alpha_mode: ddsfile::AlphaMode::Unknown,
        })
        .unwrap();

        let mut bytes = Vec::new();
        dds.write(&mut bytes).unwrap();

        let err = decode_dds_image(&bytes).unwrap_err();
        assert!(
            matches!(err, Error::InputDecoding(_)),
            "expected InputDecoding, got {err:?}"
        );
    }

    /// Huge dimensions must produce a clean error from the size helper rather
    /// than overflowing (panic in debug, silent wrap in release).
    #[test]
    fn huge_dimensions_error_not_overflow() {
        // Uncompressed: width * bpp overflows u32.
        let r = compute_mip_size_and_stride(u32::MAX, 1, ktx2::Format::R32G32B32A32_SFLOAT);
        assert!(matches!(r, Err(Error::InputDecoding(_))), "got {r:?}");

        // Compressed: block count * bytes-per-block overflows u32.
        let r = compute_mip_size_and_stride(u32::MAX, u32::MAX, ktx2::Format::BC7_UNORM_BLOCK);
        assert!(matches!(r, Err(Error::InputDecoding(_))), "got {r:?}");
    }

    #[test]
    fn cubemap_array_layer_overflow_rejected() {
        let mut dds = Dds::new_dxgi(ddsfile::NewDxgiParams {
            height: 1,
            width: 1,
            depth: None,
            format: DxgiFormat::R8_UNorm,
            mipmap_levels: Some(1),
            array_layers: Some(6),
            caps2: None,
            is_cubemap: true,
            resource_dimension: D3D10ResourceDimension::Texture2D,
            alpha_mode: ddsfile::AlphaMode::Unknown,
        })
        .unwrap();
        dds.header10.as_mut().unwrap().array_size = u32::MAX;

        let err = checked_array_layers(&dds, true).unwrap_err();
        assert!(err.to_string().contains("overflow"), "got: {err}");
    }

    #[test]
    fn volume_total_size_overflow_returns_error() {
        let dds = Dds::new_dxgi(ddsfile::NewDxgiParams {
            height: 1,
            width: 1,
            depth: Some(4),
            format: DxgiFormat::R8_UNorm,
            mipmap_levels: Some(1),
            array_layers: Some(1),
            caps2: None,
            is_cubemap: false,
            resource_dimension: D3D10ResourceDimension::Texture3D,
            alpha_mode: ddsfile::AlphaMode::Unknown,
        })
        .unwrap();
        let mut bytes = Vec::new();
        dds.write(&mut bytes).unwrap();
        bytes[12..16].copy_from_slice(&(1u32 << 31).to_le_bytes());
        bytes[16..20].copy_from_slice(&(1u32 << 31).to_le_bytes());
        bytes[24..28].copy_from_slice(&4u32.to_le_bytes());

        let result = std::panic::catch_unwind(|| decode_dds_image(&bytes));
        assert!(result.is_ok(), "malformed DDS must not panic");
        let err = result.unwrap().unwrap_err();
        assert!(err.to_string().contains("overflow"), "got: {err}");
    }
}
