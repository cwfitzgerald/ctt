use crate::vk_format::FormatExt as _;

/// KTX2 file identifier (12 bytes).
const KTX2_MAGIC: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// Encode an [`Image`](crate::surface::Image) as a KTX2 file.
///
/// Uses the `ktx2::Format` from the first surface directly as the VkFormat value,
/// recombined with the surface's color space via [`FormatExt::denormalize`](crate::vk_format::FormatExt::denormalize).
pub fn encode_ktx2_image(image: &crate::surface::Image) -> crate::error::Result<Vec<u8>> {
    let first = &image.surfaces[0][0];
    let vk_format = first.format.denormalize(first.color_space);
    let vk_format_value = vk_format.value();

    log::debug!(
        "KTX2: vk_format={} ({:?}), {} layers, {} mips",
        vk_format_value,
        vk_format,
        image.surfaces.len(),
        image.surfaces[0].len()
    );

    let level_count = image.surfaces[0].len() as u32;
    let face_count = if image.is_cubemap { 6u32 } else { 1u32 };
    let layer_count = if image.is_cubemap {
        0u32
    } else if image.surfaces.len() > 1 {
        image.surfaces.len() as u32
    } else {
        0u32
    };

    // Collect all level data.
    let mut level_data: Vec<Vec<u8>> = Vec::with_capacity(level_count as usize);
    for mip_idx in 0..level_count as usize {
        let mut mip_data = Vec::new();
        for layer in &image.surfaces {
            mip_data.extend_from_slice(&layer[mip_idx].data);
        }
        level_data.push(mip_data);
    }

    let header_size = 80u64;
    let level_index_size = level_count as u64 * 24;
    let dfd_byte_length = 0u32;
    let kvd_byte_length = 0u32;
    let data_start = header_size + level_index_size;

    let mut level_offsets: Vec<(u64, u64)> = Vec::with_capacity(level_count as usize);
    let mut current_offset = data_start;
    for level in &level_data {
        let len = level.len() as u64;
        level_offsets.push((current_offset, len));
        current_offset += (len + 15) & !15;
    }

    let total_size = current_offset as usize;
    let mut output = Vec::with_capacity(total_size);

    // Header
    output.extend_from_slice(&KTX2_MAGIC);
    output.extend_from_slice(&vk_format_value.to_le_bytes());
    output.extend_from_slice(&1u32.to_le_bytes()); // typeSize
    output.extend_from_slice(&first.width.to_le_bytes());
    output.extend_from_slice(&first.height.to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
    output.extend_from_slice(&layer_count.to_le_bytes());
    output.extend_from_slice(&face_count.to_le_bytes());
    output.extend_from_slice(&level_count.to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

    let dfd_byte_offset = if dfd_byte_length > 0 {
        data_start as u32
    } else {
        0
    };
    output.extend_from_slice(&dfd_byte_offset.to_le_bytes());
    output.extend_from_slice(&dfd_byte_length.to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
    output.extend_from_slice(&kvd_byte_length.to_le_bytes());
    output.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
    output.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

    assert_eq!(output.len(), header_size as usize);

    // Level index
    for (offset, len) in &level_offsets {
        output.extend_from_slice(&offset.to_le_bytes());
        output.extend_from_slice(&len.to_le_bytes());
        output.extend_from_slice(&len.to_le_bytes());
    }

    assert_eq!(output.len(), data_start as usize);

    // Level data
    for (i, level) in level_data.iter().enumerate() {
        output.extend_from_slice(level);
        let aligned_len = (level.len() + 15) & !15;
        let padding = aligned_len - level.len();
        if padding > 0 && i + 1 < level_data.len() {
            output.extend(std::iter::repeat_n(0u8, padding));
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::ColorSpace;
    use ktx2::Format as F;

    #[test]
    fn bc1_srgb_vk_format() {
        // BC1 sRGB should produce VkFormat 134 via denormalize
        let fmt = F::BC1_RGBA_UNORM_BLOCK;
        let full = fmt.denormalize(ColorSpace::Srgb);
        assert_eq!(full.value(), 134);
    }

    #[test]
    fn bc7_linear_vk_format() {
        let fmt = F::BC7_UNORM_BLOCK;
        let full = fmt.denormalize(ColorSpace::Linear);
        assert_eq!(full.value(), 145);
    }

    #[test]
    fn astc_4x4_srgb_vk_format() {
        let fmt = F::ASTC_4x4_UNORM_BLOCK;
        let full = fmt.denormalize(ColorSpace::Srgb);
        assert_eq!(full.value(), 158);
    }

    #[test]
    fn etc1_maps_to_etc2_rgb() {
        let fmt = F::ETC2_R8G8B8_UNORM_BLOCK;
        let full = fmt.denormalize(ColorSpace::Linear);
        assert_eq!(full.value(), 147);
    }
}
