use crate::compress::CompressedTexture;
use crate::error::{Error, Result};
use crate::format::{ColorSpace, CompressedFormat};

/// KTX2 file identifier (12 bytes).
const KTX2_MAGIC: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// Map a compressed format + color space to a VkFormat u32 value.
pub fn to_vk_format(format: CompressedFormat, color_space: ColorSpace) -> Result<u32> {
    let srgb = color_space == ColorSpace::Srgb;
    match format {
        CompressedFormat::Bc1 => Ok(if srgb { 134 } else { 133 }), // BC1_RGBA_{SRGB,UNORM}_BLOCK
        CompressedFormat::Bc3 => Ok(if srgb { 138 } else { 137 }), // BC3_{SRGB,UNORM}_BLOCK
        CompressedFormat::Bc4 => Ok(139),                          // BC4_UNORM_BLOCK
        CompressedFormat::Bc5 => Ok(141),                          // BC5_UNORM_BLOCK
        CompressedFormat::Bc6h => Ok(143),                         // BC6H_UFLOAT_BLOCK
        CompressedFormat::Bc7 => Ok(if srgb { 146 } else { 145 }), // BC7_{SRGB,UNORM}_BLOCK
        CompressedFormat::Etc1 => Ok(if srgb { 148 } else { 147 }), // ETC2_R8G8B8_{SRGB,UNORM}_BLOCK
        CompressedFormat::Astc {
            block_width,
            block_height,
        } => astc_vk_format(block_width, block_height, srgb),
    }
}

fn astc_vk_format(bw: u8, bh: u8, srgb: bool) -> Result<u32> {
    // VkFormat ASTC values: each block size has a pair (UNORM, SRGB)
    let base = match (bw, bh) {
        (4, 4) => 157,
        (5, 4) => 159,
        (5, 5) => 161,
        (6, 5) => 163,
        (6, 6) => 165,
        (8, 5) => 167,
        (8, 6) => 169,
        (8, 8) => 171,
        (10, 5) => 173,
        (10, 6) => 175,
        (10, 8) => 177,
        (10, 10) => 179,
        (12, 10) => 181,
        (12, 12) => 183,
        _ => {
            return Err(Error::UnsupportedFormat(format!(
                "unsupported ASTC block size {bw}x{bh}"
            )));
        }
    };
    Ok(if srgb { base + 1 } else { base })
}

/// Encode a compressed texture as a KTX2 file.
///
/// Writes the KTX2 binary format directly since the `ktx2` crate is read-only.
pub fn encode_ktx2(texture: &CompressedTexture) -> Result<Vec<u8>> {
    let first = &texture.layers[0][0];
    let vk_format = to_vk_format(first.format, texture.color_space)?;

    let level_count = texture.layers[0].len() as u32;
    let face_count = if texture.is_cubemap { 6u32 } else { 1u32 };
    let layer_count = if texture.is_cubemap {
        0u32 // KTX2 spec: 0 means "not an array texture"
    } else if texture.layers.len() > 1 {
        texture.layers.len() as u32
    } else {
        0u32
    };

    // Collect all level data. For cubemaps, faces are interleaved within each level.
    let mut level_data: Vec<Vec<u8>> = Vec::with_capacity(level_count as usize);
    for mip_idx in 0..level_count as usize {
        let mut mip_data = Vec::new();
        for layer in &texture.layers {
            mip_data.extend_from_slice(&layer[mip_idx].data);
        }
        level_data.push(mip_data);
    }

    // Calculate layout sizes.
    // Header: 80 bytes
    // Level index: level_count * 24 bytes
    // DFD: 0 bytes (omitted for simplicity)
    // KVD: 0 bytes
    // SGD: 0 bytes
    let header_size = 80u64;
    let level_index_size = level_count as u64 * 24;
    let dfd_byte_length = 0u32;
    let kvd_byte_length = 0u32;

    let data_start = header_size + level_index_size;

    // Build level index and compute offsets.
    let mut level_offsets: Vec<(u64, u64)> = Vec::with_capacity(level_count as usize);
    let mut current_offset = data_start;
    for level in &level_data {
        let len = level.len() as u64;
        level_offsets.push((current_offset, len));
        // Align to next multiple of `lcm(texel_block_size, 4)` — use 16 for simplicity.
        current_offset += (len + 15) & !15;
    }

    let total_size = current_offset as usize;
    let mut output = Vec::with_capacity(total_size);

    // Write header.
    output.extend_from_slice(&KTX2_MAGIC);
    output.extend_from_slice(&vk_format.to_le_bytes());
    output.extend_from_slice(&1u32.to_le_bytes()); // typeSize
    output.extend_from_slice(&first.width.to_le_bytes()); // pixelWidth
    output.extend_from_slice(&first.height.to_le_bytes()); // pixelHeight
    output.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
    output.extend_from_slice(&layer_count.to_le_bytes()); // layerCount
    output.extend_from_slice(&face_count.to_le_bytes()); // faceCount
    output.extend_from_slice(&level_count.to_le_bytes()); // levelCount
    output.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

    // DFD
    let dfd_byte_offset = if dfd_byte_length > 0 {
        data_start as u32
    } else {
        0
    };
    output.extend_from_slice(&dfd_byte_offset.to_le_bytes());
    output.extend_from_slice(&dfd_byte_length.to_le_bytes());

    // KVD
    output.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
    output.extend_from_slice(&kvd_byte_length.to_le_bytes());

    // SGD
    output.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
    output.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

    assert_eq!(output.len(), header_size as usize);

    // Write level index.
    for (offset, len) in &level_offsets {
        output.extend_from_slice(&offset.to_le_bytes()); // byteOffset
        output.extend_from_slice(&len.to_le_bytes()); // byteLength
        output.extend_from_slice(&len.to_le_bytes()); // uncompressedByteLength (no supercompression)
    }

    assert_eq!(output.len(), data_start as usize);

    // Write level data with alignment padding.
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

    #[test]
    fn bc1_srgb_vk_format() {
        assert_eq!(
            to_vk_format(CompressedFormat::Bc1, ColorSpace::Srgb).unwrap(),
            134
        );
    }

    #[test]
    fn bc7_linear_vk_format() {
        assert_eq!(
            to_vk_format(CompressedFormat::Bc7, ColorSpace::Linear).unwrap(),
            145
        );
    }

    #[test]
    fn astc_4x4_srgb_vk_format() {
        assert_eq!(
            to_vk_format(
                CompressedFormat::Astc {
                    block_width: 4,
                    block_height: 4
                },
                ColorSpace::Srgb
            )
            .unwrap(),
            158
        );
    }

    #[test]
    fn etc1_maps_to_etc2_rgb() {
        assert_eq!(
            to_vk_format(CompressedFormat::Etc1, ColorSpace::Linear).unwrap(),
            147
        );
    }

    #[test]
    fn unsupported_astc_block_size() {
        assert!(
            to_vk_format(
                CompressedFormat::Astc {
                    block_width: 3,
                    block_height: 3
                },
                ColorSpace::Srgb
            )
            .is_err()
        );
    }
}
