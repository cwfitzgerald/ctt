use ktx2::dfd;

use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::surface::Image;
use crate::vk_format::FormatExt as _;

/// Encode an [`Image`] as a KTX2 file.
///
/// Uses the new `ktx2` crate APIs for header serialization, DFD generation,
/// and level index construction.
pub fn encode_ktx2_image(image: &Image) -> Result<Vec<u8>> {
    let first = &image.surfaces[0][0];
    let vk_format = first.format.denormalize(first.color_space);

    let alpha_premultiplied = first.alpha == AlphaMode::Premultiplied;

    let (basic_dfd, type_size) =
        dfd::Basic::from_format_with(vk_format, alpha_premultiplied, None, None, None).map_err(
            |e| Error::OutputEncoding(format!("DFD generation failed for {vk_format:?}: {e}")),
        )?;

    log::debug!(
        "KTX2: vk_format={} ({:?}), {} layers, {} mips, alpha={:?}",
        vk_format.value(),
        vk_format,
        image.surfaces.len(),
        image.surfaces[0].len(),
        first.alpha,
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

    // bytes_planes[0] is the texel block size: bytes per block for compressed
    // formats, bytes per pixel for uncompressed.
    let texel_block_size = basic_dfd.bytes_planes[0] as u32;

    // Build DFD section bytes: 4-byte total size + block data.
    let dfd_block = dfd::Block::Basic(basic_dfd);
    let dfd_block_bytes = dfd_block.to_vec();
    let dfd_total_size = 4 + dfd_block_bytes.len();

    // Build KVD section: KTXwriter key.
    let kvd_bytes = build_kvd();

    // Layout offsets.
    let level_index_size = level_count as usize * ktx2::LevelIndex::LENGTH;
    let dfd_offset = ktx2::Header::LENGTH + level_index_size;
    let kvd_offset = dfd_offset + dfd_total_size;
    let after_kvd = kvd_offset + kvd_bytes.len();

    // Level data must be aligned to lcm(texel_block_size, 4) per the spec.
    let alignment = lcm(texel_block_size, 4) as usize;
    let data_start = align_up(after_kvd, alignment);

    // KTX2 spec requires level data stored smallest-to-largest (smallest mip
    // at the lowest file offset). The level index is still ordered 0..N-1
    // (largest first), but the byte_offsets point into the reversed layout.
    let mut level_indices: Vec<ktx2::LevelIndex> = vec![
        ktx2::LevelIndex {
            byte_offset: 0,
            byte_length: 0,
            uncompressed_byte_length: 0,
        };
        level_count as usize
    ];
    let mut current_offset = data_start;
    for i in (0..level_count as usize).rev() {
        let len = level_data[i].len();
        level_indices[i] = ktx2::LevelIndex {
            byte_offset: current_offset as u64,
            byte_length: len as u64,
            uncompressed_byte_length: len as u64,
        };
        current_offset = align_up(current_offset + len, alignment);
    }

    let total_size = if level_data.is_empty() {
        data_start
    } else {
        // Level 0 (largest) is last in the file; no trailing alignment needed.
        let idx = &level_indices[0];
        idx.byte_offset as usize + idx.byte_length as usize
    };

    // Build header.
    let header = ktx2::Header {
        format: Some(vk_format),
        type_size,
        pixel_width: first.width,
        pixel_height: first.height,
        pixel_depth: 0,
        layer_count,
        face_count,
        level_count,
        supercompression_scheme: None,
        index: ktx2::Index {
            dfd_byte_offset: dfd_offset as u32,
            dfd_byte_length: dfd_total_size as u32,
            kvd_byte_offset: kvd_offset as u32,
            kvd_byte_length: kvd_bytes.len() as u32,
            sgd_byte_offset: 0,
            sgd_byte_length: 0,
        },
    };

    let mut output = vec![0u8; total_size];

    // Write header.
    output[..ktx2::Header::LENGTH].copy_from_slice(&header.as_bytes());

    // Write level index.
    for (i, idx) in level_indices.iter().enumerate() {
        let offset = ktx2::Header::LENGTH + i * ktx2::LevelIndex::LENGTH;
        output[offset..offset + ktx2::LevelIndex::LENGTH].copy_from_slice(&idx.as_bytes());
    }

    // Write DFD section.
    output[dfd_offset..dfd_offset + 4].copy_from_slice(&(dfd_total_size as u32).to_le_bytes());
    output[dfd_offset + 4..dfd_offset + 4 + dfd_block_bytes.len()]
        .copy_from_slice(&dfd_block_bytes);

    // Write KVD section.
    output[kvd_offset..kvd_offset + kvd_bytes.len()].copy_from_slice(&kvd_bytes);

    // Write level data.
    for (i, level) in level_data.iter().enumerate() {
        let offset = level_indices[i].byte_offset as usize;
        output[offset..offset + level.len()].copy_from_slice(level);
    }

    Ok(output)
}

/// Build the Key/Value Data section containing a KTXwriter entry.
fn build_kvd() -> Vec<u8> {
    let key = b"KTXwriter";
    let ctt_version = env!("CARGO_PKG_VERSION");
    let value = format!("ctt {ctt_version} / ktx2 crate 0.5.0");
    let value_bytes = value.as_bytes();

    // key + NUL + value + NUL
    let kv_len = key.len() + 1 + value_bytes.len() + 1;

    // 4-byte length prefix + key/value data, padded to 4-byte alignment.
    let entry_size = 4 + align_up(kv_len, 4);
    let mut kvd = vec![0u8; entry_size];

    kvd[0..4].copy_from_slice(&(kv_len as u32).to_le_bytes());
    kvd[4..4 + key.len()].copy_from_slice(key);
    // NUL separator at kvd[4 + key.len()] is already 0.
    let value_start = 4 + key.len() + 1;
    kvd[value_start..value_start + value_bytes.len()].copy_from_slice(value_bytes);
    // Trailing NUL and padding are already 0.

    kvd
}

fn lcm(a: u32, b: u32) -> u32 {
    a / gcd(a, b) * b
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::{ColorSpace, Surface};
    use ktx2::Format as F;

    fn make_test_image(format: F, color_space: ColorSpace, alpha: AlphaMode) -> Image {
        let bpp = format.bytes_per_pixel().unwrap_or(16);
        let data = vec![0u8; 4 * 4 * bpp];
        Image {
            surfaces: vec![vec![Surface {
                data,
                width: 4,
                height: 4,
                stride: 4 * bpp as u32,
                format,
                color_space,
                alpha,
            }]],
            is_cubemap: false,
        }
    }

    #[test]
    fn roundtrip_rgba8_srgb() {
        let image = make_test_image(F::R8G8B8A8_UNORM, ColorSpace::Srgb, AlphaMode::Straight);
        let bytes = encode_ktx2_image(&image).unwrap();
        let reader = ktx2::Reader::new(&bytes[..]).expect("valid KTX2");
        let header = reader.header();
        assert_eq!(header.format, Some(F::R8G8B8A8_SRGB));
        assert_eq!(header.pixel_width, 4);
        assert_eq!(header.pixel_height, 4);
        assert_eq!(header.level_count, 1);

        // Verify DFD exists and has correct transfer function.
        let basic = reader.basic_dfd().expect("should have basic DFD");
        assert_eq!(basic.transfer_function, Some(ktx2::TransferFunction::SRGB));

        // Verify KTXwriter.
        let writer = reader.writer().expect("should have KTXwriter");
        assert!(writer.starts_with("ctt "));
    }

    #[test]
    fn roundtrip_bc7_linear() {
        let image = Image {
            surfaces: vec![vec![Surface {
                data: vec![0u8; 16], // 1 block = 16 bytes for 4x4 BC7
                width: 4,
                height: 4,
                stride: 16,
                format: F::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            is_cubemap: false,
        };
        let bytes = encode_ktx2_image(&image).unwrap();
        let reader = ktx2::Reader::new(&bytes[..]).expect("valid KTX2");
        let header = reader.header();
        assert_eq!(header.format, Some(F::BC7_UNORM_BLOCK));

        let basic = reader.basic_dfd().expect("should have basic DFD");
        assert_eq!(
            basic.transfer_function,
            Some(ktx2::TransferFunction::Linear)
        );
    }

    #[test]
    fn premultiplied_alpha_flag() {
        let image = make_test_image(
            F::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Premultiplied,
        );
        let bytes = encode_ktx2_image(&image).unwrap();
        let reader = ktx2::Reader::new(&bytes[..]).expect("valid KTX2");
        assert_eq!(reader.is_alpha_premultiplied(), Some(true));
    }

    #[test]
    fn straight_alpha_flag() {
        let image = make_test_image(F::R8G8B8A8_UNORM, ColorSpace::Linear, AlphaMode::Straight);
        let bytes = encode_ktx2_image(&image).unwrap();
        let reader = ktx2::Reader::new(&bytes[..]).expect("valid KTX2");
        assert_eq!(reader.is_alpha_premultiplied(), Some(false));
    }

    #[test]
    fn level_data_preserved() {
        let data = vec![42u8; 4 * 4 * 4]; // 4x4 RGBA8
        let image = Image {
            surfaces: vec![vec![Surface {
                data: data.clone(),
                width: 4,
                height: 4,
                stride: 16,
                format: F::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            is_cubemap: false,
        };
        let bytes = encode_ktx2_image(&image).unwrap();
        let reader = ktx2::Reader::new(&bytes[..]).expect("valid KTX2");
        let levels: Vec<_> = reader.levels().collect();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].data, &data[..]);
    }

    #[test]
    fn mip_levels_stored_smallest_first() {
        // 4x4, 2x2, 1x1 — three mip levels of RGBA8
        let mip0 = vec![0xAAu8; 4 * 4 * 4];
        let mip1 = vec![0xBBu8; 2 * 2 * 4];
        let mip2 = vec![0xCCu8; 4];
        let image = Image {
            surfaces: vec![vec![
                Surface {
                    data: mip0.clone(),
                    width: 4,
                    height: 4,
                    stride: 16,
                    format: F::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
                Surface {
                    data: mip1.clone(),
                    width: 2,
                    height: 2,
                    stride: 8,
                    format: F::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
                Surface {
                    data: mip2.clone(),
                    width: 1,
                    height: 1,
                    stride: 4,
                    format: F::R8G8B8A8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                },
            ]],
            is_cubemap: false,
        };
        let bytes = encode_ktx2_image(&image).unwrap();
        let reader = ktx2::Reader::new(&bytes[..]).expect("valid KTX2");
        assert_eq!(reader.header().level_count, 3);

        let levels: Vec<_> = reader.levels().collect();
        assert_eq!(levels[0].data, &mip0[..]);
        assert_eq!(levels[1].data, &mip1[..]);
        assert_eq!(levels[2].data, &mip2[..]);

        // Verify file layout: smallest mip (level 2) at lowest offset,
        // largest mip (level 0) at highest offset.
        let header = reader.header();
        let idx: Vec<_> = (0..3)
            .map(|i| {
                let off = ktx2::Header::LENGTH + i * ktx2::LevelIndex::LENGTH;
                ktx2::LevelIndex::from_bytes(&bytes[off..off + 24].try_into().unwrap())
            })
            .collect();
        assert!(
            idx[2].byte_offset < idx[1].byte_offset,
            "level 2 (1x1) should be before level 1 (2x2) in file, but {} >= {}",
            idx[2].byte_offset,
            idx[1].byte_offset,
        );
        assert!(
            idx[1].byte_offset < idx[0].byte_offset,
            "level 1 (2x2) should be before level 0 (4x4) in file, but {} >= {}",
            idx[1].byte_offset,
            idx[0].byte_offset,
        );
        let _ = header;
    }
}
