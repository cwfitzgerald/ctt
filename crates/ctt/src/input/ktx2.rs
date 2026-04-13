use std::borrow::Cow;

use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::surface::{Image, Surface};
use crate::vk_format::FormatExt as _;

/// Decode a KTX2 file from raw bytes into an [`Image`].
///
/// Reads format, color space, alpha mode, layers, faces, and all mip levels
/// from the container. Errors on supercompressed files or missing format info.
pub fn decode_ktx2_image(data: &[u8]) -> Result<Image> {
    let reader = ktx2::Reader::new(data)
        .map_err(|e| Error::InputDecoding(format!("KTX2 parse failed: {e}")))?;

    let header = reader.header();

    let supercompression = header.supercompression_scheme;

    let full_format = header.format.ok_or_else(|| {
        Error::InputDecoding("KTX2 has VK_FORMAT_UNDEFINED (Basis Universal); not supported".into())
    })?;

    let (format, color_space) = full_format.normalize();

    let alpha = match reader.is_alpha_premultiplied() {
        Some(true) => AlphaMode::Premultiplied,
        _ => AlphaMode::Straight,
    };

    let is_cubemap = header.face_count == 6;
    let face_count = header.face_count as usize;
    let layer_count = header.layer_count.max(1) as usize;
    let level_count = header.level_count.max(1) as usize;

    // Total number of "slices" (layers × faces). Each becomes one entry in
    // Image::surfaces (i.e. one layer in ctt's model).
    let slice_count = layer_count * face_count;

    // Pre-compute per-mip sizes so we can split each level's data blob.
    let mip_slice_sizes =
        compute_mip_slice_sizes(header.pixel_width, header.pixel_height, level_count, format)?;

    // Allocate: surfaces[slice][mip]
    let mut surfaces: Vec<Vec<Surface>> = (0..slice_count)
        .map(|_| Vec::with_capacity(level_count))
        .collect();

    for (mip_idx, level) in reader.levels().enumerate() {
        let level_data = decompress_level(
            supercompression,
            level.data,
            level.uncompressed_byte_length,
            mip_idx,
        )?;

        let expected_slice_size = mip_slice_sizes[mip_idx];
        let mip_w = (header.pixel_width >> mip_idx).max(1);
        let mip_h = (header.pixel_height >> mip_idx).max(1);

        let stride = compute_stride(mip_w, format)?;

        for (slice_idx, slice_surfaces) in surfaces.iter_mut().enumerate() {
            let offset = slice_idx * expected_slice_size;
            let end = offset + expected_slice_size;

            if end > level_data.len() {
                return Err(Error::InputDecoding(format!(
                    "KTX2 level {mip_idx} slice {slice_idx}: expected {expected_slice_size} bytes \
                     at offset {offset}, but level data is only {} bytes",
                    level_data.len(),
                )));
            }

            slice_surfaces.push(Surface {
                data: level_data[offset..end].to_vec(),
                width: mip_w,
                height: mip_h,
                stride,
                format,
                color_space,
                alpha,
            });
        }
    }

    log::debug!(
        "KTX2 input: {:?}, {}x{}, {} slices, {} mips, cubemap={}",
        format,
        header.pixel_width,
        header.pixel_height,
        slice_count,
        level_count,
        is_cubemap,
    );

    Ok(Image {
        surfaces,
        is_cubemap,
    })
}

/// Decompress a single mip level's data according to the supercompression scheme.
fn decompress_level<'a>(
    scheme: Option<ktx2::SupercompressionScheme>,
    data: &'a [u8],
    uncompressed_size: u64,
    level_idx: usize,
) -> Result<Cow<'a, [u8]>> {
    let Some(scheme) = scheme else {
        return Ok(Cow::Borrowed(data));
    };

    if scheme == ktx2::SupercompressionScheme::Zstandard {
        profiling::scope!("decompress_zstd");
        let decompressed =
            zstd::bulk::decompress(data, uncompressed_size as usize).map_err(|e| {
                Error::InputDecoding(format!(
                    "zstd decompression failed at level {level_idx}: {e}"
                ))
            })?;
        Ok(Cow::Owned(decompressed))
    } else if scheme == ktx2::SupercompressionScheme::ZLIB {
        profiling::scope!("decompress_zlib");
        let decompressed = miniz_oxide::inflate::decompress_to_vec_zlib(data).map_err(|e| {
            Error::InputDecoding(format!(
                "zlib decompression failed at level {level_idx}: {e:?}"
            ))
        })?;
        Ok(Cow::Owned(decompressed))
    } else {
        Err(Error::InputDecoding(format!(
            "unsupported KTX2 supercompression scheme: {scheme:?}"
        )))
    }
}

/// Compute the byte size of one slice (one layer×face) at each mip level.
fn compute_mip_slice_sizes(
    base_width: u32,
    base_height: u32,
    level_count: usize,
    format: ktx2::Format,
) -> Result<Vec<usize>> {
    let mut sizes = Vec::with_capacity(level_count);

    for mip in 0..level_count {
        let w = (base_width >> mip).max(1);
        let h = (base_height >> mip).max(1);

        let size = if format.is_compressed() {
            let (bw, bh) = format.block_size().unwrap();
            let bpb = format.bytes_per_block().unwrap();
            let blocks_x = w.div_ceil(bw as u32) as usize;
            let blocks_y = h.div_ceil(bh as u32) as usize;
            blocks_x * blocks_y * bpb
        } else {
            let bpp = format
                .bytes_per_pixel()
                .ok_or_else(|| Error::InputDecoding(format!("unknown bpp for {format:?}")))?;
            w as usize * h as usize * bpp
        };

        sizes.push(size);
    }

    Ok(sizes)
}

/// Compute the stride (bytes per row / bytes per row-of-blocks) for a format and width.
fn compute_stride(width: u32, format: ktx2::Format) -> Result<u32> {
    if format.is_compressed() {
        let (bw, _) = format.block_size().unwrap();
        let bpb = format.bytes_per_block().unwrap();
        Ok(width.div_ceil(bw as u32) * bpb as u32)
    } else {
        let bpp = format
            .bytes_per_pixel()
            .ok_or_else(|| Error::InputDecoding(format!("unknown bpp for {format:?}")))?;
        Ok(width * bpp as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::ktx2::encode_ktx2_image;
    use crate::surface::ColorSpace;

    /// Round-trip: encode an Image to KTX2 bytes, then decode it back.
    #[test]
    fn roundtrip_rgba8_srgb() {
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

        let encoded = encode_ktx2_image(&original, None).unwrap();
        let decoded = decode_ktx2_image(&encoded).unwrap();

        assert_eq!(decoded.surfaces.len(), 1);
        assert_eq!(decoded.surfaces[0].len(), 1);
        let s = &decoded.surfaces[0][0];
        assert_eq!(s.width, 4);
        assert_eq!(s.height, 4);
        assert_eq!(s.format, ktx2::Format::R8G8B8A8_UNORM);
        assert_eq!(s.color_space, ColorSpace::Srgb);
        assert_eq!(s.data, vec![42u8; 64]);
    }

    /// Round-trip with multiple mip levels.
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

        let encoded = encode_ktx2_image(&original, None).unwrap();
        let decoded = decode_ktx2_image(&encoded).unwrap();

        assert_eq!(decoded.surfaces.len(), 1);
        assert_eq!(decoded.surfaces[0].len(), 3);
        assert_eq!(decoded.surfaces[0][0].data, vec![0xAA; 64]);
        assert_eq!(decoded.surfaces[0][1].data, vec![0xBB; 16]);
        assert_eq!(decoded.surfaces[0][2].data, vec![0xCC; 4]);
    }

    /// Round-trip a BC7 compressed single block.
    #[test]
    fn roundtrip_bc7() {
        let original = Image {
            surfaces: vec![vec![Surface {
                data: vec![0xFF; 16],
                width: 4,
                height: 4,
                stride: 16,
                format: ktx2::Format::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Srgb,
                alpha: AlphaMode::Straight,
            }]],
            is_cubemap: false,
        };

        let encoded = encode_ktx2_image(&original, None).unwrap();
        let decoded = decode_ktx2_image(&encoded).unwrap();

        assert_eq!(decoded.surfaces[0][0].format, ktx2::Format::BC7_UNORM_BLOCK);
        assert_eq!(decoded.surfaces[0][0].color_space, ColorSpace::Srgb);
        assert_eq!(decoded.surfaces[0][0].data, vec![0xFF; 16]);
    }

    /// Round-trip a cubemap (6 faces).
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

        let encoded = encode_ktx2_image(&original, None).unwrap();
        let decoded = decode_ktx2_image(&encoded).unwrap();

        assert!(decoded.is_cubemap);
        assert_eq!(decoded.surfaces.len(), 6);
        for i in 0..6 {
            assert_eq!(decoded.surfaces[i][0].data, vec![i as u8; 64]);
        }
    }

    /// Premultiplied alpha survives a round-trip.
    #[test]
    fn roundtrip_premultiplied_alpha() {
        let original = Image {
            surfaces: vec![vec![Surface {
                data: vec![0; 4 * 4 * 4],
                width: 4,
                height: 4,
                stride: 16,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Premultiplied,
            }]],
            is_cubemap: false,
        };

        let encoded = encode_ktx2_image(&original, None).unwrap();
        let decoded = decode_ktx2_image(&encoded).unwrap();
        assert_eq!(decoded.surfaces[0][0].alpha, AlphaMode::Premultiplied);
    }
}
