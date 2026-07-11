use std::borrow::Cow;
use std::io::Read;

use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::surface::{Image, Surface, TextureKind};
use crate::vk_format::FormatExt as _;

// A texture with this many independently addressable layer/face surfaces is
// already far beyond practical container/GPU limits. Capping the count keeps a
// tiny malicious header from driving multi-gigabyte descriptor allocations
// before any payload can be validated.
const MAX_SURFACE_COUNT: usize = 1 << 14;

// Decoding duplicates each level temporarily while splitting it into owned
// surfaces, so keep the supported decoded payload bounded well below the
// address-space limit. This is a resource policy, not a format limit.
const MAX_DECODED_BYTES: usize = 1 << 30;

/// Decode a KTX2 file from raw bytes into an [`Image`].
///
/// Reads format, color space, alpha mode, layers, faces, and all mip levels
/// from the container. Supports zstd/zlib supercompression and errors on
/// missing format information.
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

    let face_count = header.face_count as usize;
    let layer_count = header.layer_count.max(1) as usize;
    let level_count = header.level_count.max(1) as usize;
    let depth = header.pixel_depth.max(1);
    let is_cubemap = face_count == 6;
    let is_3d = header.pixel_depth > 0;

    if header.pixel_width == 0 {
        return Err(Error::InputDecoding("KTX2 width must be at least 1".into()));
    }
    if is_3d && header.pixel_height == 0 {
        return Err(Error::InputDecoding(
            "KTX2 3D textures must have a nonzero height".into(),
        ));
    }
    if !matches!(face_count, 1 | 6) {
        return Err(Error::InputDecoding(format!(
            "KTX2 face count must be 1 or 6, got {face_count}"
        )));
    }
    if is_cubemap
        && (header.pixel_height == 0 || is_3d || header.pixel_width != header.pixel_height)
    {
        return Err(Error::InputDecoding(format!(
            "KTX2 cubemap faces must be square 2D images, got {}x{}x{}",
            header.pixel_width, header.pixel_height, depth
        )));
    }

    if is_3d && (is_cubemap || layer_count > 1) {
        return Err(Error::InputDecoding(
            "KTX2 3D textures cannot be combined with cubemap faces or array layers".into(),
        ));
    }

    let kind = if is_cubemap {
        TextureKind::Cubemap
    } else if is_3d {
        TextureKind::Texture3D
    } else {
        TextureKind::Texture2D
    };

    // Reject a level count larger than the full mip chain the dimensions can
    // produce. This rejects malformed files and keeps the `>> mip` shifts below
    // in range (a count >= 32 would otherwise shift-overflow-panic in debug).
    let max_dim = header
        .pixel_width
        .max(header.pixel_height)
        .max(depth)
        .max(1);
    let max_levels = (32 - max_dim.leading_zeros()) as usize;
    if level_count > max_levels {
        return Err(Error::InputDecoding(format!(
            "KTX2 declares {level_count} mip levels but dimensions {}x{}x{} allow at most \
             {max_levels}",
            header.pixel_width, header.pixel_height, depth,
        )));
    }

    // Total number of "slices" (layers × faces). Each becomes one entry in
    // Image::surfaces (i.e. one layer in ctt's model). 3D textures use a
    // single slice; the depth axis is folded into Surface::data.
    let slice_count = layer_count.checked_mul(face_count).ok_or_else(|| {
        Error::InputDecoding("KTX2 layer × face count overflows this platform".into())
    })?;
    if slice_count > MAX_SURFACE_COUNT {
        return Err(Error::InputDecoding(format!(
            "KTX2 declares {slice_count} layer/face surfaces; the supported maximum is \
             {MAX_SURFACE_COUNT}"
        )));
    }

    // Pre-compute per-mip sizes so we can split each level's data blob. For
    // 3D, each "slice" actually carries `depth_at_mip` Z-slices stacked.
    let mip_slice_sizes = compute_mip_slice_sizes(
        header.pixel_width,
        header.pixel_height,
        depth,
        level_count,
        format,
    )?;

    decoded_size_with_limit(&mip_slice_sizes, slice_count, MAX_DECODED_BYTES)?;

    // Allocate: surfaces[slice][mip]
    let mut surfaces = Vec::new();
    surfaces.try_reserve_exact(slice_count).map_err(|e| {
        Error::InputDecoding(format!(
            "KTX2 layer × face count {slice_count} is too large to allocate: {e}"
        ))
    })?;
    surfaces.resize_with(slice_count, Vec::new);

    for (mip_idx, level) in reader.levels().enumerate() {
        let expected_slice_size = mip_slice_sizes[mip_idx];

        // Total uncompressed bytes this level should hold (all slices). Used to
        // bound decompression so a tiny payload declaring a huge
        // `uncompressed_byte_length` can't trigger a giant allocation.
        let expected_level_size =
            expected_slice_size
                .checked_mul(slice_count)
                .ok_or_else(|| {
                    Error::InputDecoding(format!("KTX2 level {mip_idx}: total size overflow"))
                })?;

        let level_data = decompress_level(
            supercompression,
            level.data,
            level.uncompressed_byte_length,
            expected_level_size,
            mip_idx,
        )?;

        let mip_w = (header.pixel_width >> mip_idx).max(1);
        let mip_h = (header.pixel_height >> mip_idx).max(1);
        let mip_d = (depth >> mip_idx).max(1);

        let stride = compute_stride(mip_w, format)?;
        let single_slice_bytes = expected_slice_size / mip_d as usize;
        let surface_slice_stride = if is_3d { single_slice_bytes as u32 } else { 0 };

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
                depth: mip_d,
                stride,
                slice_stride: surface_slice_stride,
                format,
                color_space,
                alpha,
            });
        }
    }

    log::debug!(
        "KTX2 input: {:?}, {}x{}x{}, {} slices, {} mips, kind={:?}",
        format,
        header.pixel_width,
        header.pixel_height,
        depth,
        slice_count,
        level_count,
        kind,
    );

    Ok(Image { surfaces, kind })
}

/// Decompress a single mip level's data according to the supercompression scheme.
///
/// `expected_size` is the uncompressed byte count implied by the level's
/// dimensions and format. It bounds the output: a crafted file that declares a
/// huge `uncompressed_size` (or whose stream expands beyond what its dimensions
/// justify) is rejected before any large allocation, guarding against
/// decompression bombs.
fn decompress_level<'a>(
    scheme: Option<ktx2::SupercompressionScheme>,
    data: &'a [u8],
    uncompressed_size: u64,
    expected_size: usize,
    level_idx: usize,
) -> Result<Cow<'a, [u8]>> {
    // The KTX2 level index carries the exact uncompressed byte length. Requiring
    // it to match the size implied by the dimensions prevents either value from
    // becoming an attacker-controlled allocation hint for the other.
    let uncompressed_size = usize::try_from(uncompressed_size).map_err(|_| {
        Error::InputDecoding(format!(
            "KTX2 level {level_idx}: declared uncompressed size does not fit this platform"
        ))
    })?;
    if uncompressed_size != expected_size {
        return Err(Error::InputDecoding(format!(
            "KTX2 level {level_idx}: declared uncompressed size {uncompressed_size} does not \
             match the {expected_size} bytes implied by its dimensions"
        )));
    }

    let Some(scheme) = scheme else {
        if data.len() != expected_size {
            return Err(Error::InputDecoding(format!(
                "KTX2 level {level_idx} contains {} bytes, expected {expected_size}",
                data.len()
            )));
        }
        return Ok(Cow::Borrowed(data));
    };

    if scheme == ktx2::SupercompressionScheme::Zstandard {
        profiling::scope!("decompress_zstd");
        let decoder = zstd::stream::read::Decoder::new(data).map_err(|e| {
            Error::InputDecoding(format!("zstd setup failed at level {level_idx}: {e}"))
        })?;
        let decompressed =
            read_decompressed_exact(decoder, expected_size, MAX_DECODED_BYTES, "zstd", level_idx)?;
        Ok(Cow::Owned(decompressed))
    } else if scheme == ktx2::SupercompressionScheme::ZLIB {
        profiling::scope!("decompress_zlib");
        let decoder = flate2::read::ZlibDecoder::new(data);
        let decompressed =
            read_decompressed_exact(decoder, expected_size, MAX_DECODED_BYTES, "zlib", level_idx)?;
        Ok(Cow::Owned(decompressed))
    } else {
        Err(Error::InputDecoding(format!(
            "unsupported KTX2 supercompression scheme: {scheme:?}"
        )))
    }
}

fn read_decompressed_exact(
    mut reader: impl Read,
    expected_size: usize,
    max_size: usize,
    scheme: &str,
    level_idx: usize,
) -> Result<Vec<u8>> {
    if expected_size > max_size {
        return Err(Error::InputDecoding(format!(
            "{scheme} level {level_idx} expands beyond the supported {max_size}-byte limit"
        )));
    }

    let mut output = Vec::new();
    output.try_reserve_exact(expected_size).map_err(|e| {
        Error::InputDecoding(format!(
            "cannot allocate {scheme} output for level {level_idx}: {e}"
        ))
    })?;
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut chunk).map_err(|e| {
            Error::InputDecoding(format!(
                "{scheme} decompression failed at level {level_idx}: {e}"
            ))
        })?;
        if read == 0 {
            break;
        }
        let new_len = output.len().checked_add(read).ok_or_else(|| {
            Error::InputDecoding(format!(
                "{scheme} output size overflow at level {level_idx}"
            ))
        })?;
        if new_len > expected_size {
            return Err(Error::InputDecoding(format!(
                "{scheme} level {level_idx} expands beyond the expected {expected_size} bytes"
            )));
        }
        output.extend_from_slice(&chunk[..read]);
    }
    if output.len() != expected_size {
        return Err(Error::InputDecoding(format!(
            "{scheme} level {level_idx} produced {} bytes, expected {expected_size}",
            output.len()
        )));
    }
    Ok(output)
}

/// Compute the byte size of one slice (one layer×face, or all Z slices for
/// a 3D texture) at each mip level.
fn compute_mip_slice_sizes(
    base_width: u32,
    base_height: u32,
    base_depth: u32,
    level_count: usize,
    format: ktx2::Format,
) -> Result<Vec<usize>> {
    let mut sizes = Vec::with_capacity(level_count);

    for mip in 0..level_count {
        let w = (base_width >> mip).max(1);
        let h = (base_height >> mip).max(1);
        let d = (base_depth >> mip).max(1) as usize;

        let overflow = || Error::InputDecoding(format!("KTX2 level {mip}: size overflow"));

        // Checked arithmetic: attacker-controlled dimensions must not overflow
        // `usize` (trivially possible on 32-bit targets).
        let size = if format.is_compressed() {
            let (bw, bh) = format.block_size().unwrap();
            let bpb = format.bytes_per_block().unwrap();
            let blocks_x = w.div_ceil(bw as u32) as usize;
            let blocks_y = h.div_ceil(bh as u32) as usize;
            blocks_x
                .checked_mul(blocks_y)
                .and_then(|v| v.checked_mul(bpb))
                .and_then(|v| v.checked_mul(d))
                .ok_or_else(overflow)?
        } else {
            let bpp = format
                .bytes_per_pixel()
                .ok_or_else(|| Error::InputDecoding(format!("unknown bpp for {format:?}")))?;
            (w as usize)
                .checked_mul(h as usize)
                .and_then(|v| v.checked_mul(bpp))
                .and_then(|v| v.checked_mul(d))
                .ok_or_else(overflow)?
        };

        sizes.push(size);
    }

    Ok(sizes)
}

fn decoded_size_with_limit(
    mip_slice_sizes: &[usize],
    slice_count: usize,
    max_bytes: usize,
) -> Result<usize> {
    let mut total = 0usize;
    for (mip_idx, &slice_size) in mip_slice_sizes.iter().enumerate() {
        let level_size = slice_size.checked_mul(slice_count).ok_or_else(|| {
            Error::InputDecoding(format!("KTX2 level {mip_idx}: total size overflow"))
        })?;
        total = total
            .checked_add(level_size)
            .ok_or_else(|| Error::InputDecoding("KTX2 decoded size overflows usize".into()))?;
        if total > max_bytes {
            return Err(Error::InputDecoding(format!(
                "KTX2 decoded payload is {total} bytes; the supported maximum is \
                 {max_bytes} bytes"
            )));
        }
    }
    Ok(total)
}

/// Compute the stride (bytes per row / bytes per row-of-blocks) for a format and width.
fn compute_stride(width: u32, format: ktx2::Format) -> Result<u32> {
    let overflow = || Error::InputDecoding(format!("KTX2 stride overflow for {format:?}"));
    if format.is_compressed() {
        let (bw, _) = format.block_size().unwrap();
        let bpb = format.bytes_per_block().unwrap();
        width
            .div_ceil(bw as u32)
            .checked_mul(bpb as u32)
            .ok_or_else(overflow)
    } else {
        let bpp = format
            .bytes_per_pixel()
            .ok_or_else(|| Error::InputDecoding(format!("unknown bpp for {format:?}")))?;
        width.checked_mul(bpp as u32).ok_or_else(overflow)
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
                depth: 1,
                stride: 16,
                slice_stride: 0,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Srgb,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture2D,
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
                depth: 1,
                stride: 16,
                slice_stride: 0,
                format: ktx2::Format::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Srgb,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture2D,
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

        let encoded = encode_ktx2_image(&original, None).unwrap();
        let decoded = decode_ktx2_image(&encoded).unwrap();

        assert_eq!(decoded.kind, TextureKind::Cubemap);
        assert_eq!(decoded.surfaces.len(), 6);
        for i in 0..6 {
            assert_eq!(decoded.surfaces[i][0].data, vec![i as u8; 64]);
        }
    }

    /// A tiny zstd payload that declares a preposterous uncompressed size must
    /// be rejected cleanly, without attempting a giant allocation.
    #[test]
    fn zstd_decompression_bomb_rejected() {
        use crate::convert::Ktx2Supercompression;

        let image = Image {
            surfaces: vec![vec![Surface {
                data: vec![0u8; 4 * 4 * 4],
                width: 4,
                height: 4,
                depth: 1,
                stride: 16,
                slice_stride: 0,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture2D,
        };

        let mut bytes =
            encode_ktx2_image(&image, Some(Ktx2Supercompression::Zstd { level: 3 })).unwrap();

        // Sanity: it decodes fine before tampering.
        assert!(decode_ktx2_image(&bytes).is_ok());

        // Patch level 0's `uncompressed_byte_length` (the third u64 in its level
        // index entry) to 100 GiB. The tiny payload now claims to expand hugely.
        let off = ktx2::Header::LENGTH + 16;
        let bomb: u64 = 100 * 1024 * 1024 * 1024;
        bytes[off..off + 8].copy_from_slice(&bomb.to_le_bytes());

        let err = decode_ktx2_image(&bytes).unwrap_err();
        assert!(
            matches!(err, Error::InputDecoding(_)),
            "expected InputDecoding, got {err:?}"
        );
    }

    /// Premultiplied alpha survives a round-trip.
    #[test]
    fn roundtrip_premultiplied_alpha() {
        let original = Image {
            surfaces: vec![vec![Surface {
                data: vec![0; 4 * 4 * 4],
                width: 4,
                height: 4,
                depth: 1,
                stride: 16,
                slice_stride: 0,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Premultiplied,
            }]],
            kind: TextureKind::Texture2D,
        };

        let encoded = encode_ktx2_image(&original, None).unwrap();
        let decoded = decode_ktx2_image(&encoded).unwrap();
        assert_eq!(decoded.surfaces[0][0].alpha, AlphaMode::Premultiplied);
    }

    #[test]
    fn invalid_face_count_rejected_before_allocation() {
        let image = Image {
            surfaces: vec![vec![Surface {
                data: vec![0; 4],
                width: 1,
                height: 1,
                depth: 1,
                stride: 4,
                slice_stride: 0,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture2D,
        };
        let mut bytes = encode_ktx2_image(&image, None).unwrap();
        bytes[36..40].copy_from_slice(&2u32.to_le_bytes());

        let err = decode_ktx2_image(&bytes).unwrap_err();
        assert!(err.to_string().contains("face count"), "got: {err}");
    }

    #[test]
    fn excessive_layer_count_rejected_before_allocation() {
        let image = Image {
            surfaces: vec![vec![Surface {
                data: vec![0; 4],
                width: 1,
                height: 1,
                depth: 1,
                stride: 4,
                slice_stride: 0,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture2D,
        };
        let mut bytes = encode_ktx2_image(&image, None).unwrap();
        bytes[32..36].copy_from_slice(&u32::MAX.to_le_bytes());

        let err = decode_ktx2_image(&bytes).unwrap_err();
        assert!(err.to_string().contains("supported maximum"), "got: {err}");
    }

    #[test]
    fn one_dimensional_texture_decodes_as_height_one() {
        let image = Image {
            surfaces: vec![vec![Surface {
                data: vec![1, 2, 3, 4],
                width: 4,
                height: 1,
                depth: 1,
                stride: 4,
                slice_stride: 0,
                format: ktx2::Format::R8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture2D,
        };
        let mut bytes = encode_ktx2_image(&image, None).unwrap();
        bytes[24..28].copy_from_slice(&0u32.to_le_bytes());

        let decoded = decode_ktx2_image(&bytes).unwrap();
        assert_eq!(decoded.surfaces[0][0].height, 1);
        assert_eq!(decoded.surfaces[0][0].data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn cubemap_dimensions_must_be_square_and_two_dimensional() {
        let faces = (0..6)
            .map(|_| {
                vec![Surface {
                    data: vec![0; 4],
                    width: 2,
                    height: 2,
                    depth: 1,
                    stride: 2,
                    slice_stride: 0,
                    format: ktx2::Format::R8_UNORM,
                    color_space: ColorSpace::Linear,
                    alpha: AlphaMode::Straight,
                }]
            })
            .collect();
        let image = Image {
            surfaces: faces,
            kind: TextureKind::Cubemap,
        };
        let encoded = encode_ktx2_image(&image, None).unwrap();

        for bad_height in [0u32, 1] {
            let mut bytes = encoded.clone();
            bytes[24..28].copy_from_slice(&bad_height.to_le_bytes());
            let err = decode_ktx2_image(&bytes).unwrap_err();
            assert!(err.to_string().contains("square 2D"), "got: {err}");
        }
    }

    #[test]
    fn declared_level_size_must_match_dimensions() {
        let compressed = zstd::bulk::compress(&[1, 2, 3, 4], 1).unwrap();
        let err = decompress_level(
            Some(ktx2::SupercompressionScheme::Zstandard),
            &compressed,
            4,
            16,
            0,
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not match"), "got: {err}");
    }

    #[test]
    fn expanding_streams_stop_at_resource_limit() {
        use std::io::Write as _;

        let payload = vec![0u8; 4096];

        let zstd = zstd::bulk::compress(&payload, 1).unwrap();
        let zstd_decoder = zstd::stream::read::Decoder::new(zstd.as_slice()).unwrap();
        let err =
            read_decompressed_exact(zstd_decoder, payload.len(), 1024, "zstd", 0).unwrap_err();
        assert!(err.to_string().contains("supported"), "got: {err}");

        let mut zlib = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        zlib.write_all(&payload).unwrap();
        let zlib = zlib.finish().unwrap();
        let zlib_decoder = flate2::read::ZlibDecoder::new(zlib.as_slice());
        let err =
            read_decompressed_exact(zlib_decoder, payload.len(), 1024, "zlib", 0).unwrap_err();
        assert!(err.to_string().contains("supported"), "got: {err}");
    }

    #[test]
    fn total_decoded_payload_is_bounded() {
        assert_eq!(decoded_size_with_limit(&[400, 300], 1, 700).unwrap(), 700);
        let err = decoded_size_with_limit(&[400, 301], 1, 700).unwrap_err();
        assert!(err.to_string().contains("supported maximum"), "got: {err}");
    }
}
