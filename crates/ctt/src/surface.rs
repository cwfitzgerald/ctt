use std::borrow::Cow;
use std::fmt;

use crate::alpha::AlphaMode;
use crate::error::{Error, Result};
use crate::vk_format::FormatExt;

/// Color space metadata for a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum ColorSpace {
    #[default]
    Srgb,
    Linear,
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Srgb => f.write_str("srgb"),
            Self::Linear => f.write_str("linear"),
        }
    }
}

/// A single image surface — either raw pixels or compressed blocks.
///
/// 2D surfaces use `depth == 1`; 3D (volume) surfaces use `depth > 1` with all
/// Z slices packed contiguously in `data`. The format field determines whether
/// the data is uncompressed pixel data or compressed block data — use
/// [`FormatExt::is_compressed`] to check.
#[derive(Debug, Clone)]
pub struct Surface {
    /// Raw bytes: uncompressed pixels or compressed blocks, laid out row by
    /// row (and slice by slice when `depth > 1`) at the declared strides.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of Z slices. `1` for 2D textures, `>= 1` for 3D textures.
    pub depth: u32,
    /// Bytes between adjacent rows in `data` (a.k.a. row pitch). May include
    /// trailing padding; the minimum sensible value is one packed row:
    /// `width * bytes_per_pixel` for uncompressed formats, or
    /// `ceil(width / block_w) * bytes_per_block` for compressed formats.
    pub stride: u32,
    /// Bytes between adjacent Z slices in `data` (a.k.a. depth pitch). May
    /// include trailing padding; the minimum sensible value is one packed slice
    /// (`stride * ceil(height / block_h)` for compressed, `stride * height`
    /// for uncompressed). Meaningful only when `depth > 1`; set to `0` for 2D
    /// surfaces.
    pub slice_stride: u32,
    /// Pixel or block format of `data`.
    pub format: ktx2::Format,
    /// Color space the pixel values live in (sRGB or linear).
    pub color_space: ColorSpace,
    /// How the alpha channel relates to the color channels.
    pub alpha: AlphaMode,
}

/// Texture topology — distinguishes 2D, cubemap, and 3D textures.
///
/// Array-ness is implicit in `Image::surfaces.len()`:
/// - `Texture2D`: one layer per surface entry (`surfaces.len()` is the array layer count).
/// - `Cubemap`: six surfaces per cube (`surfaces.len() / 6` is the cube count;
///   `surfaces.len() % 6` must be 0).
/// - `Texture3D`: exactly one surface entry; depth is carried on the Surface.
///   3D arrays are not supported (KTX2 / D3D do not allow them).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureKind {
    /// Flat 2D texture (or a 2D array — one surface entry per array layer).
    Texture2D,
    /// Cubemap (or cubemap array): six faces per cube, `surfaces.len()` a
    /// multiple of 6.
    Cubemap,
    /// 3D (volume) texture: a single surface entry whose `depth > 1`.
    Texture3D,
}

/// Multi-layer, multi-mip image.
///
/// `surfaces[i][j]` is slice `i` (layer or face), mip level `j`. The meaning
/// of the slice axis depends on `kind`; see [`TextureKind`].
#[derive(Debug, Clone)]
pub struct Image {
    pub surfaces: Vec<Vec<Surface>>,
    pub kind: TextureKind,
}

impl Image {
    /// Verify that this `Image` satisfies the invariants implied by its `kind`
    /// and that all surfaces share the metadata the pipeline assumes is
    /// uniform (format, color space, alpha mode, mip count).
    pub fn validate(&self) -> Result<()> {
        // 1. Layout shape.
        if self.surfaces.is_empty() {
            return Err(Error::InvalidImage("image has no surfaces".into()));
        }
        for (layer_idx, layer) in self.surfaces.iter().enumerate() {
            if layer.is_empty() {
                return Err(Error::InvalidImage(format!(
                    "layer {layer_idx} has no mip levels",
                )));
            }
        }

        // 2. Cross-surface uniformity (mip count, format, color space, alpha)
        // and depth >= 1.
        let expected_mips = self.surfaces[0].len();
        let head = &self.surfaces[0][0];
        let expected_format = head.format;
        let expected_cs = head.color_space;
        let expected_alpha = head.alpha;

        for (layer_idx, layer) in self.surfaces.iter().enumerate() {
            if layer.len() != expected_mips {
                return Err(Error::InvalidImage(format!(
                    "layer {layer_idx} has {} mip(s); layer 0 has {expected_mips}",
                    layer.len(),
                )));
            }
            for (mip_idx, s) in layer.iter().enumerate() {
                if s.format != expected_format {
                    return Err(Error::InvalidImage(format!(
                        "layer {layer_idx} mip {mip_idx}: format {:?} differs from layer 0 ({:?})",
                        s.format, expected_format,
                    )));
                }
                if s.color_space != expected_cs {
                    return Err(Error::InvalidImage(format!(
                        "layer {layer_idx} mip {mip_idx}: color space {:?} differs from layer 0 ({:?})",
                        s.color_space, expected_cs,
                    )));
                }
                if s.alpha != expected_alpha {
                    return Err(Error::InvalidImage(format!(
                        "layer {layer_idx} mip {mip_idx}: alpha {:?} differs from layer 0 ({:?})",
                        s.alpha, expected_alpha,
                    )));
                }
                if s.depth == 0 {
                    return Err(Error::InvalidImage(format!(
                        "layer {layer_idx} mip {mip_idx}: depth must be >= 1",
                    )));
                }
                if s.width == 0 || s.height == 0 {
                    return Err(Error::InvalidImage(format!(
                        "layer {layer_idx} mip {mip_idx}: width and height must be >= 1, \
                         got {}x{}",
                        s.width, s.height,
                    )));
                }
            }
        }

        // 3. Kind invariants. Done before stride/length so that "Texture2D
        // with depth>1" errors with the structural message instead of
        // "slice_stride below minimum".
        match self.kind {
            TextureKind::Texture2D => {
                self.validate_no_volume_slices()?;
            }
            TextureKind::Cubemap => {
                if !self.surfaces.len().is_multiple_of(6) {
                    return Err(Error::InvalidImage(format!(
                        "cubemap surface count must be a multiple of 6, got {}",
                        self.surfaces.len(),
                    )));
                }
                self.validate_no_volume_slices()?;
            }
            TextureKind::Texture3D => {
                if self.surfaces.len() != 1 {
                    return Err(Error::InvalidImage(format!(
                        "3D textures must have exactly one surface entry, got {}",
                        self.surfaces.len(),
                    )));
                }
                let layer = &self.surfaces[0];
                let base_depth = layer[0].depth;
                for (mip_idx, s) in layer.iter().enumerate() {
                    let expected = base_depth.checked_shr(mip_idx as u32).unwrap_or(0).max(1);
                    if s.depth != expected {
                        return Err(Error::InvalidImage(format!(
                            "3D mip {mip_idx}: depth {} does not match expected {expected} \
                             (base_depth={base_depth})",
                            s.depth,
                        )));
                    }
                }
            }
        }

        // 4. Stride/length checks. By here we know depth==1 implies a 2D-ish
        // image with no slice axis to worry about, and depth>1 implies 3D
        // with the per-mip depth chain already verified.
        for (layer_idx, layer) in self.surfaces.iter().enumerate() {
            for (mip_idx, s) in layer.iter().enumerate() {
                let Some(tight_row) = s.tight_row_bytes() else {
                    return Err(Error::InvalidImage(format!(
                        "layer {layer_idx} mip {mip_idx}: format {:?} has no known pixel/block size",
                        s.format,
                    )));
                };
                if s.stride < tight_row {
                    return Err(Error::InvalidImage(format!(
                        "layer {layer_idx} mip {mip_idx}: stride {} is below the tight \
                         minimum {tight_row} for {:?} at width={}",
                        s.stride, s.format, s.width,
                    )));
                }
                let rows = if let Some((_, bh)) = s.format.block_size() {
                    s.height.div_ceil(bh as u32) as usize
                } else {
                    s.height as usize
                };
                // Height zero was rejected above, so at least one physical row
                // (or block row) is present.
                let row_span = (rows - 1)
                    .checked_mul(s.stride as usize)
                    .and_then(|v| v.checked_add(tight_row as usize))
                    .ok_or_else(|| {
                        Error::InvalidImage(format!(
                            "layer {layer_idx} mip {mip_idx}: row/slice span overflows"
                        ))
                    })?;
                if s.depth > 1 {
                    if row_span > u32::MAX as usize {
                        return Err(Error::InvalidImage(format!(
                            "layer {layer_idx} mip {mip_idx}: padded slice span {row_span} \
                             overflows the u32 slice_stride representation"
                        )));
                    }
                    if (s.slice_stride as usize) < row_span {
                        return Err(Error::InvalidImage(format!(
                            "layer {layer_idx} mip {mip_idx}: slice_stride {} is below the \
                             minimum {row_span} required by the padded row layout",
                            s.slice_stride,
                        )));
                    }
                }
                let required = if s.depth > 1 {
                    (s.depth as usize - 1)
                        .checked_mul(s.slice_stride as usize)
                        .and_then(|v| v.checked_add(row_span))
                        .ok_or_else(|| {
                            Error::InvalidImage(format!(
                                "layer {layer_idx} mip {mip_idx}: total 3D data span overflows"
                            ))
                        })?
                } else {
                    row_span
                };
                if s.data.len() < required {
                    return Err(Error::InvalidImage(format!(
                        "layer {layer_idx} mip {mip_idx}: data is {} bytes, need at least \
                         {required} to read width={}, height={}, depth={} at the declared strides",
                        s.data.len(),
                        s.width,
                        s.height,
                        s.depth,
                    )));
                }
            }
        }

        Ok(())
    }

    fn validate_no_volume_slices(&self) -> Result<()> {
        for (layer_idx, layer) in self.surfaces.iter().enumerate() {
            for (mip_idx, s) in layer.iter().enumerate() {
                if s.depth != 1 {
                    return Err(Error::InvalidImage(format!(
                        "{:?} layer {layer_idx} mip {mip_idx}: depth must be 1, got {}",
                        self.kind, s.depth,
                    )));
                }
            }
        }
        Ok(())
    }
}

impl Surface {
    /// Number of rows or rows-of-blocks in this surface — `height` for
    /// uncompressed formats, `ceil(height / block_h)` for compressed.
    fn rows_in_image(&self) -> u32 {
        if let Some((_, bh)) = self.format.block_size() {
            self.height.div_ceil(bh as u32)
        } else {
            self.height
        }
    }

    /// Bytes for one tightly-packed row (or row-of-blocks for compressed
    /// formats). Returns `None` if the format's pixel/block size is unknown or
    /// the byte count overflows `u32`.
    pub fn tight_row_bytes(&self) -> Option<u32> {
        if let Some((bw, _)) = self.format.block_size() {
            let bpb = self.format.bytes_per_block()? as u32;
            self.width.div_ceil(bw as u32).checked_mul(bpb)
        } else {
            let bpp = self.format.bytes_per_pixel()? as u32;
            self.width.checked_mul(bpp)
        }
    }

    /// Bytes for one tightly-packed Z slice. For 2D surfaces this is the
    /// whole image; for 3D it's one entry along the depth axis. Returns `None`
    /// if the format size is unknown or the byte count overflows `u32`.
    pub fn tight_slice_bytes(&self) -> Option<u32> {
        self.tight_row_bytes()?.checked_mul(self.rows_in_image())
    }

    /// True when `stride` and (for `depth > 1`) `slice_stride` already match
    /// the tightly-packed minimums and `data` is exactly the right length —
    /// i.e., `data` can be reused without repacking.
    pub fn is_tightly_packed(&self) -> bool {
        let Some(tight_row) = self.tight_row_bytes() else {
            return false;
        };
        let Some(tight_slice) = self.tight_slice_bytes() else {
            return false;
        };
        let expected_total = tight_slice as usize * self.depth as usize;
        if self.stride != tight_row || self.data.len() != expected_total {
            return false;
        }
        if self.depth > 1 && self.slice_stride != tight_slice {
            return false;
        }
        true
    }

    /// Borrow the surface bytes if already tight, or build a fresh tightly-
    /// packed copy that strips per-row and per-slice padding.
    ///
    /// Container output formats (KTX2, DDS) require tight packing; this is
    /// the bridge for surfaces that carry padded strides through the
    /// passthrough fast path. Panics if the format's pixel/block size is
    /// unknown — `Image::validate` is the place that should reject those.
    pub fn tight_data(&self) -> Cow<'_, [u8]> {
        if self.is_tightly_packed() {
            return Cow::Borrowed(&self.data);
        }
        let tight_row = self
            .tight_row_bytes()
            .expect("tight_data requires a known format size");
        let tight_slice = self
            .tight_slice_bytes()
            .expect("tight_data requires a known format size");
        let rows = self.rows_in_image() as usize;
        let row = tight_row as usize;
        let slice_in = self.slice_stride as usize;
        let row_in = self.stride as usize;
        let mut out = Vec::with_capacity(tight_slice as usize * self.depth as usize);
        for z in 0..self.depth as usize {
            let slice_base = z * slice_in;
            for y in 0..rows {
                let row_start = slice_base + y * row_in;
                out.extend_from_slice(&self.data[row_start..row_start + row]);
            }
        }
        Cow::Owned(out)
    }

    /// Tile the surface into tightly-packed blocks for block-level encoders.
    ///
    /// Each block is `block_w * block_h * bytes_per_pixel` bytes of contiguous
    /// pixel data. Partial blocks at the right/bottom edges replicate the
    /// nearest edge pixel (clamp-to-edge) so encoders don't see black padding.
    ///
    /// Panics if the format is compressed or has unknown bytes-per-pixel, or
    /// if the surface is empty (width or height of 0).
    pub fn tile_to_blocks(&self, block_w: u32, block_h: u32) -> Vec<u8> {
        let bpp = self
            .format
            .bytes_per_pixel()
            .expect("tile_to_blocks requires an uncompressed format with known bpp")
            as u32;
        assert!(
            self.width > 0 && self.height > 0,
            "tile_to_blocks requires non-empty surface"
        );

        let blocks_x = self.width.div_ceil(block_w);
        let blocks_y = self.height.div_ceil(block_h);
        let block_bytes = (block_w * block_h * bpp) as usize;
        // Widen to usize before multiplying: for very large surfaces the block
        // count and per-row byte offsets overflow u32 (a >32768-wide RGBA8
        // surface overflows `y * stride`).
        let mut out = vec![0u8; blocks_x as usize * blocks_y as usize * block_bytes];

        let max_x = self.width - 1;
        let max_y = self.height - 1;
        let stride = self.stride as usize;
        let bpp = bpp as usize;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = (by * blocks_x + bx) as usize;
                let block_start = block_idx * block_bytes;

                for py in 0..block_h {
                    let y = (by * block_h + py).min(max_y) as usize;
                    for px in 0..block_w {
                        let x = (bx * block_w + px).min(max_x) as usize;
                        let src = y * stride + x * bpp;
                        let dst = block_start + ((py * block_w + px) as usize) * bpp;
                        out[dst..dst + bpp].copy_from_slice(&self.data[src..src + bpp]);
                    }
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_to_blocks_basic() {
        // 2x2 RGBA8 image, tile into 4x4 blocks (padded)
        let surface = Surface {
            data: vec![
                1, 2, 3, 4, 5, 6, 7, 8, // row 0
                9, 10, 11, 12, 13, 14, 15, 16, // row 1
            ],
            width: 2,
            height: 2,
            depth: 1,
            stride: 8,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };

        let blocks = surface.tile_to_blocks(4, 4);
        // 1 block of 4x4 pixels, 4 bytes each = 64 bytes
        assert_eq!(blocks.len(), 64);
        // First pixel should be (1,2,3,4)
        assert_eq!(&blocks[0..4], &[1, 2, 3, 4]);
        // Second pixel should be (5,6,7,8)
        assert_eq!(&blocks[4..8], &[5, 6, 7, 8]);
        // Right-padded pixel at (2,0) should replicate the pixel at (1,0) = (5,6,7,8)
        assert_eq!(&blocks[8..12], &[5, 6, 7, 8]);
        // Right-padded pixel at (3,0) should also replicate (5,6,7,8)
        assert_eq!(&blocks[12..16], &[5, 6, 7, 8]);
        // Bottom-padded row 2: replicates row 1 (pixels (9,10,11,12), (13,14,15,16), ...)
        assert_eq!(&blocks[32..36], &[9, 10, 11, 12]);
        assert_eq!(&blocks[36..40], &[13, 14, 15, 16]);
        // And the right-padded portion of that replicated row still replicates the last column.
        assert_eq!(&blocks[40..44], &[13, 14, 15, 16]);
        assert_eq!(&blocks[44..48], &[13, 14, 15, 16]);
    }

    #[test]
    fn tile_to_blocks_non_multiple_size() {
        // 3x3 RGBA8 (not a multiple of 4) — should edge-replicate into one 4x4 block.
        let mut data = Vec::new();
        for y in 0..3u8 {
            for x in 0..3u8 {
                data.extend_from_slice(&[x, y, 0, 255]);
            }
        }
        let surface = Surface {
            data,
            width: 3,
            height: 3,
            depth: 1,
            stride: 3 * 4,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };

        let blocks = surface.tile_to_blocks(4, 4);
        assert_eq!(blocks.len(), 64);

        // Pixel (3, 0) should replicate pixel (2, 0) = (2, 0, 0, 255).
        assert_eq!(&blocks[12..16], &[2, 0, 0, 255]);
        // Pixel (0, 3) should replicate pixel (0, 2) = (0, 2, 0, 255).
        assert_eq!(&blocks[48..52], &[0, 2, 0, 255]);
        // Corner pixel (3, 3) should replicate pixel (2, 2) = (2, 2, 0, 255).
        assert_eq!(&blocks[60..64], &[2, 2, 0, 255]);
    }

    /// 2×2 RGBA8 with `stride` deliberately wider than the row payload — the
    /// extra bytes between rows must be ignored when tiling.
    #[test]
    fn tile_to_blocks_padded_stride() {
        let pad = 0xCCu8;
        let mut data = Vec::new();
        // Row 0: 2 RGBA8 pixels + 4 padding bytes = stride 12.
        data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        data.extend_from_slice(&[pad; 4]);
        // Row 1: 2 RGBA8 pixels + 4 padding bytes.
        data.extend_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
        data.extend_from_slice(&[pad; 4]);

        let surface = Surface {
            data,
            width: 2,
            height: 2,
            depth: 1,
            stride: 12, // 8 bytes of pixels + 4 bytes of padding
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };

        let blocks = surface.tile_to_blocks(4, 4);
        assert_eq!(blocks.len(), 64);
        // The padding byte 0xCC must never appear in the tiled output —
        // every pixel comes from the real width-2 source data, edge-replicated
        // for the padded right/bottom of the 4×4 block.
        assert!(
            !blocks.contains(&pad),
            "tile_to_blocks must not read past the row payload",
        );
        // Pixel (0,0) = real (0,0).
        assert_eq!(&blocks[0..4], &[1, 2, 3, 4]);
        // Pixel (1,0) = real (1,0).
        assert_eq!(&blocks[4..8], &[5, 6, 7, 8]);
        // Pixel (0,1) = real (0,1) — proves we hopped by `stride`, not by
        // `width * bpp`, when moving to the next row.
        assert_eq!(&blocks[16..20], &[9, 10, 11, 12]);
    }

    fn s2d(width: u32, height: u32) -> Surface {
        Surface {
            data: vec![0u8; (width * height * 4) as usize],
            width,
            height,
            depth: 1,
            stride: width * 4,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        }
    }

    #[test]
    fn validate_2d_ok() {
        let img = Image {
            surfaces: vec![vec![s2d(4, 4)]],
            kind: TextureKind::Texture2D,
        };
        img.validate().unwrap();
    }

    #[test]
    fn validate_zero_width_rejected() {
        let mut s = s2d(4, 4);
        s.width = 0;
        s.stride = 0;
        s.data.clear();
        let img = Image {
            surfaces: vec![vec![s]],
            kind: TextureKind::Texture2D,
        };
        let err = img.validate().unwrap_err();
        assert!(
            err.to_string().contains("width and height must be >= 1"),
            "got: {err}",
        );
    }

    #[test]
    fn validate_zero_height_rejected() {
        let mut s = s2d(4, 4);
        s.height = 0;
        s.data.clear();
        let img = Image {
            surfaces: vec![vec![s]],
            kind: TextureKind::Texture2D,
        };
        let err = img.validate().unwrap_err();
        assert!(
            err.to_string().contains("width and height must be >= 1"),
            "got: {err}",
        );
    }

    #[test]
    fn validate_cubemap_wrong_face_count() {
        let img = Image {
            surfaces: vec![vec![s2d(4, 4)]; 5],
            kind: TextureKind::Cubemap,
        };
        let err = img.validate().unwrap_err();
        assert!(err.to_string().contains("multiple of 6"), "got: {err}");
    }

    #[test]
    fn validate_cubemap_array_ok() {
        let img = Image {
            surfaces: vec![vec![s2d(4, 4)]; 12],
            kind: TextureKind::Cubemap,
        };
        img.validate().unwrap();
    }

    #[test]
    fn validate_2d_with_depth_rejected() {
        let mut s = s2d(4, 4);
        s.depth = 2;
        let img = Image {
            surfaces: vec![vec![s]],
            kind: TextureKind::Texture2D,
        };
        let err = img.validate().unwrap_err();
        assert!(err.to_string().contains("depth must be 1"), "got: {err}");
    }

    /// A 3D surface whose per-slice byte count overflows `u32` must be
    /// rejected by `validate`, not panic on the internal `tight_slice_bytes`
    /// lookup. width*bpp fits in u32 here, but *height does not.
    #[test]
    fn validate_3d_slice_size_overflow_rejected() {
        let img = Image {
            surfaces: vec![vec![Surface {
                data: vec![0u8; 16],
                width: 65536,
                height: 65536,
                depth: 2,
                stride: 65536 * 4,
                slice_stride: u32::MAX,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture3D,
        };
        let err = img.validate().unwrap_err();
        assert!(err.to_string().contains("overflows"), "got: {err}");
    }

    #[test]
    fn validate_3d_rejects_overlapping_padded_slices() {
        let img = Image {
            surfaces: vec![vec![Surface {
                data: vec![0; 7],
                width: 1,
                height: 2,
                depth: 2,
                stride: 4,
                slice_stride: 2,
                format: ktx2::Format::R8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture3D,
        };

        let err = img.validate().unwrap_err();
        assert!(err.to_string().contains("padded row layout"), "got: {err}");
    }

    #[test]
    fn validate_3d_extreme_spans_return_error_without_panicking() {
        let img = Image {
            surfaces: vec![vec![Surface {
                data: Vec::new(),
                width: 1,
                height: 4,
                depth: u32::MAX,
                stride: u32::MAX,
                slice_stride: u32::MAX,
                format: ktx2::Format::R8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: TextureKind::Texture3D,
        };

        let result = std::panic::catch_unwind(|| img.validate());
        assert!(result.is_ok(), "extreme layout must not panic");
        assert!(result.unwrap().is_err(), "extreme layout must be rejected");
    }

    #[test]
    fn validate_3d_must_have_one_layer() {
        let img = Image {
            surfaces: vec![vec![s2d(4, 4)], vec![s2d(4, 4)]],
            kind: TextureKind::Texture3D,
        };
        let err = img.validate().unwrap_err();
        assert!(
            err.to_string().contains("exactly one surface"),
            "got: {err}",
        );
    }

    #[test]
    fn validate_3d_mip_depth_chain() {
        let mut base = s2d(4, 4);
        base.depth = 4;
        base.slice_stride = base.stride * base.height;
        base.data = vec![0u8; (base.slice_stride * base.depth) as usize];
        let mut mip1 = s2d(2, 2);
        // Wrong: depth at mip 1 should be 2, not 4.
        mip1.depth = 4;
        mip1.slice_stride = mip1.stride * mip1.height;
        mip1.data = vec![0u8; (mip1.slice_stride * mip1.depth) as usize];
        let img = Image {
            surfaces: vec![vec![base, mip1]],
            kind: TextureKind::Texture3D,
        };
        let err = img.validate().unwrap_err();
        assert!(
            err.to_string().contains("does not match expected"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_3d_many_mips_does_not_shift_panic() {
        let img = Image {
            surfaces: vec![(0..33).map(|_| s2d(1, 1)).collect()],
            kind: TextureKind::Texture3D,
        };

        let result = std::panic::catch_unwind(|| img.validate());
        assert!(result.is_ok(), "large mip count must not panic");
        assert!(result.unwrap().is_ok());
    }

    #[test]
    fn validate_format_uniformity() {
        let mut a = s2d(4, 4);
        let mut b = s2d(4, 4);
        b.format = ktx2::Format::R8G8B8A8_SRGB;
        a.color_space = ColorSpace::Linear;
        b.color_space = ColorSpace::Srgb;
        let img = Image {
            surfaces: vec![vec![a], vec![b]],
            kind: TextureKind::Texture2D,
        };
        let err = img.validate().unwrap_err();
        assert!(err.to_string().contains("format"), "got: {err}");
    }

    #[test]
    fn validate_mip_count_uniformity() {
        let img = Image {
            surfaces: vec![
                vec![s2d(4, 4), s2d(2, 2)],
                vec![s2d(4, 4)], // missing mip 1
            ],
            kind: TextureKind::Texture2D,
        };
        let err = img.validate().unwrap_err();
        assert!(err.to_string().contains("mip"), "got: {err}");
    }
}
