use ctt_intel_texture_compressor as itc;

use crate::encoders::edge;
use crate::encoders::{Encoder, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::surface::Surface;
use crate::vk_format::FormatExt as _;

/// ISPC-specific encoder settings.
#[derive(Debug, Clone, Copy)]
pub struct IspcSettings {
    /// Whether to encode alpha channel (for BC7).
    pub alpha: bool,
}

impl EncoderSettings for IspcSettings {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct IspcEncoder;

impl Encoder for IspcEncoder {
    fn name(&self) -> &str {
        "intel"
    }

    fn supported_formats(&self) -> &[ktx2::Format] {
        &[
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            ktx2::Format::BC3_UNORM_BLOCK,
            ktx2::Format::BC4_UNORM_BLOCK,
            ktx2::Format::BC5_UNORM_BLOCK,
            ktx2::Format::BC6H_UFLOAT_BLOCK,
            ktx2::Format::BC7_UNORM_BLOCK,
            ktx2::Format::ETC2_R8G8B8_UNORM_BLOCK,
        ]
    }

    fn required_input_format(&self, format: ktx2::Format) -> ktx2::Format {
        use ktx2::Format as F;
        match format {
            F::BC4_UNORM_BLOCK => F::R8_UNORM,
            F::BC5_UNORM_BLOCK => F::R8G8_UNORM,
            F::BC6H_UFLOAT_BLOCK => F::R16G16B16A16_SFLOAT,
            _ => F::R8G8B8A8_UNORM,
        }
    }

    fn compress(
        &self,
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>> {
        let (base, _) = format.normalize();

        // The ISPC C API holds dimensions as int32. Reject pathological inputs
        // here instead of panicking inside the lower crate.
        check_i32_dims(surface.width, surface.height, surface.stride)?;

        let (data, width, height, stride) = (
            &*surface.data,
            surface.width,
            surface.height,
            surface.stride,
        );
        use ktx2::Format as F;
        match base {
            F::BC1_RGBA_UNORM_BLOCK => Ok(encode_unaligned(
                data,
                width,
                height,
                stride,
                4,
                8,
                |s, w, h, st, dst| {
                    let surf = itc::RgbaSurface::new(s, w, h, st);
                    itc::bc1::compress_blocks_into(&surf, dst);
                },
            )),
            F::BC3_UNORM_BLOCK => Ok(encode_unaligned(
                data,
                width,
                height,
                stride,
                4,
                16,
                |s, w, h, st, dst| {
                    let surf = itc::RgbaSurface::new(s, w, h, st);
                    itc::bc3::compress_blocks_into(&surf, dst);
                },
            )),
            F::BC4_UNORM_BLOCK => Ok(encode_unaligned(
                data,
                width,
                height,
                stride,
                1,
                8,
                |s, w, h, st, dst| {
                    let surf = itc::RSurface::new(s, w, h, st);
                    itc::bc4::compress_blocks_into(&surf, dst);
                },
            )),
            F::BC5_UNORM_BLOCK => Ok(encode_unaligned(
                data,
                width,
                height,
                stride,
                2,
                16,
                |s, w, h, st, dst| {
                    let surf = itc::RgSurface::new(s, w, h, st);
                    itc::bc5::compress_blocks_into(&surf, dst);
                },
            )),
            F::BC6H_UFLOAT_BLOCK => {
                let bc6_settings = bc6h_settings(quality);
                Ok(encode_unaligned(
                    data,
                    width,
                    height,
                    stride,
                    8,
                    16,
                    |s, w, h, st, dst| {
                        let surf = itc::RgbaF16Surface::new(s, w, h, st);
                        itc::bc6h::compress_blocks_into(&bc6_settings, &surf, dst);
                    },
                ))
            }
            F::BC7_UNORM_BLOCK => {
                let alpha = settings
                    .and_then(|s| s.as_any().downcast_ref::<IspcSettings>())
                    .is_some_and(|s| s.alpha);
                let bc7_settings = bc7_settings(quality, alpha);
                Ok(encode_unaligned(
                    data,
                    width,
                    height,
                    stride,
                    4,
                    16,
                    |s, w, h, st, dst| {
                        let surf = itc::RgbaSurface::new(s, w, h, st);
                        itc::bc7::compress_blocks_into(&bc7_settings, &surf, dst);
                    },
                ))
            }
            F::ETC2_R8G8B8_UNORM_BLOCK => {
                let etc_settings = etc1_settings();
                Ok(encode_unaligned(
                    data,
                    width,
                    height,
                    stride,
                    4,
                    8,
                    |s, w, h, st, dst| {
                        let surf = itc::RgbaSurface::new(s, w, h, st);
                        itc::etc1::compress_blocks_into(&etc_settings, &surf, dst);
                    },
                ))
            }
            _ => unreachable!("format not in supported_formats()"),
        }
    }
}

fn check_i32_dims(width: u32, height: u32, stride: u32) -> Result<()> {
    if i32::try_from(width).is_err()
        || i32::try_from(height).is_err()
        || i32::try_from(stride).is_err()
    {
        return Err(Error::InvalidDimensions(format!(
            "ISPC encoder requires width/height/stride to fit in i32, got {width}x{height} stride {stride}"
        )));
    }
    Ok(())
}

/// Encode an image whose dimensions may not be multiples of 4 by decomposing
/// into block-row calls.
///
/// For aligned dims this is a single zero-copy call. Otherwise:
/// - each interior block-row calls the encoder on the original source with
///   `width = floor(W/4)*4` (no pixel copy); then synthesizes a 4×4 scratch
///   for the right-edge block (one per row) if width isn't aligned,
/// - the bottom partial row builds one `ceil_W × 4` scratch with edge
///   replication and encodes it in one call.
///
/// Output is written directly into the final buffer — no intermediate
/// compressed scratch — since the ISPC kernels write single-row-of-blocks
/// calls starting at `dst + 0`.
fn encode_unaligned(
    data: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    bpp: u32,
    bytes_per_block: u32,
    encode_aligned: impl Fn(&[u8], u32, u32, u32, &mut [u8]),
) -> Vec<u8> {
    let bx = width.div_ceil(4);
    let by = height.div_ceil(4);
    let fx = width / 4;
    let fy = height / 4;

    let row_bytes = (bx * bytes_per_block) as usize;
    let block_bytes = bytes_per_block as usize;
    let mut out = vec![0u8; (bx as usize) * (by as usize) * block_bytes];

    // Fast path: fully aligned dimensions — single zero-copy call.
    if fx == bx && fy == by {
        encode_aligned(data, width, height, stride, &mut out);
        return out;
    }

    let edge_scratch_stride = 4 * bpp;
    let mut edge_scratch = vec![0u8; (4 * 4 * bpp) as usize];

    for by_idx in 0..fy {
        let src_row = &data[(by_idx * 4 * stride) as usize..];
        let dst_row = &mut out[by_idx as usize * row_bytes..(by_idx as usize + 1) * row_bytes];

        // Interior portion: fx blocks, zero-copy from original.
        if fx > 0 {
            let interior_end = (fx * bytes_per_block) as usize;
            encode_aligned(src_row, fx * 4, 4, stride, &mut dst_row[..interior_end]);
        }

        // Right-edge block (one per row) if the width isn't aligned.
        if fx < bx {
            edge::fill_clamped_block(
                data,
                width,
                height,
                stride,
                bpp,
                fx,
                by_idx,
                &mut edge_scratch,
            );
            let start = (fx * bytes_per_block) as usize;
            encode_aligned(
                &edge_scratch,
                4,
                4,
                edge_scratch_stride,
                &mut dst_row[start..],
            );
        }
    }

    // Bottom partial block-row (at most one), in a single call.
    if fy < by {
        let bottom_w = bx * 4;
        let bottom_stride = bottom_w * bpp;
        let mut bottom_scratch = vec![0u8; (bottom_w * 4 * bpp) as usize];
        edge::fill_clamped_block_row(
            data,
            width,
            height,
            stride,
            bpp,
            fy,
            bottom_w,
            &mut bottom_scratch,
        );
        let dst_start = fy as usize * row_bytes;
        encode_aligned(
            &bottom_scratch,
            bottom_w,
            4,
            bottom_stride,
            &mut out[dst_start..],
        );
    }

    out
}

fn bc6h_settings(quality: Quality) -> itc::bc6h::EncodeSettings {
    match quality {
        Quality::UltraFast | Quality::VeryFast => itc::bc6h::very_fast_settings(),
        Quality::Fast => itc::bc6h::fast_settings(),
        Quality::Basic => itc::bc6h::basic_settings(),
        Quality::Slow => itc::bc6h::slow_settings(),
        Quality::VerySlow => itc::bc6h::very_slow_settings(),
    }
}

fn bc7_settings(quality: Quality, alpha: bool) -> itc::bc7::EncodeSettings {
    match (alpha, quality) {
        (false, Quality::UltraFast) => itc::bc7::opaque_ultra_fast_settings(),
        (false, Quality::VeryFast) => itc::bc7::opaque_very_fast_settings(),
        (false, Quality::Fast) => itc::bc7::opaque_fast_settings(),
        (false, Quality::Basic) => itc::bc7::opaque_basic_settings(),
        (false, Quality::Slow | Quality::VerySlow) => itc::bc7::opaque_slow_settings(),
        (true, Quality::UltraFast) => itc::bc7::alpha_ultra_fast_settings(),
        (true, Quality::VeryFast) => itc::bc7::alpha_very_fast_settings(),
        (true, Quality::Fast) => itc::bc7::alpha_fast_settings(),
        (true, Quality::Basic) => itc::bc7::alpha_basic_settings(),
        (true, Quality::Slow | Quality::VerySlow) => itc::bc7::alpha_slow_settings(),
    }
}

fn etc1_settings() -> itc::etc1::EncodeSettings {
    itc::etc1::slow_settings()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    fn solid_red_surface(width: u32, height: u32) -> Surface {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            data.extend_from_slice(&[255, 0, 0, 255]);
        }
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride: width * 4,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    #[cfg(feature = "encoder-etcpak")]
    #[test]
    fn bc7_non_aligned_5x5() {
        let surface = solid_red_surface(5, 5);
        let encoder = IspcEncoder;
        let out = encoder
            .compress(
                &surface,
                ktx2::Format::BC7_UNORM_BLOCK,
                Quality::UltraFast,
                None,
            )
            .unwrap();
        // 5x5 rounds up to 8x8 blocks: 2×2 = 4 blocks × 16 bytes = 64 bytes.
        assert_eq!(out.len(), 2 * 2 * 16);
        // Every block should decode to close-to-red. Borrow the etcpak decoder.
        let decoded = ctt_etcpak::decode::decode_bc7(&out, 8, 8);
        for pixel in decoded.chunks_exact(4) {
            // BC7 ultrafast should still keep red channel > 200 for solid red.
            assert!(
                pixel[0] > 200,
                "decoded pixel R={} should be near 255",
                pixel[0]
            );
        }
    }

    #[cfg(feature = "encoder-amd")]
    #[test]
    fn bc1_non_aligned_7x3() {
        // compressonator has a working per-block BC1 decoder (etcpak's
        // DecodeBc1 is upstream-buggy — it writes only the first row of each
        // block), so round-trip through it.
        let surface = solid_red_surface(7, 3);
        let encoder = IspcEncoder;
        let out = encoder
            .compress(
                &surface,
                ktx2::Format::BC1_RGBA_UNORM_BLOCK,
                Quality::UltraFast,
                None,
            )
            .unwrap();
        // 7×3 rounds up to 8×4 pixels = 2 blocks × 8 bytes = 16 bytes.
        assert_eq!(out.len(), 2 * 8);
        for chunk in out.chunks_exact(8) {
            let block: [u8; 8] = chunk.try_into().unwrap();
            let decoded = ctt_compressonator::bc1::decompress_block(&block).unwrap();
            for pixel in decoded.chunks_exact(4) {
                assert!(pixel[0] > 200, "edge-block decodes R={}", pixel[0]);
            }
        }
    }

    #[test]
    fn aligned_fast_path_matches_single_call() {
        // 4×4 aligned image: fast path and slow path should both work.
        let surface = solid_red_surface(4, 4);
        let encoder = IspcEncoder;
        let out = encoder
            .compress(
                &surface,
                ktx2::Format::BC7_UNORM_BLOCK,
                Quality::UltraFast,
                None,
            )
            .unwrap();
        assert_eq!(out.len(), 16);
    }
}
