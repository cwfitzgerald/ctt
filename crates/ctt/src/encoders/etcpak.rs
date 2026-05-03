use ctt_etcpak as ep;

use crate::encoders::Quality;
use crate::encoders::backend::Encoder;
use crate::encoders::edge;
use crate::error::Result;
use crate::surface::Surface;
use crate::vk_format::FormatExt as _;

/// etcpak-specific encoder settings.
#[derive(Debug, Clone, Copy, Default)]
pub struct EtcpakSettings {
    /// Enable dithering for ETC1 and BC1 compression.
    pub dither: bool,
    /// Enable heuristic-based fast compression mode selection for ETC2 RGB/RGBA.
    pub use_heuristics: bool,
}

pub struct EtcpakEncoder;

impl Encoder for EtcpakEncoder {
    type Settings = EtcpakSettings;

    fn name() -> &'static str {
        "etcpak"
    }

    fn supported_formats() -> &'static [ktx2::Format] {
        &[
            ktx2::Format::ETC2_R8G8B8_UNORM_BLOCK,
            ktx2::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
            ktx2::Format::EAC_R11_UNORM_BLOCK,
            ktx2::Format::EAC_R11G11_UNORM_BLOCK,
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            ktx2::Format::BC3_UNORM_BLOCK,
            ktx2::Format::BC4_UNORM_BLOCK,
            ktx2::Format::BC5_UNORM_BLOCK,
        ]
    }

    fn required_input_format(format: ktx2::Format) -> ktx2::Format {
        use ktx2::Format as F;
        match format {
            // ETC/EAC codecs expect BGRA pixel layout.
            F::ETC2_R8G8B8_UNORM_BLOCK
            | F::ETC2_R8G8B8A8_UNORM_BLOCK
            | F::EAC_R11_UNORM_BLOCK
            | F::EAC_R11G11_UNORM_BLOCK => F::B8G8R8A8_UNORM,
            // BC codecs expect RGBA pixel layout.
            _ => F::R8G8B8A8_UNORM,
        }
    }

    fn compress(
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: &EtcpakSettings,
    ) -> Result<Vec<u8>> {
        let (base, _) = format.normalize();

        let use_heuristics = settings.use_heuristics;
        let dither = settings.dither;

        let bytes_per_block =
            base.bytes_per_block()
                .expect("supported etcpak format has known block size") as u32;

        use ktx2::Format as F;
        let encode_aligned = |s: &[u8], w: u32, h: u32, dst: &mut [u8]| {
            let surf = ep::Surface::new(s, w, h);
            match base {
                F::ETC2_R8G8B8_UNORM_BLOCK => {
                    if dither {
                        ep::etc1::compress_blocks_dither_into(&surf, dst);
                    } else {
                        match quality {
                            // ETC1 is faster, ETC2 is higher quality.
                            Quality::UltraFast | Quality::VeryFast | Quality::Fast => {
                                ep::etc1::compress_blocks_into(&surf, dst);
                            }
                            _ => ep::etc2_rgb::compress_blocks_into(&surf, use_heuristics, dst),
                        }
                    }
                }
                F::ETC2_R8G8B8A8_UNORM_BLOCK => {
                    ep::etc2_rgba::compress_blocks_into(&surf, use_heuristics, dst);
                }
                F::EAC_R11_UNORM_BLOCK => ep::eac_r::compress_blocks_into(&surf, dst),
                F::EAC_R11G11_UNORM_BLOCK => ep::eac_rg::compress_blocks_into(&surf, dst),
                F::BC1_RGBA_UNORM_BLOCK => {
                    if dither {
                        ep::bc1::compress_blocks_dither_into(&surf, dst);
                    } else {
                        ep::bc1::compress_blocks_into(&surf, dst);
                    }
                }
                F::BC3_UNORM_BLOCK => ep::bc3::compress_blocks_into(&surf, dst),
                F::BC4_UNORM_BLOCK => ep::bc4::compress_blocks_into(&surf, dst),
                F::BC5_UNORM_BLOCK => ep::bc5::compress_blocks_into(&surf, dst),
                _ => unreachable!("format not in supported_formats()"),
            }
        };

        Ok(encode_with_edges(
            &surface.data,
            surface.width,
            surface.height,
            surface.stride,
            bytes_per_block,
            encode_aligned,
        ))
    }
}

/// etcpak requires tight-packed, multiple-of-4 input. For aligned dims this
/// is a zero-copy direct call. For unaligned dims we can't avoid copying
/// interior pixels — etcpak has no stride parameter — so we walk the image
/// one block-row at a time through a reusable `ceil_W × 4` scratch.
///
/// The output buffer is allocated at final size and written directly (each
/// block-row call's output lands at its final offset, since etcpak writes
/// `blocks` entries sequentially starting at the passed `dst` pointer).
fn encode_with_edges(
    data: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    bytes_per_block: u32,
    encode: impl Fn(&[u8], u32, u32, &mut [u8]),
) -> Vec<u8> {
    // etcpak's fixed pixel layout is 4 bytes/pixel for every format it exposes.
    const BPP: u32 = 4;

    let bx = width.div_ceil(4);
    let by = height.div_ceil(4);
    let block_bytes = bytes_per_block as usize;
    let row_bytes = (bx * bytes_per_block) as usize;
    let mut out = vec![0u8; (bx as usize) * (by as usize) * block_bytes];

    // Fully aligned and tight-packed: single zero-copy call.
    if width.is_multiple_of(4) && height.is_multiple_of(4) && stride == width * BPP {
        encode(data, width, height, &mut out);
        return out;
    }

    // Reusable per-row scratch: ceil_W × 4 tightly packed.
    let scratch_w = bx * 4;
    let mut scratch = vec![0u8; (scratch_w * 4 * BPP) as usize];

    for by_idx in 0..by {
        edge::fill_clamped_block_row(
            data,
            width,
            height,
            stride,
            BPP,
            by_idx,
            scratch_w,
            &mut scratch,
        );
        let dst_start = by_idx as usize * row_bytes;
        encode(
            &scratch,
            scratch_w,
            4,
            &mut out[dst_start..dst_start + row_bytes],
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    fn solid_surface(width: u32, height: u32, pixel: [u8; 4]) -> Surface {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            data.extend_from_slice(&pixel);
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

    #[test]
    fn bc3_non_aligned_7x5() {
        // BC3 has a working etcpak decoder, so round-trip solid red.
        let surface = solid_surface(7, 5, [255, 0, 0, 255]);
        let out = EtcpakEncoder::compress(
            &surface,
            ktx2::Format::BC3_UNORM_BLOCK,
            Quality::Fast,
            &EtcpakSettings::default(),
        )
        .unwrap();
        // 7×5 → 8×8 → 2×2 blocks × 16 bytes = 64 bytes.
        assert_eq!(out.len(), 2 * 2 * 16);
        let decoded = ctt_etcpak::decode::decode_bc3(&out, 8, 8);
        for pixel in decoded.chunks_exact(4) {
            assert!(pixel[0] > 200, "decoded BC3 edge pixel R={}", pixel[0]);
        }
    }

    #[test]
    fn etc2_rgba_non_aligned_5x5() {
        // etcpak expects BGRA input for ETC codecs. Encode solid white which is
        // palette-invariant, then verify the round-trip through decode_rgba.
        let surface = solid_surface(5, 5, [255, 255, 255, 255]);
        let out = EtcpakEncoder::compress(
            &surface,
            ktx2::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
            Quality::Fast,
            &EtcpakSettings::default(),
        )
        .unwrap();
        // 5×5 → 8×8 → 4 blocks × 16 bytes = 64 bytes.
        assert_eq!(out.len(), 4 * 16);
        let decoded = ctt_etcpak::decode::decode_rgba(&out, 8, 8);
        for pixel in decoded.chunks_exact(4) {
            // etcpak decodes to BGRA — white is invariant.
            assert!(
                pixel[0] > 240 && pixel[1] > 240 && pixel[2] > 240,
                "ETC2 decode {:?}",
                pixel
            );
        }
    }

    #[test]
    fn aligned_zero_copy_path() {
        // 8×8 is aligned and tight-packed → fast path direct call.
        let surface = solid_surface(8, 8, [128, 128, 128, 255]);
        let out = EtcpakEncoder::compress(
            &surface,
            ktx2::Format::BC3_UNORM_BLOCK,
            Quality::Fast,
            &EtcpakSettings::default(),
        )
        .unwrap();
        assert_eq!(out.len(), 4 * 16);
    }
}
