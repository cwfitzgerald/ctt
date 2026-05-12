use ctt_compressonator as cmp;

use crate::encoders::Quality;
use crate::encoders::backend::Encoder;
use crate::error::{Error, Result};
use crate::surface::Surface;
use crate::vk_format::FormatExt as _;

#[derive(Debug, Clone, Copy, Default)]
pub struct AmdSettings;

pub struct CompressonatorEncoder;

impl Encoder for CompressonatorEncoder {
    type Settings = AmdSettings;

    fn name() -> &'static str {
        "amd"
    }

    fn supported_formats() -> &'static [ktx2::Format] {
        &[
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            ktx2::Format::BC2_UNORM_BLOCK,
            ktx2::Format::BC3_UNORM_BLOCK,
            ktx2::Format::BC4_UNORM_BLOCK,
            ktx2::Format::BC4_SNORM_BLOCK,
            ktx2::Format::BC5_UNORM_BLOCK,
            ktx2::Format::BC5_SNORM_BLOCK,
            ktx2::Format::BC6H_UFLOAT_BLOCK,
            ktx2::Format::BC6H_SFLOAT_BLOCK,
            ktx2::Format::BC7_UNORM_BLOCK,
        ]
    }

    fn required_input_format(format: ktx2::Format, _settings: &AmdSettings) -> ktx2::Format {
        use ktx2::Format as F;
        match format {
            F::BC4_UNORM_BLOCK | F::BC4_SNORM_BLOCK => F::R8_UNORM,
            F::BC5_UNORM_BLOCK | F::BC5_SNORM_BLOCK => F::R8G8_UNORM,
            F::BC6H_UFLOAT_BLOCK | F::BC6H_SFLOAT_BLOCK => F::R16G16B16_SFLOAT,
            _ => F::R8G8B8A8_UNORM,
        }
    }

    fn compress(
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        _settings: &AmdSettings,
    ) -> Result<Vec<u8>> {
        let q = quality_to_float(quality);
        let (base, _) = format.normalize();
        let (data, width, height) = (&*surface.data, surface.width, surface.height);

        use ktx2::Format as F;
        match base {
            F::BC1_RGBA_UNORM_BLOCK => {
                let mut opts = cmp::bc1::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc1::compress_blocks(data, width, height, &opts).map_err(cmp_err)
            }
            F::BC2_UNORM_BLOCK => {
                let mut opts = cmp::bc2::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc2::compress_blocks(data, width, height, &opts).map_err(cmp_err)
            }
            F::BC3_UNORM_BLOCK => {
                let mut opts = cmp::bc3::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc3::compress_blocks(data, width, height, &opts).map_err(cmp_err)
            }
            F::BC4_UNORM_BLOCK => {
                let mut opts = cmp::bc4::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc4::compress_blocks(data, width, height, &opts).map_err(cmp_err)
            }
            F::BC4_SNORM_BLOCK => {
                let mut opts = cmp::bc4::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let src: &[i8] = bytemuck::cast_slice(data);
                cmp::bc4s::compress_blocks(src, width, height, &opts).map_err(cmp_err)
            }
            F::BC5_UNORM_BLOCK => {
                let mut opts = cmp::bc5::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc5::compress_blocks(data, width, height, &opts).map_err(cmp_err)
            }
            F::BC5_SNORM_BLOCK => {
                let mut opts = cmp::bc5::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let src: &[i8] = bytemuck::cast_slice(data);
                cmp::bc5s::compress_blocks(src, width, height, &opts).map_err(cmp_err)
            }
            F::BC6H_UFLOAT_BLOCK => {
                let mut opts = cmp::bc6h::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let src: &[u16] = bytemuck::cast_slice(data);
                cmp::bc6h::compress_blocks(src, width, height, &opts).map_err(cmp_err)
            }
            F::BC6H_SFLOAT_BLOCK => {
                let mut opts = cmp::bc6h::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                opts.set_signed(true).map_err(cmp_err)?;
                let src: &[u16] = bytemuck::cast_slice(data);
                cmp::bc6h::compress_blocks(src, width, height, &opts).map_err(cmp_err)
            }
            F::BC7_UNORM_BLOCK => {
                let mut opts = cmp::bc7::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                cmp::bc7::compress_blocks(data, width, height, &opts).map_err(cmp_err)
            }
            _ => unreachable!("format not in supported_formats()"),
        }
    }
}

fn quality_to_float(quality: Quality) -> f32 {
    match quality {
        Quality::UltraFast => 0.01,
        Quality::VeryFast => 0.05,
        Quality::Fast => 0.1,
        Quality::Basic => 0.5,
        Quality::Slow => 0.8,
        Quality::VerySlow => 1.0,
    }
}

fn cmp_err(e: cmp::Error) -> Error {
    Error::Compression(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    fn solid_red(width: u32, height: u32) -> Surface {
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

    #[test]
    fn bc7_non_aligned_5x5() {
        let surface = solid_red(5, 5);
        // Compressonator BC7 at UltraFast produces R=0 output on non-MSVC
        // toolchains (Linux, macOS). Use Slow so the NPOT coverage is
        // independent of that upstream quirk.
        let out = CompressonatorEncoder::compress(
            &surface,
            ktx2::Format::BC7_UNORM_BLOCK,
            Quality::Slow,
            &AmdSettings,
        )
        .unwrap();
        // 5x5 → 8×8 → 4 blocks × 16 bytes.
        assert_eq!(out.len(), 4 * 16);
        for chunk in out.chunks_exact(16) {
            let block: [u8; 16] = chunk.try_into().unwrap();
            let decoded = ctt_compressonator::bc7::decompress_block(&block).unwrap();
            for pixel in decoded.chunks_exact(4) {
                assert!(pixel[0] > 200, "compressonator BC7 edge R={}", pixel[0]);
            }
        }
    }

    #[test]
    fn bc1_non_aligned_7x3() {
        let surface = solid_red(7, 3);
        let out = CompressonatorEncoder::compress(
            &surface,
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            Quality::UltraFast,
            &AmdSettings,
        )
        .unwrap();
        // 7×3 → 8×4 → 2 blocks × 8 bytes.
        assert_eq!(out.len(), 2 * 8);
        for chunk in out.chunks_exact(8) {
            let block: [u8; 8] = chunk.try_into().unwrap();
            let decoded = ctt_compressonator::bc1::decompress_block(&block).unwrap();
            for pixel in decoded.chunks_exact(4) {
                assert!(pixel[0] > 200, "compressonator BC1 edge R={}", pixel[0]);
            }
        }
    }
}
