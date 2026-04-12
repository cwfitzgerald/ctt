use ctt_compressonator as cmp;

use crate::encoders::{Encoder, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::surface::Surface;
use crate::vk_format::FormatExt as _;

pub struct CompressonatorEncoder;

impl Encoder for CompressonatorEncoder {
    fn name(&self) -> &str {
        "amd"
    }

    fn supported_formats(&self) -> &[ktx2::Format] {
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

    fn required_input_format(&self, format: ktx2::Format) -> ktx2::Format {
        use ktx2::Format as F;
        match format {
            F::BC4_UNORM_BLOCK | F::BC4_SNORM_BLOCK => F::R8_UNORM,
            F::BC5_UNORM_BLOCK | F::BC5_SNORM_BLOCK => F::R8G8_UNORM,
            F::BC6H_UFLOAT_BLOCK | F::BC6H_SFLOAT_BLOCK => F::R16G16B16_SFLOAT,
            _ => F::R8G8B8A8_UNORM,
        }
    }

    fn compress(
        &self,
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        _settings: Option<&dyn EncoderSettings>,
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
