use ctt_astcenc as astc;

use crate::encoder::{Encoder, EncoderSettings, Quality};
use crate::error::Result;
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::FormatExt as _;

/// All 14 valid ASTC 2D block sizes.
const SUPPORTED_FORMATS: &[ktx2::Format] = &[
    ktx2::Format::ASTC_4x4_UNORM_BLOCK,
    ktx2::Format::ASTC_5x4_UNORM_BLOCK,
    ktx2::Format::ASTC_5x5_UNORM_BLOCK,
    ktx2::Format::ASTC_6x5_UNORM_BLOCK,
    ktx2::Format::ASTC_6x6_UNORM_BLOCK,
    ktx2::Format::ASTC_8x5_UNORM_BLOCK,
    ktx2::Format::ASTC_8x6_UNORM_BLOCK,
    ktx2::Format::ASTC_8x8_UNORM_BLOCK,
    ktx2::Format::ASTC_10x5_UNORM_BLOCK,
    ktx2::Format::ASTC_10x6_UNORM_BLOCK,
    ktx2::Format::ASTC_10x8_UNORM_BLOCK,
    ktx2::Format::ASTC_10x10_UNORM_BLOCK,
    ktx2::Format::ASTC_12x10_UNORM_BLOCK,
    ktx2::Format::ASTC_12x12_UNORM_BLOCK,
];

pub struct AstcencEncoder;

impl Encoder for AstcencEncoder {
    fn name(&self) -> &str {
        "astcenc"
    }

    fn supported_formats(&self) -> &[ktx2::Format] {
        SUPPORTED_FORMATS
    }

    fn required_input_format(&self, _format: ktx2::Format) -> ktx2::Format {
        // astcenc LDR mode expects RGBA U8.
        ktx2::Format::R8G8B8A8_UNORM
    }

    fn compress(
        &self,
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        _settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>> {
        let (base, color_space) = format.normalize();

        let (block_width, block_height) =
            base.block_size().expect("ASTC format must have block size");

        let profile = match color_space {
            ColorSpace::Srgb => astc::astcenc_profile_ASTCENC_PRF_LDR_SRGB,
            ColorSpace::Linear => astc::astcenc_profile_ASTCENC_PRF_LDR,
        };

        let quality_preset = match quality {
            Quality::UltraFast => astc::ASTCENC_PRE_FASTEST,
            Quality::VeryFast => astc::ASTCENC_PRE_FAST,
            Quality::Fast => astc::ASTCENC_PRE_MEDIUM,
            Quality::Basic => astc::ASTCENC_PRE_MEDIUM,
            Quality::Slow => astc::ASTCENC_PRE_THOROUGH,
            Quality::VerySlow => astc::ASTCENC_PRE_EXHAUSTIVE,
        };

        let config = astc::config_init(
            profile,
            block_width as u32,
            block_height as u32,
            1, // 2D (z=1)
            quality_preset,
            0, // no special flags
        )
        .map_err(|e| crate::error::Error::Compression(e.to_string()))?;

        let mut ctx = astc::Context::new(&config)
            .map_err(|e| crate::error::Error::Compression(e.to_string()))?;

        // Build the astcenc_image pointing at the raw pixel data.
        let mut data_ptr = surface.data.as_ptr() as *mut std::ffi::c_void;
        let mut img = astc::astcenc_image {
            dim_x: surface.width,
            dim_y: surface.height,
            dim_z: 1,
            data_type: astc::astcenc_type_ASTCENC_TYPE_U8,
            data: &mut data_ptr,
        };

        let swizzle = astc::astcenc_swizzle {
            r: astc::astcenc_swz_ASTCENC_SWZ_R,
            g: astc::astcenc_swz_ASTCENC_SWZ_G,
            b: astc::astcenc_swz_ASTCENC_SWZ_B,
            a: astc::astcenc_swz_ASTCENC_SWZ_A,
        };

        // ASTC block is always 16 bytes (128 bits).
        let blocks_x = surface.width.div_ceil(block_width as u32);
        let blocks_y = surface.height.div_ceil(block_height as u32);
        let output_size = (blocks_x * blocks_y * 16) as usize;
        let mut output = vec![0u8; output_size];

        ctx.compress(&mut img, &swizzle, &mut output)
            .map_err(|e| crate::error::Error::Compression(e.to_string()))?;

        Ok(output)
    }
}
