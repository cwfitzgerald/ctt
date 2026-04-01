use ctt_astcenc as astc;

use crate::encoder::{Encoder, EncoderSettings, Quality};
use crate::error::Result;
use crate::format::{ChannelType, ColorSpace, CompressedFormat, PixelComponents, PixelFormat};
use crate::image::RawImage;

/// All 14 valid ASTC 2D block sizes.
const SUPPORTED_FORMATS: &[CompressedFormat] = &[
    CompressedFormat::Astc {
        block_width: 4,
        block_height: 4,
    },
    CompressedFormat::Astc {
        block_width: 5,
        block_height: 4,
    },
    CompressedFormat::Astc {
        block_width: 5,
        block_height: 5,
    },
    CompressedFormat::Astc {
        block_width: 6,
        block_height: 5,
    },
    CompressedFormat::Astc {
        block_width: 6,
        block_height: 6,
    },
    CompressedFormat::Astc {
        block_width: 8,
        block_height: 5,
    },
    CompressedFormat::Astc {
        block_width: 8,
        block_height: 6,
    },
    CompressedFormat::Astc {
        block_width: 8,
        block_height: 8,
    },
    CompressedFormat::Astc {
        block_width: 10,
        block_height: 5,
    },
    CompressedFormat::Astc {
        block_width: 10,
        block_height: 6,
    },
    CompressedFormat::Astc {
        block_width: 10,
        block_height: 8,
    },
    CompressedFormat::Astc {
        block_width: 10,
        block_height: 10,
    },
    CompressedFormat::Astc {
        block_width: 12,
        block_height: 10,
    },
    CompressedFormat::Astc {
        block_width: 12,
        block_height: 12,
    },
];

pub struct AstcencEncoder;

impl Encoder for AstcencEncoder {
    fn name(&self) -> &str {
        "astcenc"
    }

    fn supported_formats(&self) -> &[CompressedFormat] {
        SUPPORTED_FORMATS
    }

    fn required_input_format(
        &self,
        _format: CompressedFormat,
        color_space: ColorSpace,
    ) -> PixelFormat {
        // astcenc LDR mode expects RGBA U8.
        PixelFormat {
            components: PixelComponents::Rgba,
            channel_type: ChannelType::U8,
            color_space,
        }
    }

    fn compress(
        &self,
        image: &RawImage,
        format: CompressedFormat,
        quality: Quality,
        _settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>> {
        let CompressedFormat::Astc {
            block_width,
            block_height,
        } = format
        else {
            unreachable!("AstcencEncoder only supports ASTC formats")
        };

        let profile = match image.pixel_format.color_space {
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
        let mut data_ptr = image.data.as_ptr() as *mut std::ffi::c_void;
        let mut img = astc::astcenc_image {
            dim_x: image.width,
            dim_y: image.height,
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
        let blocks_x = image.width.div_ceil(block_width as u32);
        let blocks_y = image.height.div_ceil(block_height as u32);
        let output_size = (blocks_x * blocks_y * 16) as usize;
        let mut output = vec![0u8; output_size];

        ctx.compress(&mut img, &swizzle, &mut output)
            .map_err(|e| crate::error::Error::Compression(e.to_string()))?;

        Ok(output)
    }
}
