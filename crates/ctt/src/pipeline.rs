use crate::compress::compress_layout;
use crate::config::{CompressConfig, EncodeSettings, OutputFormat};
use crate::error::{Error, Result};
use crate::format::CompressedFormat;
use crate::image::ImageLayout;
use crate::output::{dds, ktx2};
use crate::transform::convert::convert_image;
use crate::transform::swizzle::apply_swizzle;

/// Run the full compression pipeline: swizzle -> convert -> compress -> encode.
///
/// Returns the encoded file bytes (DDS or KTX2).
pub fn run(config: &CompressConfig, mut layout: ImageLayout) -> Result<Vec<u8>> {
    validate(config, &layout)?;

    // Apply swizzle if configured.
    if let Some(ref swizzle) = config.swizzle {
        log::info!("Applying swizzle: {:?}", swizzle.0);
        for layer in &mut layout.layers {
            for image in layer {
                apply_swizzle(image, swizzle)?;
            }
        }
    }

    // Convert images to the format required by the compressor.
    let required_format = config.format.required_input_format(config.color_space);
    log::info!("Converting images to {}", required_format);
    for layer in &mut layout.layers {
        for image in layer {
            if image.pixel_format != required_format {
                log::debug!("Converting {} -> {}", image.pixel_format, required_format);
                *image = convert_image(image, required_format)?;
            } else {
                log::debug!("Already in required format");
            }
        }
    }

    // Compress all layers and mip levels.
    let settings = config
        .encode_settings
        .unwrap_or_else(|| EncodeSettings::default_for(config.format));
    log::info!("Compressing as {:?}", config.format);
    let compressed = compress_layout(&layout, config.format, config.color_space, &settings)?;

    // Encode to the requested output format.
    log::info!("Encoding as {:?}", config.output_format);
    match config.output_format {
        OutputFormat::Dds => dds::encode_dds(&compressed),
        OutputFormat::Ktx2 => ktx2::encode_ktx2(&compressed),
    }
}

/// Validate the config and layout before doing any work.
fn validate(config: &CompressConfig, layout: &ImageLayout) -> Result<()> {
    // ETC1 and ASTC are not supported in DDS.
    if config.output_format == OutputFormat::Dds {
        match config.format {
            CompressedFormat::Etc1 => {
                return Err(Error::UnsupportedFormat(
                    "ETC1 is not supported in DDS".into(),
                ));
            }
            CompressedFormat::Astc { .. } => {
                return Err(Error::UnsupportedFormat(
                    "ASTC is not supported in DDS".into(),
                ));
            }
            _ => {}
        }
    }

    // ASTC compression is not implemented.
    if matches!(config.format, CompressedFormat::Astc { .. }) {
        return Err(Error::CompressionNotImplemented(config.format));
    }

    // Cubemaps must have exactly 6 layers.
    if layout.is_cubemap && layout.layers.len() != 6 {
        return Err(Error::CubemapFaceCount(layout.layers.len()));
    }

    Ok(())
}
