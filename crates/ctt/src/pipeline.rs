use crate::compress::compress_layout;
use crate::config::{CompressConfig, EncodeSettings, OutputFormat};
use crate::error::Result;
use crate::image::ImageLayout;
use crate::output::{dds, ktx2};
use crate::transform::convert::convert_image;
use crate::transform::swizzle::apply_swizzle;

/// Run the full compression pipeline: swizzle -> convert -> compress -> encode.
///
/// Returns the encoded file bytes (DDS or KTX2).
pub fn run(config: &CompressConfig, mut layout: ImageLayout) -> Result<Vec<u8>> {
    // Apply swizzle if configured.
    if let Some(ref swizzle) = config.swizzle {
        for layer in &mut layout.layers {
            for image in layer {
                apply_swizzle(image, swizzle)?;
            }
        }
    }

    // Convert images to the format required by the compressor.
    let required_format = config.format.required_input_format(config.color_space);
    for layer in &mut layout.layers {
        for image in layer {
            if image.pixel_format != required_format {
                *image = convert_image(image, required_format)?;
            }
        }
    }

    // Compress all layers and mip levels.
    let settings = config
        .encode_settings
        .unwrap_or_else(|| EncodeSettings::default_for(config.format));
    let compressed = compress_layout(&layout, config.format, config.color_space, &settings)?;

    // Encode to the requested output format.
    match config.output_format {
        OutputFormat::Dds => dds::encode_dds(&compressed),
        OutputFormat::Ktx2 => ktx2::encode_ktx2(&compressed),
    }
}
