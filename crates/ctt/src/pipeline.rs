use crate::compress::compress_layout;
use crate::config::{CompressConfig, OutputFormat};
use crate::error::Result;
use crate::image::ImageLayout;
use crate::output::{dds, ktx2};
use crate::transform::swizzle::apply_swizzle;

/// Run the full compression pipeline: swizzle -> compress -> encode.
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

    // Compress all layers and mip levels.
    let compressed = compress_layout(&layout, config.format, config.color_space)?;

    // Encode to the requested output format.
    match config.output_format {
        OutputFormat::Dds => dds::encode_dds(&compressed),
        OutputFormat::Ktx2 => ktx2::encode_ktx2(&compressed),
    }
}
