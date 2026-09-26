//! Compressed-in == compressed-out fast path.
//!
//! When the input is already compressed in the target format we skip decode/
//! re-encode and hand the image straight to the container encoder.

use crate::convert::Container;
use crate::error::{Error, Result};
use crate::surface::Image;

use super::PipelineOutput;

/// Run the passthrough path for a compressed input whose format already
/// matches the target.
pub fn run(
    image: Image,
    target_format: ktx2::Format,
    container: Container,
) -> Result<PipelineOutput> {
    let first_fmt = image.surfaces[0][0].format;

    if first_fmt != target_format {
        return Err(Error::UnsupportedConversion(format!(
            "passthrough: input format {first_fmt:?} does not match target {target_format:?}"
        )));
    }

    emit(image, container)
}

/// Encode an image into the requested container, or return it raw.
pub fn emit(image: Image, container: Container) -> Result<PipelineOutput> {
    match container {
        Container::Dds => {
            profiling::scope!("encode_dds");
            let bytes = crate::output::dds::encode_dds_image(&image)?;
            Ok(PipelineOutput::Encoded(bytes))
        }
        Container::Ktx2(sc) => {
            profiling::scope!("encode_ktx2");
            let bytes = crate::output::ktx2::encode_ktx2_image(&image, sc)?;
            Ok(PipelineOutput::Encoded(bytes))
        }
        Container::Raw => Ok(PipelineOutput::Raw(image)),
    }
}
