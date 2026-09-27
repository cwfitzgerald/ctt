//! Compressed-in == compressed-out fast path.
//!
//! When the input is already compressed in the target format we skip decode/
//! re-encode and hand the image straight to the container encoder.

use crate::convert::Container;
use crate::error::{Error, Result};
use crate::surface::{Image, ImageRef};

use super::PipelineOutput;

/// Run the passthrough path for a compressed input whose format already
/// matches the target.
pub fn run(
    image: ImageRef<'_>,
    target_format: ktx2::Format,
    container: Container,
) -> Result<PipelineOutput> {
    let first_fmt = image.desc.format;

    if first_fmt != target_format {
        return Err(Error::UnsupportedConversion(format!(
            "passthrough: input format {first_fmt:?} does not match target {target_format:?}"
        )));
    }

    emit_ref(image, container)
}

/// Encode an image into the requested container, or return it raw.
pub fn emit(image: Image, container: Container) -> Result<PipelineOutput> {
    match container {
        Container::Raw => Ok(PipelineOutput::Raw(image)),
        _ => emit_ref(image.to_ref(), container),
    }
}

/// Encode a borrowed image into the requested container, or return an owned
/// copy for [`Container::Raw`].
fn emit_ref(image: ImageRef<'_>, container: Container) -> Result<PipelineOutput> {
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
        Container::Raw => Ok(PipelineOutput::Raw(image.to_owned())),
    }
}
