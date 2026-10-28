//! Compressed-in == compressed-out fast path.
//!
//! When the input is already compressed in the target format we skip decode/
//! re-encode and hand the image straight to the caller.

use crate::error::{Error, Result};
use crate::surface::ImageRef;

/// Run the passthrough path for a compressed input whose format already
/// matches the target. Returns the input unchanged.
pub fn run(image: ImageRef<'_>, target_format: ktx2::Format) -> Result<ImageRef<'_>> {
    let first_fmt = image.desc.format;

    if first_fmt != target_format {
        return Err(Error::UnsupportedConversion(format!(
            "passthrough: input format {first_fmt:?} does not match target {target_format:?}"
        )));
    }

    Ok(image)
}
