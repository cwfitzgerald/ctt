use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::Result;
use crate::surface::ColorSpace;
use crate::surface::Image;

/// A single step in the transformation pipeline.
///
/// Each transform declares what format states it accepts ([`FormatConstraint`]) and what it
/// produces. The pipeline resolver uses this information to automatically insert format
/// conversions where needed.
pub trait Transform: Send + Sync {
    /// Human-readable name for error messages and logging.
    fn name(&self) -> &str;

    /// What format states this transform accepts as input.
    fn constraint(&self) -> FormatConstraint;

    /// Given an input format state, what does this transform produce?
    fn output_format(
        &self,
        input: ktx2::Format,
        cs: ColorSpace,
        alpha: AlphaMode,
    ) -> (ktx2::Format, ColorSpace, AlphaMode);

    /// Execute the transform on an image.
    fn execute(&self, image: Image) -> Result<Image>;
}

pub(crate) mod compress;
pub(crate) mod format_convert;
pub(crate) mod mipmap;
pub(crate) mod output_state;
pub(crate) mod swizzle;
pub(crate) mod target_format;

pub use compress::CompressTransform;
pub use mipmap::MipmapTransform;
pub use output_state::OutputStateTransform;
pub use swizzle::{SwizzleTransform, apply_swizzle};
pub use target_format::TargetFormatTransform;
