use crate::constraint::FormatConstraint;
use crate::alpha::AlphaMode;
use crate::error::Result;
use crate::format::ColorSpace;
use crate::surface::Image;

/// Describes the layer/mip structure of an image without the actual data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutInfo {
    pub layer_count: usize,
    pub mip_count: usize,
    pub is_cubemap: bool,
}

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

    /// How does this transform change the layer/mip structure?
    ///
    /// Most transforms return the input unchanged. Mip generation would increase `mip_count`.
    fn output_layout(&self, input: &LayoutInfo) -> LayoutInfo {
        input.clone()
    }

    /// Execute the transform on an image.
    fn execute(&self, image: Image) -> Result<Image>;
}
