use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::Result;
use crate::surface::ColorSpace;
use crate::surface::Image;
use crate::transforms::Transform;

/// A no-op transform that constrains the pipeline to a specific uncompressed format.
///
/// The pipeline resolver will automatically insert format conversion steps before this
/// transform to satisfy its constraint. The transform itself is an identity operation.
pub struct TargetFormatTransform {
    target_format: ktx2::Format,
}

impl TargetFormatTransform {
    pub fn new(target_format: ktx2::Format) -> Self {
        Self { target_format }
    }
}

impl Transform for TargetFormatTransform {
    fn name(&self) -> &str {
        "target_format"
    }

    fn constraint(&self) -> FormatConstraint {
        FormatConstraint {
            formats: Some(vec![self.target_format]),
            color_spaces: None,
            alpha_modes: None,
        }
    }

    fn output_format(
        &self,
        input: ktx2::Format,
        cs: ColorSpace,
        alpha: AlphaMode,
    ) -> (ktx2::Format, ColorSpace, AlphaMode) {
        (input, cs, alpha)
    }

    fn execute(&self, image: Image) -> Result<Image> {
        Ok(image)
    }
}
