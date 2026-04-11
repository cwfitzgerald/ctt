use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::Result;
use crate::surface::ColorSpace;
use crate::surface::Image;
use crate::transforms::Transform;

/// A no-op transform that constrains the pipeline to a target format state.
///
/// Any combination of format, color space, and alpha mode can be constrained independently.
/// The pipeline resolver will automatically insert conversion steps before this transform
/// to satisfy its constraints. The transform itself is an identity operation.
pub struct OutputStateTransform {
    target_format: Option<ktx2::Format>,
    target_color_space: Option<ColorSpace>,
    target_alpha: Option<AlphaMode>,
}

impl OutputStateTransform {
    pub fn new(
        target_format: Option<ktx2::Format>,
        target_color_space: Option<ColorSpace>,
        target_alpha: Option<AlphaMode>,
    ) -> Self {
        Self {
            target_format,
            target_color_space,
            target_alpha,
        }
    }
}

impl Transform for OutputStateTransform {
    fn name(&self) -> &str {
        "output_state"
    }

    fn constraint(&self) -> FormatConstraint {
        FormatConstraint {
            formats: self.target_format.map(|f| vec![f]),
            color_spaces: self.target_color_space.map(|cs| vec![cs]),
            alpha_modes: self.target_alpha.map(|a| vec![a]),
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
