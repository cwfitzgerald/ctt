use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::conversion_graph::SurfaceConverter;
use crate::error::Result;
use crate::surface::ColorSpace;
use crate::surface::Image;
use crate::transform_node::{LayoutInfo, Transform};

/// A transform that converts between formats, inserted automatically by the pipeline resolver.
///
/// Each instance handles a single conversion step (one edge in the conversion graph).
pub struct FormatConvertTransform {
    target_format: ktx2::Format,
    target_color_space: ColorSpace,
    target_alpha: AlphaMode,
    converter: SurfaceConverter,
}

impl FormatConvertTransform {
    pub fn new(
        target_format: ktx2::Format,
        target_color_space: ColorSpace,
        target_alpha: AlphaMode,
        converter: SurfaceConverter,
    ) -> Self {
        Self {
            target_format,
            target_color_space,
            target_alpha,
            converter,
        }
    }
}

impl Transform for FormatConvertTransform {
    fn name(&self) -> &str {
        "format_convert"
    }

    fn constraint(&self) -> FormatConstraint {
        // This transform accepts anything — it's only inserted by the resolver
        // when the input format is known to be convertible.
        FormatConstraint::any()
    }

    fn output_format(
        &self,
        _input: ktx2::Format,
        _cs: ColorSpace,
        _alpha: AlphaMode,
    ) -> (ktx2::Format, ColorSpace, AlphaMode) {
        (
            self.target_format,
            self.target_color_space,
            self.target_alpha,
        )
    }

    fn output_layout(&self, input: &LayoutInfo) -> LayoutInfo {
        input.clone()
    }

    fn execute(&self, image: Image) -> Result<Image> {
        log::debug!(
            "Converting to {:?} ({:?}, {:?})",
            self.target_format,
            self.target_color_space,
            self.target_alpha
        );

        let mut new_surfaces = Vec::with_capacity(image.surfaces.len());
        for layer in &image.surfaces {
            let mut new_mips = Vec::with_capacity(layer.len());
            for surface in layer {
                new_mips.push((self.converter)(surface)?);
            }
            new_surfaces.push(new_mips);
        }

        Ok(Image {
            surfaces: new_surfaces,
            is_cubemap: image.is_cubemap,
        })
    }
}
