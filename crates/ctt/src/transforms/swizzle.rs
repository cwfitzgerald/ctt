use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::Result;
use crate::format::ColorSpace;
use crate::surface::{Image, Surface};
use crate::transform::swizzle::{apply_swizzle, Swizzle};
use crate::transform_node::{LayoutInfo, Transform};

/// A transform that remaps RGBA channels according to a swizzle pattern.
pub struct SwizzleTransform {
    swizzle: Swizzle,
}

impl SwizzleTransform {
    pub fn new(swizzle: Swizzle) -> Self {
        Self { swizzle }
    }
}

impl Transform for SwizzleTransform {
    fn name(&self) -> &str {
        "swizzle"
    }

    fn constraint(&self) -> FormatConstraint {
        // Swizzle requires RGBA uncompressed formats with U8/U16/F16/F32
        FormatConstraint {
            formats: Some(vec![
                ktx2::Format::R8G8B8A8_UNORM,
                ktx2::Format::R16G16B16A16_UNORM,
                ktx2::Format::R16G16B16A16_SFLOAT,
                ktx2::Format::R32G32B32A32_SFLOAT,
            ]),
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

    fn output_layout(&self, input: &LayoutInfo) -> LayoutInfo {
        input.clone()
    }

    fn execute(&self, image: Image) -> Result<Image> {
        if self.swizzle == Swizzle::IDENTITY {
            log::debug!("Swizzle: identity, skipping");
            return Ok(image);
        }

        log::info!("Applying swizzle: {:?}", self.swizzle.0);

        let mut new_surfaces = Vec::with_capacity(image.surfaces.len());
        for layer in &image.surfaces {
            let mut new_mips = Vec::with_capacity(layer.len());
            for surface in layer {
                let mut raw = surface.to_raw_image()?;
                apply_swizzle(&mut raw, &self.swizzle)?;
                let mut result = Surface::from_raw_image(raw);
                result.color_space = surface.color_space;
                result.alpha = surface.alpha;
                new_mips.push(result);
            }
            new_surfaces.push(new_mips);
        }

        Ok(Image {
            surfaces: new_surfaces,
            is_cubemap: image.is_cubemap,
        })
    }
}
