use std::sync::Arc;

use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::encoder::{Encoder, EncoderRegistry, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::surface::{ColorSpace, Image, Surface};
use crate::transforms::Transform;
use crate::vk_format::FormatExt;

/// A transform that compresses uncompressed surfaces using a block encoder.
pub struct CompressTransform {
    target_format: ktx2::Format,
    quality: Quality,
    encoder_name: Option<String>,
    encoder_settings: Option<Box<dyn EncoderSettings>>,
    registry: Arc<EncoderRegistry>,
}

impl CompressTransform {
    pub fn new(
        target_format: ktx2::Format,
        quality: Quality,
        encoder_name: Option<String>,
        encoder_settings: Option<Box<dyn EncoderSettings>>,
        registry: Arc<EncoderRegistry>,
    ) -> Self {
        Self {
            target_format,
            quality,
            encoder_name,
            encoder_settings,
            registry,
        }
    }

    fn resolve_encoder(&self) -> Result<&dyn Encoder> {
        if let Some(ref name) = self.encoder_name {
            self.registry
                .find_by_name(name, self.target_format)
                .ok_or_else(|| {
                    Error::UnsupportedFormat(format!(
                        "no encoder named '{}' supports {:?}",
                        name, self.target_format
                    ))
                })
        } else {
            self.registry.find(self.target_format).ok_or_else(|| {
                Error::UnsupportedFormat(format!("no encoder supports {:?}", self.target_format))
            })
        }
    }
}

impl Transform for CompressTransform {
    fn name(&self) -> &str {
        "compress"
    }

    fn constraint(&self) -> FormatConstraint {
        let Ok(encoder) = self.resolve_encoder() else {
            return FormatConstraint {
                formats: Some(Vec::new()),
                color_spaces: None,
                alpha_modes: None,
            };
        };

        let required = encoder.required_input_format(self.target_format);
        let (required, _) = required.normalize();

        FormatConstraint {
            formats: Some(vec![required]),
            color_spaces: None,
            alpha_modes: None,
        }
    }

    fn output_format(
        &self,
        _input: ktx2::Format,
        cs: ColorSpace,
        alpha: AlphaMode,
    ) -> (ktx2::Format, ColorSpace, AlphaMode) {
        (self.target_format, cs, alpha)
    }

    fn execute(&self, image: Image) -> Result<Image> {
        let encoder = self.resolve_encoder()?;
        let settings_ref = self.encoder_settings.as_deref();

        let mut new_surfaces = Vec::with_capacity(image.surfaces.len());
        for (layer_idx, layer) in image.surfaces.iter().enumerate() {
            let mut new_mips = Vec::with_capacity(layer.len());
            for (mip_idx, surface) in layer.iter().enumerate() {
                log::debug!(
                    "Compressing layer {layer_idx}, mip {mip_idx}: {}x{} to {:?} using {}",
                    surface.width,
                    surface.height,
                    self.target_format,
                    encoder.name(),
                );

                // Pass the denormalized format so the encoder can recover color space.
                let output_format = self.target_format.denormalize(surface.color_space);
                let data = encoder.compress(surface, output_format, self.quality, settings_ref)?;

                let bpp_block = self.target_format.bytes_per_block().unwrap_or(16) as u32;
                let (bw, _bh) = self.target_format.block_size().unwrap_or((4, 4));
                let blocks_x = surface.width.div_ceil(bw as u32);

                new_mips.push(Surface {
                    data,
                    width: surface.width,
                    height: surface.height,
                    stride: blocks_x * bpp_block,
                    format: self.target_format,
                    color_space: surface.color_space,
                    alpha: surface.alpha,
                });
            }
            new_surfaces.push(new_mips);
        }

        Ok(Image {
            surfaces: new_surfaces,
            is_cubemap: image.is_cubemap,
        })
    }
}
