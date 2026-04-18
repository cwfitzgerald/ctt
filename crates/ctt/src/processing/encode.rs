//! Block-compression step — runs an [`Encoder`] across every surface in an
//! [`Image`].
//!
//! Replaces the old `CompressTransform`. Input must already be the format
//! the encoder requires (typically `R8G8B8A8_UNORM` / `R32G32B32A32_SFLOAT`).

use std::sync::Arc;

use crate::encoders::{Encoder, EncoderRegistry, EncoderSettings};
use crate::error::{Error, Result};
use crate::quality::Quality;
use crate::surface::{Image, Surface};
use crate::vk_format::FormatExt;

/// A resolved encoder step: what to encode to, which encoder, what settings.
pub struct EncoderStep {
    pub target_format: ktx2::Format,
    pub encoder_name: Option<String>,
    pub quality: Quality,
    pub settings: Option<Box<dyn EncoderSettings>>,
    pub registry: Arc<EncoderRegistry>,
}

impl EncoderStep {
    fn resolve(&self) -> Result<&dyn Encoder> {
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

    /// The uncompressed format the encoder requires as input.
    pub fn required_input(&self) -> Result<ktx2::Format> {
        let encoder = self.resolve()?;
        let (required, _) = encoder
            .required_input_format(self.target_format)
            .normalize();
        Ok(required)
    }
}

/// Run the encoder across every surface of the image.
pub fn encode_all(image: Image, step: &EncoderStep) -> Result<Image> {
    profiling::scope!("encode_all");
    let encoder = step.resolve()?;
    let settings_ref = step.settings.as_deref();

    let mut new_surfaces = Vec::with_capacity(image.surfaces.len());
    for (layer_idx, layer) in image.surfaces.iter().enumerate() {
        let mut new_mips = Vec::with_capacity(layer.len());
        for (mip_idx, surface) in layer.iter().enumerate() {
            profiling::scope!("encode_mip", encoder.name());
            log::debug!(
                "Compressing layer {layer_idx}, mip {mip_idx}: {}x{} to {:?} using {}",
                surface.width,
                surface.height,
                step.target_format,
                encoder.name(),
            );

            let output_format = step.target_format.denormalize(surface.color_space);
            let data = encoder.compress(surface, output_format, step.quality, settings_ref)?;

            let bpp_block = step.target_format.bytes_per_block().unwrap_or(16) as u32;
            let (bw, _bh) = step.target_format.block_size().unwrap_or((4, 4));
            let blocks_x = surface.width.div_ceil(bw as u32);

            new_mips.push(Surface {
                data,
                width: surface.width,
                height: surface.height,
                stride: blocks_x * bpp_block,
                format: step.target_format,
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
