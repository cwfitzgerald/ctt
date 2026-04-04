use std::sync::Arc;

use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::encoder::{Encoder, EncoderRegistry, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::format::{ColorSpace, CompressedFormat};
use crate::surface::{Image, Surface};
use crate::transform_node::{LayoutInfo, Transform};
use crate::vk_format::FormatExt;

/// A transform that compresses uncompressed surfaces using a block encoder.
///
/// This wraps the existing [`Encoder`] trait, bridging between the new `Surface` types and
/// the legacy `RawImage`-based encoder interface.
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
        let legacy_format = vk_to_legacy_compressed(self.target_format)?;

        if let Some(ref name) = self.encoder_name {
            self.registry
                .find_by_name(name, legacy_format)
                .ok_or_else(|| {
                    Error::UnsupportedFormat(format!(
                        "no encoder named '{}' supports {:?}",
                        name, self.target_format
                    ))
                })
        } else {
            self.registry.find(legacy_format).ok_or_else(|| {
                Error::UnsupportedFormat(format!(
                    "no encoder supports {:?}",
                    self.target_format
                ))
            })
        }
    }
}

impl Transform for CompressTransform {
    fn name(&self) -> &str {
        "compress"
    }

    fn constraint(&self) -> FormatConstraint {
        // Determine what the encoder needs
        let Ok(encoder) = self.resolve_encoder() else {
            // If we can't resolve the encoder, return an impossible constraint.
            // The resolver will produce an error.
            return FormatConstraint {
                formats: Some(Vec::new()),
                color_spaces: None,
                alpha_modes: None,
            };
        };

        let legacy_format =
            vk_to_legacy_compressed(self.target_format).expect("already validated in resolve");

        // The encoder tells us what pixel format it needs. We convert that to ktx2::Format.
        // We need to check both sRGB and linear since the constraint should accept both.
        let required_linear = encoder.required_input_format(legacy_format, ColorSpace::Linear);
        let vk_linear = ktx2::Format::from_pixel_format(required_linear);
        let (vk_linear, _) = vk_linear.normalize();

        let required_srgb = encoder.required_input_format(legacy_format, ColorSpace::Srgb);
        let vk_srgb = ktx2::Format::from_pixel_format(required_srgb);
        let (vk_srgb, _) = vk_srgb.normalize();

        let mut formats = vec![vk_linear];
        if vk_srgb != vk_linear {
            formats.push(vk_srgb);
        }

        FormatConstraint {
            formats: Some(formats),
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

    fn output_layout(&self, input: &LayoutInfo) -> LayoutInfo {
        input.clone()
    }

    fn execute(&self, image: Image) -> Result<Image> {
        let encoder = self.resolve_encoder()?;
        let legacy_format = vk_to_legacy_compressed(self.target_format)?;
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

                let raw = surface.to_raw_image()?;
                let data = encoder.compress(&raw, legacy_format, self.quality, settings_ref)?;

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

/// Convert a `ktx2::Format` to the legacy `CompressedFormat` for encoder interop.
fn vk_to_legacy_compressed(format: ktx2::Format) -> Result<CompressedFormat> {
    use ktx2::Format as F;
    let (base, _) = format.normalize();
    Ok(match base {
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGB_UNORM_BLOCK => CompressedFormat::Bc1,
        F::BC2_UNORM_BLOCK => CompressedFormat::Bc2,
        F::BC3_UNORM_BLOCK => CompressedFormat::Bc3,
        F::BC4_UNORM_BLOCK => CompressedFormat::Bc4,
        F::BC4_SNORM_BLOCK => CompressedFormat::Bc4s,
        F::BC5_UNORM_BLOCK => CompressedFormat::Bc5,
        F::BC5_SNORM_BLOCK => CompressedFormat::Bc5s,
        F::BC6H_UFLOAT_BLOCK => CompressedFormat::Bc6h,
        F::BC6H_SFLOAT_BLOCK => CompressedFormat::Bc6hSf,
        F::BC7_UNORM_BLOCK => CompressedFormat::Bc7,
        F::ETC2_R8G8B8_UNORM_BLOCK => CompressedFormat::Etc1,
        _ => {
            // Try ASTC
            if let Some((bw, bh)) = format.block_size() {
                if format.is_compressed() {
                    return Ok(CompressedFormat::Astc {
                        block_width: bw,
                        block_height: bh,
                    });
                }
            }
            return Err(Error::UnsupportedFormat(format!(
                "{format:?} is not a supported compressed format"
            )));
        }
    })
}
