//! High-level conversion entry point.

use std::sync::Arc;

use crate::alpha::AlphaMode;
use crate::encoders::{EncoderRegistry, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::format::TargetFormat;
use crate::processing::{
    self, Buffer, PipelineOutput, Swizzle, Variant, encode, load, mipmap, passthrough, store,
    swizzle,
};
use crate::surface::{ColorSpace, Image};
use crate::vk_format::FormatExt;

/// Output container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Container {
    Dds,
    Ktx2(Option<Ktx2Supercompression>),
    /// Return the processed [`Image`] directly, without encoding into a file format.
    Raw,
}

/// Supercompression to apply when writing KTX2 files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ktx2Supercompression {
    /// Zstandard compression. `level` is passed directly to the `zstd` crate.
    Zstd { level: i32 },
    /// ZLIB compression (deflate with zlib framing).
    Zlib { level: u8 },
}

impl Container {
    pub fn ktx2() -> Self {
        Container::Ktx2(None)
    }
    pub fn ktx2_zstd(level: i32) -> Self {
        Container::Ktx2(Some(Ktx2Supercompression::Zstd { level }))
    }
    pub fn ktx2_zlib(level: u8) -> Self {
        Container::Ktx2(Some(Ktx2Supercompression::Zlib { level }))
    }
}

/// Settings for the high-level [`convert`] function.
pub struct ConvertSettings {
    /// Target format. If `None`, the input format is preserved without compression.
    pub format: Option<TargetFormat>,
    pub container: Container,
    pub quality: Quality,
    pub output_color_space: Option<ColorSpace>,
    pub output_alpha: Option<AlphaMode>,
    pub swizzle: Option<Swizzle>,
    pub mipmap: bool,
    pub mipmap_count: Option<usize>,
    pub mipmap_filter: mipmap::MipmapFilter,
    pub encoder_settings: Option<Box<dyn EncoderSettings>>,
    pub registry: Option<Arc<EncoderRegistry>>,
}

impl Default for ConvertSettings {
    fn default() -> Self {
        Self {
            format: None,
            container: Container::Ktx2(None),
            quality: Quality::default(),
            output_color_space: None,
            output_alpha: None,
            swizzle: None,
            mipmap: false,
            mipmap_count: None,
            mipmap_filter: mipmap::MipmapFilter::default(),
            encoder_settings: None,
            registry: None,
        }
    }
}

/// Convert an image.
///
/// The input [`Image`] should already be fully assembled. Use
/// [`split_cubemap`](crate::split_cubemap) to prepare cubemap inputs beforehand.
pub fn convert(image: Image, mut settings: ConvertSettings) -> Result<PipelineOutput> {
    profiling::scope!("convert");
    let registry = settings
        .registry
        .take()
        .unwrap_or_else(|| Arc::new(EncoderRegistry::default_registry()));

    let input_base = &image.surfaces[0][0];
    let input_fmt = input_base.format;
    let input_cs = input_base.color_space;
    let input_alpha = input_base.alpha;

    let (target_fmt, encoder_step) = resolve_target(input_fmt, &mut settings, &registry)?;

    let target_cs = settings.output_color_space.unwrap_or(input_cs);
    let target_alpha = settings.output_alpha.unwrap_or(input_alpha);

    // Format that ends up in the output container: the encoder's target when
    // one is set, otherwise the resolved target format.
    let final_target_fmt = encoder_step
        .as_ref()
        .map(|s| s.target_format)
        .unwrap_or(target_fmt);

    log::debug!(
        "convert: {input_fmt:?} ({input_cs:?}, {input_alpha:?}) → \
         {final_target_fmt:?} ({target_cs:?}, {target_alpha:?}) \
         container={:?} swizzle={} mipmap={}",
        settings.container,
        settings.swizzle.is_some(),
        settings.mipmap,
    );

    // Passthrough whenever the bytes already represent the requested output:
    // identical format (modulo the sRGB tag, which rides on the surface),
    // matching color space and alpha, and no pixel-level rewrite. Covers
    // compressed-in == compressed-out and avoids a lossy load/store roundtrip
    // for uncompressed identity conversions.
    let (input_base_fmt, _) = input_fmt.normalize();
    let (target_base_fmt, _) = final_target_fmt.normalize();
    let formats_match = input_base_fmt == target_base_fmt;
    let no_pixel_work = settings.swizzle.is_none() && !settings.mipmap;

    if formats_match && input_cs == target_cs && input_alpha == target_alpha && no_pixel_work {
        log::debug!("convert: taking passthrough path (format and processing match)");
        return passthrough::run(image, final_target_fmt, settings.container);
    }

    // Compressed inputs that didn't qualify for passthrough have nowhere to
    // go — the float/integer pipelines can't decode compressed data. Defer
    // to passthrough::run's strict format check so the error is meaningful.
    if input_fmt.is_compressed() {
        log::debug!(
            "convert: compressed input did not qualify for passthrough; \
             falling back to passthrough format check"
        );
        return passthrough::run(image, final_target_fmt, settings.container);
    }

    let variant = processing::pick_variant(input_fmt, target_fmt).ok_or_else(|| {
        Error::UnsupportedConversion(format!(
            "cannot derive pipeline variant from {input_fmt:?} → {target_fmt:?}"
        ))
    })?;

    if !processing::families_compatible(input_fmt, target_fmt) {
        return Err(Error::UnsupportedConversion(format!(
            "integer/float family mismatch: {input_fmt:?} → {target_fmt:?}"
        )));
    }

    log::debug!("convert: routing through {variant:?} pipeline");

    match variant {
        Variant::F32 => convert_f32(image, settings, target_fmt, encoder_step),
        Variant::F64 => convert_f64(image, settings, target_fmt, encoder_step),
        Variant::U32 => convert_u32(image, settings, target_fmt, encoder_step),
        Variant::U64 => convert_u64(image, settings, target_fmt, encoder_step),
    }
}

/// Resolve the final output format and optional encoder step from settings.
///
/// Returns the format the store step should produce (= encoder input for
/// compressed targets, = `TargetFormat::Uncompressed` or input for non-compressed).
fn resolve_target(
    input_fmt: ktx2::Format,
    settings: &mut ConvertSettings,
    registry: &Arc<EncoderRegistry>,
) -> Result<(ktx2::Format, Option<encode::EncoderStep>)> {
    match settings.format.take() {
        Some(TargetFormat::Compressed {
            encoder_name,
            format,
        }) => {
            let step = encode::EncoderStep {
                target_format: format,
                encoder_name,
                quality: settings.quality,
                settings: settings.encoder_settings.take(),
                registry: Arc::clone(registry),
            };
            let required_input = step.required_input()?;
            Ok((required_input, Some(step)))
        }
        Some(TargetFormat::Uncompressed(fmt)) => Ok((fmt, None)),
        None => Ok((input_fmt, None)),
    }
}

fn convert_f32(
    image: Image,
    settings: ConvertSettings,
    target_fmt: ktx2::Format,
    encoder_step: Option<encode::EncoderStep>,
) -> Result<PipelineOutput> {
    let input_base = &image.surfaces[0][0];
    let target_color_space = settings
        .output_color_space
        .unwrap_or(input_base.color_space);
    let target_alpha = settings.output_alpha.unwrap_or(input_base.alpha);

    let mut out_layers = Vec::with_capacity(image.surfaces.len());
    for layer in image.surfaces {
        profiling::scope!("convert_f32_layer");
        let base = layer
            .into_iter()
            .next()
            .ok_or_else(|| Error::UnsupportedFormat("empty layer".into()))?;

        let mut buf: Buffer<f32> = load::load_f32(&base)?;

        if let Some(sw) = &settings.swizzle {
            swizzle::apply_f32(&mut buf, sw);
        }

        let bufs = if settings.mipmap {
            mipmap::generate(buf, settings.mipmap_filter, settings.mipmap_count)?
        } else {
            vec![buf]
        };

        let mut mips = Vec::with_capacity(bufs.len());
        for b in bufs {
            mips.push(store::store_f32(
                b,
                target_fmt,
                target_color_space,
                target_alpha,
            )?);
        }
        out_layers.push(mips);
    }

    let processed = Image {
        surfaces: out_layers,
        is_cubemap: image.is_cubemap,
    };

    let final_image = match encoder_step {
        Some(step) => encode::encode_all(processed, &step)?,
        None => processed,
    };

    passthrough::emit(final_image, settings.container)
}

fn convert_f64(
    image: Image,
    settings: ConvertSettings,
    target_fmt: ktx2::Format,
    encoder_step: Option<encode::EncoderStep>,
) -> Result<PipelineOutput> {
    if settings.mipmap {
        return Err(Error::UnsupportedFormat(
            "f64 pipeline does not yet support mipmap generation".into(),
        ));
    }

    let input_base = &image.surfaces[0][0];
    let target_color_space = settings
        .output_color_space
        .unwrap_or(input_base.color_space);
    let target_alpha = settings.output_alpha.unwrap_or(input_base.alpha);

    let mut out_layers = Vec::with_capacity(image.surfaces.len());
    for layer in image.surfaces {
        profiling::scope!("convert_f64_layer");
        let mut mips = Vec::with_capacity(layer.len());
        for base in layer {
            let mut buf = load::load_f64(&base)?;
            if let Some(sw) = &settings.swizzle {
                swizzle::apply_f64(&mut buf, sw);
            }
            mips.push(store::store_f64(
                buf,
                target_fmt,
                target_color_space,
                target_alpha,
            )?);
        }
        out_layers.push(mips);
    }

    let processed = Image {
        surfaces: out_layers,
        is_cubemap: image.is_cubemap,
    };

    let final_image = match encoder_step {
        Some(step) => encode::encode_all(processed, &step)?,
        None => processed,
    };

    passthrough::emit(final_image, settings.container)
}

fn convert_u32(
    image: Image,
    settings: ConvertSettings,
    target_fmt: ktx2::Format,
    encoder_step: Option<encode::EncoderStep>,
) -> Result<PipelineOutput> {
    let input_alpha = image.surfaces[0][0].alpha;
    check_uint_unsupported(&settings, input_alpha)?;
    let target_alpha = settings.output_alpha.unwrap_or(input_alpha);

    let mut out_layers = Vec::with_capacity(image.surfaces.len());
    for layer in image.surfaces {
        profiling::scope!("convert_u32_layer");
        let mut mips = Vec::with_capacity(layer.len());
        for base in layer {
            let mut buf = load::load_u32(&base)?;
            if let Some(sw) = &settings.swizzle {
                swizzle::apply_u32(&mut buf, sw);
            }
            mips.push(store::store_u32(buf, target_fmt, target_alpha)?);
        }
        out_layers.push(mips);
    }

    let processed = Image {
        surfaces: out_layers,
        is_cubemap: image.is_cubemap,
    };

    if encoder_step.is_some() {
        return Err(Error::UnsupportedConversion(
            "integer (uint/sint) formats cannot be block-compressed".into(),
        ));
    }

    passthrough::emit(processed, settings.container)
}

fn convert_u64(
    image: Image,
    settings: ConvertSettings,
    target_fmt: ktx2::Format,
    encoder_step: Option<encode::EncoderStep>,
) -> Result<PipelineOutput> {
    let input_alpha = image.surfaces[0][0].alpha;
    check_uint_unsupported(&settings, input_alpha)?;
    let target_alpha = settings.output_alpha.unwrap_or(input_alpha);

    let mut out_layers = Vec::with_capacity(image.surfaces.len());
    for layer in image.surfaces {
        profiling::scope!("convert_u64_layer");
        let mut mips = Vec::with_capacity(layer.len());
        for base in layer {
            let mut buf = load::load_u64(&base)?;
            if let Some(sw) = &settings.swizzle {
                swizzle::apply_u64(&mut buf, sw);
            }
            mips.push(store::store_u64(buf, target_fmt, target_alpha)?);
        }
        out_layers.push(mips);
    }

    let processed = Image {
        surfaces: out_layers,
        is_cubemap: image.is_cubemap,
    };

    if encoder_step.is_some() {
        return Err(Error::UnsupportedConversion(
            "integer (uint/sint) formats cannot be block-compressed".into(),
        ));
    }

    passthrough::emit(processed, settings.container)
}

fn check_uint_unsupported(settings: &ConvertSettings, input_alpha: AlphaMode) -> Result<()> {
    if settings.mipmap {
        return Err(Error::UnsupportedFormat(
            "integer pipeline does not support mipmap generation".into(),
        ));
    }
    if settings.output_color_space.is_some() {
        return Err(Error::UnsupportedFormat(
            "integer pipeline does not support output_color_space change".into(),
        ));
    }
    if let Some(out_alpha) = settings.output_alpha
        && out_alpha != input_alpha
    {
        return Err(Error::UnsupportedFormat(
            "integer pipeline does not support output_alpha change".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::Surface;

    fn make_image(
        data: Vec<u8>,
        width: u32,
        height: u32,
        format: ktx2::Format,
        cs: ColorSpace,
        alpha: AlphaMode,
    ) -> Image {
        let bpp = format.bytes_per_pixel().unwrap() as u32;
        Image {
            surfaces: vec![vec![Surface {
                data,
                width,
                height,
                stride: width * bpp,
                format,
                color_space: cs,
                alpha,
            }]],
            is_cubemap: false,
        }
    }

    #[test]
    fn raw_roundtrip_rgba8_linear() {
        let image = make_image(
            vec![10, 20, 30, 40, 50, 60, 70, 80],
            2,
            1,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        );
        let out = convert(
            image,
            ConvertSettings {
                container: Container::Raw,
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Raw(img) => {
                assert_eq!(
                    img.surfaces[0][0].data,
                    vec![10, 20, 30, 40, 50, 60, 70, 80]
                );
            }
            _ => panic!("expected Raw output"),
        }
    }

    #[test]
    fn convert_rgba8_to_r8_channel_drop() {
        let image = make_image(
            vec![100, 150, 200, 255],
            1,
            1,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        );
        let out = convert(
            image,
            ConvertSettings {
                format: Some(TargetFormat::Uncompressed(ktx2::Format::R8_UNORM)),
                container: Container::Raw,
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Raw(img) => {
                assert_eq!(img.surfaces[0][0].format, ktx2::Format::R8_UNORM);
                assert_eq!(img.surfaces[0][0].data, vec![100]);
            }
            _ => panic!("expected Raw output"),
        }
    }

    #[test]
    fn convert_with_mipmap() {
        let data = vec![128u8; 8 * 8 * 4];
        let image = make_image(
            data,
            8,
            8,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        );
        let out = convert(
            image,
            ConvertSettings {
                format: Some(TargetFormat::Uncompressed(ktx2::Format::R8G8B8A8_UNORM)),
                container: Container::Raw,
                mipmap: true,
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Raw(img) => {
                assert_eq!(img.surfaces[0].len(), 4); // 8,4,2,1
            }
            _ => panic!("expected Raw output"),
        }
    }

    #[test]
    fn convert_integer_family_mismatch_errors() {
        let image = make_image(
            vec![100, 150, 200, 255],
            1,
            1,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        );
        let err = convert(
            image,
            ConvertSettings {
                format: Some(TargetFormat::Uncompressed(ktx2::Format::R8G8B8A8_UINT)),
                container: Container::Raw,
                ..Default::default()
            },
        )
        .unwrap_err();
        match err {
            Error::UnsupportedConversion(_) => {}
            _ => panic!("expected UnsupportedConversion, got {err:?}"),
        }
    }

    #[test]
    fn convert_rgba8_to_bc7_ktx2() {
        // 4x4 opaque image, encode to BC7 via default encoder, wrap in KTX2.
        let image = make_image(
            vec![128u8; 4 * 4 * 4],
            4,
            4,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        );
        let out = convert(
            image,
            ConvertSettings {
                format: Some(TargetFormat::Compressed {
                    encoder_name: None,
                    format: ktx2::Format::BC7_UNORM_BLOCK,
                }),
                container: Container::ktx2(),
                quality: Quality::UltraFast,
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Encoded(bytes) => {
                // Verify KTX2 magic.
                assert!(bytes.len() > 12);
                assert_eq!(&bytes[0..12], b"\xabKTX 20\xbb\r\n\x1a\n");
            }
            _ => panic!("expected Encoded output"),
        }
    }

    #[test]
    fn convert_passthrough_compressed() {
        // BC7 input → BC7 output should use the passthrough fast path.
        let bc7_bytes = vec![0xFFu8; 16]; // 1 BC7 block
        let image = Image {
            surfaces: vec![vec![Surface {
                data: bc7_bytes.clone(),
                width: 4,
                height: 4,
                stride: 16,
                format: ktx2::Format::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Opaque,
            }]],
            is_cubemap: false,
        };
        let out = convert(
            image,
            ConvertSettings {
                container: Container::Raw,
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Raw(img) => {
                assert_eq!(img.surfaces[0][0].data, bc7_bytes);
            }
            _ => panic!("expected Raw output"),
        }
    }

    #[test]
    fn convert_uint_pipeline_with_swizzle() {
        // 4 u32 values = 16 bytes.
        let mut data = Vec::new();
        for v in &[10u32, 20, 30, 40] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let image = make_image(
            data,
            1,
            1,
            ktx2::Format::R32G32B32A32_UINT,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        );
        let out = convert(
            image,
            ConvertSettings {
                format: Some(TargetFormat::Uncompressed(ktx2::Format::R32G32B32A32_UINT)),
                container: Container::Raw,
                swizzle: Some(Swizzle([
                    processing::SwizzleChannel::A,
                    processing::SwizzleChannel::B,
                    processing::SwizzleChannel::G,
                    processing::SwizzleChannel::R,
                ])),
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Raw(img) => {
                let bytes = &img.surfaces[0][0].data;
                assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 40);
                assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 30);
                assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 20);
                assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 10);
            }
            _ => panic!("expected Raw output"),
        }
    }
}
