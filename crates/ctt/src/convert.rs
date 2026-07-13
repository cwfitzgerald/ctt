//! High-level conversion entry point.

use crate::alpha::AlphaMode;
use crate::encoders::Quality;
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
    /// DirectDraw Surface (`.dds`) file.
    Dds,
    /// KTX2 (`.ktx2`) file, optionally supercompressed. `None` means no
    /// supercompression.
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
    /// KTX2 output with no supercompression.
    pub fn ktx2() -> Self {
        Container::Ktx2(None)
    }
    /// KTX2 output with Zstandard supercompression at the given `zstd` level.
    pub fn ktx2_zstd(level: i32) -> Self {
        Container::Ktx2(Some(Ktx2Supercompression::Zstd { level }))
    }
    /// KTX2 output with ZLIB supercompression at the given deflate level.
    pub fn ktx2_zlib(level: u8) -> Self {
        Container::Ktx2(Some(Ktx2Supercompression::Zlib { level }))
    }
}

/// Settings for the high-level [`convert`] function.
#[derive(Default)]
pub struct ConvertSettings {
    /// Target format. If `None`, the input format is preserved without compression.
    pub format: Option<TargetFormat>,
    /// Output container format. Defaults to KTX2 with no supercompression
    /// ([`Container::Ktx2(None)`](Container::Ktx2)).
    pub container: Container,
    /// Encoder quality preset for compressed targets. Ignored for
    /// uncompressed output.
    pub quality: Quality,
    /// Override the output color space. `None` keeps the input's color space.
    pub output_color_space: Option<ColorSpace>,
    /// Override the output alpha mode. `None` keeps the input's alpha mode.
    pub output_alpha: Option<AlphaMode>,
    /// Suppress the warning emitted when a meaningful (`Straight`) alpha
    /// channel is dropped because the target format has none. Does not change
    /// pixel output.
    pub allow_discarding_alpha: bool,
    /// Optional channel swizzle applied to each pixel before encoding.
    pub swizzle: Option<Swizzle>,
    /// Complete the mip chain when `true`, preserving existing levels and
    /// generating only the missing tail. When `false`, any existing mip levels
    /// are preserved (and converted).
    pub mipmap: bool,
    /// Number of mip levels to retain or generate when `mipmap` is `true`.
    /// `None` builds the full chain; larger-than-full values are clamped.
    pub mipmap_count: Option<usize>,
    /// Downsampling filter used for mipmap generation.
    pub mipmap_filter: mipmap::MipmapFilter,
}

impl Default for Container {
    fn default() -> Self {
        Container::Ktx2(None)
    }
}

/// Convert an image.
///
/// The input [`Image`] should already be fully assembled. Use
/// [`split_cubemap`](crate::split_cubemap) to prepare cubemap inputs beforehand.
pub fn convert(image: Image, mut settings: ConvertSettings) -> Result<PipelineOutput> {
    profiling::scope!("convert");
    image.validate()?;

    let input_base = &image.surfaces[0][0];
    let input_fmt = input_base.format;
    let input_cs = input_base.color_space;
    let input_alpha = input_base.alpha;

    let (target_fmt, encoder_step) = resolve_target(input_fmt, &mut settings)?;

    let target_cs = settings.output_color_space.unwrap_or(input_cs);
    let target_alpha = settings.output_alpha.unwrap_or(input_alpha);

    // Format that ends up in the output container: the encoder's target when
    // one is set, otherwise the resolved target format.
    let final_target_fmt = encoder_step
        .as_ref()
        .map(|s| s.target_format)
        .unwrap_or(target_fmt);

    let final_has_alpha = final_target_fmt.has_alpha_channel();
    if warn_discarding_alpha(
        input_fmt.has_alpha_channel(),
        final_has_alpha,
        target_alpha,
        settings.allow_discarding_alpha,
    ) {
        log::warn!(
            "{final_target_fmt:?} has no alpha channel; discarding the source's straight \
             alpha and keeping color. Set output alpha to premultiplied to bake it in, opaque \
             to mark the texture opaque, or allow_discarding_alpha to silence."
        );
    }

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
    // go — the float/integer pipelines can't decode compressed data. A
    // swizzle or mipmap request needs pixel-level access, so it can't be
    // honored; fail loudly rather than silently dropping it in passthrough.
    if input_fmt.is_compressed() {
        if settings.swizzle.is_some() || settings.mipmap {
            return Err(Error::UnsupportedConversion(
                "cannot swizzle or mipmap a compressed input; decode it to an \
                 uncompressed format first"
                    .into(),
            ));
        }
        if input_cs != target_cs || input_alpha != target_alpha {
            return Err(Error::UnsupportedConversion(
                "cannot change color space or alpha mode for a compressed input; decode it to an \
                 uncompressed format first"
                    .into(),
            ));
        }
        log::debug!(
            "convert: compressed input did not qualify for passthrough; \
             falling back to passthrough format check"
        );
        return passthrough::run(image, final_target_fmt, settings.container);
    }

    // 3D textures only flow through the passthrough fast path. The f32/u32
    // pipelines treat each Surface as a single 2D plane; they can't yet
    // process the stacked Z slices that 3D textures carry.
    if matches!(image.kind, crate::TextureKind::Texture3D) {
        return Err(Error::UnsupportedConversion(
            "3D textures are only supported in passthrough mode (no format change, swizzle, or mipmap generation)".into(),
        ));
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
        Variant::F32 => convert_f32(image, settings, target_fmt, encoder_step, final_has_alpha),
        Variant::F64 => convert_f64(image, settings, target_fmt, encoder_step, final_has_alpha),
        Variant::U32 => convert_u32(image, settings, target_fmt, encoder_step),
        Variant::U64 => convert_u64(image, settings, target_fmt, encoder_step),
    }
}

/// Whether to warn that a meaningful straight alpha is being dropped.
///
/// The warning fires only when the source has alpha, the final output format
/// does not, the effective output alpha is `Straight`, and the caller has not
/// acknowledged the drop. `Opaque` and `Premultiplied` outputs are deliberate
/// and therefore silent.
fn warn_discarding_alpha(
    source_has_alpha: bool,
    final_has_alpha: bool,
    effective_out: AlphaMode,
    allow: bool,
) -> bool {
    source_has_alpha && !final_has_alpha && effective_out == AlphaMode::Straight && !allow
}

/// Pick the `(load, store)` alpha modes that apply the single direct
/// conversion an alpha-less target needs, given the effective `output_alpha`
/// and the source's alpha mode.
///
/// `load_f32` premultiplies iff the load alpha is `Straight`; `store_f32`
/// unpremultiplies iff the store alpha is `Straight`. Choosing the pair below
/// therefore applies at most one premultiply or one unpremultiply, avoiding
/// the destructive round-trip that zeros RGB at α=0.
fn alpha_less_load_store(target_alpha: AlphaMode, src_alpha: AlphaMode) -> (AlphaMode, AlphaMode) {
    use AlphaMode::*;
    match (target_alpha, src_alpha) {
        (Premultiplied, Straight) => (Straight, Opaque), // bake: premultiply once
        (Premultiplied, _) => (Opaque, Opaque),          // already premul / opaque
        (_, Premultiplied) => (Opaque, Straight),        // keep color: unpremultiply once
        (_, _) => (Opaque, Opaque),                      // keep color: pass through
    }
}

/// Resolve the final output format and optional encoder step from settings.
///
/// Returns the format the store step should produce (= encoder input for
/// compressed targets, = `TargetFormat::Uncompressed` or input for non-compressed).
fn resolve_target(
    input_fmt: ktx2::Format,
    settings: &mut ConvertSettings,
) -> Result<(ktx2::Format, Option<encode::EncoderStep>)> {
    match settings.format.take() {
        Some(TargetFormat::Compressed { format, encoder }) => {
            let step = encode::EncoderStep {
                target_format: format,
                quality: settings.quality,
                encoder,
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
    final_has_alpha: bool,
) -> Result<PipelineOutput> {
    let input_base = &image.surfaces[0][0];
    let target_color_space = settings
        .output_color_space
        .unwrap_or(input_base.color_space);
    let target_alpha = settings.output_alpha.unwrap_or(input_base.alpha);

    let mut out_layers = Vec::with_capacity(image.surfaces.len());
    for layer in image.surfaces {
        profiling::scope!("convert_f32_layer");

        let mips = if settings.mipmap {
            let target_count = settings
                .mipmap_count
                .unwrap_or_else(|| mipmap::full_mip_count(layer[0].width, layer[0].height));
            let mut bufs = Vec::with_capacity(layer.len().min(target_count));
            let mut store_alphas = Vec::with_capacity(bufs.capacity());
            for mut surface in layer.into_iter().take(target_count) {
                // An alpha-less final target must not force the premultiplied
                // round-trip: it zeros RGB wherever alpha=0. Apply only the
                // single direct conversion the effective output mode requires.
                let (load_alpha, store_alpha) = if final_has_alpha {
                    (surface.alpha, target_alpha)
                } else {
                    alpha_less_load_store(target_alpha, surface.alpha)
                };
                surface.alpha = load_alpha;
                let mut buf: Buffer<f32> = load::load_f32(&surface)?;
                if let Some(sw) = &settings.swizzle {
                    swizzle::apply_f32(&mut buf, sw);
                }
                bufs.push(buf);
                store_alphas.push(store_alpha);
            }
            let generated_store_alpha = *store_alphas
                .last()
                .ok_or_else(|| Error::UnsupportedFormat("mipmap count must be >= 1".into()))?;
            let bufs = mipmap::complete(bufs, settings.mipmap_filter, settings.mipmap_count)?;
            store_alphas.resize(bufs.len(), generated_store_alpha);
            let mut mips = Vec::with_capacity(bufs.len());
            for (b, store_alpha) in bufs.into_iter().zip(store_alphas) {
                let mut surface = store::store_f32(b, target_fmt, target_color_space, store_alpha)?;
                surface.alpha = target_alpha;
                mips.push(surface);
            }
            mips
        } else {
            // Convert every existing mip level (matching the f64/integer
            // paths) so an input mip chain isn't silently dropped.
            let mut mips = Vec::with_capacity(layer.len());
            for mut base in layer {
                let (load_alpha, store_alpha) = if final_has_alpha {
                    (base.alpha, target_alpha)
                } else {
                    alpha_less_load_store(target_alpha, base.alpha)
                };
                base.alpha = load_alpha;
                let mut buf: Buffer<f32> = load::load_f32(&base)?;
                if let Some(sw) = &settings.swizzle {
                    swizzle::apply_f32(&mut buf, sw);
                }
                let mut surface =
                    store::store_f32(buf, target_fmt, target_color_space, store_alpha)?;
                surface.alpha = target_alpha;
                mips.push(surface);
            }
            mips
        };
        out_layers.push(mips);
    }

    let processed = Image {
        surfaces: out_layers,
        kind: image.kind,
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
    final_has_alpha: bool,
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
        for mut base in layer {
            // See `convert_f32`: an alpha-less target skips the destructive
            // premultiplied round-trip.
            let (load_alpha, store_alpha) = if final_has_alpha {
                (base.alpha, target_alpha)
            } else {
                alpha_less_load_store(target_alpha, base.alpha)
            };
            base.alpha = load_alpha;
            let mut buf = load::load_f64(&base)?;
            if let Some(sw) = &settings.swizzle {
                swizzle::apply_f64(&mut buf, sw);
            }
            let mut surface = store::store_f64(buf, target_fmt, target_color_space, store_alpha)?;
            surface.alpha = target_alpha;
            mips.push(surface);
        }
        out_layers.push(mips);
    }

    let processed = Image {
        surfaces: out_layers,
        kind: image.kind,
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
        kind: image.kind,
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
        kind: image.kind,
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
                depth: 1,
                stride: width * bpp,
                slice_stride: 0,
                format,
                color_space: cs,
                alpha,
            }]],
            kind: crate::TextureKind::Texture2D,
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

    /// Build an RGBA8 image with three explicit mip levels (8×8, 4×4, 2×2).
    fn three_mip_rgba8() -> Image {
        let mip = |w: u32, h: u32, fill: u8| Surface {
            data: vec![fill; (w * h * 4) as usize],
            width: w,
            height: h,
            depth: 1,
            stride: w * 4,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        };
        Image {
            surfaces: vec![vec![mip(8, 8, 10), mip(4, 4, 20), mip(2, 2, 30)]],
            kind: crate::TextureKind::Texture2D,
        }
    }

    #[test]
    fn convert_f32_preserves_existing_mips_when_no_mipmap() {
        // mipmap = false must convert (not drop) every input mip level. Target
        // R8_UNORM (channel drop) so this routes through the f32 pipeline
        // rather than the format-identity passthrough fast path.
        let out = convert(
            three_mip_rgba8(),
            ConvertSettings {
                format: Some(TargetFormat::Uncompressed(ktx2::Format::R8_UNORM)),
                container: Container::Raw,
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Raw(img) => {
                assert_eq!(img.surfaces[0].len(), 3, "existing 3-mip chain preserved");
                assert_eq!(
                    (img.surfaces[0][2].width, img.surfaces[0][2].height),
                    (2, 2),
                );
            }
            _ => panic!("expected Raw output"),
        }
    }

    #[test]
    fn convert_f32_completes_existing_chain_with_mipmap() {
        // The supplied 8×8, 4×4, and 2×2 levels are retained; only the
        // missing 1×1 tail is generated from the final existing level.
        let out = convert(
            three_mip_rgba8(),
            ConvertSettings {
                format: Some(TargetFormat::Uncompressed(ktx2::Format::R8_UNORM)),
                container: Container::Raw,
                mipmap: true,
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Raw(img) => {
                assert_eq!(img.surfaces[0].len(), 4, "full chain completed");
                assert_eq!(img.surfaces[0][0].data[0], 10);
                assert_eq!(img.surfaces[0][1].data[0], 20);
                assert_eq!(img.surfaces[0][2].data[0], 30);
                assert_eq!(img.surfaces[0][3].data[0], 30);
                assert_eq!(
                    (img.surfaces[0][3].width, img.surfaces[0][3].height),
                    (1, 1),
                );
            }
            _ => panic!("expected Raw output"),
        }
    }

    #[test]
    fn convert_f32_mipmap_count_truncates_existing_chain() {
        let out = convert(
            three_mip_rgba8(),
            ConvertSettings {
                format: Some(TargetFormat::Uncompressed(ktx2::Format::R8_UNORM)),
                container: Container::Raw,
                mipmap: true,
                mipmap_count: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Raw(img) => {
                assert_eq!(img.surfaces[0].len(), 2);
                assert_eq!(img.surfaces[0][0].data[0], 10);
                assert_eq!(img.surfaces[0][1].data[0], 20);
            }
            _ => panic!("expected Raw output"),
        }
    }

    fn bc7_1block_image() -> Image {
        Image {
            surfaces: vec![vec![Surface {
                data: vec![0xFFu8; 16],
                width: 4,
                height: 4,
                depth: 1,
                stride: 16,
                slice_stride: 0,
                format: ktx2::Format::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Opaque,
            }]],
            kind: crate::TextureKind::Texture2D,
        }
    }

    #[test]
    fn convert_compressed_input_swizzle_errors() {
        let err = convert(
            bc7_1block_image(),
            ConvertSettings {
                container: Container::Raw,
                swizzle: Some(Swizzle([
                    processing::SwizzleChannel::B,
                    processing::SwizzleChannel::G,
                    processing::SwizzleChannel::R,
                    processing::SwizzleChannel::A,
                ])),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedConversion(_)),
            "expected UnsupportedConversion, got {err:?}",
        );
    }

    #[test]
    fn convert_compressed_input_mipmap_errors() {
        let err = convert(
            bc7_1block_image(),
            ConvertSettings {
                container: Container::Raw,
                mipmap: true,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedConversion(_)),
            "expected UnsupportedConversion, got {err:?}",
        );
    }

    #[test]
    fn convert_compressed_input_color_space_change_errors() {
        let err = convert(
            bc7_1block_image(),
            ConvertSettings {
                container: Container::Raw,
                output_color_space: Some(ColorSpace::Srgb),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedConversion(_)),
            "expected UnsupportedConversion, got {err:?}",
        );
    }

    #[test]
    fn convert_compressed_input_alpha_change_errors() {
        let err = convert(
            bc7_1block_image(),
            ConvertSettings {
                container: Container::Raw,
                output_alpha: Some(AlphaMode::Premultiplied),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedConversion(_)),
            "expected UnsupportedConversion, got {err:?}",
        );
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
                    format: ktx2::Format::BC7_UNORM_BLOCK,
                    encoder: crate::encoders::Encoder::Auto,
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
                depth: 1,
                stride: 16,
                slice_stride: 0,
                format: ktx2::Format::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Opaque,
            }]],
            kind: crate::TextureKind::Texture2D,
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

    /// Build a 4×2 RGBA8 image whose row stride is `width * 4 + 8` bytes —
    /// each row carries 8 bytes of trailing padding that the pipeline must
    /// skip. The padding is filled with 0xCC so any padding-leak shows up
    /// loudly in the output.
    fn padded_rgba8_4x2() -> Image {
        let pad = 0xCCu8;
        // Row 0: 4 distinct pixels, then 8 bytes padding (stride = 24).
        let mut data = Vec::new();
        for x in 0..4u8 {
            data.extend_from_slice(&[10 + x, 20 + x, 30 + x, 255]);
        }
        data.extend_from_slice(&[pad; 8]);
        for x in 0..4u8 {
            data.extend_from_slice(&[100 + x, 110 + x, 120 + x, 255]);
        }
        data.extend_from_slice(&[pad; 8]);

        Image {
            surfaces: vec![vec![Surface {
                data,
                width: 4,
                height: 2,
                depth: 1,
                stride: 4 * 4 + 8,
                slice_stride: 0,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Straight,
            }]],
            kind: crate::TextureKind::Texture2D,
        }
    }

    /// Encoding path: padded-stride RGBA8 → swizzle to BGRA → Raw.
    /// The store step always writes a tight stride, and no padding byte
    /// (0xCC) should bleed into the output.
    #[test]
    fn convert_padded_stride_swizzle_to_raw_is_tight() {
        let image = padded_rgba8_4x2();
        let out = convert(
            image,
            ConvertSettings {
                container: Container::Raw,
                swizzle: Some(Swizzle([
                    processing::SwizzleChannel::B,
                    processing::SwizzleChannel::G,
                    processing::SwizzleChannel::R,
                    processing::SwizzleChannel::A,
                ])),
                ..Default::default()
            },
        )
        .unwrap();
        let img = match out {
            PipelineOutput::Raw(img) => img,
            _ => panic!("expected Raw output"),
        };
        let s = &img.surfaces[0][0];
        // Output must be tight.
        assert_eq!(s.stride, 4 * 4);
        assert_eq!(s.data.len(), 4 * 4 * 2);
        // Padding byte must not appear.
        assert!(
            !s.data.contains(&0xCC),
            "padding leaked into output: {:?}",
            s.data,
        );
        // First pixel of row 0: original (10,20,30,255) → BGRA = (30,20,10,255).
        assert_eq!(&s.data[0..4], &[30, 20, 10, 255]);
        // First pixel of row 1: original (100,110,120,255) → (120,110,100,255).
        let row1 = (4 * 4) as usize;
        assert_eq!(&s.data[row1..row1 + 4], &[120, 110, 100, 255]);
    }

    /// Encoding path: padded-stride RGBA8 → BC7 → KTX2.
    /// Just verifies the pipeline accepts padded input and produces a valid
    /// KTX2 file. Re-decoding BC7 to check pixel exactness is too brittle
    /// for an unrelated test, so we assert structural validity only.
    #[test]
    fn convert_padded_stride_to_bc7_succeeds() {
        let image = padded_rgba8_4x2();
        // 4×2 isn't 4×4-aligned, but tile_to_blocks edge-replicates so this
        // still produces 1×1 blocks (rounded up to 4×4).
        let out = convert(
            image,
            ConvertSettings {
                format: Some(TargetFormat::Compressed {
                    format: ktx2::Format::BC7_UNORM_BLOCK,
                    encoder: crate::encoders::Encoder::Auto,
                }),
                container: Container::ktx2(),
                quality: Quality::UltraFast,
                ..Default::default()
            },
        )
        .unwrap();
        match out {
            PipelineOutput::Encoded(bytes) => {
                assert_eq!(&bytes[0..12], b"\xabKTX 20\xbb\r\n\x1a\n");
            }
            _ => panic!("expected Encoded output"),
        }
    }

    /// Passthrough path: padded-stride RGBA8 → KTX2 (format identity, no
    /// pixel work). The output must be a valid KTX2 file containing the
    /// tightly-packed pixels — the per-row padding from the input must not
    /// appear in the level data.
    #[test]
    fn convert_padded_stride_uncompressed_passthrough_is_tight() {
        let image = padded_rgba8_4x2();
        let out = convert(
            image,
            ConvertSettings {
                container: Container::ktx2(),
                ..Default::default()
            },
        )
        .unwrap();
        let bytes = match out {
            PipelineOutput::Encoded(bytes) => bytes,
            _ => panic!("expected Encoded output"),
        };
        // Round-trip through the KTX2 decoder; each pixel must match the
        // original padded source row-by-row.
        let decoded = crate::input::ktx2::decode_ktx2_image(&bytes).unwrap();
        let s = &decoded.surfaces[0][0];
        assert_eq!(s.width, 4);
        assert_eq!(s.height, 2);
        assert_eq!(s.stride, 4 * 4); // tight on the way out
        assert!(
            !s.data.contains(&0xCC),
            "padding leaked into KTX2 level data"
        );
        // Spot-check a pixel from each row.
        assert_eq!(&s.data[0..4], &[10, 20, 30, 255]);
        assert_eq!(&s.data[16..20], &[100, 110, 120, 255]);
    }

    /// Passthrough path: padded-stride BC7 → KTX2.
    /// 8×4 pixels = 2×1 blocks per row of blocks, but with a row-of-blocks
    /// stride deliberately wider than 32 bytes (one 16-byte pad block per row).
    #[test]
    fn convert_padded_stride_compressed_passthrough_is_tight() {
        let pad = 0xCCu8;
        // 2 real BC7 blocks (32 bytes) + 1 block of padding (16 bytes) per
        // row of blocks. 8×4 has 1 row of blocks vertically (4 / 4 = 1).
        let block0 = [0x11u8; 16];
        let block1 = [0x22u8; 16];
        let mut data = Vec::new();
        data.extend_from_slice(&block0);
        data.extend_from_slice(&block1);
        data.extend_from_slice(&[pad; 16]);

        let image = Image {
            surfaces: vec![vec![Surface {
                data,
                width: 8,
                height: 4,
                depth: 1,
                stride: 2 * 16 + 16, // 2 blocks of payload + 1 block of padding
                slice_stride: 0,
                format: ktx2::Format::BC7_UNORM_BLOCK,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Opaque,
            }]],
            kind: crate::TextureKind::Texture2D,
        };

        let out = convert(
            image,
            ConvertSettings {
                container: Container::ktx2(),
                ..Default::default()
            },
        )
        .unwrap();
        let bytes = match out {
            PipelineOutput::Encoded(bytes) => bytes,
            _ => panic!("expected Encoded output"),
        };
        // Inspect the encoded KTX2 directly: the level index must say 32
        // bytes (2 BC7 blocks, tight), not 48 (which would mean the padding
        // block was written into the file). The decoder discards trailing
        // bytes, so we have to look at the writer's output to catch this.
        let reader = ktx2::Reader::new(&bytes[..]).expect("valid KTX2");
        let levels: Vec<_> = reader.levels().collect();
        assert_eq!(levels.len(), 1);
        assert_eq!(
            levels[0].data.len(),
            2 * 16,
            "level payload must be tight (2 BC7 blocks), not include the padding block",
        );
        assert!(
            !levels[0].data.contains(&pad),
            "BC7 padding leaked into KTX2 level data",
        );

        let decoded = crate::input::ktx2::decode_ktx2_image(&bytes).unwrap();
        let s = &decoded.surfaces[0][0];
        assert_eq!(s.format, ktx2::Format::BC7_UNORM_BLOCK);
        let mut expected = Vec::new();
        expected.extend_from_slice(&block0);
        expected.extend_from_slice(&block1);
        assert_eq!(s.data, expected);
    }

    /// Passthrough path: 3D RGBA8 with both row and slice padding → KTX2.
    /// Each Z slice is `4×2 RGBA8 + 8 bytes row pad + 8 bytes slice pad`,
    /// none of which may surface in the encoded file.
    #[test]
    fn convert_padded_slice_stride_3d_passthrough_is_tight() {
        let pad = 0xCCu8;
        let row_pad = 8;
        let slice_pad = 8;
        let stride = 4 * 4 + row_pad;
        let slice_payload = stride * 2;
        let slice_stride = slice_payload + slice_pad;
        let depth = 3u32;

        let mut data = Vec::with_capacity((slice_stride * depth) as usize);
        for z in 0..depth as u8 {
            for y in 0..2u8 {
                for x in 0..4u8 {
                    data.extend_from_slice(&[z * 50 + x, y * 30, 0, 255]);
                }
                data.extend_from_slice(&[pad; 8]);
            }
            data.extend_from_slice(&[pad; 8]);
        }

        let image = Image {
            surfaces: vec![vec![Surface {
                data,
                width: 4,
                height: 2,
                depth,
                stride,
                slice_stride,
                format: ktx2::Format::R8G8B8A8_UNORM,
                color_space: ColorSpace::Linear,
                alpha: AlphaMode::Opaque,
            }]],
            kind: crate::TextureKind::Texture3D,
        };

        let out = convert(
            image,
            ConvertSettings {
                container: Container::ktx2(),
                ..Default::default()
            },
        )
        .unwrap();
        let bytes = match out {
            PipelineOutput::Encoded(bytes) => bytes,
            _ => panic!("expected Encoded output"),
        };
        let decoded = crate::input::ktx2::decode_ktx2_image(&bytes).unwrap();
        let s = &decoded.surfaces[0][0];
        assert_eq!(s.depth, depth);
        assert_eq!(s.stride, 4 * 4);
        assert_eq!(s.slice_stride, 4 * 4 * 2);
        assert!(
            !s.data.contains(&pad),
            "padding leaked into 3D KTX2 level data",
        );
        // Spot-check pixel (0,0,z) for each slice.
        for z in 0..depth as usize {
            let base = z * (4 * 4 * 2);
            assert_eq!(
                &s.data[base..base + 4],
                &[(z as u8) * 50, 0, 0, 255],
                "slice {z} pixel (0,0)",
            );
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

    #[test]
    fn warn_discarding_alpha_truth_table() {
        use AlphaMode::*;
        // Alpha-bearing final target: never warn regardless of alpha mode.
        for mode in [Straight, Premultiplied, Opaque] {
            assert!(
                !warn_discarding_alpha(true, true, mode, false),
                "has-alpha {mode:?}"
            );
        }
        // An alpha-less source has nothing meaningful to discard.
        assert!(!warn_discarding_alpha(false, false, Straight, false));
        // Alpha-less final target with effective Straight output: warn unless
        // the caller acknowledged the drop.
        assert!(warn_discarding_alpha(true, false, Straight, false));
        assert!(!warn_discarding_alpha(true, false, Straight, true));
        // Alpha-less target with a deliberate disposition: always silent.
        assert!(!warn_discarding_alpha(true, false, Premultiplied, false));
        assert!(!warn_discarding_alpha(true, false, Opaque, false));
    }

    #[test]
    fn alpha_less_output_keeps_requested_metadata() {
        for target_alpha in [AlphaMode::Premultiplied, AlphaMode::Opaque] {
            let mut data = Vec::new();
            for value in [0.8f32, 0.4, 0.2, 0.5] {
                data.extend_from_slice(&value.to_le_bytes());
            }
            let image = make_image(
                data,
                1,
                1,
                ktx2::Format::R32G32B32A32_SFLOAT,
                ColorSpace::Linear,
                AlphaMode::Straight,
            );
            let output = convert(
                image,
                ConvertSettings {
                    format: Some(TargetFormat::Uncompressed(ktx2::Format::R32G32B32_SFLOAT)),
                    output_alpha: Some(target_alpha),
                    container: Container::Raw,
                    ..Default::default()
                },
            )
            .unwrap();
            let PipelineOutput::Raw(image) = output else {
                panic!("expected raw output");
            };
            assert_eq!(image.surfaces[0][0].alpha, target_alpha);
        }
    }
}

/// Regression tests for the f32 → alpha-less (BC6H) alpha handling. Gated on
/// the AMD encoder because it both encodes BC6H and can decode a block back.
#[cfg(all(test, feature = "encoder-amd"))]
mod bc6h_alpha_tests {
    use super::*;
    use crate::surface::Surface;

    /// Build a 4×4 `R32G32B32A32_SFLOAT` image with every texel set to
    /// `(r, g, b, a)`, tagged with the given source alpha mode.
    fn solid_rgba_f32(r: f32, g: f32, b: f32, a: f32, alpha: AlphaMode) -> Image {
        let mut data = Vec::with_capacity(4 * 4 * 16);
        for _ in 0..(4 * 4) {
            for c in [r, g, b, a] {
                data.extend_from_slice(&c.to_le_bytes());
            }
        }
        Image {
            surfaces: vec![vec![Surface {
                data,
                width: 4,
                height: 4,
                depth: 1,
                stride: 4 * 16,
                slice_stride: 0,
                format: ktx2::Format::R32G32B32A32_SFLOAT,
                color_space: ColorSpace::Linear,
                alpha,
            }]],
            kind: crate::TextureKind::Texture2D,
        }
    }

    /// Encode `image` to BC6H (Raw container) and decode the first texel's RGB.
    fn encode_bc6h_first_texel(image: Image, output_alpha: Option<AlphaMode>) -> [f32; 3] {
        let out = convert(
            image,
            ConvertSettings {
                format: Some(TargetFormat::Compressed {
                    format: ktx2::Format::BC6H_UFLOAT_BLOCK,
                    encoder: crate::encoders::Encoder::Auto,
                }),
                container: Container::Raw,
                quality: Quality::UltraFast,
                output_alpha,
                ..Default::default()
            },
        )
        .unwrap();
        let img = match out {
            PipelineOutput::Raw(img) => img,
            _ => panic!("expected Raw output"),
        };
        let block: [u8; 16] = img.surfaces[0][0].data.as_slice().try_into().unwrap();
        let texels = ctt_compressonator::bc6h::decompress_block(&block).unwrap();
        [
            half::f16::from_bits(texels[0]).to_f32(),
            half::f16::from_bits(texels[1]).to_f32(),
            half::f16::from_bits(texels[2]).to_f32(),
        ]
    }

    fn assert_rgb_close(actual: [f32; 3], expected: [f32; 3]) {
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() <= 0.02,
                "channel {i}: got {}, expected {} (full {actual:?} vs {expected:?})",
                actual[i],
                expected[i],
            );
        }
    }

    #[test]
    fn zero_alpha_straight_keeps_color() {
        // The core fix: α=0 must not zero RGB when the target has no alpha.
        let img = solid_rgba_f32(4.0, 2.0, 1.0, 0.0, AlphaMode::Straight);
        assert_rgb_close(encode_bc6h_first_texel(img, None), [4.0, 2.0, 1.0]);
    }

    #[test]
    fn partial_alpha_straight_keeps_color() {
        let img = solid_rgba_f32(4.0, 2.0, 1.0, 0.5, AlphaMode::Straight);
        assert_rgb_close(encode_bc6h_first_texel(img, None), [4.0, 2.0, 1.0]);
    }

    #[test]
    fn zero_alpha_opaque_source_keeps_color() {
        let img = solid_rgba_f32(4.0, 2.0, 1.0, 0.0, AlphaMode::Opaque);
        assert_rgb_close(encode_bc6h_first_texel(img, None), [4.0, 2.0, 1.0]);
    }

    #[test]
    fn premultiplied_output_bakes_partial_alpha() {
        let img = solid_rgba_f32(4.0, 2.0, 1.0, 0.5, AlphaMode::Straight);
        let out = encode_bc6h_first_texel(img, Some(AlphaMode::Premultiplied));
        assert_rgb_close(out, [2.0, 1.0, 0.5]);
    }

    #[test]
    fn premultiplied_output_bakes_zero_alpha_to_black() {
        let img = solid_rgba_f32(4.0, 2.0, 1.0, 0.0, AlphaMode::Straight);
        let out = encode_bc6h_first_texel(img, Some(AlphaMode::Premultiplied));
        assert_rgb_close(out, [0.0, 0.0, 0.0]);
    }
}
