pub use ctt_astcenc as astc;

use crate::encoders::Quality;
use crate::encoders::backend::Encoder;
use crate::error::Result;
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::FormatExt as _;

/// How a normal map's X/Y components are laid out across the four astcenc
/// input channels.
///
/// **Only meaningful inside [`AstcencUsage::NormalMap`]** — other usages
/// derive their swizzle from the data shape, not from this enum. The codec
/// stores normals as a 2-channel signal and the shader reconstructs Z as
/// `sqrt(1 - x² - y²)`. The two variants differ only in which input channel
/// carries X vs. Y; pick the one your shader's sample swizzle already
/// expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NormalSwizzle {
    /// `rrrg` — X replicated into RGB, Y in alpha. The astcenc default.
    #[default]
    AstcDefault,
    /// `gggr` — Y replicated into RGB, X in alpha. Matches BC5n's layout
    /// so shader code written against BC5n normals keeps working unchanged
    /// after swapping to ASTC.
    Bc5Compat,
}

/// What the texture data represents. Drives profile, flags, and swizzle.
///
/// Picking the right usage is the single most important quality lever: it
/// avoids spending encoding bits on components you don't actually need.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AstcencUsage {
    /// Generic color texture. Uses LDR or LDR sRGB profile based on the
    /// surface's color space; passes RGBA through unchanged.
    #[default]
    Color,
    /// 2-channel tangent-space normal map. Sets `MAP_NORMAL`, optimizes for
    /// angular error, and applies the chosen 2-channel input swizzle. Z must
    /// be reconstructed in the shader.
    NormalMap { swizzle: NormalSwizzle },
    /// Single-channel mask data (roughness, AO, height, ...). Uses the `rrr1`
    /// swizzle so only the red channel is encoded.
    SingleChannel,
    /// Two-channel mask data (e.g. metallic+roughness). Uses the `rrrg`
    /// swizzle so only red and green are encoded.
    TwoChannel,
    /// HDR RGB with LDR alpha — the recommended HDR mode. Requires fp16 input.
    HdrRgb,
    /// All-HDR RGBA. Requires fp16 input.
    HdrRgba,
    /// HDR data preprocessed into LDR RGBM form.
    ///
    /// RGBM is a "fake HDR" packing: HDR color is stored in a 4-channel LDR
    /// texture where RGB holds a normalized color and the M (alpha) channel
    /// holds a per-pixel shared multiplier. The shader reconstructs the HDR
    /// value as `rgb * m * rgbm_m_scale`. Use this to ship HDR-ish content
    /// through formats and platforms that only support LDR sampling (common
    /// for lightmaps, reflection probes, low-end mobile, and any pipeline
    /// where true fp16/HDR textures aren't an option). The trade-offs are
    /// banding in highlights, a fixed dynamic range capped at
    /// `rgbm_m_scale`, and a hard floor on `m` — values that quantize to
    /// zero produce black or NaN pixels.
    ///
    /// **The caller must do the RGBM packing before handing the surface to
    /// the encoder** — this variant only flips on codec heuristics. Tune
    /// the scale via [`AstcencSettings::rgbm_m_scale`] and follow the
    /// upstream guidance to floor `m` at ~16/255 or 32/255 before encoding.
    Rgbm,
}

/// astcenc-specific encoder settings.
///
/// The `usage` field captures the high-level intent (color, normal map,
/// mask, HDR, RGBM) and is the most important knob — it drives the profile,
/// flag bits, and input swizzle. The remaining bools toggle orthogonal
/// codec features; the `Option` fields override codec defaults only when
/// `Some`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AstcencSettings {
    /// What the texture represents. Defaults to [`AstcencUsage::Color`].
    pub usage: AstcencUsage,
    /// Override the quality preset that would otherwise be derived from the
    /// pipeline's generic [`Quality`]. Use this to pin a specific astcenc
    /// effort level or any custom point on the `[0.0, 100.0]` continuum.
    pub preset: Option<astc::Preset>,
    /// Weight RGB error by the alpha channel — improves alpha precision in
    /// transparent regions at the cost of RGB fidelity there. Recommended for
    /// textures sampled with linear filtering on cutout/transparent content.
    pub use_alpha_weight: bool,
    /// Optimize for perceptual error rather than PSNR. Only meaningful for
    /// color and normal-map usages.
    pub perceptual: bool,
    /// Tune for the `decode_unorm8` ASTC decode mode instead of `decode_fp16`.
    ///
    /// ASTC blocks can be expanded by the GPU two ways: as fp16 (the
    /// historical default, exact intermediate values) or as unorm8 (rounded
    /// to 8-bit during decode). The two paths round differently in the last
    /// bit, so the encoder picks slightly different bit patterns to land on
    /// whichever the runtime will use. Mismatched flag + decode mode costs
    /// a small amount of quality; matched gains it back.
    ///
    /// Set this when the texture will be sampled as a unorm8 texel format
    /// at runtime — the common case for color textures on mobile and most
    /// modern desktop pipelines. Leave it off for HDR content sampled as
    /// fp16. LDR sRGB always decodes via unorm8 regardless, so this flag
    /// is a no-op for sRGB color usages.
    pub decode_unorm8: bool,
    /// Custom per-channel error weights `[r, g, b, a]`. Higher values make
    /// the codec spend more bits on that channel. `None` keeps the codec's
    /// defaults (which already include usage-specific tuning, e.g. RGBM
    /// alpha boost).
    pub channel_weights: Option<[f32; 4]>,
    /// Override the RGBM shared-multiplier scale (default 5.0 in RGBM mode).
    /// Ignored unless `usage` is [`AstcencUsage::Rgbm`]. When changing this,
    /// also bump `channel_weights[3]` to roughly `2 * scale` so the M
    /// channel stays accurate.
    pub rgbm_m_scale: Option<f32>,
}

/// All 14 valid ASTC 2D block sizes.
const SUPPORTED_FORMATS: &[ktx2::Format] = &[
    ktx2::Format::ASTC_4x4_UNORM_BLOCK,
    ktx2::Format::ASTC_5x4_UNORM_BLOCK,
    ktx2::Format::ASTC_5x5_UNORM_BLOCK,
    ktx2::Format::ASTC_6x5_UNORM_BLOCK,
    ktx2::Format::ASTC_6x6_UNORM_BLOCK,
    ktx2::Format::ASTC_8x5_UNORM_BLOCK,
    ktx2::Format::ASTC_8x6_UNORM_BLOCK,
    ktx2::Format::ASTC_8x8_UNORM_BLOCK,
    ktx2::Format::ASTC_10x5_UNORM_BLOCK,
    ktx2::Format::ASTC_10x6_UNORM_BLOCK,
    ktx2::Format::ASTC_10x8_UNORM_BLOCK,
    ktx2::Format::ASTC_10x10_UNORM_BLOCK,
    ktx2::Format::ASTC_12x10_UNORM_BLOCK,
    ktx2::Format::ASTC_12x12_UNORM_BLOCK,
];

pub struct AstcencEncoder;

impl Encoder for AstcencEncoder {
    type Settings = AstcencSettings;

    fn name() -> &'static str {
        "astcenc"
    }

    fn supported_formats() -> &'static [ktx2::Format] {
        SUPPORTED_FORMATS
    }

    fn required_input_format(_format: ktx2::Format, settings: &AstcencSettings) -> ktx2::Format {
        match settings.usage {
            AstcencUsage::HdrRgb | AstcencUsage::HdrRgba => ktx2::Format::R16G16B16A16_SFLOAT,
            _ => ktx2::Format::R8G8B8A8_UNORM,
        }
    }

    fn compress(
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: &AstcencSettings,
    ) -> Result<Vec<u8>> {
        let (base, color_space) = format.normalize();

        let (block_width, block_height) =
            base.block_size().expect("ASTC format must have block size");

        // astcenc's image dimensions cross the FFI boundary into signed-int
        // arithmetic inside the C library; reject inputs that don't fit in i32
        // rather than risking overflow there.
        check_i32_dims(surface.width, surface.height)?;

        let profile = profile_for(settings.usage, color_space);
        let preset = settings.preset.unwrap_or_else(|| default_preset(quality));
        let flags = flags_for(settings);
        let swizzle = swizzle_for(settings.usage);
        let data_type = data_type_for(settings.usage);

        let mut config = astc::config_init(
            profile,
            block_width as u32,
            block_height as u32,
            1,
            preset,
            flags,
        )
        .map_err(|e| crate::error::Error::Compression(e.to_string()))?;

        if let Some([r, g, b, a]) = settings.channel_weights {
            config.cw_r_weight = r;
            config.cw_g_weight = g;
            config.cw_b_weight = b;
            config.cw_a_weight = a;
        }
        if let Some(scale) = settings.rgbm_m_scale {
            config.rgbm_m_scale = scale;
        }

        #[cfg(feature = "rayon")]
        let ctx = astc::Context::new_parallel(&config);
        #[cfg(not(feature = "rayon"))]
        let ctx = astc::Context::new(&config);
        let mut ctx = ctx.map_err(|e| crate::error::Error::Compression(e.to_string()))?;

        // astcenc_image has no row-stride field, so the C side assumes tightly
        // packed input. Repack when the surface carries padded rows; this costs
        // one full-image copy but only on the padded path (tight surfaces
        // borrow their data unchanged).
        let tight = surface.tight_data();
        let mut data_ptr = tight.as_ptr() as *mut std::ffi::c_void;
        let mut img = astc::bindings::astcenc_image {
            dim_x: surface.width,
            dim_y: surface.height,
            dim_z: 1,
            data_type,
            data: &mut data_ptr,
        };

        let output_size = astc_output_size(
            surface.width,
            surface.height,
            block_width as u32,
            block_height as u32,
        )?;
        let mut output = Vec::new();
        output.try_reserve_exact(output_size).map_err(|e| {
            crate::error::Error::Compression(format!(
                "cannot allocate {output_size} bytes for ASTC output: {e}"
            ))
        })?;
        output.resize(output_size, 0);

        ctx.compress(&mut img, swizzle, &mut output)
            .map_err(|e| crate::error::Error::Compression(e.to_string()))?;

        Ok(output)
    }
}

fn profile_for(usage: AstcencUsage, color_space: ColorSpace) -> astc::Profile {
    match usage {
        AstcencUsage::HdrRgb => astc::Profile::HdrRgbLdrA,
        AstcencUsage::HdrRgba => astc::Profile::Hdr,
        // Normal maps and single/two-channel masks are intrinsically linear.
        AstcencUsage::NormalMap { .. } | AstcencUsage::SingleChannel | AstcencUsage::TwoChannel => {
            astc::Profile::Ldr
        }
        AstcencUsage::Color | AstcencUsage::Rgbm => match color_space {
            ColorSpace::Srgb => astc::Profile::LdrSrgb,
            ColorSpace::Linear => astc::Profile::Ldr,
        },
    }
}

fn flags_for(settings: &AstcencSettings) -> astc::Flags {
    // SELF_DECOMPRESS_ONLY is a pure compression-side optimization (smaller
    // lookup tables, faster context creation) — safe to always set because
    // ctt never decompresses through this codec instance.
    let mut flags = astc::Flags::SELF_DECOMPRESS_ONLY;
    match settings.usage {
        AstcencUsage::NormalMap { .. } => flags |= astc::Flags::MAP_NORMAL,
        AstcencUsage::Rgbm => flags |= astc::Flags::MAP_RGBM,
        _ => {}
    }
    if settings.use_alpha_weight {
        flags |= astc::Flags::USE_ALPHA_WEIGHT;
    }
    if settings.perceptual {
        flags |= astc::Flags::USE_PERCEPTUAL;
    }
    if settings.decode_unorm8 {
        flags |= astc::Flags::USE_DECODE_UNORM8;
    }
    flags
}

fn swizzle_for(usage: AstcencUsage) -> astc::Swizzle {
    match usage {
        AstcencUsage::NormalMap {
            swizzle: NormalSwizzle::AstcDefault,
        } => astc::Swizzle::RRRG,
        AstcencUsage::NormalMap {
            swizzle: NormalSwizzle::Bc5Compat,
        } => astc::Swizzle::GGGR,
        AstcencUsage::SingleChannel => astc::Swizzle::RRR1,
        AstcencUsage::TwoChannel => astc::Swizzle::RRRG,
        AstcencUsage::Color | AstcencUsage::HdrRgb | AstcencUsage::HdrRgba | AstcencUsage::Rgbm => {
            astc::Swizzle::IDENTITY
        }
    }
}

fn data_type_for(usage: AstcencUsage) -> astc::bindings::astcenc_type {
    use astc::bindings::{astcenc_type_ASTCENC_TYPE_F16, astcenc_type_ASTCENC_TYPE_U8};
    match usage {
        AstcencUsage::HdrRgb | AstcencUsage::HdrRgba => astcenc_type_ASTCENC_TYPE_F16,
        _ => astcenc_type_ASTCENC_TYPE_U8,
    }
}

fn default_preset(quality: Quality) -> astc::Preset {
    // astcenc quality is a float on [0.0, 100.0]; the named presets are just
    // points on that continuum (Fastest=0, Fast=10, Medium=60, Thorough=98,
    // VeryThorough=99, Exhaustive=100). Keep all six tiers distinct and
    // monotonic, but hold Basic at Medium (60.0) to preserve default
    // performance.
    match quality {
        Quality::UltraFast => astc::Preset::Fastest, // 0.0
        Quality::VeryFast => astc::Preset::Fast,     // 10.0
        Quality::Fast => astc::Preset::Custom(35.0),
        Quality::Basic => astc::Preset::Medium,        // 60.0
        Quality::Slow => astc::Preset::Thorough,       // 98.0
        Quality::VerySlow => astc::Preset::Exhaustive, // 100.0
    }
}

/// Bytes of ASTC output for a `width × height` image at the given block size.
///
/// Every ASTC block is a fixed 16 bytes.
fn astc_output_size(width: u32, height: u32, block_width: u32, block_height: u32) -> Result<usize> {
    let blocks_x = width.div_ceil(block_width) as usize;
    let blocks_y = height.div_ceil(block_height) as usize;
    blocks_x
        .checked_mul(blocks_y)
        .and_then(|blocks| blocks.checked_mul(16))
        .ok_or_else(|| {
            crate::error::Error::InvalidDimensions(format!(
                "ASTC output size overflows usize for {width}x{height} with \
                 {block_width}x{block_height} blocks"
            ))
        })
}

/// Reject dimensions that don't fit in `i32`. astcenc holds image extents as
/// C ints internally, so pathological inputs would overflow there; fail
/// cleanly at the boundary instead.
fn check_i32_dims(width: u32, height: u32) -> Result<()> {
    if i32::try_from(width).is_err() || i32::try_from(height).is_err() {
        return Err(crate::error::Error::InvalidDimensions(format!(
            "astcenc encoder requires width/height to fit in i32, got {width}x{height}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    /// Build an RGBA8 surface with a non-uniform pattern and the given row
    /// stride. Bytes between the packed row payload and `stride` are filled
    /// with a sentinel so a stride-ignoring read would corrupt the result.
    fn patterned(width: u32, height: u32, stride: u32) -> Surface {
        assert!(stride >= width * 4);
        let mut data = vec![0xABu8; (stride * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let off = (y * stride + x * 4) as usize;
                let v = (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)) & 0xff) as u8;
                data[off..off + 4].copy_from_slice(&[
                    v,
                    v.wrapping_add(50),
                    v.wrapping_add(100),
                    255,
                ]);
            }
        }
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn output_size_does_not_wrap() {
        // 65536×65536 at 4×4 → 16384×16384 blocks × 16 bytes. The u32 product
        // (4_294_967_296) wraps to 0; the widened computation must not.
        let size = astc_output_size(65536, 65536, 4, 4).unwrap();
        assert_eq!(size, 16384usize * 16384 * 16);
        assert!(size > u32::MAX as usize);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn output_size_overflow_is_rejected() {
        assert!(astc_output_size(65536, 65536, 4, 4).is_err());
    }

    #[test]
    fn quality_presets_distinct_and_monotonic() {
        let tiers = [
            Quality::UltraFast,
            Quality::VeryFast,
            Quality::Fast,
            Quality::Basic,
            Quality::Slow,
            Quality::VerySlow,
        ];
        let values: Vec<f32> = tiers
            .iter()
            .map(|&q| default_preset(q).as_float())
            .collect();
        for pair in values.windows(2) {
            assert!(
                pair[0] < pair[1],
                "quality presets must strictly increase, got {values:?}",
            );
        }
        // Basic must stay at Medium to preserve default performance.
        assert_eq!(default_preset(Quality::Basic).as_float(), 60.0);
    }

    #[test]
    fn padded_stride_matches_tight() {
        let tight = patterned(8, 8, 8 * 4);
        let padded = patterned(8, 8, 8 * 4 + 16);
        let a = AstcencEncoder::compress(
            &tight,
            ktx2::Format::ASTC_4x4_UNORM_BLOCK,
            Quality::Fast,
            &AstcencSettings::default(),
        )
        .unwrap();
        let b = AstcencEncoder::compress(
            &padded,
            ktx2::Format::ASTC_4x4_UNORM_BLOCK,
            Quality::Fast,
            &AstcencSettings::default(),
        )
        .unwrap();
        assert_eq!(a, b, "padded-stride encode must match tight encode");
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn shared_context_matches_single_worker() {
        let surface = patterned(19, 13, 19 * 4 + 12);
        crate::encoders::assert_parallel_matches_serial(|| {
            AstcencEncoder::compress(
                &surface,
                ktx2::Format::ASTC_4x4_UNORM_BLOCK,
                Quality::Fast,
                &AstcencSettings::default(),
            )
            .unwrap()
        });
    }
}
