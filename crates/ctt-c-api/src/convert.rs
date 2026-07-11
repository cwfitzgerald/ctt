use crate::error::{Status, catch_panic, map_error, set_last_error};
use crate::formats::to_ctt_format;
use crate::image::{Image, take_image};
use crate::output::PipelineOutput;
use crate::types::{
    AlphaMode, ColorSpace, Format, MipmapFilter, OptionalAlphaMode, OptionalColorSpace,
    OptionalSize, OptionalSwizzle, Quality,
};

// ---------------------------------------------------------------------------
// bc7enc settings
// ---------------------------------------------------------------------------

/// Optional `bool` (matches Rust's `Option<bool>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalBool {
    pub present: bool,
    pub value: bool,
}

/// Optional 32-bit unsigned int (matches Rust's `Option<u32>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalU32 {
    pub present: bool,
    pub value: u32,
}

/// Optional 8-bit unsigned int (matches Rust's `Option<u8>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalU8 {
    pub present: bool,
    pub value: u8,
}

/// Optional per-channel error weights `[r, g, b, a]` as integers
/// (matches Rust's `Option<[u32; 4]>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalChannelWeightsU32 {
    pub present: bool,
    pub value: [u32; 4],
}

/// Optional RGB error weights `[r, g, b]` as floats
/// (matches Rust's `Option<[f32; 3]>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalRgbWeights {
    pub present: bool,
    pub value: [f32; 3],
}

/// Settings for the `bc7enc-rdo` encoder.
///
/// `perceptual` and `mode6_only` are plain toggles. The optional fields
/// override the values baked into the quality-derived preset only when
/// `present == true`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Bc7encSettings {
    /// Use perceptual error metrics (default `true`).
    pub perceptual: bool,
    /// Restrict to BC7 mode 6 (RGBA, no partitions). Fastest mode,
    /// lower quality on high-contrast blocks.
    pub mode6_only: bool,
    /// Override the preset's parity-bit search choice.
    pub pbit_search: OptionalBool,
    /// Override the preset's "uber" refinement level (0..=4).
    pub uber_level: OptionalU32,
    /// Override the preset's per-channel error weights `[r, g, b, a]`.
    pub channel_weights: OptionalChannelWeightsU32,
}

/// Default settings for the `bc7enc-rdo` encoder.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_bc7enc_settings_default() -> Bc7encSettings {
    Bc7encSettings {
        perceptual: true,
        mode6_only: false,
        pbit_search: OptionalBool {
            present: false,
            value: false,
        },
        uber_level: OptionalU32 {
            present: false,
            value: 0,
        },
        channel_weights: OptionalChannelWeightsU32 {
            present: false,
            value: [1, 1, 1, 1],
        },
    }
}

// ---------------------------------------------------------------------------
// Intel ISPC settings
// ---------------------------------------------------------------------------

/// How the Intel ISPC BC7 encoder should treat the alpha channel.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntelBc7Alpha {
    /// Derive from the surface's [`AlphaMode`]: opaque → opaque presets,
    /// anything else → alpha-aware presets.
    Auto,
    /// Force opaque presets — RGB-only modes 0–3.
    Opaque,
    /// Force alpha-aware presets — modes 4–7 are searched.
    Alpha,
}

/// Settings for the Intel ISPC texture compressor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IntelSettings {
    /// How BC7 should handle the alpha channel. Ignored for other formats.
    pub bc7_alpha: IntelBc7Alpha,
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_intel_settings_default() -> IntelSettings {
    IntelSettings {
        bc7_alpha: IntelBc7Alpha::Auto,
    }
}

// ---------------------------------------------------------------------------
// etcpak settings
// ---------------------------------------------------------------------------

/// Settings for the etcpak encoder.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EtcpakSettings {
    /// Enable dithering for ETC1 / BC1.
    pub dither: bool,
    /// Enable heuristic-based fast mode selection for ETC2 RGB / RGBA.
    pub use_heuristics: bool,
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_etcpak_settings_default() -> EtcpakSettings {
    EtcpakSettings {
        dither: false,
        use_heuristics: false,
    }
}

// ---------------------------------------------------------------------------
// AMD Compressonator settings
// ---------------------------------------------------------------------------

/// What the texture data represents — drives default BC1/BC2/BC3 channel
/// weighting and the `auto` branch of [`AmdBc7Alpha`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdUsage {
    /// Generic color texture. BC1/2/3 use BT.601 luminance weights
    /// `[0.3086, 0.6094, 0.0820]`.
    Color,
    /// Tangent-space normals packed into BC1/2/3. Uses uniform RGB
    /// weights so X/Y aren't sacrificed to a luminance prior.
    NormalMap,
    /// Mask / data channels (metallic+roughness+AO, etc.) packed into
    /// BC1/2/3. Uses uniform RGB weights — perceptual luminance bias is
    /// the wrong model when the channels aren't color.
    Data,
}

/// How the BC7 encoder should treat the alpha channel.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdBc7Alpha {
    /// Derive from the surface's [`AlphaMode`]: opaque → behaves like
    /// `Opaque`, anything else → `Full`.
    Auto,
    /// No meaningful alpha — concentrate on RGB modes.
    Opaque,
    /// Alpha is meaningful; full mode search, no palette restrictions
    /// (highest alpha quality, slowest encode).
    Full,
    /// Alpha is meaningful but with restricted color + alpha palettes
    /// (faster, marginally lower alpha quality).
    Restricted,
}

/// Settings for the AMD Compressonator encoder.
///
/// Most fields apply only to a subset of formats; the docs name which.
/// Picking the right [`AmdUsage`] is the cheapest quality lever for
/// BC1/BC2/BC3 — it controls per-channel error weighting.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdSettings {
    /// What the texture represents.
    pub usage: AmdUsage,
    /// Explicit RGB error weights for BC1/BC2/BC3. Ignored otherwise.
    pub channel_weights: OptionalRgbWeights,
    /// BC7 alpha handling. Ignored for non-BC7 formats.
    pub bc7_alpha: AmdBc7Alpha,
    /// BC7 mode mask — bit `n` enables mode `n` (0..=7).
    /// Ignored for non-BC7 formats.
    pub bc7_mode_mask: OptionalU8,
    /// BC6H mode mask — 14-bit bitfield. Ignored for non-BC6H formats.
    pub bc6h_mode_mask: OptionalU32,
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_amd_settings_default() -> AmdSettings {
    AmdSettings {
        usage: AmdUsage::Color,
        channel_weights: OptionalRgbWeights {
            present: false,
            value: [1.0, 1.0, 1.0],
        },
        bc7_alpha: AmdBc7Alpha::Auto,
        bc7_mode_mask: OptionalU8 {
            present: false,
            value: 0xFF,
        },
        bc6h_mode_mask: OptionalU32 {
            present: false,
            value: 0,
        },
    }
}

// ---------------------------------------------------------------------------
// astcenc settings
// ---------------------------------------------------------------------------

/// Layout of a normal map's X and Y across the four input channels.
///
/// **Only meaningful when the enclosing `ctt_astcenc_usage` has tag
/// `CTT_ASTCENC_USAGE_NORMAL_MAP`.** For every other usage, the swizzle is
/// determined by the data shape (single-channel, two-channel, color, ...)
/// and this field is ignored.
///
/// astcenc encodes only X and Y; the shader reconstructs Z as
/// `sqrt(1 - x*x - y*y)`. Pick the variant that matches your shader's
/// sample swizzle:
///
///   - `ASTC_DEFAULT` (`rrrg`) — the astcenc default. X is replicated into
///     RGB and Y is in alpha.
///   - `BC5_COMPAT` (`gggr`) — matches BC5n's layout so existing BC5n
///     shader code keeps working after the swap to ASTC.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstcencNormalSwizzle {
    AstcDefault = 0,
    Bc5Compat = 1,
}

/// What the texture data represents. Drives profile, flag bits, and swizzle.
///
/// Picking the right usage is the most important quality lever — it avoids
/// wasting encoding bits on components the asset doesn't actually need.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum AstcencUsage {
    /// Generic color (LDR or LDR sRGB based on the surface's color space).
    Color,
    /// 2-channel tangent-space normal map. Sets `MAP_NORMAL`.
    NormalMap(AstcencNormalSwizzle),
    /// Single-channel mask (roughness, AO, height); only red is encoded.
    SingleChannel,
    /// Two-channel mask (e.g. metallic+roughness); only red and green are encoded.
    TwoChannel,
    /// HDR RGB + LDR alpha. Requires fp16 input.
    HdrRgb,
    /// All-HDR RGBA. Requires fp16 input.
    HdrRgba,
    /// HDR data preprocessed into LDR RGBM form. Sets `MAP_RGBM`.
    ///
    /// RGBM is a "fake HDR" packing: HDR color is stored in a 4-channel
    /// LDR texture where RGB holds a normalized color and the M (alpha)
    /// channel holds a per-pixel shared multiplier. The shader
    /// reconstructs the HDR value as `rgb * m * rgbm_m_scale`. Use this to
    /// ship HDR-ish content through formats and platforms that only
    /// support LDR sampling (lightmaps, reflection probes, low-end mobile,
    /// any pipeline where true fp16/HDR textures aren't an option).
    /// Trade-offs: banding in highlights, a fixed dynamic range capped at
    /// `rgbm_m_scale`, and a hard floor on `m` (values that quantize to
    /// zero produce black or NaN pixels).
    ///
    /// **The caller must do the RGBM packing before passing the surface
    /// in** — this tag only flips on codec heuristics. Tune the scale via
    /// `ctt_astcenc_settings.rgbm_m_scale` and follow the upstream
    /// guidance to floor `m` at ~16/255 or 32/255 before encoding.
    Rgbm,
}

/// astcenc effort preset. `CUSTOM` carries any value in `[0.0, 100.0]`.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum AstcencPreset {
    /// 0.0 — fastest, lowest quality.
    Fastest,
    /// 10.0 — fast.
    Fast,
    /// 60.0 — medium.
    Medium,
    /// 98.0 — thorough.
    Thorough,
    /// 99.0 — very thorough.
    VeryThorough,
    /// 100.0 — exhaustive, highest quality.
    Exhaustive,
    /// Any value in `[0.0, 100.0]`; clamped on use.
    Custom(f32),
}

/// Optional [`AstcencPreset`] (matches Rust's `Option<AstcencPreset>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalAstcencPreset {
    pub present: bool,
    pub value: AstcencPreset,
}

/// Optional per-channel error weights `[r, g, b, a]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalChannelWeights {
    pub present: bool,
    pub value: [f32; 4],
}

/// Optional 32-bit float (matches Rust's `Option<f32>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalF32 {
    pub present: bool,
    pub value: f32,
}

/// Settings for the `astcenc` encoder.
///
/// `usage` is the single most important field — it picks the profile, flags,
/// and input swizzle automatically. The bool fields toggle orthogonal codec
/// features; the optional fields override codec defaults only when present.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AstcencSettings {
    /// What the texture represents. See [`AstcencUsage`].
    pub usage: AstcencUsage,
    /// Override the quality preset that would otherwise be derived from
    /// `ctt_convert_settings.quality`.
    pub preset: OptionalAstcencPreset,
    /// Weight RGB error by alpha — improves alpha precision in transparent
    /// regions at the cost of RGB fidelity there.
    pub use_alpha_weight: bool,
    /// Optimize for perceptual error rather than PSNR. Only meaningful for
    /// color and normal-map usages.
    pub perceptual: bool,
    /// Tune for the `decode_unorm8` ASTC decode mode instead of `decode_fp16`.
    ///
    /// ASTC blocks can be expanded by the GPU two ways: as fp16 (the
    /// historical default, exact intermediate values) or as unorm8
    /// (rounded to 8-bit during decode). The two paths round differently
    /// in the last bit, so the encoder picks slightly different bit
    /// patterns to land on whichever the runtime will use. Mismatched
    /// flag + decode mode costs a small amount of quality; matched gains
    /// it back.
    ///
    /// Set this when the texture will be sampled as a unorm8 texel format
    /// at runtime — the common case for color textures on mobile and most
    /// modern desktop pipelines. Leave it off for HDR content sampled as
    /// fp16. LDR sRGB always decodes via unorm8 regardless, so this flag
    /// is a no-op for sRGB color usages.
    pub decode_unorm8: bool,
    /// Custom per-channel error weights. Higher values spend more bits on
    /// that channel; leave absent to keep codec defaults.
    pub channel_weights: OptionalChannelWeights,
    /// Override the RGBM shared-multiplier scale (default 5.0). Ignored
    /// unless `usage.tag == CTT_ASTCENC_USAGE_RGBM`. See
    /// [`AstcencUsage::Rgbm`] for what RGBM is and why you'd use it. When
    /// raising this, also bump `channel_weights.value[3]` to roughly
    /// `2 * scale` so the M channel stays accurate.
    pub rgbm_m_scale: OptionalF32,
}

/// Default settings: `Color` usage, codec defaults for everything else.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_astcenc_settings_default() -> AstcencSettings {
    AstcencSettings {
        usage: AstcencUsage::Color,
        preset: OptionalAstcencPreset {
            present: false,
            value: AstcencPreset::Medium,
        },
        use_alpha_weight: false,
        perceptual: false,
        decode_unorm8: false,
        channel_weights: OptionalChannelWeights {
            present: false,
            value: [1.0, 1.0, 1.0, 1.0],
        },
        rgbm_m_scale: OptionalF32 {
            present: false,
            value: 5.0,
        },
    }
}

#[cfg(feature = "encoder-astcenc")]
fn astcenc_preset_into(p: AstcencPreset) -> ctt::encoders::astcenc::astc::Preset {
    use ctt::encoders::astcenc::astc::Preset;
    match p {
        AstcencPreset::Fastest => Preset::Fastest,
        AstcencPreset::Fast => Preset::Fast,
        AstcencPreset::Medium => Preset::Medium,
        AstcencPreset::Thorough => Preset::Thorough,
        AstcencPreset::VeryThorough => Preset::VeryThorough,
        AstcencPreset::Exhaustive => Preset::Exhaustive,
        AstcencPreset::Custom(v) => Preset::Custom(v),
    }
}

#[cfg(feature = "encoder-amd")]
fn amd_usage_into(u: AmdUsage) -> ctt::encoders::compressonator::AmdUsage {
    use ctt::encoders::compressonator::AmdUsage as U;
    match u {
        AmdUsage::Color => U::Color,
        AmdUsage::NormalMap => U::NormalMap,
        AmdUsage::Data => U::Data,
    }
}

#[cfg(feature = "encoder-amd")]
fn amd_bc7_alpha_into(a: AmdBc7Alpha) -> ctt::encoders::compressonator::AmdBc7Alpha {
    use ctt::encoders::compressonator::AmdBc7Alpha as A;
    match a {
        AmdBc7Alpha::Auto => A::Auto,
        AmdBc7Alpha::Opaque => A::Opaque,
        AmdBc7Alpha::Full => A::Full,
        AmdBc7Alpha::Restricted => A::Restricted,
    }
}

#[cfg(feature = "encoder-intel")]
fn intel_bc7_alpha_into(a: IntelBc7Alpha) -> ctt::encoders::ispc::IspcBc7Alpha {
    use ctt::encoders::ispc::IspcBc7Alpha as A;
    match a {
        IntelBc7Alpha::Auto => A::Auto,
        IntelBc7Alpha::Opaque => A::Opaque,
        IntelBc7Alpha::Alpha => A::Alpha,
    }
}

#[cfg(feature = "encoder-astcenc")]
fn astcenc_usage_into(u: AstcencUsage) -> ctt::encoders::astcenc::AstcencUsage {
    use ctt::encoders::astcenc::{AstcencUsage as U, NormalSwizzle};
    match u {
        AstcencUsage::Color => U::Color,
        AstcencUsage::NormalMap(s) => U::NormalMap {
            swizzle: match s {
                AstcencNormalSwizzle::AstcDefault => NormalSwizzle::AstcDefault,
                AstcencNormalSwizzle::Bc5Compat => NormalSwizzle::Bc5Compat,
            },
        },
        AstcencUsage::SingleChannel => U::SingleChannel,
        AstcencUsage::TwoChannel => U::TwoChannel,
        AstcencUsage::HdrRgb => U::HdrRgb,
        AstcencUsage::HdrRgba => U::HdrRgba,
        AstcencUsage::Rgbm => U::Rgbm,
    }
}

// ---------------------------------------------------------------------------
// Encoder (tagged union)
// ---------------------------------------------------------------------------

/// User-facing encoder choice for compressed targets.
///
/// `Auto` picks the best compiled-in encoder for the requested format. Other
/// variants pin a specific backend and carry its settings. All variants are
/// always present in the ABI; selecting an encoder whose feature is disabled
/// at compile time returns `CTT_STATUS_ENCODER_NOT_COMPILED_IN`.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum Encoder {
    Auto,
    Bc7enc(Bc7encSettings),
    Intel(IntelSettings),
    Etcpak(EtcpakSettings),
    Amd(AmdSettings),
    Astcenc(AstcencSettings),
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_encoder_auto() -> Encoder {
    Encoder::Auto
}

impl Encoder {
    fn into_inner(self) -> Result<ctt::encoders::Encoder, Status> {
        match self {
            Encoder::Auto => Ok(ctt::encoders::Encoder::Auto),
            Encoder::Bc7enc(_settings) => {
                #[cfg(feature = "encoder-bc7enc")]
                {
                    Ok(ctt::encoders::Encoder::Bc7enc(
                        ctt::encoders::bc7enc::Bc7encSettings {
                            perceptual: _settings.perceptual,
                            mode6_only: _settings.mode6_only,
                            pbit_search: _settings
                                .pbit_search
                                .present
                                .then_some(_settings.pbit_search.value),
                            uber_level: _settings
                                .uber_level
                                .present
                                .then_some(_settings.uber_level.value),
                            channel_weights: _settings
                                .channel_weights
                                .present
                                .then_some(_settings.channel_weights.value),
                        },
                    ))
                }
                #[cfg(not(feature = "encoder-bc7enc"))]
                {
                    set_last_error("encoder 'bc7enc' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
            Encoder::Intel(_settings) => {
                #[cfg(feature = "encoder-intel")]
                {
                    Ok(ctt::encoders::Encoder::Intel(
                        ctt::encoders::ispc::IspcSettings {
                            bc7_alpha: intel_bc7_alpha_into(_settings.bc7_alpha),
                        },
                    ))
                }
                #[cfg(not(feature = "encoder-intel"))]
                {
                    set_last_error("encoder 'intel' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
            Encoder::Etcpak(_settings) => {
                #[cfg(feature = "encoder-etcpak")]
                {
                    Ok(ctt::encoders::Encoder::Etcpak(
                        ctt::encoders::etcpak::EtcpakSettings {
                            dither: _settings.dither,
                            use_heuristics: _settings.use_heuristics,
                        },
                    ))
                }
                #[cfg(not(feature = "encoder-etcpak"))]
                {
                    set_last_error("encoder 'etcpak' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
            Encoder::Amd(_settings) => {
                #[cfg(feature = "encoder-amd")]
                {
                    Ok(ctt::encoders::Encoder::Amd(
                        ctt::encoders::compressonator::AmdSettings {
                            usage: amd_usage_into(_settings.usage),
                            channel_weights: _settings
                                .channel_weights
                                .present
                                .then_some(_settings.channel_weights.value),
                            bc7_alpha: amd_bc7_alpha_into(_settings.bc7_alpha),
                            bc7_mode_mask: _settings
                                .bc7_mode_mask
                                .present
                                .then_some(_settings.bc7_mode_mask.value),
                            bc6h_mode_mask: _settings
                                .bc6h_mode_mask
                                .present
                                .then_some(_settings.bc6h_mode_mask.value),
                        },
                    ))
                }
                #[cfg(not(feature = "encoder-amd"))]
                {
                    set_last_error("encoder 'amd' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
            Encoder::Astcenc(_settings) => {
                #[cfg(feature = "encoder-astcenc")]
                {
                    Ok(ctt::encoders::Encoder::Astcenc(
                        ctt::encoders::astcenc::AstcencSettings {
                            usage: astcenc_usage_into(_settings.usage),
                            preset: _settings
                                .preset
                                .present
                                .then(|| astcenc_preset_into(_settings.preset.value)),
                            use_alpha_weight: _settings.use_alpha_weight,
                            perceptual: _settings.perceptual,
                            decode_unorm8: _settings.decode_unorm8,
                            channel_weights: _settings
                                .channel_weights
                                .present
                                .then_some(_settings.channel_weights.value),
                            rgbm_m_scale: _settings
                                .rgbm_m_scale
                                .present
                                .then_some(_settings.rgbm_m_scale.value),
                        },
                    ))
                }
                #[cfg(not(feature = "encoder-astcenc"))]
                {
                    set_last_error("encoder 'astcenc' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TargetFormat (tagged union)
// ---------------------------------------------------------------------------

/// Inner payload of the [`TargetFormat::Compressed`] variant.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompressedTargetFormat {
    pub format: Format,
    pub encoder: Encoder,
}

/// The target format for a conversion.
///
/// `None` keeps the input format (no conversion). `Uncompressed` produces a
/// plain pixel format. `Compressed` block-encodes with the chosen
/// [`Encoder`].
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum TargetFormat {
    None,
    Uncompressed(Format),
    Compressed(CompressedTargetFormat),
}

impl TargetFormat {
    fn into_inner(self) -> Result<Option<ctt::TargetFormat>, Status> {
        match self {
            TargetFormat::None => Ok(None),
            TargetFormat::Uncompressed(f) => {
                let Some(fmt) = to_ctt_format(f) else {
                    set_last_error(
                        "TargetFormat::Uncompressed: format must be a non-zero VkFormat",
                    );
                    return Err(Status::InvalidArgument);
                };
                Ok(Some(ctt::TargetFormat::Uncompressed(fmt)))
            }
            TargetFormat::Compressed(body) => {
                let Some(fmt) = to_ctt_format(body.format) else {
                    set_last_error("TargetFormat::Compressed: format must be a non-zero VkFormat");
                    return Err(Status::InvalidArgument);
                };
                let encoder = body.encoder.into_inner()?;
                Ok(Some(ctt::TargetFormat::Compressed {
                    format: fmt,
                    encoder,
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Container (tagged union)
// ---------------------------------------------------------------------------

/// Output container format.
///
/// `Ktx2` writes a plain KTX2 file. `Ktx2Zstd` / `Ktx2Zlib` apply the
/// corresponding KTX2 supercompression scheme; the carried integer is the
/// compression level passed to the underlying codec. `Dds` writes a DDS file.
/// `Raw` returns the processed image without serializing into a file format.
///
/// `Ktx2` is tag `0` so that a zero-initialized
/// [`ConvertSettings`] selects the same default container as
/// `ctt_convert_settings_default`.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum Container {
    Ktx2,
    Ktx2Zstd(i32),
    Ktx2Zlib(u8),
    Dds,
    Raw,
}

impl From<Container> for ctt::Container {
    fn from(c: Container) -> Self {
        match c {
            Container::Dds => ctt::Container::Dds,
            Container::Ktx2 => ctt::Container::Ktx2(None),
            Container::Ktx2Zstd(level) => {
                ctt::Container::Ktx2(Some(ctt::Ktx2Supercompression::Zstd { level }))
            }
            Container::Ktx2Zlib(level) => {
                ctt::Container::Ktx2(Some(ctt::Ktx2Supercompression::Zlib { level }))
            }
            Container::Raw => ctt::Container::Raw,
        }
    }
}

// ---------------------------------------------------------------------------
// ConvertSettings
// ---------------------------------------------------------------------------

/// Settings controlling a [`ctt_convert`] call.
///
/// Build via [`ctt_convert_settings_default`] and overwrite fields, or
/// fill the struct directly. Optional fields have a `present` flag — set
/// it to `false` to keep the default behavior.
///
/// The enum discriminants are chosen so that a fully zero-initialized
/// `ctt_convert_settings` (e.g. via `memset(&cfg, 0, sizeof cfg)`) behaves
/// identically to `ctt_convert_settings_default()`: preserve input format,
/// KTX2 container, `Basic` quality, triangle mipmap filter, and no overrides.
/// Calling `ctt_convert_settings_default()` is still recommended — it is
/// self-documenting and stays correct if defaults ever change.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConvertSettings {
    pub format: TargetFormat,
    pub container: Container,
    pub quality: Quality,
    pub output_color_space: OptionalColorSpace,
    pub output_alpha: OptionalAlphaMode,
    pub swizzle: OptionalSwizzle,
    pub mipmap: bool,
    pub mipmap_count: OptionalSize,
    pub mipmap_filter: MipmapFilter,
}

/// Default-constructed settings: input format preserved, KTX2 container,
/// `Basic` quality, no swizzle / mipmaps / overrides.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_convert_settings_default() -> ConvertSettings {
    ConvertSettings {
        format: TargetFormat::None,
        container: Container::Ktx2,
        quality: Quality::Basic,
        output_color_space: OptionalColorSpace {
            present: false,
            value: ColorSpace::Linear,
        },
        output_alpha: OptionalAlphaMode {
            present: false,
            value: AlphaMode::Straight,
        },
        swizzle: OptionalSwizzle {
            present: false,
            value: crate::types::Swizzle {
                channels: [
                    crate::types::SwizzleChannel::R,
                    crate::types::SwizzleChannel::G,
                    crate::types::SwizzleChannel::B,
                    crate::types::SwizzleChannel::A,
                ],
            },
        },
        mipmap: false,
        mipmap_count: OptionalSize {
            present: false,
            value: 0,
        },
        mipmap_filter: MipmapFilter::Triangle,
    }
}

// ---------------------------------------------------------------------------
// convert
// ---------------------------------------------------------------------------

/// Run the conversion pipeline.
///
/// **Consumes** `image` on both success and failure — the handle must not be
/// destroyed by the caller after this call. On success, writes a freshly
/// allocated `ctt_pipeline_output_t` handle into `*out` (caller frees with
/// `ctt_pipeline_output_destroy`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_convert(
    image: *mut Image,
    settings: *const ConvertSettings,
    out: *mut *mut PipelineOutput,
) -> Status {
    catch_panic(Status::Internal, || {
        if out.is_null() {
            if !image.is_null() {
                drop(unsafe { Box::from_raw(image) });
            }
            set_last_error("ctt_convert: out is null");
            return Status::NullPointer;
        }
        let Some(settings) = (unsafe { settings.as_ref() }) else {
            if !image.is_null() {
                drop(unsafe { Box::from_raw(image) });
            }
            set_last_error("ctt_convert: settings is null");
            return Status::NullPointer;
        };

        let image = match unsafe { take_image(image) } {
            Ok(i) => i,
            Err(s) => return s,
        };

        let format = match settings.format.into_inner() {
            Ok(f) => f,
            Err(s) => return s,
        };

        let swizzle = if settings.swizzle.present {
            Some(settings.swizzle.value.into())
        } else {
            None
        };

        let inner = ctt::ConvertSettings {
            format,
            container: settings.container.into(),
            quality: settings.quality.into_inner(),
            output_color_space: settings
                .output_color_space
                .present
                .then(|| settings.output_color_space.value.into()),
            output_alpha: settings
                .output_alpha
                .present
                .then(|| settings.output_alpha.value.into()),
            swizzle,
            mipmap: settings.mipmap,
            mipmap_count: settings
                .mipmap_count
                .present
                .then_some(settings.mipmap_count.value),
            mipmap_filter: settings.mipmap_filter.into(),
        };

        match ctt::convert(image, inner) {
            Ok(output) => {
                let boxed = Box::into_raw(Box::new(PipelineOutput(Some(output))));
                unsafe {
                    *out = boxed;
                }
                Status::Ok
            }
            Err(e) => map_error(e),
        }
    })
}
