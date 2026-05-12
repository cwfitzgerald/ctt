//! CLI-facing shim for `ctt::encoders::astcenc::AstcencSettings`.
//!
//! The library type uses a structured `AstcencUsage::NormalMap { swizzle }`
//! variant that doesn't map cleanly to `key=val` syntax. This shim flattens
//! `usage` and `normal_swizzle` into separate CLI keys and reconstructs the
//! structured form in [`Opts::into_settings`]. The struct-literal construction
//! there is the drift tripwire: adding a field to the library type forces a
//! compile error in `into_settings`.

use ctt::encoders::astcenc::{AstcencSettings, AstcencUsage, astc};
use facet::Facet;

use super::{OptsShim, ParseError, parse_helpers};

/// astcenc options exposed via `--astcenc-opts key=val[;key=val...]`.
#[derive(Facet, Debug, Clone, Default)]
#[facet(rename_all = "kebab-case")]
pub struct Opts {
    /// Texture role; drives profile, flags, and swizzle.
    /// One of: color, normal-map, single-channel, two-channel,
    /// hdr-rgb, hdr-rgba, rgbm.
    pub usage: Usage,

    /// Normal-map input layout. Only meaningful when usage=normal-map.
    /// astc-default = rrrg (X in RGB, Y in A), bc5-compat = gggr.
    pub normal_swizzle: NormalSwizzle,

    /// Encoder effort on a 0.0..=100.0 continuum (matches astcenc's `-q`).
    /// Overrides the top-level --quality mapping when set. Reference
    /// points: 0=fastest, 10=fast, 60=medium, 98=thorough,
    /// 99=very-thorough, 100=exhaustive.
    pub quality: Option<f32>,

    /// Weight RGB error by alpha. Improves alpha precision in
    /// transparent regions at the cost of RGB fidelity there.
    pub use_alpha_weight: bool,

    /// Optimize for perceptual error rather than PSNR. Only meaningful
    /// for color and normal-map usages.
    pub perceptual: bool,

    /// Tune for the decode_unorm8 ASTC decode mode. Set when the
    /// texture will be sampled as unorm8 at runtime (common on mobile
    /// and most desktop pipelines).
    pub decode_unorm8: bool,

    /// Per-channel error weights [r, g, b, a]. Higher = more bits
    /// spent on that channel.
    pub channel_weights: Option<[f32; 4]>,

    /// RGBM shared-multiplier scale. Only meaningful when usage=rgbm.
    pub rgbm_m_scale: Option<f32>,
}

/// Flat mirror of `AstcencUsage` (the `NormalMap` variant is stripped of its
/// payload here; the swizzle is carried on [`Opts::normal_swizzle`] instead).
#[derive(Facet, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
pub enum Usage {
    #[default]
    Color,
    NormalMap,
    SingleChannel,
    TwoChannel,
    HdrRgb,
    HdrRgba,
    Rgbm,
}

/// Flat mirror of `ctt::encoders::astcenc::NormalSwizzle`. Re-defined here so
/// the library crate doesn't need a facet dep.
#[derive(Facet, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
pub enum NormalSwizzle {
    #[default]
    AstcDefault,
    Bc5Compat,
}

impl Opts {
    /// Reconstruct the library settings struct.
    ///
    /// Uses a full struct literal — if `AstcencSettings` gains a field, this
    /// stops compiling until both the field and a corresponding CLI key
    /// (or an explicit choice not to expose it) are added.
    pub fn into_settings(self) -> AstcencSettings {
        AstcencSettings {
            usage: match self.usage {
                Usage::Color => AstcencUsage::Color,
                Usage::NormalMap => AstcencUsage::NormalMap {
                    swizzle: match self.normal_swizzle {
                        NormalSwizzle::AstcDefault => {
                            ctt::encoders::astcenc::NormalSwizzle::AstcDefault
                        }
                        NormalSwizzle::Bc5Compat => {
                            ctt::encoders::astcenc::NormalSwizzle::Bc5Compat
                        }
                    },
                },
                Usage::SingleChannel => AstcencUsage::SingleChannel,
                Usage::TwoChannel => AstcencUsage::TwoChannel,
                Usage::HdrRgb => AstcencUsage::HdrRgb,
                Usage::HdrRgba => AstcencUsage::HdrRgba,
                Usage::Rgbm => AstcencUsage::Rgbm,
            },
            preset: self.quality.map(astc::Preset::Custom),
            use_alpha_weight: self.use_alpha_weight,
            perceptual: self.perceptual,
            decode_unorm8: self.decode_unorm8,
            channel_weights: self.channel_weights,
            rgbm_m_scale: self.rgbm_m_scale,
        }
    }

    fn parse_usage(value: &str) -> Result<Usage, ParseError> {
        match value {
            "color" => Ok(Usage::Color),
            "normal-map" => Ok(Usage::NormalMap),
            "single-channel" => Ok(Usage::SingleChannel),
            "two-channel" => Ok(Usage::TwoChannel),
            "hdr-rgb" => Ok(Usage::HdrRgb),
            "hdr-rgba" => Ok(Usage::HdrRgba),
            "rgbm" => Ok(Usage::Rgbm),
            other => Err(ParseError::BadValue {
                key: "usage".into(),
                message: format!("unknown usage `{other}`"),
            }),
        }
    }

    fn parse_normal_swizzle(value: &str) -> Result<NormalSwizzle, ParseError> {
        match value {
            "astc-default" => Ok(NormalSwizzle::AstcDefault),
            "bc5-compat" => Ok(NormalSwizzle::Bc5Compat),
            other => Err(ParseError::BadValue {
                key: "normal-swizzle".into(),
                message: format!("unknown swizzle `{other}`"),
            }),
        }
    }

    /// Surface "set but ignored" combinations. These are informational —
    /// the encoder will silently drop them, so the user wouldn't otherwise
    /// notice the mistake.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.normal_swizzle != NormalSwizzle::default() && self.usage != Usage::NormalMap {
            w.push("normal-swizzle set but usage != normal-map (ignored)".into());
        }
        if self.rgbm_m_scale.is_some() && self.usage != Usage::Rgbm {
            w.push("rgbm-m-scale set but usage != rgbm (ignored)".into());
        }
        w
    }
}

impl OptsShim for Opts {
    fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError> {
        match key {
            "usage" => self.usage = Self::parse_usage(value)?,
            "normal-swizzle" => self.normal_swizzle = Self::parse_normal_swizzle(value)?,
            "quality" => self.quality = Some(parse_helpers::f32(key, value)?),
            "use-alpha-weight" => self.use_alpha_weight = parse_helpers::bool(key, value)?,
            "perceptual" => self.perceptual = parse_helpers::bool(key, value)?,
            "decode-unorm8" => self.decode_unorm8 = parse_helpers::bool(key, value)?,
            "channel-weights" => {
                self.channel_weights = Some(parse_helpers::f32_array::<4>(key, value)?)
            }
            "rgbm-m-scale" => self.rgbm_m_scale = Some(parse_helpers::f32(key, value)?),
            _ => unreachable!("parser pre-validates keys against Facet's field list"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder_opts::parse_opts;

    #[test]
    fn into_settings_round_trip_defaults() {
        let opts = Opts::default();
        let s = opts.into_settings();
        // Default usage is Color, so AstcencSettings's default applies cleanly.
        assert_eq!(s.usage, AstcencUsage::Color);
        assert!(s.preset.is_none());
        // (library field `preset` corresponds to CLI `quality`)
        assert!(!s.use_alpha_weight);
    }

    #[test]
    fn parses_usage_normal_map_with_swizzle() {
        let parsed = parse_opts::<Opts>("usage=normal-map;normal-swizzle=bc5-compat").unwrap();
        let s = parsed.value.into_settings();
        match s.usage {
            AstcencUsage::NormalMap { swizzle } => {
                assert_eq!(swizzle, ctt::encoders::astcenc::NormalSwizzle::Bc5Compat);
            }
            other => panic!("expected NormalMap, got {other:?}"),
        }
    }

    #[test]
    fn warns_on_orphan_normal_swizzle() {
        let parsed = parse_opts::<Opts>("normal-swizzle=bc5-compat").unwrap();
        let w = parsed.value.warnings();
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("normal-swizzle"));
    }

    #[test]
    fn warns_on_orphan_rgbm_m_scale() {
        let parsed = parse_opts::<Opts>("rgbm-m-scale=8.0").unwrap();
        let w = parsed.value.warnings();
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("rgbm-m-scale"));
    }

    #[test]
    fn channel_weights_array_parses() {
        let parsed = parse_opts::<Opts>("channel-weights=1,1,1,2").unwrap();
        assert_eq!(parsed.value.channel_weights, Some([1.0, 1.0, 1.0, 2.0]));
    }
}
