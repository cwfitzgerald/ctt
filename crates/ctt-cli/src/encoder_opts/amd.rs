//! CLI-facing shim for `ctt::encoders::compressonator::AmdSettings`.
//!
//! Compressonator spans BC1–BC7; the same shim covers all targets, with
//! each field's docstring naming which formats it actually affects.

use ctt::encoders::compressonator::{AmdBc7Alpha, AmdSettings, AmdUsage};
use facet::Facet;

use super::{OptsShim, ParseError, parse_helpers};

/// AMD-compressonator options exposed via `--amd-opts key=val[;key=val...]`.
#[derive(Facet, Debug, Clone, Default)]
#[facet(rename_all = "kebab-case")]
pub struct Opts {
    /// Texture role; drives BC1/BC2/BC3 channel-weight defaults and the
    /// `auto` branch of `bc7-alpha`. One of: color, normal-map, data.
    pub usage: Usage,

    /// Explicit RGB error weights [r, g, b] for BC1/BC2/BC3. Ratios
    /// matter — values are normalized internally. Ignored for other targets.
    pub channel_weights: Option<[f32; 3]>,

    /// BC7 alpha handling. One of: auto (derive from surface alpha mode),
    /// opaque (no alpha modes), full (alpha modes, no restrictions),
    /// restricted (alpha modes with palette restrictions).
    pub bc7_alpha: Bc7Alpha,

    /// BC7 mode mask (bit `n` enables mode `n`, 0..=7). Ignored for other
    /// targets.
    pub bc7_mode_mask: Option<u8>,

    /// BC6H mode mask (14-bit field). Ignored for other targets.
    pub bc6h_mode_mask: Option<u32>,
}

/// Flat mirror of `AmdUsage`.
#[derive(Facet, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
pub enum Usage {
    #[default]
    Color,
    NormalMap,
    Data,
}

/// Flat mirror of `AmdBc7Alpha`.
#[derive(Facet, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
pub enum Bc7Alpha {
    #[default]
    Auto,
    Opaque,
    Full,
    Restricted,
}

impl Opts {
    pub fn into_settings(self) -> AmdSettings {
        AmdSettings {
            usage: match self.usage {
                Usage::Color => AmdUsage::Color,
                Usage::NormalMap => AmdUsage::NormalMap,
                Usage::Data => AmdUsage::Data,
            },
            channel_weights: self.channel_weights,
            bc7_alpha: match self.bc7_alpha {
                Bc7Alpha::Auto => AmdBc7Alpha::Auto,
                Bc7Alpha::Opaque => AmdBc7Alpha::Opaque,
                Bc7Alpha::Full => AmdBc7Alpha::Full,
                Bc7Alpha::Restricted => AmdBc7Alpha::Restricted,
            },
            bc7_mode_mask: self.bc7_mode_mask,
            bc6h_mode_mask: self.bc6h_mode_mask,
        }
    }

    fn parse_usage(value: &str) -> Result<Usage, ParseError> {
        match value {
            "color" => Ok(Usage::Color),
            "normal-map" => Ok(Usage::NormalMap),
            "data" => Ok(Usage::Data),
            other => Err(ParseError::BadValue {
                key: "usage".into(),
                message: format!("unknown usage `{other}`"),
            }),
        }
    }

    fn parse_bc7_alpha(value: &str) -> Result<Bc7Alpha, ParseError> {
        match value {
            "auto" => Ok(Bc7Alpha::Auto),
            "opaque" => Ok(Bc7Alpha::Opaque),
            "full" => Ok(Bc7Alpha::Full),
            "restricted" => Ok(Bc7Alpha::Restricted),
            other => Err(ParseError::BadValue {
                key: "bc7-alpha".into(),
                message: format!("unknown bc7-alpha mode `{other}`"),
            }),
        }
    }
}

impl OptsShim for Opts {
    fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError> {
        match key {
            "usage" => self.usage = Self::parse_usage(value)?,
            "channel-weights" => {
                self.channel_weights = Some(parse_helpers::f32_array::<3>(key, value)?)
            }
            "bc7-alpha" => self.bc7_alpha = Self::parse_bc7_alpha(value)?,
            "bc7-mode-mask" => self.bc7_mode_mask = Some(parse_helpers::u8(key, value)?),
            "bc6h-mode-mask" => self.bc6h_mode_mask = Some(parse_helpers::u32(key, value)?),
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
    fn parses_usage_and_weights() {
        let parsed = parse_opts::<Opts>("usage=normal-map;channel-weights=1,1,1").unwrap();
        let s = parsed.value.into_settings();
        assert_eq!(s.usage, AmdUsage::NormalMap);
        assert_eq!(s.channel_weights, Some([1.0, 1.0, 1.0]));
    }

    #[test]
    fn parses_bc7_alpha() {
        let parsed = parse_opts::<Opts>("bc7-alpha=restricted").unwrap();
        let s = parsed.value.into_settings();
        assert_eq!(s.bc7_alpha, AmdBc7Alpha::Restricted);
    }

    #[test]
    fn defaults_match_lib() {
        let s = Opts::default().into_settings();
        let lib = AmdSettings::default();
        assert_eq!(s.usage, lib.usage);
        assert_eq!(s.bc7_alpha, lib.bc7_alpha);
        assert!(s.channel_weights.is_none());
        assert!(s.bc7_mode_mask.is_none());
        assert!(s.bc6h_mode_mask.is_none());
    }
}
