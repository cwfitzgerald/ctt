//! CLI-facing shim for `ctt::encoders::ispc::IspcSettings`.

use ctt::encoders::ispc::{IspcBc7Alpha, IspcSettings};
use facet::Facet;

use super::{OptsShim, ParseError};

/// intel/ispc-encoder options exposed via `--intel-opts key=val[;key=val...]`.
#[derive(Facet, Debug, Clone, Default)]
#[facet(rename_all = "kebab-case")]
pub struct Opts {
    /// BC7 alpha handling. One of: auto (derive from surface alpha mode),
    /// opaque (force opaque presets — RGB-only modes), alpha (force
    /// alpha-aware presets — modes 4–7). Ignored for non-BC7 targets.
    pub bc7_alpha: Bc7Alpha,
}

/// Flat mirror of `IspcBc7Alpha`.
#[derive(Facet, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
pub enum Bc7Alpha {
    #[default]
    Auto,
    Opaque,
    Alpha,
}

impl Opts {
    pub fn into_settings(self) -> IspcSettings {
        IspcSettings {
            bc7_alpha: match self.bc7_alpha {
                Bc7Alpha::Auto => IspcBc7Alpha::Auto,
                Bc7Alpha::Opaque => IspcBc7Alpha::Opaque,
                Bc7Alpha::Alpha => IspcBc7Alpha::Alpha,
            },
        }
    }

    fn parse_bc7_alpha(value: &str) -> Result<Bc7Alpha, ParseError> {
        match value {
            "auto" => Ok(Bc7Alpha::Auto),
            "opaque" => Ok(Bc7Alpha::Opaque),
            "alpha" => Ok(Bc7Alpha::Alpha),
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
            "bc7-alpha" => self.bc7_alpha = Self::parse_bc7_alpha(value)?,
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
    fn parses_bc7_alpha() {
        let parsed = parse_opts::<Opts>("bc7-alpha=alpha").unwrap();
        let s = parsed.value.into_settings();
        assert_eq!(s.bc7_alpha, IspcBc7Alpha::Alpha);
    }

    #[test]
    fn defaults_match_lib() {
        let s = Opts::default().into_settings();
        let lib = IspcSettings::default();
        assert_eq!(s.bc7_alpha, lib.bc7_alpha);
    }
}
