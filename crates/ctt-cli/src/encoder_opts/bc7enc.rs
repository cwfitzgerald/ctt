//! CLI-facing shim for `ctt::encoders::bc7enc::Bc7encSettings`.

use ctt::encoders::bc7enc::Bc7encSettings;
use facet::Facet;

use super::{OptsShim, ParseError, parse_helpers};

/// bc7enc-rdo options exposed via `--bc7e-opts key=val[;key=val...]`.
///
/// Default mirrors `Bc7encSettings::default()` (perceptual=true).
#[derive(Facet, Debug, Clone)]
#[facet(rename_all = "kebab-case")]
pub struct Opts {
    /// Use perceptual quality metrics. Defaults to true.
    pub perceptual: bool,
}

impl Default for Opts {
    fn default() -> Self {
        // Match the library's Default rather than `bool::default()` (false).
        let lib = Bc7encSettings::default();
        Self {
            perceptual: lib.perceptual,
        }
    }
}

impl Opts {
    pub fn into_settings(self) -> Bc7encSettings {
        Bc7encSettings {
            perceptual: self.perceptual,
        }
    }
}

impl OptsShim for Opts {
    fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError> {
        match key {
            "perceptual" => self.perceptual = parse_helpers::bool(key, value)?,
            _ => unreachable!("parser pre-validates keys against Facet's field list"),
        }
        Ok(())
    }
}
