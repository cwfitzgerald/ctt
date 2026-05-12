//! CLI-facing shim for `ctt::encoders::ispc::IspcSettings`.

use ctt::encoders::ispc::IspcSettings;
use facet::Facet;

use super::{OptsShim, ParseError, parse_helpers};

/// intel/ispc-encoder options exposed via `--intel-opts key=val[;key=val...]`.
#[derive(Facet, Debug, Clone, Default)]
#[facet(rename_all = "kebab-case")]
pub struct Opts {
    /// Encode the alpha channel (BC7 only).
    pub alpha: bool,
}

impl Opts {
    pub fn into_settings(self) -> IspcSettings {
        IspcSettings { alpha: self.alpha }
    }
}

impl OptsShim for Opts {
    fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError> {
        match key {
            "alpha" => self.alpha = parse_helpers::bool(key, value)?,
            _ => unreachable!("parser pre-validates keys against Facet's field list"),
        }
        Ok(())
    }
}
