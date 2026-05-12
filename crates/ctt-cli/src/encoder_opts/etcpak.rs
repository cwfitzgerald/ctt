//! CLI-facing shim for `ctt::encoders::etcpak::EtcpakSettings`.

use ctt::encoders::etcpak::EtcpakSettings;
use facet::Facet;

use super::{OptsShim, ParseError, parse_helpers};

/// etcpak-encoder options exposed via `--etcpak-opts key=val[;key=val...]`.
#[derive(Facet, Debug, Clone, Default)]
#[facet(rename_all = "kebab-case")]
pub struct Opts {
    /// Enable dithering for ETC1 and BC1 compression.
    pub dither: bool,
    /// Enable heuristic-based fast mode selection for ETC2 RGB/RGBA.
    pub use_heuristics: bool,
}

impl Opts {
    pub fn into_settings(self) -> EtcpakSettings {
        EtcpakSettings {
            dither: self.dither,
            use_heuristics: self.use_heuristics,
        }
    }
}

impl OptsShim for Opts {
    fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError> {
        match key {
            "dither" => self.dither = parse_helpers::bool(key, value)?,
            "use-heuristics" => self.use_heuristics = parse_helpers::bool(key, value)?,
            _ => unreachable!("parser pre-validates keys against Facet's field list"),
        }
        Ok(())
    }
}
