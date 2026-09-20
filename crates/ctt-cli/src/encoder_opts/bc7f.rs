//! CLI-facing shim for `ctt::encoders::bc7f::Bc7fSettings`.

use ctt::encoders::bc7f::Bc7fSettings;
use facet::Facet;

use super::{OptsShim, ParseError, parse_helpers};

/// bc7f-encoder options exposed via `--bc7f-opts key=val[;key=val...]`.
#[derive(Facet, Debug, Clone, Default)]
#[facet(rename_all = "kebab-case")]
pub struct Opts {
    /// Enable astc_compatibleing for ETC1 and BC1 compression.
    pub astc_compatible: bool,
    /// Disable separate RGB planes. A separate alpha plane remains available.
    pub disable_rgb_dual_plane: bool,
}

impl Opts {
    pub fn into_settings(self) -> Bc7fSettings {
        Bc7fSettings {
            astc_compatible: self.astc_compatible,
            disable_rgb_dual_plane: self.disable_rgb_dual_plane,
        }
    }
}

impl OptsShim for Opts {
    fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError> {
        match key {
            "astc-compatible" => self.astc_compatible = parse_helpers::bool(key, value)?,
            "disable-rgb-dual-plane" => {
                self.disable_rgb_dual_plane = parse_helpers::bool(key, value)?
            }
            _ => unreachable!("parser pre-validates keys against Facet's field list"),
        }
        Ok(())
    }
}
