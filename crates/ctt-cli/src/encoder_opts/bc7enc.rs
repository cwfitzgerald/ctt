//! CLI-facing shim for `ctt::encoders::bc7enc::Bc7encSettings`.

use ctt::encoders::bc7enc::Bc7encSettings;
use facet::Facet;

use super::{OptsShim, ParseError, parse_helpers};

/// bc7enc-rdo options exposed via `--bc7e-opts key=val[;key=val...]`.
///
/// Defaults mirror `Bc7encSettings::default()` (perceptual=true, no
/// preset overrides).
#[derive(Facet, Debug, Clone)]
#[facet(rename_all = "kebab-case")]
pub struct Opts {
    /// Use perceptual error metrics. Defaults to true. Turn off for
    /// normal maps and data textures where channels aren't color.
    pub perceptual: bool,

    /// Restrict to BC7 mode 6 only — fastest mode, lower quality on
    /// blocks with sharp two-color regions.
    pub mode6_only: bool,

    /// Override the preset's parity-bit search choice. Slow presets
    /// already enable it; setting to true forces it on for fast presets.
    pub pbit_search: Option<bool>,

    /// Override the preset's "uber" refinement level (0..=4). Higher =
    /// more re-encode passes on bad-fit blocks. Clamped on the codec side.
    pub uber_level: Option<u32>,

    /// Custom per-channel error weights [r, g, b, a]. Integer values;
    /// perceptual default is [128, 64, 16, 256], uniform is [1, 1, 1, 1].
    pub channel_weights: Option<[u32; 4]>,
}

impl Default for Opts {
    fn default() -> Self {
        // Match the library's Default rather than `bool::default()` (false).
        let lib = Bc7encSettings::default();
        Self {
            perceptual: lib.perceptual,
            mode6_only: lib.mode6_only,
            pbit_search: lib.pbit_search,
            uber_level: lib.uber_level,
            channel_weights: lib.channel_weights,
        }
    }
}

impl Opts {
    pub fn into_settings(self) -> Bc7encSettings {
        Bc7encSettings {
            perceptual: self.perceptual,
            mode6_only: self.mode6_only,
            pbit_search: self.pbit_search,
            uber_level: self.uber_level,
            channel_weights: self.channel_weights,
        }
    }
}

impl OptsShim for Opts {
    fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError> {
        match key {
            "perceptual" => self.perceptual = parse_helpers::bool(key, value)?,
            "mode6-only" => self.mode6_only = parse_helpers::bool(key, value)?,
            "pbit-search" => self.pbit_search = Some(parse_helpers::bool(key, value)?),
            "uber-level" => self.uber_level = Some(parse_helpers::u32(key, value)?),
            "channel-weights" => {
                self.channel_weights = Some(parse_helpers::u32_array::<4>(key, value)?)
            }
            _ => unreachable!("parser pre-validates keys against Facet's field list"),
        }
        Ok(())
    }
}
