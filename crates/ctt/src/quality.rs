/// Universal quality preset all encoders understand.
///
/// Presets trade encode speed for output quality, from fastest/lowest to
/// slowest/highest. Each encoder maps these onto its own native effort levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Quality {
    /// Fastest encode, lowest quality.
    UltraFast,
    /// Very fast encode, low quality.
    VeryFast,
    /// Fast encode, moderate quality.
    Fast,
    /// Balanced speed/quality — the default.
    #[default]
    Basic,
    /// Slow encode, high quality.
    Slow,
    /// Slowest encode, highest quality.
    VerySlow,
}
