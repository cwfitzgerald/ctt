use crate::format::{ColorSpace, CompressedFormat};
use crate::transform::swizzle::Swizzle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    Dds,
    Ktx2,
}

/// Quality preset for BC6H compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Bc6hQuality {
    VeryFast,
    Fast,
    #[default]
    Basic,
    Slow,
    VerySlow,
}

/// Quality preset for BC7 compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Bc7Quality {
    UltraFast,
    VeryFast,
    Fast,
    #[default]
    Basic,
    Slow,
}

/// BC7 encode settings: quality and whether to encode alpha.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bc7Settings {
    pub quality: Bc7Quality,
    pub alpha: bool,
}

/// Quality preset for ETC1 compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Etc1Quality {
    #[default]
    Slow,
}

/// Per-format encode settings.
///
/// When `None` is used in [`CompressConfig`], sensible defaults are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodeSettings {
    Bc1,
    Bc3,
    Bc4,
    Bc5,
    Bc6h(Bc6hQuality),
    Bc7(Bc7Settings),
    Etc1(Etc1Quality),
    Astc,
}

impl EncodeSettings {
    /// Create default settings for the given compressed format.
    pub fn default_for(format: CompressedFormat) -> Self {
        match format {
            CompressedFormat::Bc1 => Self::Bc1,
            CompressedFormat::Bc3 => Self::Bc3,
            CompressedFormat::Bc4 => Self::Bc4,
            CompressedFormat::Bc5 => Self::Bc5,
            CompressedFormat::Bc6h => Self::Bc6h(Bc6hQuality::default()),
            CompressedFormat::Bc7 => Self::Bc7(Bc7Settings::default()),
            CompressedFormat::Etc1 => Self::Etc1(Etc1Quality::default()),
            CompressedFormat::Astc { .. } => Self::Astc,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressConfig {
    pub format: CompressedFormat,
    pub output_format: OutputFormat,
    pub swizzle: Option<Swizzle>,
    pub color_space: ColorSpace,
    pub encode_settings: Option<EncodeSettings>,
}
