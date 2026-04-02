use crate::encoder::{EncoderSettings, Quality};
use crate::format::{ColorSpace, CompressedFormat};
use crate::transform::swizzle::Swizzle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    Dds,
    Ktx2,
}

/// Full compression configuration.
pub struct CompressConfig {
    pub format: CompressedFormat,
    pub output_format: OutputFormat,
    pub swizzle: Option<Swizzle>,
    pub color_space: ColorSpace,
    pub quality: Quality,
    /// Optional encoder name to force a specific encoder (e.g., "intel", "bc7e").
    pub encoder_name: Option<String>,
    /// Optional encoder-specific settings (downcast by the encoder impl).
    pub encoder_settings: Option<Box<dyn EncoderSettings>>,
}
