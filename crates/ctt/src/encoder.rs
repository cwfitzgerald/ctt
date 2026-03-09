use std::any::Any;

use crate::error::Result;
use crate::format::{ColorSpace, CompressedFormat, PixelFormat};
use crate::image::RawImage;

/// Universal quality preset all encoders understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Quality {
    UltraFast,
    VeryFast,
    Fast,
    #[default]
    Basic,
    Slow,
    VerySlow,
}

/// Marker trait for encoder-specific settings. Acts as a bounded `dyn Any`:
/// only types explicitly marked as encoder settings can be passed through.
pub trait EncoderSettings: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

/// An encoder backend that can compress raw images.
pub trait Encoder: Send + Sync {
    /// Short name used as prefix in format strings (e.g., "ispc", "bc7e").
    fn name(&self) -> &str;

    /// Formats this encoder supports.
    fn supported_formats(&self) -> &[CompressedFormat];

    /// What pixel format this encoder needs for a given compressed format.
    fn required_input_format(&self, format: CompressedFormat, color_space: ColorSpace)
        -> PixelFormat;

    /// Compress a single image.
    ///
    /// `quality` is the universal quality level.
    /// `settings` is an optional encoder-specific settings object (downcast by impl).
    fn compress(
        &self,
        image: &RawImage,
        format: CompressedFormat,
        quality: Quality,
        settings: Option<&dyn EncoderSettings>,
    ) -> Result<Vec<u8>>;
}

/// Registry of available encoder backends.
pub struct EncoderRegistry {
    encoders: Vec<Box<dyn Encoder>>,
}

impl EncoderRegistry {
    pub fn new() -> Self {
        Self {
            encoders: Vec::new(),
        }
    }

    /// Build default registry from compile-time-enabled encoders.
    /// Registration order = priority order (first registered wins).
    pub fn default_registry() -> Self {
        #[allow(unused_mut)]
        let mut r = Self::new();
        #[cfg(feature = "encoder-bc7enc")]
        r.register(Box::new(crate::encoders::bc7enc::Bc7encEncoder::new()));
        #[cfg(feature = "encoder-ispc")]
        r.register(Box::new(crate::encoders::ispc::IspcEncoder));
        r
    }

    pub fn register(&mut self, encoder: Box<dyn Encoder>) {
        self.encoders.push(encoder);
    }

    /// Find best encoder for a format (first registered that supports it).
    pub fn find(&self, format: CompressedFormat) -> Option<&dyn Encoder> {
        self.encoders
            .iter()
            .find(|e| e.supported_formats().contains(&format))
            .map(|e| e.as_ref())
    }

    /// Find encoder by name + format (for "ispc_bc7", "bc7e_bc7" style).
    pub fn find_by_name(&self, name: &str, format: CompressedFormat) -> Option<&dyn Encoder> {
        self.encoders
            .iter()
            .find(|e| e.name() == name && e.supported_formats().contains(&format))
            .map(|e| e.as_ref())
    }

    /// Access the registered encoders.
    pub fn encoders(&self) -> &[Box<dyn Encoder>] {
        &self.encoders
    }

    /// List all available format strings (e.g., ["bc7e_bc7", "ispc_bc1", ...]).
    pub fn available_formats(&self) -> Vec<String> {
        let mut formats = Vec::new();
        for encoder in &self.encoders {
            for &fmt in encoder.supported_formats() {
                formats.push(format!("{}_{:?}", encoder.name(), fmt).to_lowercase());
            }
        }
        formats
    }
}

impl Default for EncoderRegistry {
    fn default() -> Self {
        Self::new()
    }
}
