use std::any::Any;

use crate::error::Result;
pub(crate) use crate::quality::Quality;
use crate::surface::Surface;

/// Marker trait for encoder-specific settings. Acts as a bounded `dyn Any`:
/// only types explicitly marked as encoder settings can be passed through.
pub trait EncoderSettings: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

/// An encoder backend that can compress pixel data into block-compressed formats.
pub trait Encoder: Send + Sync {
    /// Short name used as prefix in format strings (e.g., "intel", "bc7e").
    fn name(&self) -> &str;

    /// Compressed formats this encoder supports (normalized, i.e. UNORM/UFLOAT variants).
    fn supported_formats(&self) -> &[ktx2::Format];

    /// What uncompressed format this encoder needs as input for a given compressed format.
    ///
    /// `format` is the normalized target format (e.g. `BC7_UNORM_BLOCK`).
    fn required_input_format(&self, format: ktx2::Format) -> ktx2::Format;

    /// Compress a single image.
    ///
    /// `surface` is the uncompressed input surface.
    /// `format` may be the sRGB variant to indicate color space (call `.normalize()` to recover).
    /// `quality` is the universal quality level.
    /// `settings` is an optional encoder-specific settings object (downcast by impl).
    fn compress(
        &self,
        surface: &Surface,
        format: ktx2::Format,
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
        r.register(Box::new(self::bc7enc::Bc7encEncoder));
        #[cfg(feature = "encoder-intel")]
        r.register(Box::new(self::ispc::IspcEncoder));
        #[cfg(feature = "encoder-amd")]
        r.register(Box::new(self::compressonator::CompressonatorEncoder));
        #[cfg(feature = "encoder-astcenc")]
        r.register(Box::new(self::astcenc::AstcencEncoder));
        r
    }

    pub fn register(&mut self, encoder: Box<dyn Encoder>) {
        self.encoders.push(encoder);
    }

    /// Find best encoder for a format (first registered that supports it).
    pub fn find(&self, format: ktx2::Format) -> Option<&dyn Encoder> {
        self.encoders
            .iter()
            .find(|e| e.supported_formats().contains(&format))
            .map(|e| e.as_ref())
    }

    /// Find encoder by name + format (for "intel_bc7", "bc7e_bc7" style).
    pub fn find_by_name(&self, name: &str, format: ktx2::Format) -> Option<&dyn Encoder> {
        self.encoders
            .iter()
            .find(|e| e.name() == name && e.supported_formats().contains(&format))
            .map(|e| e.as_ref())
    }

    /// Access the registered encoders.
    pub fn encoders(&self) -> &[Box<dyn Encoder>] {
        &self.encoders
    }

    /// List all available format strings (e.g., ["bc7e_bc7", "intel_bc1", ...]).
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

#[cfg(feature = "encoder-intel")]
pub mod ispc;

#[cfg(feature = "encoder-bc7enc")]
pub mod bc7enc;

#[cfg(feature = "encoder-astcenc")]
pub mod astcenc;

#[cfg(feature = "encoder-amd")]
pub mod compressonator;
