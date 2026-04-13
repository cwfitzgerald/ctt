//! High-level "CLI-like" conversion API.
//!
//! Most users don't need to build a [`Pipeline`](crate::pipeline::Pipeline) manually.
//! Instead, pass an [`Image`] and [`ConvertSettings`] to [`convert`] and get encoded
//! bytes back.
//!
//! For advanced use cases (custom transforms, multi-branch assembly, custom conversion
//! graphs), use the [`pipeline`](crate::pipeline) module directly.

use std::sync::Arc;

use crate::alpha::AlphaMode;
use crate::encoders::{EncoderRegistry, EncoderSettings, Quality};
use crate::error::{Error, Result};
use crate::format::TargetFormat;
use crate::pipeline::{AssemblyNode, InputBranch, InputNode, Pipeline, PipelineOutput};
use crate::surface::{ColorSpace, Image};
use crate::transforms::Transform;
use crate::transforms::compress::CompressTransform;
use crate::transforms::mipmap::{MipmapFilter, MipmapTransform};
use crate::transforms::output_state::OutputStateTransform;
use crate::transforms::swizzle::{Swizzle, SwizzleTransform};

/// Output container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Container {
    Dds,
    Ktx2(Option<Ktx2Supercompression>),
    /// Return the processed [`Image`] directly, without encoding into a file format.
    Raw,
}

/// Supercompression to apply when writing KTX2 files.
///
/// Each mip level is compressed independently per the KTX2 spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ktx2Supercompression {
    /// Zstandard compression. `level` is passed directly to the `zstd` crate.
    /// Valid range: negative values (fast mode) through 22 (maximum compression).
    /// Level 0 is a special sentinel that maps to the library default (currently 3).
    Zstd { level: i32 },
    /// ZLIB compression (deflate with zlib framing).
    /// Valid range: 1 (fastest) through 10 (maximum compression). 0 means no
    /// compression (stored). The conventional default is 6.
    Zlib { level: u8 },
}

impl Container {
    /// KTX2 without supercompression.
    pub fn ktx2() -> Self {
        Container::Ktx2(None)
    }

    /// KTX2 with zstd supercompression at the given level.
    ///
    /// Valid range: negative values (fast mode) through 22 (maximum compression).
    /// Pass 0 to use the zstd library's default compression level (currently 3).
    pub fn ktx2_zstd(level: i32) -> Self {
        Container::Ktx2(Some(Ktx2Supercompression::Zstd { level }))
    }

    /// KTX2 with zlib supercompression at the given level.
    ///
    /// Valid range: 1 (fastest) through 10 (maximum compression). 0 means no
    /// compression (stored). The conventional default is 6.
    pub fn ktx2_zlib(level: u8) -> Self {
        Container::Ktx2(Some(Ktx2Supercompression::Zlib { level }))
    }
}

/// Settings for the high-level [`convert`] function.
///
/// Use [`Default::default()`] and override only the fields you care about:
///
/// ```ignore
/// use ctt::{Format, Container, ConvertSettings, TargetFormat, convert};
///
/// let bytes = convert(image, ConvertSettings {
///     format: Some(TargetFormat::Compressed {
///         encoder_name: None,
///         format: Format::BC7_UNORM_BLOCK,
///     }),
///     container: Container::ktx2(),
///     ..Default::default()
/// })?;
/// ```
///
/// To parse a format from a string (e.g. `"bc7"`, `"intel_bc7"`, `"rgba8unorm"`),
/// use [`parse_format`](crate::parse_format).
pub struct ConvertSettings {
    /// Target format. If `None`, the input format is preserved without compression.
    ///
    /// Use [`parse_format`](crate::parse_format) to build this from a string,
    /// or construct a [`TargetFormat`] directly.
    pub format: Option<TargetFormat>,

    /// Output container format.
    pub container: Container,

    /// Compression quality preset.
    pub quality: Quality,

    /// Desired output color space. If `None`, matches the input.
    pub output_color_space: Option<ColorSpace>,

    /// Desired output alpha mode. If `None`, matches the input.
    pub output_alpha: Option<AlphaMode>,

    /// Channel swizzle pattern to apply before compression.
    pub swizzle: Option<Swizzle>,

    /// Generate mipmaps.
    pub mipmap: bool,

    /// Number of mip levels (including base). `None` = full chain down to 1x1.
    pub mipmap_count: Option<usize>,

    /// Mipmap downsampling filter.
    pub mipmap_filter: MipmapFilter,

    /// Allow lossy auto-inserted format conversions in the pipeline.
    pub allow_lossy: bool,

    /// Encoder-specific settings (e.g., [`IspcSettings`](crate::encoders::ispc::IspcSettings)).
    pub encoder_settings: Option<Box<dyn EncoderSettings>>,

    /// Encoder registry. If `None`, uses [`EncoderRegistry::default_registry`].
    pub registry: Option<Arc<EncoderRegistry>>,
}

impl Default for ConvertSettings {
    fn default() -> Self {
        Self {
            format: None,
            container: Container::Ktx2(None),
            quality: Quality::default(),
            output_color_space: None,
            output_alpha: None,
            swizzle: None,
            mipmap: false,
            mipmap_count: None,
            mipmap_filter: MipmapFilter::default(),
            allow_lossy: false,
            encoder_settings: None,
            registry: None,
        }
    }
}

/// Convert an image using a simple, CLI-like interface.
///
/// Builds and executes a [`Pipeline`](crate::pipeline::Pipeline) internally based on
/// the given settings. Returns [`PipelineOutput::Encoded`] for DDS/KTX2 containers,
/// or [`PipelineOutput::Raw`] when using [`Container::Raw`].
///
/// The input [`Image`] should already be fully assembled (including cubemap layers
/// or array layers). Use [`split_cubemap`](crate::split_cubemap) to prepare
/// cubemap inputs before calling this function.
pub fn convert(image: Image, settings: ConvertSettings) -> Result<PipelineOutput> {
    let registry = settings
        .registry
        .unwrap_or_else(|| Arc::new(EncoderRegistry::default_registry()));

    // Build transforms.
    let mut transforms: Vec<Box<dyn Transform>> = Vec::new();

    if let Some(ref swizzle) = settings.swizzle {
        transforms.push(Box::new(SwizzleTransform::new(*swizzle)));
    }

    if settings.mipmap {
        transforms.push(Box::new(MipmapTransform::new(
            settings.mipmap_count,
            settings.mipmap_filter,
        )));
    }

    match settings.format {
        Some(TargetFormat::Compressed {
            encoder_name,
            format: target_format,
        }) => {
            if settings.output_color_space.is_some() || settings.output_alpha.is_some() {
                transforms.push(Box::new(OutputStateTransform::new(
                    None,
                    settings.output_color_space,
                    settings.output_alpha,
                )));
            }
            transforms.push(Box::new(CompressTransform::new(
                target_format,
                settings.quality,
                encoder_name,
                settings.encoder_settings,
                registry,
            )));
        }
        Some(TargetFormat::Uncompressed(target_format)) => {
            transforms.push(Box::new(OutputStateTransform::new(
                Some(target_format),
                settings.output_color_space,
                settings.output_alpha,
            )));
        }
        None => {
            if settings.output_color_space.is_some() || settings.output_alpha.is_some() {
                transforms.push(Box::new(OutputStateTransform::new(
                    None,
                    settings.output_color_space,
                    settings.output_alpha,
                )));
            }
        }
    }

    let pipeline = Pipeline {
        inputs: vec![InputBranch {
            input: InputNode::Raw(image),
            transforms: Vec::new(),
        }],
        assembly: AssemblyNode::Identity,
        transforms,
        container: settings.container,
        allow_lossy_intermediates: settings.allow_lossy,
    };

    let resolved = pipeline.resolve().map_err(|errors| {
        let messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        Error::UnsupportedFormat(messages.join("; "))
    })?;

    resolved.execute()
}
