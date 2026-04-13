//! `ctt` — a texture compression and conversion library.
//!
//! # Quick start
//!
//! Most users only need [`convert`] and [`ConvertSettings`]:
//!
//! ```ignore
//! use ctt::{convert, ConvertSettings, Container, TargetFormat, Format, Image, Surface, ColorSpace, AlphaMode};
//!
//! // Build an Image from your pixel data.
//! let surface = Surface {
//!     data: pixel_bytes,
//!     width: 512,
//!     height: 512,
//!     stride: 512 * 4,
//!     format: Format::R8G8B8A8_UNORM,
//!     color_space: ColorSpace::Srgb,
//!     alpha: AlphaMode::Straight,
//! };
//! let image = Image {
//!     surfaces: vec![vec![surface]],
//!     is_cubemap: false,
//! };
//!
//! // Convert to BC7, output as KTX2.
//! let ktx2_bytes = convert(image, ConvertSettings {
//!     format: Some(TargetFormat::Compressed {
//!         encoder_name: None,
//!         format: Format::BC7_UNORM_BLOCK,
//!     }),
//!     container: Container::ktx2(),
//!     ..Default::default()
//! })?;
//! ```
//!
//! Use [`parse_format`] to build a [`TargetFormat`] from a string like `"bc7"`,
//! `"intel_bc7"`, or `"rgba8unorm"`.
//!
//! # Advanced usage
//!
//! For full control over the conversion pipeline — custom transforms, multi-branch
//! assembly, or custom format conversion graphs — use the [`pipeline`] module directly.
//! The [`transforms`] module provides the built-in transform types, and [`encoders`]
//! exposes the encoder registry and individual encoder backends.

// ---- Core types ----

pub use ktx2::Format;

pub use alpha::AlphaMode;
pub use convert::{Container, ConvertSettings, Ktx2Supercompression, convert};
pub use cubemap::{CubemapInput, split_cubemap};
pub use error::{Error, Result};
pub use format::{TargetFormat, format_short_name, parse_format};
pub use pipeline::PipelineOutput;
pub use quality::Quality;
pub use surface::{ColorSpace, Image, Surface};
pub use transforms::mipmap::MipmapFilter;
pub use transforms::swizzle::{Swizzle, SwizzleChannel};
pub use vk_format::{ChannelKind, FormatExt};

// ---- Public modules for advanced use ----
//
// Most users should use `ctt::convert()`. If you need full control over
// the pipeline (custom transforms, multi-branch assembly, custom conversion
// graphs), use these modules directly.

pub mod encoders;
pub mod pipeline;
pub mod transforms;

// ---- Semi-public (needed by custom Transform impls) ----

pub mod constraint;

// ---- Internal ----

mod alpha;
pub(crate) mod conversion;
mod convert;
mod cubemap;
mod error;
mod format;
pub mod input;
pub(crate) mod output;
mod quality;
#[allow(dead_code)]
pub(crate) mod sample;
mod surface;
pub(crate) mod vk_format;
