//! `ctt` — a texture compression and conversion library.
//!
//! # Quick start
//!
//! ```ignore
//! use ctt::{convert, ConvertSettings, Container, TargetFormat, Format, Image, Surface, ColorSpace, AlphaMode};
//!
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

// ---- Core types ----

pub use ktx2::Format;

pub use alpha::AlphaMode;
pub use convert::{Container, ConvertSettings, Ktx2Supercompression, convert};
pub use cubemap::{CubemapInput, split_cubemap};
pub use error::{Error, Result};
pub use format::{TargetFormat, format_short_name, parse_format};
pub use processing::{MipmapFilter, PipelineOutput, Swizzle, SwizzleChannel};
pub use quality::Quality;
pub use surface::{ColorSpace, Image, Surface};
pub use vk_format::{ChannelKind, FormatExt};

// ---- Public modules for advanced use ----

pub mod encoders;

// ---- Internal ----

mod alpha;
mod convert;
mod cubemap;
mod error;
mod format;
mod format_kind;
pub mod input;
pub(crate) mod output;
mod processing;
mod quality;
mod surface;
pub(crate) mod vk_format;
