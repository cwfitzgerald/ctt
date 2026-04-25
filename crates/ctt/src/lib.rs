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

/// **Not part of the public API.** Re-exports the internal sRGB kernels so
/// that `benches/` (a separate crate) can measure them directly without
/// going through the public dispatch layer. Items here have no stability
/// guarantees — they may be renamed, removed, or behave differently across
/// patch releases. Real callers should go through the crate's normal
/// public entry points (`convert`, etc.), which dispatch to the best
/// available kernel at runtime.
#[doc(hidden)]
pub mod bench_internals {
    pub use crate::processing::Buffer;

    pub use crate::processing::load_kernels::srgb::load_srgb8_f32_serial;
    pub use crate::processing::load_kernels::{
        load_bgr8_srgb_f32, load_bgra8_srgb_f32, load_srgb8_f32,
    };
    pub use crate::processing::store_kernels::srgb::{
        store_bgra8_srgb_f32_serial, store_srgb8_f32_serial,
    };
    pub use crate::processing::store_kernels::{
        store_bgr8_srgb_f32, store_bgra8_srgb_f32, store_srgb8_f32,
    };

    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::load_kernels::srgb::{
        load_srgb8_rgba_f32_avx2_fma, load_srgb8_rgba_f32_avx512, load_srgb8_rgba_f32_sse4_1,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::store_kernels::srgb::{
        store_srgb8_f32_avx2_fma, store_srgb8_f32_avx512, store_srgb8_f32_sse4_1,
    };

    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::load_kernels::srgb::load_srgb8_rgba_f32_neon;
    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::store_kernels::srgb::store_srgb8_f32_neon;
}
