//! `ctt` — a texture compression and conversion library.
//!
//! # Quick start
//!
//! ```no_run
//! use ctt::{convert, ConvertSettings, Container, TargetFormat, Format, Image, Surface, ColorSpace, AlphaMode, TextureKind};
//! use ctt::encoders::Encoder;
//!
//! # fn main() -> Result<(), ctt::Error> {
//! let pixel_bytes = vec![0u8; 512 * 512 * 4];
//! let surface = Surface {
//!     data: pixel_bytes,
//!     width: 512,
//!     height: 512,
//!     depth: 1,
//!     stride: 512 * 4,
//!     slice_stride: 0,
//!     format: Format::R8G8B8A8_UNORM,
//!     color_space: ColorSpace::Srgb,
//!     alpha: AlphaMode::Straight,
//! };
//! let image = Image {
//!     surfaces: vec![vec![surface]],
//!     kind: TextureKind::Texture2D,
//! };
//!
//! let _ktx2_bytes = convert(image, ConvertSettings {
//!     format: Some(TargetFormat::Compressed {
//!         format: Format::BC7_UNORM_BLOCK,
//!         encoder: Encoder::Auto,
//!     }),
//!     container: Container::ktx2(),
//!     ..Default::default()
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! Use [`parse_format`] to build a [`TargetFormat`] from a string like `"bc7"`,
//! `"intel_bc7"`, or `"rgba8unorm"`.
//!
//! # Feature flags
//!
//! Each encoder backend is an independent feature, all enabled by default:
//! `encoder-bc7enc`, `encoder-intel`, `encoder-etcpak`, `encoder-amd`, and
//! `encoder-astcenc`. `ispc-prebuilt` (default) links prebuilt ISPC kernels;
//! `ispc-build-from-source` compiles them instead and requires `ispc` on
//! `PATH` — enable exactly one of the two. The default-off `rayon` feature
//! parallelizes compression within each image on the active Rayon pool.

// Pixel decode/encode paths reinterpret little-endian file bytes in place and
// index whole images with `usize`.
#[cfg(any(target_endian = "big", target_pointer_width = "16"))]
compile_error!("ctt only supports little-endian targets with 32-bit or wider pointers");

/// Compile-checks the Rust code blocks in `README.md` as doctests so the
/// library example there cannot drift out of sync with the current API.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct ReadmeDoctests;

// ---- Core types ----

pub use ktx2::Format;

pub use alpha::AlphaMode;
pub use convert::{Container, ConvertSettings, Ktx2Supercompression, convert};
pub use cubemap::{CubemapInput, split_cubemap};
pub use error::{Error, Result};
pub use format::{TargetFormat, format_short_name, parse_format};
pub use processing::equirectangular::{EquirectangularFront, EquirectangularOrientation};
pub use processing::{MipmapFilter, PipelineOutput, Swizzle, SwizzleChannel};
pub use quality::Quality;
pub use surface::{ColorSpace, Image, Surface, TextureKind};
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

/// **Not part of the public API.** Re-exports internal kernels so `benches/`
/// (a separate crate) can measure them directly; no stability guarantees.
#[doc(hidden)]
pub mod bench_internals {
    pub use crate::processing::Buffer;
    pub use crate::processing::kernels::{Fallback, Level, constructible_levels};

    // ---- sRGB ----

    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::kernels::srgb::store_srgb8_f32_avx512_escape;
    pub use crate::processing::kernels::srgb::{
        load_srgb8_f32_at, srgb_eotf_in_place_f32_at, srgb_oetf_in_place_f32_at,
        store_srgb8_f32_at, store_srgb8_f32_generic_at,
    };
    pub use crate::processing::load_kernels::{
        load_bgr8_srgb_f32, load_bgra8_srgb_f32, load_srgb8_f32, srgb_eotf_in_place_f32,
    };
    pub use crate::processing::store_kernels::{
        srgb_oetf_in_place_f32, store_bgr8_srgb_f32, store_bgra8_srgb_f32, store_f16_f32,
        store_srgb8_f32,
    };

    pub use crate::processing::load_kernels::load_f16_f32;

    // ---- Packed 32-bit formats ----

    pub use crate::processing::kernels::a2_10_10_10::{
        A2B_R_SHIFT, load_a2_f32_at, load_a2_u32_at, store_a2_f32_at, store_a2_u32_at,
    };
    pub use crate::processing::kernels::b10g11r11::{
        load_b10g11r11_f32_at, store_b10g11r11_f32_at,
    };
    pub use crate::processing::kernels::e5b9g9r9::{load_e5b9g9r9_f32_at, store_e5b9g9r9_f32_at};
    pub use crate::processing::load_kernels::{
        load_a2b10g10r10_sint_u32, load_a2b10g10r10_snorm_f32, load_a2b10g10r10_uint_u32,
        load_a2b10g10r10_unorm_f32, load_b10g11r11_f32, load_e5b9g9r9_f32,
    };
    pub use crate::processing::store_kernels::{
        store_a2b10g10r10_sint_u32, store_a2b10g10r10_snorm_f32, store_a2b10g10r10_uint_u32,
        store_a2b10g10r10_unorm_f32, store_b10g11r11_f32, store_e5b9g9r9_f32,
    };

    // ---- Equirectangular → cubemap projection ----

    pub use crate::processing::equirectangular::{EquirectangularPyramid, project_f32};
    pub use crate::processing::kernels::equirectangular::project_f32_at;
}
