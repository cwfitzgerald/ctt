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

    pub use crate::processing::load_kernels::srgb::{
        load_srgb8_f32_serial, srgb_eotf_in_place_f32_serial,
    };
    pub use crate::processing::load_kernels::{
        load_bgr8_srgb_f32, load_bgra8_srgb_f32, load_srgb8_f32, srgb_eotf_in_place_f32,
    };
    pub use crate::processing::store_kernels::srgb::{
        srgb_oetf_in_place_f32_serial, store_bgra8_srgb_f32_serial, store_srgb8_f32_serial,
    };
    pub use crate::processing::store_kernels::{
        srgb_oetf_in_place_f32, store_bgr8_srgb_f32, store_bgra8_srgb_f32, store_f16_f32,
        store_srgb8_f32,
    };

    pub use crate::processing::load_kernels::load_f16_f32;

    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::load_kernels::srgb::{
        load_srgb8_rgba_f32_avx2_fma, load_srgb8_rgba_f32_avx512, load_srgb8_rgba_f32_sse4_1,
        srgb_eotf_in_place_f32_avx2_fma, srgb_eotf_in_place_f32_avx512,
        srgb_eotf_in_place_f32_sse4_1,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::store_kernels::srgb::{
        srgb_oetf_in_place_f32_avx2_fma, srgb_oetf_in_place_f32_avx512,
        srgb_oetf_in_place_f32_sse4_1, store_srgb8_f32_avx2_fma, store_srgb8_f32_avx512,
        store_srgb8_f32_sse4_1,
    };

    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::load_kernels::srgb::{
        load_srgb8_rgba_f32_neon, srgb_eotf_in_place_f32_neon,
    };
    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::store_kernels::srgb::{
        srgb_oetf_in_place_f32_neon, store_srgb8_f32_neon,
    };

    // ---- Packed 32-bit formats ----

    pub use crate::processing::load_kernels::a2_10_10_10::{A2B_R_SHIFT, load_a2_unorm_serial};
    pub use crate::processing::load_kernels::b10g11r11::load_b10g11r11_f32_serial;
    pub use crate::processing::store_kernels::a2_10_10_10::store_a2_unorm_serial;
    pub use crate::processing::store_kernels::b10g11r11::store_b10g11r11_f32_serial;
    pub use crate::processing::store_kernels::e5b9g9r9::store_e5b9g9r9_f32_serial;

    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::x86::has_avx512;

    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::load_kernels::a2_10_10_10::{
        load_a2_f32_avx2, load_a2_f32_avx512, load_a2_f32_sse4_1,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::load_kernels::b10g11r11::{
        load_b10g11r11_f32_avx2_fma, load_b10g11r11_f32_avx512, load_b10g11r11_f32_sse4_1,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::store_kernels::a2_10_10_10::{
        store_a2_f32_avx2_fma, store_a2_f32_avx512, store_a2_f32_sse4_1,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::store_kernels::b10g11r11::{
        store_b10g11r11_f32_avx2, store_b10g11r11_f32_avx512,
    };
    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::store_kernels::e5b9g9r9::{
        store_e5b9g9r9_f32_avx2_fma, store_e5b9g9r9_f32_avx512, store_e5b9g9r9_f32_sse4_1,
    };

    // ---- Equirectangular → cubemap projection ----

    pub use crate::processing::equirectangular::{
        EquirectangularPyramid, project_f32, project_f32_serial,
    };

    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::equirectangular::x86::{project_f32_avx2_fma, project_f32_avx512};

    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::equirectangular::neon::project_f32_neon;

    #[cfg(target_arch = "x86_64")]
    pub use crate::processing::x86::has_avx2_fma;

    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::load_kernels::a2_10_10_10::load_a2_f32_neon;
    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::load_kernels::b10g11r11::load_b10g11r11_f32_neon;
    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::store_kernels::a2_10_10_10::store_a2_f32_neon;
    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::store_kernels::b10g11r11::store_b10g11r11_f32_neon;
    #[cfg(target_arch = "aarch64")]
    pub use crate::processing::store_kernels::e5b9g9r9::store_e5b9g9r9_f32_neon;
}
