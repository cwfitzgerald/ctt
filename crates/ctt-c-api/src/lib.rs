//! C bindings for the [`ctt`] texture compression library.
//!
//! All allocations happen on the Rust side. Each opaque type has a `*_destroy`
//! function; APIs documented as "consuming" take ownership of the passed
//! handle on both success and failure.
//!
//! The binding lives behind a stable, hand-curated header that is regenerated
//! by `cargo xtask generate-c-header`.

#![allow(clippy::missing_safety_doc)]

mod convert;
mod cubemap;
mod error;
mod formats;
mod image;
mod input;
mod output;
mod surface;
mod types;

pub use convert::*;
pub use cubemap::*;
pub use error::*;
pub use formats::*;
pub use image::*;
pub use input::*;
pub use output::*;
pub use surface::*;
pub use types::*;
