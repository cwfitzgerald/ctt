//! The crate's only `fearless_simd` boundary.
//!
//! Every explicitly vectorized kernel lives in this subtree, and this subtree is
//! the only place in `crates/ctt/src` allowed to name `fearless_simd`. The rule
//! is mechanical: `grep -rn fearless_simd crates/ctt/src` must report hits only
//! under `processing/kernels/`.
//!
//! What crosses the boundary is plain safe Rust:
//!
//! * arguments and results are [`Surface`](crate::surface::Surface)s,
//!   [`Buffer<T>`](crate::processing::Buffer)s, and slices — never a vector type;
//! * no exported function is generic over `S: Simd`, so no caller has to name a
//!   token type or satisfy a SIMD trait bound;
//! * the only SIMD-flavored type that leaves is the opaque [`Level`] selector,
//!   re-exported here (with [`Fallback`] and [`constructible_levels`]) purely so
//!   tests and benches can force a specific backend through the `_at` entry
//!   points. The single exception is
//!   [`srgb::store_srgb8_f32_avx512_escape`], which takes the concrete
//!   `Avx512` token so the benches can measure the intrinsic escape against the
//!   generic kernel; callers obtain it from `Level::as_avx512()`, an inherent
//!   method, so they still never import `fearless_simd`.
//!
//! Inside the boundary the `docs/fearless-simd.md` notes apply in full — in
//! particular every function between a dispatch point and the vector ops,
//! closure literals included, must be `#[inline(always)]`. [`driver`] carries
//! that note; the format modules reference it.
//!
//! Layout: [`driver`] holds the format-independent scaffolding, [`curve_pass`]
//! the shared in-place transfer-curve loop, [`alpha`] the autovectorized
//! premultiply passes, [`equirectangular`] the projection kernels, and one
//! module per packed pixel format ([`srgb`], [`a2_10_10_10`], [`b10g11r11`],
//! [`e5b9g9r9`]) holding that format's load *and* store side, plus any scalar
//! production path cohesive with them. The format-facing surface stays in
//! [`load_kernels`](crate::processing::load_kernels) /
//! [`store_kernels`](crate::processing::store_kernels), which re-export the
//! entry points below.

pub(crate) mod a2_10_10_10;
pub(crate) mod alpha;
pub(crate) mod b10g11r11;
pub(crate) mod curve_pass;
pub(crate) mod driver;
pub(crate) mod e5b9g9r9;
pub(crate) mod equirectangular;
pub(crate) mod srgb;

// `pub` rather than `pub(crate)` only so `crate::bench_internals` can re-export
// them; `processing` is private, so this widens nothing.
pub use fearless_simd::{Fallback, Level};

/// Every constructible level on the host, so tests validate and benches measure
/// each backend. The names are the row IDs the benchmark groups use.
#[doc(hidden)]
pub fn constructible_levels() -> Vec<(&'static str, Level)> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    use fearless_simd::Simd as _;

    let detected = Level::new();
    let mut out = Vec::new();
    // `dispatch!` normalizes a level against the compile-time baseline before
    // matching, so a `Fallback` token only reaches the scalar backend where the
    // target guarantees no SIMD level. On aarch64 NEON is the architectural
    // baseline: a "fallback" row there would silently re-run the NEON backend
    // (with NEON semantics, e.g. NaN-propagating `max`), not scalar code.
    #[cfg(not(target_arch = "aarch64"))]
    out.push(("fallback", Level::Fallback(Fallback::new())));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(t) = detected.as_sse4_2() {
            out.push(("sse4_2", t.level()));
        }
        if let Some(t) = detected.as_avx2() {
            out.push(("avx2", t.level()));
        }
        if let Some(t) = detected.as_avx512() {
            out.push(("avx512", t.level()));
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = detected.as_neon() {
        out.push(("neon", t.level()));
    }
    out
}
