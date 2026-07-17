#![allow(dead_code)]

//! Shared helpers for the ctt kernel micro-benchmarks.
//!
//! Each bench binary (`srgb`, `packed`, `f16`, `equirectangular`) is its own
//! crate. `autobenches = false` keeps this file from becoming a bench target
//! of its own; the explicit `[[bench]]` entries in `Cargo.toml` register the
//! real binaries, and each pulls this module in with a plain `mod common;`.
//! Not every binary uses every helper, hence the crate-level `dead_code`
//! allow above.
//!
//! Every group sweeps the constructible SIMD levels via [`bench_levels`], one
//! row per level, so the comparison is tier-against-tier on the same host.
//!
//! ## Allocation in the timed region
//!
//! The packed and sRGB load/store kernels return freshly allocated output
//! buffers, so those rows time the kernel *plus* the allocator. This is
//! inherent to the kernel API (the benches deliberately do not change the
//! signatures). The allocation cost is identical across every SIMD tier, so
//! relative comparisons between the forced-level rows remain valid.

use criterion::measurement::Measurement;
use criterion::{Bencher, BenchmarkGroup};
use ctt::bench_internals::{Level, constructible_levels};

/// Side length of the square test images shared across the packed, sRGB, and
/// f16 benches.
pub const SIDE: u32 = 1024;
/// Pixel count of a [`SIDE`]×[`SIDE`] image.
pub const PIXEL_COUNT: u64 = (SIDE as u64) * (SIDE as u64);

/// Register one bench per constructible SIMD level, emitting the IDs
/// `fallback{suffix}`, `sse4_2{suffix}`,
/// `avx2{suffix}`, `avx512{suffix}` (x86/x86_64) and
/// `neon{suffix}` (aarch64). Levels the host cannot execute are skipped. Pass
/// `""` for the canonical rows or a suffix such as `"_bgra"` for an additional
/// channel-order sweep.
///
/// `run` receives the forced [`Level`] for each row and drives the measurement
/// (`b.iter(..)`, `b.iter_batched_ref(..)`, etc.).
pub fn bench_levels<M: Measurement>(
    g: &mut BenchmarkGroup<'_, M>,
    suffix: &str,
    mut run: impl FnMut(&mut Bencher<'_, M>, Level),
) {
    for (name, level) in constructible_levels() {
        g.bench_function(format!("{name}{suffix}"), |b| run(b, level));
    }
}
