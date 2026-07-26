//! Micro-benchmarks for the equirectangular → cubemap projection kernels.
//!
//! One row per constructible SIMD level (levels the host cannot execute are
//! skipped). The pyramid is built once outside the measured region — the
//! benchmark isolates the projection itself. Throughput is reported in output
//! (cubemap) texels per second across all six faces.
//!
//! Note: the projection parallelizes across faces and row bands when the
//! `rayon` feature is on; run with `--features rayon` to measure the
//! multithreaded configuration, or without for single-thread kernels.

use std::f32::consts::PI;
use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ctt::EquirectangularOrientation;
use ctt::bench_internals::{Buffer, EquirectangularPyramid};

mod common;

/// Smooth HDR-looking source: low-frequency angular gradients plus a bright
/// "sun" spot, so LOD/aniso paths see realistic variation.
fn smooth_equirectangular(w: u32, h: u32) -> Buffer<f32> {
    let mut pixels = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            let phi = (u - 0.5) * 2.0 * PI;
            let theta = v * PI;
            let sun = (-((phi - 0.7).powi(2) + (theta - 0.9).powi(2)) * 40.0).exp() * 500.0;
            pixels.push([
                phi.sin() * 0.5 + 0.6 + sun,
                theta.cos() * 0.5 + 0.6 + sun,
                phi.cos() * theta.sin() * 0.5 + 0.6 + sun,
                1.0,
            ]);
        }
    }
    Buffer {
        pixels,
        width: w,
        height: h,
    }
}

fn bench_size(c: &mut Criterion, src_w: u32, src_h: u32) {
    use ctt::bench_internals as k;
    let face = src_w / 4;
    let out_texels = 6 * face as u64 * face as u64;
    let pyramid =
        EquirectangularPyramid::new(smooth_equirectangular(src_w, src_h)).expect("pyramid");
    let orientation = EquirectangularOrientation::default();

    let mut group = c.benchmark_group(format!("equirectangular_{src_w}x{src_h}_to_{face}"));
    group.throughput(Throughput::Elements(out_texels));
    // Each output texel is resampled with mip/aniso filtering, so a sample here
    // is far costlier than the packed/sRGB kernels; 20 keeps wall-clock per
    // bench reasonable without starving criterion's estimator.
    group.sample_size(20);

    common::bench_levels(&mut group, "", |b, level| {
        b.iter(|| {
            black_box(k::project_f32_at(
                level,
                black_box(&pyramid),
                face,
                orientation,
            ))
        });
    });

    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_size(c, 2048, 1024);
    bench_size(c, 4096, 2048);
}

criterion_group!(equirectangular_benches, benches);
criterion_main!(equirectangular_benches);
