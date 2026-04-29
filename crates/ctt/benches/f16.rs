//! Micro-benchmarks for f16 ↔ f32 load/store kernels.
//!
//! The benchmark calls the public `load_f16_f32` / `store_f16_f32` dispatch,
//! so swapping the kernel between the scalar and `half`-bulk paths changes
//! what's measured. Run with `--save-baseline` on each variant and compare.
//!
//! Throughput is reported in pixels/second.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ctt::bench_internals::{Buffer, load_f16_f32, store_f16_f32};
use ctt::{AlphaMode, ColorSpace, Format, Surface};
use half::f16;

const SIDE: u32 = 1024;
const PIXEL_COUNT: u64 = (SIDE as u64) * (SIDE as u64);

/// Build an f16 surface with `channels` channels and the matching
/// `Format::*_SFLOAT`. Pattern covers a range of values so neither the
/// scalar nor the bulk path can short-circuit on a constant.
fn make_f16_surface(channels: usize) -> Surface {
    let format = match channels {
        1 => Format::R16_SFLOAT,
        2 => Format::R16G16_SFLOAT,
        4 => Format::R16G16B16A16_SFLOAT,
        _ => unreachable!(),
    };
    let n = (SIDE as usize) * (SIDE as usize);
    let mut data = vec![0u8; n * channels * 2];
    for i in 0..n {
        for c in 0..channels {
            let v = ((i.wrapping_mul(37 + c).wrapping_add(c * 11)) as f32 / 65535.0).fract();
            let bytes = f16::from_f32(v).to_le_bytes();
            let base = (i * channels + c) * 2;
            data[base] = bytes[0];
            data[base + 1] = bytes[1];
        }
    }
    Surface {
        data,
        width: SIDE,
        height: SIDE,
        depth: 1,
        stride: SIDE * channels as u32 * 2,
        slice_stride: 0,
        format,
        color_space: ColorSpace::Linear,
        alpha: AlphaMode::Opaque,
    }
}

/// Build a `Buffer<f32>` whose lanes hold values representable by f16.
/// Lane defaults (alpha=1.0) are preserved for sub-4-channel stores.
fn make_f32_buffer() -> Buffer<f32> {
    let n = PIXEL_COUNT as usize;
    let mut pixels = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f32 / n as f32).fract();
        // Round-trip through f16 so the store has nothing to clamp.
        let r = f16::from_f32(t).to_f32();
        let g = f16::from_f32((t * 0.5 + 0.2).clamp(0.0, 1.0)).to_f32();
        let b = f16::from_f32(t * t).to_f32();
        let a = f16::from_f32(0.75 + (t * 0.25)).to_f32();
        pixels.push([r, g, b, a]);
    }
    Buffer {
        pixels,
        width: SIDE,
        height: SIDE,
    }
}

fn bench_load(c: &mut Criterion) {
    let surfaces: Vec<(usize, Surface)> = vec![
        (1, make_f16_surface(1)),
        (2, make_f16_surface(2)),
        (4, make_f16_surface(4)),
    ];

    let mut g = c.benchmark_group("f16_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    for (channels, surface) in &surfaces {
        let name = format!("load_f16_f32_ch{channels}");
        g.bench_function(&name, |b| {
            b.iter(|| load_f16_f32(black_box(surface), *channels).unwrap());
        });
    }

    g.finish();
}

fn bench_store(c: &mut Criterion) {
    let buf = make_f32_buffer();

    let mut g = c.benchmark_group("f16_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    for &channels in &[1usize, 2, 4] {
        let name = format!("store_f16_f32_ch{channels}");
        g.bench_function(&name, |b| {
            b.iter(|| store_f16_f32(black_box(&buf), channels));
        });
    }

    g.finish();
}

criterion_group!(benches, bench_load, bench_store);
criterion_main!(benches);
