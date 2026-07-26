//! Micro-benchmarks for the packed 32-bit format kernels.
//!
//! Representative paths:
//!   * `A2B10G10R10_UNORM` load — reciprocal-multiply field normalization.
//!   * `B10G11R11_UFLOAT` load — FMA-fused small-float decode.
//!   * `E5B9G9R9_UFLOAT` load — shared-exponent decode.
//!   * `A2B10G10R10_UNORM` store — FMA-fused rounding and bit packing.
//!   * `B10G11R11_UFLOAT` store — integer small-float encoding and bit packing.
//!   * `E5B9G9R9_UFLOAT` store — reciprocal-power-of-two + FMA-fused rounding.
//!
//! The `A2B10G10R10` block covers each distinct codegen path: SNORM
//! (sign-extension/copysign), UINT (integer min-clamp), and SINT (signed clamp)
//! for both load and store, alongside the UNORM f32 paths above.
//!
//! Each group emits one row per constructible SIMD level (levels the host
//! cannot execute are skipped), so the same bench is meaningful on x86_64 and
//! aarch64. Throughput is reported in pixels/second.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ctt::bench_internals::{A2B_R_SHIFT, Buffer};
use ctt::{AlphaMode, ColorSpace, Format, Surface};

mod common;

use common::{PIXEL_COUNT, SIDE};

/// Build a packed 32-bit surface (4 bytes/pixel) with a deterministic word
/// pattern that spreads bits across all four fields.
fn make_packed_surface(format: Format) -> Surface {
    let n = (SIDE as usize) * (SIDE as usize);
    let mut data = vec![0u8; n * 4];
    for (i, word) in data.chunks_exact_mut(4).enumerate() {
        let v = (i as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(0x9e37_79b9);
        word.copy_from_slice(&v.to_le_bytes());
    }
    Surface {
        data,
        width: SIDE,
        height: SIDE,
        depth: 1,
        stride: SIDE * 4,
        slice_stride: 0,
        format,
        color_space: ColorSpace::Linear,
        alpha: AlphaMode::Straight,
    }
}

/// Build a 4-channel linear-f32 buffer spanning a wide HDR magnitude range so
/// the shared-exponent store exercises many exponents.
fn make_buffer() -> Buffer<f32> {
    let n = PIXEL_COUNT as usize;
    let mut pixels = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        let m = t * 64.0; // 0..64
        pixels.push([m, m * 0.5 + 0.1, t * t * 12.0, 1.0]);
    }
    Buffer {
        pixels,
        width: SIDE,
        height: SIDE,
    }
}

/// Build a 4-channel buffer in the UNORM range, including exact halfway points
/// and out-of-range values so the store clamp + round-half-away path is
/// exercised.
fn make_unorm_buffer() -> Buffer<f32> {
    let n = PIXEL_COUNT as usize;
    let mut pixels = Vec::with_capacity(n);
    for i in 0..n {
        let k = (i % 1025) as f32;
        let v = (k + 0.5) / 1023.0; // straddles rounding boundaries, some > 1
        pixels.push([v, 1.0 - v, v * 0.5, ((i % 4) as f32) / 3.0]);
    }
    Buffer {
        pixels,
        width: SIDE,
        height: SIDE,
    }
}

/// Build a 4-channel buffer in the SNORM range, sweeping across zero and past
/// both sign boundaries so the store's copysign rounding and clamp fire on
/// negative inputs.
fn make_snorm_buffer() -> Buffer<f32> {
    let n = PIXEL_COUNT as usize;
    let mut pixels = Vec::with_capacity(n);
    for i in 0..n {
        let k = (i % 1025) as f32;
        let v = (k + 0.5) / 511.0 - 1.0; // spans [-1, 1+], straddles boundaries
        pixels.push([v, -v, v * 0.5, ((i % 3) as f32) - 1.0]);
    }
    Buffer {
        pixels,
        width: SIDE,
        height: SIDE,
    }
}

/// Build a 4-channel u32 buffer for the UINT store, spanning past the 10-bit
/// (and 2-bit alpha) field maxima so the integer min-clamp path fires.
fn make_uint_buffer() -> Buffer<u32> {
    let n = PIXEL_COUNT as usize;
    let mut pixels = Vec::with_capacity(n);
    for i in 0..n {
        let v = (i as u32) % 1200; // 0..1199, exceeds the 10-bit max of 1023
        pixels.push([v, 1199 - v, v / 2, (i as u32) % 5]); // alpha 0..4 exceeds 3
    }
    Buffer {
        pixels,
        width: SIDE,
        height: SIDE,
    }
}

/// Build a 4-channel u32 buffer for the SINT store: each lane is an i32 value
/// reinterpreted as u32, sweeping across zero and past both signed field
/// boundaries so the signed-clamp path fires on negative inputs.
fn make_sint_buffer() -> Buffer<u32> {
    let n = PIXEL_COUNT as usize;
    let mut pixels = Vec::with_capacity(n);
    for i in 0..n {
        let v = (i as i32) % 1200 - 600; // -600..599, past [-512, 511]
        let a = (i as i32) % 5 - 2; // -2..2, past [-2, 1]
        pixels.push([v as u32, (-v) as u32, (v / 2) as u32, a as u32]);
    }
    Buffer {
        pixels,
        width: SIDE,
        height: SIDE,
    }
}

fn bench_a2b10g10r10_load(c: &mut Criterion) {
    let surface = make_packed_surface(Format::A2B10G10R10_UNORM_PACK32);

    let mut g = c.benchmark_group("a2b10g10r10_unorm_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| {
            ctt::bench_internals::load_a2_f32_at::<A2B_R_SHIFT, false>(level, black_box(&surface))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_b10g11r11_load(c: &mut Criterion) {
    let surface = make_packed_surface(Format::B10G11R11_UFLOAT_PACK32);

    let mut g = c.benchmark_group("b10g11r11_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| ctt::bench_internals::load_b10g11r11_f32_at(level, black_box(&surface)).unwrap());
    });

    g.finish();
}

fn bench_e5b9g9r9_load(c: &mut Criterion) {
    let surface = make_packed_surface(Format::E5B9G9R9_UFLOAT_PACK32);

    let mut g = c.benchmark_group("e5b9g9r9_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| ctt::bench_internals::load_e5b9g9r9_f32_at(level, black_box(&surface)).unwrap());
    });

    g.finish();
}

fn bench_e5b9g9r9_store(c: &mut Criterion) {
    let buf = make_buffer();

    let mut g = c.benchmark_group("e5b9g9r9_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| ctt::bench_internals::store_e5b9g9r9_f32_at(level, black_box(&buf)));
    });

    g.finish();
}

fn bench_a2b10g10r10_store(c: &mut Criterion) {
    let buf = make_unorm_buffer();

    let mut g = c.benchmark_group("a2b10g10r10_unorm_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| {
            ctt::bench_internals::store_a2_f32_at::<A2B_R_SHIFT, false>(level, black_box(&buf))
        });
    });

    g.finish();
}

fn bench_b10g11r11_store(c: &mut Criterion) {
    let buf = make_buffer();

    let mut g = c.benchmark_group("b10g11r11_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| ctt::bench_internals::store_b10g11r11_f32_at(level, black_box(&buf)));
    });

    g.finish();
}

fn bench_a2b10g10r10_snorm_load(c: &mut Criterion) {
    let surface = make_packed_surface(Format::A2B10G10R10_SNORM_PACK32);

    let mut g = c.benchmark_group("a2b10g10r10_snorm_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| {
            ctt::bench_internals::load_a2_f32_at::<A2B_R_SHIFT, true>(level, black_box(&surface))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_a2b10g10r10_snorm_store(c: &mut Criterion) {
    let buf = make_snorm_buffer();

    let mut g = c.benchmark_group("a2b10g10r10_snorm_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| {
            ctt::bench_internals::store_a2_f32_at::<A2B_R_SHIFT, true>(level, black_box(&buf))
        });
    });

    g.finish();
}

fn bench_a2b10g10r10_uint_load(c: &mut Criterion) {
    let surface = make_packed_surface(Format::A2B10G10R10_UINT_PACK32);

    let mut g = c.benchmark_group("a2b10g10r10_uint_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| {
            ctt::bench_internals::load_a2_u32_at::<A2B_R_SHIFT, false>(level, black_box(&surface))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_a2b10g10r10_uint_store(c: &mut Criterion) {
    let buf = make_uint_buffer();

    let mut g = c.benchmark_group("a2b10g10r10_uint_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| {
            ctt::bench_internals::store_a2_u32_at::<A2B_R_SHIFT, false>(level, black_box(&buf))
        });
    });

    g.finish();
}

fn bench_a2b10g10r10_sint_load(c: &mut Criterion) {
    let surface = make_packed_surface(Format::A2B10G10R10_SINT_PACK32);

    let mut g = c.benchmark_group("a2b10g10r10_sint_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| {
            ctt::bench_internals::load_a2_u32_at::<A2B_R_SHIFT, true>(level, black_box(&surface))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_a2b10g10r10_sint_store(c: &mut Criterion) {
    let buf = make_sint_buffer();

    let mut g = c.benchmark_group("a2b10g10r10_sint_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| {
            ctt::bench_internals::store_a2_u32_at::<A2B_R_SHIFT, true>(level, black_box(&buf))
        });
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_a2b10g10r10_load,
    bench_b10g11r11_load,
    bench_e5b9g9r9_load,
    bench_a2b10g10r10_store,
    bench_b10g11r11_store,
    bench_e5b9g9r9_store,
    bench_a2b10g10r10_snorm_load,
    bench_a2b10g10r10_snorm_store,
    bench_a2b10g10r10_uint_load,
    bench_a2b10g10r10_uint_store,
    bench_a2b10g10r10_sint_load,
    bench_a2b10g10r10_sint_store
);
criterion_main!(benches);
