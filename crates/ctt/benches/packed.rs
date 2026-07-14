//! Micro-benchmarks for the packed 32-bit format kernels.
//!
//! Five representative paths:
//!   * `A2B10G10R10_UNORM` load — reciprocal-multiply field normalization.
//!   * `B10G11R11_UFLOAT` load — FMA-fused small-float decode.
//!   * `A2B10G10R10_UNORM` store — FMA-fused rounding and bit packing.
//!   * `B10G11R11_UFLOAT` store — integer small-float encoding and bit packing.
//!   * `E5B9G9R9_UFLOAT` store — reciprocal-power-of-two + FMA-fused rounding.
//!
//! Each runs scalar plus every SIMD tier the host supports (skipped otherwise),
//! so the same bench is meaningful on x86_64 and aarch64. Throughput is
//! reported in pixels/second.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ctt::bench_internals::{A2B_R_SHIFT, Buffer};
use ctt::{AlphaMode, ColorSpace, Format, Surface};

const SIDE: u32 = 1024;
const PIXEL_COUNT: u64 = (SIDE as u64) * (SIDE as u64);

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

fn bench_a2b10g10r10_load(c: &mut Criterion) {
    let surface = make_packed_surface(Format::A2B10G10R10_UNORM_PACK32);

    let mut g = c.benchmark_group("a2b10g10r10_unorm_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    g.bench_function("serial", |b| {
        b.iter(|| {
            ctt::bench_internals::load_a2_unorm_serial::<A2B_R_SHIFT>(black_box(&surface)).unwrap()
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        use ctt::bench_internals::{load_a2_f32_avx2, load_a2_f32_avx512, load_a2_f32_sse4_1};

        if is_x86_feature_detected!("sse4.1") {
            g.bench_function("sse4_1", |b| {
                // SAFETY: runtime check confirmed sse4.1 is available.
                b.iter(|| unsafe {
                    load_a2_f32_sse4_1::<A2B_R_SHIFT, false>(black_box(&surface)).unwrap()
                });
            });
        }
        if is_x86_feature_detected!("avx2") {
            g.bench_function("avx2", |b| {
                // SAFETY: runtime check confirmed avx2 is available.
                b.iter(|| unsafe {
                    load_a2_f32_avx2::<A2B_R_SHIFT, false>(black_box(&surface)).unwrap()
                });
            });
        }
        if ctt::bench_internals::has_avx512() {
            g.bench_function("avx512", |b| {
                // SAFETY: runtime check confirmed avx512f+vl+bw are available.
                b.iter(|| unsafe {
                    load_a2_f32_avx512::<A2B_R_SHIFT, false>(black_box(&surface)).unwrap()
                });
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use ctt::bench_internals::load_a2_f32_neon;

        if std::arch::is_aarch64_feature_detected!("neon") {
            g.bench_function("neon", |b| {
                // SAFETY: runtime check confirmed NEON is available.
                b.iter(|| unsafe {
                    load_a2_f32_neon::<A2B_R_SHIFT, false>(black_box(&surface)).unwrap()
                });
            });
        }
    }

    g.finish();
}

fn bench_b10g11r11_load(c: &mut Criterion) {
    let surface = make_packed_surface(Format::B10G11R11_UFLOAT_PACK32);

    let mut g = c.benchmark_group("b10g11r11_load_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    g.bench_function("serial", |b| {
        b.iter(|| ctt::bench_internals::load_b10g11r11_f32_serial(black_box(&surface)).unwrap());
    });

    #[cfg(target_arch = "x86_64")]
    {
        use ctt::bench_internals::{
            load_b10g11r11_f32_avx2_fma, load_b10g11r11_f32_avx512, load_b10g11r11_f32_sse4_1,
        };

        if is_x86_feature_detected!("sse4.1") {
            g.bench_function("sse4_1", |b| {
                // SAFETY: runtime check confirmed sse4.1 is available.
                b.iter(|| unsafe { load_b10g11r11_f32_sse4_1(black_box(&surface)).unwrap() });
            });
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            g.bench_function("avx2_fma", |b| {
                // SAFETY: runtime check confirmed avx2+fma are available.
                b.iter(|| unsafe { load_b10g11r11_f32_avx2_fma(black_box(&surface)).unwrap() });
            });
        }
        if ctt::bench_internals::has_avx512() {
            g.bench_function("avx512", |b| {
                // SAFETY: runtime check confirmed avx512f+vl+bw are available.
                b.iter(|| unsafe { load_b10g11r11_f32_avx512(black_box(&surface)).unwrap() });
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use ctt::bench_internals::load_b10g11r11_f32_neon;

        if std::arch::is_aarch64_feature_detected!("neon") {
            g.bench_function("neon", |b| {
                // SAFETY: runtime check confirmed NEON is available.
                b.iter(|| unsafe { load_b10g11r11_f32_neon(black_box(&surface)).unwrap() });
            });
        }
    }

    g.finish();
}

fn bench_e5b9g9r9_store(c: &mut Criterion) {
    let buf = make_buffer();

    let mut g = c.benchmark_group("e5b9g9r9_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    g.bench_function("serial", |b| {
        b.iter(|| ctt::bench_internals::store_e5b9g9r9_f32_serial(black_box(&buf)));
    });

    #[cfg(target_arch = "x86_64")]
    {
        use ctt::bench_internals::{
            store_e5b9g9r9_f32_avx2_fma, store_e5b9g9r9_f32_avx512, store_e5b9g9r9_f32_sse4_1,
        };

        if is_x86_feature_detected!("sse4.1") {
            g.bench_function("sse4_1", |b| {
                // SAFETY: runtime check confirmed sse4.1 is available.
                b.iter(|| unsafe { store_e5b9g9r9_f32_sse4_1(black_box(&buf)) });
            });
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            g.bench_function("avx2_fma", |b| {
                // SAFETY: runtime check confirmed avx2+fma are available.
                b.iter(|| unsafe { store_e5b9g9r9_f32_avx2_fma(black_box(&buf)) });
            });
        }
        if ctt::bench_internals::has_avx512() {
            g.bench_function("avx512", |b| {
                // SAFETY: runtime check confirmed avx512f+vl+bw are available.
                b.iter(|| unsafe { store_e5b9g9r9_f32_avx512(black_box(&buf)) });
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use ctt::bench_internals::store_e5b9g9r9_f32_neon;

        if std::arch::is_aarch64_feature_detected!("neon") {
            g.bench_function("neon", |b| {
                // SAFETY: runtime check confirmed NEON is available.
                b.iter(|| unsafe { store_e5b9g9r9_f32_neon(black_box(&buf)) });
            });
        }
    }

    g.finish();
}

fn bench_a2b10g10r10_store(c: &mut Criterion) {
    let buf = make_unorm_buffer();

    let mut g = c.benchmark_group("a2b10g10r10_unorm_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    g.bench_function("serial", |b| {
        b.iter(|| ctt::bench_internals::store_a2_unorm_serial::<A2B_R_SHIFT>(black_box(&buf)));
    });

    #[cfg(target_arch = "x86_64")]
    {
        use ctt::bench_internals::{
            store_a2_f32_avx2_fma, store_a2_f32_avx512, store_a2_f32_sse4_1,
        };

        if is_x86_feature_detected!("sse4.1") {
            g.bench_function("sse4_1", |b| {
                // SAFETY: runtime check confirmed sse4.1 is available.
                b.iter(|| unsafe { store_a2_f32_sse4_1::<A2B_R_SHIFT, false>(black_box(&buf)) });
            });
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            g.bench_function("avx2_fma", |b| {
                // SAFETY: runtime check confirmed avx2+fma are available.
                b.iter(|| unsafe { store_a2_f32_avx2_fma::<A2B_R_SHIFT, false>(black_box(&buf)) });
            });
        }
        if ctt::bench_internals::has_avx512() {
            g.bench_function("avx512", |b| {
                // SAFETY: runtime check confirmed avx512f+vl+bw are available.
                b.iter(|| unsafe { store_a2_f32_avx512::<A2B_R_SHIFT, false>(black_box(&buf)) });
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use ctt::bench_internals::store_a2_f32_neon;

        if std::arch::is_aarch64_feature_detected!("neon") {
            g.bench_function("neon", |b| {
                // SAFETY: runtime check confirmed NEON is available.
                b.iter(|| unsafe { store_a2_f32_neon::<A2B_R_SHIFT, false>(black_box(&buf)) });
            });
        }
    }

    g.finish();
}

fn bench_b10g11r11_store(c: &mut Criterion) {
    let buf = make_buffer();

    let mut g = c.benchmark_group("b10g11r11_store_1024x1024");
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    g.bench_function("serial", |b| {
        b.iter(|| ctt::bench_internals::store_b10g11r11_f32_serial(black_box(&buf)));
    });

    #[cfg(target_arch = "x86_64")]
    {
        use ctt::bench_internals::{store_b10g11r11_f32_avx2, store_b10g11r11_f32_avx512};

        if is_x86_feature_detected!("avx2") {
            g.bench_function("avx2", |b| {
                // SAFETY: runtime check confirmed avx2 is available.
                b.iter(|| unsafe { store_b10g11r11_f32_avx2(black_box(&buf)) });
            });
        }
        if ctt::bench_internals::has_avx512() {
            g.bench_function("avx512", |b| {
                // SAFETY: runtime check confirmed avx512f+vl+bw are available.
                b.iter(|| unsafe { store_b10g11r11_f32_avx512(black_box(&buf)) });
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use ctt::bench_internals::store_b10g11r11_f32_neon;

        if std::arch::is_aarch64_feature_detected!("neon") {
            g.bench_function("neon", |b| {
                // SAFETY: runtime check confirmed NEON is available.
                b.iter(|| unsafe { store_b10g11r11_f32_neon(black_box(&buf)) });
            });
        }
    }

    g.finish();
}

criterion_group!(
    benches,
    bench_a2b10g10r10_load,
    bench_b10g11r11_load,
    bench_a2b10g10r10_store,
    bench_b10g11r11_store,
    bench_e5b9g9r9_store
);
criterion_main!(benches);
