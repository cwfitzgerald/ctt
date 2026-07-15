//! Micro-benchmarks for the sRGB load and store kernels.
//!
//! All benches run on the same [`SIDE`]×[`SIDE`] image. Kernels gated on CPU
//! features are skipped if the host lacks the feature, so this bench is
//! safe to run on x86_64 and aarch64 machines — the subset that fires is
//! whatever the host can execute.
//!
//! Throughput is reported in pixels/second via
//! `Throughput::Elements(pixel_count)`.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use ctt::bench_internals::Buffer;
use ctt::{AlphaMode, ColorSpace, Format, Surface};

const SIDE: u32 = 1024;
const PIXEL_COUNT: u64 = (SIDE as u64) * (SIDE as u64);

/// Build a 4-channel sRGB8 surface with a deterministic pattern that
/// exercises both the linear (`byte <= 10`) and curve branches of the
/// EOTF approximation.
fn make_rgba_surface() -> Surface {
    let n = (SIDE as usize) * (SIDE as usize);
    let mut data = vec![0u8; n * 4];
    for i in 0..n {
        let base = i * 4;
        data[base] = (i.wrapping_mul(37)) as u8;
        data[base + 1] = (i.wrapping_mul(59).wrapping_add(11)) as u8;
        data[base + 2] = (i.wrapping_mul(97).wrapping_add(3)) as u8;
        data[base + 3] = (i.wrapping_mul(13)) as u8;
    }
    Surface {
        data,
        width: SIDE,
        height: SIDE,
        depth: 1,
        stride: SIDE * 4,
        slice_stride: 0,
        format: Format::R8G8B8A8_SRGB,
        color_space: ColorSpace::Srgb,
        alpha: AlphaMode::Straight,
    }
}

/// Same pattern as [`make_rgba_surface`] but 3 bytes per pixel for
/// [`load_bgr8_srgb_f32`].
fn make_bgr_surface() -> Surface {
    let n = (SIDE as usize) * (SIDE as usize);
    let mut data = vec![0u8; n * 3];
    for i in 0..n {
        let base = i * 3;
        data[base] = (i.wrapping_mul(37)) as u8;
        data[base + 1] = (i.wrapping_mul(59).wrapping_add(11)) as u8;
        data[base + 2] = (i.wrapping_mul(97).wrapping_add(3)) as u8;
    }
    Surface {
        data,
        width: SIDE,
        height: SIDE,
        depth: 1,
        stride: SIDE * 3,
        slice_stride: 0,
        format: Format::B8G8R8_SRGB,
        color_space: ColorSpace::Srgb,
        alpha: AlphaMode::Opaque,
    }
}

/// Build a 4-channel linear-f32 buffer covering the full [0, 1] range so
/// both OETF branches (linear-segment and curve) get exercised.
fn make_buffer() -> Buffer<f32> {
    let n = PIXEL_COUNT as usize;
    let mut pixels = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        pixels.push([t, (t * 0.5 + 0.2).clamp(0.0, 1.0), t * t, t]);
    }
    Buffer {
        pixels,
        width: SIDE,
        height: SIDE,
    }
}

fn bench_load(c: &mut Criterion) {
    let rgba = make_rgba_surface();
    let bgr = make_bgr_surface();

    let mut g = c.benchmark_group(format!("srgb_load_{SIDE}x{SIDE}"));
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    g.bench_function("load_srgb8_f32_serial_rgba", |b| {
        b.iter(|| ctt::bench_internals::load_srgb8_f32_serial(black_box(&rgba), 4).unwrap());
    });

    // Scalar comparison paths that are not covered by the 4-channel SIMD
    // specializations.
    g.bench_function("scalar_bgra", |b| {
        b.iter(|| ctt::bench_internals::load_bgra8_srgb_f32(black_box(&rgba)).unwrap());
    });
    g.bench_function("scalar_bgr", |b| {
        b.iter(|| ctt::bench_internals::load_bgr8_srgb_f32(black_box(&bgr)).unwrap());
    });

    #[cfg(target_arch = "x86_64")]
    {
        use ctt::bench_internals::{
            load_srgb8_rgba_f32_avx2_fma, load_srgb8_rgba_f32_avx512, load_srgb8_rgba_f32_sse4_1,
        };

        if is_x86_feature_detected!("sse4.1") {
            g.bench_function("sse4_1", |b| {
                // SAFETY: runtime feature check confirmed sse4.1 is available.
                b.iter(|| unsafe { load_srgb8_rgba_f32_sse4_1(black_box(&rgba)).unwrap() });
            });
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            g.bench_function("avx2_fma", |b| {
                // SAFETY: runtime feature check confirmed avx2+fma are available.
                b.iter(|| unsafe { load_srgb8_rgba_f32_avx2_fma(black_box(&rgba)).unwrap() });
            });
        }
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            g.bench_function("avx512", |b| {
                // SAFETY: runtime feature check confirmed avx512f+bw+vl are available.
                b.iter(|| unsafe { load_srgb8_rgba_f32_avx512(black_box(&rgba)).unwrap() });
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use ctt::bench_internals::load_srgb8_rgba_f32_neon;

        if std::arch::is_aarch64_feature_detected!("neon") {
            g.bench_function("neon", |b| {
                // SAFETY: runtime feature check confirmed NEON is available.
                b.iter(|| unsafe { load_srgb8_rgba_f32_neon(black_box(&rgba)).unwrap() });
            });
        }
    }

    g.finish();
}

fn bench_store(c: &mut Criterion) {
    let buf = make_buffer();
    let buf3 = Buffer {
        // `store_bgr8_srgb_f32` only reads lanes 0-2 from each pixel, so the
        // 4-lane buffer feeds it fine.
        pixels: buf.pixels.clone(),
        width: SIDE,
        height: SIDE,
    };

    let mut g = c.benchmark_group(format!("srgb_store_{SIDE}x{SIDE}"));
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    g.bench_function("store_srgb8_f32_serial_rgba", |b| {
        b.iter(|| ctt::bench_internals::store_srgb8_f32_serial(black_box(&buf), 4));
    });
    g.bench_function("store_bgra8_srgb_f32_serial", |b| {
        b.iter(|| ctt::bench_internals::store_bgra8_srgb_f32_serial(black_box(&buf)));
    });

    g.bench_function("scalar_bgr", |b| {
        b.iter(|| ctt::bench_internals::store_bgr8_srgb_f32(black_box(&buf3)));
    });

    #[cfg(target_arch = "x86_64")]
    {
        use ctt::bench_internals::{
            store_srgb8_f32_avx2_fma, store_srgb8_f32_avx512, store_srgb8_f32_sse4_1,
        };

        if is_x86_feature_detected!("sse4.1") {
            g.bench_function("sse4_1_rgba", |b| {
                // SAFETY: runtime feature check confirmed sse4.1 is available.
                b.iter(|| unsafe { store_srgb8_f32_sse4_1::<false>(black_box(&buf)) });
            });
            g.bench_function("sse4_1_bgra", |b| {
                // SAFETY: runtime feature check confirmed sse4.1 is available.
                b.iter(|| unsafe { store_srgb8_f32_sse4_1::<true>(black_box(&buf)) });
            });
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            g.bench_function("avx2_fma_rgba", |b| {
                // SAFETY: runtime feature check confirmed avx2+fma are available.
                b.iter(|| unsafe { store_srgb8_f32_avx2_fma::<false>(black_box(&buf)) });
            });
            g.bench_function("avx2_fma_bgra", |b| {
                // SAFETY: runtime feature check confirmed avx2+fma are available.
                b.iter(|| unsafe { store_srgb8_f32_avx2_fma::<true>(black_box(&buf)) });
            });
        }
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            g.bench_function("avx512_rgba", |b| {
                // SAFETY: runtime feature check confirmed avx512f+bw+vl are available.
                b.iter(|| unsafe { store_srgb8_f32_avx512::<false>(black_box(&buf)) });
            });
            g.bench_function("avx512_bgra", |b| {
                // SAFETY: runtime feature check confirmed avx512f+bw+vl are available.
                b.iter(|| unsafe { store_srgb8_f32_avx512::<true>(black_box(&buf)) });
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use ctt::bench_internals::store_srgb8_f32_neon;

        if std::arch::is_aarch64_feature_detected!("neon") {
            g.bench_function("neon_rgba", |b| {
                // SAFETY: runtime feature check confirmed NEON is available.
                b.iter(|| unsafe { store_srgb8_f32_neon::<false>(black_box(&buf)) });
            });
            g.bench_function("neon_bgra", |b| {
                // SAFETY: runtime feature check confirmed NEON is available.
                b.iter(|| unsafe { store_srgb8_f32_neon::<true>(black_box(&buf)) });
            });
        }
    }

    g.finish();
}

/// One runtime-selectable variant of an in-place pass.
type InPlaceVariant = (&'static str, fn(&mut Vec<[f32; 4]>));

/// Build the `(name, kernel)` list for one in-place pass: the serial path
/// always, each SIMD kernel only when the host supports its ISA. The wrapper
/// closures' `unsafe` is sound because a kernel is only listed after its
/// runtime feature check passes.
macro_rules! in_place_variants {
    ($serial:path, $sse4_1:path, $avx2_fma:path, $avx512:path, $neon:path $(,)?) => {{
        let mut variants: Vec<InPlaceVariant> = vec![("serial", |pixels| $serial(pixels))];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
                variants.push(("sse4_1", |pixels| unsafe { $sse4_1(pixels) }));
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                variants.push(("avx2_fma", |pixels| unsafe { $avx2_fma(pixels) }));
            }
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
            {
                variants.push(("avx512", |pixels| unsafe { $avx512(pixels) }));
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                variants.push(("neon", |pixels| unsafe { $neon(pixels) }));
            }
        }

        variants
    }};
}

/// Bench every listed variant of an in-place pass. The pass mutates its
/// input, so every iteration gets a fresh clone via `iter_batched_ref`; the
/// clone happens outside the timed region.
fn bench_in_place(c: &mut Criterion, group: &str, variants: &[InPlaceVariant]) {
    let buf = make_buffer();

    let mut g = c.benchmark_group(group);
    g.throughput(Throughput::Elements(PIXEL_COUNT));
    for &(name, kernel) in variants {
        g.bench_function(name, |b| {
            b.iter_batched_ref(|| buf.pixels.clone(), kernel, BatchSize::LargeInput);
        });
    }
    g.finish();
}

fn bench_oetf_in_place(c: &mut Criterion) {
    use ctt::bench_internals as k;
    bench_in_place(
        c,
        "srgb_oetf_in_place",
        &in_place_variants!(
            k::srgb_oetf_in_place_f32_serial,
            k::srgb_oetf_in_place_f32_sse4_1,
            k::srgb_oetf_in_place_f32_avx2_fma,
            k::srgb_oetf_in_place_f32_avx512,
            k::srgb_oetf_in_place_f32_neon,
        ),
    );
}

fn bench_eotf_in_place(c: &mut Criterion) {
    use ctt::bench_internals as k;
    bench_in_place(
        c,
        "srgb_eotf_in_place",
        &in_place_variants!(
            k::srgb_eotf_in_place_f32_serial,
            k::srgb_eotf_in_place_f32_sse4_1,
            k::srgb_eotf_in_place_f32_avx2_fma,
            k::srgb_eotf_in_place_f32_avx512,
            k::srgb_eotf_in_place_f32_neon,
        ),
    );
}

criterion_group!(
    benches,
    bench_load,
    bench_store,
    bench_oetf_in_place,
    bench_eotf_in_place
);
criterion_main!(benches);
