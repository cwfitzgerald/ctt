//! Micro-benchmarks for the sRGB load and store kernels.
//!
//! All benches run on the same [`SIDE`]×[`SIDE`] image, one row per
//! constructible SIMD level. Levels the host cannot execute are skipped, so this
//! bench is safe to run on x86_64 and aarch64 machines — the subset that fires
//! is whatever the host can execute.
//!
//! Groups:
//!   * `srgb_load_*` — 4-channel sRGB8 → linear f32, `_bgra` for the swapped
//!     byte order, plus `bgr8` for the 3-channel scalar production path (no
//!     packed-word kernel exists at 3 bytes per pixel).
//!   * `srgb_store_*` — the inverse, with the AVX-512 tier split into the
//!     generic-rsqrt kernel and the `rsqrt14` intrinsic escape, and `bgr8` for
//!     the 3-channel scalar production path.
//!   * `srgb_{oetf,eotf}_in_place` — the f32 curve passes used by the 16+ bit
//!     formats.
//!
//! Throughput is reported in pixels/second via
//! `Throughput::Elements(pixel_count)`.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use ctt::bench_internals::{Buffer, Level};
use ctt::{AlphaMode, ColorSpace, Format, Surface};

mod common;

use common::{PIXEL_COUNT, SIDE};

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
    use ctt::bench_internals as k;
    let rgba = make_rgba_surface();
    let bgr = make_bgr_surface();

    let mut g = c.benchmark_group(format!("srgb_load_{SIDE}x{SIDE}"));
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    // RGBA sweep — `load_srgb8_f32` (4 channels) routes through this path.
    common::bench_levels(&mut g, "", |b, level| {
        b.iter(|| k::load_srgb8_f32_at::<false>(level, black_box(&rgba)).unwrap());
    });
    // BGRA sweep — `load_bgra8_srgb_f32` routes through this path.
    common::bench_levels(&mut g, "_bgra", |b, level| {
        b.iter(|| k::load_srgb8_f32_at::<true>(level, black_box(&rgba)).unwrap());
    });

    // 3 bytes per pixel is not one packed word, so this production path is
    // per-pixel scalar at every level — one row, not a level sweep.
    g.bench_function("bgr8_scalar", |b| {
        b.iter(|| k::load_bgr8_srgb_f32(black_box(&bgr)).unwrap());
    });

    g.finish();
}

fn bench_store(c: &mut Criterion) {
    use ctt::bench_internals as k;
    let buf = make_buffer();

    let mut g = c.benchmark_group(format!("srgb_store_{SIDE}x{SIDE}"));
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    bench_srgb_store(&mut g, &buf);

    // 3 bytes per pixel is not one packed word, so this production path is
    // per-pixel scalar at every level — one row, not a level sweep.
    // `store_bgr8_srgb_f32` only reads lanes 0-2, so the 4-lane `buf` feeds it.
    g.bench_function("bgr8_scalar", |b| {
        b.iter(|| k::store_bgr8_srgb_f32(black_box(&buf)));
    });

    g.finish();
}

/// sRGB store per-level sweep for both channel orders. Emits `_rgba` and
/// `_bgra` variants of `fallback`, `sse4_2`, `avx2` (x86/x86_64) and `neon`
/// (aarch64). The AVX-512 tier is deliberately split into the generic-rsqrt
/// kernel (`avx512_generic_*`) and the `rsqrt14` intrinsic escape
/// (`avx512_escape_*`, x86_64 only) rather than a single `avx512_*` row, since
/// comparing those two is the point of keeping the escape.
fn bench_srgb_store<M: criterion::measurement::Measurement>(
    g: &mut criterion::BenchmarkGroup<'_, M>,
    buf: &Buffer<f32>,
) {
    use ctt::bench_internals::{constructible_levels, store_srgb8_f32_at};

    for (name, level) in constructible_levels() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if name == "avx512" {
            use ctt::bench_internals::store_srgb8_f32_generic_at;
            g.bench_function("avx512_generic_rgba", |b| {
                b.iter(|| store_srgb8_f32_generic_at::<false>(level, black_box(buf)));
            });
            g.bench_function("avx512_generic_bgra", |b| {
                b.iter(|| store_srgb8_f32_generic_at::<true>(level, black_box(buf)));
            });
            #[cfg(target_arch = "x86_64")]
            if let Some(avx512) = level.as_avx512() {
                use ctt::bench_internals::store_srgb8_f32_avx512_escape;
                g.bench_function("avx512_escape_rgba", |b| {
                    b.iter(|| store_srgb8_f32_avx512_escape::<false>(avx512, black_box(buf)));
                });
                g.bench_function("avx512_escape_bgra", |b| {
                    b.iter(|| store_srgb8_f32_avx512_escape::<true>(avx512, black_box(buf)));
                });
            }
            continue;
        }

        g.bench_function(format!("{name}_rgba"), |b| {
            b.iter(|| store_srgb8_f32_at::<false>(level, black_box(buf)));
        });
        g.bench_function(format!("{name}_bgra"), |b| {
            b.iter(|| store_srgb8_f32_at::<true>(level, black_box(buf)));
        });
    }
}

/// Bench an in-place pass: one row per forced level. The pass mutates its input,
/// so every iteration gets a fresh clone via `iter_batched_ref`; the clone
/// happens outside the timed region. The buffer is built once and shared across
/// every row.
fn bench_in_place(c: &mut Criterion, group: &str, at: fn(Level, &mut [[f32; 4]])) {
    let buf = make_buffer();

    let mut g = c.benchmark_group(group);
    g.throughput(Throughput::Elements(PIXEL_COUNT));

    common::bench_levels(&mut g, "", |b, level| {
        b.iter_batched_ref(
            || buf.pixels.clone(),
            |pixels| at(level, pixels),
            BatchSize::LargeInput,
        );
    });

    g.finish();
}

fn bench_oetf_in_place(c: &mut Criterion) {
    use ctt::bench_internals as k;
    bench_in_place(c, "srgb_oetf_in_place", k::srgb_oetf_in_place_f32_at);
}

fn bench_eotf_in_place(c: &mut Criterion) {
    use ctt::bench_internals as k;
    bench_in_place(c, "srgb_eotf_in_place", k::srgb_eotf_in_place_f32_at);
}

criterion_group!(
    benches,
    bench_load,
    bench_store,
    bench_oetf_in_place,
    bench_eotf_in_place
);
criterion_main!(benches);
