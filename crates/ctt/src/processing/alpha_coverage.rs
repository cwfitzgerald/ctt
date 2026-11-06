//! Mip levels that keep an alpha-tested texture's coverage.
//!
//! Filtered mips average thin opaque features into partial alpha under the
//! alpha-test cutoff, so alpha-tested foliage thins out and vanishes with
//! distance. Each generated level's alpha is scaled so the share of it passing
//! the cutoff matches the base level's (Castaño, "Computing Alpha Mipmaps",
//! 2010).
//!
//! Coverage is measured as the GPU samples the texture, bilinearly between
//! texels: a level's isolated texels just above the cutoff pass only near
//! their centers, so counting texels would overstate it. Each texel is
//! measured at [`SUBSAMPLES`] × [`SUBSAMPLES`] points.

use super::buffer::Buffer;
use super::par_map_infallible;

/// Subsamples per texel along each axis when measuring coverage.
const SUBSAMPLES: u32 = 4;

/// Low bits of an `f32` alpha scale dropped to find its histogram bin: the 10
/// mantissa bits left split each doubling of the scale into 1024 bins.
const BIN_SHIFT: u32 = 13;

/// Doublings of the alpha scale above `cutoff` the histogram covers. Scales
/// past `cutoff * 2^16` only lift alpha below 1/65536 over the cutoff.
const OCTAVES: usize = 16;

const BINS: usize = OCTAVES << (f32::MANTISSA_DIGITS - 1 - BIN_SHIFT);

/// Upper bound on the bands one level's measurement splits into, so the
/// per-band histograms stay small however tall the level is.
const MAX_BANDS: usize = 64;

/// Scale the alpha of `chain[first_generated..]` so each level's coverage of
/// `cutoff` is as near `chain[0]`'s as its alpha values allow.
///
/// Levels are scaled in place after the whole chain is built, so each was
/// filtered from the unscaled level above. Pixels are premultiplied: scaling a
/// texel's alpha scales its color by the same ratio, keeping the straight
/// color.
pub fn preserve(chain: &mut [Buffer<f32>], first_generated: usize, cutoff: f32) {
    profiling::scope!("alpha_coverage::preserve");
    let first_generated = first_generated.max(1);
    if first_generated >= chain.len() {
        return;
    }
    let target = passing_share(&chain[0], cutoff);
    // Nothing passes at the base, so there is no coverage to keep.
    if target == 0.0 {
        return;
    }
    for level in &mut chain[first_generated..] {
        let scale = coverage_scale(level, cutoff, target);
        apply_scale(level, scale);
    }
}

/// Consecutive subsamples along one axis that interpolate the same two
/// texels.
struct Run {
    first: usize,
    second: usize,
    /// Each subsample's weight of `second`.
    weights: Vec<f32>,
}

/// The runs of subsamples along an axis of `size` texels, sampled with clamped
/// edges.
fn runs(size: u32) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for s in 0..size * SUBSAMPLES {
        let at = (s as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
        let floor = at.floor();
        let clamp = |i: f32| (i.max(0.0) as u32).min(size - 1) as usize;
        let (first, second) = (clamp(floor), clamp(floor + 1.0));
        match runs.last_mut() {
            Some(run) if (run.first, run.second) == (first, second) => run.weights.push(at - floor),
            _ => runs.push(Run {
                first,
                second,
                weights: vec![at - floor],
            }),
        }
    }
    runs
}

/// The alpha lane of `level`, clamped to `[0, 1]`; NaN reads as 0.
fn clamped_alpha(level: &Buffer<f32>) -> Vec<f32> {
    level
        .pixels
        .iter()
        .map(|p| {
            if p[3].is_nan() {
                0.0
            } else {
                p[3].clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Split `len` row runs into contiguous bands for parallel measurement.
fn bands(len: usize) -> Vec<std::ops::Range<usize>> {
    let size = len.div_ceil(MAX_BANDS).max(1);
    (0..len)
        .step_by(size)
        .map(|start| start..(start + size).min(len))
        .collect()
}

/// A cell between four texels: the subsamples of one row run and one column
/// run, which all interpolate the same texels.
struct Cell<'a> {
    /// Corner alphas: top left, top right, bottom left, bottom right.
    corners: [f32; 4],
    /// Each subsample row's weight of the bottom corners.
    ys: &'a [f32],
    /// Each subsample column's weight of the right corners.
    xs: &'a [f32],
}

impl Cell<'_> {
    fn subsamples(&self) -> usize {
        self.ys.len() * self.xs.len()
    }

    fn min_max(&self) -> (f32, f32) {
        let [a, b, c, d] = self.corners;
        (a.min(b).min(c.min(d)), a.max(b).max(c.max(d)))
    }

    /// The four bilinear taps of each subsample as `(weight, alpha)` pairs.
    fn taps(&self) -> impl Iterator<Item = [(f32, f32); 4]> + '_ {
        let [tl, tr, bl, br] = self.corners;
        self.ys.iter().flat_map(move |&ty| {
            self.xs.iter().map(move |&tx| {
                [
                    ((1.0 - tx) * (1.0 - ty), tl),
                    (tx * (1.0 - ty), tr),
                    ((1.0 - tx) * ty, bl),
                    (tx * ty, br),
                ]
            })
        })
    }
}

/// Measure `level` in parallel bands of cells: `measure` folds each cell into
/// its band's accumulator, starting from `init`. Returns each band's result
/// and the level's subsample count.
fn measure_cells<T: Send>(
    level: &Buffer<f32>,
    init: impl Fn() -> T + Sync + Send,
    measure: impl Fn(&mut T, Cell<'_>) + Sync + Send,
) -> (Vec<T>, usize) {
    let alpha = clamped_alpha(level);
    let width = level.width as usize;
    let (rows, columns) = (runs(level.height), runs(level.width));
    let bands = par_map_infallible(bands(rows.len()), |band| {
        let mut acc = init();
        for row in &rows[band] {
            let (top, bottom) = (row.first * width, row.second * width);
            for column in &columns {
                let (left, right) = (column.first, column.second);
                let cell = Cell {
                    corners: [
                        alpha[top + left],
                        alpha[top + right],
                        alpha[bottom + left],
                        alpha[bottom + right],
                    ],
                    ys: &row.weights,
                    xs: &column.weights,
                };
                measure(&mut acc, cell);
            }
        }
        acc
    });
    let subsamples = (level.width * SUBSAMPLES) as usize * (level.height * SUBSAMPLES) as usize;
    (bands, subsamples)
}

/// The share of `level`'s subsamples whose bilinear alpha reaches `cutoff`.
fn passing_share(level: &Buffer<f32>, cutoff: f32) -> f32 {
    let (counts, subsamples) = measure_cells(
        level,
        || 0usize,
        |passing, cell| {
            // Bilinear values lie between the corners.
            let (min, max) = cell.min_max();
            if min >= cutoff {
                *passing += cell.subsamples();
            } else if max >= cutoff {
                *passing += cell
                    .taps()
                    .filter(|taps| taps.iter().map(|&(w, a)| w * a).sum::<f32>() >= cutoff)
                    .count();
            }
        },
    );
    counts.into_iter().sum::<usize>() as f32 / subsamples as f32
}

/// The least alpha scale `s` at which `Σ w · min(a · s, 1)` over `taps`
/// reaches `cutoff`, or infinity when no scale does. Weights sum to 1 and
/// alphas lie in `[0, 1]`, so the result is at least `cutoff`.
///
/// Taps saturate at 1 in order of decreasing alpha; between saturations the
/// sum is linear in `s`, so the crossing is solved segment by segment.
fn pass_threshold(mut taps: [(f32, f32); 4], cutoff: f32) -> f32 {
    // A sorting network, by decreasing alpha.
    for (i, j) in [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)] {
        if taps[i].1 < taps[j].1 {
            taps.swap(i, j);
        }
    }
    // The sum is `saturated + slope * s` until the next tap saturates.
    let mut saturated = 0.0f32;
    let mut slope: f32 = taps.iter().map(|&(w, a)| w * a).sum();
    for &(w, a) in &taps {
        if a <= 0.0 {
            break;
        }
        // `saturated` stays under `cutoff` here, so a sum reaching it has a
        // positive slope.
        if saturated + slope / a >= cutoff {
            return (cutoff - saturated) / slope;
        }
        saturated += w;
        slope -= w * a;
    }
    f32::INFINITY
}

/// The histogram bin of pass threshold `t`, or `None` past the last bin.
///
/// Positive floats order as their bits do, and each doubling spans the same
/// number of bit patterns, so bins counted in bits from `cutoff` are spaced
/// evenly on a log scale.
fn bin(t: f32, cutoff: f32) -> Option<usize> {
    let bin = (t.to_bits().saturating_sub(cutoff.to_bits()) >> BIN_SHIFT) as usize;
    (bin < BINS).then_some(bin)
}

/// The scale at the upper edge of bin `bin`: every threshold in bins up to and
/// including `bin` passes at it.
fn bin_scale(bin: usize, cutoff: f32) -> f32 {
    f32::from_bits(cutoff.to_bits() + ((bin as u32 + 1) << BIN_SHIFT))
}

/// The alpha scale that brings the share of `level` passing `cutoff` nearest
/// `target`.
///
/// Every subsample passes from its own threshold scale up, so coverage at a
/// scale is the share of thresholds at or below it. Thresholds are binned on
/// a log scale finer than 8-bit alpha resolves, and coverage only changes in
/// steps, so of the bins either side of the step that crosses `target` the
/// nearer wins.
fn coverage_scale(level: &Buffer<f32>, cutoff: f32, target: f32) -> f32 {
    let (histograms, subsamples) = measure_cells(
        level,
        || vec![0u32; BINS],
        |histogram, cell| {
            let (min, max) = cell.min_max();
            if min == max {
                // Uniform: every subsample shares one threshold, infinite when
                // transparent.
                if let Some(bin) = bin(cutoff / max, cutoff) {
                    histogram[bin] += cell.subsamples() as u32;
                }
            } else {
                for taps in cell.taps() {
                    if let Some(bin) = bin(pass_threshold(taps, cutoff), cutoff) {
                        histogram[bin] += 1;
                    }
                }
            }
        },
    );
    let mut histogram = vec![0u64; BINS];
    for band in histograms {
        for (total, count) in histogram.iter_mut().zip(band) {
            *total += u64::from(count);
        }
    }

    let wanted = f64::from(target) * subsamples as f64;
    let mut through = 0u64;
    let Some(crossing) = histogram.iter().position(|&count| {
        through += count;
        through as f64 >= wanted
    }) else {
        // Even the largest scale falls short of the target.
        return bin_scale(BINS - 1, cutoff);
    };
    // `through` now counts up to and including the crossing bin. Ties keep
    // the higher coverage.
    let below = through - histogram[crossing];
    if crossing > 0 && wanted - (below as f64) < through as f64 - wanted {
        return bin_scale(crossing - 1, cutoff);
    }
    // Coverage holds from the upper edge of the crossing bin up to the lower
    // edge of the next occupied one. Of those scales, the one nearest 1
    // changes alpha least: an opaque level keeps its alpha.
    let low = bin_scale(crossing, cutoff);
    let high = histogram[crossing + 1..]
        .iter()
        .position(|&count| count > 0)
        .map_or(f32::INFINITY, |next| bin_scale(crossing + next, cutoff));
    1.0f32.clamp(low, high)
}

/// Scale `level`'s alpha by `scale`, saturating at 1, and its premultiplied
/// color by the same ratio.
fn apply_scale(level: &mut Buffer<f32>, scale: f32) {
    if scale == 1.0 {
        return;
    }
    for pixel in &mut level.pixels {
        let alpha = pixel[3];
        if alpha > 0.0 {
            let scaled = (alpha * scale).min(1.0);
            let ratio = scaled / alpha;
            for c in &mut pixel[..3] {
                *c *= ratio;
            }
            pixel[3] = scaled;
        }
    }
}

/// Fixtures shared with the `convert` tests.
#[cfg(test)]
pub(crate) mod test_support {
    /// Whether texel `i` of a sparse leaf mask is opaque: an eighth of texels,
    /// scattered so neighbours land independently (murmur3's finaliser).
    pub(crate) fn is_leaf(i: u32) -> bool {
        let mut hash = i.wrapping_mul(0x9E37_79B9);
        hash = (hash ^ (hash >> 16)).wrapping_mul(0x85EB_CA6B);
        hash = (hash ^ (hash >> 13)).wrapping_mul(0xC2B2_AE35);
        hash ^= hash >> 16;
        hash.is_multiple_of(8)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::is_leaf;
    use super::*;
    use crate::processing::mipmap::{MipmapFilter, complete};

    /// A `size` × `size` level of sparse leaves: filtered mips average them
    /// towards 0.125 alpha, under a 0.5 cutoff.
    fn sparse_leaves(size: u32) -> Buffer<f32> {
        let pixels = (0..size * size)
            .map(|i| {
                let a = if is_leaf(i) { 1.0 } else { 0.0 };
                [0.2 * a, 0.6 * a, 0.1 * a, a]
            })
            .collect();
        Buffer {
            pixels,
            width: size,
            height: size,
        }
    }

    fn chain(base: Buffer<f32>) -> Vec<Buffer<f32>> {
        complete(vec![base], MipmapFilter::Triangle, None).unwrap()
    }

    #[test]
    fn levels_keep_the_base_coverage() {
        // Mid-chain levels of sparse atlases end as isolated texels just over
        // the cutoff, which bilinear sampling mostly blends under it: matching
        // texel counts let drawn coverage halve by the 8x8 level.
        let mut chain = chain(sparse_leaves(512));
        let base = passing_share(&chain[0], 0.5);
        // Unscaled, the leaves fade under the cutoff.
        assert!(passing_share(&chain[6], 0.5) < base / 2.0);
        preserve(&mut chain, 1, 0.5);
        // Within 5% of the base down to 4x4, below which levels have too few
        // subsamples.
        for level in &chain[2..=7] {
            let got = passing_share(level, 0.5);
            assert!(
                (got - base).abs() < 0.05 * base,
                "{}x{}: {got} vs {base}",
                level.width,
                level.height
            );
        }
    }

    #[test]
    fn base_and_supplied_levels_are_untouched() {
        let mut chain = chain(sparse_leaves(32));
        let before = chain.clone();
        preserve(&mut chain, 2, 0.5);
        for level in 0..2 {
            assert_eq!(chain[level].pixels, before[level].pixels, "level {level}");
        }
        assert_ne!(chain[2].pixels, before[2].pixels);
    }

    #[test]
    fn scaling_keeps_straight_color() {
        let mut level = Buffer {
            pixels: vec![[0.1, 0.2, 0.05, 0.25], [0.4, 0.2, 0.1, 0.8], [0.0; 4]],
            width: 3,
            height: 1,
        };
        let straight = |p: [f32; 4]| [p[0] / p[3], p[1] / p[3], p[2] / p[3]];
        let before = level.pixels.clone();
        apply_scale(&mut level, 2.0);
        assert_eq!(level.pixels[0][3], 0.5);
        // Saturates at 1.
        assert_eq!(level.pixels[1][3], 1.0);
        assert_eq!(level.pixels[2], [0.0; 4]);
        for (&scaled, &unscaled) in level.pixels.iter().zip(&before).take(2) {
            let (got, want) = (straight(scaled), straight(unscaled));
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}");
            }
        }
    }

    #[test]
    fn transparent_base_is_left_alone() {
        let mut chain = chain(Buffer {
            pixels: vec![[0.0, 0.0, 0.0, 0.2]; 16 * 16],
            width: 16,
            height: 16,
        });
        let before = chain.clone();
        preserve(&mut chain, 1, 0.5);
        for (got, want) in chain.iter().zip(&before) {
            assert_eq!(got.pixels, want.pixels);
        }
    }

    #[test]
    fn opaque_odd_sizes_keep_their_alpha() {
        let mut chain = chain(Buffer {
            pixels: vec![[0.5, 0.5, 0.5, 1.0]; 5 * 3],
            width: 5,
            height: 3,
        });
        preserve(&mut chain, 1, 0.5);
        for level in &chain {
            assert_eq!(passing_share(level, 0.5), 1.0);
            // Opaque levels already match the base, so keep their alpha.
            assert!(level.pixels.iter().all(|p| p[3] == 1.0));
        }
    }

    #[test]
    fn cells_match_sampling_each_subsample() {
        // Odd sizes exercise the clamped edge runs.
        let (width, height) = (7u32, 5u32);
        let pixels: Vec<[f32; 4]> = (0..width * height)
            .map(|i| [0.0, 0.0, 0.0, ((i * 37) % 11) as f32 / 10.0])
            .collect();
        let level = Buffer {
            pixels,
            width,
            height,
        };
        let alpha = |x: i64, y: i64| {
            let (x, y) = (x.clamp(0, width as i64 - 1), y.clamp(0, height as i64 - 1));
            level.pixels[(y * width as i64 + x) as usize][3]
        };
        for cutoff in [0.15, 0.5, 0.85] {
            let mut passing = 0;
            for sy in 0..height * SUBSAMPLES {
                for sx in 0..width * SUBSAMPLES {
                    let at = |s: u32| (s as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                    let (x, y) = (at(sx), at(sy));
                    let (x0, y0) = (x.floor(), y.floor());
                    let (tx, ty) = (x - x0, y - y0);
                    let (x0, y0) = (x0 as i64, y0 as i64);
                    let top = alpha(x0, y0) * (1.0 - tx) + alpha(x0 + 1, y0) * tx;
                    let bottom = alpha(x0, y0 + 1) * (1.0 - tx) + alpha(x0 + 1, y0 + 1) * tx;
                    if top * (1.0 - ty) + bottom * ty >= cutoff - 1e-6 {
                        passing += 1;
                    }
                }
            }
            let want = passing as f32 / (width * height * SUBSAMPLES * SUBSAMPLES) as f32;
            let got = passing_share(&level, cutoff);
            assert!((got - want).abs() < 0.01, "{cutoff}: {got} vs {want}");
        }
    }

    #[test]
    fn threshold_matches_a_direct_evaluation() {
        let cases = [
            [(0.25, 1.0), (0.25, 0.5), (0.25, 0.1), (0.25, 0.0)],
            [(0.5, 0.3), (0.5, 0.3), (0.0, 1.0), (0.0, 0.0)],
            [(0.1, 0.9), (0.2, 0.05), (0.3, 0.6), (0.4, 0.2)],
            [(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        ];
        for taps in cases {
            for cutoff in [0.1, 0.5, 0.9] {
                let value =
                    |s: f32| -> f32 { taps.iter().map(|&(w, a)| w * (a * s).min(1.0)).sum() };
                let t = pass_threshold(taps, cutoff);
                if t.is_finite() {
                    assert!((value(t) - cutoff).abs() < 1e-5, "{taps:?} @ {cutoff}: {t}");
                    assert!(
                        value(t * 0.999) < cutoff,
                        "{taps:?} @ {cutoff}: {t} not least"
                    );
                } else {
                    assert!(value(1e9) < cutoff, "{taps:?} @ {cutoff}: should pass");
                }
            }
        }
    }

    #[test]
    fn bins_bracket_their_thresholds() {
        for cutoff in [0.05f32, 0.5, 1.0] {
            for i in 0..2000 {
                let t = cutoff * (1.0 + i as f32 * 0.37).powf(1.3);
                let Some(bin) = bin(t, cutoff) else {
                    assert!(t > cutoff * 65535.0, "{t} binless at {cutoff}");
                    continue;
                };
                assert!(t < bin_scale(bin, cutoff), "{t} above bin {bin}");
                if bin > 0 {
                    assert!(t >= bin_scale(bin - 1, cutoff), "{t} below bin {bin}");
                }
            }
        }
    }

    #[test]
    fn nan_alpha_reads_as_transparent() {
        let level = Buffer {
            pixels: vec![[0.0, 0.0, 0.0, f32::NAN], [0.0, 0.0, 0.0, 2.0]],
            width: 2,
            height: 1,
        };
        assert_eq!(clamped_alpha(&level), [0.0, 1.0]);
    }
}
