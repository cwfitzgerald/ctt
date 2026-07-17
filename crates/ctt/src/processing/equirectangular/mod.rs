//! Equirectangular (lat-long) → cubemap projection.
//!
//! The projection is a gather: every cubemap texel maps to a direction on the
//! sphere, the direction to a coordinate in the equirectangular source, and the
//! source is sampled there with anisotropic mip filtering. Face order and
//! orientation follow the Vulkan/KTX2 cube map face selection rules
//! (+X, -X, +Y, -Y, +Z, -Z; s right, t down). The equirectangular convention is:
//! image center faces +Z, u increases toward +X, the top row is +Y (up).
//!
//! Filtering: the source footprint of an output texel is strongly
//! anisotropic near the poles (a long horizontal ellipse in the equirectangular),
//! so an isotropic trilinear lookup either sparkles (LOD from the minor
//! axis) or over-blurs latitudes together (LOD from the major axis).
//! Instead, each texel takes `N` trilinear taps marching along the major
//! gradient axis with the LOD chosen from the minor axis — the same scheme
//! GPU anisotropic filtering uses. `N` is a power of two capped at
//! `2^MAX_ANISO_LOG2`; footprints longer than the cap get the residual folded
//! back into the LOD, which over-blurs only a few texels at the very center
//! of the ±Y faces.
//!
//! Every kernel (serial and SIMD) computes the same algorithm; the SIMD
//! paths approximate `atan2` and `log2` with polynomials whose error is far
//! below one source texel (see `ATAN_COEFFS` / `LOG2_COEFFS`).

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

use std::f32::consts::PI;

use crate::error::{Error, Result};

use super::buffer::Buffer;
use super::dispatch::dispatch_simd;
use super::mipmap::{self, MipmapFilter};

/// Cap on anisotropic taps per output texel: `2^MAX_ANISO_LOG2 = 16`.
pub(crate) const MAX_ANISO_LOG2: u32 = 4;

/// Rows per parallel work item.
const BAND_ROWS: u32 = 16;

/// Largest accepted face size. 2^15 keeps every index comfortably inside
/// `i32`/`f32` exact-integer range.
const MAX_FACE_SIZE: u32 = 1 << 15;

/// Odd polynomial for `atan(t)`, `t ∈ [0, 1]`: `t * poly(t²)`.
/// Max error ~1.4e-7 rad; at an 8k source that is ~2e-4 texels.
pub(crate) const ATAN_COEFFS: [f32; 8] = [
    9.999_994_4e-1,
    -3.333_010_7e-1,
    1.994_851e-1,
    -1.391_580_2e-1,
    9.656_256_5e-2,
    -5.606_317_7e-2,
    2.194_661e-2,
    -4.073_309_5e-3,
];

/// Cap on gradient magnitude (mip-0 texels per output texel). Footprints
/// beyond a billion texels are indistinguishable; the clamp keeps the
/// singular-value math below f32 overflow.
pub(crate) const GRAD_MAX: f32 = 1.0e9;

/// Relative floor for the major-axis direction norm, `2^-60`. Directions
/// smaller than `smax² · 2^-60` are degenerate (isotropic footprint) and
/// must scale to a zero step instead of NaN.
pub(crate) const DIR_NORM_GUARD: f32 = 1.0 / (1u64 << 60) as f32;

/// `log2(m)`, `m ∈ [1, 2)`, as `poly(m - 1)`. Max error ~2e-4 — only used
/// for mip LOD selection.
pub(crate) const LOG2_COEFFS: [f32; 5] = [
    2.042_570_2e-4,
    1.436_097_1,
    -6.695_125e-1,
    3.122_116e-1,
    -7.914_958e-2,
];

/// Which world axis the panorama center (`u = 0.5`) faces.
///
/// Combined with [`EquirectangularOrientation::mirror`] this spans the
/// conventions in common use; see [`EquirectangularOrientation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EquirectangularFront {
    #[default]
    PosZ,
    NegZ,
    PosX,
    NegX,
}

/// Panorama orientation convention.
///
/// Unmirrored, longitude (`u`) increases along the `+Z → +X → -Z → -X`
/// yaw cycle starting from `front`; `mirror` reverses it. Verified
/// tool conventions:
///
/// - default (`PosZ`, unmirrored): `u = 0.5 + atan2(x, z)/2π` — matches
///   Filament's `cmgen`.
/// - `PosX` + `mirror`: `u = 0.5 + atan2(z, x)/2π` — matches three.js
///   (`equirectangularUv`) and the Khronos glTF-IBL-Sampler (`dirToUV`).
///
/// The top image row is always up (+Y).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EquirectangularOrientation {
    pub front: EquirectangularFront,
    pub mirror: bool,
}

impl EquirectangularOrientation {
    /// World-to-convention transform: a yaw taking `front` to +Z, then an
    /// optional X mirror. Feeding transformed directions to the base
    /// mapping (`u = 0.5 + atan2(x, z)/2π`) realizes the convention.
    fn apply(self, v: [f32; 3]) -> [f32; 3] {
        let [x, y, z] = v;
        let [x, y, z] = match self.front {
            EquirectangularFront::PosZ => [x, y, z],
            EquirectangularFront::NegZ => [-x, y, -z],
            EquirectangularFront::PosX => [-z, y, x],
            EquirectangularFront::NegX => [z, y, -x],
        };
        if self.mirror { [-x, y, z] } else { [x, y, z] }
    }
}

/// One cube face: `dir(a, b) = axis + a·u + b·v` with `a, b ∈ [-1, 1]`,
/// `a` increasing rightward (s) and `b` downward (t) in the face image.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FaceBasis {
    pub axis: [f32; 3],
    pub u: [f32; 3],
    pub v: [f32; 3],
}

/// Vulkan cube map face selection, inverted: +X, -X, +Y, -Y, +Z, -Z.
pub(crate) const FACE_BASES: [FaceBasis; 6] = [
    FaceBasis {
        axis: [1.0, 0.0, 0.0],
        u: [0.0, 0.0, -1.0],
        v: [0.0, -1.0, 0.0],
    },
    FaceBasis {
        axis: [-1.0, 0.0, 0.0],
        u: [0.0, 0.0, 1.0],
        v: [0.0, -1.0, 0.0],
    },
    FaceBasis {
        axis: [0.0, 1.0, 0.0],
        u: [1.0, 0.0, 0.0],
        v: [0.0, 0.0, 1.0],
    },
    FaceBasis {
        axis: [0.0, -1.0, 0.0],
        u: [1.0, 0.0, 0.0],
        v: [0.0, 0.0, -1.0],
    },
    FaceBasis {
        axis: [0.0, 0.0, 1.0],
        u: [1.0, 0.0, 0.0],
        v: [0.0, -1.0, 0.0],
    },
    FaceBasis {
        axis: [0.0, 0.0, -1.0],
        u: [-1.0, 0.0, 0.0],
        v: [0.0, -1.0, 0.0],
    },
];

/// Prefiltered mip pyramid over an equirectangular source. Built once, shared by
/// every face and every SIMD tier.
#[derive(Debug, Clone)]
pub struct EquirectangularPyramid {
    levels: Vec<Buffer<f32>>,
}

impl EquirectangularPyramid {
    /// Build the full triangle-filtered mip chain for `src`.
    pub fn new(src: Buffer<f32>) -> Result<Self> {
        profiling::scope!("EquirectangularPyramid::new");
        if src.width == 0 || src.height == 0 {
            return Err(Error::InvalidDimensions(
                "equirectangular source must be non-empty".into(),
            ));
        }
        if src.pixels.len() != src.width as usize * src.height as usize {
            return Err(Error::DataLengthMismatch {
                expected: src.width as usize * src.height as usize,
                actual: src.pixels.len(),
            });
        }
        let levels = mipmap::complete(vec![src], MipmapFilter::Triangle, None)?;
        Ok(Self { levels })
    }

    pub fn levels(&self) -> &[Buffer<f32>] {
        &self.levels
    }

    pub fn width(&self) -> u32 {
        self.levels[0].width
    }

    pub fn height(&self) -> u32 {
        self.levels[0].height
    }

    /// Default face size for this source: a quarter of the equirectangular width
    /// (matching sampling rates at the equator), at least 1.
    pub fn default_face_size(&self) -> u32 {
        (self.width() / 4).max(1)
    }
}

/// Per-texel sample command produced by the coordinate phase and consumed by
/// the fetch phase. Coordinates and steps are in normalized source units.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TexelCmd {
    pub u: f32,
    pub v: f32,
    pub lod: f32,
    pub step_u: f32,
    pub step_v: f32,
    /// log2 of the tap count; taps = `1 << taps_log2`, capped at
    /// `2^`[`MAX_ANISO_LOG2`].
    pub taps_log2: u32,
}

/// One prefiltered level, unpacked for the SIMD kernels' narrow phase.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) struct LevelInfo<'a> {
    pub px: &'a [[f32; 4]],
    pub w: i32,
    pub h: i32,
    pub wf: f32,
    pub hf: f32,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn level_table(pyr: &EquirectangularPyramid) -> Vec<LevelInfo<'_>> {
    pyr.levels()
        .iter()
        .map(|l| LevelInfo {
            px: &l.pixels,
            w: l.width as i32,
            h: l.height as i32,
            wf: l.width as f32,
            hf: l.height as f32,
        })
        .collect()
}

/// Wide-phase output for one SIMD lane bundle, spilled to the stack for the
/// narrow phase.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) struct LaneCmds<const N: usize> {
    pub u: [f32; N],
    pub v: [f32; N],
    pub lod: [f32; N],
    pub step_u: [f32; N],
    pub step_v: [f32; N],
    pub taps_log2: [i32; N],
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl<const N: usize> LaneCmds<N> {
    #[inline]
    pub fn zeroed() -> Self {
        Self {
            u: [0.0; N],
            v: [0.0; N],
            lod: [0.0; N],
            step_u: [0.0; N],
            step_v: [0.0; N],
            taps_log2: [0; N],
        }
    }

    /// Reassemble lane `i` as a scalar [`TexelCmd`] for the narrow phase.
    #[inline]
    pub fn get(&self, i: usize) -> TexelCmd {
        TexelCmd {
            u: self.u[i],
            v: self.v[i],
            lod: self.lod[i],
            step_u: self.step_u[i],
            step_v: self.step_v[i],
            taps_log2: self.taps_log2[i] as u32,
        }
    }
}

/// Constants shared by every texel of one projection run.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProjectParams {
    /// Source-texel-per-face-texel gradient scale for u: `w0 / (π·n)`
    /// (`du/dφ = w0/2π` texels × `da = 2/n` per face texel).
    pub ku: f32,
    /// Source-texel-per-face-texel gradient scale for v: `2·h0 / (π·n)`
    /// (`dv/dθ = h0/π` texels × `da = 2/n` per face texel).
    pub kv: f32,
    pub inv_w0: f32,
    pub inv_h0: f32,
    pub max_lod: f32,
}

impl ProjectParams {
    pub(crate) fn new(pyr: &EquirectangularPyramid, face_size: u32) -> Self {
        let w0 = pyr.width() as f32;
        let h0 = pyr.height() as f32;
        let n = face_size as f32;
        Self {
            ku: w0 / (PI * n),
            kv: 2.0 * h0 / (PI * n),
            inv_w0: 1.0 / w0,
            inv_h0: 1.0 / h0,
            max_lod: (pyr.levels.len() - 1) as f32,
        }
    }
}

/// Project an equirectangular pyramid onto the six cube faces (`face_size` ×
/// `face_size` each), dispatching to the best available SIMD kernel.
pub fn project_f32(
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> Result<[Buffer<f32>; 6]> {
    profiling::scope!("equirectangular::project_f32");
    validate_face_size(face_size)?;
    dispatch_simd! {
        x86_64: {
            avx512: Ok(x86::project_f32_avx512(pyr, face_size, orientation)),
            avx2_fma: Ok(x86::project_f32_avx2_fma(pyr, face_size, orientation)),
        },
        aarch64: {
            neon: Ok(neon::project_f32_neon(pyr, face_size, orientation)),
        },
    }
    Ok(project_f32_serial(pyr, face_size, orientation))
}

/// The Vulkan face bases with the panorama orientation baked in: the
/// convention transform is applied to every basis vector once, so the
/// kernels stay orientation-agnostic at zero per-texel cost.
pub(crate) fn oriented_bases(orientation: EquirectangularOrientation) -> [FaceBasis; 6] {
    FACE_BASES.map(|b| FaceBasis {
        axis: orientation.apply(b.axis),
        u: orientation.apply(b.u),
        v: orientation.apply(b.v),
    })
}

fn validate_face_size(face_size: u32) -> Result<()> {
    if face_size == 0 || face_size > MAX_FACE_SIZE {
        return Err(Error::InvalidDimensions(format!(
            "cubemap face size must be in 1..={MAX_FACE_SIZE}, got {face_size}"
        )));
    }
    Ok(())
}

/// Serial reference projection.
///
/// **Not part of the public API.** Exposed so benchmarks and tests can
/// compare the scalar implementation against the SIMD kernels.
#[doc(hidden)]
pub fn project_f32_serial(
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> [Buffer<f32>; 6] {
    profiling::scope!("equirectangular::project_f32_serial");
    project_with(pyr, face_size, orientation, project_band_serial)
}

/// Run one projection: allocate the six faces and hand every
/// [`BAND_ROWS`]-row band of every face to `band`, in parallel with the
/// `rayon` feature.
pub(crate) fn project_with<F>(
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
    band: F,
) -> [Buffer<f32>; 6]
where
    F: Fn(&EquirectangularPyramid, &FaceBasis, u32, u32, &mut [[f32; 4]]) + Sync,
{
    let bases = oriented_bases(orientation);
    let n = face_size;
    let px = n as usize * n as usize;
    let mut faces: [Buffer<f32>; 6] = std::array::from_fn(|_| Buffer {
        pixels: vec![[0.0f32; 4]; px],
        width: n,
        height: n,
    });

    // One flat work item per band of every face: band cost varies ~16×
    // with latitude, so a single list load-balances better than nesting
    // face- and band-level parallelism.
    let band_px = (BAND_ROWS * n) as usize;
    let bands: Vec<(&FaceBasis, u32, &mut [[f32; 4]])> = faces
        .iter_mut()
        .zip(&bases)
        .flat_map(|(buf, basis)| {
            buf.pixels
                .chunks_mut(band_px)
                .enumerate()
                .map(move |(i, chunk)| (basis, i as u32 * BAND_ROWS, chunk))
        })
        .collect();

    let run = |(basis, y_start, chunk): (&FaceBasis, u32, &mut [[f32; 4]])| {
        profiling::scope!("equirectangular_band");
        band(pyr, basis, n, y_start, chunk);
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        bands.into_par_iter().for_each(run);
    }
    #[cfg(not(feature = "rayon"))]
    bands.into_iter().for_each(run);

    faces
}

/// Scalar projection of one row band of one face.
pub(crate) fn project_band_serial(
    pyr: &EquirectangularPyramid,
    basis: &FaceBasis,
    face_size: u32,
    y_start: u32,
    out: &mut [[f32; 4]],
) {
    let params = ProjectParams::new(pyr, face_size);
    let inv_n = 1.0 / face_size as f32;
    for (r, row) in out.chunks_exact_mut(face_size as usize).enumerate() {
        let b = (2 * (y_start + r as u32) + 1) as f32 * inv_n - 1.0;
        for (x, px) in row.iter_mut().enumerate() {
            let a = (2 * x + 1) as f32 * inv_n - 1.0;
            let cmd = map_texel(basis, a, b, &params);
            *px = sample_aniso(&pyr.levels, &cmd);
        }
    }
}

/// Map one face texel (face coords `a, b ∈ [-1, 1]`) to its sample command.
/// This is the scalar reference for the SIMD coordinate phases.
pub(crate) fn map_texel(basis: &FaceBasis, a: f32, b: f32, p: &ProjectParams) -> TexelCmd {
    let [ox, oy, oz] = basis.axis;
    let [ux, uy, uz] = basis.u;
    let [vx, vy, vz] = basis.v;
    let x = ox + a * ux + b * vx;
    let y = oy + a * uy + b * vy;
    let z = oz + a * uz + b * vz;

    let xz2 = x * x + z * z;
    let hyp = xz2.sqrt();
    let len2 = xz2 + y * y;

    // Longitude φ ∈ (-π, π], colatitude θ ∈ [0, π]. Both are
    // scale-invariant in the direction, so it is never normalized.
    let phi = x.atan2(z);
    let theta = hyp.atan2(y);
    let u = phi * (0.5 / PI) + 0.5;
    let v = theta * (1.0 / PI);

    // Analytic gradients of the source coordinate (in mip-0 texels) per
    // output texel, for both face axes. `xz2 → 0` only at the exact ±Y
    // pole, where the footprint is genuinely unbounded; the guard keeps the
    // math finite and the LOD clamp does the rest.
    let inv_xz2 = 1.0 / xz2.max(f32::MIN_POSITIVE);
    let inv_len2 = 1.0 / len2;
    let inv_hyp = hyp * inv_xz2; // hyp/xz² = 1/hyp, reusing the guard
    let dphi_da = (z * ux - x * uz) * inv_xz2;
    let dphi_db = (z * vx - x * vz) * inv_xz2;
    let dtheta_da = (y * (x * ux + z * uz) * inv_hyp - hyp * uy) * inv_len2;
    let dtheta_db = (y * (x * vx + z * vz) * inv_hyp - hyp * vy) * inv_len2;

    let gua = (dphi_da * p.ku).clamp(-GRAD_MAX, GRAD_MAX);
    let gva = (dtheta_da * p.kv).clamp(-GRAD_MAX, GRAD_MAX);
    let gub = (dphi_db * p.ku).clamp(-GRAD_MAX, GRAD_MAX);
    let gvb = (dtheta_db * p.kv).clamp(-GRAD_MAX, GRAD_MAX);

    // The footprint ellipse axes are the singular values of the Jacobian
    // J = [[gua, gub], [gva, gvb]] — the face-axis gradient vectors
    // themselves are *not* the principal axes (near the poles both are
    // huge and nearly parallel, which would fake isotropy). For the 2×2
    // case the singular values and the major eigenvector of J·Jᵀ have a
    // closed form.
    let am = gua * gua + gub * gub;
    let bm = gva * gva + gvb * gvb;
    let cm = gua * gva + gub * gvb;
    let half_diff = 0.5 * (am - bm);
    let disc = (half_diff * half_diff + cm * cm).sqrt();
    let mean = 0.5 * (am + bm);
    let smax2 = mean + disc;
    let smin2 = (mean - disc).max(f32::MIN_POSITIVE);
    // Major-axis direction: eigenvector of [[am, cm], [cm, bm]] for the
    // larger eigenvalue, picking the numerically larger of the two
    // equivalent forms. Degenerate (isotropic) footprints give a ~zero
    // vector, which the guarded normalization below turns into a zero step.
    let (dir_u, dir_v) = if am >= bm {
        (disc + half_diff, cm)
    } else {
        (cm, disc - half_diff)
    };
    let norm2 = dir_u * dir_u + dir_v * dir_v;
    let scale = (smax2 / norm2.max(smax2 * DIR_NORM_GUARD).max(f32::MIN_POSITIVE)).sqrt();
    let gu = dir_u * scale;
    let gv = dir_v * scale;

    let lod_minor = 0.5 * smin2.log2();
    let lod_major = 0.5 * smax2.max(f32::MIN_POSITIVE).log2();
    // Power-of-two tap count covering the anisotropy ratio, capped; the
    // remainder is folded back into the LOD.
    let delta = (lod_major - lod_minor).max(0.0);
    let taps_log2 = (delta.ceil() as u32).min(MAX_ANISO_LOG2);
    let lod = (lod_minor + (delta - taps_log2 as f32).max(0.0)).clamp(0.0, p.max_lod);

    // Tap step along the major gradient, converted to normalized source
    // units and divided among the taps. One normalized unit already spans
    // the whole source, so the clamp only tames the unbounded gradients at
    // the exact ±Y poles (where every longitude is equivalent anyway).
    let inv_taps = 1.0 / (1u32 << taps_log2) as f32;
    TexelCmd {
        u,
        v,
        lod,
        step_u: (gu * p.inv_w0 * inv_taps).clamp(-1.0, 1.0),
        step_v: (gv * p.inv_h0 * inv_taps).clamp(-1.0, 1.0),
        taps_log2,
    }
}

/// Accumulate the anisotropic tap line for one texel.
pub(crate) fn sample_aniso(levels: &[Buffer<f32>], cmd: &TexelCmd) -> [f32; 4] {
    let taps = 1u32 << cmd.taps_log2;
    let center = 0.5 * (taps - 1) as f32;
    let mut acc = [0.0f32; 4];
    for k in 0..taps {
        let t = k as f32 - center;
        let s = sample_trilinear(
            levels,
            cmd.u + t * cmd.step_u,
            cmd.v + t * cmd.step_v,
            cmd.lod,
        );
        for (a, s) in acc.iter_mut().zip(s) {
            *a += s;
        }
    }
    let inv = 1.0 / taps as f32;
    acc.map(|c| c * inv)
}

fn sample_trilinear(levels: &[Buffer<f32>], u: f32, v: f32, lod: f32) -> [f32; 4] {
    let l0 = lod as usize;
    let frac = lod - l0 as f32;
    let c0 = sample_bilinear(&levels[l0], u, v);
    if frac <= 0.0 {
        return c0;
    }
    let c1 = sample_bilinear(&levels[(l0 + 1).min(levels.len() - 1)], u, v);
    std::array::from_fn(|i| c0[i] + (c1[i] - c0[i]) * frac)
}

/// Bilinear tap with horizontal wrap and vertical clamp. `u` may lie
/// outside `[0, 1]` (anisotropic tap lines cross the seam); it wraps.
fn sample_bilinear(level: &Buffer<f32>, u: f32, v: f32) -> [f32; 4] {
    let w = level.width as i32;
    let h = level.height as i32;
    let up = u * level.width as f32 - 0.5;
    let vp = v * level.height as f32 - 0.5;
    let x0f = up.floor();
    let y0f = vp.floor();
    let fx = up - x0f;
    let fy = vp - y0f;
    let x0 = (x0f as i32).rem_euclid(w);
    let x1 = (x0 + 1) % w;
    let y0 = (y0f as i32).clamp(0, h - 1);
    let y1 = (y0f as i32 + 1).clamp(0, h - 1);

    let row0 = y0 as usize * w as usize;
    let row1 = y1 as usize * w as usize;
    let p00 = level.pixels[row0 + x0 as usize];
    let p01 = level.pixels[row0 + x1 as usize];
    let p10 = level.pixels[row1 + x0 as usize];
    let p11 = level.pixels[row1 + x1 as usize];

    std::array::from_fn(|i| {
        let top = p00[i] + (p01[i] - p00[i]) * fx;
        let bot = p10[i] + (p11[i] - p10[i]) * fx;
        top + (bot - top) * fy
    })
}

#[cfg(test)]
mod tests;
