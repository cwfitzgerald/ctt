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
//! The projection runs entirely through the width-generic kernels in
//! [`kernels::equirectangular`](crate::processing::kernels::equirectangular)
//! (scalar `Fallback` included, so every architecture is covered); they
//! approximate `atan2` and `log2` with polynomials whose error is far below one
//! source texel (see [`ATAN_COEFFS`] / [`LOG2_COEFFS`]). This module holds the
//! geometry, the public types, and the band scheduler the kernels drive.

use std::f32::consts::PI;

use crate::error::{Error, Result};
use crate::processing::kernels::equirectangular as kernel;

use super::buffer::Buffer;
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
/// `face_size` each), dispatching to the best available SIMD level (scalar
/// fallback included, so this runs on every architecture).
pub fn project_f32(
    pyr: &EquirectangularPyramid,
    face_size: u32,
    orientation: EquirectangularOrientation,
) -> Result<[Buffer<f32>; 6]> {
    profiling::scope!("equirectangular::project_f32");
    validate_face_size(face_size)?;
    Ok(kernel::project_f32(pyr, face_size, orientation))
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

pub(crate) fn validate_face_size(face_size: u32) -> Result<()> {
    if face_size == 0 || face_size > MAX_FACE_SIZE {
        return Err(Error::InvalidDimensions(format!(
            "cubemap face size must be in 1..={MAX_FACE_SIZE}, got {face_size}"
        )));
    }
    Ok(())
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

/// Test sources and absolute property checks for the projection: every backend
/// must satisfy the same seam-continuity and pole-averaging guarantees, so the
/// fixtures live in one place.
#[cfg(test)]
pub(crate) mod test_support {
    use std::f32::consts::PI;

    use super::Buffer;

    /// Face size for the seam-continuity check.
    pub(crate) const SEAM_FACE_SIZE: u32 = 32;
    /// Face size for the pole-averaging check.
    pub(crate) const POLE_FACE_SIZE: u32 = 64;

    /// Smooth longitude-only pattern; must cross the ±π seam (center of the
    /// −Z face) without a jump.
    pub(crate) fn seam_source() -> Buffer<f32> {
        let (w, h) = (128u32, 64u32);
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for _y in 0..h {
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let phi = (u - 0.5) * 2.0 * PI;
                pixels.push([phi.sin() * 0.5 + 0.5, phi.cos() * 0.5 + 0.5, 0.0, 1.0]);
            }
        }
        Buffer {
            pixels,
            width: w,
            height: h,
        }
    }

    /// Alternating columns — the worst horizontal-aliasing case.
    pub(crate) fn stripe_source() -> Buffer<f32> {
        let (w, h) = (256u32, 128u32);
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for _y in 0..h {
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let stripe = if ((u * 256.0) as u32).is_multiple_of(2) {
                    1.0
                } else {
                    0.0
                };
                pixels.push([stripe, stripe, stripe, 1.0]);
            }
        }
        Buffer {
            pixels,
            width: w,
            height: h,
        }
    }

    /// Adjacent texels along the −Z face equator must not jump across the seam.
    pub(crate) fn assert_neg_z_seam_continuous(faces: &[Buffer<f32>; 6], n: u32, what: &str) {
        let face = &faces[5]; // −Z spans the seam
        let y = n / 2;
        for x in 0..n - 1 {
            let a = face.pixels[(y * n + x) as usize];
            let b = face.pixels[(y * n + x + 1) as usize];
            for c in 0..2 {
                assert!(
                    (a[c] - b[c]).abs() < 0.1,
                    "{what}: seam jump at x={x} chan {c}: {} -> {}",
                    a[c],
                    b[c],
                );
            }
        }
    }

    /// The central quarter of the +Y face looks at the pole cap; each output
    /// texel covers many stripes, so it must converge to their mean.
    pub(crate) fn assert_pole_region_averaged(faces: &[Buffer<f32>; 6], n: u32, what: &str) {
        let face = &faces[2]; // +Y pole cap
        for y in (n / 2 - 8)..(n / 2 + 8) {
            for x in (n / 2 - 8)..(n / 2 + 8) {
                let px = face.pixels[(y * n + x) as usize];
                assert!(
                    (px[0] - 0.5).abs() < 0.15,
                    "{what}: pole texel ({x},{y}) not averaged: {}",
                    px[0],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
