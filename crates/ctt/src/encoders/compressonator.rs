use std::borrow::Cow;

use ctt_compressonator as cmp;

use crate::alpha::AlphaMode;
use crate::encoders::Quality;
use crate::encoders::backend::Encoder;
use crate::error::{Error, Result};
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::FormatExt as _;

/// What the texture data represents. Drives default channel weighting for
/// BC1/BC2/BC3 and the auto branch of [`AmdSettings::bc7_alpha`].
///
/// Format selection (BC4 vs BC5 vs BC7 vs ...) is independent — usage only
/// tunes how the codec weighs error inside the format you've already chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmdUsage {
    /// Generic color texture (default). BC1/BC2/BC3 use ITU-R BT.601
    /// luminance weights `[0.3086, 0.6094, 0.0820]` — the codec spends
    /// most of its bits on green, where the eye is most sensitive.
    #[default]
    Color,
    /// Tangent-space normal data packed into an RGB-shaped format
    /// (BC1/BC2/BC3 normals — uncommon, BC5 is normally preferred).
    /// Channel weights default to uniform `[1.0, 1.0, 1.0]` so X/Y aren't
    /// sacrificed to a luminance prior. No effect on BC4/BC5/BC6H/BC7.
    NormalMap,
    /// Mask / data channels packed into RGB (e.g. metallic+roughness+AO).
    /// Same uniform weighting as `NormalMap` — perceptual luminance bias
    /// is the wrong model when the channels aren't color.
    Data,
}

/// How the BC7 encoder should treat the alpha channel.
///
/// BC7 mode 6 carries RGBA. Modes 4–5 add per-block index rotations for
/// alpha quality. Modes 0–3 are RGB-only. Disabling alpha modes when the
/// asset is fully opaque gives the encoder more RGB-mode partitions to
/// pick from and can measurably improve color fidelity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmdBc7Alpha {
    /// Derive from the surface's [`AlphaMode`]: `Opaque` → behaves like
    /// `Opaque` below; anything else → `Full`.
    #[default]
    Auto,
    /// Tell the encoder there is no meaningful alpha. Compressonator
    /// concentrates on RGB modes and skips the alpha-aware palette
    /// restrictions.
    Opaque,
    /// Alpha is meaningful; let the encoder search the full mode set
    /// without palette restrictions. Highest alpha quality, slowest
    /// encode.
    Full,
    /// Alpha is meaningful but encoding speed matters more — restrict
    /// the color and alpha palettes when both are present. Faster than
    /// `Full`, marginally lower alpha quality.
    Restricted,
}

/// AMD Compressonator encoder settings.
///
/// Compressonator spans BC1 through BC7 with one quality knob per format
/// plus a handful of format-specific options. Most fields below only apply
/// to the format named in the doc comment — the rest are no-ops.
///
/// `usage` is the broadest-impact knob: it picks BT.601 luminance weights
/// for color textures and uniform weights for data/normal textures encoded
/// to BC1/BC2/BC3. Each format-specific Option override beats the
/// usage-derived default when present.
#[derive(Debug, Clone, Copy, Default)]
pub struct AmdSettings {
    /// What the texture represents. Drives default BC1/2/3 channel weights
    /// and the `Auto` branch of [`Self::bc7_alpha`]. See [`AmdUsage`].
    pub usage: AmdUsage,

    /// Explicit RGB error weights for BC1, BC2, and BC3. `None` derives
    /// the weights from `usage`. Values aren't required to sum to 1 — the
    /// codec normalizes them internally; their ratios are what matters.
    /// Ignored for BC4/BC5/BC6H/BC7.
    pub channel_weights: Option<[f32; 3]>,

    /// How BC7 should handle the alpha channel. See [`AmdBc7Alpha`].
    /// Ignored for non-BC7 formats.
    pub bc7_alpha: AmdBc7Alpha,

    /// Restrict which BC7 modes the encoder considers (bit `n` enables
    /// mode `n`, modes 0–7). `None` keeps the codec default (all modes).
    /// Restricting to a single mode is useful for benchmarking but rarely
    /// improves quality. Ignored for non-BC7 formats.
    pub bc7_mode_mask: Option<u8>,

    /// Restrict which BC6H modes the encoder considers (14-mode bitfield,
    /// bit `n` enables mode `n`). `None` keeps the codec default.
    /// Ignored for non-BC6H formats.
    pub bc6h_mode_mask: Option<u32>,
}

pub struct CompressonatorEncoder;

impl Encoder for CompressonatorEncoder {
    type Settings = AmdSettings;

    fn name() -> &'static str {
        "amd"
    }

    fn supported_formats() -> &'static [ktx2::Format] {
        &[
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            ktx2::Format::BC2_UNORM_BLOCK,
            ktx2::Format::BC3_UNORM_BLOCK,
            ktx2::Format::BC4_UNORM_BLOCK,
            ktx2::Format::BC4_SNORM_BLOCK,
            ktx2::Format::BC5_UNORM_BLOCK,
            ktx2::Format::BC5_SNORM_BLOCK,
            ktx2::Format::BC6H_UFLOAT_BLOCK,
            ktx2::Format::BC6H_SFLOAT_BLOCK,
            ktx2::Format::BC7_UNORM_BLOCK,
        ]
    }

    fn required_input_format(format: ktx2::Format, _settings: &AmdSettings) -> ktx2::Format {
        use ktx2::Format as F;
        match format {
            F::BC4_UNORM_BLOCK | F::BC4_SNORM_BLOCK => F::R8_UNORM,
            F::BC5_UNORM_BLOCK | F::BC5_SNORM_BLOCK => F::R8G8_UNORM,
            F::BC6H_UFLOAT_BLOCK | F::BC6H_SFLOAT_BLOCK => F::R16G16B16_SFLOAT,
            _ => F::R8G8B8A8_UNORM,
        }
    }

    fn compress(
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: &AmdSettings,
    ) -> Result<Vec<u8>> {
        let q = quality_to_float(quality);
        let (base, _) = format.normalize();
        // Compressonator's block APIs assume tightly packed input (they derive
        // row stride from width internally). Repack when the surface carries
        // padded rows; this copies the image once, only on the padded path
        // (tight surfaces borrow their data unchanged).
        let tight = surface.tight_data();
        let (data, width, height) = (&*tight, surface.width, surface.height);
        let is_srgb = surface.color_space == ColorSpace::Srgb;
        let weights = settings
            .channel_weights
            .unwrap_or_else(|| default_rgb_weights(settings.usage));

        use ktx2::Format as F;
        match base {
            F::BC1_RGBA_UNORM_BLOCK => {
                let mut opts = cmp::bc1::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                opts.set_channel_weights(weights[0], weights[1], weights[2])
                    .map_err(cmp_err)?;
                opts.set_srgb(is_srgb).map_err(cmp_err)?;
                compress_rows(data, width, height, 4, 8, |src, w, h, dst| {
                    cmp::bc1::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC2_UNORM_BLOCK => {
                let mut opts = cmp::bc2::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                opts.set_channel_weights(weights[0], weights[1], weights[2])
                    .map_err(cmp_err)?;
                opts.set_srgb(is_srgb).map_err(cmp_err)?;
                compress_rows(data, width, height, 4, 16, |src, w, h, dst| {
                    cmp::bc2::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC3_UNORM_BLOCK => {
                let mut opts = cmp::bc3::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                opts.set_channel_weights(weights[0], weights[1], weights[2])
                    .map_err(cmp_err)?;
                opts.set_srgb(is_srgb).map_err(cmp_err)?;
                compress_rows(data, width, height, 4, 16, |src, w, h, dst| {
                    cmp::bc3::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC4_UNORM_BLOCK => {
                let mut opts = cmp::bc4::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                compress_rows(data, width, height, 1, 8, |src, w, h, dst| {
                    cmp::bc4::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC4_SNORM_BLOCK => {
                let mut opts = cmp::bc4::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let src: &[i8] = bytemuck::cast_slice(data);
                compress_rows(src, width, height, 1, 8, |src, w, h, dst| {
                    cmp::bc4s::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC5_UNORM_BLOCK => {
                let mut opts = cmp::bc5::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                compress_rows(data, width, height, 2, 16, |src, w, h, dst| {
                    cmp::bc5::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC5_SNORM_BLOCK => {
                let mut opts = cmp::bc5::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let src: &[i8] = bytemuck::cast_slice(data);
                compress_rows(src, width, height, 2, 16, |src, w, h, dst| {
                    cmp::bc5s::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC6H_UFLOAT_BLOCK => {
                let mut opts = cmp::bc6h::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                if let Some(mask) = settings.bc6h_mode_mask {
                    opts.set_mask(mask).map_err(cmp_err)?;
                }
                let src = u16_slice(data);
                compress_rows(&src, width, height, 3, 16, |src, w, h, dst| {
                    cmp::bc6h::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC6H_SFLOAT_BLOCK => {
                let mut opts = cmp::bc6h::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                opts.set_signed(true).map_err(cmp_err)?;
                if let Some(mask) = settings.bc6h_mode_mask {
                    opts.set_mask(mask).map_err(cmp_err)?;
                }
                let src = u16_slice(data);
                compress_rows(&src, width, height, 3, 16, |src, w, h, dst| {
                    cmp::bc6h::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            F::BC7_UNORM_BLOCK => {
                let mut opts = cmp::bc7::Options::new().map_err(cmp_err)?;
                opts.set_quality(q).map_err(cmp_err)?;
                let (image_needs_alpha, colour_restrict, alpha_restrict) =
                    resolve_bc7_alpha(settings.bc7_alpha, surface.alpha);
                opts.set_alpha_options(image_needs_alpha, colour_restrict, alpha_restrict)
                    .map_err(cmp_err)?;
                if let Some(mask) = settings.bc7_mode_mask {
                    opts.set_mask(mask).map_err(cmp_err)?;
                }
                compress_rows(data, width, height, 4, 16, |src, w, h, dst| {
                    cmp::bc7::compress_blocks_into(src, w, h, &opts, dst)
                })
            }
            _ => unreachable!("format not in supported_formats()"),
        }
    }
}

fn compress_rows<T: Sync>(
    data: &[T],
    width: u32,
    height: u32,
    values_per_pixel: usize,
    bytes_per_block: usize,
    compress: impl Fn(&[T], u32, u32, &mut [u8]) -> std::result::Result<(), cmp::Error> + Send + Sync,
) -> Result<Vec<u8>> {
    let blocks_x = width.div_ceil(4) as usize;
    let blocks_y = height.div_ceil(4) as usize;
    let row_bytes = blocks_x * bytes_per_block;
    let mut output = vec![0u8; blocks_y * row_bytes];

    crate::encoders::parallel::try_for_each_row_chunk(
        &mut output,
        row_bytes,
        |start_row, row_count, dst| {
            let start_y = start_row * 4;
            let chunk_height = (height as usize - start_y).min(row_count * 4) as u32;
            let src_start = start_y * width as usize * values_per_pixel;
            compress(&data[src_start..], width, chunk_height, dst)
        },
    )
    .map_err(cmp_err)?;

    Ok(output)
}

/// BT.601 luminance weights for color content; uniform for data/normals.
fn default_rgb_weights(usage: AmdUsage) -> [f32; 3] {
    match usage {
        AmdUsage::Color => [0.3086, 0.6094, 0.0820],
        AmdUsage::NormalMap | AmdUsage::Data => [1.0, 1.0, 1.0],
    }
}

/// Resolve the `(image_needs_alpha, colour_restrict, alpha_restrict)` tuple
/// passed to Compressonator's BC7 `set_alpha_options`.
fn resolve_bc7_alpha(choice: AmdBc7Alpha, surface_alpha: AlphaMode) -> (bool, bool, bool) {
    let resolved = match choice {
        AmdBc7Alpha::Auto => match surface_alpha {
            AlphaMode::Opaque => AmdBc7Alpha::Opaque,
            _ => AmdBc7Alpha::Full,
        },
        other => other,
    };
    match resolved {
        AmdBc7Alpha::Opaque => (false, false, false),
        AmdBc7Alpha::Full => (true, false, false),
        AmdBc7Alpha::Restricted => (true, true, true),
        // Auto was already resolved above.
        AmdBc7Alpha::Auto => unreachable!(),
    }
}

fn quality_to_float(quality: Quality) -> f32 {
    match quality {
        Quality::UltraFast => 0.01,
        Quality::VeryFast => 0.05,
        Quality::Fast => 0.1,
        Quality::Basic => 0.5,
        Quality::Slow => 0.8,
        Quality::VerySlow => 1.0,
    }
}

fn cmp_err(e: cmp::Error) -> Error {
    Error::Compression(e.to_string())
}

/// Reinterpret packed bytes as `u16` for the BC6H fp16 input. Borrows in the
/// common case; copies into a fresh, correctly aligned buffer only on the rare
/// under-aligned allocation, so `try_cast_slice` never panics.
fn u16_slice(data: &[u8]) -> Cow<'_, [u16]> {
    match bytemuck::try_cast_slice(data) {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(bytemuck::pod_collect_to_vec(data)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    fn solid_red(width: u32, height: u32) -> Surface {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            data.extend_from_slice(&[255, 0, 0, 255]);
        }
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride: width * 4,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    #[test]
    fn bc7_non_aligned_5x5() {
        let surface = solid_red(5, 5);
        // Compressonator BC7 at UltraFast produces R=0 output on non-MSVC
        // toolchains (Linux, macOS). Use Slow so the NPOT coverage is
        // independent of that upstream quirk.
        let out = CompressonatorEncoder::compress(
            &surface,
            ktx2::Format::BC7_UNORM_BLOCK,
            Quality::Slow,
            &AmdSettings::default(),
        )
        .unwrap();
        // 5x5 → 8×8 → 4 blocks × 16 bytes.
        assert_eq!(out.len(), 4 * 16);
        for chunk in out.chunks_exact(16) {
            let block: [u8; 16] = chunk.try_into().unwrap();
            let decoded = ctt_compressonator::bc7::decompress_block(&block).unwrap();
            for pixel in decoded.chunks_exact(4) {
                assert!(pixel[0] > 200, "compressonator BC7 edge R={}", pixel[0]);
            }
        }
    }

    #[test]
    fn bc1_non_aligned_7x3() {
        let surface = solid_red(7, 3);
        let out = CompressonatorEncoder::compress(
            &surface,
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            Quality::UltraFast,
            &AmdSettings::default(),
        )
        .unwrap();
        // 7×3 → 8×4 → 2 blocks × 8 bytes.
        assert_eq!(out.len(), 2 * 8);
        for chunk in out.chunks_exact(8) {
            let block: [u8; 8] = chunk.try_into().unwrap();
            let decoded = ctt_compressonator::bc1::decompress_block(&block).unwrap();
            for pixel in decoded.chunks_exact(4) {
                assert!(pixel[0] > 200, "compressonator BC1 edge R={}", pixel[0]);
            }
        }
    }

    /// RGBA8 surface with a non-uniform pattern at the given row stride;
    /// inter-row padding is a sentinel so stride bugs corrupt the output.
    fn patterned(width: u32, height: u32, stride: u32) -> Surface {
        assert!(stride >= width * 4);
        let mut data = vec![0xABu8; (stride * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let off = (y * stride + x * 4) as usize;
                let v = (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)) & 0xff) as u8;
                data[off..off + 4].copy_from_slice(&[
                    v,
                    v.wrapping_add(50),
                    v.wrapping_add(100),
                    255,
                ]);
            }
        }
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    #[test]
    fn padded_stride_matches_tight() {
        let tight = patterned(8, 8, 8 * 4);
        let padded = patterned(8, 8, 8 * 4 + 16);
        let a = CompressonatorEncoder::compress(
            &tight,
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            Quality::Fast,
            &AmdSettings::default(),
        )
        .unwrap();
        let b = CompressonatorEncoder::compress(
            &padded,
            ktx2::Format::BC1_RGBA_UNORM_BLOCK,
            Quality::Fast,
            &AmdSettings::default(),
        )
        .unwrap();
        assert_eq!(a, b, "padded-stride encode must match tight encode");
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn parallel_matches_single_worker() {
        let surface = patterned(19, 13, 19 * 4 + 12);
        crate::encoders::assert_parallel_matches_serial(|| {
            CompressonatorEncoder::compress(
                &surface,
                ktx2::Format::BC1_RGBA_UNORM_BLOCK,
                Quality::Fast,
                &AmdSettings::default(),
            )
            .unwrap()
        });
    }

    #[test]
    fn bc7_alpha_auto_follows_surface() {
        // Opaque surface → Opaque-equivalent tuple.
        assert_eq!(
            resolve_bc7_alpha(AmdBc7Alpha::Auto, AlphaMode::Opaque),
            (false, false, false),
        );
        // Anything else → Full-equivalent tuple.
        assert_eq!(
            resolve_bc7_alpha(AmdBc7Alpha::Auto, AlphaMode::Straight),
            (true, false, false),
        );
        assert_eq!(
            resolve_bc7_alpha(AmdBc7Alpha::Auto, AlphaMode::Premultiplied),
            (true, false, false),
        );
    }

    #[test]
    fn bc7_alpha_explicit_overrides_surface() {
        // Explicit choice ignores the surface alpha mode.
        assert_eq!(
            resolve_bc7_alpha(AmdBc7Alpha::Opaque, AlphaMode::Straight),
            (false, false, false),
        );
        assert_eq!(
            resolve_bc7_alpha(AmdBc7Alpha::Restricted, AlphaMode::Opaque),
            (true, true, true),
        );
    }
}
