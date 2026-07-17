//! Internal processing pipeline: load → (swizzle/mipmap/alpha) → store → encode.
//!
//! The public entry point is [`crate::convert::convert`]. Everything here is
//! a collection of plain functions over [`Buffer`][buffer::Buffer] and
//! [`Surface`][crate::surface::Surface].

pub(crate) mod alpha;
pub(crate) mod buffer;
pub(crate) mod curve_pass;
pub(crate) mod dispatch;
pub(crate) mod encode;
pub(crate) mod load;
pub(crate) mod load_kernels;
pub(crate) mod mipmap;
pub(crate) mod passthrough;
pub(crate) mod store;
pub(crate) mod store_kernels;
pub(crate) mod swizzle;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

pub use buffer::{Buffer, Variant};
pub use mipmap::MipmapFilter;
pub use swizzle::{Swizzle, SwizzleChannel};

use crate::error::Result;
use crate::format_kind::{FormatFamily, classify};
use crate::surface::{ColorSpace, Image};

/// Output of [`crate::convert::convert`].
#[derive(Debug)]
pub enum PipelineOutput {
    /// Encoded file bytes (DDS or KTX2).
    Encoded(Vec<u8>),
    /// Raw image (when the caller requested [`crate::convert::Container::Raw`]).
    Raw(Image),
}

/// Map a fallible transform over owned items, in parallel with the `rayon`
/// feature. Output order matches input order.
pub(crate) fn par_map<T: Send, U: Send>(
    items: Vec<T>,
    f: impl Fn(T) -> Result<U> + Sync + Send,
) -> Result<Vec<U>> {
    #[cfg(feature = "rayon")]
    let mapped = {
        use rayon::prelude::*;
        items.into_par_iter().map(f).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let mapped = items.into_iter().map(f).collect();
    mapped
}

/// Map a fallible transform over every item of a nested list, in parallel
/// with the `rayon` feature, preserving the outer structure.
pub(crate) fn map_nested<T: Send, U: Send>(
    nested: Vec<Vec<T>>,
    f: impl Fn(T) -> Result<U> + Sync + Send,
) -> Result<Vec<Vec<U>>> {
    let lens: Vec<usize> = nested.iter().map(Vec::len).collect();
    let flat = par_map(nested.into_iter().flatten().collect(), f)?;
    let mut flat = flat.into_iter();
    Ok(lens
        .into_iter()
        .map(|len| flat.by_ref().take(len).collect())
        .collect())
}

/// Pick the internal representation to use for a run, based on input + target formats.
///
/// Returns `None` if the families are incompatible (integer source with float target, etc).
pub fn pick_variant(input: ktx2::Format, target: ktx2::Format) -> Option<Variant> {
    use ktx2::Format as F;

    // Compressed targets always route through the float pipeline; they drop
    // out of classification so handle them up front. sRGB-ness doesn't change
    // the integer-vs-float routing, so pass Linear.
    let input_info = classify(input, ColorSpace::Linear);
    let target_info = classify(target, ColorSpace::Linear);

    // R64 dominates the variant choice.
    let has_r64_int = matches!(
        input,
        F::R64_UINT
            | F::R64_SINT
            | F::R64G64_UINT
            | F::R64G64_SINT
            | F::R64G64B64_UINT
            | F::R64G64B64_SINT
            | F::R64G64B64A64_UINT
            | F::R64G64B64A64_SINT,
    ) || matches!(
        target,
        F::R64_UINT
            | F::R64_SINT
            | F::R64G64_UINT
            | F::R64G64_SINT
            | F::R64G64B64_UINT
            | F::R64G64B64_SINT
            | F::R64G64B64A64_UINT
            | F::R64G64B64A64_SINT,
    );
    let has_r64_float = matches!(
        input,
        F::R64_SFLOAT | F::R64G64_SFLOAT | F::R64G64B64_SFLOAT | F::R64G64B64A64_SFLOAT,
    ) || matches!(
        target,
        F::R64_SFLOAT | F::R64G64_SFLOAT | F::R64G64B64_SFLOAT | F::R64G64B64A64_SFLOAT,
    );

    if has_r64_int {
        return Some(Variant::U64);
    }
    if has_r64_float {
        return Some(Variant::F64);
    }

    // Integer family anywhere → integer pipeline. Family mismatches are a
    // separate error; the caller enforces it.
    let input_family = input_info.map(|i| i.family);
    let target_family = target_info.map(|i| i.family);

    if matches!(
        input_family,
        Some(FormatFamily::Uint) | Some(FormatFamily::Sint)
    ) || matches!(
        target_family,
        Some(FormatFamily::Uint) | Some(FormatFamily::Sint)
    ) {
        return Some(Variant::U32);
    }

    Some(Variant::F32)
}

/// Check whether an input and target format are in compatible families.
///
/// The new pipeline does not bridge integer ↔ float; mismatches error at
/// settings resolution.
pub fn families_compatible(input: ktx2::Format, target: ktx2::Format) -> bool {
    // sRGB-ness doesn't change the integer-vs-float decision.
    let i = classify(input, ColorSpace::Linear).map(|i| i.family);
    let t = classify(target, ColorSpace::Linear).map(|i| i.family);
    match (i, t) {
        (Some(a), Some(b)) => a.is_integer() == b.is_integer(),
        // Unknown target (e.g. compressed) — compressed targets are always
        // routed through the float pipeline, so only float-side inputs are valid.
        (Some(a), None) => a.is_float_side(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::{ColorSpace, Surface};

    fn make_surface(data: Vec<u8>, width: u32, height: u32, format: ktx2::Format) -> Surface {
        use crate::vk_format::FormatExt as _;
        let bpp = format.bytes_per_pixel().unwrap() as u32;
        Surface {
            data,
            width,
            height,
            depth: 1,
            stride: width * bpp,
            slice_stride: 0,
            format,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        }
    }

    #[test]
    fn roundtrip_rgba8_unorm_linear_opaque() {
        let pixels = vec![10u8, 20, 30, 40, 200, 150, 100, 50];
        let surface = make_surface(pixels.clone(), 2, 1, ktx2::Format::R8G8B8A8_UNORM);
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(out.data, pixels);
    }

    #[test]
    fn roundtrip_rgba8_srgb_opaque() {
        // Opaque alpha => no premultiply; sRGB roundtrip should be lossless
        // for every u8 value.
        let surface = Surface {
            data: vec![128, 64, 32, 200],
            width: 1,
            height: 1,
            depth: 1,
            stride: 4,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_SRGB,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Opaque,
        };
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::R8G8B8A8_SRGB,
            ColorSpace::Srgb,
            AlphaMode::Opaque,
        )
        .unwrap();
        for i in 0..4 {
            assert_eq!(
                out.data[i], surface.data[i],
                "srgb roundtrip diverged at channel {i}"
            );
        }
    }

    /// Full-chain u8 roundtrip for every byte value on every channel.
    ///
    /// Sweeps the 256-entry u8 domain through the load SIMD approximation,
    /// then back out through the store SIMD approximation, and asserts the
    /// recovered bytes exactly match the input. Both approximations are
    /// individually inside the ±0.5/255 margin; chaining them is a stronger
    /// check that error compounding near the linear/curve threshold stays
    /// within a 1-byte tolerance.
    fn full_chain_srgb_roundtrip(format: ktx2::Format) {
        let mut data = vec![0u8; 256 * 4];
        for b in 0..256usize {
            let base = b * 4;
            data[base] = b as u8;
            data[base + 1] = (255 - b) as u8;
            data[base + 2] = ((b * 7) & 0xff) as u8;
            data[base + 3] = b as u8;
        }
        let surface = Surface {
            data: data.clone(),
            width: 256,
            height: 1,
            depth: 1,
            stride: 256 * 4,
            slice_stride: 0,
            format,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Opaque,
        };
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(buf, format, ColorSpace::Srgb, AlphaMode::Opaque).unwrap();

        let mut mismatches: Vec<(usize, u8, u8)> = Vec::new();
        for (i, (&got, &want)) in out.data.iter().zip(&data).enumerate() {
            if got != want {
                mismatches.push((i, want, got));
            }
        }
        assert!(
            mismatches.is_empty(),
            "{format:?} roundtrip diverged at {} byte(s): {:?}",
            mismatches.len(),
            mismatches
                .iter()
                .take(16)
                .map(|(i, w, g)| format!("pos {i} want {w} got {g}"))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn roundtrip_rgba8_srgb_full_chain() {
        full_chain_srgb_roundtrip(ktx2::Format::R8G8B8A8_SRGB);
    }

    #[test]
    fn roundtrip_bgra8_srgb_full_chain() {
        full_chain_srgb_roundtrip(ktx2::Format::B8G8R8A8_SRGB);
    }

    #[test]
    fn bgra_byte_swap() {
        // BGRA input 0xB 0xG 0xR 0xA (decimal 10,20,30,40) reads as
        // R=30, G=20, B=10, A=40. Store back to RGBA swaps it to (30,20,10,40).
        let surface = make_surface(vec![10u8, 20, 30, 40], 1, 1, ktx2::Format::B8G8R8A8_UNORM);
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(out.data, vec![30, 20, 10, 40]);
    }

    #[test]
    fn rgba_to_r_channel_drop() {
        // 4-channel → 1-channel keeps just R.
        let surface = make_surface(
            vec![100u8, 150, 200, 255],
            1,
            1,
            ktx2::Format::R8G8B8A8_UNORM,
        );
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::R8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(out.data, vec![100]);
    }

    #[test]
    fn r_to_rgba_channel_expansion_fills_alpha() {
        let surface = make_surface(vec![100u8], 1, 1, ktx2::Format::R8_UNORM);
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        // R=100, G=0, B=0, A=255.
        assert_eq!(out.data, vec![100, 0, 0, 255]);
    }

    #[test]
    fn premultiply_straight_roundtrip() {
        let surface = Surface {
            data: vec![200u8, 100, 50, 128],
            width: 1,
            height: 1,
            depth: 1,
            stride: 4,
            slice_stride: 0,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Straight,
        )
        .unwrap();
        // Premul→unpremul roundtrip is within ±1 for low alpha values.
        for i in 0..4 {
            let diff = (out.data[i] as i16 - surface.data[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "channel {i}: {} vs {}",
                out.data[i],
                surface.data[i]
            );
        }
    }

    #[test]
    fn u16_unorm_roundtrip() {
        let pixels: Vec<u8> = vec![0x34, 0x12, 0x78, 0x56];
        let surface = make_surface(pixels.clone(), 1, 1, ktx2::Format::R16G16_UNORM);
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::R16G16_UNORM,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(out.data, pixels);
    }

    #[test]
    fn f32_roundtrip_exact() {
        let mut data = Vec::new();
        for v in &[0.25f32, 0.5, 0.75, 1.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let surface = make_surface(data.clone(), 1, 1, ktx2::Format::R32G32B32A32_SFLOAT);
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::R32G32B32A32_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(out.data, data);
    }

    #[test]
    fn f16_rgba_roundtrip_exact() {
        // RGBA f16 (channels = 4) exercises the bulk fast path that maps
        // 4×f16 directly onto 4×f32 lanes.
        use half::f16;
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.125 - 1.0).collect();
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        let surface = make_surface(data.clone(), 4, 1, ktx2::Format::R16G16B16A16_SFLOAT);
        let buf = load::load_f32(&surface).unwrap();
        // Lanes match what we encoded.
        for (i, pixel) in buf.pixels.iter().enumerate() {
            for c in 0..4 {
                let want = f16::from_f32(values[i * 4 + c]).to_f32();
                assert_eq!(
                    pixel[c], want,
                    "pixel {i} chan {c}: got {} want {want}",
                    pixel[c]
                );
            }
        }
        let out = store::store_f32(
            buf,
            ktx2::Format::R16G16B16A16_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(out.data, data);
    }

    #[test]
    fn f16_rg_roundtrip_with_default_alpha() {
        // 2-channel f16 exercises the scatter path. Missing channels must
        // load as 0 and alpha must default to 1.0.
        use half::f16;
        let values = [0.25f32, -0.5, 1.0, 0.75];
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        let surface = make_surface(data.clone(), 2, 1, ktx2::Format::R16G16_SFLOAT);
        let buf = load::load_f32(&surface).unwrap();
        for (i, pixel) in buf.pixels.iter().enumerate() {
            assert_eq!(pixel[0], f16::from_f32(values[i * 2]).to_f32());
            assert_eq!(pixel[1], f16::from_f32(values[i * 2 + 1]).to_f32());
            assert_eq!(pixel[2], 0.0);
            assert_eq!(pixel[3], 1.0);
        }
        let out = store::store_f32(
            buf,
            ktx2::Format::R16G16_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(out.data, data);
    }

    #[test]
    fn u32_uint_roundtrip() {
        let vals: [u32; 4] = [1, 2, 3, 4];
        let mut data = Vec::new();
        for v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let surface = Surface {
            data: data.clone(),
            width: 1,
            height: 1,
            depth: 1,
            stride: 16,
            slice_stride: 0,
            format: ktx2::Format::R32G32B32A32_UINT,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        };
        let buf = load::load_u32(&surface).unwrap();
        let out =
            store::store_u32(buf, ktx2::Format::R32G32B32A32_UINT, AlphaMode::Opaque).unwrap();
        assert_eq!(out.data, data);
    }

    // ---- Packed 32-bit formats ----

    /// Build a surface from packed 32-bit words (one per pixel), laid out as a
    /// single row.
    fn packed_surface(words: &[u32], format: ktx2::Format) -> Surface {
        let mut data = Vec::with_capacity(words.len() * 4);
        for w in words {
            data.extend_from_slice(&w.to_le_bytes());
        }
        make_surface(data, words.len() as u32, 1, format)
    }

    fn stored_words(surface: &Surface) -> Vec<u32> {
        surface
            .data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn roundtrip_a2b10g10r10_unorm_bit_exact() {
        // UNORM 10-bit and 2-bit codes are a bijection through the pipeline, so
        // every valid word must round-trip bit-exactly.
        let pack = |r: u32, g: u32, b: u32, a: u32| (a << 30) | (b << 20) | (g << 10) | r;
        let mut words = Vec::new();
        for &r in &[0u32, 1, 511, 512, 1022, 1023] {
            for &a in &[0u32, 1, 2, 3] {
                words.push(pack(r, 1023 - r, (r * 3) & 0x3ff, a));
            }
        }
        let surface = packed_surface(&words, ktx2::Format::A2B10G10R10_UNORM_PACK32);
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::A2B10G10R10_UNORM_PACK32,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(stored_words(&out), words);
    }

    #[test]
    fn a2r10g10b10_swaps_r_and_b_vs_a2b() {
        // Same packed word interpreted under both channel orders must yield
        // R and B swapped in the decoded buffer.
        let word = (0b10u32 << 30) | (300 << 20) | (200 << 10) | 100; // A=2, slot2=300, G=200, slot0=100
        let a2b = packed_surface(&[word], ktx2::Format::A2B10G10R10_UNORM_PACK32);
        let a2r = packed_surface(&[word], ktx2::Format::A2R10G10B10_UNORM_PACK32);
        let b_buf = load::load_f32(&a2b).unwrap();
        let r_buf = load::load_f32(&a2r).unwrap();
        // A2B: R=slot0=100, B=slot2=300. A2R: R=slot2=300, B=slot0=100.
        assert!((b_buf.pixels[0][0] - 100.0 / 1023.0).abs() < 1e-6);
        assert!((b_buf.pixels[0][2] - 300.0 / 1023.0).abs() < 1e-6);
        assert!((r_buf.pixels[0][0] - 300.0 / 1023.0).abs() < 1e-6);
        assert!((r_buf.pixels[0][2] - 100.0 / 1023.0).abs() < 1e-6);
    }

    #[test]
    fn roundtrip_a2b10g10r10_uint_bit_exact() {
        let pack = |r: u32, g: u32, b: u32, a: u32| (a << 30) | (b << 20) | (g << 10) | r;
        let words = vec![
            pack(0, 1023, 512, 3),
            pack(1023, 0, 1, 0),
            pack(500, 600, 700, 2),
        ];
        let surface = packed_surface(&words, ktx2::Format::A2B10G10R10_UINT_PACK32);
        let buf = load::load_u32(&surface).unwrap();
        let out = store::store_u32(
            buf,
            ktx2::Format::A2B10G10R10_UINT_PACK32,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(stored_words(&out), words);
    }

    #[test]
    fn roundtrip_a2b10g10r10_sint_bit_exact() {
        // Full signed range (incl. -512) round-trips: store clamps to [-512,511].
        let pack = |r: u32, g: u32, b: u32, a: u32| (a << 30) | (b << 20) | (g << 10) | r;
        let words = vec![
            pack(0x200, 0x1ff, 0, 0b10), // R=-512, G=511, B=0, A=-2
            pack(0x3ff, 1, 0x201, 0b01), // R=-1, G=1, B=-511, A=1
            pack(0, 0x3ff, 0x1ff, 0b11), // R=0, G=-1, B=511, A=-1
        ];
        let surface = packed_surface(&words, ktx2::Format::A2B10G10R10_SINT_PACK32);
        let buf = load::load_u32(&surface).unwrap();
        let out = store::store_u32(
            buf,
            ktx2::Format::A2B10G10R10_SINT_PACK32,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(stored_words(&out), words);
    }

    #[test]
    fn roundtrip_a2b10g10r10_snorm_values() {
        // SNORM value-level round-trip for representable codes (excludes -512,
        // which maps to -1 == code -511).
        let pack = |r: u32, g: u32, b: u32, a: u32| (a << 30) | (b << 20) | (g << 10) | r;
        let words = vec![
            pack(0x1ff, 0x201, 0, 0b01), // R=1.0, G=-1.0, B=0.0, A=1.0
            pack(0, 511, 100, 0),
        ];
        let surface = packed_surface(&words, ktx2::Format::A2B10G10R10_SNORM_PACK32);
        let buf = load::load_f32(&surface).unwrap();
        let out = store::store_f32(
            buf,
            ktx2::Format::A2B10G10R10_SNORM_PACK32,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        assert_eq!(stored_words(&out), words);
    }

    #[test]
    fn roundtrip_e5b9g9r9_values() {
        // Shared-exponent RGB: encode a set of values, decode, and check the
        // relative error is within the format's ~1/512 mantissa resolution.
        let vals = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.5, 0.25],
            [0.1, 0.2, 0.3],
            [10.0, 20.0, 5.0],
            [100.0, 0.01, 1.0],
        ];
        let pixels: Vec<[f32; 4]> = vals.iter().map(|v| [v[0], v[1], v[2], 1.0]).collect();
        let buf = Buffer {
            pixels,
            width: vals.len() as u32,
            height: 1,
        };
        let out = store::store_f32(
            buf,
            ktx2::Format::E5B9G9R9_UFLOAT_PACK32,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        let reload = load::load_f32(&out).unwrap();
        for (got, want) in reload.pixels.iter().zip(&vals) {
            // The mantissa step is set by the largest channel (shared exponent),
            // so accuracy is bounded relative to that, not per-channel.
            let max_c = want.iter().cloned().fold(0.0f32, f32::max);
            for c in 0..3 {
                let diff = (got[c] - want[c]).abs();
                assert!(
                    diff <= max_c / 256.0 + 1e-6,
                    "channel {c}: got {} want {}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    #[test]
    fn roundtrip_b10g11r11_values() {
        let vals = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.5, 0.25],
            [0.1, 0.2, 0.3],
            [12.5, 3.0, 7.0],
        ];
        let pixels: Vec<[f32; 4]> = vals.iter().map(|v| [v[0], v[1], v[2], 1.0]).collect();
        let buf = Buffer {
            pixels,
            width: vals.len() as u32,
            height: 1,
        };
        let out = store::store_f32(
            buf,
            ktx2::Format::B10G11R11_UFLOAT_PACK32,
            ColorSpace::Linear,
            AlphaMode::Opaque,
        )
        .unwrap();
        let reload = load::load_f32(&out).unwrap();
        for (got, want) in reload.pixels.iter().zip(&vals) {
            // R,G have 6 mantissa bits (~1/64), B has 5 (~1/32).
            let tol = [1.0 / 64.0, 1.0 / 64.0, 1.0 / 32.0];
            for c in 0..3 {
                let w = want[c];
                let diff = (got[c] - w).abs();
                assert!(
                    diff <= w.abs() * tol[c] + 1e-6,
                    "channel {c}: got {} want {w}",
                    got[c]
                );
            }
        }
    }

    #[test]
    fn pick_variant_rgba8_to_bc7_is_f32() {
        let v = pick_variant(ktx2::Format::R8G8B8A8_UNORM, ktx2::Format::BC7_UNORM_BLOCK);
        assert_eq!(v, Some(Variant::F32));
    }

    #[test]
    fn pick_variant_r32uint_is_u32() {
        let v = pick_variant(ktx2::Format::R32_UINT, ktx2::Format::R32_UINT);
        assert_eq!(v, Some(Variant::U32));
    }

    #[test]
    fn pick_variant_r64_uint_is_u64() {
        let v = pick_variant(ktx2::Format::R64_UINT, ktx2::Format::R64_UINT);
        assert_eq!(v, Some(Variant::U64));
    }

    #[test]
    fn families_incompatible_uint_to_unorm() {
        // R8_UINT -> R8_UNORM is a family mismatch.
        assert!(!families_compatible(
            ktx2::Format::R8_UINT,
            ktx2::Format::R8_UNORM
        ));
    }

    #[test]
    fn families_compatible_unorm_to_bc7() {
        assert!(families_compatible(
            ktx2::Format::R8G8B8A8_UNORM,
            ktx2::Format::BC7_UNORM_BLOCK
        ));
    }
}
