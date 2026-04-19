//! Internal processing pipeline: load → (swizzle/mipmap/alpha) → store → encode.
//!
//! The public entry point is [`crate::convert::convert`]. Everything here is
//! a collection of plain functions over [`Buffer`][buffer::Buffer] and
//! [`Surface`][crate::surface::Surface].

pub(crate) mod alpha;
pub(crate) mod buffer;
pub(crate) mod encode;
pub(crate) mod load;
pub(crate) mod load_kernels;
pub(crate) mod mipmap;
pub(crate) mod passthrough;
pub(crate) mod store;
pub(crate) mod store_kernels;
pub(crate) mod swizzle;

pub use buffer::{Buffer, Variant};
pub use mipmap::MipmapFilter;
pub use swizzle::{Swizzle, SwizzleChannel};

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
            stride: width * bpp,
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
            stride: 4,
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
            stride: 256 * 4,
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
            stride: 4,
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
            stride: 16,
            format: ktx2::Format::R32G32B32A32_UINT,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Opaque,
        };
        let buf = load::load_u32(&surface).unwrap();
        let out =
            store::store_u32(buf, ktx2::Format::R32G32B32A32_UINT, AlphaMode::Opaque).unwrap();
        assert_eq!(out.data, data);
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
