use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::{Error, Result};
use crate::surface::{ColorSpace, Image, Surface};
use crate::transform_node::{LayoutInfo, Transform};
use crate::vk_format::{ChannelKind, FormatExt};

/// A single channel source for swizzling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwizzleChannel {
    R,
    G,
    B,
    A,
    Zero,
    One,
}

/// A 4-component swizzle pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swizzle(pub [SwizzleChannel; 4]);

impl Swizzle {
    pub const IDENTITY: Self = Self([
        SwizzleChannel::R,
        SwizzleChannel::G,
        SwizzleChannel::B,
        SwizzleChannel::A,
    ]);
}

/// Apply a swizzle operation to a surface in place.
///
/// The surface must have 4-channel format (RGBA). The swizzle remaps channels
/// per-pixel according to the given pattern.
pub fn apply_swizzle(surface: &mut Surface, swizzle: &Swizzle) -> Result<()> {
    let channels = surface
        .format
        .channel_count()
        .ok_or_else(|| Error::InvalidSwizzle("unknown format channel count".into()))?;
    if channels != 4 {
        return Err(Error::InvalidSwizzle(
            "swizzle requires 4-channel format".into(),
        ));
    }

    if *swizzle == Swizzle::IDENTITY {
        log::debug!("Swizzle: identity, skipping");
        return Ok(());
    }

    log::debug!("Swizzling {}x{} surface", surface.width, surface.height);

    let width = surface.width as usize;
    let height = surface.height as usize;
    let stride = surface.stride as usize;
    let ck = surface
        .format
        .channel_kind()
        .ok_or_else(|| Error::InvalidSwizzle("unknown format channel kind".into()))?;
    let cs = ck.byte_size();
    let pixel_bytes = 4 * cs;

    // Pre-compute the "one" value for this channel type.
    let one = one_value(ck);

    let mut tmp = vec![0u8; pixel_bytes];

    for y in 0..height {
        for x in 0..width {
            let offset = y * stride + x * pixel_bytes;

            for ch in 0..4 {
                let src_ch = match swizzle.0[ch] {
                    SwizzleChannel::R => Some(0),
                    SwizzleChannel::G => Some(1),
                    SwizzleChannel::B => Some(2),
                    SwizzleChannel::A => Some(3),
                    SwizzleChannel::Zero => None,
                    SwizzleChannel::One => None,
                };

                let dst = &mut tmp[ch * cs..(ch + 1) * cs];
                match src_ch {
                    Some(src_idx) => {
                        let src_start = offset + src_idx * cs;
                        dst.copy_from_slice(&surface.data[src_start..src_start + cs]);
                    }
                    None => match swizzle.0[ch] {
                        SwizzleChannel::Zero => dst.fill(0),
                        SwizzleChannel::One => dst.copy_from_slice(&one),
                        _ => unreachable!(),
                    },
                }
            }

            surface.data[offset..offset + pixel_bytes].copy_from_slice(&tmp);
        }
    }

    Ok(())
}

/// Returns the byte representation of "1.0" / max value for a given channel kind.
fn one_value(ck: ChannelKind) -> Vec<u8> {
    match ck {
        ChannelKind::U8 => vec![255],
        ChannelKind::U16 => u16::MAX.to_le_bytes().to_vec(),
        ChannelKind::F16 => half::f16::from_f64(1.0).to_le_bytes().to_vec(),
        ChannelKind::F32 => 1.0f32.to_le_bytes().to_vec(),
        ChannelKind::U32 => u32::MAX.to_le_bytes().to_vec(),
    }
}

/// A transform that remaps RGBA channels according to a swizzle pattern.
pub struct SwizzleTransform {
    swizzle: Swizzle,
}

impl SwizzleTransform {
    pub fn new(swizzle: Swizzle) -> Self {
        Self { swizzle }
    }
}

impl Transform for SwizzleTransform {
    fn name(&self) -> &str {
        "swizzle"
    }

    fn constraint(&self) -> FormatConstraint {
        FormatConstraint {
            formats: Some(vec![
                ktx2::Format::R8G8B8A8_UNORM,
                ktx2::Format::R16G16B16A16_UNORM,
                ktx2::Format::R16G16B16A16_SFLOAT,
                ktx2::Format::R32G32B32A32_SFLOAT,
            ]),
            color_spaces: None,
            alpha_modes: None,
        }
    }

    fn output_format(
        &self,
        input: ktx2::Format,
        cs: ColorSpace,
        alpha: AlphaMode,
    ) -> (ktx2::Format, ColorSpace, AlphaMode) {
        (input, cs, alpha)
    }

    fn output_layout(&self, input: &LayoutInfo) -> LayoutInfo {
        input.clone()
    }

    fn execute(&self, mut image: Image) -> Result<Image> {
        if self.swizzle == Swizzle::IDENTITY {
            log::debug!("Swizzle: identity, skipping");
            return Ok(image);
        }

        log::info!("Applying swizzle: {:?}", self.swizzle.0);

        for layer in &mut image.surfaces {
            for surface in layer {
                apply_swizzle(surface, &self.swizzle)?;
            }
        }

        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba_surface(pixels: &[[u8; 4]]) -> Surface {
        let width = pixels.len() as u32;
        let data: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
        Surface {
            data,
            width,
            height: 1,
            stride: width * 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Straight,
        }
    }

    #[test]
    fn identity_swizzle_is_noop() {
        let original = [100u8, 150, 200, 255];
        let mut surface = make_rgba_surface(&[original]);
        apply_swizzle(&mut surface, &Swizzle::IDENTITY).unwrap();
        assert_eq!(&surface.data, &original);
    }

    #[test]
    fn bgra_swap() {
        let mut surface = make_rgba_surface(&[[10, 20, 30, 40]]);
        let bgra = Swizzle([
            SwizzleChannel::B,
            SwizzleChannel::G,
            SwizzleChannel::R,
            SwizzleChannel::A,
        ]);
        apply_swizzle(&mut surface, &bgra).unwrap();
        assert_eq!(&surface.data, &[30, 20, 10, 40]);
    }

    #[test]
    fn zero_and_one_channels() {
        let mut surface = make_rgba_surface(&[[10, 20, 30, 40]]);
        let swizzle = Swizzle([
            SwizzleChannel::Zero,
            SwizzleChannel::One,
            SwizzleChannel::R,
            SwizzleChannel::Zero,
        ]);
        apply_swizzle(&mut surface, &swizzle).unwrap();
        assert_eq!(&surface.data, &[0, 255, 10, 0]);
    }
}
