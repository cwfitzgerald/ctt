use crate::error::{Error, Result};
use crate::format::PixelComponents;
use crate::image::RawImage;

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

/// Apply a swizzle operation to a raw image in place.
///
/// The image must have RGBA pixel components. The swizzle remaps channels
/// per-pixel according to the given pattern. Works with any channel type
/// (U8, U16, F16, F32).
pub fn apply_swizzle(image: &mut RawImage, swizzle: &Swizzle) -> Result<()> {
    if image.pixel_format.components != PixelComponents::Rgba {
        return Err(Error::InvalidSwizzle(
            "swizzle requires RGBA pixel format".into(),
        ));
    }

    if *swizzle == Swizzle::IDENTITY {
        log::debug!("Swizzle: identity, skipping");
        return Ok(());
    }

    log::debug!("Swizzling {}x{} image", image.width, image.height);

    let width = image.width as usize;
    let height = image.height as usize;
    let stride = image.stride as usize;
    let cs = image.pixel_format.channel_type.byte_size();
    let bpp = 4 * cs; // RGBA = 4 channels

    // Pre-compute the "one" value for this channel type.
    let one = one_value(cs);

    let mut tmp = vec![0u8; bpp];

    for y in 0..height {
        for x in 0..width {
            let offset = y * stride + x * bpp;

            // Read all 4 channels into tmp.
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
                        dst.copy_from_slice(&image.data[src_start..src_start + cs]);
                    }
                    None => match swizzle.0[ch] {
                        SwizzleChannel::Zero => dst.fill(0),
                        SwizzleChannel::One => dst.copy_from_slice(&one),
                        _ => unreachable!(),
                    },
                }
            }

            image.data[offset..offset + bpp].copy_from_slice(&tmp);
        }
    }

    Ok(())
}

/// Returns the byte representation of "1.0" / max value for a given channel byte size.
fn one_value(channel_byte_size: usize) -> Vec<u8> {
    match channel_byte_size {
        1 => vec![255],
        2 => u16::MAX.to_le_bytes().to_vec(),
        4 => 1.0f32.to_le_bytes().to_vec(),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ChannelType, ColorSpace, PixelFormat};

    fn make_rgba_image(pixels: &[[u8; 4]]) -> RawImage {
        let width = pixels.len() as u32;
        let data: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
        RawImage {
            data,
            width,
            height: 1,
            stride: width * 4,
            pixel_format: PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U8,
                color_space: ColorSpace::Srgb,
            },
        }
    }

    #[test]
    fn identity_swizzle_is_noop() {
        let original = [100u8, 150, 200, 255];
        let mut image = make_rgba_image(&[original]);
        apply_swizzle(&mut image, &Swizzle::IDENTITY).unwrap();
        assert_eq!(&image.data, &original);
    }

    #[test]
    fn bgra_swap() {
        let mut image = make_rgba_image(&[[10, 20, 30, 40]]);
        let bgra = Swizzle([
            SwizzleChannel::B,
            SwizzleChannel::G,
            SwizzleChannel::R,
            SwizzleChannel::A,
        ]);
        apply_swizzle(&mut image, &bgra).unwrap();
        assert_eq!(&image.data, &[30, 20, 10, 40]);
    }

    #[test]
    fn zero_and_one_channels() {
        let mut image = make_rgba_image(&[[10, 20, 30, 40]]);
        let swizzle = Swizzle([
            SwizzleChannel::Zero,
            SwizzleChannel::One,
            SwizzleChannel::R,
            SwizzleChannel::Zero,
        ]);
        apply_swizzle(&mut image, &swizzle).unwrap();
        assert_eq!(&image.data, &[0, 255, 10, 0]);
    }
}
