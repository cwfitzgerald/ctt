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
/// per-pixel according to the given pattern.
pub fn apply_swizzle(image: &mut RawImage, swizzle: &Swizzle) -> Result<()> {
    if image.pixel_format.components != PixelComponents::Rgba {
        return Err(Error::InvalidSwizzle(
            "swizzle requires RGBA pixel format".into(),
        ));
    }

    if *swizzle == Swizzle::IDENTITY {
        return Ok(());
    }

    let width = image.width as usize;
    let height = image.height as usize;
    let stride = image.stride as usize;

    for y in 0..height {
        for x in 0..width {
            let offset = y * stride + x * 4;
            let r = image.data[offset];
            let g = image.data[offset + 1];
            let b = image.data[offset + 2];
            let a = image.data[offset + 3];

            for (i, channel) in swizzle.0.iter().enumerate() {
                image.data[offset + i] = match channel {
                    SwizzleChannel::R => r,
                    SwizzleChannel::G => g,
                    SwizzleChannel::B => b,
                    SwizzleChannel::A => a,
                    SwizzleChannel::Zero => 0,
                    SwizzleChannel::One => 255,
                };
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ColorSpace, PixelFormat};

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
