use crate::alpha::AlphaMode;
use crate::error::Result;
use crate::surface::Surface;
use crate::vk_format::FormatExt;

use super::{read_channel, write_channel};

/// Premultiply alpha: RGB *= A. Operates on normalized [0,1] values.
pub(crate) fn premultiply_alpha(surface: &Surface) -> Result<Surface> {
    let cc = surface
        .format
        .channel_count()
        .expect("unknown channel count");
    let ck = surface.format.channel_kind().expect("unknown channel kind");
    let cs = ck.byte_size();
    let bpp = cc * cs;

    assert!(cc == 4, "premultiply_alpha requires 4-channel format");

    let width = surface.width as usize;
    let height = surface.height as usize;
    let stride = surface.stride as usize;

    let mut out = surface.data.clone();

    for y in 0..height {
        for x in 0..width {
            let off = y * stride + x * bpp;
            let alpha = read_channel(&surface.data, off + 3 * cs, ck);

            for ch in 0..3 {
                let val = read_channel(&surface.data, off + ch * cs, ck);
                write_channel(&mut out, off + ch * cs, ck, val * alpha);
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: surface.stride,
        format: surface.format,
        color_space: surface.color_space,
        alpha: AlphaMode::Premultiplied,
    })
}

/// Unpremultiply alpha: RGB /= A. Operates on normalized [0,1] values.
pub(crate) fn unpremultiply_alpha(surface: &Surface) -> Result<Surface> {
    let cc = surface
        .format
        .channel_count()
        .expect("unknown channel count");
    let ck = surface.format.channel_kind().expect("unknown channel kind");
    let cs = ck.byte_size();
    let bpp = cc * cs;

    assert!(cc == 4, "unpremultiply_alpha requires 4-channel format");

    let width = surface.width as usize;
    let height = surface.height as usize;
    let stride = surface.stride as usize;

    let mut out = surface.data.clone();

    for y in 0..height {
        for x in 0..width {
            let off = y * stride + x * bpp;
            let alpha = read_channel(&surface.data, off + 3 * cs, ck);

            if alpha > 0.0 {
                for ch in 0..3 {
                    let val = read_channel(&surface.data, off + ch * cs, ck);
                    write_channel(&mut out, off + ch * cs, ck, val / alpha);
                }
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: surface.stride,
        format: surface.format,
        color_space: surface.color_space,
        alpha: AlphaMode::Straight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::ColorSpace;

    #[test]
    fn premultiply_roundtrip_surface() {
        // 1x1 pixel with alpha=0.5 (128/255)
        let surface = Surface {
            data: vec![200, 100, 50, 128],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };

        let premul = premultiply_alpha(&surface).unwrap();
        assert_eq!(premul.alpha, AlphaMode::Premultiplied);
        // Alpha should be unchanged
        assert_eq!(premul.data[3], 128);

        let back = unpremultiply_alpha(&premul).unwrap();
        assert_eq!(back.alpha, AlphaMode::Straight);
        // Should round-trip within +-1
        for i in 0..4 {
            assert!(
                (back.data[i] as i16 - surface.data[i] as i16).unsigned_abs() <= 1,
                "channel {i}: {} vs {}",
                back.data[i],
                surface.data[i],
            );
        }
    }
}
