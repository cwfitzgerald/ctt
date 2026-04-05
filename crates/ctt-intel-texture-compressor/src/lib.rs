#[cfg(feature = "prebuilt")]
extern crate ctt_intel_texture_compressor_prebuilt;

pub mod bindings;

pub mod bc1;
pub mod bc3;
pub mod bc4;
pub mod bc5;
pub mod bc6h;
pub mod bc7;
pub mod etc1;

/// Describes a 2D image to block-compress.
///
/// `COMPONENTS` is the number of **bytes per pixel** in the source data. This
/// determines the expected tightly-packed stride (`width * COMPONENTS`) and is
/// used to validate that `data` is large enough. The actual channel layout
/// depends on which encoder consumes the surface — see the individual format
/// modules for details.
///
/// # Available surface type aliases
///
/// | Alias | `COMPONENTS` | Layout | Used by |
/// |---|---|---|---|
/// | [`RSurface`] | 1 | `R8` (1 byte/pixel) | [`bc4`] |
/// | [`RgSurface`] | 2 | `R8 G8` interleaved (2 bytes/pixel) | [`bc5`] |
/// | [`RgbaSurface`] | 4 | `R8 G8 B8 A8` interleaved (4 bytes/pixel) | [`bc1`], [`bc3`], [`bc7`], [`etc1`] |
/// | [`RgbaF16Surface`] | 8 | `R16 G16 B16 A16` interleaved (8 bytes/pixel) | [`bc6h`] |
#[derive(Debug, Copy, Clone)]
pub struct Surface<'a, const COMPONENTS: usize> {
    /// The raw pixel data for the image.
    ///
    /// The byte interpretation depends on the encoder that consumes this
    /// surface. For most formats, each byte is one u8 channel sample. For
    /// [`bc6h`], every two bytes form a little-endian f16 channel sample.
    ///
    /// The data does not need to be tightly packed, but if it isn't, `stride`
    /// must differ from `width * COMPONENTS`.
    ///
    /// Expected to be at least `stride * height` bytes.
    pub data: &'a [u8],
    /// The width of the image in texels.
    pub width: u32,
    /// The height of the image in texels.
    pub height: u32,
    /// The stride between rows of the image, in bytes.
    ///
    /// For tightly-packed data this is `width * COMPONENTS`.
    pub stride: u32,
}

impl<'a, const COMPONENTS: usize> Surface<'a, COMPONENTS> {
    /// Creates a new surface, validating that dimensions fit within `i32` (required
    /// by the underlying C API) and that `data` is large enough for the given
    /// stride and height.
    ///
    /// # Panics
    ///
    /// Panics if `width`, `height`, or `stride` exceed `i32::MAX`, or if
    /// `data.len()` is less than `stride * height`.
    pub fn new(data: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
        assert!(
            i32::try_from(width).is_ok(),
            "width {width} exceeds i32::MAX"
        );
        assert!(
            i32::try_from(height).is_ok(),
            "height {height} exceeds i32::MAX"
        );
        assert!(
            i32::try_from(stride).is_ok(),
            "stride {stride} exceeds i32::MAX"
        );
        let width_components = (width as usize)
            .checked_mul(COMPONENTS)
            .expect("width * COMPONENTS overflows usize");
        assert!(
            stride as usize >= width_components,
            "stride {stride} is less than width * COMPONENTS ({} * {COMPONENTS} = {width_components})",
            width,
        );
        let required = (stride as usize)
            .checked_mul(height as usize)
            .expect("stride * height overflows usize");
        assert!(
            data.len() >= required,
            "data length {} is less than stride * height ({required})",
            data.len()
        );
        Self {
            data,
            width,
            height,
            stride,
        }
    }
}

/// 4-channel, 8-bit surface: `R8 G8 B8 A8` — 4 bytes per pixel.
///
/// Used by [`bc1`], [`bc3`], [`bc7`], and [`etc1`].
pub type RgbaSurface<'a> = Surface<'a, 4>;

/// 2-channel, 8-bit surface: `R8 G8` interleaved — 2 bytes per pixel.
///
/// Used by [`bc5`].
pub type RgSurface<'a> = Surface<'a, 2>;

/// 1-channel, 8-bit surface: `R8` — 1 byte per pixel.
///
/// Used by [`bc4`].
pub type RSurface<'a> = Surface<'a, 1>;

/// 4-channel, 16-bit surface: `R16 G16 B16 A16` — 8 bytes per pixel.
///
/// Each channel is a little-endian IEEE 754 binary16 (half-precision) float.
/// The alpha channel is present in the layout but ignored by [`bc6h`].
pub type RgbaF16Surface<'a> = Surface<'a, 8>;
