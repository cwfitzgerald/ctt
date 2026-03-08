pub mod bindings {
    use ispc_rt::ispc_module;
    ispc_module!(kernel);
    ispc_module!(kernel_astc);
}

pub mod astc;
pub mod bc1;
pub mod bc3;
pub mod bc4;
pub mod bc5;
pub mod bc6h;
pub mod bc7;
pub mod etc1;

/// Describes a 2D image to block-compress.
#[derive(Debug, Copy, Clone)]
pub struct Surface<'a, const COMPONENTS: usize> {
    /// The pixel data for the image.
    /// The data does not need to be tightly packed, but if it isn't, stride must be different from `width * COMPONENTS`.
    ///
    /// Expected to be at least `stride * height`.
    pub data: &'a [u8],
    /// The width of the image in texels.
    pub width: u32,
    /// The height of the image in texels.
    pub height: u32,
    /// The stride between the rows of the image, in bytes.
    /// If `data` is tightly packed, this is expected to be `width * COMPONENTS`.
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
        let required = stride as usize * height as usize;
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

pub type RgbaSurface<'a> = Surface<'a, 4>;
pub type RgSurface<'a> = Surface<'a, 2>;
pub type RSurface<'a> = Surface<'a, 1>;
