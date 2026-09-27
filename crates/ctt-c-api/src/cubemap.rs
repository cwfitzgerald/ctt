use crate::error::{Status, catch_panic, map_error, set_last_error};
use crate::image::Image;
use crate::surface::Surface;
use crate::types::FormatDesc;

/// Split a horizontal (4×3) or vertical (3×4) cross-layout atlas into a
/// cubemap image with one mip level per face.
///
/// Does not consume `surface`. On success writes a new image handle into
/// `*out_image`; the faces are in `+X, -X, +Y, -Y, +Z, -Z` order. On failure
/// leaves `*out_image` untouched.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_split_cubemap_cross(
    surface: *const Surface,
    desc: FormatDesc,
    out_image: *mut *mut Image,
) -> Status {
    unsafe {
        split(
            "ctt_split_cubemap_cross",
            surface,
            desc,
            out_image,
            |surface, desc| ctt::CubemapInput::Cross { surface, desc },
        )
    }
}

/// Split a horizontal strip of six faces into a cubemap image with one mip
/// level per face.
///
/// Does not consume `surface`. On success writes a new image handle into
/// `*out_image`; the faces are in `+X, -X, +Y, -Y, +Z, -Z` order. On failure
/// leaves `*out_image` untouched.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_split_cubemap_strip(
    surface: *const Surface,
    desc: FormatDesc,
    out_image: *mut *mut Image,
) -> Status {
    unsafe {
        split(
            "ctt_split_cubemap_strip",
            surface,
            desc,
            out_image,
            |surface, desc| ctt::CubemapInput::Strip { surface, desc },
        )
    }
}

/// Shared body of the `ctt_split_cubemap_*` entry points. `name` prefixes
/// the error messages.
unsafe fn split(
    name: &str,
    surface: *const Surface,
    desc: FormatDesc,
    out_image: *mut *mut Image,
    input: impl FnOnce(ctt::SurfaceRef<'_>, ctt::FormatDesc) -> ctt::CubemapInput<'_>,
) -> Status {
    catch_panic(Status::Internal, || {
        let Some(surface) = (unsafe { surface.as_ref() }) else {
            set_last_error(format!("{name}: surface is null"));
            return Status::NullPointer;
        };
        if out_image.is_null() {
            set_last_error(format!("{name}: out_image is null"));
            return Status::NullPointer;
        }
        let desc = match desc.into_inner() {
            Ok(desc) => desc,
            Err(msg) => {
                set_last_error(format!("{name}: {msg}"));
                return Status::InvalidArgument;
            }
        };
        match ctt::split_cubemap(input(surface.0.as_ref(), desc)) {
            Ok(image) => {
                unsafe { *out_image = Box::into_raw(Box::new(Image(image))) };
                Status::Ok
            }
            Err(e) => map_error(e),
        }
    })
}
