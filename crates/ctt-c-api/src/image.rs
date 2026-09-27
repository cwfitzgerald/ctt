use crate::error::{Status, set_last_error};
use crate::surface::{Surface, take_surface};
use crate::types::{AlphaMode, ColorSpace, FormatDesc, TextureKind};

/// Opaque handle to a multi-layer / multi-mip image.
///
/// `surfaces[i][j]` is layer `i`, mip level `j`. The meaning of the layer
/// axis depends on [`TextureKind`]: one entry per array layer for `Texture2D`,
/// six entries per cube for `Cubemap`, exactly one entry for `Texture3D`
/// (which carries depth on each surface).
pub struct Image(pub(crate) ctt::Image);

/// Create an empty image of the given kind whose surfaces are read as `desc`.
///
/// `desc.format` must be non-zero and agree with `desc.color_space`.
///
/// On failure returns `NULL` and sets the thread-local error message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_create(kind: TextureKind, desc: FormatDesc) -> *mut Image {
    let desc = match desc.into_inner() {
        Ok(desc) => desc,
        Err(msg) => {
            set_last_error(format!("ctt_image_create: {msg}"));
            return std::ptr::null_mut();
        }
    };
    let img = ctt::Image {
        surfaces: Vec::new(),
        kind: kind.into(),
        desc,
    };
    Box::into_raw(Box::new(Image(img)))
}

/// Destroy an image. `img` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_destroy(img: *mut Image) {
    if img.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(img) });
}

/// Append a new (empty) layer to the image.
///
/// Writes the new layer's index into `*out_layer` if non-null. Returns
/// `CTT_STATUS_NULL_POINTER` if `img` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_add_layer(img: *mut Image, out_layer: *mut usize) -> Status {
    let Some(image) = (unsafe { img.as_mut() }) else {
        set_last_error("ctt_image_add_layer: image is null");
        return Status::NullPointer;
    };
    image.0.surfaces.push(Vec::new());
    if let Some(out) = unsafe { out_layer.as_mut() } {
        *out = image.0.surfaces.len() - 1;
    }
    Status::Ok
}

/// Push a mip level onto an existing layer. **Consumes** the surface — on
/// both success and failure the surface handle is destroyed; the caller
/// must not call `ctt_surface_destroy` on it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_push_mip(
    img: *mut Image,
    layer: usize,
    surface: *mut Surface,
) -> Status {
    let surface = match unsafe { take_surface(surface) } {
        Ok(s) => s,
        Err(status) => return status,
    };
    let Some(image) = (unsafe { img.as_mut() }) else {
        set_last_error("ctt_image_push_mip: image is null");
        return Status::NullPointer;
    };
    let Some(slot) = image.0.surfaces.get_mut(layer) else {
        set_last_error(format!(
            "ctt_image_push_mip: layer index {layer} is out of range (image has {} layers)",
            image.0.surfaces.len()
        ));
        return Status::InvalidArgument;
    };
    slot.push(surface);
    Status::Ok
}

/// Number of layers in the image. Returns `0` if `img` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_layer_count(img: *const Image) -> usize {
    unsafe { img.as_ref() }.map_or(0, |i| i.0.surfaces.len())
}

/// Number of mip levels stored in `layer`. Returns `0` if `img` is null or
/// `layer` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_mip_count(img: *const Image, layer: usize) -> usize {
    unsafe { img.as_ref() }
        .and_then(|i| i.0.surfaces.get(layer))
        .map_or(0, Vec::len)
}

/// Texture kind of the image. Returns `CTT_TEXTURE_KIND_TEXTURE2D` if `img`
/// is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_kind(img: *const Image) -> TextureKind {
    unsafe { img.as_ref() }.map_or(TextureKind::Texture2d, |i| i.0.kind.into())
}

/// How to read the bytes of every surface in the image. Returns a desc with
/// format `0` (`VK_FORMAT_UNDEFINED`), `CTT_COLOR_SPACE_LINEAR`, and
/// `CTT_ALPHA_MODE_STRAIGHT` if `img` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_desc(img: *const Image) -> FormatDesc {
    unsafe { img.as_ref() }.map_or(
        FormatDesc {
            format: 0,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        },
        |i| i.0.desc.into(),
    )
}

fn surface_at(img: &Image, layer: usize, mip: usize) -> Option<&ctt::Surface> {
    img.0.surfaces.get(layer)?.get(mip)
}

/// Allocate a freshly cloned `ctt_surface_t` for the surface at the given
/// `(layer, mip)` coordinates. The returned handle is independent of the
/// image and must be destroyed via `ctt_surface_destroy`.
///
/// Returns NULL if `img` is null or the indices are out of range. Cloning
/// copies pixel/block data, which can be expensive for large surfaces; use
/// the field accessors below to read metadata or borrow the data buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_clone_surface(
    img: *const Image,
    layer: usize,
    mip: usize,
) -> *mut Surface {
    let Some(image) = (unsafe { img.as_ref() }) else {
        return std::ptr::null_mut();
    };
    let Some(s) = surface_at(image, layer, mip) else {
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(Surface(s.clone())))
}

/// Pointer to the data buffer of one surface inside the image. Valid for as
/// long as the image lives and is not mutated. Returns `NULL` if `img` is
/// null or `(layer, mip)` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_surface_data(
    img: *const Image,
    layer: usize,
    mip: usize,
) -> *const u8 {
    let Some(image) = (unsafe { img.as_ref() }) else {
        return std::ptr::null();
    };
    surface_at(image, layer, mip).map_or(std::ptr::null(), |s| s.data.as_ptr())
}

/// Length in bytes of one surface's data buffer. Returns `0` if `img` is
/// null or `(layer, mip)` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_surface_data_len(
    img: *const Image,
    layer: usize,
    mip: usize,
) -> usize {
    let Some(image) = (unsafe { img.as_ref() }) else {
        return 0;
    };
    surface_at(image, layer, mip).map_or(0, |s| s.data.len())
}

/// Width in pixels of the surface at `(layer, mip)`. Returns `0` if `img` is
/// null or `(layer, mip)` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_surface_width(
    img: *const Image,
    layer: usize,
    mip: usize,
) -> u32 {
    let Some(image) = (unsafe { img.as_ref() }) else {
        return 0;
    };
    surface_at(image, layer, mip).map_or(0, |s| s.width)
}

/// Height in pixels of the surface at `(layer, mip)`. Returns `0` if `img`
/// is null or `(layer, mip)` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_surface_height(
    img: *const Image,
    layer: usize,
    mip: usize,
) -> u32 {
    let Some(image) = (unsafe { img.as_ref() }) else {
        return 0;
    };
    surface_at(image, layer, mip).map_or(0, |s| s.height)
}

/// Depth (Z slice count) of the surface at `(layer, mip)`. `1` for 2D
/// surfaces. Returns `0` if `img` is null or `(layer, mip)` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_surface_depth(
    img: *const Image,
    layer: usize,
    mip: usize,
) -> u32 {
    let Some(image) = (unsafe { img.as_ref() }) else {
        return 0;
    };
    surface_at(image, layer, mip).map_or(0, |s| s.depth)
}

/// Row stride in bytes of the surface at `(layer, mip)`. Returns `0` if
/// `img` is null or `(layer, mip)` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_surface_stride(
    img: *const Image,
    layer: usize,
    mip: usize,
) -> u32 {
    let Some(image) = (unsafe { img.as_ref() }) else {
        return 0;
    };
    surface_at(image, layer, mip).map_or(0, |s| s.stride)
}

/// Z-slice stride in bytes of the surface at `(layer, mip)`. Meaningful only
/// when `depth > 1`. Returns `0` if `img` is null or `(layer, mip)` is out
/// of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_image_surface_slice_stride(
    img: *const Image,
    layer: usize,
    mip: usize,
) -> u32 {
    let Some(image) = (unsafe { img.as_ref() }) else {
        return 0;
    };
    surface_at(image, layer, mip).map_or(0, |s| s.slice_stride)
}
