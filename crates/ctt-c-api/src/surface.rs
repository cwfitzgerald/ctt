use crate::error::{Status, set_last_error};

/// Opaque handle to a single image surface (raw pixels or compressed blocks).
///
/// 2D surfaces use `depth == 1`; 3D (volume) surfaces use `depth > 1` with
/// all Z slices packed contiguously. Allocation is owned by ctt — call
/// [`ctt_surface_destroy`] when finished, unless the surface has been
/// consumed by another API.
pub struct Surface(pub(crate) ctt::Surface);

/// Create a new surface, copying `data_len` bytes from `data` into Rust's
/// allocator.
///
/// `slice_stride` is only meaningful when `depth > 1`; pass `0` for 2D
/// surfaces. The [`FormatDesc`](crate::FormatDesc) of the image that holds
/// the surface tells how to read the bytes.
///
/// On failure returns `NULL` and sets the thread-local error message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_create(
    data: *const u8,
    data_len: usize,
    width: u32,
    height: u32,
    depth: u32,
    stride: u32,
    slice_stride: u32,
) -> *mut Surface {
    if data.is_null() && data_len != 0 {
        set_last_error("ctt_surface_create: data is null but data_len != 0");
        return std::ptr::null_mut();
    }
    // Safety: caller asserts data points to data_len bytes.
    let bytes = if data_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data, data_len) }.to_vec()
    };

    let surface = ctt::Surface {
        data: bytes,
        width,
        height,
        depth,
        stride,
        slice_stride,
    };
    Box::into_raw(Box::new(Surface(surface)))
}

/// Destroy a surface. `s` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_destroy(s: *mut Surface) {
    if s.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(s) });
}

/// Pointer to the surface's pixel/block bytes. Valid as long as the surface
/// exists and is not modified or consumed. Returns `NULL` if `s` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_data(s: *const Surface) -> *const u8 {
    let Some(s) = (unsafe { s.as_ref() }) else {
        return std::ptr::null();
    };
    s.0.data.as_ptr()
}

/// Length in bytes of the surface's data buffer. Returns `0` if `s` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_data_len(s: *const Surface) -> usize {
    let Some(s) = (unsafe { s.as_ref() }) else {
        return 0;
    };
    s.0.data.len()
}

/// Width in pixels of the surface. Returns `0` if `s` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_width(s: *const Surface) -> u32 {
    unsafe { s.as_ref() }.map_or(0, |s| s.0.width)
}

/// Height in pixels of the surface. Returns `0` if `s` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_height(s: *const Surface) -> u32 {
    unsafe { s.as_ref() }.map_or(0, |s| s.0.height)
}

/// Depth (Z slice count) of the surface. `1` for 2D surfaces. Returns `0`
/// if `s` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_depth(s: *const Surface) -> u32 {
    unsafe { s.as_ref() }.map_or(0, |s| s.0.depth)
}

/// Row stride in bytes. Returns `0` if `s` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_stride(s: *const Surface) -> u32 {
    unsafe { s.as_ref() }.map_or(0, |s| s.0.stride)
}

/// Z-slice stride in bytes. Meaningful only when `depth > 1`. Returns `0`
/// if `s` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_slice_stride(s: *const Surface) -> u32 {
    unsafe { s.as_ref() }.map_or(0, |s| s.0.slice_stride)
}

/// Deep-copy a surface (data and layout).
///
/// On failure returns `NULL` and sets the thread-local error message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_surface_clone(s: *const Surface) -> *mut Surface {
    let Some(src) = (unsafe { s.as_ref() }) else {
        set_last_error("ctt_surface_clone: source is null");
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(Surface(src.0.clone())))
}

pub(crate) unsafe fn take_surface(ptr: *mut Surface) -> Result<ctt::Surface, Status> {
    if ptr.is_null() {
        set_last_error("expected non-null surface handle");
        return Err(Status::NullPointer);
    }
    let boxed = unsafe { Box::from_raw(ptr) };
    Ok(boxed.0)
}
