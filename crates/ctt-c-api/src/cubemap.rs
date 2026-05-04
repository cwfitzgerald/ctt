use crate::error::{Status, map_error, set_last_error};
use crate::surface::{Surface, take_surface};

/// Opaque handle to a cubemap input — either six separate face surfaces, a
/// cross-layout image, or a horizontal strip of six faces.
///
/// Pass to [`ctt_split_cubemap`] to extract the six face surfaces.
pub struct CubemapInput(pub(crate) ctt::CubemapInput);

/// Build a cubemap input from six separate face surfaces, ordered
/// `+X, -X, +Y, -Y, +Z, -Z`.
///
/// `faces` must point to an array of exactly six surface handles; **all
/// six are consumed** on both success and failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_cubemap_input_separate_faces(
    faces: *mut *mut Surface,
) -> *mut CubemapInput {
    if faces.is_null() {
        set_last_error("ctt_cubemap_input_separate_faces: faces pointer is null");
        return std::ptr::null_mut();
    }
    let face_ptrs: [*mut Surface; 6] = unsafe { std::ptr::read(faces.cast::<[*mut Surface; 6]>()) };
    let mut taken: Vec<ctt::Surface> = Vec::with_capacity(6);
    let mut error: Option<&'static str> = None;
    for ptr in face_ptrs {
        match unsafe { take_surface(ptr) } {
            Ok(s) => taken.push(s),
            Err(_) => {
                error = Some("ctt_cubemap_input_separate_faces: a face handle is null");
                break;
            }
        }
    }
    if let Some(msg) = error {
        set_last_error(msg);
        return std::ptr::null_mut();
    }
    let arr: [ctt::Surface; 6] = taken
        .try_into()
        .expect("collected exactly 6 elements above");
    let input = ctt::CubemapInput::SeparateFaces(Box::new(arr));
    Box::into_raw(Box::new(CubemapInput(input)))
}

/// Build a cubemap input from a 4×3 cross-layout image. Consumes the surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_cubemap_input_cross(surface: *mut Surface) -> *mut CubemapInput {
    let s = match unsafe { take_surface(surface) } {
        Ok(s) => s,
        Err(_) => {
            set_last_error("ctt_cubemap_input_cross: surface is null");
            return std::ptr::null_mut();
        }
    };
    Box::into_raw(Box::new(CubemapInput(ctt::CubemapInput::Cross(s))))
}

/// Build a cubemap input from a horizontal strip of 6 faces. Consumes the
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_cubemap_input_strip(surface: *mut Surface) -> *mut CubemapInput {
    let s = match unsafe { take_surface(surface) } {
        Ok(s) => s,
        Err(_) => {
            set_last_error("ctt_cubemap_input_strip: surface is null");
            return std::ptr::null_mut();
        }
    };
    Box::into_raw(Box::new(CubemapInput(ctt::CubemapInput::Strip(s))))
}

/// Destroy a cubemap input. `input` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_cubemap_input_destroy(input: *mut CubemapInput) {
    if input.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(input) });
}

/// Split a cubemap input into six face surfaces.
///
/// **Consumes** `input` on both success and failure. On success writes six
/// new surface handles into `out_faces[0..6]`, in `+X, -X, +Y, -Y, +Z, -Z`
/// order. On failure leaves `out_faces` untouched.
///
/// `out_faces` must point to space for six `ctt_surface_t` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_split_cubemap(
    input: *mut CubemapInput,
    out_faces: *mut *mut Surface,
) -> Status {
    if input.is_null() {
        set_last_error("ctt_split_cubemap: input is null");
        return Status::NullPointer;
    }
    if out_faces.is_null() {
        // Still consume the input.
        drop(unsafe { Box::from_raw(input) });
        set_last_error("ctt_split_cubemap: out_faces is null");
        return Status::NullPointer;
    }

    let boxed = unsafe { Box::from_raw(input) };
    let result = ctt::split_cubemap(boxed.0);

    match result {
        Ok(faces) => {
            let face_ptrs: [*mut Surface; 6] = faces.map(|s| Box::into_raw(Box::new(Surface(s))));
            unsafe {
                std::ptr::write(out_faces.cast::<[*mut Surface; 6]>(), face_ptrs);
            }
            Status::Ok
        }
        Err(e) => map_error(e),
    }
}
