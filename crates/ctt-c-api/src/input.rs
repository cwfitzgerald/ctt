use crate::error::{Status, map_error, set_last_error};
use crate::image::Image;
use crate::types::{AlphaMode, ColorSpace, OptionalAlphaMode, OptionalColorSpace};

/// Detected container format.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Ktx2 = 0,
    Dds = 1,
}

impl From<InputFormat> for ctt::input::InputFormat {
    fn from(f: InputFormat) -> Self {
        match f {
            InputFormat::Ktx2 => ctt::input::InputFormat::Ktx2,
            InputFormat::Dds => ctt::input::InputFormat::Dds,
        }
    }
}

impl From<ctt::input::InputFormat> for InputFormat {
    fn from(f: ctt::input::InputFormat) -> Self {
        match f {
            ctt::input::InputFormat::Ktx2 => InputFormat::Ktx2,
            ctt::input::InputFormat::Dds => InputFormat::Dds,
        }
    }
}

/// Optional metadata overrides applied to every surface in a decoded image.
///
/// Each field's `present` flag selects whether to override the value the
/// container would otherwise provide.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InputOverrides {
    pub color_space: OptionalColorSpace,
    pub alpha: OptionalAlphaMode,
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_input_overrides_default() -> InputOverrides {
    InputOverrides {
        color_space: OptionalColorSpace {
            present: false,
            value: ColorSpace::Linear,
        },
        alpha: OptionalAlphaMode {
            present: false,
            value: AlphaMode::Straight,
        },
    }
}

impl From<InputOverrides> for ctt::input::InputOverrides {
    fn from(o: InputOverrides) -> Self {
        ctt::input::InputOverrides {
            color_space: o.color_space.present.then(|| o.color_space.value.into()),
            alpha: o.alpha.present.then(|| o.alpha.value.into()),
        }
    }
}

/// Detect whether `data` is a recognized container.
///
/// Returns `true` and writes the format into `*out_format` if recognized,
/// `false` otherwise. `out_format` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_detect_container(
    data: *const u8,
    len: usize,
    out_format: *mut InputFormat,
) -> bool {
    if data.is_null() && len != 0 {
        return false;
    }
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    match ctt::input::detect_container(bytes) {
        Some(f) => {
            if let Some(out) = unsafe { out_format.as_mut() } {
                *out = f.into();
            }
            true
        }
        None => false,
    }
}

/// Decode a container, auto-detecting format from magic bytes.
///
/// On success writes a freshly allocated [`ctt_image_t`] into `*out_image`.
/// `*out_recognized` (if non-null) is set to `true` when the data was a
/// recognized container; when it is `false`, `*out_image` is set to NULL
/// and the status code is `CTT_STATUS_OK` (not an error).
///
/// `overrides` may be NULL to use defaults.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_decode_container(
    data: *const u8,
    len: usize,
    overrides: *const InputOverrides,
    out_image: *mut *mut Image,
    out_recognized: *mut bool,
) -> Status {
    if out_image.is_null() {
        set_last_error("ctt_decode_container: out_image is null");
        return Status::NullPointer;
    }
    if data.is_null() && len != 0 {
        set_last_error("ctt_decode_container: data is null but len != 0");
        return Status::NullPointer;
    }
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    let overrides_inner: ctt::input::InputOverrides = match unsafe { overrides.as_ref() } {
        Some(o) => (*o).into(),
        None => ctt::input::InputOverrides::default(),
    };

    match ctt::input::decode_container(bytes, overrides_inner) {
        Ok(Some(image)) => {
            unsafe {
                *out_image = Box::into_raw(Box::new(Image(image)));
            }
            if let Some(r) = unsafe { out_recognized.as_mut() } {
                *r = true;
            }
            Status::Ok
        }
        Ok(None) => {
            unsafe {
                *out_image = std::ptr::null_mut();
            }
            if let Some(r) = unsafe { out_recognized.as_mut() } {
                *r = false;
            }
            Status::Ok
        }
        Err(e) => map_error(e),
    }
}

/// Decode a container with the format chosen explicitly, bypassing magic
/// detection. Useful when the caller already knows which container they are
/// pointing at.
///
/// `overrides` may be NULL to use defaults.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_decode_container_as(
    data: *const u8,
    len: usize,
    format: InputFormat,
    overrides: *const InputOverrides,
    out_image: *mut *mut Image,
) -> Status {
    if out_image.is_null() {
        set_last_error("ctt_decode_container_as: out_image is null");
        return Status::NullPointer;
    }
    if data.is_null() && len != 0 {
        set_last_error("ctt_decode_container_as: data is null but len != 0");
        return Status::NullPointer;
    }
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    let overrides_inner: ctt::input::InputOverrides = match unsafe { overrides.as_ref() } {
        Some(o) => (*o).into(),
        None => ctt::input::InputOverrides::default(),
    };

    match ctt::input::decode_container_as(bytes, format.into(), overrides_inner) {
        Ok(image) => {
            unsafe {
                *out_image = Box::into_raw(Box::new(Image(image)));
            }
            Status::Ok
        }
        Err(e) => map_error(e),
    }
}
