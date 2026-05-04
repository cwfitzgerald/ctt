use std::cell::RefCell;
use std::ffi::{CString, c_char};

/// Status codes returned by ctt entry points.
///
/// `CTT_STATUS_OK` (0) means success. Negative codes are errors. The most
/// recent error message is available via [`ctt_last_error_message`] until
/// the next ctt call on this thread.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    InvalidDimensions = -1,
    UnsupportedFormat = -2,
    InvalidSwizzle = -3,
    CubemapFaceCount = -4,
    CubemapNonUniformFaces = -5,
    Compression = -6,
    OutputEncoding = -7,
    InputDecoding = -8,
    DataLengthMismatch = -9,
    UnsupportedConversion = -10,
    LossyConversion = -11,
    InvalidImage = -12,
    NullPointer = -100,
    EncoderNotCompiledIn = -101,
    InvalidArgument = -102,
    Internal = -200,
}

impl From<&ctt::Error> for Status {
    fn from(e: &ctt::Error) -> Self {
        match e {
            ctt::Error::InvalidDimensions(_) => Self::InvalidDimensions,
            ctt::Error::UnsupportedFormat(_) => Self::UnsupportedFormat,
            ctt::Error::InvalidSwizzle(_) => Self::InvalidSwizzle,
            ctt::Error::CubemapFaceCount(_) => Self::CubemapFaceCount,
            ctt::Error::CubemapNonUniformFaces => Self::CubemapNonUniformFaces,
            ctt::Error::Compression(_) => Self::Compression,
            ctt::Error::OutputEncoding(_) => Self::OutputEncoding,
            ctt::Error::InputDecoding(_) => Self::InputDecoding,
            ctt::Error::DataLengthMismatch { .. } => Self::DataLengthMismatch,
            ctt::Error::UnsupportedConversion(_) => Self::UnsupportedConversion,
            ctt::Error::LossyConversion { .. } => Self::LossyConversion,
            ctt::Error::InvalidImage(_) => Self::InvalidImage,
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub(crate) fn set_last_error(msg: impl Into<String>) {
    let c =
        CString::new(msg.into()).unwrap_or_else(|_| CString::new("invalid error message").unwrap());
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(c));
}

pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|cell| *cell.borrow_mut() = None);
}

pub(crate) fn map_error(e: ctt::Error) -> Status {
    let status = Status::from(&e);
    set_last_error(e.to_string());
    status
}

/// Pointer to a NUL-terminated UTF-8 string describing the most recent error
/// produced on this thread, or `NULL` if there is no recorded error.
///
/// The pointer is valid only until the next ctt call on this thread; copy
/// the string before making further calls if you need to keep it.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr())
    })
}

/// Clear the thread-local error message slot.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_clear_last_error() {
    clear_last_error();
}
