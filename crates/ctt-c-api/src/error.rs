use std::cell::RefCell;
use std::ffi::{CString, c_char};

/// Status codes returned by ctt entry points.
///
/// `CTT_STATUS_OK` (0) means success. Negative codes are errors. The most
/// recent error message is available via [`ctt_last_error_message`] until
/// the next ctt call on this thread.
///
/// `CTT_STATUS_INTERNAL` signals an unexpected internal failure — most
/// commonly a Rust panic that was caught at the FFI boundary (see the panic
/// note in the header overview). It always leaves a message in the
/// thread-local error slot.
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
    ThreadPoolAlreadyInitialized = -103,
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

/// Run `f`, containing any panic that unwinds out of it.
///
/// Edition 2024 turns an unwind across an `extern "C"` boundary into an
/// immediate process abort, so every entry point that calls into non-trivial
/// ctt core code (decode / convert / encode) routes its body through this
/// shim. A caught panic is recorded in the thread-local error slot and the
/// caller-supplied `sentinel` is returned in its place (`Status::Internal`
/// for status-returning functions, a null pointer / `false` for others).
///
/// The closure is wrapped in [`AssertUnwindSafe`] because these entry points
/// operate on raw pointers and already own the "consumed on failure" contract
/// documented per function; a panic drops any owned locals as the stack
/// unwinds into the `catch_unwind`, so no double-free or leak results.
pub(crate) fn catch_panic<T>(sentinel: T, f: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            set_last_error(format!(
                "internal error: caught panic: {}",
                panic_message(payload.as_ref())
            ));
            sentinel
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn catch_panic_returns_sentinel_and_records_message() {
        clear_last_error();
        let result = catch_panic(Status::Internal, || panic!("ffi test panic"));
        assert_eq!(result, Status::Internal);

        let ptr = ctt_last_error_message();
        assert!(!ptr.is_null());
        let message = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert!(message.contains("ffi test panic"), "got: {message}");
    }

    #[test]
    fn catch_panic_drops_owned_locals_once() {
        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let owned = DropCounter(Arc::clone(&drops));
        let result = catch_panic(false, move || {
            let _owned = owned;
            panic!("drop test");
        });

        assert!(!result);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
