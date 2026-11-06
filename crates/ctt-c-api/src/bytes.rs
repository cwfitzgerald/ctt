/// Opaque handle to an owned byte buffer, such as an encoded container file.
///
/// Free with [`ctt_bytes_destroy`].
pub struct Bytes(pub(crate) Vec<u8>);

/// Destroy a byte buffer. `bytes` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_bytes_destroy(bytes: *mut Bytes) {
    if bytes.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(bytes) });
}

/// Pointer to the bytes. Valid until the handle is destroyed. Returns NULL
/// when `bytes` is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_bytes_data(bytes: *const Bytes) -> *const u8 {
    match unsafe { bytes.as_ref() } {
        Some(b) => b.0.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Length of the buffer in bytes, or `0` when `bytes` is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_bytes_len(bytes: *const Bytes) -> usize {
    match unsafe { bytes.as_ref() } {
        Some(b) => b.0.len(),
        None => 0,
    }
}
