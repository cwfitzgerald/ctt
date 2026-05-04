use crate::error::set_last_error;
use crate::image::Image;

/// Tag for [`PipelineOutput`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineOutputKind {
    Encoded = 0,
    Raw = 1,
}

/// Opaque handle to a conversion result.
///
/// The output is one of two variants, distinguished by
/// [`ctt_pipeline_output_get_kind`]:
///
/// - `CTT_PIPELINE_OUTPUT_KIND_ENCODED` (any `Container` other than `Raw`):
///   the result is a serialized container file (DDS or KTX2). Read the bytes
///   with [`ctt_pipeline_output_encoded_data`] / [`ctt_pipeline_output_encoded_len`]
///   and copy them out (e.g. write to disk) before destroying the output —
///   the pointer is borrowed and becomes invalid once the output is freed.
///
/// - `CTT_PIPELINE_OUTPUT_KIND_RAW` (only when `Container::Raw` was requested):
///   the result is a processed [`ctt_image_t`]. Call
///   [`ctt_pipeline_output_take_image`] to transfer ownership to the caller;
///   from then on, walk the image with `ctt_image_layer_count` /
///   `ctt_image_mip_count` and the `ctt_image_surface_*` accessors. The taken
///   image must be freed with [`ctt_image_destroy`].
///
/// Always destroy the output with [`ctt_pipeline_output_destroy`] when done,
/// even after taking the image out of a `Raw` output.
pub struct PipelineOutput(pub(crate) Option<ctt::PipelineOutput>);

/// Destroy a pipeline output. `out` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_pipeline_output_destroy(out: *mut PipelineOutput) {
    if out.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(out) });
}

/// Return the variant tag of the output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_pipeline_output_get_kind(
    out: *const PipelineOutput,
) -> PipelineOutputKind {
    let Some(o) = (unsafe { out.as_ref() }) else {
        return PipelineOutputKind::Encoded;
    };
    match &o.0 {
        Some(ctt::PipelineOutput::Encoded(_)) => PipelineOutputKind::Encoded,
        _ => PipelineOutputKind::Raw,
    }
}

/// Pointer to the encoded container bytes. Valid until the output is
/// destroyed. Returns NULL when `out` is null or the tag is `Raw`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_pipeline_output_encoded_data(out: *const PipelineOutput) -> *const u8 {
    let Some(o) = (unsafe { out.as_ref() }) else {
        return std::ptr::null();
    };
    match &o.0 {
        Some(ctt::PipelineOutput::Encoded(bytes)) => bytes.as_ptr(),
        _ => std::ptr::null(),
    }
}

/// Length in bytes of the encoded data, or `0` if the tag is `Raw`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_pipeline_output_encoded_len(out: *const PipelineOutput) -> usize {
    let Some(o) = (unsafe { out.as_ref() }) else {
        return 0;
    };
    match &o.0 {
        Some(ctt::PipelineOutput::Encoded(bytes)) => bytes.len(),
        _ => 0,
    }
}

/// Take the [`ctt_image_t`] out of a `Raw` output, transferring ownership
/// to the caller. The output handle remains live for tag queries and must
/// still be destroyed via [`ctt_pipeline_output_destroy`], but a second
/// call to this function returns NULL.
///
/// Returns NULL if `out` is null, the tag is not `Raw`, or the image has
/// already been taken.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_pipeline_output_take_image(out: *mut PipelineOutput) -> *mut Image {
    let Some(o) = (unsafe { out.as_mut() }) else {
        set_last_error("ctt_pipeline_output_take_image: output is null");
        return std::ptr::null_mut();
    };
    match o.0.take() {
        Some(ctt::PipelineOutput::Raw(img)) => Box::into_raw(Box::new(Image(img))),
        Some(other) => {
            // Put it back; not a Raw output.
            o.0 = Some(other);
            set_last_error("ctt_pipeline_output_take_image: output tag is not Raw");
            std::ptr::null_mut()
        }
        None => {
            set_last_error("ctt_pipeline_output_take_image: image already taken");
            std::ptr::null_mut()
        }
    }
}
