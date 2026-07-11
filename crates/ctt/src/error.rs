pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced anywhere in the ctt pipeline.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Image dimensions are zero, negative, or otherwise invalid for the operation.
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// The pixel format is not supported by the requested operation or container.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    /// A swizzle specification is malformed or references invalid channels.
    #[error("invalid swizzle: {0}")]
    InvalidSwizzle(String),

    /// A cubemap was given a number of faces that is not exactly 6 (per cube).
    #[error("cubemap requires exactly 6 faces, got {0}")]
    CubemapFaceCount(usize),

    /// The faces of a cubemap do not all share the same dimensions.
    #[error("cubemap faces must have uniform dimensions")]
    CubemapNonUniformFaces,

    /// Block or supercompression of image data failed.
    #[error("compression failed: {0}")]
    Compression(String),

    /// Serializing an [`Image`](crate::surface::Image) into a container failed.
    #[error("output encoding error: {0}")]
    OutputEncoding(String),

    /// Parsing or decoding an input container (KTX2/DDS) failed, including
    /// malformed, truncated, or maliciously crafted headers.
    #[error("input decoding error: {0}")]
    InputDecoding(String),

    /// A buffer's length did not match the size implied by its format and dimensions.
    #[error("data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },

    /// No conversion path exists between the source and destination pixel formats.
    #[error("unsupported pixel format conversion: {0}")]
    UnsupportedConversion(String),

    /// A transform would require a lossy automatic format conversion that was not permitted.
    #[error(
        "lossy auto-conversion from {from} to {to} required by transform '{transform}': {reason}"
    )]
    LossyConversion {
        /// The source format the conversion would start from.
        from: String,
        /// The destination format the conversion would produce.
        to: String,
        /// The transform that triggered the conversion requirement.
        transform: String,
        /// Why the conversion is considered lossy.
        reason: String,
    },

    /// The image failed a structural validity check (e.g. inconsistent surfaces).
    #[error("invalid image: {0}")]
    InvalidImage(String),
}
