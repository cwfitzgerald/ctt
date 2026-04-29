pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("invalid swizzle: {0}")]
    InvalidSwizzle(String),

    #[error("cubemap requires exactly 6 faces, got {0}")]
    CubemapFaceCount(usize),

    #[error("cubemap faces must have uniform dimensions")]
    CubemapNonUniformFaces,

    #[error("compression failed: {0}")]
    Compression(String),

    #[error("output encoding error: {0}")]
    OutputEncoding(String),

    #[error("input decoding error: {0}")]
    InputDecoding(String),

    #[error("data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },

    #[error("unsupported pixel format conversion: {0}")]
    UnsupportedConversion(String),

    #[error(
        "lossy auto-conversion from {from} to {to} required by transform '{transform}': {reason}"
    )]
    LossyConversion {
        from: String,
        to: String,
        transform: String,
        reason: String,
    },

    #[error("invalid image: {0}")]
    InvalidImage(String),
}
