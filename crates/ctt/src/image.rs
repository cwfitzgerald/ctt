use crate::format::PixelFormat;

/// A raw, uncompressed image stored as a flat pixel buffer.
#[derive(Debug, Clone)]
pub struct RawImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pixel_format: PixelFormat,
}

/// Describes the layout of one or more images for compression.
///
/// `layers[i][j]` is layer `i`, mip level `j`.
/// A 2D texture has 1 layer; a cubemap has 6.
/// Currently only a single mip level (index 0) is supported.
#[derive(Debug, Clone)]
pub struct ImageLayout {
    pub layers: Vec<Vec<RawImage>>,
    pub is_cubemap: bool,
}
