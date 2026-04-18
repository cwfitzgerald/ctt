//! Internal pixel buffer used by the processing pipeline.
//!
//! Pixels are stored as 4-wide lane tuples (`[T; 4]`) to keep them 16-byte aligned
//! and friendly to autovectorization. RGB sources get lane 3 defaulted on load.

use bytemuck::Pod;

/// A 2D pixel buffer with 4 lanes per pixel.
///
/// Internal shape used during the float/integer processing pipelines.
#[derive(Debug, Clone)]
pub struct Buffer<T: Copy + Pod> {
    pub pixels: Vec<[T; 4]>,
    pub width: u32,
    pub height: u32,
}

/// Which internal representation a single conversion run uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    F32,
    F64,
    U32,
    U64,
}
