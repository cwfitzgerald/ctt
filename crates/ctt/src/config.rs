use crate::format::{ColorSpace, CompressedFormat};
use crate::transform::swizzle::Swizzle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    Dds,
    Ktx2,
}

#[derive(Debug, Clone)]
pub struct CompressConfig {
    pub format: CompressedFormat,
    pub output_format: OutputFormat,
    pub swizzle: Option<Swizzle>,
    pub color_space: ColorSpace,
}
