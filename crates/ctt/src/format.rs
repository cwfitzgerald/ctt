#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressedFormat {
    Bc1,
    Bc3,
    Bc4,
    Bc5,
    Bc6h,
    Bc7,
    Etc1,
    Astc {
        block_width: u8,
        block_height: u8,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorSpace {
    #[default]
    Srgb,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelComponents {
    R,
    Rg,
    Rgb,
    Rgba,
}

impl PixelComponents {
    pub fn channel_count(self) -> usize {
        match self {
            Self::R => 1,
            Self::Rg => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelFormat {
    pub components: PixelComponents,
    pub color_space: ColorSpace,
}
