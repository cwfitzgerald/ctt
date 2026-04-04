#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressedFormat {
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc4s,
    Bc5,
    Bc5s,
    Bc6h,
    Bc6hSf,
    Bc7,
    Etc1,
    Astc { block_width: u8, block_height: u8 },
}

impl std::fmt::Display for CompressedFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bc1 => f.write_str("BC1"),
            Self::Bc2 => f.write_str("BC2"),
            Self::Bc3 => f.write_str("BC3"),
            Self::Bc4 => f.write_str("BC4"),
            Self::Bc4s => f.write_str("BC4S"),
            Self::Bc5 => f.write_str("BC5"),
            Self::Bc5s => f.write_str("BC5S"),
            Self::Bc6h => f.write_str("BC6H"),
            Self::Bc6hSf => f.write_str("BC6H_SF"),
            Self::Bc7 => f.write_str("BC7"),
            Self::Etc1 => f.write_str("ETC1"),
            Self::Astc {
                block_width,
                block_height,
            } => write!(f, "ASTC {block_width}x{block_height}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorSpace {
    #[default]
    Srgb,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChannelType {
    #[default]
    U8,
    U16,
    F16,
    F32,
}

impl ChannelType {
    pub fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
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
    pub channel_type: ChannelType,
    pub color_space: ColorSpace,
}

impl std::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let components = match self.components {
            PixelComponents::R => "R",
            PixelComponents::Rg => "RG",
            PixelComponents::Rgb => "RGB",
            PixelComponents::Rgba => "RGBA",
        };
        let channel = match self.channel_type {
            ChannelType::U8 => "8",
            ChannelType::U16 => "16",
            ChannelType::F16 => "16f",
            ChannelType::F32 => "32f",
        };
        let cs = match self.color_space {
            ColorSpace::Srgb => "sRGB",
            ColorSpace::Linear => "linear",
        };
        write!(f, "{components}{channel} ({cs})")
    }
}

impl PixelFormat {
    pub fn bytes_per_pixel(self) -> usize {
        self.components.channel_count() * self.channel_type.byte_size()
    }
}
