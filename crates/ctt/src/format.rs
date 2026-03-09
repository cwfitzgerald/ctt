#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressedFormat {
    Bc1,
    Bc3,
    Bc4,
    Bc5,
    Bc6h,
    Bc7,
    Etc1,
    Astc { block_width: u8, block_height: u8 },
}

impl CompressedFormat {
    /// The pixel format that the underlying compressor expects as input.
    ///
    /// `color_space` is passed through — the compressor doesn't change it.
    pub fn required_input_format(self, color_space: ColorSpace) -> PixelFormat {
        match self {
            Self::Bc1 | Self::Bc3 | Self::Bc7 | Self::Etc1 | Self::Astc { .. } => PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U8,
                color_space,
            },
            Self::Bc4 => PixelFormat {
                components: PixelComponents::R,
                channel_type: ChannelType::U8,
                color_space,
            },
            Self::Bc5 => PixelFormat {
                components: PixelComponents::Rg,
                channel_type: ChannelType::U8,
                color_space,
            },
            Self::Bc6h => PixelFormat {
                components: PixelComponents::Rgba,
                channel_type: ChannelType::U16,
                color_space,
            },
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
