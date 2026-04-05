use crate::surface::ColorSpace;

/// Channel data type for uncompressed formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelKind {
    U8,
    U16,
    F16,
    F32,
    U32,
}

impl ChannelKind {
    pub fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::F16 => 2,
            Self::F32 | Self::U32 => 4,
        }
    }
}

/// Extension trait providing introspection and utility methods on [`ktx2::Format`].
pub trait FormatExt {
    /// Returns `true` if this is a block-compressed format (BC, ETC2, ASTC, EAC).
    fn is_compressed(&self) -> bool;

    /// Bytes per pixel for uncompressed formats. Returns `None` for compressed or unknown formats.
    fn bytes_per_pixel(&self) -> Option<usize>;

    /// Block dimensions `(width, height)` for compressed formats.
    /// Returns `None` for uncompressed or unknown formats.
    fn block_size(&self) -> Option<(u8, u8)>;

    /// Bytes per block for compressed formats. Returns `None` for uncompressed or unknown formats.
    fn bytes_per_block(&self) -> Option<usize>;

    /// Number of color channels. Returns `None` for unknown formats.
    fn channel_count(&self) -> Option<usize>;

    /// Channel data type for uncompressed formats. Returns `None` for compressed or unknown formats.
    fn channel_kind(&self) -> Option<ChannelKind>;

    /// Strip any sRGB variant to its UNORM/UFLOAT base and return the color space separately.
    ///
    /// For example, `R8G8B8A8_SRGB` becomes `(R8G8B8A8_UNORM, Srgb)`.
    /// Non-sRGB formats return `(self, Linear)`.
    fn normalize(&self) -> (ktx2::Format, ColorSpace);

    /// Recombine a base format with a color space for container output.
    ///
    /// Returns the sRGB variant when `cs == Srgb` and one exists, otherwise returns `self`.
    fn denormalize(&self, cs: ColorSpace) -> ktx2::Format;
}

impl FormatExt for ktx2::Format {
    fn is_compressed(&self) -> bool {
        let v = self.value();
        // BC1 RGB through ASTC 12x12 sRGB (includes ETC2/EAC at 147-156)
        (131..=184).contains(&v)
            // ASTC SFLOAT extension
            || (1000066000..=1000066013).contains(&v)
    }

    fn bytes_per_pixel(&self) -> Option<usize> {
        use ktx2::Format as F;
        Some(match *self {
            // 1 byte
            F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT | F::R8_SRGB => 1,

            // 2 bytes
            F::R8G8_UNORM | F::R8G8_SNORM | F::R8G8_UINT | F::R8G8_SINT | F::R8G8_SRGB => 2,
            F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT | F::R16_SFLOAT => 2,
            F::R4G4B4A4_UNORM_PACK16
            | F::B4G4R4A4_UNORM_PACK16
            | F::R5G6B5_UNORM_PACK16
            | F::B5G6R5_UNORM_PACK16
            | F::R5G5B5A1_UNORM_PACK16
            | F::B5G5R5A1_UNORM_PACK16
            | F::A1R5G5B5_UNORM_PACK16 => 2,

            // 3 bytes
            F::R8G8B8_UNORM
            | F::R8G8B8_SNORM
            | F::R8G8B8_UINT
            | F::R8G8B8_SINT
            | F::R8G8B8_SRGB
            | F::B8G8R8_UNORM
            | F::B8G8R8_SNORM
            | F::B8G8R8_UINT
            | F::B8G8R8_SINT
            | F::B8G8R8_SRGB => 3,

            // 4 bytes
            F::R8G8B8A8_UNORM
            | F::R8G8B8A8_SNORM
            | F::R8G8B8A8_UINT
            | F::R8G8B8A8_SINT
            | F::R8G8B8A8_SRGB
            | F::B8G8R8A8_UNORM
            | F::B8G8R8A8_SNORM
            | F::B8G8R8A8_UINT
            | F::B8G8R8A8_SINT
            | F::B8G8R8A8_SRGB => 4,
            F::A2R10G10B10_UNORM_PACK32
            | F::A2R10G10B10_SNORM_PACK32
            | F::A2R10G10B10_UINT_PACK32
            | F::A2R10G10B10_SINT_PACK32
            | F::A2B10G10R10_UNORM_PACK32
            | F::A2B10G10R10_SNORM_PACK32
            | F::A2B10G10R10_UINT_PACK32
            | F::A2B10G10R10_SINT_PACK32 => 4,
            F::R16G16_UNORM
            | F::R16G16_SNORM
            | F::R16G16_UINT
            | F::R16G16_SINT
            | F::R16G16_SFLOAT => 4,
            F::R32_UINT | F::R32_SINT | F::R32_SFLOAT => 4,
            F::B10G11R11_UFLOAT_PACK32 | F::E5B9G9R9_UFLOAT_PACK32 => 4,

            // 6 bytes
            F::R16G16B16_UNORM
            | F::R16G16B16_SNORM
            | F::R16G16B16_UINT
            | F::R16G16B16_SINT
            | F::R16G16B16_SFLOAT => 6,

            // 8 bytes
            F::R16G16B16A16_UNORM
            | F::R16G16B16A16_SNORM
            | F::R16G16B16A16_UINT
            | F::R16G16B16A16_SINT
            | F::R16G16B16A16_SFLOAT => 8,
            F::R32G32_UINT | F::R32G32_SINT | F::R32G32_SFLOAT => 8,
            F::R64_UINT | F::R64_SINT | F::R64_SFLOAT => 8,

            // 12 bytes
            F::R32G32B32_UINT | F::R32G32B32_SINT | F::R32G32B32_SFLOAT => 12,

            // 16 bytes
            F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => 16,
            F::R64G64_UINT | F::R64G64_SINT | F::R64G64_SFLOAT => 16,

            // 24 bytes
            F::R64G64B64_UINT | F::R64G64B64_SINT | F::R64G64B64_SFLOAT => 24,

            // 32 bytes
            F::R64G64B64A64_UINT | F::R64G64B64A64_SINT | F::R64G64B64A64_SFLOAT => 32,

            // Depth/stencil
            F::D16_UNORM => 2,
            F::D32_SFLOAT => 4,
            F::S8_UINT => 1,
            F::X8_D24_UNORM_PACK32 | F::D24_UNORM_S8_UINT => 4,
            F::D16_UNORM_S8_UINT => 3,
            F::D32_SFLOAT_S8_UINT => 5,

            _ => return None,
        })
    }

    fn block_size(&self) -> Option<(u8, u8)> {
        use ktx2::Format as F;
        Some(match *self {
            // BC formats: all 4x4
            F::BC1_RGB_UNORM_BLOCK
            | F::BC1_RGB_SRGB_BLOCK
            | F::BC1_RGBA_UNORM_BLOCK
            | F::BC1_RGBA_SRGB_BLOCK
            | F::BC2_UNORM_BLOCK
            | F::BC2_SRGB_BLOCK
            | F::BC3_UNORM_BLOCK
            | F::BC3_SRGB_BLOCK
            | F::BC4_UNORM_BLOCK
            | F::BC4_SNORM_BLOCK
            | F::BC5_UNORM_BLOCK
            | F::BC5_SNORM_BLOCK
            | F::BC6H_UFLOAT_BLOCK
            | F::BC6H_SFLOAT_BLOCK
            | F::BC7_UNORM_BLOCK
            | F::BC7_SRGB_BLOCK => (4, 4),

            // ETC2 / EAC: all 4x4
            F::ETC2_R8G8B8_UNORM_BLOCK
            | F::ETC2_R8G8B8_SRGB_BLOCK
            | F::ETC2_R8G8B8A1_UNORM_BLOCK
            | F::ETC2_R8G8B8A1_SRGB_BLOCK
            | F::ETC2_R8G8B8A8_UNORM_BLOCK
            | F::ETC2_R8G8B8A8_SRGB_BLOCK
            | F::EAC_R11_UNORM_BLOCK
            | F::EAC_R11_SNORM_BLOCK
            | F::EAC_R11G11_UNORM_BLOCK
            | F::EAC_R11G11_SNORM_BLOCK => (4, 4),

            // ASTC
            F::ASTC_4x4_UNORM_BLOCK | F::ASTC_4x4_SRGB_BLOCK | F::ASTC_4x4_SFLOAT_BLOCK => (4, 4),
            F::ASTC_5x4_UNORM_BLOCK | F::ASTC_5x4_SRGB_BLOCK | F::ASTC_5x4_SFLOAT_BLOCK => (5, 4),
            F::ASTC_5x5_UNORM_BLOCK | F::ASTC_5x5_SRGB_BLOCK | F::ASTC_5x5_SFLOAT_BLOCK => (5, 5),
            F::ASTC_6x5_UNORM_BLOCK | F::ASTC_6x5_SRGB_BLOCK | F::ASTC_6x5_SFLOAT_BLOCK => (6, 5),
            F::ASTC_6x6_UNORM_BLOCK | F::ASTC_6x6_SRGB_BLOCK | F::ASTC_6x6_SFLOAT_BLOCK => (6, 6),
            F::ASTC_8x5_UNORM_BLOCK | F::ASTC_8x5_SRGB_BLOCK | F::ASTC_8x5_SFLOAT_BLOCK => (8, 5),
            F::ASTC_8x6_UNORM_BLOCK | F::ASTC_8x6_SRGB_BLOCK | F::ASTC_8x6_SFLOAT_BLOCK => (8, 6),
            F::ASTC_8x8_UNORM_BLOCK | F::ASTC_8x8_SRGB_BLOCK | F::ASTC_8x8_SFLOAT_BLOCK => (8, 8),
            F::ASTC_10x5_UNORM_BLOCK | F::ASTC_10x5_SRGB_BLOCK | F::ASTC_10x5_SFLOAT_BLOCK => {
                (10, 5)
            }
            F::ASTC_10x6_UNORM_BLOCK | F::ASTC_10x6_SRGB_BLOCK | F::ASTC_10x6_SFLOAT_BLOCK => {
                (10, 6)
            }
            F::ASTC_10x8_UNORM_BLOCK | F::ASTC_10x8_SRGB_BLOCK | F::ASTC_10x8_SFLOAT_BLOCK => {
                (10, 8)
            }
            F::ASTC_10x10_UNORM_BLOCK | F::ASTC_10x10_SRGB_BLOCK | F::ASTC_10x10_SFLOAT_BLOCK => {
                (10, 10)
            }
            F::ASTC_12x10_UNORM_BLOCK | F::ASTC_12x10_SRGB_BLOCK | F::ASTC_12x10_SFLOAT_BLOCK => {
                (12, 10)
            }
            F::ASTC_12x12_UNORM_BLOCK | F::ASTC_12x12_SRGB_BLOCK | F::ASTC_12x12_SFLOAT_BLOCK => {
                (12, 12)
            }

            _ => return None,
        })
    }

    fn bytes_per_block(&self) -> Option<usize> {
        use ktx2::Format as F;
        Some(match *self {
            // 8 bytes per block
            F::BC1_RGB_UNORM_BLOCK
            | F::BC1_RGB_SRGB_BLOCK
            | F::BC1_RGBA_UNORM_BLOCK
            | F::BC1_RGBA_SRGB_BLOCK
            | F::BC4_UNORM_BLOCK
            | F::BC4_SNORM_BLOCK => 8,

            F::ETC2_R8G8B8_UNORM_BLOCK
            | F::ETC2_R8G8B8_SRGB_BLOCK
            | F::ETC2_R8G8B8A1_UNORM_BLOCK
            | F::ETC2_R8G8B8A1_SRGB_BLOCK
            | F::EAC_R11_UNORM_BLOCK
            | F::EAC_R11_SNORM_BLOCK => 8,

            // 16 bytes per block
            F::BC2_UNORM_BLOCK
            | F::BC2_SRGB_BLOCK
            | F::BC3_UNORM_BLOCK
            | F::BC3_SRGB_BLOCK
            | F::BC5_UNORM_BLOCK
            | F::BC5_SNORM_BLOCK
            | F::BC6H_UFLOAT_BLOCK
            | F::BC6H_SFLOAT_BLOCK
            | F::BC7_UNORM_BLOCK
            | F::BC7_SRGB_BLOCK => 16,

            F::ETC2_R8G8B8A8_UNORM_BLOCK
            | F::ETC2_R8G8B8A8_SRGB_BLOCK
            | F::EAC_R11G11_UNORM_BLOCK
            | F::EAC_R11G11_SNORM_BLOCK => 16,

            // ASTC: always 16 bytes per block regardless of block size
            F::ASTC_4x4_UNORM_BLOCK
            | F::ASTC_4x4_SRGB_BLOCK
            | F::ASTC_4x4_SFLOAT_BLOCK
            | F::ASTC_5x4_UNORM_BLOCK
            | F::ASTC_5x4_SRGB_BLOCK
            | F::ASTC_5x4_SFLOAT_BLOCK
            | F::ASTC_5x5_UNORM_BLOCK
            | F::ASTC_5x5_SRGB_BLOCK
            | F::ASTC_5x5_SFLOAT_BLOCK
            | F::ASTC_6x5_UNORM_BLOCK
            | F::ASTC_6x5_SRGB_BLOCK
            | F::ASTC_6x5_SFLOAT_BLOCK
            | F::ASTC_6x6_UNORM_BLOCK
            | F::ASTC_6x6_SRGB_BLOCK
            | F::ASTC_6x6_SFLOAT_BLOCK
            | F::ASTC_8x5_UNORM_BLOCK
            | F::ASTC_8x5_SRGB_BLOCK
            | F::ASTC_8x5_SFLOAT_BLOCK
            | F::ASTC_8x6_UNORM_BLOCK
            | F::ASTC_8x6_SRGB_BLOCK
            | F::ASTC_8x6_SFLOAT_BLOCK
            | F::ASTC_8x8_UNORM_BLOCK
            | F::ASTC_8x8_SRGB_BLOCK
            | F::ASTC_8x8_SFLOAT_BLOCK
            | F::ASTC_10x5_UNORM_BLOCK
            | F::ASTC_10x5_SRGB_BLOCK
            | F::ASTC_10x5_SFLOAT_BLOCK
            | F::ASTC_10x6_UNORM_BLOCK
            | F::ASTC_10x6_SRGB_BLOCK
            | F::ASTC_10x6_SFLOAT_BLOCK
            | F::ASTC_10x8_UNORM_BLOCK
            | F::ASTC_10x8_SRGB_BLOCK
            | F::ASTC_10x8_SFLOAT_BLOCK
            | F::ASTC_10x10_UNORM_BLOCK
            | F::ASTC_10x10_SRGB_BLOCK
            | F::ASTC_10x10_SFLOAT_BLOCK
            | F::ASTC_12x10_UNORM_BLOCK
            | F::ASTC_12x10_SRGB_BLOCK
            | F::ASTC_12x10_SFLOAT_BLOCK
            | F::ASTC_12x12_UNORM_BLOCK
            | F::ASTC_12x12_SRGB_BLOCK
            | F::ASTC_12x12_SFLOAT_BLOCK => 16,

            _ => return None,
        })
    }

    fn channel_count(&self) -> Option<usize> {
        use ktx2::Format as F;
        Some(match *self {
            // 1 channel
            F::R8_UNORM
            | F::R8_SNORM
            | F::R8_UINT
            | F::R8_SINT
            | F::R8_SRGB
            | F::R16_UNORM
            | F::R16_SNORM
            | F::R16_UINT
            | F::R16_SINT
            | F::R16_SFLOAT
            | F::R32_UINT
            | F::R32_SINT
            | F::R32_SFLOAT
            | F::R64_UINT
            | F::R64_SINT
            | F::R64_SFLOAT => 1,

            // 1 channel compressed
            F::BC4_UNORM_BLOCK | F::BC4_SNORM_BLOCK => 1,
            F::EAC_R11_UNORM_BLOCK | F::EAC_R11_SNORM_BLOCK => 1,

            // 2 channels
            F::R8G8_UNORM
            | F::R8G8_SNORM
            | F::R8G8_UINT
            | F::R8G8_SINT
            | F::R8G8_SRGB
            | F::R16G16_UNORM
            | F::R16G16_SNORM
            | F::R16G16_UINT
            | F::R16G16_SINT
            | F::R16G16_SFLOAT
            | F::R32G32_UINT
            | F::R32G32_SINT
            | F::R32G32_SFLOAT
            | F::R64G64_UINT
            | F::R64G64_SINT
            | F::R64G64_SFLOAT => 2,

            // 2 channel compressed
            F::BC5_UNORM_BLOCK | F::BC5_SNORM_BLOCK => 2,
            F::EAC_R11G11_UNORM_BLOCK | F::EAC_R11G11_SNORM_BLOCK => 2,

            // 3 channels
            F::R8G8B8_UNORM
            | F::R8G8B8_SNORM
            | F::R8G8B8_UINT
            | F::R8G8B8_SINT
            | F::R8G8B8_SRGB
            | F::B8G8R8_UNORM
            | F::B8G8R8_SNORM
            | F::B8G8R8_UINT
            | F::B8G8R8_SINT
            | F::B8G8R8_SRGB
            | F::R16G16B16_UNORM
            | F::R16G16B16_SNORM
            | F::R16G16B16_UINT
            | F::R16G16B16_SINT
            | F::R16G16B16_SFLOAT
            | F::R32G32B32_UINT
            | F::R32G32B32_SINT
            | F::R32G32B32_SFLOAT
            | F::R64G64B64_UINT
            | F::R64G64B64_SINT
            | F::R64G64B64_SFLOAT
            | F::B10G11R11_UFLOAT_PACK32
            | F::E5B9G9R9_UFLOAT_PACK32
            | F::R5G6B5_UNORM_PACK16
            | F::B5G6R5_UNORM_PACK16 => 3,

            // 3 channel compressed
            F::BC1_RGB_UNORM_BLOCK
            | F::BC1_RGB_SRGB_BLOCK
            | F::BC6H_UFLOAT_BLOCK
            | F::BC6H_SFLOAT_BLOCK => 3,
            F::ETC2_R8G8B8_UNORM_BLOCK | F::ETC2_R8G8B8_SRGB_BLOCK => 3,

            // 4 channels
            F::R8G8B8A8_UNORM
            | F::R8G8B8A8_SNORM
            | F::R8G8B8A8_UINT
            | F::R8G8B8A8_SINT
            | F::R8G8B8A8_SRGB
            | F::B8G8R8A8_UNORM
            | F::B8G8R8A8_SNORM
            | F::B8G8R8A8_UINT
            | F::B8G8R8A8_SINT
            | F::B8G8R8A8_SRGB
            | F::A2R10G10B10_UNORM_PACK32
            | F::A2R10G10B10_SNORM_PACK32
            | F::A2R10G10B10_UINT_PACK32
            | F::A2R10G10B10_SINT_PACK32
            | F::A2B10G10R10_UNORM_PACK32
            | F::A2B10G10R10_SNORM_PACK32
            | F::A2B10G10R10_UINT_PACK32
            | F::A2B10G10R10_SINT_PACK32
            | F::R16G16B16A16_UNORM
            | F::R16G16B16A16_SNORM
            | F::R16G16B16A16_UINT
            | F::R16G16B16A16_SINT
            | F::R16G16B16A16_SFLOAT
            | F::R32G32B32A32_UINT
            | F::R32G32B32A32_SINT
            | F::R32G32B32A32_SFLOAT
            | F::R64G64B64A64_UINT
            | F::R64G64B64A64_SINT
            | F::R64G64B64A64_SFLOAT
            | F::R4G4B4A4_UNORM_PACK16
            | F::B4G4R4A4_UNORM_PACK16
            | F::R5G5B5A1_UNORM_PACK16
            | F::B5G5R5A1_UNORM_PACK16
            | F::A1R5G5B5_UNORM_PACK16 => 4,

            // 4 channel compressed
            F::BC1_RGBA_UNORM_BLOCK
            | F::BC1_RGBA_SRGB_BLOCK
            | F::BC2_UNORM_BLOCK
            | F::BC2_SRGB_BLOCK
            | F::BC3_UNORM_BLOCK
            | F::BC3_SRGB_BLOCK
            | F::BC7_UNORM_BLOCK
            | F::BC7_SRGB_BLOCK => 4,
            F::ETC2_R8G8B8A1_UNORM_BLOCK
            | F::ETC2_R8G8B8A1_SRGB_BLOCK
            | F::ETC2_R8G8B8A8_UNORM_BLOCK
            | F::ETC2_R8G8B8A8_SRGB_BLOCK => 4,

            // ASTC: always 4 channels
            F::ASTC_4x4_UNORM_BLOCK
            | F::ASTC_4x4_SRGB_BLOCK
            | F::ASTC_4x4_SFLOAT_BLOCK
            | F::ASTC_5x4_UNORM_BLOCK
            | F::ASTC_5x4_SRGB_BLOCK
            | F::ASTC_5x4_SFLOAT_BLOCK
            | F::ASTC_5x5_UNORM_BLOCK
            | F::ASTC_5x5_SRGB_BLOCK
            | F::ASTC_5x5_SFLOAT_BLOCK
            | F::ASTC_6x5_UNORM_BLOCK
            | F::ASTC_6x5_SRGB_BLOCK
            | F::ASTC_6x5_SFLOAT_BLOCK
            | F::ASTC_6x6_UNORM_BLOCK
            | F::ASTC_6x6_SRGB_BLOCK
            | F::ASTC_6x6_SFLOAT_BLOCK
            | F::ASTC_8x5_UNORM_BLOCK
            | F::ASTC_8x5_SRGB_BLOCK
            | F::ASTC_8x5_SFLOAT_BLOCK
            | F::ASTC_8x6_UNORM_BLOCK
            | F::ASTC_8x6_SRGB_BLOCK
            | F::ASTC_8x6_SFLOAT_BLOCK
            | F::ASTC_8x8_UNORM_BLOCK
            | F::ASTC_8x8_SRGB_BLOCK
            | F::ASTC_8x8_SFLOAT_BLOCK
            | F::ASTC_10x5_UNORM_BLOCK
            | F::ASTC_10x5_SRGB_BLOCK
            | F::ASTC_10x5_SFLOAT_BLOCK
            | F::ASTC_10x6_UNORM_BLOCK
            | F::ASTC_10x6_SRGB_BLOCK
            | F::ASTC_10x6_SFLOAT_BLOCK
            | F::ASTC_10x8_UNORM_BLOCK
            | F::ASTC_10x8_SRGB_BLOCK
            | F::ASTC_10x8_SFLOAT_BLOCK
            | F::ASTC_10x10_UNORM_BLOCK
            | F::ASTC_10x10_SRGB_BLOCK
            | F::ASTC_10x10_SFLOAT_BLOCK
            | F::ASTC_12x10_UNORM_BLOCK
            | F::ASTC_12x10_SRGB_BLOCK
            | F::ASTC_12x10_SFLOAT_BLOCK
            | F::ASTC_12x12_UNORM_BLOCK
            | F::ASTC_12x12_SRGB_BLOCK
            | F::ASTC_12x12_SFLOAT_BLOCK => 4,

            _ => return None,
        })
    }

    fn normalize(&self) -> (ktx2::Format, ColorSpace) {
        use ktx2::Format as F;
        match *self {
            // Uncompressed sRGB -> UNORM
            F::R8_SRGB => (F::R8_UNORM, ColorSpace::Srgb),
            F::R8G8_SRGB => (F::R8G8_UNORM, ColorSpace::Srgb),
            F::R8G8B8_SRGB => (F::R8G8B8_UNORM, ColorSpace::Srgb),
            F::B8G8R8_SRGB => (F::B8G8R8_UNORM, ColorSpace::Srgb),
            F::R8G8B8A8_SRGB => (F::R8G8B8A8_UNORM, ColorSpace::Srgb),
            F::B8G8R8A8_SRGB => (F::B8G8R8A8_UNORM, ColorSpace::Srgb),

            // BC sRGB -> UNORM
            F::BC1_RGB_SRGB_BLOCK => (F::BC1_RGB_UNORM_BLOCK, ColorSpace::Srgb),
            F::BC1_RGBA_SRGB_BLOCK => (F::BC1_RGBA_UNORM_BLOCK, ColorSpace::Srgb),
            F::BC2_SRGB_BLOCK => (F::BC2_UNORM_BLOCK, ColorSpace::Srgb),
            F::BC3_SRGB_BLOCK => (F::BC3_UNORM_BLOCK, ColorSpace::Srgb),
            F::BC7_SRGB_BLOCK => (F::BC7_UNORM_BLOCK, ColorSpace::Srgb),

            // ETC2 sRGB -> UNORM
            F::ETC2_R8G8B8_SRGB_BLOCK => (F::ETC2_R8G8B8_UNORM_BLOCK, ColorSpace::Srgb),
            F::ETC2_R8G8B8A1_SRGB_BLOCK => (F::ETC2_R8G8B8A1_UNORM_BLOCK, ColorSpace::Srgb),
            F::ETC2_R8G8B8A8_SRGB_BLOCK => (F::ETC2_R8G8B8A8_UNORM_BLOCK, ColorSpace::Srgb),

            // ASTC sRGB -> UNORM
            F::ASTC_4x4_SRGB_BLOCK => (F::ASTC_4x4_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_5x4_SRGB_BLOCK => (F::ASTC_5x4_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_5x5_SRGB_BLOCK => (F::ASTC_5x5_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_6x5_SRGB_BLOCK => (F::ASTC_6x5_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_6x6_SRGB_BLOCK => (F::ASTC_6x6_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_8x5_SRGB_BLOCK => (F::ASTC_8x5_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_8x6_SRGB_BLOCK => (F::ASTC_8x6_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_8x8_SRGB_BLOCK => (F::ASTC_8x8_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_10x5_SRGB_BLOCK => (F::ASTC_10x5_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_10x6_SRGB_BLOCK => (F::ASTC_10x6_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_10x8_SRGB_BLOCK => (F::ASTC_10x8_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_10x10_SRGB_BLOCK => (F::ASTC_10x10_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_12x10_SRGB_BLOCK => (F::ASTC_12x10_UNORM_BLOCK, ColorSpace::Srgb),
            F::ASTC_12x12_SRGB_BLOCK => (F::ASTC_12x12_UNORM_BLOCK, ColorSpace::Srgb),

            // Everything else is already a base format
            other => (other, ColorSpace::Linear),
        }
    }

    fn denormalize(&self, cs: ColorSpace) -> ktx2::Format {
        if cs == ColorSpace::Linear {
            return *self;
        }

        use ktx2::Format as F;
        match *self {
            // Uncompressed UNORM -> sRGB
            F::R8_UNORM => F::R8_SRGB,
            F::R8G8_UNORM => F::R8G8_SRGB,
            F::R8G8B8_UNORM => F::R8G8B8_SRGB,
            F::B8G8R8_UNORM => F::B8G8R8_SRGB,
            F::R8G8B8A8_UNORM => F::R8G8B8A8_SRGB,
            F::B8G8R8A8_UNORM => F::B8G8R8A8_SRGB,

            // BC UNORM -> sRGB
            F::BC1_RGB_UNORM_BLOCK => F::BC1_RGB_SRGB_BLOCK,
            F::BC1_RGBA_UNORM_BLOCK => F::BC1_RGBA_SRGB_BLOCK,
            F::BC2_UNORM_BLOCK => F::BC2_SRGB_BLOCK,
            F::BC3_UNORM_BLOCK => F::BC3_SRGB_BLOCK,
            F::BC7_UNORM_BLOCK => F::BC7_SRGB_BLOCK,

            // ETC2 UNORM -> sRGB
            F::ETC2_R8G8B8_UNORM_BLOCK => F::ETC2_R8G8B8_SRGB_BLOCK,
            F::ETC2_R8G8B8A1_UNORM_BLOCK => F::ETC2_R8G8B8A1_SRGB_BLOCK,
            F::ETC2_R8G8B8A8_UNORM_BLOCK => F::ETC2_R8G8B8A8_SRGB_BLOCK,

            // ASTC UNORM -> sRGB
            F::ASTC_4x4_UNORM_BLOCK => F::ASTC_4x4_SRGB_BLOCK,
            F::ASTC_5x4_UNORM_BLOCK => F::ASTC_5x4_SRGB_BLOCK,
            F::ASTC_5x5_UNORM_BLOCK => F::ASTC_5x5_SRGB_BLOCK,
            F::ASTC_6x5_UNORM_BLOCK => F::ASTC_6x5_SRGB_BLOCK,
            F::ASTC_6x6_UNORM_BLOCK => F::ASTC_6x6_SRGB_BLOCK,
            F::ASTC_8x5_UNORM_BLOCK => F::ASTC_8x5_SRGB_BLOCK,
            F::ASTC_8x6_UNORM_BLOCK => F::ASTC_8x6_SRGB_BLOCK,
            F::ASTC_8x8_UNORM_BLOCK => F::ASTC_8x8_SRGB_BLOCK,
            F::ASTC_10x5_UNORM_BLOCK => F::ASTC_10x5_SRGB_BLOCK,
            F::ASTC_10x6_UNORM_BLOCK => F::ASTC_10x6_SRGB_BLOCK,
            F::ASTC_10x8_UNORM_BLOCK => F::ASTC_10x8_SRGB_BLOCK,
            F::ASTC_10x10_UNORM_BLOCK => F::ASTC_10x10_SRGB_BLOCK,
            F::ASTC_12x10_UNORM_BLOCK => F::ASTC_12x10_SRGB_BLOCK,
            F::ASTC_12x12_UNORM_BLOCK => F::ASTC_12x12_SRGB_BLOCK,

            // No sRGB variant exists — return as-is
            other => other,
        }
    }

    fn channel_kind(&self) -> Option<ChannelKind> {
        use ktx2::Format as F;
        Some(match *self {
            F::R8_UNORM
            | F::R8_SNORM
            | F::R8_UINT
            | F::R8_SINT
            | F::R8_SRGB
            | F::R8G8_UNORM
            | F::R8G8_SNORM
            | F::R8G8_UINT
            | F::R8G8_SINT
            | F::R8G8_SRGB
            | F::R8G8B8_UNORM
            | F::R8G8B8_SNORM
            | F::R8G8B8_UINT
            | F::R8G8B8_SINT
            | F::R8G8B8_SRGB
            | F::B8G8R8_UNORM
            | F::B8G8R8_SNORM
            | F::B8G8R8_UINT
            | F::B8G8R8_SINT
            | F::B8G8R8_SRGB
            | F::R8G8B8A8_UNORM
            | F::R8G8B8A8_SNORM
            | F::R8G8B8A8_UINT
            | F::R8G8B8A8_SINT
            | F::R8G8B8A8_SRGB
            | F::B8G8R8A8_UNORM
            | F::B8G8R8A8_SNORM
            | F::B8G8R8A8_UINT
            | F::B8G8R8A8_SINT
            | F::B8G8R8A8_SRGB => ChannelKind::U8,

            F::R16_UNORM
            | F::R16_SNORM
            | F::R16_UINT
            | F::R16_SINT
            | F::R16G16_UNORM
            | F::R16G16_SNORM
            | F::R16G16_UINT
            | F::R16G16_SINT
            | F::R16G16B16_UNORM
            | F::R16G16B16_SNORM
            | F::R16G16B16_UINT
            | F::R16G16B16_SINT
            | F::R16G16B16A16_UNORM
            | F::R16G16B16A16_SNORM
            | F::R16G16B16A16_UINT
            | F::R16G16B16A16_SINT => ChannelKind::U16,

            F::R16_SFLOAT | F::R16G16_SFLOAT | F::R16G16B16_SFLOAT | F::R16G16B16A16_SFLOAT => {
                ChannelKind::F16
            }

            F::R32_SFLOAT | F::R32G32_SFLOAT | F::R32G32B32_SFLOAT | F::R32G32B32A32_SFLOAT => {
                ChannelKind::F32
            }

            F::R32_UINT
            | F::R32_SINT
            | F::R32G32_UINT
            | F::R32G32_SINT
            | F::R32G32B32_UINT
            | F::R32G32B32_SINT
            | F::R32G32B32A32_UINT
            | F::R32G32B32A32_SINT => ChannelKind::U32,

            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktx2::Format as F;

    #[test]
    fn normalize_srgb_strips_color_space() {
        let (base, cs) = F::R8G8B8A8_SRGB.normalize();
        assert_eq!(base, F::R8G8B8A8_UNORM);
        assert_eq!(cs, ColorSpace::Srgb);
    }

    #[test]
    fn normalize_linear_is_identity() {
        let (base, cs) = F::R8G8B8A8_UNORM.normalize();
        assert_eq!(base, F::R8G8B8A8_UNORM);
        assert_eq!(cs, ColorSpace::Linear);
    }

    #[test]
    fn denormalize_roundtrip() {
        let original = F::BC7_SRGB_BLOCK;
        let (base, cs) = original.normalize();
        assert_eq!(base, F::BC7_UNORM_BLOCK);
        assert_eq!(cs, ColorSpace::Srgb);
        assert_eq!(base.denormalize(cs), original);
    }

    #[test]
    fn denormalize_linear_is_identity() {
        assert_eq!(
            F::R32G32B32A32_SFLOAT.denormalize(ColorSpace::Linear),
            F::R32G32B32A32_SFLOAT
        );
    }

    #[test]
    fn denormalize_no_srgb_variant_returns_self() {
        // BC6H has no sRGB variant
        assert_eq!(
            F::BC6H_UFLOAT_BLOCK.denormalize(ColorSpace::Srgb),
            F::BC6H_UFLOAT_BLOCK
        );
    }

    #[test]
    fn is_compressed_bc() {
        assert!(F::BC7_UNORM_BLOCK.is_compressed());
        assert!(F::BC1_RGBA_SRGB_BLOCK.is_compressed());
    }

    #[test]
    fn is_compressed_astc() {
        assert!(F::ASTC_4x4_UNORM_BLOCK.is_compressed());
        assert!(F::ASTC_12x12_SFLOAT_BLOCK.is_compressed());
    }

    #[test]
    fn is_compressed_uncompressed() {
        assert!(!F::R8G8B8A8_UNORM.is_compressed());
        assert!(!F::R32_SFLOAT.is_compressed());
    }

    #[test]
    fn bytes_per_pixel_common() {
        assert_eq!(F::R8_UNORM.bytes_per_pixel(), Some(1));
        assert_eq!(F::R8G8B8A8_UNORM.bytes_per_pixel(), Some(4));
        assert_eq!(F::R16G16B16A16_SFLOAT.bytes_per_pixel(), Some(8));
        assert_eq!(F::R32G32B32A32_SFLOAT.bytes_per_pixel(), Some(16));
    }

    #[test]
    fn bytes_per_pixel_compressed_returns_none() {
        assert_eq!(F::BC7_UNORM_BLOCK.bytes_per_pixel(), None);
    }

    #[test]
    fn block_size_bc() {
        assert_eq!(F::BC7_UNORM_BLOCK.block_size(), Some((4, 4)));
    }

    #[test]
    fn block_size_astc() {
        assert_eq!(F::ASTC_10x8_UNORM_BLOCK.block_size(), Some((10, 8)));
    }

    #[test]
    fn block_size_uncompressed_returns_none() {
        assert_eq!(F::R8G8B8A8_UNORM.block_size(), None);
    }

    #[test]
    fn bytes_per_block_values() {
        assert_eq!(F::BC1_RGBA_UNORM_BLOCK.bytes_per_block(), Some(8));
        assert_eq!(F::BC7_UNORM_BLOCK.bytes_per_block(), Some(16));
        assert_eq!(F::ASTC_4x4_UNORM_BLOCK.bytes_per_block(), Some(16));
    }

    #[test]
    fn channel_count_values() {
        assert_eq!(F::R8_UNORM.channel_count(), Some(1));
        assert_eq!(F::R8G8_UNORM.channel_count(), Some(2));
        assert_eq!(F::R8G8B8_UNORM.channel_count(), Some(3));
        assert_eq!(F::R8G8B8A8_UNORM.channel_count(), Some(4));
        assert_eq!(F::BC4_UNORM_BLOCK.channel_count(), Some(1));
        assert_eq!(F::BC7_UNORM_BLOCK.channel_count(), Some(4));
    }

    #[test]
    fn normalize_all_astc_srgb() {
        let srgb_astc = [
            F::ASTC_4x4_SRGB_BLOCK,
            F::ASTC_5x4_SRGB_BLOCK,
            F::ASTC_5x5_SRGB_BLOCK,
            F::ASTC_6x5_SRGB_BLOCK,
            F::ASTC_6x6_SRGB_BLOCK,
            F::ASTC_8x5_SRGB_BLOCK,
            F::ASTC_8x6_SRGB_BLOCK,
            F::ASTC_8x8_SRGB_BLOCK,
            F::ASTC_10x5_SRGB_BLOCK,
            F::ASTC_10x6_SRGB_BLOCK,
            F::ASTC_10x8_SRGB_BLOCK,
            F::ASTC_10x10_SRGB_BLOCK,
            F::ASTC_12x10_SRGB_BLOCK,
            F::ASTC_12x12_SRGB_BLOCK,
        ];
        for fmt in srgb_astc {
            let (base, cs) = fmt.normalize();
            assert_eq!(cs, ColorSpace::Srgb, "expected sRGB for {fmt:?}");
            assert_eq!(base.denormalize(cs), fmt, "roundtrip failed for {fmt:?}");
        }
    }
}
