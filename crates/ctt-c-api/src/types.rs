//! Plain enums and POD structs shared across the API.

/// A texel format. Values are Vulkan VkFormat values.
///
/// Common values are exported as `CTT_FORMAT_*` constants; arbitrary VkFormat
/// values are also accepted (useful when round-tripping container files that
/// carry a format ctt does not specifically know about).
pub type Format = u32;

/// Universal quality preset all encoders understand.
///
/// `Basic` is discriminant `0` so that a zero-initialized
/// [`ConvertSettings`](crate::ConvertSettings) selects it (the same value
/// `ctt_convert_settings_default` uses). The remaining presets keep their
/// speed ordering.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Basic = 0,
    UltraFast = 1,
    VeryFast = 2,
    Fast = 3,
    Slow = 4,
    VerySlow = 5,
}

impl Quality {
    pub(crate) fn into_inner(self) -> ctt::Quality {
        match self {
            Self::UltraFast => ctt::Quality::UltraFast,
            Self::VeryFast => ctt::Quality::VeryFast,
            Self::Fast => ctt::Quality::Fast,
            Self::Basic => ctt::Quality::Basic,
            Self::Slow => ctt::Quality::Slow,
            Self::VerySlow => ctt::Quality::VerySlow,
        }
    }
}

/// Color space metadata for a surface.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb = 0,
    Linear = 1,
}

impl From<ColorSpace> for ctt::ColorSpace {
    fn from(c: ColorSpace) -> Self {
        match c {
            ColorSpace::Srgb => ctt::ColorSpace::Srgb,
            ColorSpace::Linear => ctt::ColorSpace::Linear,
        }
    }
}

impl From<ctt::ColorSpace> for ColorSpace {
    fn from(c: ctt::ColorSpace) -> Self {
        match c {
            ctt::ColorSpace::Srgb => ColorSpace::Srgb,
            ctt::ColorSpace::Linear => ColorSpace::Linear,
        }
    }
}

/// Alpha interpretation for a surface.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Straight = 0,
    Premultiplied = 1,
    Opaque = 2,
}

impl From<AlphaMode> for ctt::AlphaMode {
    fn from(a: AlphaMode) -> Self {
        match a {
            AlphaMode::Straight => ctt::AlphaMode::Straight,
            AlphaMode::Premultiplied => ctt::AlphaMode::Premultiplied,
            AlphaMode::Opaque => ctt::AlphaMode::Opaque,
        }
    }
}

impl From<ctt::AlphaMode> for AlphaMode {
    fn from(a: ctt::AlphaMode) -> Self {
        match a {
            ctt::AlphaMode::Straight => AlphaMode::Straight,
            ctt::AlphaMode::Premultiplied => AlphaMode::Premultiplied,
            ctt::AlphaMode::Opaque => AlphaMode::Opaque,
        }
    }
}

/// How to read the bytes of every surface in an image.
///
/// `format` must agree with `color_space`:
///
/// - `CTT_COLOR_SPACE_SRGB`: if the format has an sRGB variant, the format
///   must be that variant. `CTT_FORMAT_R8G8B8A8_SRGB` is valid and
///   `CTT_FORMAT_R8G8B8A8_UNORM` is not. `CTT_FORMAT_R16G16B16A16_SFLOAT` has
///   no sRGB variant, so it is valid.
/// - `CTT_COLOR_SPACE_LINEAR`: the format must not be an sRGB variant.
///   `CTT_FORMAT_R8G8B8A8_UNORM` and `CTT_FORMAT_R16G16B16A16_SFLOAT` are
///   valid and `CTT_FORMAT_R8G8B8A8_SRGB` is not.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatDesc {
    /// VkFormat of every surface. Must be non-zero.
    pub format: Format,
    /// Color space the pixel values live in.
    pub color_space: ColorSpace,
    /// How the alpha channel relates to the color channels.
    pub alpha: AlphaMode,
}

impl FormatDesc {
    /// Convert to the Rust type. Fails if `format` is zero or does not agree
    /// with `color_space`.
    pub(crate) fn into_inner(self) -> Result<ctt::FormatDesc, String> {
        let format =
            ctt::Format::new(self.format).ok_or("format must be a non-zero VkFormat value")?;
        let desc = ctt::FormatDesc {
            format,
            color_space: self.color_space.into(),
            alpha: self.alpha.into(),
        };
        desc.validate().map_err(|e| e.to_string())?;
        Ok(desc)
    }
}

impl From<ctt::FormatDesc> for FormatDesc {
    fn from(d: ctt::FormatDesc) -> Self {
        FormatDesc {
            format: d.format.value(),
            color_space: d.color_space.into(),
            alpha: d.alpha.into(),
        }
    }
}

/// Texture topology.
///
/// Array-ness is implicit in the layer count: `Texture2D` uses one layer per
/// surface entry, `Cubemap` requires a multiple of 6 layers, `Texture3D`
/// requires exactly one layer (depth is carried on the surface).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureKind {
    Texture2d = 0,
    Cubemap = 1,
    Texture3d = 2,
}

impl From<TextureKind> for ctt::TextureKind {
    fn from(k: TextureKind) -> Self {
        match k {
            TextureKind::Texture2d => ctt::TextureKind::Texture2D,
            TextureKind::Cubemap => ctt::TextureKind::Cubemap,
            TextureKind::Texture3d => ctt::TextureKind::Texture3D,
        }
    }
}

impl From<ctt::TextureKind> for TextureKind {
    fn from(k: ctt::TextureKind) -> Self {
        match k {
            ctt::TextureKind::Texture2D => TextureKind::Texture2d,
            ctt::TextureKind::Cubemap => TextureKind::Cubemap,
            ctt::TextureKind::Texture3D => TextureKind::Texture3d,
        }
    }
}

/// One channel of a [`Swizzle`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwizzleChannel {
    R = 0,
    G = 1,
    B = 2,
    A = 3,
    Zero = 4,
    One = 5,
}

impl From<SwizzleChannel> for ctt::SwizzleChannel {
    fn from(c: SwizzleChannel) -> Self {
        match c {
            SwizzleChannel::R => ctt::SwizzleChannel::R,
            SwizzleChannel::G => ctt::SwizzleChannel::G,
            SwizzleChannel::B => ctt::SwizzleChannel::B,
            SwizzleChannel::A => ctt::SwizzleChannel::A,
            SwizzleChannel::Zero => ctt::SwizzleChannel::Zero,
            SwizzleChannel::One => ctt::SwizzleChannel::One,
        }
    }
}

/// A 4-component swizzle pattern, ordered as the destination channels R,G,B,A.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Swizzle {
    pub channels: [SwizzleChannel; 4],
}

impl From<Swizzle> for ctt::Swizzle {
    fn from(s: Swizzle) -> Self {
        ctt::Swizzle([
            s.channels[0].into(),
            s.channels[1].into(),
            s.channels[2].into(),
            s.channels[3].into(),
        ])
    }
}

/// Filter used when downsampling for mipmap generation.
///
/// `Triangle` is discriminant `0` so that a zero-initialized
/// [`ConvertSettings`](crate::ConvertSettings) selects the same default filter
/// as `ctt_convert_settings_default`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipmapFilter {
    Triangle = 0,
    Nearest = 1,
    CatmullRom = 2,
    Gaussian = 3,
    Lanczos3 = 4,
}

impl From<MipmapFilter> for ctt::MipmapFilter {
    fn from(f: MipmapFilter) -> Self {
        match f {
            MipmapFilter::Nearest => ctt::MipmapFilter::Nearest,
            MipmapFilter::Triangle => ctt::MipmapFilter::Triangle,
            MipmapFilter::CatmullRom => ctt::MipmapFilter::CatmullRom,
            MipmapFilter::Gaussian => ctt::MipmapFilter::Gaussian,
            MipmapFilter::Lanczos3 => ctt::MipmapFilter::Lanczos3,
        }
    }
}

/// Optional [`ColorSpace`] (matches Rust's `Option<ColorSpace>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalColorSpace {
    pub present: bool,
    pub value: ColorSpace,
}

/// Optional [`AlphaMode`] (matches Rust's `Option<AlphaMode>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalAlphaMode {
    pub present: bool,
    pub value: AlphaMode,
}

/// Optional [`Swizzle`] (matches Rust's `Option<Swizzle>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalSwizzle {
    pub present: bool,
    pub value: Swizzle,
}

/// Optional `size_t` value (matches Rust's `Option<usize>`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionalSize {
    pub present: bool,
    pub value: usize,
}
