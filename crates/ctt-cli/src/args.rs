use std::path::PathBuf;

use clap::Parser;

/// ctt — texture compression tool
#[derive(Debug, Parser)]
#[command(name = "ctt", version, about)]
pub struct Args {
    /// Input image file(s). For cubemaps with separate faces, provide 6 files.
    #[arg(required_unless_present = "list_encoders")]
    pub input: Vec<PathBuf>,

    /// Output file path.
    #[arg(short, long, required_unless_present = "list_encoders")]
    pub output: Option<PathBuf>,

    /// Compression format. Bare (bc1, bc7) or prefixed with encoder (intel_bc7, bc7e_bc7).
    /// ASTC formats use astc_WxH (e.g. astc_4x4, astc_8x8, astc_12x12).
    #[arg(short, long, required_unless_present = "list_encoders")]
    pub format: Option<String>,

    /// Output container format.
    #[arg(short, long, default_value = "ktx2")]
    pub container: ContainerArg,

    /// Treat input as a cubemap.
    #[arg(long)]
    pub cubemap: bool,

    /// Cubemap layout when using a single input image.
    #[arg(long, default_value = "cross")]
    pub cubemap_layout: CubemapLayoutArg,

    /// Remap RGBA channels. 4 characters from: rgba01.
    ///
    /// "bgra" = swap red/blue, "0r0g" = 2 channel normal map to BC3 packing
    /// "rgb1" = force opaque, "r000" = ignore non-red channel.
    #[arg(long)]
    pub swizzle: Option<String>,

    /// Color space of the input. Used for selecting output color space and performing mipmap generation.
    #[arg(long, default_value = "srgb")]
    pub color_space: ColorSpaceArg,

    /// Compression quality preset.
    #[arg(long, default_value = "basic")]
    pub quality: QualityArg,

    /// Encode alpha channel (for BC7).
    #[arg(long)]
    pub alpha: bool,

    /// List available encoder backends and their supported formats.
    #[arg(long)]
    pub list_encoders: bool,

    /// Increase logging verbosity (-v = debug, -vv = trace).
    #[arg(short, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ContainerArg {
    Dds,
    Ktx2,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum CubemapLayoutArg {
    Cross,
    Strip,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ColorSpaceArg {
    Srgb,
    Linear,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum QualityArg {
    UltraFast,
    VeryFast,
    Fast,
    Basic,
    Slow,
    VerySlow,
}
