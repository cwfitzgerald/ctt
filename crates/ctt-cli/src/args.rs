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

    /// Target format. Bare (bc1, bc7) or prefixed with encoder (intel_bc7, bc7e_bc7).
    /// ASTC formats use astc_WxH (e.g. astc_4x4, astc_8x8, astc_12x12).
    /// Uncompressed formats use WebGPU (rgba8unorm) or Vulkan (r8g8b8a8_unorm) names.
    /// If omitted, the input format is preserved without compression.
    #[arg(short, long)]
    pub format: Option<String>,

    /// Output container format. Inferred from the output file extension when omitted.
    #[arg(short, long)]
    pub container: Option<ContainerArg>,

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

    /// Color space of the input image(s).
    #[arg(long, visible_alias = "ic", default_value = "srgb")]
    pub input_color_space: ColorSpaceArg,

    /// Alpha mode of the input image(s).
    #[arg(long, visible_alias = "ia", default_value = "straight")]
    pub input_alpha: AlphaModeArg,

    /// Desired color space of the output. If omitted, matches the input.
    #[arg(long, visible_alias = "oc")]
    pub output_color_space: Option<ColorSpaceArg>,

    /// Desired alpha mode of the output. If omitted, matches the input.
    #[arg(long, visible_alias = "oa")]
    pub output_alpha: Option<AlphaModeArg>,

    /// Compression quality preset.
    #[arg(long, default_value = "basic")]
    pub quality: QualityArg,

    /// Encode alpha channel (for BC7).
    #[arg(long)]
    pub alpha: bool,

    /// Allow lossy auto-inserted format conversions in the pipeline.
    ///
    /// By default, the resolver will error if an intermediate conversion loses
    /// precision (e.g. f32 → u16 in a f32 → f32 pipeline). This flag suppresses
    /// those errors.
    #[arg(long)]
    pub allow_lossy_intermediates: bool,

    /// List available encoder backends and their supported formats.
    #[arg(long)]
    pub list_encoders: bool,

    /// Generate mipmaps.
    #[arg(long)]
    pub mipmap: bool,

    /// Number of mip levels (including the base). Requires --mipmap.
    /// If omitted, generates the full chain down to 1×1.
    #[arg(long, requires = "mipmap")]
    pub mipmap_count: Option<usize>,

    /// Filter used for mipmap downsampling. Requires --mipmap.
    #[arg(long, default_value = "catmull-rom", requires = "mipmap")]
    pub mipmap_filter: MipmapFilterArg,

    /// Enable zstd supercompression for KTX2 output.
    /// Optionally takes a compression level: negative (fast mode) through 22
    /// (maximum compression). Default when omitted: 0, which maps to the zstd
    /// library default (currently level 3).
    #[arg(long, num_args = 0..=1, default_missing_value = "0", value_name = "LEVEL", conflicts_with = "zlib")]
    pub zstd: Option<i32>,

    /// Enable zlib supercompression for KTX2 output.
    /// Optionally takes a compression level: 1 (fastest) through 10 (maximum
    /// compression). Default when omitted: 6.
    #[arg(long, num_args = 0..=1, default_missing_value = "6", value_name = "LEVEL", conflicts_with = "zstd")]
    pub zlib: Option<u8>,

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
pub enum AlphaModeArg {
    Straight,
    Premultiplied,
    Opaque,
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

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum MipmapFilterArg {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}
