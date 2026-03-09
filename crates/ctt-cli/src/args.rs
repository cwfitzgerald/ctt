use std::path::PathBuf;

use clap::Parser;

/// ctt — texture compression tool
#[derive(Debug, Parser)]
#[command(name = "ctt", version, about)]
pub struct Args {
    /// Input image file(s). For cubemaps with separate faces, provide 6 files.
    #[arg(required = true)]
    pub input: Vec<PathBuf>,

    /// Output file path.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Compression format (bc1, bc3, bc4, bc5, bc6h, bc7, etc1, astc_4x4, astc_6x6, ...).
    #[arg(short, long)]
    pub format: String,

    /// Output container format.
    #[arg(short, long, default_value = "ktx2")]
    pub container: ContainerArg,

    /// Treat input as a cubemap.
    #[arg(long)]
    pub cubemap: bool,

    /// Cubemap layout when using a single input image.
    #[arg(long, default_value = "cross")]
    pub cubemap_layout: CubemapLayoutArg,

    /// Channel swizzle (e.g. "rgba", "bgra", "rrrg", "rgb1").
    #[arg(long)]
    pub swizzle: Option<String>,

    /// Color space of the input.
    #[arg(long, default_value = "srgb")]
    pub color_space: ColorSpaceArg,
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
