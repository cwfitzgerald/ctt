use std::path::PathBuf;

use clap::Parser;

/// ctt — texture compression tool
#[derive(Debug, Parser)]
#[command(name = "ctt", version, about, max_term_width = 100)]
pub struct Args {
    /// Input image file(s).
    ///
    /// A single input is converted as-is. N plain inputs assemble a 2D array
    /// texture (argv order = layer order). With --cubemap, provide 6 face
    /// files for one cubemap, or N×6 files to assemble a cubemap array. A
    /// single input plus --cubemap is split according to --cubemap-layout.
    #[arg(required_unless_present_any = ["list_encoders", "help_encoder"])]
    pub input: Vec<PathBuf>,

    /// Output file path.
    #[arg(short, long, required_unless_present_any = ["list_encoders", "help_encoder"])]
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
    #[arg(long, conflicts_with = "volume")]
    pub cubemap: bool,

    /// Cubemap layout when splitting a single input image (default: cross).
    /// Applies only to a single input; rejected with multiple inputs.
    #[arg(long, requires = "cubemap")]
    pub cubemap_layout: Option<CubemapLayoutArg>,

    /// Treat each input as a Z slice of a 3D (volume) texture, stacked in
    /// argv order. Mip generation is unsupported for 3D textures.
    #[arg(long, conflicts_with = "cubemap", conflicts_with = "mipmap")]
    pub volume: bool,

    /// Remap RGBA channels. 4 characters from: rgba01.
    ///
    /// "bgra" = swap red/blue, "0r0g" = 2 channel normal map to BC3 packing
    /// "rgb1" = force opaque, "r000" = ignore non-red channel.
    #[arg(long)]
    pub swizzle: Option<String>,

    /// Override the input color space.
    ///
    /// Container formats (KTX2, DDS) carry color-space metadata, which is
    /// honored by default. For formats without metadata (PNG, JPEG, …) the
    /// fallback is sRGB. Pass this flag to override either.
    #[arg(long, visible_alias = "ic")]
    pub input_color_space: Option<ColorSpaceArg>,

    /// Override the input alpha mode.
    ///
    /// Container formats (KTX2, DDS) carry alpha metadata, which is honored
    /// by default. For formats without metadata the fallback is straight.
    /// Pass this flag to override either.
    #[arg(long, visible_alias = "ia")]
    pub input_alpha: Option<AlphaModeArg>,

    /// Desired color space of the output. If omitted, matches the input.
    #[arg(long, visible_alias = "oc")]
    pub output_color_space: Option<ColorSpaceArg>,

    /// Desired alpha mode of the output. If omitted, matches the input.
    #[arg(long, visible_alias = "oa")]
    pub output_alpha: Option<AlphaModeArg>,

    /// Compression quality preset.
    #[arg(long, default_value = "basic")]
    pub quality: QualityArg,

    /// List available encoder backends and their supported formats.
    #[arg(long)]
    pub list_encoders: bool,

    /// Show the available `--<encoder>-opts` keys for an encoder, with
    /// types and doc strings, then exit.
    #[arg(long, value_name = "NAME")]
    pub help_encoder: Option<String>,

    /// astcenc-specific options. Format: `key=val[;key=val...]`.
    /// Run `--help-encoder astcenc` to see the available keys.
    #[arg(long, value_name = "OPTS")]
    pub astcenc_opts: Option<String>,

    /// bc7enc-rdo-specific options. Format: `key=val[;key=val...]`.
    /// Run `--help-encoder bc7e` to see the available keys.
    #[arg(long, value_name = "OPTS")]
    pub bc7e_opts: Option<String>,

    /// intel/ispc-encoder-specific options. Format: `key=val[;key=val...]`.
    /// Run `--help-encoder intel` to see the available keys.
    #[arg(long, value_name = "OPTS")]
    pub intel_opts: Option<String>,

    /// etcpak-encoder-specific options. Format: `key=val[;key=val...]`.
    /// Run `--help-encoder etcpak` to see the available keys.
    #[arg(long, value_name = "OPTS")]
    pub etcpak_opts: Option<String>,

    /// AMD-compressonator-specific options. Format: `key=val[;key=val...]`.
    /// Run `--help-encoder amd` to see the available keys.
    #[arg(long, value_name = "OPTS")]
    pub amd_opts: Option<String>,

    /// Generate mipmaps.
    #[arg(long)]
    pub mipmap: bool,

    /// Number of mip levels (including the base). Requires --mipmap.
    /// If omitted, generates the full chain down to 1×1.
    #[arg(long, requires = "mipmap")]
    pub mipmap_count: Option<usize>,

    /// Filter used for mipmap downsampling. Requires --mipmap.
    #[arg(long, default_value = "triangle", requires = "mipmap")]
    pub mipmap_filter: MipmapFilterArg,

    /// Enable zstd supercompression for KTX2 output.
    ///
    /// Optionally takes a compression level attached with `=`: negative (fast
    /// mode) through 22 (maximum compression). Use `--zstd` for the default or
    /// `--zstd=<LEVEL>` (e.g. `--zstd=19`); `--zstd <LEVEL>` is not accepted.
    /// Default when omitted: 0, which maps to the zstd library default
    /// (currently level 3).
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "0",
        value_name = "LEVEL",
        value_parser = zstd_level_parser(),
        conflicts_with = "zlib",
    )]
    pub zstd: Option<i32>,

    /// Enable zlib supercompression for KTX2 output.
    ///
    /// Optionally takes a compression level attached with `=`: 1 (fastest)
    /// through 10 (maximum compression). Use `--zlib` for the default or
    /// `--zlib=<LEVEL>` (e.g. `--zlib=9`); `--zlib <LEVEL>` is not accepted.
    /// Default when omitted: 6.
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "6",
        value_name = "LEVEL",
        value_parser = clap::value_parser!(u8).range(1..=10),
        conflicts_with = "zstd",
    )]
    pub zlib: Option<u8>,

    /// Increase logging verbosity (-v = debug, -vv = trace).
    #[arg(short, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

/// Value parser for `--zstd` levels, bounded by the range the linked zstd
/// library actually accepts (`ZSTD_minCLevel()..=ZSTD_maxCLevel()`) — a large
/// negative fast-mode floor up to 22.
fn zstd_level_parser() -> clap::builder::RangedI64ValueParser<i32> {
    let range = zstd::compression_level_range();
    clap::value_parser!(i32).range((*range.start() as i64)..=(*range.end() as i64))
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
