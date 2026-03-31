use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::util::workspace_root;

#[derive(Parser)]
pub struct BuildIspcArgs {
    /// Which crate to build ISPC libraries for.
    #[arg(long = "crate", value_enum)]
    krate: CrateChoice,
    /// Rust target triple (e.g. x86_64-unknown-linux-gnu).
    #[arg(long)]
    target: String,
    /// Directory to write output libraries to.
    #[arg(long)]
    output_dir: PathBuf,
}

#[derive(Clone, clap::ValueEnum)]
enum CrateChoice {
    IntelTextureCompressor,
    Bc7encRdo,
    Both,
}

pub fn build_ispc(args: BuildIspcArgs) -> Result<()> {
    let workspace_root = workspace_root();
    std::fs::create_dir_all(&args.output_dir)?;

    let target =
        ispc_build_utils::CompileTarget::from_triple(&args.target, args.output_dir.clone());

    let build_intel = matches!(
        args.krate,
        CrateChoice::IntelTextureCompressor | CrateChoice::Both
    );
    let build_bc7enc = matches!(args.krate, CrateChoice::Bc7encRdo | CrateChoice::Both);

    if build_intel {
        println!("Building intel-texture-compressor ISPC libraries...");
        let ispc_dir = workspace_root.join("crates/ctt-intel-texture-compressor/ispc");

        let mut kernel = ispc_build_utils::Config::new();
        kernel
            .file(ispc_dir.join("kernel.ispc"))
            .opt_level(2)
            .woff();
        kernel.compile_to("kernel", &target);

        let mut kernel_astc = ispc_build_utils::Config::new();
        kernel_astc
            .file(ispc_dir.join("kernel_astc.ispc"))
            .opt_level(2)
            .woff();
        kernel_astc.compile_to("kernel_astc", &target);

        println!("  kernel and kernel_astc -> {}", args.output_dir.display());
    }

    if build_bc7enc {
        println!("Building bc7enc-rdo ISPC library...");
        let ispc_dir = workspace_root.join("crates/ctt-bc7enc-rdo/ispc");

        let mut bc7e = ispc_build_utils::Config::new();
        bc7e.file(ispc_dir.join("bc7e.ispc"))
            .opt_level(2)
            .woff()
            .fast_math()
            .disable_assertions();
        bc7e.compile_to("bc7e", &target);

        println!("  bc7e -> {}", args.output_dir.display());
    }

    Ok(())
}
