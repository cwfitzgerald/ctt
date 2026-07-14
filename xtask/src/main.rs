use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod build_ispc;
mod generate_bindings;
mod generate_c_header;
mod util;
mod vendor;
mod verify_binaries;

#[derive(Parser)]
#[command(name = "xtask", about = "Build automation for ctt")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Verify attestations of prebuilt binaries.
    VerifyBinaries,
    /// Build ISPC libraries for a target platform.
    BuildIspc(build_ispc::BuildIspcArgs),
    /// Regenerate FFI binding files from C/ISPC headers.
    ///
    /// Requires ISPC to be installed (for bc7e bindings).
    GenerateBindings,
    /// Regenerate the C header (`crates/ctt-c-api/include/ctt.h`) from the
    /// `ctt-c-api` Rust crate using cbindgen.
    GenerateCHeader,
    /// Vendor third-party source code into crate directories.
    ///
    /// With no target, vendors all targets.
    Vendor(vendor::VendorArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Cmd::VerifyBinaries => verify_binaries::verify_binaries(),
        Cmd::BuildIspc(args) => build_ispc::build_ispc(args),
        Cmd::GenerateBindings => generate_bindings::generate_bindings(),
        Cmd::GenerateCHeader => generate_c_header::generate_c_header(),
        Cmd::Vendor(args) => vendor::vendor(args),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:?}");
            ExitCode::FAILURE
        }
    }
}
