use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod build_ispc;
mod generate_bindings;
mod util;
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
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Cmd::VerifyBinaries => verify_binaries::verify_binaries(),
        Cmd::BuildIspc(args) => build_ispc::build_ispc(args),
        Cmd::GenerateBindings => generate_bindings::generate_bindings(),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:?}");
            ExitCode::FAILURE
        }
    }
}
