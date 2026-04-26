use std::process::ExitCode;

use clap::Parser;

use ctt_cli::{Args, run, setup_logger};

fn main() -> ExitCode {
    let args = Args::parse();
    setup_logger(args.verbose);
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("{e}");
            ExitCode::FAILURE
        }
    }
}
