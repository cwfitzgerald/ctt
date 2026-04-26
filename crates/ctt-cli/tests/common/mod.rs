//! Shared test helpers.
//!
//! Tests run `ctt_cli::run` directly (no subprocess) against synthesized
//! inputs in a tempdir owned by [`TestFixture`]. Helpers handle synthesis
//! (`synth`) and output validation (`assert`).
//!
//! ## Environment knobs
//!
//! - `CTT_TEST_KEEP_TMP=1` — disable tempdir cleanup; the path is printed
//!   to stderr when the fixture is dropped.
//! - `CTT_TEST_VERBOSE=1` — initialize the test logger at debug level so
//!   per-fixture path logs are emitted (visible with `--nocapture`).

#![allow(dead_code)]

pub mod assert;
pub mod synth;

use std::path::PathBuf;
use std::sync::OnceLock;

use clap::Parser;

pub use ctt_cli::Args;

/// Per-test scratch directory + path helpers.
///
/// On drop the tempdir is deleted unless `CTT_TEST_KEEP_TMP` is set in
/// the environment, in which case the path is left in place and printed
/// to stderr so a developer can inspect the artifacts.
pub struct TestFixture {
    tmp: tempfile::TempDir,
}

impl TestFixture {
    pub fn new() -> Self {
        init_logger();
        let mut tmp = tempfile::tempdir().expect("create tempdir");
        if std::env::var_os("CTT_TEST_KEEP_TMP").is_some() {
            tmp.disable_cleanup(true);
            eprintln!("CTT_TEST_KEEP_TMP set; preserving {}", tmp.path().display());
        }
        log::debug!("fixture tempdir: {}", tmp.path().display());
        Self { tmp }
    }

    /// Resolve a checked-in golden input under `tests/data/`.
    pub fn data_file(&self, name: &str) -> PathBuf {
        let p = data_dir().join(name);
        log::debug!("fixture data file: {}", p.display());
        p
    }

    /// Path inside the tempdir for a test-produced artifact. The file
    /// itself is created by whatever the test runs (e.g. `ctt_cli::run`).
    pub fn output_file(&self, name: &str) -> PathBuf {
        let p = self.tmp.path().join(name);
        log::debug!("fixture output file: {}", p.display());
        p
    }
}

impl Default for TestFixture {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the CLI with the given argv (including the leading "ctt").
pub fn run_cli<I, S>(argv: I) -> Result<(), Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString> + Clone,
{
    init_logger();
    let args = Args::try_parse_from(argv)?;
    ctt_cli::run(args)
}

/// Try to parse args without running. Useful for argument-parsing-only tests.
pub fn try_parse_args<I, S>(argv: I) -> Result<Args, clap::Error>
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString> + Clone,
{
    Args::try_parse_from(argv)
}

fn init_logger() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let verbose = if std::env::var_os("CTT_TEST_VERBOSE").is_some() {
            1
        } else {
            0
        };
        ctt_cli::setup_logger(verbose);
    });
}

/// Read a file produced into a tempdir.
pub fn read(path: impl AsRef<std::path::Path>) -> Vec<u8> {
    std::fs::read(path.as_ref()).expect("read file")
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
}
