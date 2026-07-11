//! `--<encoder>-opts` merge tests.
//!
//! These lock in the fix for per-encoder opts being silently dropped when a
//! bare format name (`-f bc7`) resolves to `Encoder::Auto`: the opts must now
//! reach the concrete backend Auto selects, while opts aimed at a *different*
//! backend than the one running are still warned about and dropped.

use crate::common::{TestFixture, read, run_cli};

/// Encode the linear RGBA8 golden to `output` with the given extra args.
fn encode(out: &std::path::Path, extra: &[&str]) {
    let input = TestFixture::new().data_file("rgba8_16x16_linear.ktx2");
    let mut argv: Vec<&str> = vec![
        "ctt",
        input.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--quality",
        "ultra-fast",
    ];
    argv.extend_from_slice(extra);
    run_cli(argv).expect("run succeeded");
}

/// `-f bc7` resolves to `Encoder::Auto` → the highest-priority BC7 backend
/// (bc7enc). `--bc7e-opts` must now apply to it, changing the output bytes.
#[test]
fn bare_format_auto_applies_bc7e_opts() {
    let f = TestFixture::new();
    let default_out = f.output_file("default.ktx2");
    let opts_out = f.output_file("opts.ktx2");

    encode(&default_out, &["-f", "bc7"]);
    encode(&opts_out, &["-f", "bc7", "--bc7e-opts", "perceptual=false"]);

    assert_ne!(
        read(&default_out),
        read(&opts_out),
        "--bc7e-opts must alter the Auto-selected bc7enc output"
    );
}

/// An explicit `bc7e_bc7` prefix also honors `--bc7e-opts`.
#[test]
fn explicit_prefix_applies_bc7e_opts() {
    let f = TestFixture::new();
    let default_out = f.output_file("default.ktx2");
    let opts_out = f.output_file("opts.ktx2");

    encode(&default_out, &["-f", "bc7e_bc7"]);
    encode(
        &opts_out,
        &["-f", "bc7e_bc7", "--bc7e-opts", "perceptual=false"],
    );

    assert_ne!(
        read(&default_out),
        read(&opts_out),
        "--bc7e-opts must alter explicit bc7e_bc7 output"
    );
}

/// A genuine mismatch — `--bc7e-opts` while `-f intel_bc7` selects the intel
/// encoder — is warned about and dropped, so the output is identical to the
/// plain `intel_bc7` run.
#[test]
fn mismatched_opts_are_warned_and_dropped() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let plain_out = f.output_file("plain.ktx2");
    let opts_out = f.output_file("opts.ktx2");

    encode(&plain_out, &["-f", "intel_bc7"]);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ctt"))
        .args([
            input.as_os_str(),
            "-o".as_ref(),
            opts_out.as_os_str(),
            "--quality".as_ref(),
            "ultra-fast".as_ref(),
            "-f".as_ref(),
            "intel_bc7".as_ref(),
            "--bc7e-opts".as_ref(),
            "perceptual=false".as_ref(),
        ])
        .output()
        .expect("run ctt subprocess");
    assert!(output.status.success(), "subprocess failed: {output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("warning: --bc7e-opts ignored"),
        "expected mismatch warning, got: {stderr}"
    );

    assert_eq!(
        read(&plain_out),
        read(&opts_out),
        "--bc7e-opts targeting a different backend must be dropped, not applied"
    );
}
