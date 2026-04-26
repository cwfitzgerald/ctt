//! Encoder selection tests.

use crate::common::{TestFixture, read, run_cli};

/// `intel_bc7` and `bc7e_bc7` are different encoders for the same target
/// format; their output bytes must differ for the same RGBA8 input.
#[test]
fn intel_bc7_vs_bc7e_bc7_produce_different_bytes() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let intel_out = f.output_file("intel.ktx2");
    let bc7e_out = f.output_file("bc7e.ktx2");

    let common_args = ["--quality", "ultra-fast"];

    let mut intel_args: Vec<&str> = vec![
        "ctt",
        input.to_str().unwrap(),
        "-o",
        intel_out.to_str().unwrap(),
        "-f",
        "intel_bc7",
    ];
    intel_args.extend_from_slice(&common_args);
    run_cli(intel_args).expect("intel_bc7 run succeeded");

    let mut bc7e_args: Vec<&str> = vec![
        "ctt",
        input.to_str().unwrap(),
        "-o",
        bc7e_out.to_str().unwrap(),
        "-f",
        "bc7e_bc7",
    ];
    bc7e_args.extend_from_slice(&common_args);
    run_cli(bc7e_args).expect("bc7e_bc7 run succeeded");

    assert_ne!(
        read(&intel_out),
        read(&bc7e_out),
        "intel and bc7e BC7 encoders should produce different bytes"
    );
}
