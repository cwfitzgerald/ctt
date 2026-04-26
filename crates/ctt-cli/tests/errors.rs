//! Error-case tests.

use crate::common::{TestFixture, run_cli, try_parse_args};

/// DDS does not support ETC formats — the CLI must reject this combination.
#[test]
fn dds_etc_rejected() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.dds");

    let result = run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        "etc2_rgba",
    ]);
    assert!(result.is_err(), "DDS + ETC should be rejected");
}

/// `--cubemap` requires either 1 or 6 inputs. Five inputs must error.
#[test]
fn cubemap_with_five_inputs_rejected() {
    let f = TestFixture::new();
    let inputs = [
        f.data_file("cube_face_pos_x.ktx2"),
        f.data_file("cube_face_neg_x.ktx2"),
        f.data_file("cube_face_pos_y.ktx2"),
        f.data_file("cube_face_neg_y.ktx2"),
        f.data_file("cube_face_pos_z.ktx2"),
    ];
    let output = f.output_file("cube.ktx2");

    let mut argv: Vec<String> = vec!["ctt".to_string()];
    for p in &inputs {
        argv.push(p.to_str().unwrap().to_string());
    }
    argv.extend([
        "-o".to_string(),
        output.to_str().unwrap().to_string(),
        "--cubemap".to_string(),
    ]);

    let result = run_cli(argv);
    assert!(result.is_err(), "5 inputs + --cubemap should be rejected");
}

/// `--swizzle` must be exactly 4 characters.
#[test]
fn swizzle_three_chars_rejected() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    let argv = [
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--swizzle",
        "bgr",
    ];

    if try_parse_args(argv).is_err() {
        return;
    }

    let result = run_cli(argv);
    assert!(result.is_err(), "3-character swizzle must be rejected");
}
