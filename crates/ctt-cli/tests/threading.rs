use crate::common::{TestFixture, read, run_cli, synth, try_parse_args};

#[test]
fn thread_count_defaults_to_zero() {
    let args = try_parse_args(["ctt", "in.png", "-o", "out.ktx2"]).unwrap();
    assert_eq!(args.threads, 0);
}

#[test]
fn thread_count_accepts_zero_one_and_larger_counts() {
    for (flag, value, expected) in [("-t", "0", 0), ("-t", "1", 1), ("--threads", "4", 4)] {
        let args = try_parse_args(["ctt", "in.png", "-o", "out.ktx2", flag, value]).unwrap();
        assert_eq!(args.threads, expected);
    }
}

#[test]
fn thread_counts_produce_identical_output() {
    let fixture = TestFixture::new();
    let input = fixture.output_file("input.ktx2");
    let image = synth::make_image(
        synth::rgba8_gradient(64, 64),
        64,
        64,
        ctt::Format::R8G8B8A8_UNORM,
        ctt::ColorSpace::Linear,
        ctt::AlphaMode::Opaque,
    );
    synth::write_ktx2(image, &input);

    let mut outputs = Vec::new();
    for (name, count) in [
        ("default", None),
        ("zero", Some(0)),
        ("one", Some(1)),
        ("four", Some(4)),
    ] {
        let output_name = format!("{name}.ktx2");
        let output = fixture.output_file(&output_name);
        let mut argv = vec![
            "ctt".to_string(),
            input.to_string_lossy().into_owned(),
            "-o".to_string(),
            output.to_string_lossy().into_owned(),
            "-f".to_string(),
            "bc7".to_string(),
            "--quality".to_string(),
            "ultra-fast".to_string(),
        ];
        if let Some(count) = count {
            argv.extend(["--threads".to_string(), count.to_string()]);
        }
        run_cli(argv).unwrap();
        outputs.push(read(output));
    }

    for output in &outputs[1..] {
        assert_eq!(output, &outputs[0]);
    }
}
