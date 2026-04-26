//! Container conversion tests (KTX2 ↔ DDS).

use crate::common::{TestFixture, assert, read, run_cli};

/// BC7 KTX2 in → DDS out: payload bytes survive through the container
/// conversion and the DDS file has a valid header.
#[test]
fn bc7_ktx2_to_dds_payload_preserved() {
    let f = TestFixture::new();
    let input = f.data_file("bc7_4x4.ktx2");
    let output = f.output_file("out.dds");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ])
    .expect("run succeeded");

    let out_bytes = read(&output);
    assert!(
        out_bytes.starts_with(assert::DDS_MAGIC),
        "output must start with DDS magic"
    );

    let dds_info = assert::parse_dds(&out_bytes);
    assert_eq!(dds_info.width, 4);
    assert_eq!(dds_info.height, 4);

    assert::assert_payload_eq(&read(&input), &out_bytes);
}
