//! Compression smoke tests.

use crate::common::{TestFixture, assert, read, run_cli};

/// RGBA8 KTX2 in → BC7 KTX2 out: result has a valid header, format=BC7,
/// dimensions match the input, and parses back through ctt.
#[test]
fn rgba8_to_bc7_ktx2_smoke() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-f",
        "bc7",
        "--quality",
        "ultra-fast",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.width, 16);
    assert_eq!(info.height, 16);
    assert_eq!(info.face_count, 1);
    assert_eq!(info.level_count, 1);
    assert_eq!(info.format, Some(ktx2::Format::BC7_UNORM_BLOCK));

    let decoded = assert::decode(&bytes);
    assert_eq!(decoded.surfaces.len(), 1);
    assert_eq!(decoded.surfaces[0].len(), 1);
    assert_eq!(decoded.surfaces[0][0].width, 16);
    assert_eq!(decoded.surfaces[0][0].height, 16);
    assert_eq!(decoded.surfaces[0][0].format, ktx2::Format::BC7_UNORM_BLOCK);
}
