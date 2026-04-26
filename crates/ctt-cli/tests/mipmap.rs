//! Mipmap generation tests.

use crate::common::{TestFixture, assert, read, run_cli};

/// 16×16 RGBA8 + `--mipmap` → KTX2 with a 5-level chain (16, 8, 4, 2, 1)
/// and the per-level dimensions match.
#[test]
fn full_mip_chain_16x16() {
    let f = TestFixture::new();
    let input = f.data_file("rgba8_16x16_linear.ktx2");
    let output = f.output_file("out.ktx2");

    run_cli([
        "ctt",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--mipmap",
    ])
    .expect("run succeeded");

    let bytes = read(&output);
    let info = assert::parse_ktx2(&bytes);
    assert_eq!(info.width, 16);
    assert_eq!(info.height, 16);
    assert_eq!(info.level_count, 5, "16×16 produces a 5-level chain");

    let decoded = assert::decode(&bytes);
    let expected_dims = [(16u32, 16u32), (8, 8), (4, 4), (2, 2), (1, 1)];
    for (i, (w, h)) in expected_dims.iter().enumerate() {
        let s = &decoded.surfaces[0][i];
        assert_eq!(s.width, *w, "mip {i} width");
        assert_eq!(s.height, *h, "mip {i} height");
    }
}
