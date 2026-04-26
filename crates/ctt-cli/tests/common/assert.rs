//! Output validation helpers.
//!
//! Read KTX2/DDS bytes back through `ktx2`/`ddsfile` to inspect headers,
//! and through `ctt::input::decode_container` for full-image roundtrips.

use ctt::input::{InputOverrides, decode_container};
use ctt::{Image, Surface};

/// KTX2 magic bytes (first 12 bytes of every KTX2 file).
pub const KTX2_MAGIC: &[u8; 12] = &ktx2::MAGIC;

/// DDS magic ("DDS ").
pub const DDS_MAGIC: &[u8; 4] = b"DDS ";

/// Subset of KTX2 header fields tests typically care about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ktx2Info {
    pub format: Option<ktx2::Format>,
    pub width: u32,
    pub height: u32,
    pub layer_count: u32,
    pub face_count: u32,
    pub level_count: u32,
    pub supercompression: Option<ktx2::SupercompressionScheme>,
}

/// Parse a KTX2 file's header.
pub fn parse_ktx2(bytes: &[u8]) -> Ktx2Info {
    let reader = ktx2::Reader::new(bytes).expect("valid KTX2");
    let h = reader.header();
    Ktx2Info {
        format: h.format,
        width: h.pixel_width,
        height: h.pixel_height,
        layer_count: h.layer_count,
        face_count: h.face_count,
        level_count: h.level_count,
        supercompression: h.supercompression_scheme,
    }
}

/// Subset of DDS header fields.
#[derive(Debug, Clone)]
pub struct DdsInfo {
    pub width: u32,
    pub height: u32,
    pub array_layers: u32,
    pub mipmap_levels: u32,
}

/// Parse a DDS file's header.
pub fn parse_dds(bytes: &[u8]) -> DdsInfo {
    let dds = ddsfile::Dds::read(bytes).expect("valid DDS");
    DdsInfo {
        width: dds.get_width(),
        height: dds.get_height(),
        array_layers: dds.get_num_array_layers(),
        mipmap_levels: dds.get_num_mipmap_levels(),
    }
}

/// Decode a KTX2 or DDS file into the canonical [`Image`] representation.
pub fn decode(bytes: &[u8]) -> Image {
    decode_container(bytes, InputOverrides::default())
        .expect("decode succeeded")
        .expect("recognized container")
}

/// Compare a single surface's payload bytes-for-bytes.
#[track_caller]
pub fn assert_surface_data_eq(a: &Surface, b: &Surface, label: &str) {
    assert_eq!(a.width, b.width, "{label}: width mismatch");
    assert_eq!(a.height, b.height, "{label}: height mismatch");
    assert_eq!(a.format, b.format, "{label}: format mismatch");
    assert_eq!(a.data, b.data, "{label}: pixel data mismatch");
}

/// Decode both files and assert the per-surface payload matches across
/// every layer/mip. Header-level differences (e.g. array vs face encoding)
/// are intentionally ignored so this works for KTX2↔DDS conversions.
#[track_caller]
pub fn assert_payload_eq(a: &[u8], b: &[u8]) {
    let img_a = decode(a);
    let img_b = decode(b);
    assert_eq!(
        img_a.surfaces.len(),
        img_b.surfaces.len(),
        "layer count differs: {} vs {}",
        img_a.surfaces.len(),
        img_b.surfaces.len()
    );
    for (li, (la, lb)) in img_a.surfaces.iter().zip(&img_b.surfaces).enumerate() {
        assert_eq!(
            la.len(),
            lb.len(),
            "layer {li}: mip count differs: {} vs {}",
            la.len(),
            lb.len()
        );
        for (mi, (sa, sb)) in la.iter().zip(lb).enumerate() {
            assert_surface_data_eq(sa, sb, &format!("layer {li} mip {mi}"));
        }
    }
}
