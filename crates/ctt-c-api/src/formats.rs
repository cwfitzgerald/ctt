//! Curated set of common [`Format`](crate::Format) values exposed as
//! constants for convenience.
//!
//! The full set of values is identical to Vulkan's `VkFormat` enumeration;
//! arbitrary VkFormat values may be passed wherever a [`Format`](crate::Format)
//! is expected.
//!
//! `0` corresponds to `VK_FORMAT_UNDEFINED` and is **not** a valid input
//! format for any ctt entry point.

use crate::Format;

// Uncompressed — single channel
pub const CTT_FORMAT_R8_UNORM: Format = 9;
pub const CTT_FORMAT_R8_SNORM: Format = 10;
pub const CTT_FORMAT_R8_UINT: Format = 13;
pub const CTT_FORMAT_R8_SINT: Format = 14;
pub const CTT_FORMAT_R8_SRGB: Format = 15;

// Two channel
pub const CTT_FORMAT_R8G8_UNORM: Format = 16;
pub const CTT_FORMAT_R8G8_SNORM: Format = 17;
pub const CTT_FORMAT_R8G8_UINT: Format = 20;
pub const CTT_FORMAT_R8G8_SINT: Format = 21;
pub const CTT_FORMAT_R8G8_SRGB: Format = 22;

// Four channel — RGBA8
pub const CTT_FORMAT_R8G8B8A8_UNORM: Format = 37;
pub const CTT_FORMAT_R8G8B8A8_SNORM: Format = 38;
pub const CTT_FORMAT_R8G8B8A8_UINT: Format = 41;
pub const CTT_FORMAT_R8G8B8A8_SINT: Format = 42;
pub const CTT_FORMAT_R8G8B8A8_SRGB: Format = 43;

// Four channel — BGRA8
pub const CTT_FORMAT_B8G8R8A8_UNORM: Format = 44;
pub const CTT_FORMAT_B8G8R8A8_SNORM: Format = 45;
pub const CTT_FORMAT_B8G8R8A8_UINT: Format = 48;
pub const CTT_FORMAT_B8G8R8A8_SINT: Format = 49;
pub const CTT_FORMAT_B8G8R8A8_SRGB: Format = 50;

// Packed 32-bit
pub const CTT_FORMAT_A2B10G10R10_UNORM_PACK32: Format = 64;
pub const CTT_FORMAT_A2B10G10R10_UINT_PACK32: Format = 68;
pub const CTT_FORMAT_B10G11R11_UFLOAT_PACK32: Format = 122;

// 16-bit per channel
pub const CTT_FORMAT_R16_UNORM: Format = 70;
pub const CTT_FORMAT_R16_SNORM: Format = 71;
pub const CTT_FORMAT_R16_UINT: Format = 74;
pub const CTT_FORMAT_R16_SINT: Format = 75;
pub const CTT_FORMAT_R16_SFLOAT: Format = 76;

pub const CTT_FORMAT_R16G16_UNORM: Format = 77;
pub const CTT_FORMAT_R16G16_SNORM: Format = 78;
pub const CTT_FORMAT_R16G16_UINT: Format = 81;
pub const CTT_FORMAT_R16G16_SINT: Format = 82;
pub const CTT_FORMAT_R16G16_SFLOAT: Format = 83;

pub const CTT_FORMAT_R16G16B16A16_UNORM: Format = 91;
pub const CTT_FORMAT_R16G16B16A16_SNORM: Format = 92;
pub const CTT_FORMAT_R16G16B16A16_UINT: Format = 95;
pub const CTT_FORMAT_R16G16B16A16_SINT: Format = 96;
pub const CTT_FORMAT_R16G16B16A16_SFLOAT: Format = 97;

// 32-bit per channel
pub const CTT_FORMAT_R32_UINT: Format = 98;
pub const CTT_FORMAT_R32_SINT: Format = 99;
pub const CTT_FORMAT_R32_SFLOAT: Format = 100;
pub const CTT_FORMAT_R32G32_UINT: Format = 101;
pub const CTT_FORMAT_R32G32_SINT: Format = 102;
pub const CTT_FORMAT_R32G32_SFLOAT: Format = 103;
pub const CTT_FORMAT_R32G32B32A32_UINT: Format = 107;
pub const CTT_FORMAT_R32G32B32A32_SINT: Format = 108;
pub const CTT_FORMAT_R32G32B32A32_SFLOAT: Format = 109;

// Block-compressed — BC family
pub const CTT_FORMAT_BC1_RGB_UNORM_BLOCK: Format = 131;
pub const CTT_FORMAT_BC1_RGB_SRGB_BLOCK: Format = 132;
pub const CTT_FORMAT_BC1_RGBA_UNORM_BLOCK: Format = 133;
pub const CTT_FORMAT_BC1_RGBA_SRGB_BLOCK: Format = 134;
pub const CTT_FORMAT_BC2_UNORM_BLOCK: Format = 135;
pub const CTT_FORMAT_BC2_SRGB_BLOCK: Format = 136;
pub const CTT_FORMAT_BC3_UNORM_BLOCK: Format = 137;
pub const CTT_FORMAT_BC3_SRGB_BLOCK: Format = 138;
pub const CTT_FORMAT_BC4_UNORM_BLOCK: Format = 139;
pub const CTT_FORMAT_BC4_SNORM_BLOCK: Format = 140;
pub const CTT_FORMAT_BC5_UNORM_BLOCK: Format = 141;
pub const CTT_FORMAT_BC5_SNORM_BLOCK: Format = 142;
pub const CTT_FORMAT_BC6H_UFLOAT_BLOCK: Format = 143;
pub const CTT_FORMAT_BC6H_SFLOAT_BLOCK: Format = 144;
pub const CTT_FORMAT_BC7_UNORM_BLOCK: Format = 145;
pub const CTT_FORMAT_BC7_SRGB_BLOCK: Format = 146;

// ETC2 / EAC family
pub const CTT_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: Format = 147;
pub const CTT_FORMAT_ETC2_R8G8B8_SRGB_BLOCK: Format = 148;
pub const CTT_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK: Format = 149;
pub const CTT_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK: Format = 150;
pub const CTT_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: Format = 151;
pub const CTT_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK: Format = 152;
pub const CTT_FORMAT_EAC_R11_UNORM_BLOCK: Format = 153;
pub const CTT_FORMAT_EAC_R11_SNORM_BLOCK: Format = 154;
pub const CTT_FORMAT_EAC_R11G11_UNORM_BLOCK: Format = 155;
pub const CTT_FORMAT_EAC_R11G11_SNORM_BLOCK: Format = 156;

// ASTC LDR
pub const CTT_FORMAT_ASTC_4X4_UNORM_BLOCK: Format = 157;
pub const CTT_FORMAT_ASTC_4X4_SRGB_BLOCK: Format = 158;
pub const CTT_FORMAT_ASTC_5X4_UNORM_BLOCK: Format = 159;
pub const CTT_FORMAT_ASTC_5X4_SRGB_BLOCK: Format = 160;
pub const CTT_FORMAT_ASTC_5X5_UNORM_BLOCK: Format = 161;
pub const CTT_FORMAT_ASTC_5X5_SRGB_BLOCK: Format = 162;
pub const CTT_FORMAT_ASTC_6X5_UNORM_BLOCK: Format = 163;
pub const CTT_FORMAT_ASTC_6X5_SRGB_BLOCK: Format = 164;
pub const CTT_FORMAT_ASTC_6X6_UNORM_BLOCK: Format = 165;
pub const CTT_FORMAT_ASTC_6X6_SRGB_BLOCK: Format = 166;
pub const CTT_FORMAT_ASTC_8X5_UNORM_BLOCK: Format = 167;
pub const CTT_FORMAT_ASTC_8X5_SRGB_BLOCK: Format = 168;
pub const CTT_FORMAT_ASTC_8X6_UNORM_BLOCK: Format = 169;
pub const CTT_FORMAT_ASTC_8X6_SRGB_BLOCK: Format = 170;
pub const CTT_FORMAT_ASTC_8X8_UNORM_BLOCK: Format = 171;
pub const CTT_FORMAT_ASTC_8X8_SRGB_BLOCK: Format = 172;
pub const CTT_FORMAT_ASTC_10X5_UNORM_BLOCK: Format = 173;
pub const CTT_FORMAT_ASTC_10X5_SRGB_BLOCK: Format = 174;
pub const CTT_FORMAT_ASTC_10X6_UNORM_BLOCK: Format = 175;
pub const CTT_FORMAT_ASTC_10X6_SRGB_BLOCK: Format = 176;
pub const CTT_FORMAT_ASTC_10X8_UNORM_BLOCK: Format = 177;
pub const CTT_FORMAT_ASTC_10X8_SRGB_BLOCK: Format = 178;
pub const CTT_FORMAT_ASTC_10X10_UNORM_BLOCK: Format = 179;
pub const CTT_FORMAT_ASTC_10X10_SRGB_BLOCK: Format = 180;
pub const CTT_FORMAT_ASTC_12X10_UNORM_BLOCK: Format = 181;
pub const CTT_FORMAT_ASTC_12X10_SRGB_BLOCK: Format = 182;
pub const CTT_FORMAT_ASTC_12X12_UNORM_BLOCK: Format = 183;
pub const CTT_FORMAT_ASTC_12X12_SRGB_BLOCK: Format = 184;

/// Convert a [`Format`] to `ctt::Format` if it is a known/valid value.
/// Returns `None` for `0` (VK_FORMAT_UNDEFINED).
pub(crate) fn to_ctt_format(f: Format) -> Option<ctt::Format> {
    ctt::Format::new(f)
}
