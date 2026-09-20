fn main() {
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=src/ffi_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/ffi_wrapper.h");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .warnings(false)
        // Basis Universal requires aliasing through different pointer types.
        .flag_if_supported("-fno-strict-aliasing")
        .include("cpp")
        .define("BASISD_SUPPORT_KTX2", "0")
        .define("BASISD_SUPPORT_KTX2_ZSTD", "0")
        .define("BASISD_SUPPORT_DXT1", "0")
        .define("BASISD_SUPPORT_DXT5A", "0")
        .define("BASISD_SUPPORT_BC7_MODE5", "0")
        .define("BASISD_SUPPORT_PVRTC1", "0")
        .define("BASISD_SUPPORT_ETC2_EAC_A8", "0")
        .define("BASISD_SUPPORT_ASTC", "0")
        .define("BASISD_SUPPORT_ATC", "0")
        .define("BASISD_SUPPORT_ETC2_EAC_RG11", "0")
        .define("BASISD_SUPPORT_FXT1", "0")
        .define("BASISD_SUPPORT_PVRTC2", "0")
        .define(
            "BASISD_IS_BIG_ENDIAN",
            if std::env::var("CARGO_CFG_TARGET_ENDIAN").unwrap() == "big" {
                "1"
            } else {
                "0"
            },
        )
        .file("cpp/basisu_transcoder.cpp")
        .file("src/ffi_wrapper.cpp")
        .compile("ctt_bc7f");
}
