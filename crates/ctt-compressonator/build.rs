use std::env;
use std::path::PathBuf;

/// Main source files (compiled without special SIMD flags).
const MAIN_SOURCES: &[&str] = &[
    "cpp/source/cmp_core.cpp",
    "cpp/shaders/bc1_encode_kernel.cpp",
    "cpp/shaders/bc2_encode_kernel.cpp",
    "cpp/shaders/bc3_encode_kernel.cpp",
    "cpp/shaders/bc4_encode_kernel.cpp",
    "cpp/shaders/bc5_encode_kernel.cpp",
    "cpp/shaders/bc6_encode_kernel.cpp",
    "cpp/shaders/bc7_encode_kernel.cpp",
    "cpp/cmp_math/cpu_extensions.cpp",
    "cpp/cmp_math/cmp_math_common.cpp",
];

fn main() {
    println!("cargo:rerun-if-changed=cpp");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let is_msvc = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc";
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Main build — all source files except SIMD variants.
    let mut main_build = cc::Build::new();
    main_build
        .cpp(true)
        .std("c++14")
        .warnings(false)
        .include("cpp/source")
        .include("cpp/shaders")
        .include("cpp/cmp_math");

    for &src in MAIN_SOURCES {
        main_build.file(src);
    }

    main_build.compile("cmp_core_main");

    // SIMD variant builds (x86_64 only).
    if target_arch == "x86_64" {
        build_simd(
            "sse",
            "cmp_core_sse",
            "cpp/source/core_simd_sse.cpp",
            if is_msvc { &[] } else { &["-msse4.1"] },
            &out_dir,
        );
        build_simd(
            "avx",
            "cmp_core_avx",
            "cpp/source/core_simd_avx.cpp",
            if is_msvc { &["/arch:AVX"] } else { &["-mavx"] },
            &out_dir,
        );
        build_simd(
            "avx512",
            "cmp_core_avx512",
            "cpp/source/core_simd_avx512.cpp",
            if is_msvc {
                &["/arch:AVX512"]
            } else {
                &["-mavx512f"]
            },
            &out_dir,
        );
    }
}

fn build_simd(name: &str, lib_name: &str, source: &str, flags: &[&str], out_dir: &std::path::Path) {
    let variant_dir = out_dir.join(name);

    let mut build = cc::Build::new();
    build
        .out_dir(&variant_dir)
        .cpp(true)
        .std("c++14")
        .warnings(false)
        .include("cpp/source")
        .include("cpp/shaders")
        .include("cpp/cmp_math")
        .file(source);

    for &flag in flags {
        build.flag(flag);
    }

    build.compile(lib_name);
}
