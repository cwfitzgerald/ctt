use std::env;

/// All 22 core library source files.
const SOURCES: &[&str] = &[
    "cpp/astcenc_averages_and_directions.cpp",
    "cpp/astcenc_block_sizes.cpp",
    "cpp/astcenc_color_quantize.cpp",
    "cpp/astcenc_color_unquantize.cpp",
    "cpp/astcenc_compress_symbolic.cpp",
    "cpp/astcenc_compute_variance.cpp",
    "cpp/astcenc_decompress_symbolic.cpp",
    "cpp/astcenc_diagnostic_trace.cpp",
    "cpp/astcenc_entry.cpp",
    "cpp/astcenc_find_best_partitioning.cpp",
    "cpp/astcenc_ideal_endpoints_and_weights.cpp",
    "cpp/astcenc_image.cpp",
    "cpp/astcenc_integer_sequence.cpp",
    "cpp/astcenc_mathlib.cpp",
    "cpp/astcenc_mathlib_softfloat.cpp",
    "cpp/astcenc_partition_tables.cpp",
    "cpp/astcenc_percentile_tables.cpp",
    "cpp/astcenc_pick_best_endpoint_format.cpp",
    "cpp/astcenc_quantization.cpp",
    "cpp/astcenc_symbolic_physical.cpp",
    "cpp/astcenc_weight_align.cpp",
    "cpp/astcenc_weight_quant_xfer_tables.cpp",
];

/// Public API functions that need per-ISA symbol renaming.
const PUBLIC_FUNCTIONS: &[&str] = &[
    "astcenc_config_init",
    "astcenc_context_alloc",
    "astcenc_context_free",
    "astcenc_compress_image",
    "astcenc_compress_reset",
    "astcenc_compress_cancel",
    "astcenc_decompress_image",
    "astcenc_decompress_reset",
    "astcenc_get_block_info",
    "astcenc_get_error_string",
];

struct IsaVariant {
    name: &'static str,
    lib_name: &'static str,
    defines: &'static [(&'static str, &'static str)],
    /// Compiler flags for GCC/Clang only.
    gnu_flags: &'static [&'static str],
    /// Compiler flags for MSVC only.
    msvc_flags: &'static [&'static str],
}

fn main() {
    println!("cargo:rerun-if-changed=cpp");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let is_msvc = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc";

    let variants: Vec<IsaVariant> = match target_arch.as_str() {
        "x86_64" => vec![
            IsaVariant {
                name: "sse2",
                lib_name: "astcenc_sse2",
                defines: &[
                    ("ASTCENC_SSE", "20"),
                    ("ASTCENC_AVX", "0"),
                    ("ASTCENC_NEON", "0"),
                    ("ASTCENC_SVE", "0"),
                    ("ASTCENC_POPCNT", "0"),
                    ("ASTCENC_F16C", "0"),
                    ("ASTCENC_X86_GATHERS", "0"),
                ],
                gnu_flags: &["-msse2"],
                msvc_flags: &[],
            },
            IsaVariant {
                name: "sse41",
                lib_name: "astcenc_sse41",
                defines: &[
                    ("ASTCENC_SSE", "41"),
                    ("ASTCENC_AVX", "0"),
                    ("ASTCENC_NEON", "0"),
                    ("ASTCENC_SVE", "0"),
                    ("ASTCENC_POPCNT", "1"),
                    ("ASTCENC_F16C", "0"),
                    ("ASTCENC_X86_GATHERS", "0"),
                ],
                gnu_flags: &["-msse4.1", "-mpopcnt"],
                msvc_flags: &[],
            },
            IsaVariant {
                name: "avx2",
                lib_name: "astcenc_avx2",
                defines: &[
                    ("ASTCENC_SSE", "41"),
                    ("ASTCENC_AVX", "2"),
                    ("ASTCENC_NEON", "0"),
                    ("ASTCENC_SVE", "0"),
                    ("ASTCENC_POPCNT", "1"),
                    ("ASTCENC_F16C", "1"),
                    ("ASTCENC_X86_GATHERS", "0"),
                ],
                gnu_flags: &["-mavx2", "-mpopcnt", "-mf16c"],
                msvc_flags: &["/arch:AVX2"],
            },
        ],
        "aarch64" => vec![IsaVariant {
            name: "neon",
            lib_name: "astcenc_neon",
            defines: &[
                ("ASTCENC_NEON", "1"),
                ("ASTCENC_SVE", "0"),
                ("ASTCENC_SSE", "0"),
                ("ASTCENC_AVX", "0"),
                ("ASTCENC_POPCNT", "0"),
                ("ASTCENC_F16C", "0"),
            ],
            gnu_flags: &[],
            msvc_flags: &[],
        }],
        other => panic!("unsupported target architecture: {other}"),
    };

    for variant in &variants {
        build_variant(variant, is_msvc);
    }
}

fn build_variant(variant: &IsaVariant, is_msvc: bool) {
    let mut build = cc::Build::new();

    build.cpp(true).std("c++14").include("cpp").warnings(false);

    // ISA-specific defines.
    for &(key, value) in variant.defines {
        build.define(key, value);
    }

    // Performance: allow FP contraction.
    build.define("ASTCENC_NO_INVARIANCE", "1");

    // Force extern "C" linkage on public API functions so Rust can call them.
    // This also adds dllexport/visibility attributes which are harmless for static libs.
    build.define("ASTCENC_DYNAMIC_LIBRARY", None);

    // Symbol renaming: prefix public API functions with ISA name.
    for &func in PUBLIC_FUNCTIONS {
        let renamed = format!(
            "astcenc_{name}_{bare}",
            name = variant.name,
            bare = &func["astcenc_".len()..]
        );
        build.define(func, renamed.as_str());
    }

    // ISA-specific compiler flags.
    if is_msvc {
        for &flag in variant.msvc_flags {
            build.flag(flag);
        }
        // FP contraction for non-invariant builds.
        build.flag("/fp:precise");
    } else {
        for &flag in variant.gnu_flags {
            build.flag(flag);
        }
        build.flag("-ffp-contract=fast");
    }

    for &src in SOURCES {
        build.file(src);
    }

    build.compile(variant.lib_name);
}
