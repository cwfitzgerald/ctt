use std::env;
use std::path::PathBuf;

/// C++ source files to compile per-ISA inside a namespace wrapper.
const CPP_SOURCES: &[&str] = &[
    "cpp/ProcessRGB.cpp",
    "cpp/ProcessDxtc.cpp",
    "cpp/Decode.cpp",
    "cpp/Tables.cpp",
    "cpp/Dither.cpp",
    "cpp/ColorSpace.cpp",
    "cpp/bc7enc.cpp",
];

struct IsaVariant {
    name: &'static str,
    lib_name: &'static str,
    /// Preprocessor defines to add (name, optional value).
    defines: &'static [(&'static str, Option<&'static str>)],
    /// Compiler flags for GCC/Clang only.
    gnu_flags: &'static [&'static str],
    /// Compiler flags for MSVC only.
    msvc_flags: &'static [&'static str],
}

fn main() {
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=src/ffi_wrapper.cpp");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let is_msvc = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc";

    let variants: Vec<IsaVariant> = match target_arch.as_str() {
        "x86_64" => vec![
            IsaVariant {
                name: "scalar",
                lib_name: "etcpak_scalar",
                // No SIMD defines — etcpak falls back to scalar code paths.
                defines: &[],
                gnu_flags: &[],
                msvc_flags: &[],
            },
            IsaVariant {
                name: "sse41",
                lib_name: "etcpak_sse41",
                defines: &[("__SSE4_1__", None)],
                gnu_flags: &["-msse4.1", "-mpopcnt"],
                // MSVC enables SSE4.1 intrinsics unconditionally via <intrin.h>,
                // but does not define __SSE4_1__. We define it manually.
                msvc_flags: &[],
            },
            IsaVariant {
                name: "avx2",
                lib_name: "etcpak_avx2",
                // MSVC /arch:AVX2 defines __AVX2__ and __AVX__ but NOT __SSE4_1__.
                // etcpak code guarded by __AVX2__ references symbols from __SSE4_1__ blocks.
                defines: &[("__SSE4_1__", None)],
                gnu_flags: &["-mavx2", "-mfma", "-mpopcnt", "-mf16c", "-mbmi", "-mbmi2"],
                msvc_flags: &["/arch:AVX2"],
            },
        ],
        "aarch64" => vec![IsaVariant {
            name: "neon",
            lib_name: "etcpak_neon",
            defines: &[],
            gnu_flags: &[],
            msvc_flags: &[],
        }],
        _ => vec![IsaVariant {
            name: "scalar",
            lib_name: "etcpak_scalar",
            defines: &[],
            gnu_flags: &[],
            msvc_flags: &[],
        }],
    };

    // Compile bcdec.c once as plain C (no SIMD, no namespace).
    build_bcdec();

    // Compile each ISA variant.
    for variant in &variants {
        build_variant(variant, is_msvc);
    }
}

fn build_bcdec() {
    let mut build = cc::Build::new();
    build
        .file("cpp/bcdec.c")
        .warnings(false)
        .include("cpp")
        .std("c11");
    build.compile("etcpak_bcdec");
}

fn build_variant(variant: &IsaVariant, is_msvc: bool) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let variant_dir = out_dir.join(variant.name);
    std::fs::create_dir_all(&variant_dir).expect("failed to create variant output dir");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = manifest_dir.join("cpp");

    // All standard/system headers used anywhere in the etcpak source tree.
    // Pre-including them before the namespace ensures they resolve in the
    // global namespace as the standard requires.
    let std_headers = build_std_headers(is_msvc, variant.name);

    // Build the namespace-wrapped C++ source files.
    let mut build = cc::Build::new();
    build.out_dir(&variant_dir);
    build.cpp(true).warnings(false).include(&cpp_dir);

    if is_msvc {
        build.std("c++20");
        build.flag("/fp:precise");
    } else {
        build.std("c++20");
        build.flag("-ffp-contract=fast");
    }

    // Prevent Windows.h from defining min/max macros that break std::min/max.
    if is_msvc {
        build.define("NOMINMAX", None);
        build.define("WIN32_LEAN_AND_MEAN", None);
    }

    // ISA-specific defines.
    for &(key, value) in variant.defines {
        build.define(key, value);
    }

    // ISA-specific compiler flags.
    if is_msvc {
        for &flag in variant.msvc_flags {
            build.flag(flag);
        }
    } else {
        for &flag in variant.gnu_flags {
            build.flag(flag);
        }
    }

    // Generate namespace wrappers for each C++ source file.
    for &src in CPP_SOURCES {
        let file_name = std::path::Path::new(src)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        let abs_src = cpp_dir.join(file_name);
        let wrapper_path = variant_dir.join(file_name);

        let wrapper = format!(
            "{std_headers}\
             namespace etcpak_{ns} {{\n\
             using ::abs;\n\
             using ::memcpy;\n\
             using ::memset;\n\
             using ::pow;\n\
             using ::sqrt;\n\
             using ::floor;\n\
             using ::ceil;\n\
             using ::log2;\n\
             #include \"{src}\"\n\
             }} // namespace etcpak_{ns}\n",
            std_headers = std_headers,
            ns = variant.name,
            src = abs_src.to_str().unwrap().replace('\\', "/"),
        );

        let needs_write = std::fs::read_to_string(&wrapper_path)
            .map(|existing| existing != wrapper)
            .unwrap_or(true);
        if needs_write {
            std::fs::write(&wrapper_path, &wrapper).expect("failed to write wrapper");
        }

        build.file(&wrapper_path);
    }

    // Compile the extern "C" forwarding wrapper with ISA-specific defines.
    let ffi_wrapper_src = manifest_dir.join("src/ffi_wrapper.cpp");
    let ffi_wrapper_dst = variant_dir.join(format!("ffi_wrapper_{}.cpp", variant.name));

    // Copy the wrapper to the variant dir so it compiles with the right flags.
    let ffi_content =
        std::fs::read_to_string(&ffi_wrapper_src).expect("failed to read ffi_wrapper.cpp");
    let needs_write = std::fs::read_to_string(&ffi_wrapper_dst)
        .map(|existing| existing != ffi_content)
        .unwrap_or(true);
    if needs_write {
        std::fs::write(&ffi_wrapper_dst, &ffi_content).expect("failed to write ffi_wrapper copy");
    }

    build.file(&ffi_wrapper_dst);
    build.define("ETCPAK_NS", format!("etcpak_{}", variant.name).as_str());
    build.define(
        "ETCPAK_PREFIX",
        format!("etcpak_{}_", variant.name).as_str(),
    );

    build.compile(variant.lib_name);
}

fn build_std_headers(is_msvc: bool, variant_name: &str) -> String {
    let mut headers = String::new();

    // C standard headers
    headers.push_str("#include <assert.h>\n");
    headers.push_str("#include <limits.h>\n");
    headers.push_str("#include <math.h>\n");
    headers.push_str("#include <memory.h>\n");
    headers.push_str("#include <stddef.h>\n");
    headers.push_str("#include <stdint.h>\n");
    headers.push_str("#include <stdio.h>\n");
    headers.push_str("#include <stdlib.h>\n");
    headers.push_str("#include <string.h>\n");

    // C++ standard headers — must be pre-included before the namespace opens
    // so that internal std:: references resolve in the global namespace.
    headers.push_str("#include <algorithm>\n");
    headers.push_str("#include <array>\n");
    headers.push_str("#include <bit>\n");
    headers.push_str("#include <cmath>\n");
    headers.push_str("#include <cstdint>\n");
    headers.push_str("#include <cstdlib>\n");
    headers.push_str("#include <cstring>\n");
    headers.push_str("#include <limits>\n");
    headers.push_str("#include <new>\n");
    headers.push_str("#include <type_traits>\n");
    headers.push_str("#include <utility>\n");

    // SIMD headers (platform-dependent)
    match variant_name {
        "neon" => {
            if is_msvc {
                headers.push_str("#include <intrin.h>\n");
                headers.push_str("#include <Windows.h>\n");
                // Windows.h defines RGB(r,g,b) macro via wingdi.h which
                // clashes with ColorSpace.hpp's RGB() method name.
                headers.push_str("#undef RGB\n");
            }
            headers.push_str("#include <arm_neon.h>\n");
        }
        "scalar" => {
            // No SIMD headers for scalar fallback.
            // Still need intrin.h on MSVC for _byteswap intrinsics used
            // behind `#ifdef _MSC_VER` guards (not behind SIMD guards).
            if is_msvc {
                headers.push_str("#include <intrin.h>\n");
            }
        }
        _ => {
            // x86 SIMD variants
            if is_msvc {
                headers.push_str("#include <intrin.h>\n");
            } else {
                headers.push_str("#include <x86intrin.h>\n");
            }

            if variant_name == "sse41" {
                headers.push_str("#include <smmintrin.h>\n");
            }
            if variant_name == "avx2" {
                headers.push_str("#include <immintrin.h>\n");
            }
        }
    }

    headers
}
