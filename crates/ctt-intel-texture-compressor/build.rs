use ispc_compile::{Config, TargetISA, bindgen::builder};

fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_isas = match target_arch.as_str() {
        "x86" | "x86_64" => vec![
            TargetISA::SSE2i32x4,
            TargetISA::SSE4i32x4,
            TargetISA::AVX1i32x8,
            TargetISA::AVX2i32x8,
            TargetISA::AVX512SKXx16,
        ],
        "arm" | "aarch64" => vec![TargetISA::Neoni32x8],
        x => panic!("Unsupported target architecture {x}"),
    };

    Config::new()
        .opt_level(2)
        .woff()
        .target_isas(target_isas.clone())
        .file("ispc/kernel.ispc")
        .bindgen_builder(builder().allowlist_function(r#"CompressBlocks(BC\dH?|ETC1)_ispc"#))
        .compile("kernel");

    Config::new()
        .opt_level(2)
        .woff()
        .target_isas(target_isas)
        .file("ispc/kernel_astc.ispc")
        .bindgen_builder(
            builder()
                .allowlist_function("astc_rank_ispc")
                .allowlist_function("astc_encode_ispc")
                .allowlist_function("get_programCount"),
        )
        .compile("kernel_astc");

    // ASTC encoder `extern "C"`'s some code, so we need to make sure to link
    // and compile that in. The relevant codepath using this functionality is
    // completely commented out and only results in linker errors on MSVC which
    // is unable to deadstrip it and the requirement for a single symbol.
    let out_dir = std::env::var("OUT_DIR").unwrap();

    cc::Build::new()
        .include(&out_dir)
        .include("ispc")
        .file("ispc/ispc_texcomp_astc.cpp")
        .cpp(true)
        .compile("ispc_texcomp_astc");
}
