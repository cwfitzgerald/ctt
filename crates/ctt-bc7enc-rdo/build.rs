use ispc_compile::{Config, OptimizationOpt, TargetISA, bindgen::builder};

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
        .optimization_opt(OptimizationOpt::FastMath)
        .optimization_opt(OptimizationOpt::DisableAssertions)
        .target_isas(target_isas)
        .file("ispc/bc7e.ispc")
        .bindgen_builder(
            builder()
                .allowlist_function("bc7e_compress_block_init")
                .allowlist_function("bc7e_compress_blocks")
                .allowlist_function("bc7e_compress_block_params_init.*"),
        )
        .compile("bc7e");
}
