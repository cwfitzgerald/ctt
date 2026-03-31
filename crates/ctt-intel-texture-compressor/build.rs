fn main() {
    // Always compile the C++ ASTC support code that the ISPC kernel_astc
    // module extern "C"'s into.
    cc::Build::new()
        .include("ispc")
        .file("ispc/ispc_texcomp_astc.cpp")
        .cpp(true)
        .compile("ispc_texcomp_astc");

    #[cfg(feature = "build-from-source")]
    {
        let mut kernel = ispc_build_utils::Config::new();
        kernel.file("ispc/kernel.ispc").opt_level(2).woff();
        kernel.compile("kernel");

        let mut kernel_astc = ispc_build_utils::Config::new();
        kernel_astc
            .file("ispc/kernel_astc.ispc")
            .opt_level(2)
            .woff();
        kernel_astc.compile("kernel_astc");
    }

    #[cfg(all(feature = "prebuilt", not(feature = "build-from-source")))]
    {
        let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        ispc_build_utils::prebuilt::link_prebuilt_from(
            &["kernel", "kernel_astc"],
            &manifest_dir.join("prebuilt/bins"),
        );
    }
}
