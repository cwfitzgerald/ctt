fn main() {
    #[cfg(all(feature = "prebuilt", feature = "build-from-source"))]
    compile_error!("Cannot enable both 'prebuilt' and 'build-from-source' — pick one");

    #[cfg(feature = "prebuilt")]
    {
        let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        ispc_build_utils::prebuilt::link_prebuilt_from(
            &["bc7e"],
            &manifest_dir.join("prebuilt/bins"),
        );
    }

    #[cfg(feature = "build-from-source")]
    {
        let mut config = ispc_build_utils::Config::new();
        config
            .file("ispc/bc7e.ispc")
            .opt_level(2)
            .woff()
            .fast_math()
            .disable_assertions();
        config.compile("bc7e");
    }
}
