fn main() {
    #[cfg(all(feature = "prebuilt", feature = "build-from-source"))]
    compile_error!("features `prebuilt` and `build-from-source` are mutually exclusive — pick one");

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
