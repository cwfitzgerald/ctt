fn main() {
    #[cfg(all(feature = "prebuilt", feature = "build-from-source"))]
    compile_error!("features `prebuilt` and `build-from-source` are mutually exclusive — pick one");

    #[cfg(feature = "build-from-source")]
    {
        let mut kernel = ispc_build_utils::Config::new();
        kernel.file("ispc/kernel.ispc").opt_level(2).woff();
        kernel.compile("kernel");
    }
}
