use std::path::PathBuf;

/// Returns the platform directory name (e.g. `"linux-x86_64"`) for the current
/// Cargo build target.
pub fn platform_dir() -> &'static str {
    let target_arch =
        std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH not set");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS not set");

    platform_dir_for(&target_os, &target_arch)
}

/// Returns the platform directory name for the given OS and architecture.
pub fn platform_dir_for(target_os: &str, target_arch: &str) -> &'static str {
    match (target_os, target_arch) {
        ("linux", "x86_64") => "linux-x86_64",
        ("linux", "aarch64") => "linux-aarch64",
        ("windows", "x86_64") => "windows-x86_64",
        ("windows", "aarch64") => "windows-aarch64",
        ("macos", "x86_64") => "macos-x86_64",
        ("macos", "aarch64") => "macos-aarch64",
        (os, arch) => panic!(
            "no prebuilt ISPC binary available for {os}-{arch}. \
             Use the 'build-from-source' feature instead."
        ),
    }
}

/// Copy prebuilt static libraries from `bins/<platform>/` relative to
/// `CARGO_MANIFEST_DIR` into `OUT_DIR`, and emit the appropriate
/// `cargo:rustc-link-lib` and `cargo:rustc-link-search` directives.
///
/// `lib_names` should be the bare library names (e.g. `["bc7e"]` or
/// `["kernel", "kernel_astc"]`).
pub fn link_prebuilt(lib_names: &[&str]) {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    let platform = platform_dir();
    let bins_dir = manifest_dir.join("bins").join(platform);

    for &lib_name in lib_names {
        let filename = if target_env == "msvc" {
            format!("{lib_name}.lib")
        } else {
            format!("lib{lib_name}.a")
        };

        let src = bins_dir.join(&filename);
        assert!(
            src.exists(),
            "prebuilt binary not found: {}. Run the build-ispc workflow to generate it.",
            src.display()
        );

        println!("cargo:rerun-if-changed={}", src.display());

        let dst = out_dir.join(&filename);
        std::fs::copy(&src, &dst).unwrap_or_else(|e| {
            panic!("failed to copy {} to {}: {e}", src.display(), dst.display())
        });

        println!("cargo:rustc-link-lib=static={lib_name}");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
