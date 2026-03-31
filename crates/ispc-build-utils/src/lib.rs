use std::path::{Path, PathBuf};
use std::process::Command;

pub mod prebuilt;

/// Describes the target platform for ISPC compilation.
pub struct CompileTarget {
    pub out_dir: PathBuf,
    pub target_arch: String,
    pub target_os: String,
    pub target_env: String,
}

impl CompileTarget {
    /// Read target information from Cargo build-script environment variables.
    pub fn from_cargo_env() -> Self {
        Self {
            out_dir: PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set")),
            target_arch: env("CARGO_CFG_TARGET_ARCH"),
            target_os: env("CARGO_CFG_TARGET_OS"),
            target_env: std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default(),
        }
    }

    /// Parse a Rust target triple (e.g. `x86_64-unknown-linux-gnu`) and an
    /// output directory into a `CompileTarget`.
    ///
    /// Handles the common Rust triple formats:
    /// - `x86_64-unknown-linux-gnu` → os=linux, env=gnu
    /// - `x86_64-pc-windows-msvc` → os=windows, env=msvc
    /// - `aarch64-apple-darwin` → os=macos, env=""
    pub fn from_triple(triple: &str, out_dir: PathBuf) -> Self {
        let parts: Vec<&str> = triple.split('-').collect();
        assert!(
            parts.len() >= 3,
            "invalid target triple: {triple} (expected at least arch-vendor-os)"
        );

        let target_arch = parts[0].to_string();

        // The remaining components after arch-vendor encode OS and optionally
        // environment. We join them and match known patterns to extract the
        // Cargo-style target_os and target_env values.
        let rest: Vec<&str> = parts[2..].to_vec();
        let (target_os, target_env) = match rest.as_slice() {
            ["darwin"] => ("macos".to_string(), String::new()),
            ["linux", env] => ("linux".to_string(), env.to_string()),
            ["windows", env] => ("windows".to_string(), env.to_string()),
            ["ios"] => ("ios".to_string(), String::new()),
            ["android" | "androideabi", ..] => ("android".to_string(), String::new()),
            [os] => (os.to_string(), String::new()),
            [os, env, ..] => (os.to_string(), env.to_string()),
            [] => panic!("invalid target triple: {triple} (no OS component)"),
        };

        Self {
            out_dir,
            target_arch,
            target_os,
            target_env,
        }
    }

    /// Returns the object file extension for this target.
    pub fn obj_ext(&self) -> &'static str {
        if self.target_env == "msvc" {
            "obj"
        } else {
            "o"
        }
    }

    /// Returns the static library filename for the given library name.
    pub fn lib_filename(&self, lib_name: &str) -> String {
        if self.target_env == "msvc" {
            format!("{lib_name}.lib")
        } else {
            format!("lib{lib_name}.a")
        }
    }
}

/// Builder for compiling ISPC source files into a static library.
///
/// Designed to be used in `build.rs` scripts. Finds the ISPC compiler on
/// `PATH`, optionally validates its SHA-256 hash, invokes it with the correct
/// flags for the current target triple, and archives the resulting objects into
/// a static library.
pub struct Config {
    files: Vec<PathBuf>,
    opt_level: u32,
    woff: bool,
    fast_math: bool,
    disable_assertions: bool,
}

impl Config {
    #[must_use]
    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            opt_level: 0,
            woff: false,
            fast_math: false,
            disable_assertions: false,
        }
    }

    /// Add an ISPC source file to compile.
    pub fn file(&mut self, path: impl AsRef<Path>) -> &mut Self {
        self.files.push(path.as_ref().to_path_buf());
        self
    }

    /// Set the optimization level (0-3). Default is 0.
    pub fn opt_level(&mut self, level: u32) -> &mut Self {
        assert!(level <= 3, "ISPC optimization level must be 0-3");
        self.opt_level = level;
        self
    }

    /// Suppress all ISPC warnings.
    pub fn woff(&mut self) -> &mut Self {
        self.woff = true;
        self
    }

    /// Enable fast-math optimizations.
    pub fn fast_math(&mut self) -> &mut Self {
        self.fast_math = true;
        self
    }

    /// Disable assertions in ISPC code.
    pub fn disable_assertions(&mut self) -> &mut Self {
        self.disable_assertions = true;
        self
    }

    /// Compile all added ISPC files and archive them into a static library
    /// named `lib_name`. Reads target information from Cargo build-script
    /// environment variables. Emits `cargo:rustc-link-lib` and
    /// `cargo:rustc-link-search` instructions.
    ///
    /// The library and any generated headers are placed in `OUT_DIR`.
    pub fn compile(&self, lib_name: &str) {
        let target = CompileTarget::from_cargo_env();
        self.compile_to(lib_name, &target);
        println!("cargo:rustc-link-lib=static={lib_name}");
        println!(
            "cargo:rustc-link-search=native={}",
            target.out_dir.display()
        );
    }

    /// Compile all added ISPC files and archive them into a static library
    /// named `lib_name` for the given target. Does NOT emit cargo directives.
    ///
    /// The library and any generated headers are placed in `target.out_dir`.
    pub fn compile_to(&self, lib_name: &str, target: &CompileTarget) {
        assert!(!self.files.is_empty(), "no ISPC files added to compile");

        let ispc_path = find_ispc();

        let (ispc_targets, ispc_arch) = targets_for_arch(&target.target_arch);
        let ispc_target_os = target_os_for_cargo(&target.target_os);
        let obj_ext = target.obj_ext();

        // Use a subdirectory per library to avoid object file name conflicts
        // when compiling multiple ISPC files in the same out_dir.
        let compile_dir = target.out_dir.join(format!("ispc_{lib_name}"));
        std::fs::create_dir_all(&compile_dir).expect("failed to create ISPC compile directory");

        for file in &self.files {
            let stem = file
                .file_stem()
                .expect("ISPC file has no stem")
                .to_str()
                .expect("ISPC file stem is not UTF-8");

            let out_obj = compile_dir.join(format!("{stem}.{obj_ext}"));
            let out_header = compile_dir.join(format!("{stem}_ispc.h"));

            let mut cmd = Command::new(&ispc_path);
            cmd.arg(file);
            cmd.arg(format!("--target={ispc_targets}"));
            cmd.arg(format!("--arch={ispc_arch}"));
            cmd.arg(format!("--target-os={ispc_target_os}"));
            cmd.arg(format!("-O{}", self.opt_level));
            cmd.arg("-o").arg(&out_obj);
            cmd.arg("-h").arg(&out_header);

            if self.woff {
                cmd.arg("--woff");
            }
            if self.fast_math {
                cmd.arg("--opt=fast-math");
            }
            if self.disable_assertions {
                cmd.arg("--opt=disable-assertions");
            }

            // Position-independent code on non-Windows platforms.
            if target.target_os != "windows" {
                cmd.arg("--pic");
            }

            let status = cmd
                .status()
                .expect("failed to execute ispc - is it on PATH?");
            assert!(
                status.success(),
                "ispc compilation of {} failed",
                file.display()
            );

            // Only emit rerun-if-changed when running inside a build script.
            if std::env::var_os("OUT_DIR").is_some() {
                println!("cargo:rerun-if-changed={}", file.display());
            }
        }

        // Collect all generated object files from the compile directory.
        let objects = collect_objects(&compile_dir, obj_ext);
        assert!(
            !objects.is_empty(),
            "no object files found after ISPC compilation"
        );

        // Archive into a static library in out_dir.
        let lib_filename = target.lib_filename(lib_name);
        let lib_path = target.out_dir.join(&lib_filename);

        create_archive(&lib_path, &objects, &target.target_arch, &target.target_env);
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns the ISPC target list and architecture flag for the given Cargo
/// target architecture.
pub fn targets_for_arch(cargo_arch: &str) -> (&'static str, &'static str) {
    match cargo_arch {
        "x86" | "x86_64" => (
            "sse2-i32x4,sse4-i32x4,avx1-i32x8,avx2-i32x8,avx512skx-i32x16",
            if cargo_arch == "x86" { "x86" } else { "x86-64" },
        ),
        "aarch64" => ("neon-i32x8", "aarch64"),
        other => panic!("unsupported target architecture for ISPC: {other}"),
    }
}

/// Maps a Cargo `target_os` to the ISPC `--target-os` value.
pub fn target_os_for_cargo(cargo_os: &str) -> &'static str {
    match cargo_os {
        "linux" => "linux",
        "macos" => "macos",
        "windows" => "windows",
        other => panic!("unsupported target OS for ISPC: {other}"),
    }
}

/// Find the ISPC compiler binary. Checks `ISPC_PATH` env var first, then
/// searches `PATH`.
fn find_ispc() -> PathBuf {
    if let Ok(path) = std::env::var("ISPC_PATH") {
        let path = PathBuf::from(path);
        assert!(
            path.exists(),
            "ISPC_PATH points to non-existent file: {}",
            path.display()
        );
        return path;
    }

    // Resolve the full path via `which`/`where` so we can compute its hash.
    let output = if cfg!(target_os = "windows") {
        Command::new("where").arg("ispc").output()
    } else {
        Command::new("which").arg("ispc").output()
    };

    let output = output.expect("failed to run which/where to locate ispc");
    assert!(
        output.status.success(),
        "ispc not found on PATH. Install ISPC and ensure it is on your PATH, \
         or set the ISPC_PATH environment variable."
    );

    let stdout = String::from_utf8(output.stdout).expect("non-UTF-8 output from which/where");
    // `where` on Windows may return multiple lines; take the first.
    let first_line = stdout
        .lines()
        .next()
        .expect("empty output from which/where");
    PathBuf::from(first_line.trim())
}

/// Collect all object files from a directory.
fn collect_objects(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut objects = Vec::new();
    for entry in std::fs::read_dir(dir).expect("failed to read compile directory") {
        let entry = entry.expect("failed to read directory entry");
        let path = entry.path();
        if path.extension().is_some_and(|e| e == ext) {
            objects.push(path);
        }
    }
    objects.sort();
    objects
}

/// Create a static library archive from object files.
fn create_archive(lib_path: &Path, objects: &[PathBuf], target_arch: &str, target_env: &str) {
    if target_env == "msvc" {
        let mut cmd = find_msvc_tools::find(target_arch, "lib.exe")
            .expect("failed to find lib.exe - ensure MSVC build tools are installed");
        cmd.arg("/NOLOGO");
        cmd.arg(format!("/OUT:{}", lib_path.display()));
        for obj in objects {
            cmd.arg(obj);
        }
        let status = cmd.status().expect("failed to run lib.exe");
        assert!(status.success(), "lib.exe archiver failed");
    } else {
        let mut cmd = Command::new("ar");
        cmd.arg("rcs");
        cmd.arg(lib_path);
        for obj in objects {
            cmd.arg(obj);
        }
        let status = cmd
            .status()
            .expect("failed to run ar - ensure binutils are installed");
        assert!(status.success(), "ar archiver failed");
    }
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} not set"))
}
