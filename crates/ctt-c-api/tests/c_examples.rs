//! Compile every C file under `examples/` with the platform compiler (via
//! the `cc` crate's toolchain detection), link it once against the static
//! `libctt_capi` and once against the dynamic library, and run the resulting
//! binary. Failure to compile, link, or run is a test failure.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Linkage {
    Static,
    Dynamic,
}

impl Linkage {
    fn label(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Dynamic => "dynamic",
        }
    }
}

/// System libs the Rust staticlib pulls in on Windows (MSVC). `msvcrt` is
/// already linked via `/defaultlib` and pulls the C++ runtime with it.
const MSVC_STATIC_SYS_LIBS: &[&str] = &[
    "kernel32.lib",
    "ntdll.lib",
    "userenv.lib",
    "ws2_32.lib",
    "dbghelp.lib",
    "advapi32.lib",
];

/// System libs the Rust staticlib pulls in on Linux. `-lstdc++` resolves the
/// C++ symbols introduced by the astcenc / compressonator bundled C++ sources;
/// the C compiler driver does not link it on its own.
const LINUX_STATIC_SYS_LIBS: &[&str] = &[
    "-lpthread",
    "-ldl",
    "-lm",
    "-lrt",
    "-lutil",
    "-lgcc_s",
    "-lstdc++",
];

/// System libs and frameworks the Rust staticlib pulls in on macOS. `-lc++`
/// covers the same C++ object files described in [`LINUX_STATIC_SYS_LIBS`].
const MACOS_STATIC_SYS_LIBS: &[&str] = &[
    "-framework",
    "Security",
    "-framework",
    "CoreFoundation",
    "-liconv",
    "-lresolv",
    "-lc++",
];

fn artifacts_dir() -> PathBuf {
    let mut p = std::env::current_exe().expect("current_exe");
    p.pop();
    p
}

fn lib_path(dir: &Path, linkage: Linkage) -> PathBuf {
    if cfg!(target_os = "windows") {
        match linkage {
            Linkage::Static => dir.join("ctt_capi.lib"),
            Linkage::Dynamic => dir.join("ctt_capi.dll.lib"),
        }
    } else if cfg!(target_os = "macos") {
        match linkage {
            Linkage::Static => dir.join("libctt_capi.a"),
            Linkage::Dynamic => dir.join("libctt_capi.dylib"),
        }
    } else {
        match linkage {
            Linkage::Static => dir.join("libctt_capi.a"),
            Linkage::Dynamic => dir.join("libctt_capi.so"),
        }
    }
}

fn collect_examples(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir({}): {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "c"))
        .collect();
    out.sort();
    out
}

/// Pick the static-link system libs for the current host OS, or `&[]` for
/// Windows (the MSVC path uses [`MSVC_STATIC_SYS_LIBS`] separately).
fn unix_static_sys_libs() -> &'static [&'static str] {
    if cfg!(target_os = "linux") {
        LINUX_STATIC_SYS_LIBS
    } else if cfg!(target_os = "macos") {
        MACOS_STATIC_SYS_LIBS
    } else {
        &[]
    }
}

/// Append `cl.exe`-style arguments to compile `c_file` and link it against
/// `lib` with the appropriate system libs for `linkage`.
fn add_msvc_args(
    cmd: &mut Command,
    c_file: &Path,
    include_dir: &Path,
    exe_path: &Path,
    obj_dir: &Path,
    lib: &Path,
    linkage: Linkage,
) {
    let mut fo: OsString = "/Fo:".into();
    fo.push(obj_dir.as_os_str());
    fo.push("\\");
    let mut fe: OsString = "/Fe:".into();
    fe.push(exe_path.as_os_str());
    let mut inc: OsString = "/I".into();
    inc.push(include_dir.as_os_str());

    cmd.arg(c_file)
        .arg("/nologo")
        .arg(inc)
        .arg(fo)
        .arg(fe)
        .arg("/link")
        .arg(lib);
    if linkage == Linkage::Static {
        cmd.args(MSVC_STATIC_SYS_LIBS);
    }
}

/// Append gcc/clang-style arguments. Static linkage pulls in the platform's
/// system libs; dynamic linkage embeds an rpath so the binary finds the
/// shared object at runtime.
fn add_unix_args(
    cmd: &mut Command,
    c_file: &Path,
    include_dir: &Path,
    exe_path: &Path,
    lib: &Path,
    lib_dir: &Path,
    linkage: Linkage,
) {
    cmd.arg(c_file)
        .arg("-I")
        .arg(include_dir)
        .arg("-o")
        .arg(exe_path)
        .arg(lib);
    match linkage {
        Linkage::Dynamic => {
            cmd.arg(format!("-Wl,-rpath,{}", lib_dir.display()));
        }
        Linkage::Static => {
            cmd.args(unix_static_sys_libs());
        }
    }
}

#[test]
fn c_examples_link_and_run() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tmp_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let target = env!("CTT_C_API_TARGET");
    let host = env!("CTT_C_API_HOST");
    let lib_dir = artifacts_dir();
    let include_dir = crate_dir.join("include");
    let example_dir = crate_dir.join("examples");
    let examples = collect_examples(&example_dir);
    assert!(
        !examples.is_empty(),
        "no .c examples in {}",
        example_dir.display()
    );

    let tool = cc::Build::new()
        .target(target)
        .host(host)
        .opt_level(0)
        .cargo_metadata(false)
        .cargo_warnings(false)
        .get_compiler();
    let is_msvc = tool.is_like_msvc();
    let exe_suffix = if cfg!(windows) { ".exe" } else { "" };

    for c_file in &examples {
        let stem = c_file
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("example file_stem");
        for &linkage in &[Linkage::Static, Linkage::Dynamic] {
            let exe_name = format!("{stem}_{}{exe_suffix}", linkage.label());
            let exe_path = tmp_dir.join(&exe_name);
            let lib = lib_path(&lib_dir, linkage);
            assert!(
                lib.exists(),
                "expected {linkage:?} library at `{}` — run `cargo build -p ctt-c-api` first",
                lib.display()
            );

            let mut cmd = tool.to_command();
            if is_msvc {
                let obj_dir = tmp_dir.join(format!("{stem}_{}", linkage.label()));
                std::fs::create_dir_all(&obj_dir).expect("create obj dir");
                add_msvc_args(
                    &mut cmd,
                    c_file,
                    &include_dir,
                    &exe_path,
                    &obj_dir,
                    &lib,
                    linkage,
                );
            } else {
                add_unix_args(
                    &mut cmd,
                    c_file,
                    &include_dir,
                    &exe_path,
                    &lib,
                    &lib_dir,
                    linkage,
                );
            }

            let output = cmd
                .output()
                .unwrap_or_else(|e| panic!("spawn {cmd:?}: {e}"));
            assert!(
                output.status.success(),
                "compile failed for `{}` ({:?})\ncommand: {:?}\n--stdout--\n{}\n--stderr--\n{}",
                c_file.display(),
                linkage,
                cmd,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );

            let mut run = Command::new(&exe_path);
            run.current_dir(&tmp_dir);
            if linkage == Linkage::Dynamic && cfg!(windows) {
                let mut new_path = OsString::from(lib_dir.as_os_str());
                new_path.push(";");
                new_path.push(std::env::var_os("PATH").unwrap_or_default());
                run.env("PATH", new_path);
            }
            let output = run
                .output()
                .unwrap_or_else(|e| panic!("spawn {exe_path:?}: {e}"));
            assert!(
                output.status.success(),
                "binary `{}` ({:?}) exited with status {:?}\n--stdout--\n{}\n--stderr--\n{}",
                exe_path.display(),
                linkage,
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }
}
