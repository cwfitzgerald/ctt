use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use crate::util::workspace_root;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
pub struct VendorArgs {
    #[command(subcommand)]
    target: Option<VendorTarget>,
}

#[derive(Subcommand, Clone)]
pub enum VendorTarget {
    /// Vendor bc7enc_rdo ISPC source.
    Bc7encRdo {
        /// Path to the bc7enc_rdo source checkout.
        #[arg(long)]
        src: Option<PathBuf>,
    },
    /// Vendor astc-encoder C++ source.
    Astcenc {
        /// Path to the astc-encoder Source directory.
        #[arg(long)]
        src: Option<PathBuf>,
    },
    /// Vendor compressonator CMP_Core source.
    Compressonator {
        /// Path to the compressonator source checkout.
        #[arg(long)]
        src: Option<PathBuf>,
    },
}

pub fn vendor(args: VendorArgs) -> Result<()> {
    let targets: Vec<VendorTarget> = match args.target {
        Some(t) => vec![t],
        None => {
            // Vendor all
            vec![
                VendorTarget::Bc7encRdo { src: None },
                VendorTarget::Astcenc { src: None },
                VendorTarget::Compressonator { src: None },
            ]
        }
    };

    for target in targets {
        match target {
            VendorTarget::Bc7encRdo { src } => vendor_bc7enc_rdo(src)?,
            VendorTarget::Astcenc { src } => vendor_astcenc(src)?,
            VendorTarget::Compressonator { src } => vendor_compressonator(src)?,
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

fn require_dir(path: &Path) -> Result<()> {
    if !path.is_dir() {
        bail!("source directory not found: {}", path.display());
    }
    Ok(())
}

fn clean_and_create(dir: &Path) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    Ok(())
}

fn copy_file(src: &Path, dst: &Path) -> Result<()> {
    std::fs::copy(src, dst)?;
    Ok(())
}

fn copy_files(src_dir: &Path, dst_dir: &Path, files: &[&str]) -> Result<()> {
    for f in files {
        copy_file(&src_dir.join(f), &dst_dir.join(f))?;
    }
    Ok(())
}

fn read_text(path: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

fn write_text(path: &Path, content: &str) -> Result<()> {
    // Normalize to LF
    let content = content.replace("\r\n", "\n");
    Ok(std::fs::write(path, content)?)
}

/// Replace all occurrences of `from` with `to` in `text`, requiring at least
/// one match. Returns an error naming `label` if the pattern is not found.
fn replace_required(text: &mut String, from: &str, to: &str, label: &str) -> Result<()> {
    if !text.contains(from) {
        bail!("patch {label:?}: pattern not found in source");
    }
    *text = text.replace(from, to);
    Ok(())
}

/// Get the current git HEAD revision of a repository at `repo_dir`.
fn git_revision(repo_dir: &Path) -> Result<String> {
    let output = Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(repo_dir)
        .output()?;
    if !output.status.success() {
        bail!(
            "failed to get git revision in {}: {}",
            repo_dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

// ---------------------------------------------------------------------------
// bc7enc_rdo
// ---------------------------------------------------------------------------

/// Automatically vendored from <https://github.com/richgel999/bc7enc_rdo>.
/// Default source: `../bc7enc_rdo` relative to the workspace root.
/// Regenerate with: `cargo xtask vendor bc7enc-rdo [--src <path>]`
fn vendor_bc7enc_rdo(src: Option<PathBuf>) -> Result<()> {
    let ws = workspace_root();
    let src_dir = src.unwrap_or_else(|| ws.join("../bc7enc_rdo"));
    require_dir(&src_dir)?;

    let rev = git_revision(&src_dir)?;

    let dst_dir = ws.join("crates/ctt-bc7enc-rdo/ispc");
    clean_and_create(&dst_dir)?;

    copy_file(&src_dir.join("bc7e.ispc"), &dst_dir.join("bc7e.ispc"))?;

    println!("Vendored bc7enc_rdo from {} (rev {rev})", src_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// astcenc
// ---------------------------------------------------------------------------

/// Automatically vendored from <https://github.com/ARM-software/astc-encoder>.
/// Default source: `../astc-encoder/Source` relative to the workspace root.
/// Regenerate with: `cargo xtask vendor astcenc [--src <path>]`
fn vendor_astcenc(src: Option<PathBuf>) -> Result<()> {
    let ws = workspace_root();
    let src_dir = src.unwrap_or_else(|| ws.join("../astc-encoder/Source"));
    require_dir(&src_dir)?;

    // The git repo root is one level above the Source directory.
    let rev = git_revision(&src_dir.join(".."))?;

    let crate_dir = ws.join("crates/ctt-astcenc");
    let dst_dir = crate_dir.join("cpp");
    clean_and_create(&dst_dir)?;

    // Core library source files
    copy_files(
        &src_dir,
        &dst_dir,
        &[
            "astcenc_averages_and_directions.cpp",
            "astcenc_block_sizes.cpp",
            "astcenc_color_quantize.cpp",
            "astcenc_color_unquantize.cpp",
            "astcenc_compress_symbolic.cpp",
            "astcenc_compute_variance.cpp",
            "astcenc_decompress_symbolic.cpp",
            "astcenc_diagnostic_trace.cpp",
            "astcenc_entry.cpp",
            "astcenc_find_best_partitioning.cpp",
            "astcenc_ideal_endpoints_and_weights.cpp",
            "astcenc_image.cpp",
            "astcenc_integer_sequence.cpp",
            "astcenc_mathlib.cpp",
            "astcenc_mathlib_softfloat.cpp",
            "astcenc_partition_tables.cpp",
            "astcenc_percentile_tables.cpp",
            "astcenc_pick_best_endpoint_format.cpp",
            "astcenc_quantization.cpp",
            "astcenc_symbolic_physical.cpp",
            "astcenc_weight_align.cpp",
            "astcenc_weight_quant_xfer_tables.cpp",
        ],
    )?;

    // Public API header
    copy_file(&src_dir.join("astcenc.h"), &dst_dir.join("astcenc.h"))?;

    // Internal headers
    copy_files(
        &src_dir,
        &dst_dir,
        &[
            "astcenc_internal.h",
            "astcenc_internal_entry.h",
            "astcenc_mathlib.h",
            "astcenc_diagnostic_trace.h",
        ],
    )?;

    // SIMD vector math headers
    copy_files(
        &src_dir,
        &dst_dir,
        &[
            "astcenc_vecmathlib.h",
            "astcenc_vecmathlib_sse_4.h",
            "astcenc_vecmathlib_avx2_8.h",
            "astcenc_vecmathlib_neon_4.h",
            "astcenc_vecmathlib_sve_8.h",
            "astcenc_vecmathlib_none_4.h",
            "astcenc_vecmathlib_common_4.h",
        ],
    )?;

    // License
    copy_file(
        &src_dir.join("../LICENSE.txt"),
        &crate_dir.join("LICENSE-APACHE-ASTCENC.md"),
    )?;

    println!("Vendored astcenc from {} (rev {rev})", src_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// compressonator
// ---------------------------------------------------------------------------

/// Automatically vendored from <https://github.com/GPUOpen-Tools/compressonator>.
/// Default source: `../compressonator` relative to the workspace root.
/// Regenerate with: `cargo xtask vendor compressonator [--src <path>]`
fn vendor_compressonator(src: Option<PathBuf>) -> Result<()> {
    let ws = workspace_root();
    let src_dir = src.unwrap_or_else(|| ws.join("../compressonator"));
    require_dir(&src_dir)?;

    let rev = git_revision(&src_dir)?;

    let crate_dir = ws.join("crates/ctt-compressonator");
    let dst_dir = crate_dir.join("cpp");
    clean_and_create(&dst_dir)?;
    std::fs::create_dir_all(dst_dir.join("source"))?;
    std::fs::create_dir_all(dst_dir.join("shaders"))?;
    std::fs::create_dir_all(dst_dir.join("cmp_math"))?;

    // cmp_core/source/
    copy_files(
        &src_dir.join("cmp_core/source"),
        &dst_dir.join("source"),
        &[
            "cmp_core.h",
            "cmp_core.cpp",
            "core_simd.h",
            "core_simd_sse.cpp",
            "core_simd_avx.cpp",
            "core_simd_avx512.cpp",
            "cmp_math_vec4.h",
            "cmp_math_func.h",
        ],
    )?;

    // cmp_core/shaders/
    copy_files(
        &src_dir.join("cmp_core/shaders"),
        &dst_dir.join("shaders"),
        &[
            "bc1_encode_kernel.cpp",
            "bc1_encode_kernel.h",
            "bc2_encode_kernel.cpp",
            "bc2_encode_kernel.h",
            "bc3_encode_kernel.cpp",
            "bc3_encode_kernel.h",
            "bc4_encode_kernel.cpp",
            "bc4_encode_kernel.h",
            "bc5_encode_kernel.cpp",
            "bc5_encode_kernel.h",
            "bc6_encode_kernel.cpp",
            "bc6_encode_kernel.h",
            "bc7_encode_kernel.cpp",
            "bc7_encode_kernel.h",
            "common_def.h",
            "bcn_common_kernel.h",
            "bcn_common_api.h",
            "bc1_cmp.h",
            "bc1_common_kernel.h",
            "bc6_common_encoder.h",
            "bc7_common_encoder.h",
            "bc7_cmpmsc.h",
        ],
    )?;

    // applications/_libs/cmp_math/
    copy_files(
        &src_dir.join("applications/_libs/cmp_math"),
        &dst_dir.join("cmp_math"),
        &[
            "cpu_extensions.cpp",
            "cpu_extensions.h",
            "cmp_math_common.cpp",
            "cmp_math_common.h",
        ],
    )?;

    // License
    copy_file(
        &src_dir.join("license/corelicense.txt"),
        &crate_dir.join("LICENSE-MIT-COMPRESSONATOR.md"),
    )?;

    // Apply patches
    patch_compressonator(&dst_dir)?;

    println!(
        "Vendored compressonator from {} (rev {rev})",
        src_dir.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// compressonator patches
// ---------------------------------------------------------------------------

fn patch_compressonator(dst_dir: &Path) -> Result<()> {
    patch_cmp_core_h(&dst_dir.join("source/cmp_core.h"))?;
    patch_extern_c_on_cpp_files(&dst_dir.join("source"))?;
    patch_extern_c_on_cpp_files(&dst_dir.join("shaders"))?;
    patch_cpu_extensions(&dst_dir.join("cmp_math/cpu_extensions.cpp"))?;
    patch_core_simd_h(&dst_dir.join("source/core_simd.h"))?;
    patch_bc1_cmp_h(&dst_dir.join("shaders/bc1_cmp.h"))?;
    Ok(())
}

fn patch_cmp_core_h(path: &Path) -> Result<()> {
    let mut text = read_text(path)?;

    replace_required(
        &mut text,
        "// API Definitions",
        concat!(
            "#ifdef __cplusplus\n",
            "extern \"C\" {\n",
            "#endif\n",
            "// API Definitions",
        ),
        "cmp_core.h: extern C opening",
    )?;

    replace_required(
        &mut text,
        "#endif  // CMP_CORE",
        concat!(
            "#ifdef __cplusplus\n",
            "}\n",
            "#endif\n",
            "#endif  // CMP_CORE",
        ),
        "cmp_core.h: extern C closing",
    )?;

    replace_required(
        &mut text,
        "const void* options = NULL)",
        "const void* options CMP_DEFAULTNULL)",
        "cmp_core.h: default arg fix",
    )?;

    write_text(path, &text)
}

fn patch_extern_c_on_cpp_files(dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("cpp") {
            let mut text = read_text(&path)?;
            // These are optional per-file — not every .cpp has CMP_CDECL functions.
            text = text.replace("\nint CMP_CDECL ", "\nextern \"C\" int CMP_CDECL ");
            text = text.replace("\nvoid CMP_CDECL ", "\nextern \"C\" void CMP_CDECL ");
            write_text(&path, &text)?;
        }
    }
    Ok(())
}

fn patch_cpu_extensions(path: &Path) -> Result<()> {
    let mut text = read_text(path)?;

    replace_required(
        &mut text,
        concat!("#ifdef _WIN32\n", "#include <intrin.h>\n", "#endif",),
        concat!(
            "#if defined(_M_X64) || defined(_M_IX86)\n",
            "#include <intrin.h>\n",
            "#elif defined(__x86_64__) || defined(__i386__)\n",
            "#include <cpuid.h>\n",
            "#endif",
        ),
        "cpu_extensions.cpp: cpuid include",
    )?;

    replace_required(
        &mut text,
        concat!(
            "#ifdef _WIN32\n",
            "    __cpuidex(outInfo, functionID, 0);  // defined in intrin.h\n",
            "#else\n",
            "    // To Do\n",
            "    //__cpuid_count(0, function_id, outInfo[0], outInfo[1], outInfo[2], outInfo[3]);\n",
            "#endif",
        ),
        concat!(
            "#if defined(_M_X64) || defined(_M_IX86)\n",
            "    __cpuidex(outInfo, functionID, 0);\n",
            "#elif defined(__x86_64__) || defined(__i386__)\n",
            "    __cpuid_count(functionID, 0, outInfo[0], outInfo[1], outInfo[2], outInfo[3]);\n",
            "#else\n",
            "    (void)functionID;\n",
            "    outInfo[0] = outInfo[1] = outInfo[2] = outInfo[3] = 0;\n",
            "#endif",
        ),
        "cpu_extensions.cpp: GetCPUID cross-platform",
    )?;

    replace_required(
        &mut text,
        "#ifndef __linux__",
        "#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)",
        "cpu_extensions.cpp: GetCPUExtensions guard",
    )?;

    write_text(path, &text)
}

fn patch_core_simd_h(path: &Path) -> Result<()> {
    let mut text = read_text(path)?;

    replace_required(
        &mut text,
        "float sse_bc1ComputeBestEndpoints",
        concat!(
            "#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)\n",
            "float sse_bc1ComputeBestEndpoints",
        ),
        "core_simd.h: x86 guard opening",
    )?;

    replace_required(
        &mut text,
        "float avx512_bc1ComputeBestEndpoints(float*, float*, float*, float*, float*, int, int);",
        concat!(
            "float avx512_bc1ComputeBestEndpoints(float*, float*, float*, float*, float*, int, int);\n",
            "#endif // x86",
        ),
        "core_simd.h: x86 guard closing",
    )?;

    write_text(path, &text)
}

fn patch_bc1_cmp_h(path: &Path) -> Result<()> {
    let mut text = read_text(path)?;

    replace_required(
        &mut text,
        concat!(
            "    if (useAVX512 && IsAvailableAVX512(extensions))\n",
            "    {\n",
            "        cpu_bc1ComputeBestEndpoints = avx512_bc1ComputeBestEndpoints;",
        ),
        concat!(
            "#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)\n",
            "    if (useAVX512 && IsAvailableAVX512(extensions))\n",
            "    {\n",
            "        cpu_bc1ComputeBestEndpoints = avx512_bc1ComputeBestEndpoints;",
        ),
        "bc1_cmp.h: SIMD dispatch guard opening",
    )?;

    replace_required(
        &mut text,
        concat!(
            "        cpu_bc1ComputeBestEndpoints = sse_bc1ComputeBestEndpoints;\n",
            "    }\n",
            "    else\n",
            "    {\n",
            "        cpu_bc1ComputeBestEndpoints = _cpu_bc1ComputeBestEndpoints;\n",
            "    }",
        ),
        concat!(
            "        cpu_bc1ComputeBestEndpoints = sse_bc1ComputeBestEndpoints;\n",
            "    }\n",
            "    else\n",
            "    {\n",
            "        cpu_bc1ComputeBestEndpoints = _cpu_bc1ComputeBestEndpoints;\n",
            "    }\n",
            "#else\n",
            "    cpu_bc1ComputeBestEndpoints = _cpu_bc1ComputeBestEndpoints;\n",
            "#endif // x86",
        ),
        "bc1_cmp.h: SIMD dispatch guard closing",
    )?;

    write_text(path, &text)
}
