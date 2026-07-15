use std::path::Path;

use anyhow::Result;

use super::{
    clean_and_create, copy_files, copy_text_file, read_text, replace_required, require_dir,
    write_text,
};
use crate::util::workspace_root;

/// Automatically vendored from <https://github.com/GPUOpen-Tools/compressonator>.
/// Regenerate with: `cargo xtask vendor compressonator [--src <path>]`
///
/// `src_dir` is the repository root.
pub fn vendor_compressonator(src_dir: &Path) -> Result<()> {
    let ws = workspace_root();
    require_dir(src_dir)?;

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
    copy_text_file(
        &src_dir.join("license/corelicense.txt"),
        &crate_dir.join("LICENSE-MIT-COMPRESSONATOR.md"),
    )?;

    // Apply patches
    patch_compressonator(&dst_dir)?;

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
