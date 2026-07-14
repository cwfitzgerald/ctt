use std::path::Path;

use anyhow::Result;

use super::{clean_and_create, copy_files, copy_text_file, require_dir};
use crate::util::workspace_root;

/// Automatically vendored from <https://github.com/ARM-software/astc-encoder>.
/// Regenerate with: `cargo xtask vendor astcenc [--src <path>]`
///
/// `src_dir` is the repository root; the C++ sources live under `Source/`.
pub fn vendor_astcenc(src_dir: &Path) -> Result<()> {
    let ws = workspace_root();
    let source = src_dir.join("Source");
    require_dir(&source)?;

    let crate_dir = ws.join("crates/ctt-astcenc");
    let dst_dir = crate_dir.join("cpp");
    clean_and_create(&dst_dir)?;

    // Core library source files
    copy_files(
        &source,
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
    copy_text_file(&source.join("astcenc.h"), &dst_dir.join("astcenc.h"))?;

    // Internal headers
    copy_files(
        &source,
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
        &source,
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
    copy_text_file(
        &src_dir.join("LICENSE.txt"),
        &crate_dir.join("LICENSE-APACHE-ASTCENC.md"),
    )?;

    Ok(())
}
