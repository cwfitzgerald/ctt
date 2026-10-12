use std::path::Path;

use anyhow::Result;

use super::{
    clean_and_create, copy_files, copy_text_file, read_text, replace_required, require_dir,
    write_text,
};
use crate::util::workspace_root;

/// Automatically vendored from <https://github.com/wolfpld/etcpak>.
/// Regenerate with: `cargo xtask vendor etcpak [--src <path>]`
///
/// `src_dir` is the repository root.
pub fn vendor_etcpak(src_dir: &Path) -> Result<()> {
    let ws = workspace_root();
    require_dir(src_dir)?;

    let crate_dir = ws.join("crates/ctt-etcpak");
    let dst_dir = crate_dir.join("cpp");
    clean_and_create(&dst_dir)?;

    // C++ source files
    copy_files(
        src_dir,
        &dst_dir,
        &[
            "ProcessRGB.cpp",
            "ProcessDxtc.cpp",
            "Decode.cpp",
            "Tables.cpp",
            "Dither.cpp",
            "bc7enc.cpp",
        ],
    )?;

    patch_constant_simd_tables(&dst_dir.join("Tables.cpp"))?;

    // C source file
    copy_text_file(&src_dir.join("bcdec.c"), &dst_dir.join("bcdec.c"))?;

    // Headers
    copy_files(
        src_dir,
        &dst_dir,
        &[
            "ProcessRGB.hpp",
            "ProcessDxtc.hpp",
            "Decode.hpp",
            "Tables.hpp",
            "Dither.hpp",
            "bc7enc.h",
            "bcdec.h",
            "Math.hpp",
            "Vector.hpp",
            "ForceInline.hpp",
            "ProcessCommon.hpp",
        ],
    )?;

    // License
    copy_text_file(
        &src_dir.join("LICENSE.txt"),
        &crate_dir.join("LICENSE-BSD-ETCPAK.md"),
    )?;

    Ok(())
}

/// Make the SIMD tables in `Tables.cpp` constant-initialized.
///
/// `build.rs` compiles each source once per ISA and links all copies. The
/// static initializers of every copy run at load, so an initializer that uses
/// `_mm_setr_*` executes AVX2 instructions on CPUs that do not have AVX2.
/// `constinit` makes all initializers of these tables run at compile time.
fn patch_constant_simd_tables(path: &Path) -> Result<()> {
    const LABEL: &str = "etcpak: constant SIMD tables";
    let mut text = read_text(path)?;

    // MSVC defines `__m128i` and `__m256i` as unions, and `std::bit_cast`
    // does not accept them in constant expressions. MSVC also does not accept
    // a `__m128i{ ... }` temporary, so its macros expand to a bare braced list.
    replace_required(
        &mut text,
        "#include \"Tables.hpp\"\n",
        r#"#include "Tables.hpp"

#if defined _MSC_VER && !defined __clang__
#  define CTT_SETR_EPI16(...) { .m128i_i16 = { __VA_ARGS__ } }
#  define CTT_SETR_EPI32(...) { .m128i_i32 = { __VA_ARGS__ } }
#  define CTT_SETR256_EPI16(...) { .m256i_i16 = { __VA_ARGS__ } }
#else
#  include <array>
#  include <bit>
#  define CTT_SETR_EPI16(...) std::bit_cast<__m128i>(std::array<int16_t, 8>{ __VA_ARGS__ })
#  define CTT_SETR_EPI32(...) std::bit_cast<__m128i>(std::array<int32_t, 4>{ __VA_ARGS__ })
#  define CTT_SETR256_EPI16(...) std::bit_cast<__m256i>(std::array<int16_t, 16>{ __VA_ARGS__ })
#endif
"#,
        LABEL,
    )?;

    // The SIMD tables read these tables in their initializers.
    replace_required(
        &mut text,
        "const int32_t g_alpha[16][8] =",
        "constexpr int32_t g_alpha[16][8] =",
        LABEL,
    )?;
    replace_required(
        &mut text,
        "const int32_t g_alphaRange[16] =",
        "constexpr int32_t g_alphaRange[16] =",
        LABEL,
    )?;

    replace_required(
        &mut text,
        "const __m128i ",
        "constinit const __m128i ",
        LABEL,
    )?;
    replace_required(
        &mut text,
        "const __m256i ",
        "constinit const __m256i ",
        LABEL,
    )?;
    replace_required(&mut text, "_mm_setr_epi16(", "CTT_SETR_EPI16(", LABEL)?;
    replace_required(&mut text, "_mm_setr_epi32(", "CTT_SETR_EPI32(", LABEL)?;
    replace_required(&mut text, "_mm256_setr_epi16(", "CTT_SETR256_EPI16(", LABEL)?;

    write_text(path, &text)
}
