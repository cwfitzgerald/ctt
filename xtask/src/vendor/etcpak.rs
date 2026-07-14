use std::path::Path;

use anyhow::Result;

use super::{clean_and_create, copy_files, copy_text_file, require_dir};
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
            "ColorSpace.cpp",
            "bc7enc.cpp",
        ],
    )?;

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
            "ColorSpace.hpp",
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
