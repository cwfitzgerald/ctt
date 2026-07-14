use std::path::Path;

use anyhow::Result;

use super::{
    clean_and_create, copy_text_file, read_text, replace_required, require_dir, write_text,
};
use crate::util::workspace_root;

/// Automatically vendored from
/// <https://github.com/GameTechDev/ISPCTextureCompressor>.
/// Regenerate with: `cargo xtask vendor intel [--src <path>]`
///
/// `src_dir` is the repository root; the ISPC sources live under
/// `ispc_texcomp/`.
pub fn vendor_intel(src_dir: &Path) -> Result<()> {
    let ws = workspace_root();
    let source = src_dir.join("ispc_texcomp");
    require_dir(&source)?;

    let crate_dir = ws.join("crates/ctt-intel-texture-compressor");
    let dst_dir = crate_dir.join("ispc");
    clean_and_create(&dst_dir)?;

    let kernel = dst_dir.join("kernel.ispc");
    copy_text_file(&source.join("kernel.ispc"), &kernel)?;
    patch_treshold_typo(&kernel)?;

    let header = dst_dir.join("ispc_texcomp.h");
    copy_text_file(&source.join("ispc_texcomp.h"), &header)?;
    patch_treshold_typo(&header)?;

    // License
    copy_text_file(
        &src_dir.join("license.txt"),
        &crate_dir.join("LICENSE-MIT-INTEL"),
    )?;

    Ok(())
}

/// Fix the upstream `fastSkipTreshold` misspelling. The Rust bindings and
/// wrappers use the corrected `fastSkipThreshold`, and the binding generator
/// reads the field names straight out of `kernel.ispc`.
fn patch_treshold_typo(path: &Path) -> Result<()> {
    let mut text = read_text(path)?;
    replace_required(
        &mut text,
        "fastSkipTreshold",
        "fastSkipThreshold",
        "intel: fastSkipTreshold typo",
    )?;
    write_text(path, &text)
}
