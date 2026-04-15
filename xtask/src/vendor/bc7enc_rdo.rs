use std::path::PathBuf;

use anyhow::Result;

use super::{clean_and_create, copy_file, git_revision, require_dir};
use crate::util::workspace_root;

/// Automatically vendored from <https://github.com/richgel999/bc7enc_rdo>.
/// Default source: `../bc7enc_rdo` relative to the workspace root.
/// Regenerate with: `cargo xtask vendor bc7enc-rdo [--src <path>]`
pub fn vendor_bc7enc_rdo(src: Option<PathBuf>) -> Result<()> {
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
