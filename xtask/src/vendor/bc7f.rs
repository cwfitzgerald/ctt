use std::path::Path;

use anyhow::{Context, Result};

use super::{
    clean_and_create, copy_files, copy_text_file, read_text, replace_required, require_dir,
    write_text,
};
use crate::util::workspace_root;

/// Copies the Basis Universal transcoder sources needed by BC7F.
pub fn vendor_bc7f(src_dir: &Path) -> Result<()> {
    let source = src_dir.join("transcoder");
    require_dir(&source)?;
    let crate_dir = workspace_root().join("crates/ctt-bc7f");
    let destination = crate_dir.join("cpp");
    clean_and_create(&destination)?;
    copy_files(
        &source,
        &destination,
        &[
            "basisu.h",
            "basisu_astc_cfgs.inl",
            "basisu_astc_hdr_core.h",
            "basisu_astc_helpers.h",
            "basisu_containers.h",
            "basisu_containers_impl.h",
            "basisu_etc1_mods.inl",
            "basisu_file_headers.h",
            "basisu_idct.h",
            "basisu_transcoder.cpp",
            "basisu_transcoder.h",
            "basisu_transcoder_uastc.h",
        ],
    )?;
    let transcoder_path = destination.join("basisu_transcoder.cpp");
    let mut transcoder = read_text(&transcoder_path)?;
    for name in [
        "basisu_dds_transcoder.inl",
        "basisu_xbc7_decoder.h",
        "basisu_xbc7_decoder.inl",
    ] {
        replace_required(&mut transcoder, &format!("#include \"{name}\""), "", name)?;
    }
    replace_required(
        &mut transcoder,
        "assert(src_tex_format == basis_tex_format::cXUBC7);\n\t\n#if BASISD_SUPPORT_XUASTC",
        "assert(src_tex_format == basis_tex_format::cXUBC7);\n\t\n#if 0 // BC7F does not use XUBC7 decoding.",
        "disable XUBC7 slice decoder",
    )?;
    write_text(&transcoder_path, &transcoder)?;

    // The upstream header contains a Windows-1252 dash in a comment.
    let mut header = std::fs::read(source.join("basisu_transcoder_internal.h"))?;
    let offset = header
        .windows(8)
        .position(|bytes| bytes == b"0.7\x960.85")
        .context("BC7F header comment patch: expected Windows-1252 dash not found")?;
    header[offset + 3] = b'-';
    write_text(
        &destination.join("basisu_transcoder_internal.h"),
        &String::from_utf8(header)?,
    )?;
    copy_text_file(
        &src_dir.join("LICENSE"),
        &crate_dir.join("LICENSE-Apache-2.0-BASISU.md"),
    )?;
    Ok(())
}
