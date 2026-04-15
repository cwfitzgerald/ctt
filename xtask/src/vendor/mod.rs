mod astcenc;
mod bc7enc_rdo;
mod compressonator;
mod etcpak;

use std::path::Path;
use std::process::Command;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

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
        src: Option<std::path::PathBuf>,
    },
    /// Vendor astc-encoder C++ source.
    Astcenc {
        /// Path to the astc-encoder Source directory.
        #[arg(long)]
        src: Option<std::path::PathBuf>,
    },
    /// Vendor compressonator CMP_Core source.
    Compressonator {
        /// Path to the compressonator source checkout.
        #[arg(long)]
        src: Option<std::path::PathBuf>,
    },
    /// Vendor etcpak C++ source.
    Etcpak {
        /// Path to the etcpak source checkout.
        #[arg(long)]
        src: Option<std::path::PathBuf>,
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
                VendorTarget::Etcpak { src: None },
            ]
        }
    };

    for target in targets {
        match target {
            VendorTarget::Bc7encRdo { src } => bc7enc_rdo::vendor_bc7enc_rdo(src)?,
            VendorTarget::Astcenc { src } => astcenc::vendor_astcenc(src)?,
            VendorTarget::Compressonator { src } => {
                compressonator::vendor_compressonator(src)?;
            }
            VendorTarget::Etcpak { src } => etcpak::vendor_etcpak(src)?,
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

pub(super) fn require_dir(path: &Path) -> Result<()> {
    if !path.is_dir() {
        bail!("source directory not found: {}", path.display());
    }
    Ok(())
}

pub(super) fn clean_and_create(dir: &Path) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    Ok(())
}

pub(super) fn copy_file(src: &Path, dst: &Path) -> Result<()> {
    std::fs::copy(src, dst)?;
    Ok(())
}

pub(super) fn copy_files(src_dir: &Path, dst_dir: &Path, files: &[&str]) -> Result<()> {
    for f in files {
        copy_file(&src_dir.join(f), &dst_dir.join(f))?;
    }
    Ok(())
}

pub(super) fn read_text(path: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

pub(super) fn write_text(path: &Path, content: &str) -> Result<()> {
    // Normalize to LF
    let content = content.replace("\r\n", "\n");
    Ok(std::fs::write(path, content)?)
}

/// Replace all occurrences of `from` with `to` in `text`, requiring at least
/// one match. Returns an error naming `label` if the pattern is not found.
pub(super) fn replace_required(text: &mut String, from: &str, to: &str, label: &str) -> Result<()> {
    if !text.contains(from) {
        bail!("patch {label:?}: pattern not found in source");
    }
    *text = text.replace(from, to);
    Ok(())
}

/// Get the current git HEAD revision of a repository at `repo_dir`.
pub(super) fn git_revision(repo_dir: &Path) -> Result<String> {
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
