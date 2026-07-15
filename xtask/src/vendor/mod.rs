mod astcenc;
mod bc7enc_rdo;
mod compressonator;
mod etcpak;
mod git;
mod intel;
mod lock;

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};

use lock::Lock;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
pub struct VendorArgs {
    /// Target to vendor. Omit to vendor every target.
    #[arg(value_enum)]
    target: Option<VendorTarget>,

    /// Vendor from this local checkout instead of pulling from git.
    #[arg(long, requires = "target", conflicts_with = "update")]
    src: Option<PathBuf>,

    /// Re-resolve each target's tracked ref (branch/tag) to its newest commit,
    /// vendor that, and record it in `vendor.lock`. Without this, the exact
    /// commit already pinned in the lock is vendored.
    #[arg(long)]
    update: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum VendorTarget {
    /// Vendor bc7enc_rdo ISPC source.
    Bc7encRdo,
    /// Vendor astc-encoder C++ source.
    Astcenc,
    /// Vendor compressonator CMP_Core source.
    Compressonator,
    /// Vendor etcpak C++ source.
    Etcpak,
    /// Vendor Intel ISPC texture compressor source.
    Intel,
}

impl VendorTarget {
    const ALL: [VendorTarget; 5] = [
        VendorTarget::Bc7encRdo,
        VendorTarget::Astcenc,
        VendorTarget::Compressonator,
        VendorTarget::Etcpak,
        VendorTarget::Intel,
    ];

    /// Canonical lock-file key for this target.
    fn name(&self) -> &'static str {
        match self {
            VendorTarget::Bc7encRdo => "bc7enc-rdo",
            VendorTarget::Astcenc => "astcenc",
            VendorTarget::Compressonator => "compressonator",
            VendorTarget::Etcpak => "etcpak",
            VendorTarget::Intel => "intel-texture-compressor",
        }
    }

    /// Copy + patch the acquired source tree at `src_dir` into the crate.
    fn vendor_from(&self, src_dir: &Path) -> Result<()> {
        match self {
            VendorTarget::Bc7encRdo => bc7enc_rdo::vendor_bc7enc_rdo(src_dir),
            VendorTarget::Astcenc => astcenc::vendor_astcenc(src_dir),
            VendorTarget::Compressonator => compressonator::vendor_compressonator(src_dir),
            VendorTarget::Etcpak => etcpak::vendor_etcpak(src_dir),
            VendorTarget::Intel => intel::vendor_intel(src_dir),
        }
    }
}

pub fn vendor(args: VendorArgs) -> Result<()> {
    let ws = crate::util::workspace_root();
    let mut lock = Lock::load(&ws)?;

    if let Some(target) = args.target {
        run_target(&ws, &mut lock, target, args.src.as_deref(), args.update)?;
    } else {
        for target in VendorTarget::ALL {
            run_target(&ws, &mut lock, target, None, args.update)?;
        }
    }

    Ok(())
}

fn run_target(
    ws: &Path,
    lock: &mut Lock,
    target: VendorTarget,
    src: Option<&Path>,
    update: bool,
) -> Result<()> {
    let name = target.name();
    let entry = lock.entry(name)?;
    let acquired = git::acquire(ws, name, &entry, src, update)?;

    target.vendor_from(&acquired.dir)?;

    if update {
        let Some(commit) = acquired.commit.as_deref() else {
            bail!("--update cannot be combined with --src");
        };
        lock.set_commit(name, commit)?;
    }

    match acquired.commit {
        Some(commit) => println!("Vendored {name} at {commit} ({})", acquired.dir.display()),
        None => println!("Vendored {name} from {}", acquired.dir.display()),
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

pub(super) fn copy_text_file(src: &Path, dst: &Path) -> Result<()> {
    let text = read_text(src)?;
    write_text(dst, &text)
}

pub(super) fn copy_files(src_dir: &Path, dst_dir: &Path, files: &[&str]) -> Result<()> {
    for f in files {
        copy_text_file(&src_dir.join(f), &dst_dir.join(f))?;
    }
    Ok(())
}

pub(super) fn read_text(path: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

pub(super) fn write_text(path: &Path, content: &str) -> Result<()> {
    // Normalize to LF
    let content = content.replace("\r\n", "\n").replace('\r', "\n");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_source_requires_a_target_and_conflicts_with_update() {
        assert!(VendorArgs::try_parse_from(["vendor", "--src", "checkout"]).is_err());
        assert!(
            VendorArgs::try_parse_from(["vendor", "astcenc", "--src", "checkout", "--update"])
                .is_err()
        );

        let args = VendorArgs::try_parse_from(["vendor", "astcenc", "--src", "checkout"]).unwrap();
        assert_eq!(args.target, Some(VendorTarget::Astcenc));
        assert_eq!(args.src, Some(PathBuf::from("checkout")));
        assert!(!args.update);
    }

    #[test]
    fn text_copy_normalizes_line_endings() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let src = dir.path().join("src.txt");
        let dst = dir.path().join("dst.txt");
        std::fs::write(&src, "crlf\r\ncr\rlf\n")?;

        copy_text_file(&src, &dst)?;

        assert_eq!(std::fs::read_to_string(dst)?, "crlf\ncr\nlf\n");
        Ok(())
    }
}
