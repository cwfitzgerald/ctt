use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use toml_edit::DocumentMut;

/// Name of the centralized lock file, relative to the workspace root.
pub const LOCK_FILE: &str = "vendor.lock";

/// The centralized vendoring lock file.
///
/// Records, per target, the upstream git remote, the branch/tag tracked by
/// `--update`, and the exact commit currently vendored. Backed by a
/// [`DocumentMut`] so writing back the resolved commit preserves the file's
/// comments and formatting.
pub struct Lock {
    path: PathBuf,
    doc: DocumentMut,
}

/// One target's recorded upstream coordinates.
pub struct LockEntry {
    /// Upstream git remote URL.
    pub repo: String,
    /// Branch or tag tracked by `cargo xtask vendor --update`.
    pub git_ref: String,
    /// Exact commit currently vendored, or `None` if not yet resolved.
    pub commit: Option<String>,
}

impl Lock {
    /// Load `vendor.lock` from the workspace root.
    pub fn load(ws: &Path) -> Result<Lock> {
        let path = ws.join(LOCK_FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading lock file {}", path.display()))?;
        let doc = text
            .parse::<DocumentMut>()
            .with_context(|| format!("parsing lock file {}", path.display()))?;
        Ok(Lock { path, doc })
    }

    /// Read the recorded entry for `target`.
    pub fn entry(&self, target: &str) -> Result<LockEntry> {
        let table = self
            .doc
            .get(target)
            .and_then(|item| item.as_table())
            .with_context(|| format!("lock file has no [{target}] section"))?;

        let get = |key: &str| -> Result<String> {
            table
                .get(key)
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .with_context(|| format!("[{target}] is missing string key {key:?}"))
        };

        let repo = get("repo")?;
        let git_ref = get("ref")?;
        let commit = table
            .get("commit")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .filter(|s| !s.is_empty());

        Ok(LockEntry {
            repo,
            git_ref,
            commit,
        })
    }

    /// Record the resolved commit for `target`, preserving surrounding
    /// formatting. Writes to disk immediately.
    pub fn set_commit(&mut self, target: &str, commit: &str) -> Result<()> {
        let table = self
            .doc
            .get_mut(target)
            .and_then(|item| item.as_table_mut())
            .with_context(|| format!("lock file has no [{target}] section"))?;
        table["commit"] = toml_edit::value(commit);
        self.save()
    }

    fn save(&self) -> Result<()> {
        std::fs::write(&self.path, self.doc.to_string())
            .with_context(|| format!("writing lock file {}", self.path.display()))
    }
}

/// Resolve the pinned commit for `target`, erroring with actionable guidance
/// when the lock has no commit recorded yet.
pub fn require_commit(entry: &LockEntry, target: &str) -> Result<String> {
    match &entry.commit {
        Some(c) => Ok(c.clone()),
        None => bail!(
            "no commit recorded for {target:?} in {}; run `cargo xtask vendor {target} --update` \
             to resolve and record one",
            LOCK_FILE
        ),
    }
}
