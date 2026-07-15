use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use super::lock::{self, LockEntry};

/// A checkout of an upstream source tree, ready to vendor from.
#[derive(Debug)]
pub struct AcquiredSource {
    /// Directory holding the upstream tree root.
    pub dir: PathBuf,
    /// The exact commit the tree is at, unless this is a local override.
    pub commit: Option<String>,
}

/// Acquire the upstream source for `target`.
///
/// With `src`, vendors from a local checkout as-is. Otherwise fetches from the
/// git remote recorded in the lock: `update` re-resolves the tracked ref to its
/// newest commit, while the default checks out the exact commit already pinned.
pub fn acquire(
    ws: &Path,
    target: &str,
    entry: &LockEntry,
    src: Option<&Path>,
    update: bool,
) -> Result<AcquiredSource> {
    if let Some(src) = src {
        return acquire_local(src);
    }
    acquire_git(ws, target, entry, update)
}

fn acquire_local(src: &Path) -> Result<AcquiredSource> {
    if !src.is_dir() {
        bail!("source directory not found: {}", src.display());
    }
    Ok(AcquiredSource {
        dir: src.to_path_buf(),
        commit: None,
    })
}

fn acquire_git(ws: &Path, target: &str, entry: &LockEntry, update: bool) -> Result<AcquiredSource> {
    let cache = ws.join("target/vendor-cache").join(target);
    ensure_repo(&cache, &entry.repo)?;
    ensure_clean(&cache)?;

    let commit = if update {
        fetch(&cache, &entry.git_ref)
            .with_context(|| format!("fetching ref {:?} for {target}", entry.git_ref))?;
        rev_parse(&cache, "FETCH_HEAD")?
    } else {
        let pinned = lock::require_commit(entry, target)?;
        if !commit_exists(&cache, &pinned)? && !try_fetch(&cache, &pinned)? {
            fetch(&cache, &entry.git_ref).with_context(|| {
                format!(
                    "fetching commit {pinned} via ref {:?} for {target}",
                    entry.git_ref
                )
            })?;
        }
        if !commit_exists(&cache, &pinned)? {
            bail!(
                "pinned commit {pinned} for {target} is unavailable after fetching ref {:?}",
                entry.git_ref
            );
        }
        pinned
    };

    checkout(&cache, &commit).with_context(|| format!("checking out {commit} for {target}"))?;
    ensure_clean(&cache)?;

    Ok(AcquiredSource {
        dir: cache,
        commit: Some(commit),
    })
}

/// Ensure `dir` is a git repo with `origin` pointing at `repo`.
fn ensure_repo(dir: &Path, repo: &str) -> Result<()> {
    if dir.join(".git").exists() {
        // Keep the remote in sync in case the lock's URL changed.
        run(dir, &["remote", "set-url", "origin", repo])?;
        return Ok(());
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating cache dir {}", dir.display()))?;
    run(dir, &["init", "-q"])?;
    run(dir, &["remote", "add", "origin", repo])?;
    Ok(())
}

fn fetch(dir: &Path, refspec: &str) -> Result<()> {
    run(dir, &["fetch", "--depth", "1", "origin", refspec])
}

fn try_fetch(dir: &Path, refspec: &str) -> Result<bool> {
    try_run(dir, &["fetch", "--depth", "1", "origin", refspec])
}

fn checkout(dir: &Path, commit: &str) -> Result<()> {
    run(dir, &["checkout", "-q", "--detach", commit])
}

fn rev_parse(dir: &Path, rev: &str) -> Result<String> {
    let out = capture(dir, &["rev-parse", rev])?;
    Ok(out.trim().to_string())
}

fn commit_exists(dir: &Path, commit: &str) -> Result<bool> {
    let commit = format!("{commit}^{{commit}}");
    try_run(dir, &["cat-file", "-e", &commit])
}

fn ensure_clean(dir: &Path) -> Result<()> {
    let status = capture(dir, &["status", "--porcelain", "--untracked-files=all"])?;
    if !status.trim().is_empty() {
        bail!(
            "vendor cache {} has uncommitted changes; remove or clean the cache before vendoring:\n{}",
            dir.display(),
            status.trim()
        );
    }
    Ok(())
}

fn run(dir: &Path, args: &[&str]) -> Result<()> {
    let output = output(dir, args)?;
    if !output.status.success() {
        bail!(
            "git {} failed in {}: {}",
            args.join(" "),
            dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn try_run(dir: &Path, args: &[&str]) -> Result<bool> {
    Ok(output(dir, args)?.status.success())
}

fn capture(dir: &Path, args: &[&str]) -> Result<String> {
    let output = output(dir, args)?;
    if !output.status.success() {
        bail!(
            "git {} failed in {}: {}",
            args.join(" "),
            dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn output(dir: &Path, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .with_context(|| format!("spawning git {}", args.join(" ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn upstream_repo() -> Result<(TempDir, String)> {
        let repo = TempDir::new()?;
        run(repo.path(), &["init", "-q"])?;
        std::fs::write(repo.path().join("source.txt"), "upstream\n")?;
        run(repo.path(), &["add", "source.txt"])?;
        run(
            repo.path(),
            &[
                "-c",
                "user.name=Vendoring Test",
                "-c",
                "user.email=vendoring@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "initial",
            ],
        )?;
        let commit = rev_parse(repo.path(), "HEAD")?;
        Ok((repo, commit))
    }

    fn entry(repo: &Path, commit: &str) -> LockEntry {
        LockEntry {
            repo: repo.display().to_string(),
            git_ref: "HEAD".to_string(),
            commit: Some(commit.to_string()),
        }
    }

    #[test]
    fn local_source_has_no_lock_commit() -> Result<()> {
        let source = TempDir::new()?;
        let acquired = acquire_local(source.path())?;
        assert_eq!(acquired.dir, source.path());
        assert_eq!(acquired.commit, None);
        Ok(())
    }

    #[test]
    fn cached_commit_does_not_require_remote() -> Result<()> {
        let (upstream, commit) = upstream_repo()?;
        let workspace = TempDir::new()?;
        let target = "fixture";

        acquire_git(
            workspace.path(),
            target,
            &entry(upstream.path(), &commit),
            false,
        )?;

        let missing_remote = workspace.path().join("missing-remote");
        let acquired = acquire_git(
            workspace.path(),
            target,
            &entry(&missing_remote, &commit),
            false,
        )?;
        assert_eq!(acquired.commit.as_deref(), Some(commit.as_str()));
        Ok(())
    }

    #[test]
    fn dirty_cache_is_rejected() -> Result<()> {
        let (upstream, commit) = upstream_repo()?;
        let workspace = TempDir::new()?;
        let target = "fixture";
        let entry = entry(upstream.path(), &commit);

        let acquired = acquire_git(workspace.path(), target, &entry, false)?;
        std::fs::write(acquired.dir.join("source.txt"), "modified\n")?;

        let error = acquire_git(workspace.path(), target, &entry, false).unwrap_err();
        assert!(error.to_string().contains("has uncommitted changes"));
        Ok(())
    }
}
