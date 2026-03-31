use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, bail};

use crate::util::workspace_root;

pub fn verify_binaries() -> Result<()> {
    let workspace_root = workspace_root();
    let prebuilt_dirs = [
        workspace_root.join("crates/ctt-intel-texture-compressor/prebuilt/bins"),
        workspace_root.join("crates/ctt-bc7enc-rdo/prebuilt/bins"),
    ];

    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut missing_attestation = 0u32;

    for bins_dir in &prebuilt_dirs {
        if !bins_dir.exists() {
            bail!("{} does not exist", bins_dir.display());
        }

        for_each_binary(bins_dir, &mut |binary| {
            let mut attestation_path = binary.as_os_str().to_owned();
            attestation_path.push(".sigstore.jsonl");
            let attestation_path = PathBuf::from(attestation_path);
            if !attestation_path.exists() {
                eprintln!("FAIL: no attestation for {}", binary.display());
                missing_attestation += 1;
                return;
            }

            match verify_attestation(binary, &attestation_path) {
                Ok(()) => {
                    println!("PASS: {}", binary.display());
                    passed += 1;
                }
                Err(e) => {
                    eprintln!("FAIL: {} — {e}", binary.display());
                    failed += 1;
                }
            }
        });
    }

    if passed == 0 {
        bail!("no binaries found to verify");
    }

    println!();
    println!(
        "Results: {passed} passed, {failed} failed, {missing_attestation} missing attestation"
    );

    if failed > 0 || missing_attestation > 0 {
        bail!(
            "{} failed, {} missing attestation",
            failed,
            missing_attestation
        );
    }

    Ok(())
}

fn verify_attestation(binary: &Path, attestation: &Path) -> Result<(), String> {
    let output = Command::new("gh")
        .arg("attestation")
        .arg("verify")
        .arg(binary)
        .arg("--bundle")
        .arg(attestation)
        .arg("--repo")
        .arg("cwfitzgerald/ctt")
        .arg("--signer-workflow")
        .arg("cwfitzgerald/ctt/.github/workflows/build-ispc.yml")
        .output()
        .map_err(|e| format!("failed to run gh: {e}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(stderr.trim().to_string())
    }
}

fn for_each_binary(dir: &Path, callback: &mut dyn FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            for_each_binary(&path, callback);
        } else if is_binary(&path) {
            callback(&path);
        }
    }
}

fn is_binary(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    ext == "a" || ext == "lib"
}
