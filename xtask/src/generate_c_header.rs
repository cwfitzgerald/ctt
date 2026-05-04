use std::path::Path;

use anyhow::{Context, Result};

use crate::util::workspace_root;

/// Regenerate `crates/ctt-c-api/include/ctt.h` from the ctt-c-api crate
/// using cbindgen.
///
/// All encoder features are enabled while parsing so the generated header
/// contains every variant; consumers can build the dynamic library with a
/// reduced feature set without producing an incompatible header.
pub fn generate_c_header() -> Result<()> {
    let workspace_root = workspace_root();
    let crate_dir = workspace_root.join("crates/ctt-c-api");
    let out_path = crate_dir.join("include/ctt.h");

    println!("Generating C header for ctt-c-api...");

    let parse_config = cbindgen::ParseConfig {
        parse_deps: false,
        ..Default::default()
    };

    let mut config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .map_err(|e| anyhow::anyhow!("failed to read cbindgen.toml: {e}"))?;
    config.parse = parse_config;

    let bindings = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .context("cbindgen failed to generate bindings")?;

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    bindings.write_to_file(&out_path);

    normalize_line_endings(&out_path);

    println!(
        "  Wrote {}",
        out_path
            .strip_prefix(&workspace_root)
            .unwrap_or(&out_path)
            .display()
    );

    Ok(())
}

fn normalize_line_endings(path: &Path) {
    let content = std::fs::read_to_string(path).expect("failed to read file for normalization");
    if content.contains('\r') {
        let normalized = content.replace("\r\n", "\n");
        std::fs::write(path, normalized).expect("failed to write normalized file");
    }
}
