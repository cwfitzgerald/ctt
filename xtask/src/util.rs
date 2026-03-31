use std::path::PathBuf;

pub fn workspace_root() -> PathBuf {
    let xtask_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    xtask_dir
        .parent()
        .expect("xtask must be in a subdirectory of the workspace root")
        .to_path_buf()
}
