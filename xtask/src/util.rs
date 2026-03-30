use std::path::PathBuf;

pub fn workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().expect("no current dir");
    loop {
        if dir.join("Cargo.toml").exists() {
            let contents = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default();
            if contents.contains("[workspace]") {
                return dir;
            }
        }
        if !dir.pop() {
            panic!("could not find workspace root");
        }
    }
}
