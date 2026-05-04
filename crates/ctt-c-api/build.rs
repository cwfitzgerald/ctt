// Re-export the cargo-provided TARGET / HOST triples as rustc env vars so
// integration tests can drive the `cc` crate without needing to construct
// the triples themselves from `cfg!`.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").expect("TARGET");
    let host = std::env::var("HOST").expect("HOST");
    println!("cargo:rustc-env=CTT_C_API_TARGET={target}");
    println!("cargo:rustc-env=CTT_C_API_HOST={host}");
}
