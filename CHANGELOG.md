# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to cargo's version of [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Per Keep a Changelog there are 6 main categories of changes:
- Added
- Changed
- Deprecated
- Removed
- Fixed
- Security

#### Table of Contents

- [Unreleased](#unreleased)

## Unreleased

### Added

- Ship prebuilt ISPC static libraries for all supported platforms (linux, macOS, Windows; x86_64 and aarch64). Builds now complete without ISPC or libclang installed. The previous `ispc_compile`/`ispc_rt` dependencies have been replaced by a lightweight `ispc-build-utils` crate and per-backend prebuilt crates.
- Every prebuilt binary is attested with [GitHub Artifact Attestations](https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds) and ships with an inline Sigstore bundle. Run `cargo xtask verify-binaries` to verify provenance locally.
- New `build-from-source` feature flag compiles the ISPC kernels from source (requires ISPC on `PATH`). The `prebuilt` and `build-from-source` features are mutually exclusive.
- `xtask` crate with `build-ispc` and `verify-binaries` commands.
- CI now tests both feature modes and verifies attestations on every push.

### Changed

- Default features now use `prebuilt` instead of compiling ISPC from source. Users no longer need ISPC, libclang, or a C++ compiler for a default build.
- GitHub Actions pinned to commit SHAs.

## Diffs
