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
- [v0.3.0](#v030)
- [v0.2.0](#v020)

## Unreleased

## v0.3.0

Released 2026-04-20

### Added

- New encoder backend: [ARM `astcenc`](https://github.com/ARM-software/astc-encoder) for ASTC (`ctt-astcenc` crate, `encoder-astcenc` feature).
- New encoder backend: [AMD Compressonator](https://github.com/GPUOpen-Tools/compressonator) for BC1–BC7 (`ctt-compressonator` crate, `encoder-amd` feature).
- New encoder backend: [etcpak](https://github.com/wolfpld/etcpak) for ETC/EAC and a subset of BCn (`ctt-etcpak` crate, `encoder-etcpak` feature).
- High-level library API: `ctt::convert` with `ConvertSettings`, plus crate-level documentation and examples covering the end-to-end pipeline.
- KTX and DDS input support: existing compressed textures can now be re-encoded or transcoded, not just raw PNG/JPEG/etc.
- KTX2 supercompression: optional zstd and zlib compression of the payload (`--zstd` / `--zlib`).
- Color space and alpha mode conversion: sRGB↔linear and straight↔premultiplied conversions are modeled in the pipeline and triggered automatically when source and target disagree.
- Mipmap generation.
- NPOT (non-power-of-two) textures are now handled correctly throughout the pipeline, including mipmap generation.
- Profiling scopes via the `profiling` crate across the pipeline.

### Changed

- Core transformation pipeline rewritten around a conversion graph and typed transform nodes, unifying how format conversion, color space conversion, alpha handling, swizzling, mipmapping, and compression are scheduled.
- Mipmap downsampling now uses [`fast_image_resize`](https://crates.io/crates/fast_image_resize); the default filter is box.
- sRGB encode/decode uses an approximate OETF/EOTF on the hot path, significantly speeding up loading and conversion of sRGB images.
- Output handling reworked: clearer container selection from extension, better diagnostics, and cleaner layering between the container writers and the pipeline.

### Removed

- Dropped the Intel ISPC Texture Compressor's built-in ASTC path in favor of `astcenc`, which produces higher-quality output.

## v0.2.0

Released 2026-03-31

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

- [Unreleased](https://github.com/cwfitzgerald/ctt/compare/v0.3.0...HEAD)
- [v0.3.0](https://github.com/cwfitzgerald/ctt/compare/v0.2.0...v0.3.0)
- [v0.2.0](https://github.com/cwfitzgerald/ctt/compare/v0.1.0...v0.2.0)
