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
- [v0.4.0](#v040)
- [v0.3.0](#v030)
- [v0.2.0](#v020)

## Unreleased

## v0.4.0

Released 2026-05-16

### Added

- C API: new `ctt-c-api` crate produces static/dynamic libraries plus a hand-curated `ctt.h` header. Signed, attested prebuilt archives are published for Windows, macOS, and Linux on x86_64 and aarch64.
- 3D (volume) texture support end-to-end: load/encode/store 3D textures via DDS and KTX2. New CLI `--volume` flag stacks input files as Z slices.
- AVX-512 sRGB load/store kernels (x86_64) and NEON sRGB load/store kernels (aarch64), selected at runtime.
- Bulk f16 ↔ f32 processing kernels.
- Per-encoder option flags: `--astcenc-opts`, `--bc7e-opts`, `--intel-opts`, `--etcpak-opts`, `--amd-opts` accept `key=val[;key=val...]` strings. `--help-encoder <name>` lists available keys with types and docs.
- Substantially expanded `astcenc` settings: usage modes (`Color`, `NormalMap` with `Bc5Compat`/`AstcDefault` swizzle, `SingleChannel`, `TwoChannel`, `HdrRgb`, `HdrRgba`, `Rgbm`), perceptual mode, alpha-weighted error, `decode_unorm8` tuning, RGBM scale, custom preset override.
- CLI integration test suite covering color/alpha conversion, containers, cubemaps, cubemap arrays, mipmaps, volumes, swizzle, supercompression, passthrough, and error paths.

### Changed

- **API:** `TargetFormat::Compressed` now carries a typed `encoder: Encoder` enum (with per-backend settings) instead of `encoder_name: Option<String>`. Use `Encoder::Auto` for "any encoder that supports this format".
- **API:** `parse_format` signature is now `parse_format(&str) -> Result<TargetFormat>` — the `EncoderRegistry` parameter is gone; compiled-in encoders are discovered statically.
- **API:** `Image::is_cubemap: bool` replaced with `Image::kind: TextureKind` (`Texture2D` / `Cubemap` / `Texture3D`).
- **API:** `Surface` gains `depth` and `slice_stride` fields for 3D surfaces (set `depth = 1`, `slice_stride = 0` for 2D).
- CLI `--help` output wraps to the terminal width.
- ASTC encoder defaults tuned for better quality out of the box.
- Passthrough path widened — more input/output combinations skip re-encoding.

### Fixed

- `ctt-compressonator` now builds on older GCC versions.

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

- [Unreleased](https://github.com/cwfitzgerald/ctt/compare/v0.4.0...HEAD)
- [v0.4.0](https://github.com/cwfitzgerald/ctt/compare/v0.3.0...v0.4.0)
- [v0.3.0](https://github.com/cwfitzgerald/ctt/compare/v0.2.0...v0.3.0)
- [v0.2.0](https://github.com/cwfitzgerald/ctt/compare/v0.1.0...v0.2.0)
