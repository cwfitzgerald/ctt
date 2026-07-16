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

### Added

- Vertical (3:4) cubemap cross layout support with automatic orientation detection. The bottom face (`-Z`) follows the standard convention of being stored rotated 180°.
- `A4R4G4B4_UNORM_PACK16` support for DDS input/output.
- `ctt::encoders::resolve_auto_encoder` returns the concrete encoder `Encoder::Auto` will select for a format.
- Rust, CLI, and C API controls for acknowledging an intentional alpha-channel discard (`allow_discarding_alpha` / `--allow-discarding-alpha`). The C settings struct includes the corresponding field.
- C API: `CTT_PIPELINE_OUTPUT_KIND_INVALID`; `ctt_pipeline_output_get_kind(NULL)` now returns it instead of masquerading as `CTT_PIPELINE_OUTPUT_KIND_ENCODED`.
- C API: new example programs covering a real BC7 encode, error paths, and zero-initialized settings; all are compiled and run in CI.
- CI now builds a matrix of encoder feature combinations, catching feature/link breakage.
- README documents 2D-array assembly, cubemap arrays, 3D/volume passthrough, `--zlib` supercompression, and the per-encoder `--<encoder>-opts` / `--help-encoder` flags.
- Feature-gated intra-image compression parallelism. The default-off Rust `rayon` feature uses the active Rayon pool; the CLI adds `--threads`, and the C API adds the process-wide `ctt_set_thread_count` configuration.
- Conversion support for the packed 32-bit formats `E5B9G9R9_UFLOAT`, `B10G11R11_UFLOAT`, and `A2B10G10R10`/`A2R10G10B10` (unorm, snorm, uint, sint), with scalar, SSE4.1, AVX2, AVX-512, and NEON load/store kernels.
- C API: `ctt_set_log_callback` and `ctt_set_log_level` deliver the library's log output to a caller-supplied callback.

### Changed

- **BREAKING (C ABI):** enum discriminants renumbered so a zero-initialized `ctt_convert_settings` behaves identically to `ctt_convert_settings_default()`. `CTT_QUALITY_BASIC`, `CTT_MIPMAP_FILTER_TRIANGLE`, and the `CTT_CONTAINER_KTX2` tag are now `0`; other `ctt_quality`/`ctt_mipmap_filter`/`ctt_container` values shifted. Recompile against the new header.
- **CLI:** `--zstd`/`--zlib` now require `=` for an explicit level (`--zstd=5`); a bare `--zstd` no longer consumes the following input path. Levels are range-validated.
- astcenc quality tiers are now all distinct and monotonic (`fast` is no longer aliased to `basic`); `basic` still maps to astcenc's medium preset, so default performance is unchanged.
- astcenc and Compressonator encoders now honor `Surface::stride` (padded rows are repacked), matching the other backends.
- Converting to an alpha-less format preserves straight-alpha RGB values by default; requesting premultiplied output still bakes alpha into RGB, and output alpha metadata now matches the requested mode.
- `ctt-intel-texture-compressor` and `ctt-bc7enc-rdo` emit a clear `compile_error!` when built without an ISPC backend (`prebuilt` or `build-from-source`) instead of failing at link time.
- The README library example and the crate-level quick start are now compile-checked doctests in CI.
- The in-place sRGB OETF/EOTF passes now dispatch to SSE4.1/AVX2/AVX-512/NEON kernels.
- Third-party C/C++ sources are vendored from pinned upstream commits recorded in `vendor.lock`; `cargo xtask vendor` re-vendors or updates them.

### Removed

- Removed an unused `criterion` entry from `ctt`'s `[dependencies]` (it remains a dev-dependency), slimming downstream dependency trees.
- The published prebuilt crates no longer package `*.sigstore.jsonl` attestation bundles.

### Fixed

- **CLI:** per-encoder option flags (`--bc7e-opts`, etc.) were silently ignored when a bare format name (e.g. `-f bc7`) auto-selected the encoder; they now apply to the encoder Auto resolves to.
- f32-pipeline conversions no longer silently drop existing mip levels; mipmap generation now preserves supplied levels and generates only the missing tail.
- Compressed inputs with a swizzle or mipmap request now error clearly instead of silently ignoring the request.
- Fixed R/B channel order for 16-bit packed DDS formats (`B5G6R5`, `B5G5R5A1`, `B4G4R4A4`) and their legacy D3D9 equivalents, in both read and write paths.
- DDS cubemaps flagged only via the DX10 header (no legacy caps2 bit) are now classified as cubemaps instead of 2D arrays.
- `split_cubemap` validates its input and returns errors instead of panicking on short data or mis-divisible cross/strip dimensions.
- `split_cubemap` accepts validated block-compressed separate faces while retaining the uncompressed requirement for cross/strip extraction.
- Zero-width/height images are rejected up front instead of panicking downstream.
- f16 surfaces with odd row strides load without panicking.
- Mipmap count is clamped to the real chain length (no panic on huge counts, no duplicate 1×1 levels); mip-chain math uses integer `ilog2`.
- Fixed a u32 overflow in ASTC output-size allocation for very large surfaces; hardened dimension/stride math across the pipeline against integer overflow.
- 3D image validation no longer shift-panics on unusually long mip lists, and ASTC output-size/allocation failures return errors on all pointer widths.
- Fixed a bc7enc BC7 bit-packing bug: an anchor index bit-width reduction (`n--`) was silently dropped because the macOS/arm64 build of ISPC miscompiles `--` on unsigned integers (the x86-64 build is unaffected; our macOS prebuilts are cross-built with the arm64 compiler), so every block was packed one bit too long. This caused an out-of-bounds write past each 16-byte block (memory corruption) and non-spec-compliant output (mis-sized anchor index). Unsigned decrements are now written `-= 1` (correct on all hosts), and the packer accumulates into two `uint64` words rather than scattering bytes.
- C API: fixed a memory leak in `ctt_cubemap_input_separate_faces` when aborting on a NULL face; all six faces are now consumed as documented.
- C API: Rust panics in decode/convert/encode entry points are contained at the FFI boundary and reported as `CTT_STATUS_INTERNAL` instead of aborting the host process.
- C API: documented the enum/bool validity contract (out-of-range values are undefined behavior) and the zero-initialization guarantee in the generated header.
- **CLI:** I/O errors now include the file path; the CLI refuses to overwrite an input file with the output; `--cubemap-layout` errors when combined with multiple inputs instead of being silently ignored.

### Security

- KTX2: a supercompressed file declaring a huge uncompressed size no longer triggers an unbounded allocation before validation (decompression bomb).
- KTX2 decoding now enforces decoded-byte and surface-count resource limits, including for otherwise self-consistent compressed headers.
- DDS/KTX2: malicious headers with absurd mip/level counts or overflowing dimensions now error cleanly instead of panicking.

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
