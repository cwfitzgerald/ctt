# Release Process

This document describes how to publish a new release of `ctt`.

## Prerequisites

- Push access to the default branch
- A crates.io API token with publish rights for all workspace crates
- `gh` CLI installed and authenticated (for creating the GitHub release)

## Steps

### 1. Determine the new version

Pick the new version number following cargo semver conventions. For this document,
we'll use `X.Y.Z` as a placeholder.

### 2. Update CHANGELOG.md

**a) Add the new version to the Table of Contents:**

Find the line:
```
- [Unreleased](#unreleased)
```
Add a new entry directly below it:
```
- [vX.Y.Z](#vXYZ)
```
(The anchor is the version with dots removed, e.g. `v0.2.0` -> `#v020`)

**b) Add a version heading under Unreleased:**

Find:
```
## Unreleased
```
Add a blank line and a new version section below it, moving all existing unreleased
items under the new heading:
```
## Unreleased

## vX.Y.Z

Released YYYY-MM-DD

- (move all previously unreleased items here)
```

**c) Update the Diffs section at the bottom:**

Find the existing unreleased diff link:
```
- [Unreleased](https://github.com/cwfitzgerald/ctt/compare/vPREVIOUS...HEAD)
```
Update it and add a new entry:
```
- [Unreleased](https://github.com/cwfitzgerald/ctt/compare/vX.Y.Z...HEAD)
- [vX.Y.Z](https://github.com/cwfitzgerald/ctt/compare/vPREVIOUS...vX.Y.Z)
```

### 3. Update Cargo.toml

Set the `version` field in `[workspace.package]` to the new version:
```toml
version = "X.Y.Z"
```

Update any intra-workspace dependency versions as needed (e.g. `ctt`, `ctt-intel-texture-compressor`, `ctt-bc7enc-rdo` entries in `[workspace.dependencies]`).

### 4. Update README.md

Update any version references (dependency snippets, compatibility tables, etc.)
to reflect the new version.

### 5. Commit and tag

```bash
jj commit -m "Release vX.Y.Z"
jj tag create vX.Y.Z
jj git push
```

Pushing the `vX.Y.Z` tag triggers `.github/workflows/publish.yml`. That
workflow runs CI, builds the release binaries (CLI + C API) for every target
with attestation, and creates the **GitHub release** with those artifacts
attached (`generate_release_notes: true`).

> **Important:** `publish.yml` does **not** publish to crates.io. Pushing the
> tag only produces the GitHub release and binaries. Publishing the crates is
> the separate manual step below.

### 6. Publish to crates.io

This is a virtual workspace, so a bare `cargo publish` does not work — each
crate must be published individually with `-p`, and they must go out in
dependency order so every crate's dependencies already exist on crates.io when
its verification build runs. Cargo waits for each freshly published crate to
become available in the index, so the commands can be run back-to-back.

Publish in this order:

```bash
# 1. Leaf build utility (build-dependency of the prebuilt crates)
cargo publish -p ispc-build-utils

# 2. Prebuilt ISPC static-library crates
cargo publish -p ctt-intel-texture-compressor-prebuilt
cargo publish -p ctt-bc7enc-rdo-prebuilt

# 3. Encoder binding crates
cargo publish -p ctt-intel-texture-compressor
cargo publish -p ctt-bc7enc-rdo
cargo publish -p ctt-astcenc
cargo publish -p ctt-compressonator
cargo publish -p ctt-etcpak

# 4. Core library
cargo publish -p ctt

# 5. Front-end crates
cargo publish -p ctt-cli
cargo publish -p ctt-c-api
```

The default (`prebuilt`) verification build links the shipped static libraries,
so `ispc` does not need to be on `PATH` to publish.

### 7. Post-release

Verify:
- [ ] The crates are visible at https://crates.io/crates/ctt/X.Y.Z
- [ ] Docs are building at https://docs.rs/ctt/X.Y.Z
- [ ] The GitHub release created by `publish.yml` exists at
      https://github.com/cwfitzgerald/ctt/releases/tag/vX.Y.Z with the CLI and
      C API binaries attached. Edit its notes from `CHANGELOG.md` if the
      auto-generated notes need refining.
