# Release Process

This document describes how to publish a new release of `ctt`.

## Prerequisites

- Push access to the default branch
- A crates.io API token with publish rights for all workspace crates
- Cargo 1.90+ (for workspace publishing)

## Steps

Pick the new version number following cargo semver conventions. For this
document, we'll use `X.Y.Z` as a placeholder.

### 1. Update CHANGELOG.md

Three edits:

- Add `- [vX.Y.Z](#vXYZ)` to the Table of Contents, directly below the
  `Unreleased` entry. (The anchor is the version with dots removed,
  e.g. `v0.2.0` -> `#v020`.)
- Add a `## vX.Y.Z` heading below `## Unreleased`, followed by
  `Released YYYY-MM-DD`, and move all unreleased items under it.
- In the `## Diffs` section at the bottom, point the `Unreleased` link at
  `vX.Y.Z...HEAD` and add a new
  `- [vX.Y.Z](https://github.com/cwfitzgerald/ctt/compare/vPREVIOUS...vX.Y.Z)`
  entry below it.

### 2. Update Cargo.toml

Set `version` in `[workspace.package]` to `X.Y.Z`, and update the `version`
fields of the intra-workspace entries in `[workspace.dependencies]` (the
`ctt*` and `ispc-build-utils` crates) if the major/minor changed.

### 3. Commit, tag, and push

```bash
jj commit -m "Release vX.Y.Z"
jj bookmark move trunk --to @-
jj tag set vX.Y.Z -r @-
jj git push
git push origin vX.Y.Z
```

(`jj git push` pushes the bookmark but not tags, hence the extra `git push`.)

Pushing the `vX.Y.Z` tag triggers `.github/workflows/publish.yml`. That
workflow runs CI, builds the release binaries (CLI + C API) for every target
with attestation, and creates the **GitHub release** with those artifacts
attached (`generate_release_notes: true`).

> **Important:** `publish.yml` does **not** publish to crates.io. Pushing the
> tag only produces the GitHub release and binaries. Publishing the crates is
> the separate manual step below.

### 4. Publish to crates.io

```bash
cargo publish --workspace
```

Cargo publishes every publishable crate in dependency order, waiting for each
one to appear in the index before publishing its dependents. `xtask` and
`regen-test-data` are `publish = false` and are skipped automatically.
(`--workspace` is required because this is a virtual workspace.)

The default (`prebuilt`) verification build links the static libraries
committed under each prebuilt crate's `bins/`, so `ispc` does not need to be
on `PATH` to publish.

### 5. Post-release

Verify:
- [ ] The crates are visible at https://crates.io/crates/ctt/X.Y.Z
- [ ] Docs are building at https://docs.rs/ctt/X.Y.Z
- [ ] The GitHub release created by `publish.yml` exists at
      https://github.com/cwfitzgerald/ctt/releases/tag/vX.Y.Z with the CLI and
      C API binaries attached. Edit its notes from `CHANGELOG.md` if the
      auto-generated notes need refining.
