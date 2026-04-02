# Prebuilt ISPC Binaries

ctt ships prebuilt static libraries for the ISPC compression kernels so that a default build doesn't require ISPC. This document explains what the binaries are, where they come from, and how to verify them.

## What's included

Two crates contain prebuilt binaries:

| Crate | Libraries | Source |
|-------|-----------|--------|
| `ctt-intel-texture-compressor-prebuilt` | `kernel` | [Intel ISPC Texture Compressor](https://github.com/GameTechDev/ISPCTextureCompressor) |
| `ctt-bc7enc-rdo-prebuilt` | `bc7e` | [bc7enc_rdo](https://github.com/richgel999/bc7enc_rdo) |

Each crate stores its binaries under `bins/<platform>/`:

```
prebuilt/bins/
  linux-x86_64/
  linux-aarch64/
  windows-x86_64/
  windows-aarch64/
  macos-x86_64/
  macos-aarch64/
```

The build script for each prebuilt crate copies the correct platform's `.a` or `.lib` into `OUT_DIR` and emits the `cargo:rustc-link-lib=static=...` directives. Nothing else from the prebuilt crate runs at build time.

## When prebuilt binaries are and aren't used

The `prebuilt` feature is **on by default**. When it's on, the prebuilt crate is pulled in as a dependency and its static libraries are linked directly -- no compilation step.

When you use `build-from-source` instead, the prebuilt crate is **not a dependency at all**. It doesn't appear in the dependency tree and its binaries are never copied or linked. CI enforces this with a `cargo tree` check that fails if `prebuilt` appears in the `build-from-source` feature tree.

The two features are mutually exclusive. Enabling both produces a compile error.

### Opting out

To build from source instead of using the prebuilt binaries:

```sh
# CLI
cargo install ctt-cli --no-default-features --features ispc-build-from-source

# Library
cargo add ctt --no-default-features --features "encoder-intel,encoder-bc7enc,ispc-build-from-source"
```

This requires ISPC and libclang on your `PATH`, plus a C++ compiler.

## How binaries are built

The prebuilt binaries are produced by the [`build-ispc.yml`](./../.github/workflows/build-ispc.yml) GitHub Actions workflow, triggered manually via `workflow_dispatch`. The workflow:

1. Checks out the repository at the triggering commit.
2. Downloads a pinned version of ISPC (see `ISPC_VERSION` in the workflow).
3. Runs `cargo xtask build-ispc` for each platform/target combination to compile the ISPC kernels into static libraries.
4. Uploads the libraries as build artifacts.
5. In a second job (`attest-and-commit`), downloads all artifacts and uses [`actions/attest-build-provenance`](https://github.com/actions/attest-build-provenance) to generate a [SLSA provenance](https://slsa.dev/) attestation for every binary.
6. Downloads the attestation bundles (Sigstore `.jsonl` files) and stores them next to each binary.
7. Commits the binaries and their attestation bundles back to the branch.

All GitHub Actions in the workflow are pinned to specific commit SHAs.

## Attestation guarantees

Every `.a` and `.lib` file has a corresponding `.sigstore.jsonl` file sitting next to it. This is a [Sigstore bundle](https://docs.sigstore.dev/about/overview/) containing a signed SLSA provenance statement. The attestation cryptographically binds:

- **The binary's content** (via SHA-256 digest).
- **The source repository** (`cwfitzgerald/ctt`).
- **The workflow that produced it** (`.github/workflows/build-ispc.yml`).
- **The commit** the workflow ran against.

The signing key is an ephemeral key issued by GitHub's OIDC identity provider and recorded in the [Sigstore transparency log](https://docs.sigstore.dev/logging/overview/). There is no long-lived signing key to compromise -- the trust root is GitHub Actions' workload identity.

In practical terms this means: if the attestation verifies, the binary was produced by GitHub Actions running the `build-ispc.yml` workflow in the `cwfitzgerald/ctt` repository. It was not built locally, and it was not modified after the workflow produced it.

## Verifying attestations locally

### Using `cargo xtask verify-binaries`

The simplest way to verify all prebuilt binaries at once:

```sh
cargo xtask verify-binaries
```

This requires the [`gh` CLI](https://cli.github.com/) to be installed and authenticated (`gh auth login`).

The command walks both prebuilt directories, and for each `.a`/`.lib` file:

1. Checks that a `.sigstore.jsonl` attestation bundle exists next to it.
2. Runs `gh attestation verify` with:
   - `--bundle <path>.sigstore.jsonl` -- the attestation bundle to check.
   - `--repo cwfitzgerald/ctt` -- the expected source repository.
   - `--signer-workflow cwfitzgerald/ctt/.github/workflows/build-ispc.yml` -- the expected workflow that produced the binary.

The output shows `PASS` or `FAIL` for each binary and a summary at the end. Any failure (missing attestation, signature mismatch, wrong signer) causes a non-zero exit.

### Verifying a single binary manually

You can verify individual files directly with `gh`:

```sh
gh attestation verify crates/ctt-intel-texture-compressor/prebuilt/bins/linux-x86_64/libkernel.a \
  --bundle crates/ctt-intel-texture-compressor/prebuilt/bins/linux-x86_64/libkernel.a.sigstore.jsonl \
  --repo cwfitzgerald/ctt \
  --signer-workflow cwfitzgerald/ctt/.github/workflows/build-ispc.yml
```

A successful verification prints the attestation details including the commit SHA, workflow ref, and digest. A failure prints the reason (signature invalid, wrong signer, digest mismatch, etc.).

## CI verification

The CI pipeline (`.github/workflows/ci.yml`) includes a `verify-attestations` job that runs `cargo xtask verify-binaries` on every push. This ensures that binaries checked into the repository always have valid, up-to-date attestations.
