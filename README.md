# ctt

[![docs.rs](https://img.shields.io/docsrs/ctt)](https://docs.rs/ctt)

A Rust library with [C bindings](#c-api), and CLI for GPU texture compression. ctt provides a unified interface over multiple encoder backends.

## Encoders

ctt binds to established open-source compression libraries and exposes them through a common `Encoder` trait. Each backend is compiled as an optional feature and can be enabled independently.

| Backend | Prefix | Feature | Description |
|---------|--------|---------|-------------|
| [**bc7enc-rdo**](https://github.com/richgel999/bc7enc_rdo) | `bc7e_` | `encoder-bc7enc` | Perceptual BC7 encoder with RDO support. |
| [**Intel ISPC Texture Compressor**](https://github.com/GameTechDev/ISPCTextureCompressor) | `intel_` | `encoder-intel` | SIMD-optimized BCn and ETC encoder. |
| [**etcpak**](https://github.com/wolfpld/etcpak) | `etcpak_` | `encoder-etcpak` | Fast ETC/EAC and BCn encoder. |
| [**AMD Compressonator**](https://github.com/GPUOpen-Tools/compressonator) | `amd_` | `encoder-amd` | AMD's BCn encoder suite. |
| [**astcenc**](https://github.com/ARM-software/astc-encoder) | `astcenc_` | `encoder-astcenc` | ARM ASTC encoder. |

Encoders are listed in priority order — when multiple encoders support the same format (e.g. BC7), the first one in the table is used. To override this, prefix the format with an encoder name (e.g. `intel_bc7`).

Use `ctt --list-encoders` to see what's available in your build:

```
$ ctt --list-encoders
Encoder    Priority     Formats
-------    --------     -------
bc7e       1            bc7
intel      2            bc1, bc3, bc4, bc5, bc6h, bc7, etc1
etcpak     3            etc1, etc2_rgba, eac_r, eac_rg, bc1, bc3, bc4, bc5
amd        4            bc1, bc2, bc3, bc4, bc4s, bc5, bc5s, bc6h, bc6hsf, bc7
astcenc    5            astc
```

## Formats

| Format | CLI name | Description |
|--------|----------|-------------|
| BC1 | `bc1` | RGB with 1-bit alpha. Opaque textures and simple cutouts. |
| BC2 | `bc2` | RGBA with explicit 4-bit alpha. Sharp alpha transitions. |
| BC3 | `bc3` | RGBA with interpolated alpha. General-purpose transparency. |
| BC4 | `bc4` | Single channel. Grayscale, heightmaps, roughness. |
| BC4S | `bc4s` | Single channel, signed. |
| BC5 | `bc5` | Two channels. Normal maps. |
| BC5S | `bc5s` | Two channels, signed. |
| BC6H | `bc6h` | HDR half-float RGB. Environment maps, HDR textures. |
| BC6H SF | `bc6hsf` | HDR signed float RGB. |
| BC7 | `bc7` | High-quality RGBA. Best LDR quality, supports alpha. |
| ETC1 | `etc1` | Mobile-friendly RGB. |
| ETC2 RGBA | `etc2_rgba` | Mobile-friendly RGBA. |
| EAC R | `eac_r` | Single channel 11-bit. |
| EAC RG | `eac_rg` | Two channel 11-bit. |
| ASTC | `astc_WxH` | Adaptive scalable texture compression. Variable block sizes (4x4 to 12x12). |

All formats support quality presets from `ultra-fast` to `very-slow` where the encoder supports them.

Uncompressed formats are also supported using WebGPU names (e.g. `rgba8unorm`) or Vulkan names (e.g. `r8g8b8a8_unorm`).

## Output containers

The container format is inferred from the output file extension (`.ktx2` or `.dds`), or can be set explicitly with `--container`.

- **KTX2** — Khronos cross-platform container. Supports all formats. Optional zstd or zlib supercompression.
- **DDS** — DirectX standard. Does not support ETC/EAC or ASTC formats.

## Installation

```sh
# Install the CLI (includes all encoders)
cargo install ctt-cli

# Or add the library to your project
cargo add ctt
```

By default the library enables all encoders. To select specific encoders:

```sh
cargo add ctt --no-default-features --features encoder-bc7enc,encoder-intel,ispc-prebuilt
```

## Library usage

The library API mirrors the CLI. Build a `Surface`, wrap it in an `Image`, and call `convert`:

```rust
use ctt::{convert, ConvertSettings, Container, TargetFormat, Format};
use ctt::{Image, Surface, ColorSpace, AlphaMode};

let surface = Surface {
    data: pixel_bytes,
    width: 512,
    height: 512,
    stride: 512 * 4,
    format: Format::R8G8B8A8_UNORM,
    color_space: ColorSpace::Srgb,
    alpha: AlphaMode::Straight,
};

let image = Image { surfaces: vec![vec![surface]], is_cubemap: false };

let ktx2_bytes = convert(image, ConvertSettings {
    format: Some(TargetFormat::Compressed {
        format: Format::BC7_UNORM_BLOCK,
        encoder: Encoder::Auto,
    }),
    container: Container::ktx2(),
    ..Default::default()
})?;
```

See the [API documentation](https://docs.rs/ctt) for the full `ConvertSettings` options and the lower-level pipeline API.

## C API

C bindings are available as a separate crate. See [`crates/ctt-c-api/README.md`](crates/ctt-c-api/README.md) for build, link, and usage instructions; the full API reference lives in the generated header at `crates/ctt-c-api/include/ctt.h`.

## CLI usage

```
ctt <INPUT>... --output <PATH> [--format <FORMAT>] [OPTIONS]
```

When `--format` is omitted the input format is preserved without compression.

### Examples

Compress to BC7 (auto-selects bc7enc-rdo when available):

```sh
ctt diffuse.png -o diffuse.ktx2 -f bc7
```

Force the Intel ISPC encoder for BC7:

```sh
ctt diffuse.png -o diffuse.ktx2 -f intel_bc7
```

Normal map to BC5 as DDS:

```sh
ctt normal.png -o normal.dds -f bc5 --input-color-space linear
```

High quality:

```sh
ctt diffuse.png -o diffuse.ktx2 -f bc7 --quality slow
```

Cubemap from a cross layout:

```sh
ctt skybox_cross.png -o skybox.ktx2 -f bc6h --cubemap --cubemap-layout cross
```

Cubemap from six separate faces:

```sh
ctt px.png nx.png py.png ny.png pz.png nz.png -o skybox.ktx2 -f bc7 --cubemap
```

Generate mipmaps:

```sh
ctt diffuse.png -o diffuse.ktx2 -f bc7 --mipmap
```

With zstd supercompression:

```sh
ctt diffuse.png -o diffuse.ktx2 -f bc7 --zstd
```

Swizzle channels:

```sh
ctt input.png -o output.ktx2 -f bc7 --swizzle bgra
```

Run `ctt --help` for a full list of options.

## Minimum Supported Rust Version (MSRV)

The MSRV is **1.88** (edition 2024). MSRV bumps are considered breaking changes.

## Prebuilt binaries

By default, ctt ships prebuilt ISPC static libraries for all supported platforms. A default build requires only a Rust toolchain and a C++ compiler.

Every prebuilt binary has a [GitHub Artifact Attestation](https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds) that cryptographically proves it was produced by this repository's CI. See [`docs/prebuilt-binaries.md`](docs/prebuilt-binaries.md) for full details on the build process, attestation guarantees, and how to verify them.

To build from source instead (requires [`ispc.exe`](https://github.com/ispc/ispc/releases) on `PATH`):

```sh
cargo install ctt-cli --no-default-features --features ispc-build-from-source
```

## License

Licensed under any of:

- MIT License
- Apache License, Version 2.0
- Zlib License

at your option.
