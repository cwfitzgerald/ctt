# ctt

A Rust library and CLI for GPU texture compression. ctt provides a unified interface over multiple encoder backends — feed it an image, pick a format, and it handles pixel conversion, compression, and container encoding.

## Encoders

ctt binds to established open-source compression libraries and exposes them through a common `Encoder` trait. Each backend is compiled as an optional feature and can be enabled independently.

| Backend | Prefix | Feature | Formats | Description |
|---------|--------|---------|---------|-------------|
| [**bc7enc-rdo**](https://github.com/richgel999/bc7enc_rdo) | `bc7e_` | `encoder-bc7enc` | BC7 | Perceptual BC7 encoder with RDO support. |
| [**Intel ISPC Texture Compressor**](https://github.com/GameTechDev/ISPCTextureCompressor) | `ispc_` | `encoder-ispc` | BC1, BC3, BC4, BC5, BC6H, BC7, ETC1 | SIMD-optimized BCn and ETC encoder. |

Encoders are listed in priority order — when multiple encoders support the same format (e.g. BC7), the first one in the table is used. To override this, prefix the format with an encoder name (e.g. `ispc_bc7`).

Use `ctt --list-encoders` to see what's available in your build:

```
$ ctt --list-encoders
Encoder    Priority     Formats
-------    --------     -------
bc7e       1            bc7
ispc       2            bc1, bc3, bc4, bc5, bc6h, bc7, etc1
```

## Formats

| Format | Description |
|--------|-------------|
| BC1 | RGB with 1-bit alpha. Opaque textures and simple cutouts. |
| BC3 | RGBA with interpolated alpha. General-purpose transparency. |
| BC4 | Single channel. Grayscale, heightmaps, roughness. |
| BC5 | Two channels. Normal maps. |
| BC6H | HDR half-float RGB. Environment maps, HDR textures. |
| BC7 | High-quality RGBA. Best LDR quality, supports alpha. |
| ETC1 | Mobile-friendly RGB. |

All formats support quality presets from `ultra-fast` to `very-slow` where the encoder supports them.

## Output containers

- **KTX2** (default) — Khronos cross-platform container. Supports all formats.
- **DDS** — DirectX standard. Does not support ETC1.

## Prebuilt binaries

By default, ctt ships prebuilt ISPC static libraries for all supported platforms (linux, macOS, Windows; x86_64 and aarch64). A default build requires only a Rust toolchain and a C++ compiler.

Every prebuilt binary has a [GitHub Artifact Attestation](https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds) that cryptographically proves it was produced by this repository's CI. See [`docs/prebuilt-binaries.md`](docs/prebuilt-binaries.md) for full details on the build process, attestation guarantees, and how to verify them.

To build from source instead (requires [`ispc.exe`](https://github.com/ispc/ispc/releases) on `PATH`):

```sh
cargo install ctt-cli --no-default-features --features ispc-build-from-source
```

## Installation

```sh
# Install the CLI (includes all encoders)
cargo install ctt-cli

# Or add the library to your project
cargo add ctt
```

By default the library enables `encoder-ispc`. To enable bc7enc-rdo as well:

```sh
cargo add ctt --features encoder-bc7enc
```

## CLI usage

```
ctt <INPUT>... --output <PATH> --format <FORMAT> [OPTIONS]
```

### Examples

Compress to BC7 (auto-selects bc7enc-rdo when available):

```sh
ctt diffuse.png -o diffuse.ktx2 -f bc7
```

Force the ISPC encoder for BC7:

```sh
ctt diffuse.png -o diffuse.ktx2 -f ispc_bc7
```

Normal map to BC5 as DDS:

```sh
ctt normal.png -o normal.dds -f bc5 -c dds --color-space linear
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

Swizzle channels:

```sh
ctt input.png -o output.ktx2 -f bc7 --swizzle bgra
```

### Options

```
Options:
  -o, --output <OUTPUT>
          Output file path

  -f, --format <FORMAT>
          Compression format. Bare (bc1, bc7) or prefixed with encoder
          (ispc_bc7, bc7e_bc7)

  -c, --container <CONTAINER>
          Output container format

          [default: ktx2]
          [possible values: dds, ktx2]

      --cubemap
          Treat input as a cubemap

      --cubemap-layout <CUBEMAP_LAYOUT>
          Cubemap layout when using a single input image

          [default: cross]
          [possible values: cross, strip]

      --swizzle <SWIZZLE>
          Remap RGBA channels. 4 characters from: rgba01.

          "bgra" = swap red/blue, "0r0g" = 2 channel normal map to BC3 packing
          "rgb1" = force opaque, "r000" = ignore non-red channel.

      --color-space <COLOR_SPACE>
          Color space of the input

          [default: srgb]
          [possible values: srgb, linear]

      --quality <QUALITY>
          Compression quality preset

          [default: basic]
          [possible values: ultra-fast, very-fast, fast, basic, slow, very-slow]

      --alpha
          Encode alpha channel (for BC7)

      --list-encoders
          List available encoder backends and their supported formats

  -v...
          Increase logging verbosity (-v = debug, -vv = trace)

  -h, --help
          Print help

  -V, --version
          Print version
```

## Library usage

```rust
use std::fs;
use ctt::config::{CompressConfig, OutputFormat};
use ctt::encoder::Quality;
use ctt::format::{ColorSpace, CompressedFormat, ChannelType, PixelComponents, PixelFormat};
use ctt::image::{ImageLayout, RawImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("diffuse.png")?.to_rgba8();
    let (width, height) = img.dimensions();

    let raw = RawImage {
        data: img.into_raw(),
        width,
        height,
        stride: width * 4,
        pixel_format: PixelFormat {
            components: PixelComponents::Rgba,
            channel_type: ChannelType::U8,
            color_space: ColorSpace::Srgb,
        },
    };

    let layout = ImageLayout {
        layers: vec![vec![raw]],
        is_cubemap: false,
    };

    let config = CompressConfig {
        format: CompressedFormat::Bc7,
        output_format: OutputFormat::Ktx2,
        swizzle: None,
        color_space: ColorSpace::Srgb,
        quality: Quality::Basic,
        encoder_name: None,        // auto-select best encoder
        encoder_settings: None,    // use encoder defaults
    };

    let output_bytes = ctt::pipeline::run(&config, layout)?;
    fs::write("diffuse.ktx2", &output_bytes)?;

    Ok(())
}
```

### Custom encoder selection

To force a specific encoder programmatically, set `encoder_name`:

```rust
let config = CompressConfig {
    format: CompressedFormat::Bc7,
    encoder_name: Some("ispc".into()), // force ISPC instead of bc7enc-rdo
    // ...
};
```

Or build your own registry with custom priority:

```rust
use ctt::encoder::EncoderRegistry;
use ctt::encoders::ispc::IspcEncoder;

let mut registry = EncoderRegistry::new();
registry.register(Box::new(IspcEncoder)); // ISPC gets priority 1

let output = ctt::pipeline::run_with_registry(&config, layout, &registry)?;
```

## Minimum Supported Rust Version (MSRV)

The MSRV is **1.85** (edition 2024). MSRV bumps are considered breaking changes.

## License

Licensed under any of:

- MIT License
- Apache License, Version 2.0
- Zlib License

at your option.
