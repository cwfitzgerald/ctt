# ctt

A texture compression tool and library powered by [Intel's ISPC Texture Compressor](https://github.com/GameTechDev/ISPCTextureCompressor). Compress images into GPU-ready block-compressed formats and package them as DDS or KTX2 files.

## Supported formats

| Format | Description | Quality presets |
|--------|-------------|-----------------|
| BC1 | RGB, 1-bit alpha. Good for opaque textures or simple cutouts. | No |
| BC3 | RGBA with interpolated alpha. General-purpose with transparency. | No |
| BC4 | Single channel (grayscale, heightmaps). | No |
| BC5 | Two channels (normal maps). | No |
| BC6H | HDR (half-float RGB). For environment maps and HDR textures. | Yes |
| BC7 | High-quality RGBA. Best quality for LDR textures. | Yes |
| ETC1 | Mobile-friendly RGB compression. | No |

Output containers:

- **KTX2** (default) — Cross-platform, supports all formats.
- **DDS** — DirectX standard. Does not support ETC1.

## Prerequisites

Building `ctt` requires the following tools to be installed and available on your `PATH`:

- **[ISPC](https://ispc.github.io/)** — The Intel SPMD Program Compiler, used to compile the compression kernels.
- **libclang** — Required by [bindgen](https://rust-lang.github.io/bindgen/) to generate Rust FFI bindings from C headers. On Windows, install via the LLVM installer or Visual Studio. On Linux, install `libclang-dev`. On macOS, it ships with Xcode command-line tools.
- **A C++ compiler** — Needed to compile ASTC helper code (MSVC on Windows, GCC/Clang on Linux/macOS).

## Installation

```sh
# Install the CLI
cargo install ctt-cli

# Or add the library to your project
cargo add ctt
```

## CLI usage

```
ctt <INPUT>... --output <PATH> --format <FORMAT> [OPTIONS]
```

### Basic examples

Compress a PNG to BC7 in KTX2:

```sh
ctt diffuse.png -o diffuse.ktx2 -f bc7
```

Compress a normal map to BC5, output as DDS:

```sh
ctt normal.png -o normal.dds -f bc5 -c dds --color-space linear
```

Compress with high quality:

```sh
ctt diffuse.png -o diffuse.ktx2 -f bc7 --quality slow
```

Compress a cubemap from a cross layout image:

```sh
ctt skybox_cross.png -o skybox.ktx2 -f bc6h --cubemap --cubemap-layout cross
```

Compress a cubemap from six separate face images:

```sh
ctt px.png nx.png py.png ny.png pz.png nz.png -o skybox.ktx2 -f bc7 --cubemap
```

Swizzle channels (swap red and blue):

```sh
ctt input.png -o output.ktx2 -f bc7 --swizzle bgra
```

### Options

```
Options:
  -o, --output <OUTPUT>
          Output file path

  -f, --format <FORMAT>
          Compression format (bc1, bc3, bc4, bc5, bc6h, bc7, etc1, astc_4x4, astc_6x6, ...)

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
          Color space of the input. Used for selecting output color space and
          performing mipmap generation

          [default: srgb]
          [possible values: srgb, linear]

      --quality <QUALITY>
          Compression quality preset

          [default: basic]
          [possible values: ultra-fast, very-fast, fast, basic, slow, very-slow]

      --alpha
          Encode alpha channel (for BC7)

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
use ctt::format::{ColorSpace, CompressedFormat, ChannelType, PixelComponents, PixelFormat};
use ctt::image::{ImageLayout, RawImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load an image (using the `image` crate or any other loader).
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

    // Wrap in a layout (single layer, single mip).
    let layout = ImageLayout {
        layers: vec![vec![raw]],
        is_cubemap: false,
    };

    // Configure compression.
    let config = CompressConfig {
        format: CompressedFormat::Bc7,
        output_format: OutputFormat::Ktx2,
        swizzle: None,
        color_space: ColorSpace::Srgb,
        encode_settings: None, // uses sensible defaults
    };

    // Run the pipeline: convert -> compress -> encode.
    let output_bytes = ctt::pipeline::run(&config, layout)?;
    fs::write("diffuse.ktx2", &output_bytes)?;

    Ok(())
}
```

## License

Licensed under any of:

- MIT License
- Apache License, Version 2.0
- Zlib License

at your option.
