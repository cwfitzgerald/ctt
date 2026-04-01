# ctt-intel-texture-compressor

Vendored Intel ISPC texture compressor bindings for the ctt workspace.

Based on [intel-tex-rs-2](https://github.com/Traverse-Research/intel-tex-rs-2) by Traverse Research & Graham Wihlidal, which wraps Intel's [ISPCTextureCompressor](https://github.com/GameTechDev/ISPCTextureCompressor).

## Supported formats

- BC1, BC3, BC4, BC5, BC6H, BC7
- ETC1

## Build requirements

- [ISPC compiler](https://ispc.github.io/) on `PATH`
- `libclang` (for bindgen)

## License

The Rust bindings are licensed under MIT OR Apache-2.0 (see `LICENSE-MIT` and `LICENSE-APACHE`).

The ISPC/C++ source code in `ispc/` is licensed under MIT by Intel Corporation (see `LICENSE-MIT-INTEL`).
