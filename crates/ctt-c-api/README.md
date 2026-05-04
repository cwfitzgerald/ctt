# ctt-c-api

C bindings for the [ctt](../../README.md) texture compression library.

The crate produces both a static and a dynamic library and a hand-curated header (`include/ctt.h`).

## Prebuilt binaries

Each archive contains `include/ctt.h`, the static and dynamic libraries under `lib/`, the runtime DLL under `bin/` (Windows only), and the project licenses.

- **Tagged releases** — every `v*` tag publishes signed, attested archives named `ctt-c-api-<target>-<tag>.{zip,tar.gz}` on the [Releases page](https://github.com/cwfitzgerald/ctt/releases).
- **Any commit on `trunk` or a PR branch** — the [CI workflow](https://github.com/cwfitzgerald/ctt/actions/workflows/ci.yml) uploads the same archives as per-target workflow artifacts named `release-<target>`. Open a run and download from the Artifacts section.

Targets: `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `x86_64-apple-darwin`, `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`.

## Building from source

```sh
cargo build -p ctt-c-api --release
```

This emits the libraries into the workspace `target/release/` directory:

| Platform | Static | Dynamic |
|----------|--------|---------|
| Windows (MSVC) | `ctt_capi.lib` | `ctt_capi.dll` + `ctt_capi.dll.lib` |
| macOS | `libctt_capi.a` | `libctt_capi.dylib` |
| Linux | `libctt_capi.a` | `libctt_capi.so` |

Encoder backends and ISPC sourcing are selected via Cargo features — see `Cargo.toml` for the list. The defaults match the Rust crate (all encoders, prebuilt ISPC).

## Using

Add `crates/ctt-c-api/include` to the include path and `#include <ctt.h>`:

```c
#include <ctt.h>
```

### Static linking

Link the static archive plus the system libraries the Rust runtime pulls in.

- Linux: `-lpthread -ldl -lm -lrt -lutil -lgcc_s -lstdc++`
- macOS: `-framework Security -framework CoreFoundation -liconv -lresolv -lc++`
- Windows (MSVC): `kernel32.lib ntdll.lib userenv.lib ws2_32.lib dbghelp.lib advapi32.lib`

The `astcenc` and `compressonator` encoders bundle C++ sources, so the C++ standard library must be linked. On Windows MSVC it is pulled in automatically by the default CRT; on Linux/macOS, link with `-lstdc++` / `-lc++` (or invoke the linker via the C++ driver, e.g. `g++` / `clang++`).

### Dynamic linking

Link against the import library (Windows) or shared object (macOS/Linux) and ensure the `.dll` / `.dylib` / `.so` is found at runtime (next to the executable, on `PATH`/`LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`, or via an embedded rpath).

A minimal end-to-end example lives in [`examples/sanity_check.c`](examples/sanity_check.c); the integration test in `tests/c_examples.rs` shows the exact compiler invocations used in CI for each platform and linkage mode.

## Documentation

The full API reference lives inline in [`include/ctt.h`](include/ctt.h). The header opens with an overview of the encode and decode pipelines, the memory ownership model, error reporting, threading, and format conventions; per-function doc comments cover the rest.

## License

Licensed under MIT OR Apache-2.0 OR Zlib at your option.
