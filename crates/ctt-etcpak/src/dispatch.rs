//! Runtime ISA dispatch for etcpak.
//!
//! Each ISA variant is compiled as a separate static library with all internal
//! C++ symbols in a per-variant namespace. The `ffi_wrapper.cpp` file provides
//! `extern "C"` forwarding functions with ISA-prefixed names
//! (e.g. `etcpak_sse41_CompressEtc1Rgb`). At startup, `init_dispatch()` probes
//! CPU features and fills a `Dispatch` function-pointer table with the best
//! available variant.

use std::sync::OnceLock;

// ── Function pointer types ───────────────────────────────────────────────────

pub(crate) type CompressFn =
    unsafe extern "C" fn(src: *const u32, dst: *mut u64, blocks: u32, width: usize);
pub(crate) type CompressBoolFn =
    unsafe extern "C" fn(src: *const u32, dst: *mut u64, blocks: u32, width: usize, flag: bool);
pub(crate) type DecodeFn =
    unsafe extern "C" fn(src: *const u64, dst: *mut u32, width: i32, height: i32);
pub(crate) type DecodeBoolFn =
    unsafe extern "C" fn(src: *const u64, dst: *mut u32, width: i32, height: i32, is_signed: bool);

// ── Dispatch table ───────────────────────────────────────────────────────────

pub(crate) struct Dispatch {
    // ETC compression
    pub compress_etc1_rgb: CompressFn,
    pub compress_etc1_rgb_dither: CompressFn,
    pub compress_etc2_rgb: CompressBoolFn,
    pub compress_etc2_rgba: CompressBoolFn,
    pub compress_eac_r: CompressFn,
    pub compress_eac_rg: CompressFn,
    // BC compression
    pub compress_bc1: CompressFn,
    pub compress_bc1_dither: CompressFn,
    pub compress_bc3: CompressFn,
    pub compress_bc4: CompressFn,
    pub compress_bc5: CompressFn,
    // Decompression
    pub decode_rgb: DecodeFn,
    pub decode_rgba: DecodeFn,
    pub decode_r: DecodeBoolFn,
    pub decode_rg: DecodeBoolFn,
    pub decode_bc1: DecodeFn,
    pub decode_bc3: DecodeFn,
    pub decode_bc4: DecodeFn,
    pub decode_bc5: DecodeFn,
    pub decode_bc7: DecodeFn,
}

// SAFETY: Dispatch only contains function pointers, which are Send+Sync.
unsafe impl Send for Dispatch {}
unsafe impl Sync for Dispatch {}

pub(crate) fn dispatch() -> &'static Dispatch {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH.get_or_init(init_dispatch)
}

// ── ISA-specific extern "C" declarations ─────────────────────────────────────

mod ffi {
    macro_rules! declare_isa {
        ($isa:ident) => {
            paste::paste! {
                unsafe extern "C" {
                    // ETC compression
                    pub fn [<etcpak_ $isa _CompressEtc1Rgb>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);
                    pub fn [<etcpak_ $isa _CompressEtc1RgbDither>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);
                    pub fn [<etcpak_ $isa _CompressEtc2Rgb>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize, use_heuristics: bool);
                    pub fn [<etcpak_ $isa _CompressEtc2Rgba>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize, use_heuristics: bool);
                    pub fn [<etcpak_ $isa _CompressEacR>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);
                    pub fn [<etcpak_ $isa _CompressEacRg>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);

                    // BC compression
                    pub fn [<etcpak_ $isa _CompressBc1>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);
                    pub fn [<etcpak_ $isa _CompressBc1Dither>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);
                    pub fn [<etcpak_ $isa _CompressBc3>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);
                    pub fn [<etcpak_ $isa _CompressBc4>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);
                    pub fn [<etcpak_ $isa _CompressBc5>](
                        src: *const u32, dst: *mut u64, blocks: u32, width: usize);

                    // Decompression
                    pub fn [<etcpak_ $isa _DecodeRGB>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32);
                    pub fn [<etcpak_ $isa _DecodeRGBA>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32);
                    pub fn [<etcpak_ $isa _DecodeR>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32, is_signed: bool);
                    pub fn [<etcpak_ $isa _DecodeRG>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32, is_signed: bool);
                    pub fn [<etcpak_ $isa _DecodeBc1>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32);
                    pub fn [<etcpak_ $isa _DecodeBc3>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32);
                    pub fn [<etcpak_ $isa _DecodeBc4>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32);
                    pub fn [<etcpak_ $isa _DecodeBc5>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32);
                    pub fn [<etcpak_ $isa _DecodeBc7>](
                        src: *const u64, dst: *mut u32, width: i32, height: i32);
                }
            }
        };
    }

    #[cfg(target_arch = "x86_64")]
    declare_isa!(scalar);
    #[cfg(target_arch = "x86_64")]
    declare_isa!(sse41);
    #[cfg(target_arch = "x86_64")]
    declare_isa!(avx2);

    #[cfg(target_arch = "aarch64")]
    declare_isa!(neon);

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    declare_isa!(scalar);
}

// ── Dispatch initialization ──────────────────────────────────────────────────

macro_rules! make_dispatch {
    ($isa:ident) => {
        paste::paste! {
            Dispatch {
                compress_etc1_rgb: ffi::[<etcpak_ $isa _CompressEtc1Rgb>],
                compress_etc1_rgb_dither: ffi::[<etcpak_ $isa _CompressEtc1RgbDither>],
                compress_etc2_rgb: ffi::[<etcpak_ $isa _CompressEtc2Rgb>],
                compress_etc2_rgba: ffi::[<etcpak_ $isa _CompressEtc2Rgba>],
                compress_eac_r: ffi::[<etcpak_ $isa _CompressEacR>],
                compress_eac_rg: ffi::[<etcpak_ $isa _CompressEacRg>],
                compress_bc1: ffi::[<etcpak_ $isa _CompressBc1>],
                compress_bc1_dither: ffi::[<etcpak_ $isa _CompressBc1Dither>],
                compress_bc3: ffi::[<etcpak_ $isa _CompressBc3>],
                compress_bc4: ffi::[<etcpak_ $isa _CompressBc4>],
                compress_bc5: ffi::[<etcpak_ $isa _CompressBc5>],
                decode_rgb: ffi::[<etcpak_ $isa _DecodeRGB>],
                decode_rgba: ffi::[<etcpak_ $isa _DecodeRGBA>],
                decode_r: ffi::[<etcpak_ $isa _DecodeR>],
                decode_rg: ffi::[<etcpak_ $isa _DecodeRG>],
                decode_bc1: ffi::[<etcpak_ $isa _DecodeBc1>],
                decode_bc3: ffi::[<etcpak_ $isa _DecodeBc3>],
                decode_bc4: ffi::[<etcpak_ $isa _DecodeBc4>],
                decode_bc5: ffi::[<etcpak_ $isa _DecodeBc5>],
                decode_bc7: ffi::[<etcpak_ $isa _DecodeBc7>],
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
fn init_dispatch() -> Dispatch {
    if is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("popcnt")
        && is_x86_feature_detected!("f16c")
    {
        make_dispatch!(avx2)
    } else if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") {
        make_dispatch!(sse41)
    } else {
        make_dispatch!(scalar)
    }
}

#[cfg(target_arch = "aarch64")]
fn init_dispatch() -> Dispatch {
    // NEON is mandatory on AArch64 (ARMv8-A).
    make_dispatch!(neon)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn init_dispatch() -> Dispatch {
    make_dispatch!(scalar)
}
