// extern "C" forwarding wrapper for etcpak ISA variants.
//
// Compiled once per ISA variant. build.rs defines:
//   ETCPAK_NS      — C++ namespace containing this variant's symbols (e.g. etcpak_sse41)
//   ETCPAK_PREFIX  — C-linkage symbol prefix                        (e.g. etcpak_sse41_)

#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// Forward-declare namespaced functions from the vendored etcpak source.
// ---------------------------------------------------------------------------

namespace ETCPAK_NS {
// ProcessRGB.hpp — ETC compression
void CompressEtc1Rgb(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);
void CompressEtc1RgbDither(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);
void CompressEtc2Rgb(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width, bool useHeuristics);
void CompressEtc2Rgba(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width, bool useHeuristics);
void CompressEacR(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);
void CompressEacRg(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);

// ProcessDxtc.hpp — BC compression (BC7 excluded from safe API)
void CompressBc1(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);
void CompressBc1Dither(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);
void CompressBc3(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);
void CompressBc4(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);
void CompressBc5(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width);

// Decode.hpp — decompression
void DecodeRGB(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height);
void DecodeRGBA(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height);
void DecodeR(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height, bool isSigned);
void DecodeRG(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height, bool isSigned);
void DecodeBc1(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height);
void DecodeBc3(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height);
void DecodeBc4(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height);
void DecodeBc5(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height);
void DecodeBc7(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height);
} // namespace ETCPAK_NS

// ---------------------------------------------------------------------------
// Token-pasting helpers
// ---------------------------------------------------------------------------

#define ETCPAK_CONCAT2(a, b) a##b
#define ETCPAK_CONCAT(a, b) ETCPAK_CONCAT2(a, b)
#define ETCPAK_FN(name) ETCPAK_CONCAT(ETCPAK_PREFIX, name)

// ---------------------------------------------------------------------------
// extern "C" forwarding functions
// ---------------------------------------------------------------------------

extern "C" {

// ── ETC compression ──────────────────────────────────────────────────────────

void ETCPAK_FN(CompressEtc1Rgb)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressEtc1Rgb(src, dst, blocks, width);
}

void ETCPAK_FN(CompressEtc1RgbDither)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressEtc1RgbDither(src, dst, blocks, width);
}

void ETCPAK_FN(CompressEtc2Rgb)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width, bool useHeuristics) {
    ETCPAK_NS::CompressEtc2Rgb(src, dst, blocks, width, useHeuristics);
}

void ETCPAK_FN(CompressEtc2Rgba)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width, bool useHeuristics) {
    ETCPAK_NS::CompressEtc2Rgba(src, dst, blocks, width, useHeuristics);
}

void ETCPAK_FN(CompressEacR)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressEacR(src, dst, blocks, width);
}

void ETCPAK_FN(CompressEacRg)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressEacRg(src, dst, blocks, width);
}

// ── BC compression ───────────────────────────────────────────────────────────

void ETCPAK_FN(CompressBc1)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressBc1(src, dst, blocks, width);
}

void ETCPAK_FN(CompressBc1Dither)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressBc1Dither(src, dst, blocks, width);
}

void ETCPAK_FN(CompressBc3)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressBc3(src, dst, blocks, width);
}

void ETCPAK_FN(CompressBc4)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressBc4(src, dst, blocks, width);
}

void ETCPAK_FN(CompressBc5)(const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width) {
    ETCPAK_NS::CompressBc5(src, dst, blocks, width);
}

// ── Decompression ────────────────────────────────────────────────────────────

void ETCPAK_FN(DecodeRGB)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height) {
    ETCPAK_NS::DecodeRGB(src, dst, width, height);
}

void ETCPAK_FN(DecodeRGBA)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height) {
    ETCPAK_NS::DecodeRGBA(src, dst, width, height);
}

void ETCPAK_FN(DecodeR)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height, bool isSigned) {
    ETCPAK_NS::DecodeR(src, dst, width, height, isSigned);
}

void ETCPAK_FN(DecodeRG)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height, bool isSigned) {
    ETCPAK_NS::DecodeRG(src, dst, width, height, isSigned);
}

void ETCPAK_FN(DecodeBc1)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height) {
    ETCPAK_NS::DecodeBc1(src, dst, width, height);
}

void ETCPAK_FN(DecodeBc3)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height) {
    ETCPAK_NS::DecodeBc3(src, dst, width, height);
}

void ETCPAK_FN(DecodeBc4)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height) {
    ETCPAK_NS::DecodeBc4(src, dst, width, height);
}

void ETCPAK_FN(DecodeBc5)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height) {
    ETCPAK_NS::DecodeBc5(src, dst, width, height);
}

void ETCPAK_FN(DecodeBc7)(const uint64_t* src, uint32_t* dst, int32_t width, int32_t height) {
    ETCPAK_NS::DecodeBc7(src, dst, width, height);
}

} // extern "C"
