#include "ffi_wrapper.h"
#include "basisu_transcoder.h"
#include <cstring>

static_assert(static_cast<uint32_t>(cPackBC7FlagUse2SubsetsRGB) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagUse2SubsetsRGB),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagUse2SubsetsRGBA) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagUse2SubsetsRGBA),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagUse3SubsetsRGB) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagUse3SubsetsRGB),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagUseDualPlaneRGB) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagUseDualPlaneRGB),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagUseDualPlaneRGBA) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagUseDualPlaneRGBA),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagPBitOpt) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagPBitOpt),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagPBitOptMode6) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagPBitOptMode6),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagUseTrivialMode6) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagUseTrivialMode6),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagPartiallyAnalyticalRGB) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagPartiallyAnalyticalRGB),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagPartiallyAnalyticalRGBA) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagPartiallyAnalyticalRGBA),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagNonAnalyticalRGB) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagNonAnalyticalRGB),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagNonAnalyticalRGBA) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagNonAnalyticalRGBA),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagASTCCompatible) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagASTCCompatible),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagDisableRGBDualPlane) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagDisableRGBDualPlane),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagDefaultFastest) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagDefaultFastest),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagDefaultFaster) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagDefaultFaster),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagDefaultFast) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagDefaultFast),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagDefault) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagDefault),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagDefaultPartiallyAnalytical) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagDefaultPartiallyAnalytical),
              "BC7F flag mismatch");
static_assert(static_cast<uint32_t>(cPackBC7FlagDefaultNonAnalytical) ==
                  static_cast<uint32_t>(basist::bc7f::cPackBC7FlagDefaultNonAnalytical),
              "BC7F flag mismatch");

extern "C" void ctt_bc7f_init() { basist::basisu_transcoder_init(); }

extern "C" void ctt_bc7f_compress_blocks(uint8_t *output, const uint8_t *pixels, size_t count,
                                         uint32_t flags) {
    static_assert(sizeof(basist::color_rgba) == 4, "RGBA pixels must occupy four bytes");
    for (size_t i = 0; i < count; ++i) {
        // BC7F reads pixels through uint32_t pointers.
        alignas(uint32_t) basist::color_rgba block[16];
        std::memcpy(block, pixels + i * 64, sizeof(block));
        basist::bc7f::fast_pack_bc7_auto_rgba(output + i * 16, block, flags);
    }
}
