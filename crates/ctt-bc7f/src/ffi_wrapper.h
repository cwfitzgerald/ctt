#pragma once
#include <stddef.h>
#include <stdint.h>

enum {
    cPackBC7FlagUse2SubsetsRGB = 1,
    cPackBC7FlagUse2SubsetsRGBA = 2,
    cPackBC7FlagUse3SubsetsRGB = 4,
    cPackBC7FlagUseDualPlaneRGB = 8,
    cPackBC7FlagUseDualPlaneRGBA = 16,
    cPackBC7FlagPBitOpt = 32,
    cPackBC7FlagPBitOptMode6 = 64,
    cPackBC7FlagUseTrivialMode6 = 128,
    cPackBC7FlagPartiallyAnalyticalRGB = 256,
    cPackBC7FlagPartiallyAnalyticalRGBA = 512,
    cPackBC7FlagNonAnalyticalRGB = 1024,
    cPackBC7FlagNonAnalyticalRGBA = 2048,
    cPackBC7FlagASTCCompatible = 4096,
    cPackBC7FlagDisableRGBDualPlane = 8192,
    cPackBC7FlagDefaultFastest = cPackBC7FlagUseTrivialMode6,
    cPackBC7FlagDefaultFaster = cPackBC7FlagPBitOpt | cPackBC7FlagUseDualPlaneRGBA |
        cPackBC7FlagUseTrivialMode6,
    cPackBC7FlagDefaultFast = cPackBC7FlagUse2SubsetsRGB | cPackBC7FlagUse2SubsetsRGBA |
        cPackBC7FlagUseDualPlaneRGBA | cPackBC7FlagPBitOpt | cPackBC7FlagUseTrivialMode6,
    cPackBC7FlagDefault = (cPackBC7FlagUse2SubsetsRGB | cPackBC7FlagUse2SubsetsRGBA |
                           cPackBC7FlagUse3SubsetsRGB) |
        (cPackBC7FlagUseDualPlaneRGB | cPackBC7FlagUseDualPlaneRGBA) |
        (cPackBC7FlagPBitOpt | cPackBC7FlagPBitOptMode6) | cPackBC7FlagUseTrivialMode6,
    cPackBC7FlagDefaultPartiallyAnalytical = cPackBC7FlagDefault |
        (cPackBC7FlagPartiallyAnalyticalRGB | cPackBC7FlagPartiallyAnalyticalRGBA),
    cPackBC7FlagDefaultNonAnalytical =
            (cPackBC7FlagDefaultPartiallyAnalytical |
             (cPackBC7FlagNonAnalyticalRGB | cPackBC7FlagNonAnalyticalRGBA)) &
        ~cPackBC7FlagUseTrivialMode6
};

#ifdef __cplusplus
extern "C" {
#endif
void ctt_bc7f_init(void);
void ctt_bc7f_compress_blocks(uint8_t *output, const uint8_t *pixels, size_t count, uint32_t flags);
#ifdef __cplusplus
}
#endif
