// This header mimics the ISPC-generated kernel_astc_ispc.h.
// It provides the `ispc` namespace declarations needed by ispc_texcomp_astc.cpp.
// Regenerate from actual ISPC output if the kernel_astc.ispc interface changes.

#pragma once
#include <stdint.h>

#ifdef __cplusplus
namespace ispc {
#endif

struct rgba_surface {
    uint8_t *ptr;
    int32_t width;
    int32_t height;
    int32_t stride;
};

struct astc_enc_settings {
    int32_t block_width;
    int32_t block_height;
    int32_t channels;
    int32_t fastSkipThreshold;
    int32_t refineIterations;
};

struct astc_enc_context {
    int32_t width;
    int32_t height;
    int32_t channels;
    bool dual_plane;
    int32_t partitions;
    int32_t color_endpoint_pairs;
};

struct astc_block {
    int32_t width;
    int32_t height;
    uint8_t dual_plane;
    int32_t weight_range;
    uint8_t weights[64];
    int32_t color_component_selector;
    int32_t partitions;
    int32_t partition_id;
    int32_t color_endpoint_pairs;
    int32_t channels;
    int32_t color_endpoint_modes[4];
    int32_t endpoint_range;
    uint8_t endpoints[18];
};

#ifdef __cplusplus
extern "C" {
#endif

    extern int32_t get_programCount();
    extern void astc_rank_ispc(
        struct rgba_surface *src,
        int32_t xx,
        int32_t yy,
        uint32_t *mode_buffer,
        struct astc_enc_settings *settings);
    extern void astc_encode_ispc(
        struct rgba_surface *src,
        float *block_scores,
        uint8_t *dst,
        uint64_t *list,
        struct astc_enc_context *list_context,
        struct astc_enc_settings *settings);

#ifdef __cplusplus
} // extern "C"
} // namespace ispc
#endif
