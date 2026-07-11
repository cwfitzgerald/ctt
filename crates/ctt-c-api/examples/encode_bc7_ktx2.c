/*
 * End-to-end encode: a 4x4 RGBA8 image is compressed to BC7 and serialized
 * into a KTX2 container in memory. Asserts the output is non-empty and begins
 * with the KTX2 identifier.
 */
#include "../include/ctt.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* The 12-byte KTX2 file identifier: «KTX 20»\r\n\x1A\n. */
static const uint8_t KTX2_MAGIC[12] = {
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

int main(void) {
    /* One BC7 block worth of pixels: a 4x4 RGBA8 gradient. */
    uint8_t pixels[4 * 4 * 4];
    for (int i = 0; i < 16; ++i) {
        pixels[i * 4 + 0] = (uint8_t)(i * 16);
        pixels[i * 4 + 1] = (uint8_t)(255 - i * 16);
        pixels[i * 4 + 2] = (uint8_t)(i * 8);
        pixels[i * 4 + 3] = 255;
    }

    ctt_surface *s = ctt_surface_create(
        pixels, sizeof pixels,
        4, 4, 1,
        4 * 4, 0,
        CTT_FORMAT_R8G8B8A8_UNORM,
        CTT_COLOR_SPACE_LINEAR,
        CTT_ALPHA_MODE_STRAIGHT);
    if (!s) {
        fprintf(stderr, "ctt_surface_create failed: %s\n", ctt_last_error_message());
        return 1;
    }

    ctt_image *img = ctt_image_create(CTT_TEXTURE_KIND_TEXTURE2D);
    if (!img) {
        fprintf(stderr, "ctt_image_create failed: %s\n", ctt_last_error_message());
        ctt_surface_destroy(s);
        return 2;
    }
    size_t layer = 0;
    if (ctt_image_add_layer(img, &layer) != CTT_STATUS_OK) {
        ctt_surface_destroy(s);
        ctt_image_destroy(img);
        return 3;
    }
    if (ctt_image_push_mip(img, layer, s) != CTT_STATUS_OK) {
        /* `s` was consumed even on failure. */
        ctt_image_destroy(img);
        return 4;
    }
    /* `s` consumed; do NOT destroy. */

    ctt_convert_settings cfg = ctt_convert_settings_default();
    cfg.format = (ctt_target_format){
        .tag = CTT_TARGET_FORMAT_COMPRESSED,
        .compressed = {CTT_FORMAT_BC7_UNORM_BLOCK, ctt_encoder_auto()},
    };
    /* container stays at its default (KTX2). */

    ctt_pipeline_output *out = NULL;
    ctt_status st = ctt_convert(img, &cfg, &out);
    /* `img` consumed regardless of result. */
    if (st != CTT_STATUS_OK) {
        fprintf(stderr, "ctt_convert failed (%d): %s\n", st, ctt_last_error_message());
        return 5;
    }

    if (ctt_pipeline_output_get_kind(out) != CTT_PIPELINE_OUTPUT_KIND_ENCODED) {
        fprintf(stderr, "expected encoded output\n");
        ctt_pipeline_output_destroy(out);
        return 6;
    }

    const uint8_t *bytes = ctt_pipeline_output_encoded_data(out);
    size_t len = ctt_pipeline_output_encoded_len(out);
    if (!bytes || len == 0) {
        fprintf(stderr, "encoded output is empty\n");
        ctt_pipeline_output_destroy(out);
        return 7;
    }
    if (len < sizeof KTX2_MAGIC || memcmp(bytes, KTX2_MAGIC, sizeof KTX2_MAGIC) != 0) {
        fprintf(stderr, "output does not start with KTX2 magic\n");
        ctt_pipeline_output_destroy(out);
        return 8;
    }

    ctt_pipeline_output_destroy(out);
    printf("ok\n");
    return 0;
}
