/*
 * Tiny C sanity check that the generated header is syntactically valid
 * and exposes the expected entry points. Built with whatever C compiler
 * is available; not part of CI by default.
 */
#include "../include/ctt.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    /* Round-trip a 1x1 RGBA8 pixel through ctt_convert. */
    uint8_t pixel[4] = {200, 120, 60, 255};
    ctt_surface *s = ctt_surface_create(
        pixel, 4,
        1, 1, 1,
        4, 0,
        CTT_FORMAT_R8G8B8A8_UNORM,
        CTT_COLOR_SPACE_LINEAR,
        CTT_ALPHA_MODE_OPAQUE);
    if (!s) {
        fprintf(stderr, "ctt_surface_create failed: %s\n", ctt_last_error_message());
        return 1;
    }

    ctt_image *img = ctt_image_create(CTT_TEXTURE_KIND_TEXTURE2D);
    size_t layer = 0;
    if (ctt_image_add_layer(img, &layer) != CTT_STATUS_OK) return 2;
    if (ctt_image_push_mip(img, layer, s) != CTT_STATUS_OK) return 3;
    /* `s` consumed; do NOT destroy. */

    ctt_convert_settings settings = ctt_convert_settings_default();
    settings.container = (ctt_container){
        .tag = CTT_CONTAINER_RAW,
    };

    ctt_pipeline_output *out = NULL;
    ctt_status st = ctt_convert(img, &settings, &out);
    /* `img` consumed regardless of result. */
    if (st != CTT_STATUS_OK) {
        fprintf(stderr, "ctt_convert failed (%d): %s\n", st, ctt_last_error_message());
        return 4;
    }

    if (ctt_pipeline_output_get_kind(out) != CTT_PIPELINE_OUTPUT_KIND_RAW) {
        fprintf(stderr, "expected raw output\n");
        ctt_pipeline_output_destroy(out);
        return 5;
    }

    ctt_image *raw = ctt_pipeline_output_take_image(out);
    if (!raw) {
        ctt_pipeline_output_destroy(out);
        return 6;
    }
    if (ctt_image_layer_count(raw) != 1
        || ctt_image_mip_count(raw, 0) != 1
        || ctt_image_surface_data_len(raw, 0, 0) != 4) {
        ctt_image_destroy(raw);
        ctt_pipeline_output_destroy(out);
        return 7;
    }
    const uint8_t *bytes = ctt_image_surface_data(raw, 0, 0);
    if (memcmp(bytes, pixel, 4) != 0) {
        ctt_image_destroy(raw);
        ctt_pipeline_output_destroy(out);
        return 8;
    }
    ctt_image_destroy(raw);
    ctt_pipeline_output_destroy(out);
    printf("ok\n");
    return 0;
}
