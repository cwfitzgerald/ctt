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
    /* Round-trip a 1x1 RGBA8 pixel through ctt_convert_raw. */
    uint8_t pixel[4] = {200, 120, 60, 255};
    ctt_surface *s = ctt_surface_create(
        pixel, 4,
        1, 1, 1,
        4, 0);
    if (!s) {
        fprintf(stderr, "ctt_surface_create failed: %s\n", ctt_last_error_message());
        return 1;
    }

    ctt_format_desc desc = {
        CTT_FORMAT_R8G8B8A8_UNORM, CTT_COLOR_SPACE_LINEAR, CTT_ALPHA_MODE_OPAQUE};
    ctt_image *img = ctt_image_create(CTT_TEXTURE_KIND_TEXTURE2D, desc);
    size_t layer = 0;
    if (ctt_image_add_layer(img, &layer) != CTT_STATUS_OK) return 2;
    if (ctt_image_push_mip(img, layer, s) != CTT_STATUS_OK) return 3;
    /* `s` consumed; do NOT destroy. */

    ctt_convert_settings settings = ctt_convert_settings_default();

    ctt_image *raw = NULL;
    ctt_status st = ctt_convert_raw(img, &settings, &raw);
    ctt_image_destroy(img); /* not consumed */
    if (st != CTT_STATUS_OK) {
        fprintf(stderr, "ctt_convert_raw failed (%d): %s\n", st, ctt_last_error_message());
        return 4;
    }

    if (ctt_image_layer_count(raw) != 1
        || ctt_image_mip_count(raw, 0) != 1
        || ctt_image_surface_data_len(raw, 0, 0) != 4) {
        ctt_image_destroy(raw);
        return 7;
    }
    const uint8_t *bytes = ctt_image_surface_data(raw, 0, 0);
    if (memcmp(bytes, pixel, 4) != 0) {
        ctt_image_destroy(raw);
        return 8;
    }

    /* Encode the raw image into a DDS file. */
    ctt_container container = {.tag = CTT_CONTAINER_DDS};
    ctt_bytes *dds = NULL;
    st = ctt_encode_container(raw, container, &dds);
    ctt_image_destroy(raw); /* not consumed */
    if (st != CTT_STATUS_OK) {
        fprintf(stderr, "ctt_encode_container failed (%d): %s\n", st, ctt_last_error_message());
        return 9;
    }
    if (ctt_bytes_len(dds) < 4 || memcmp(ctt_bytes_data(dds), "DDS ", 4) != 0) {
        fprintf(stderr, "output does not start with DDS magic\n");
        ctt_bytes_destroy(dds);
        return 10;
    }
    ctt_bytes_destroy(dds);
    printf("ok\n");
    return 0;
}
