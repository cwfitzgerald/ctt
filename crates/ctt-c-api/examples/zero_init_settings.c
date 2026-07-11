/*
 * Zero-initialization contract: a `memset(0)` ctt_convert_settings must behave
 * exactly like ctt_convert_settings_default(). We run the same input through
 * ctt_convert twice — once with each settings value — and assert the encoded
 * outputs are byte-for-byte identical.
 */
#include "../include/ctt.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static ctt_image *make_image(void) {
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
    if (!s) return NULL;
    ctt_image *img = ctt_image_create(CTT_TEXTURE_KIND_TEXTURE2D);
    size_t layer = 0;
    if (ctt_image_add_layer(img, &layer) != CTT_STATUS_OK) {
        ctt_surface_destroy(s);
        ctt_image_destroy(img);
        return NULL;
    }
    if (ctt_image_push_mip(img, layer, s) != CTT_STATUS_OK) {
        ctt_image_destroy(img);
        return NULL;
    }
    return img;
}

/* Convert `img` (consumed) with `cfg`, returning the encoded bytes copied into
 * a freshly malloc'd buffer via `*out_buf` / `*out_len`. Returns 0 on success. */
static int convert_encoded(ctt_image *img, const ctt_convert_settings *cfg,
                           uint8_t **out_buf, size_t *out_len) {
    ctt_pipeline_output *out = NULL;
    ctt_status st = ctt_convert(img, cfg, &out);
    if (st != CTT_STATUS_OK) {
        fprintf(stderr, "ctt_convert failed (%d): %s\n", st, ctt_last_error_message());
        return -1;
    }
    if (ctt_pipeline_output_get_kind(out) != CTT_PIPELINE_OUTPUT_KIND_ENCODED) {
        fprintf(stderr, "expected encoded output\n");
        ctt_pipeline_output_destroy(out);
        return -1;
    }
    const uint8_t *bytes = ctt_pipeline_output_encoded_data(out);
    size_t len = ctt_pipeline_output_encoded_len(out);
    uint8_t *copy = (uint8_t *)malloc(len ? len : 1);
    if (!copy) {
        ctt_pipeline_output_destroy(out);
        return -1;
    }
    memcpy(copy, bytes, len);
    ctt_pipeline_output_destroy(out);
    *out_buf = copy;
    *out_len = len;
    return 0;
}

int main(void) {
    ctt_convert_settings zeroed;
    memset(&zeroed, 0, sizeof zeroed);
    ctt_convert_settings defaults = ctt_convert_settings_default();

    ctt_image *img_a = make_image();
    ctt_image *img_b = make_image();
    if (!img_a || !img_b) {
        fprintf(stderr, "make_image failed: %s\n", ctt_last_error_message());
        if (img_a) {
            ctt_image_destroy(img_a);
        }
        if (img_b) {
            ctt_image_destroy(img_b);
        }
        return 1;
    }

    uint8_t *buf_zero = NULL, *buf_def = NULL;
    size_t len_zero = 0, len_def = 0;
    if (convert_encoded(img_a, &zeroed, &buf_zero, &len_zero) != 0) {
        ctt_image_destroy(img_b);
        return 2;
    }
    if (convert_encoded(img_b, &defaults, &buf_def, &len_def) != 0) {
        free(buf_zero);
        return 3;
    }

    int rc = 0;
    if (len_zero != len_def || memcmp(buf_zero, buf_def, len_zero) != 0) {
        fprintf(stderr,
                "zero-init output (%zu bytes) differs from default output (%zu bytes)\n",
                len_zero, len_def);
        rc = 4;
    }

    free(buf_zero);
    free(buf_def);
    if (rc == 0) printf("ok\n");
    return rc;
}
