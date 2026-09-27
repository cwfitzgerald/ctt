/*
 * Error-path coverage:
 *   1. A buffer carrying the KTX2 magic but a garbage body is detected as a
 *      container and then fails to decode: ctt_decode_container must report a
 *      negative status and leave a non-empty error message.
 *   2. ctt_split_cubemap_cross with a NULL surface must fail with
 *      CTT_STATUS_NULL_POINTER, and with a zero format must fail with
 *      CTT_STATUS_INVALID_ARGUMENT. Neither writes an output image.
 *   3. ctt_image_create with a format that disagrees with the color space
 *      must return NULL.
 */
#include "../include/ctt.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const uint8_t KTX2_MAGIC[12] = {
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

static ctt_surface *make_face(void) {
    uint8_t px[4] = {10, 20, 30, 255};
    return ctt_surface_create(
        px, sizeof px,
        1, 1, 1,
        4, 0);
}

int main(void) {
    /* --- 1. Garbage-but-recognized container. --- */
    uint8_t garbage[32];
    memset(garbage, 0, sizeof garbage);
    memcpy(garbage, KTX2_MAGIC, sizeof KTX2_MAGIC); /* valid magic, junk body */

    ctt_clear_last_error();
    ctt_image *decoded = (ctt_image *)0x1; /* poison: must be overwritten or unused */
    bool recognized = false;
    ctt_status st = ctt_decode_container(
        garbage, sizeof garbage, NULL, &decoded, &recognized);
    if (st >= 0) {
        fprintf(stderr, "expected negative status decoding garbage, got %d\n", st);
        return 1;
    }
    const char *msg = ctt_last_error_message();
    if (!msg || msg[0] == '\0') {
        fprintf(stderr, "expected a non-empty error message after failed decode\n");
        return 2;
    }

    /* --- 2. Cubemap split with a NULL surface or a zero format. --- */
    ctt_format_desc desc = {
        CTT_FORMAT_R8G8B8A8_UNORM, CTT_COLOR_SPACE_LINEAR, CTT_ALPHA_MODE_OPAQUE};
    ctt_image *cube = NULL;
    ctt_clear_last_error();
    st = ctt_split_cubemap_cross(NULL, desc, &cube);
    if (st != CTT_STATUS_NULL_POINTER || cube != NULL) {
        fprintf(stderr, "expected NULL_POINTER for a NULL surface, got %d\n", st);
        return 3;
    }
    msg = ctt_last_error_message();
    if (!msg || msg[0] == '\0') {
        fprintf(stderr, "expected a non-empty error message for a NULL surface\n");
        return 4;
    }

    ctt_surface *face = make_face();
    if (!face) {
        fprintf(stderr, "make_face failed: %s\n", ctt_last_error_message());
        return 5;
    }
    desc.format = 0;
    st = ctt_split_cubemap_cross(face, desc, &cube);
    ctt_surface_destroy(face); /* not consumed by the split */
    if (st != CTT_STATUS_INVALID_ARGUMENT || cube != NULL) {
        fprintf(stderr, "expected INVALID_ARGUMENT for a zero format, got %d\n", st);
        return 6;
    }

    /* --- 3. Image with a format that disagrees with the color space. --- */
    ctt_format_desc mismatched = {
        CTT_FORMAT_R8G8B8A8_UNORM, CTT_COLOR_SPACE_SRGB, CTT_ALPHA_MODE_OPAQUE};
    if (ctt_image_create(CTT_TEXTURE_KIND_TEXTURE2D, mismatched) != NULL) {
        fprintf(stderr, "expected NULL for a format that disagrees with the color space\n");
        return 7;
    }

    printf("ok\n");
    return 0;
}
