/*
 * Error-path coverage:
 *   1. A buffer carrying the KTX2 magic but a garbage body is detected as a
 *      container and then fails to decode: ctt_decode_container must report a
 *      negative status and leave a non-empty error message.
 *   2. ctt_cubemap_input_separate_faces with a NULL face must fail (returning
 *      NULL) while still consuming the non-NULL faces — this exercises the
 *      "all six consumed on failure" contract. We cannot observe the freeing
 *      directly from C, but running under a leak checker would.
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
        4, 0,
        CTT_FORMAT_R8G8B8A8_UNORM,
        CTT_COLOR_SPACE_LINEAR,
        CTT_ALPHA_MODE_OPAQUE);
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

    /* --- 2. separate-faces with a NULL face. --- */
    ctt_surface *faces[6];
    for (int i = 0; i < 6; ++i) {
        faces[i] = make_face();
        if (!faces[i]) {
            fprintf(stderr, "make_face[%d] failed: %s\n", i, ctt_last_error_message());
            for (int j = 0; j < i; ++j) ctt_surface_destroy(faces[j]);
            return 3;
        }
    }
    /* Poke a hole: face 2 is NULL. The other five must still be consumed. */
    ctt_surface_destroy(faces[2]);
    faces[2] = NULL;

    ctt_clear_last_error();
    ctt_cubemap_input *ci = ctt_cubemap_input_separate_faces(faces);
    if (ci != NULL) {
        fprintf(stderr, "expected NULL cubemap input for a NULL face\n");
        ctt_cubemap_input_destroy(ci);
        return 4;
    }
    msg = ctt_last_error_message();
    if (!msg || msg[0] == '\0') {
        fprintf(stderr, "expected a non-empty error message for NULL face\n");
        return 5;
    }
    /* faces[0,1,3,4,5] were consumed by the (failed) call; do NOT destroy. */

    printf("ok\n");
    return 0;
}
