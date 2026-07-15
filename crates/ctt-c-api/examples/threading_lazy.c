#include "../include/ctt.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    uint8_t pixel[4] = {200, 120, 60, 255};
    ctt_surface *surface = ctt_surface_create(
        pixel, sizeof(pixel), 1, 1, 1, 4, 0,
        CTT_FORMAT_R8G8B8A8_UNORM,
        CTT_COLOR_SPACE_LINEAR,
        CTT_ALPHA_MODE_OPAQUE);
    if (surface == NULL) {
        fprintf(stderr, "surface creation failed: %s\n", ctt_last_error_message());
        return 1;
    }

    ctt_image *image = ctt_image_create(CTT_TEXTURE_KIND_TEXTURE2D);
    size_t layer = 0;
    if (image == NULL || ctt_image_add_layer(image, &layer) != CTT_STATUS_OK) {
        ctt_surface_destroy(surface);
        return 2;
    }
    if (ctt_image_push_mip(image, layer, surface) != CTT_STATUS_OK) {
        ctt_surface_destroy(surface);
        ctt_image_destroy(image);
        return 3;
    }

    ctt_convert_settings settings = ctt_convert_settings_default();
    settings.container = (ctt_container){.tag = CTT_CONTAINER_RAW};
    ctt_pipeline_output *output = NULL;
    ctt_status status = ctt_convert(image, &settings, &output);
    if (status != CTT_STATUS_OK) {
        fprintf(stderr, "conversion failed: %s\n", ctt_last_error_message());
        return 4;
    }
    ctt_pipeline_output_destroy(output);

    status = ctt_set_thread_count(2);
    if (status != CTT_STATUS_THREAD_POOL_ALREADY_INITIALIZED) {
        fprintf(stderr, "late setter returned %d\n", status);
        return 5;
    }
    if (strlen(ctt_last_error_message()) == 0) {
        fprintf(stderr, "late setter did not set the last error\n");
        return 6;
    }

    return 0;
}
