/*
 * Exercise the logging API: install a callback with a user_data context,
 * drive a conversion (which makes ctt's core emit records), and assert the
 * callback saw messages, that the level filter is honored, and that clearing
 * the callback stops delivery.
 */
#include "../include/ctt.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct log_sink {
    int count;
    ctt_log_level max_seen; /* most verbose level observed */
};

static void on_log(ctt_log_level level, const char *message, void *user_data) {
    struct log_sink *sink = (struct log_sink *)user_data;
    sink->count += 1;
    if (level > sink->max_seen) {
        sink->max_seen = level;
    }
    if (!message) {
        sink->count = -1; /* contract violation: message is never NULL */
    }
}

/* Run a throwaway RAW conversion so the core emits log records. */
static int drive_conversion(void) {
    uint8_t pixel[4] = {10, 20, 30, 255};
    ctt_surface *s = ctt_surface_create(
        pixel, 4,
        1, 1, 1,
        4, 0,
        CTT_FORMAT_R8G8B8A8_UNORM,
        CTT_COLOR_SPACE_LINEAR,
        CTT_ALPHA_MODE_OPAQUE);
    if (!s) return -1;

    ctt_image *img = ctt_image_create(CTT_TEXTURE_KIND_TEXTURE2D);
    size_t layer = 0;
    if (ctt_image_add_layer(img, &layer) != CTT_STATUS_OK) {
        ctt_surface_destroy(s);
        ctt_image_destroy(img);
        return -1;
    }
    if (ctt_image_push_mip(img, layer, s) != CTT_STATUS_OK) {
        ctt_image_destroy(img);
        return -1;
    }

    ctt_convert_settings cfg = ctt_convert_settings_default();
    cfg.container = (ctt_container){.tag = CTT_CONTAINER_RAW};

    ctt_pipeline_output *out = NULL;
    ctt_status st = ctt_convert(img, &cfg, &out); /* consumes img */
    if (st != CTT_STATUS_OK) {
        fprintf(stderr, "ctt_convert failed (%d): %s\n", st, ctt_last_error_message());
        return -1;
    }
    ctt_pipeline_output_destroy(out);
    return 0;
}

int main(void) {
    struct log_sink sink = {0, CTT_LOG_LEVEL_OFF};

    /* Deliver everything down to debug, routed through our user_data. */
    ctt_set_log_callback(on_log, &sink);
    ctt_set_log_level(CTT_LOG_LEVEL_DEBUG);

    if (drive_conversion() != 0) return 1;

    if (sink.count < 0) {
        fprintf(stderr, "callback received a NULL message\n");
        return 2;
    }
    if (sink.count == 0) {
        fprintf(stderr, "expected at least one log record, got none\n");
        return 3;
    }
    if (sink.max_seen > CTT_LOG_LEVEL_DEBUG) {
        fprintf(stderr, "received a record more verbose than the debug threshold\n");
        return 4;
    }

    /* Turning logging off must silence delivery. */
    sink.count = 0;
    ctt_set_log_level(CTT_LOG_LEVEL_OFF);
    if (drive_conversion() != 0) return 5;
    if (sink.count != 0) {
        fprintf(stderr, "expected no records while OFF, got %d\n", sink.count);
        return 6;
    }

    /* Clearing the callback must also stop delivery even with logging on. */
    ctt_set_log_callback(NULL, NULL);
    ctt_set_log_level(CTT_LOG_LEVEL_TRACE);
    sink.count = 0;
    if (drive_conversion() != 0) return 7;
    if (sink.count != 0) {
        fprintf(stderr, "expected no records after clearing callback, got %d\n", sink.count);
        return 8;
    }

    printf("ok\n");
    return 0;
}
