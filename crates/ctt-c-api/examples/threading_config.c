#include "../include/ctt.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    ctt_status status = ctt_set_thread_count(2);
    if (status != CTT_STATUS_OK) {
        fprintf(stderr, "initial thread count failed: %s\n", ctt_last_error_message());
        return 1;
    }

    status = ctt_set_thread_count(0);
    if (status != CTT_STATUS_THREAD_POOL_ALREADY_INITIALIZED) {
        fprintf(stderr, "repeated setter returned %d\n", status);
        return 2;
    }
    if (strlen(ctt_last_error_message()) == 0) {
        fprintf(stderr, "repeated setter did not set the last error\n");
        return 3;
    }

    return 0;
}
