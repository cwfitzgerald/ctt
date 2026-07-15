#include "../include/ctt.h"
#include <stdio.h>

int main(void) {
    ctt_status status = ctt_set_thread_count(0);
    if (status != CTT_STATUS_OK) {
        fprintf(stderr, "default thread count failed: %s\n", ctt_last_error_message());
        return 1;
    }
    return 0;
}
