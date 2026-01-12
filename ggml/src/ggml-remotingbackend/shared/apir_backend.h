#pragma once

#include "apir_backend.gen.h"

#include <stdint.h>  // for uintptr_t
#include <time.h>    // for struct timespec, clock_gettime

#define APIR_BACKEND_INITIALIZE_SUCCESS                     0
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY    2
#define APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS     3
#define APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS        4

#define APIR_BACKEND_INITIALIZE_BACKEND_FAILED 5
// new entries here need to be added to the apir_backend_initialize_error function below

#define APIR_BACKEND_FORWARD_INDEX_INVALID 6

// 0 is fast, 1 avoids the backend to crash if an unsupported tensor is received
#define APIR_BACKEND_CHECK_SUPPORTS_OP 0

typedef uintptr_t apir_buffer_type_host_handle_t;
typedef uintptr_t apir_buffer_host_handle_t;

struct virgl_opaque_context;

struct virgl_apir_callbacks {
    void * (*get_shmem_ptr)(struct virgl_opaque_context * ctx, uint32_t res_id);
};

struct virgl_apir_context {
    struct virgl_opaque_context * virgl_ctx;

    struct virgl_apir_callbacks iface;
};

struct timer_data {
    long long    start;
    long long    total;
    long long    count;
    const char * name;
};

extern struct timer_data graph_compute_timer;
extern struct timer_data get_tensor_timer;
extern struct timer_data set_tensor_timer;
extern struct timer_data cpy_tensor_timer;
extern struct timer_data wait_host_reply_timer;
extern struct timer_data get_tensor_from_ptr_timer;
extern struct timer_data set_tensor_from_ptr_timer;

static inline void start_timer(struct timer_data * timer) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
static inline long long stop_timer(struct timer_data * timer) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->count += 1;

    return duration;
}

static inline void show_timer(struct timer_data * timer) {
    double ms    = timer->total / 1000000;
    double itl   = ms / timer->count;
    double speed = 1 / itl * 1000;

    if (!timer->total) {
        return;
    }

    INFO("%15s [%9.0f] ms for %4ld invocations | ITL %2.2f ms | throughput = %4.2f t/s (%4.2f ms/call)", timer->name,
         ms, timer->count, itl, speed, ms / timer->count);
}

static const char * apir_backend_initialize_error(int code) {
#define APIR_BACKEND_INITIALIZE_ERROR(code_name) \
    do {                                         \
        if (code == code_name)                   \
            return #code_name;                   \
    } while (0)

    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_SUCCESS);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_BACKEND_FAILED);

    return "Unknown APIR_BACKEND_INITIALIZE error:/";

#undef APIR_BACKEND_INITIALIZE_ERROR
}
