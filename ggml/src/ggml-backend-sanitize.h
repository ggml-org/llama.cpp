#pragma once

// happens-before instrumentation for the ggml backend API
//
// GGML_SCHED_SANITIZE=1  detect happens-before violations
// GGML_SCHED_SANITIZE=2  also trace synchronization edges and memory ranges

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

    int  ggml_san_level(void);

    void ggml_san_sync        (ggml_backend_t backend);
    void ggml_san_event_record(ggml_backend_event_t event, ggml_backend_t backend);
    void ggml_san_event_wait  (ggml_backend_t backend, ggml_backend_event_t event);
    void ggml_san_event_sync  (ggml_backend_event_t event);

    void ggml_san_compute     (ggml_backend_t backend, const struct ggml_cgraph * cgraph);

    // backend == NULL means the access is performed by the host thread
    void ggml_san_access      (ggml_backend_t backend, const struct ggml_tensor * tensor,
                               size_t offset, size_t size, bool write, const char * what);

    void ggml_san_cpy_async   (ggml_backend_t src_be, ggml_backend_t dst_be,
                               const struct ggml_tensor * src, const struct ggml_tensor * dst);

    void ggml_san_buffer_free (ggml_backend_buffer_t buffer);

    void ggml_san_split       (int split_id, ggml_backend_t backend, int n_inputs);

#ifdef __cplusplus
}
#endif
