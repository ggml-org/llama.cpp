#pragma once

// happens-before instrumentation for the ggml scheduler
//
// GGML_SCHED_SANITIZE=1  detect happens-before violations
// GGML_SCHED_SANITIZE=2  also trace synchronization edges and memory ranges

#include "ggml-backend.h"

struct ggml_backend_sched_backend {
    ggml_backend_t backend;

    void synchronize() const;
    void event_record(ggml_backend_event_t event) const;
    void event_wait(ggml_backend_event_t event) const;

    void tensor_set_async(ggml_tensor * tensor, const void * data, size_t offset, size_t size) const;
    void tensor_get_async(const ggml_tensor * tensor, void * data, size_t offset, size_t size) const;
    bool copy_tensor_async(ggml_backend_sched_backend src_backend, const ggml_tensor * src, ggml_tensor * dst) const;
    ggml_status graph_compute_async(ggml_cgraph * graph) const;

    static void copy_tensor(const ggml_tensor * src, ggml_tensor * dst);
    static void event_synchronize(ggml_backend_event_t event);
};

void ggml_san_buffer_free(ggml_backend_buffer_t buffer);
void ggml_san_split(int split_id, ggml_backend_t backend, int n_inputs);
