#define _GNU_SOURCE
#include "ggml-backend.h"
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static void (*real_set)(ggml_backend_t, struct ggml_tensor *, const void *, size_t, size_t);
static enum ggml_status (*real_graph)(ggml_backend_sched_t, struct ggml_cgraph *);
static _Thread_local unsigned in_graph;
static _Atomic unsigned long calls, graph_calls, graph_bytes;

__attribute__((constructor)) static void init_trace(void) {
    real_set = dlsym(RTLD_NEXT, "ggml_backend_tensor_set_async");
    real_graph = dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async");
    if (!real_set || !real_graph) {
        fputs("[xe-copy-trace] missing ggml symbols\n", stderr);
        abort();
    }
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    ++in_graph;
    enum ggml_status status = real_graph(sched, graph);
    --in_graph;
    return status;
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor,
                                 const void * data, size_t offset, size_t size) {
    unsigned long n = atomic_fetch_add(&calls, 1);
    unsigned long g = 0;
    if (in_graph) {
        g = atomic_fetch_add(&graph_calls, 1);
        atomic_fetch_add(&graph_bytes, size);
    }
    if ((!in_graph && n < 4) || (in_graph && g < 64)) {
        fprintf(stderr, "[xe-copy-trace] graph=%u name=%.*s type=%d ne=%ld,%ld,%ld,%ld offset=%zu bytes=%zu\n",
                in_graph, GGML_MAX_NAME, tensor->name, tensor->type,
                (long)tensor->ne[0], (long)tensor->ne[1], (long)tensor->ne[2], (long)tensor->ne[3], offset, size);
    }
    real_set(backend, tensor, data, offset, size);
}

__attribute__((destructor)) static void finish_trace(void) {
    fprintf(stderr, "[xe-copy-trace] total=%lu graph_calls=%lu graph_bytes=%lu\n",
            atomic_load(&calls), atomic_load(&graph_calls), atomic_load(&graph_bytes));
}
