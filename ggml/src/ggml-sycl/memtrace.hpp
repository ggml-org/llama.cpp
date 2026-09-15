#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#include <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

enum ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_FATTN_KV,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_COUNT,
};

bool ggml_sycl_memtrace_enabled();

void ggml_sycl_memtrace_add(ggml_sycl_mem_type type, const void * ptr, size_t bytes);
void ggml_sycl_memtrace_del(const void * ptr);

void ggml_sycl_memtrace_report(const char * tag);
void ggml_sycl_memtrace_report_device(const char * tag, int device, size_t dev_free, size_t dev_total);
void ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
