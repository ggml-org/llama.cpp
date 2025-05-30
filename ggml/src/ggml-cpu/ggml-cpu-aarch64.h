#pragma once

#include "ggml-cpu-traits.h"
#include "ggml.h"

// GGML internal header

ggml_backend_buffer_type_t ggml_backend_cpu_aarch64_buffer_type(void);

#ifdef __cplusplus
extern "C" {
#endif

#if defined(GGML_YIELD_BARRIER)
size_t ggml_barrier_spin_count(unsigned int n_threads);
#endif

#ifdef __cplusplus
}
#endif
