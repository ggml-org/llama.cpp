// Weak stubs for Perfetto tracing symbols used by ggml-vulkan.
// These allow building the Vulkan backend without linking the full
// llama Perfetto implementation. If the real implementations are
// linked elsewhere (e.g., from src/llama_perfetto.cpp), they will
// override these weak symbols.

#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define WEAK __attribute__((weak))
#else
#define WEAK
#endif

#ifdef __cplusplus
extern "C" {
#endif

WEAK void llama_perfetto_trace_begin(const char * name) {
    (void)name;
}

WEAK void llama_perfetto_trace_end(void) {
}

WEAK void llama_perfetto_gpu_begin(const char * name) {
    (void)name;
}

WEAK void llama_perfetto_gpu_end(void) {
}

#ifdef __cplusplus
}
#endif

