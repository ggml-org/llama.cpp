#include "ggml-profiler.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <time.h>
#    include <unistd.h>
#endif

//
// Time utilities
//

uint64_t ggml_profiler_time_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (uint64_t) (count.QuadPart * 1000000000ULL / freq.QuadPart);
#elif defined(CLOCK_MONOTONIC_RAW)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#endif
}

//
// Backend profiler registration
//

void ggml_backend_set_profiler(ggml_backend_t backend, ggml_backend_profiler_t profiler) {
    if (backend == NULL) {
        return;
    }

    // Free any existing profiler
    if (backend->profiler != NULL) {
        if (backend->profiler->free_context != NULL) {
            backend->profiler->free_context(backend->profiler->context);
        }
        delete backend->profiler;
        backend->profiler = NULL;
    }

    backend->profiler = profiler;
}

ggml_backend_profiler_t ggml_backend_get_profiler(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->profiler;
}
