#include "ggml-profiler.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

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
// Record helpers
//

void ggml_profile_record_from_tensor(ggml_profile_record * rec, const struct ggml_tensor * node) {
    if (rec == NULL) {
        return;
    }

    // Output tensor info
    if (node != NULL) {
        memcpy(rec->ne, node->ne, sizeof(rec->ne));
        rec->out_type = (int) node->type;
        memcpy(rec->op_params, node->op_params, sizeof(rec->op_params));
        // Copy the tensor name (rather than aliasing node->name) so the record stays
        // valid after the graph's meta context is reused on the next build.
        snprintf(rec->tensor_name, sizeof(rec->tensor_name), "%s", node->name);
    } else {
        memset(rec->ne, 0, sizeof(rec->ne));
        rec->out_type = -1;
        memset(rec->op_params, 0, sizeof(rec->op_params));
        rec->tensor_name[0] = '\0';
    }

    // Sub-op (UNARY/GLU)
    rec->sub_op = -1;
    if (node != NULL) {
        if (node->op == GGML_OP_UNARY) {
            rec->sub_op = (int) ggml_get_unary_op(node);
        } else if (node->op == GGML_OP_GLU) {
            rec->sub_op = (int) ggml_get_glu_op(node);
        }
    }

    // Source tensors
    rec->n_src = 0;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = (node != NULL) ? node->src[i] : NULL;
        if (src == NULL) {
            memset(rec->ne_src[i], 0, sizeof(rec->ne_src[i]));
            memset(rec->nb_src[i], 0, sizeof(rec->nb_src[i]));
            rec->type_src[i] = -1;
        } else {
            memcpy(rec->ne_src[i], src->ne, sizeof(rec->ne_src[i]));
            for (int d = 0; d < 4; d++) {
                rec->nb_src[i][d] = (int64_t) src->nb[d];
            }
            rec->type_src[i] = (int) src->type;
            rec->n_src = i + 1;
        }
    }
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
