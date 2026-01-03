#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <vector>
#include <memory>
#include <future>

#if defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
#endif

#define GGML_BLAS_NAME    "BLAS"
#define GGML_BLAS_VERSION GGML_BACKEND_API_VERSION

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend_blas_buffer {
    void * data;  // dequantized data
    size_t size;  // ggml_nelements * sizeof(float)
};

struct ggml_backend_blas_buffer_context {
    void * data;
    size_t size;
    std::vector<ggml_backend_blas_buffer *> buffers;

    ~ggml_backend_blas_buffer_context() {
        ggml_aligned_free(data, size);
        for (auto * extra : buffers) {
            ggml_aligned_free(extra->data, extra->size);
            delete extra;
        }
    }
};

struct ggml_backend_blas_buffer_type_context {
    int n_threads;

#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
};

struct ggml_backend_blas_context {
    int n_threads;
};

struct ggml_backend_blas_device_context {
    char _dummy;  // Prevent empty struct warning
};

#ifdef __cplusplus
}
#endif
