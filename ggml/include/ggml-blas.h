#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_blas_init(void);

GGML_API GGML_CALL bool ggml_backend_is_blas(ggml_backend_t backend);

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
GGML_API GGML_CALL void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads);


#ifdef  __cplusplus
}
#endif
