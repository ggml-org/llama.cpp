#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#ifdef  __cplusplus
extern "C" {
#endif

// AMD XDNA (Ryzen AI NPU) backend
//
// A narrow, CPU-cooperating backend: it uses host memory buffers and only claims
// the ops it has precompiled NPU kernels for (currently large MUL_MATs); everything
// else stays on the CPU backend via the ggml scheduler's normal graph splitting.
//
// Kernels are looked up in $GGML_XDNA_KERNEL_DIR (or the build-time default).

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_xdna_init(void);

GGML_BACKEND_API bool ggml_backend_is_xdna(ggml_backend_t backend);

// number of host threads used for converting/staging operands
GGML_BACKEND_API void ggml_backend_xdna_set_n_threads(ggml_backend_t backend_xdna, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);


#ifdef  __cplusplus
}
#endif
