#pragma once

#include "ggml.h"

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_QNN_NAME "QNN"
#define GGML_QNN_MAX_DEVICES QNN_BACKEND_COUNT

enum QNNBackend {
    QNN_BACKEND_CPU = 0,
    QNN_BACKEND_GPU,
    QNN_BACKEND_NPU,
    QNN_BACKEND_COUNT,
};

GGML_API bool ggml_backend_is_qnn(ggml_backend_t backend);

GGML_API ggml_backend_reg_t ggml_backend_qnn_reg(void);

#ifdef __cplusplus
}
#endif
