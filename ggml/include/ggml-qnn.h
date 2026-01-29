#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API ggml_backend_reg_t ggml_backend_qnn_reg(void);

#ifdef __cplusplus
}
#endif
