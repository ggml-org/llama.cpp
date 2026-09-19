#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_fakegpu_buffer_type(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_fakegpu_reg(void);

#ifdef __cplusplus
}
#endif
