#pragma once

#include "ggml-backend.h"

// GGML internal header

#define GGML_USE_TMAC
#if defined(GGML_USE_TMAC)

#ifdef  __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_tmac_buffer_type(void);
void ggml_tmac_init(void);

#ifdef __cplusplus
}
#endif

#endif
