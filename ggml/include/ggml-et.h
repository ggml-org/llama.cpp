#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_ET_NAME "ET"

// backend API
GGML_BACKEND_API ggml_guid_t     ggml_backend_et_guid(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_et_init(size_t devidx);

GGML_BACKEND_API bool ggml_backend_is_et(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_et_get_device_count(void);
GGML_BACKEND_API void ggml_backend_et_get_device_description(int devidx, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_et_get_device_memory(int devidx, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_et_buffer_type(size_t dev_num);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_et_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_et_reg(void);

#ifdef  __cplusplus
}
#endif
