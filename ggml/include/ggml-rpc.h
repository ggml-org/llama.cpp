#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_rpc_init(const char * endpoint);
GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint);

GGML_API GGML_CALL void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total);

GGML_API GGML_CALL void start_rpc_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem);

#ifdef  __cplusplus
}
#endif
