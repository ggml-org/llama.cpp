#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_REMOTING_BACKEND_NAME "RemotingBackend"

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_remoting_backend_reg();

#ifdef  __cplusplus
}
#endif
