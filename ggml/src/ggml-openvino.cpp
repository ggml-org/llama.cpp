#include "ggml-openvino.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

// backend API
GGML_API ggml_backend_t ggml_backend_openvino_init(int device) {}

GGML_API bool ggml_backend_is_openvino(ggml_backend_t backend) {}

// device buffer
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_buffer_type(int device) {}

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_split_buffer_type(const float *tensor_split) {}

// pinned host buffer for use with the CPU backend for faster copies between CPU
// and GPU
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_host_buffer_type(void) {}

GGML_API ggml_backend_reg_t ggml_backend_openvino_reg(void) {}
