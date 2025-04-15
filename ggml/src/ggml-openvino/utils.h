#include "ggml-decoder.h"
#include "ggml-backend-impl.h"

enum ggml_status openvino_frontend_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph, const int32_t start_index=0, const int32_t end_index=0);
