#include "shared/apir_backend.h"

static inline apir_buffer_handle_t
ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer) {
  // in the backend, the buffer handle is the buffer pointer
  return (apir_buffer_handle_t) buffer;
}
