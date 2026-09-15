#pragma once

#include "backend-context.h"
#include "ggml-backend.h"
#include "graph/value-map.h"

#include <cstddef>

struct ggml_tensor;

ggml_backend_hrx_buffer_context * ggml_backend_hrx_buffer_context_from_buffer(ggml_backend_buffer_t buffer);
size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor);
void * ggml_backend_hrx_buffer_base(ggml_backend_buffer_t buffer);

bool ggml_backend_hrx_tensor_binding(const ggml_tensor *                tensor,
                                     ggml_backend_hrx_buffer_context ** out_context,
                                     size_t *                           out_offset);
bool ggml_backend_hrx_resolve_value_buffer(const ggml_tensor * tensor, ggml::hrx::ValueBufferBinding & binding);
