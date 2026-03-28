#pragma once

#include "turboquant_impl.h"
#include "../../ggml/include/ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for the custom op builder
ggml_tensor * ggml_turboquant_transform(ggml_context * ctx, ggml_tensor * input, turboquant::TurboQuantKVCache * tq);

// Forward declaration for custom op builder with quantized buffer
ggml_tensor * ggml_turboquant_transform_with_buffer(
    ggml_context * ctx, 
    ggml_tensor * input, 
    turboquant::TurboQuantKVCache * tq,
    uint8_t * quantized_buffer,
    size_t bytes_per_vector);

#ifdef __cplusplus
}
#endif