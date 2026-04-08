/**
 * llama-turboquant.h — quant.cpp (turboquant) integration for llama.cpp
 *
 * Provides GGML-compatible quantize/dequantize/vec_dot wrappers for all
 * quant.cpp KV cache quantization types.
 *
 * Guarded by LLAMA_TURBOQUANT — only available when built with turboquant.
 */
#pragma once

#ifdef LLAMA_TURBOQUANT

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Register quant.cpp types with GGML.
 * Populates to_float / from_float_ref in the base type_traits table.
 * Call once during llama_backend_init().
 */
void llama_turboquant_init(void);

/**
 * Return the ggml_type for a quant.cpp CLI name, or GGML_TYPE_COUNT on failure.
 * Accepts aliases like "turbo3", "tq-uniform-4b", "uniform_4b", etc.
 */
enum ggml_type llama_turboquant_type_from_str(const char * name);

/**
 * Print all available quant.cpp KV cache types to stderr.
 */
void llama_turboquant_print_types(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TURBOQUANT
