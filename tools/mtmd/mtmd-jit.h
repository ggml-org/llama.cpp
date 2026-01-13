#ifndef MTMD_JIT_H
#define MTMD_JIT_H

// NOTE: This header intentionally does NOT define MTMD_INTERNAL_HEADER
// It provides declarations for JIT-related internal functions that are
// needed by mtmd-helper.cpp but should not be part of the public API.
// These functions are defined in mtmd.cpp.

#include "llama.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mtmd_context;
struct mtmd_input_chunk;
struct mtmd_input_chunks;

// Retrieve the JIT-initialized llama context if available (NULL if not set)
struct llama_context * mtmd_get_llm_context(mtmd_context * ctx);

// Whether pre-encode/JIT-llm flow is enabled (clip_reduced_vram)
bool mtmd_preencode_enabled(mtmd_context * ctx);

// Pre-encode the image chunk; returns 0 on success, or encode error
int32_t mtmd_preencode_image(mtmd_context * ctx, const mtmd_input_chunks * chunks);

// Invoke the registered JIT LLM init callback if not already invoked
void mtmd_invoke_llm_init_if_needed(mtmd_context * ctx);

// Query pre-encoded image state
bool mtmd_has_preencoded_image(mtmd_context * ctx);

// Retrieve the encode timing (ms) for the media chunk's underlying encoder, returns 0 if unavailable
int64_t mtmd_get_image_encode_timing(mtmd_context * ctx, const mtmd_input_chunk * chunk);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MTMD_JIT_H

