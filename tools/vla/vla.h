#pragma once

#include "llama.h"

#include <stdbool.h>
#include <stdint.h>

#if defined(VLA_SHARED)
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef VLA_BUILD
#            define VLA_API __declspec(dllexport)
#        else
#            define VLA_API __declspec(dllimport)
#        endif
#    else
#        define VLA_API __attribute__((visibility("default")))
#    endif
#else
#    define VLA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct vla_context;

typedef struct vla_context vla_context;

struct vla_context_params {
    bool    use_gpu;
    int32_t n_threads;
};

struct vla_input {
    // Per-token final hidden states in row-major [n_tokens, n_embd].
    const float * embeddings;
    int64_t       n_tokens;
    int64_t       n_embd;

    // Robot proprioception in [state_dim].
    const float * state;
    int64_t       n_state;

    // Optional row-major [horizon, action_dim] denoising noise.
    const float * noise;
    int64_t       n_noise;

    int32_t embodiment_id;
};

struct vla_output {
    // Caller-owned row-major [horizon, action_dim] buffer.
    float * actions;
    int64_t capacity;
};

VLA_API struct vla_context_params vla_context_params_default(void);

// The text model is optional. When supplied, its output embedding dimension is
// checked against the VLA conditioning dimension during initialization.
VLA_API struct vla_context * vla_init_from_file(
        const char *                    path,
        const struct llama_model *      text_model,
        struct vla_context_params       params);

VLA_API void vla_free(struct vla_context * ctx);

VLA_API const char * vla_model_type(const struct vla_context * ctx);
VLA_API int64_t vla_state_dim(const struct vla_context * ctx);
VLA_API int64_t vla_action_dim(const struct vla_context * ctx);
VLA_API int64_t vla_action_horizon(const struct vla_context * ctx);
VLA_API int64_t vla_conditioning_dim(const struct vla_context * ctx);
VLA_API int64_t vla_n_embodiments(const struct vla_context * ctx);

VLA_API bool vla_predict(
        struct vla_context *      ctx,
        const struct vla_input *  input,
        struct vla_output *       output);

#ifdef __cplusplus
}
#endif
