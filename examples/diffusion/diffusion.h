#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef bool (*diffusion_step_callback_t)(int32_t step, int32_t total_steps, const llama_token * tokens,
                                          int32_t n_tokens, void * user_data);

enum diffusion_algorithm {
    DIFFUSION_ALG_ORIGIN       = 0,
    DIFFUSION_ALG_MASKGIT_PLUS = 1,
    DIFFUSION_ALG_TOPK_MARGIN  = 2,
    DIFFUSION_ALG_ENTROPY      = 3,
};

struct diffusion_params {
    int32_t                   steps;
    float                     eps;
    float                     temperature;
    float                     top_p;
    int32_t                   top_k;
    llama_token               mask_token_id;
    enum diffusion_algorithm  algorithm;
    float                     alg_temp;
    diffusion_step_callback_t step_callback;
    void *                    step_callback_user_data;
    int32_t                   seed;
};

struct diffusion_params diffusion_default_params(void);

void diffusion_generate(llama_context * ctx, const llama_token * input_tokens, llama_token * output_tokens,
                        int32_t n_input, int32_t max_length, struct diffusion_params params, int32_t * n_generated);

#ifdef __cplusplus
}
#endif
