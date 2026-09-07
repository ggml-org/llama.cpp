#pragma once

#include "llama.h"

#include <cstdint>

enum diffusion_algorithm {
    DIFFUSION_ALGORITHM_ORIGIN           = 0,
    DIFFUSION_ALGORITHM_ENTROPY_BASED    = 1,
    DIFFUSION_ALGORITHM_MARGIN_BASED     = 2,
    DIFFUSION_ALGORITHM_RANDOM           = 3,
    DIFFUSION_ALGORITHM_CONFIDENCE_BASED = 4,
};

// Unified transfer scheduling methods
enum diffusion_transfer_schedule {
    DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED = 0,  // Dream-style: (1.0 - s/t) * remaining
    DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED    = 1,  // LLaDA-style: process in blocks with get_num_transfer_tokens
};

typedef bool (*diffusion_step_callback_t)(int32_t             step,
                                          int32_t             total_steps,
                                          const llama_token * tokens,
                                          int32_t             n_tokens,
                                          void *              user_data);

struct diffusion_params {
    int32_t                   steps                   = 0;
    float                     temperature             = 0;
    llama_token               mask_token_id           = LLAMA_TOKEN_NULL;
    diffusion_step_callback_t step_callback           = nullptr;
    void *                    step_callback_user_data = nullptr;
    int32_t                   seed                    = 0;
    bool                      visual_mode             = false;
    bool                      shift_logits            = false;  // Shift logits by -1 after decode

    float   top_p = 0.;
    int32_t top_k = 0.;

    diffusion_algorithm         algorithm = DIFFUSION_ALGORITHM_CONFIDENCE_BASED;
    diffusion_transfer_schedule schedule  = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;

    float   cfg_scale        = 0.;     // Config scale for classifier-free guidance
    float   eps              = 0.;     // Timestep scheduling
    int32_t block_length     = 0;      // Block size (for block scheduling)
    float   alg_temp         = 0;      // algorithm temperature (0.0 = deterministic)
    bool    add_gumbel_noise = false;  // Add gumbel noise to the logits if temp > 0.0

    int32_t max_length = 0;            // Maximum sequence length
};

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated);
