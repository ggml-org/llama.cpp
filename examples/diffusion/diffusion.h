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
    bool                      suppress_mask_token     = false;  // forbid revealing a position as the mask token
                                                                // (masked-diffusion models that can emit it)
    bool                      self_conditioning       = false;  // feed each step's canvas logits back into the
                                                                // next step (DiffusionGemma; no-op for others)

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

// Entropy-bound denoiser for block-diffusion canvas models (DiffusionGemma). Unlike the masked path, the
// canvas is random-initialized and non-accepted positions are renoised each step; tokens are accepted by a
// per-position entropy (mutual-information) bound, under a linear temperature schedule, with adaptive
// stopping. Writes the final argmax canvas into output_tokens[n_input .. max_length).
struct diffusion_eb_params {
    int32_t max_denoising_steps  = 48;
    float   t_min                = 0.4f;   // temperature at the last step
    float   t_max                = 0.8f;   // temperature at the first step
    float   entropy_bound        = 0.1f;   // accept lowest-entropy tokens within this MI bound
    int32_t stability_threshold  = 1;      // steps the argmax canvas must hold to count as stable
    float   confidence_threshold = 0.005f; // stop once mean canvas entropy drops below this
    int32_t seed                 = 0;
    int32_t max_length           = 0;      // n_input + canvas_length
    bool    kv_cache             = false;  // prefix-KV-cache the prompt (PREFILL once, decode canvas-only
                                           // per step) instead of re-decoding [prompt|canvas] every step
    bool    gpu_sampling         = false;  // device-resident self-conditioning: keep the prev step's canvas
                                           // logits on-device for SC instead of a per-step 268 MB host upload
                                           // (exact; the SC math/values are unchanged)
    bool    gpu_sample_reduce    = false;  // Stage-1: argmax/entropy/one multinomial draw per position done on
                                           // the GPU from sc_dev (skips the 268 MB logits D2H + host reductions).
                                           // Requires gpu_sampling. FP-equivalent: argmax exact, Z/entropy ~1e-4.

    diffusion_step_callback_t step_callback           = nullptr;
    void *                    step_callback_user_data = nullptr;
    bool                      visual_mode             = false;
};

void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated);
