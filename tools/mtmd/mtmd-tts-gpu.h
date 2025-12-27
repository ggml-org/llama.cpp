/**
 * mtmd-tts-gpu.h - GPU-accelerated Code Predictor for Qwen3-Omni TTS
 *
 * The Code Predictor is a 5-layer transformer that expands 1 codebook token
 * into 16 codebook tokens for Code2Wav synthesis. This GPU implementation
 * uses ggml graphs to keep computation on GPU, eliminating CPU-GPU transfers.
 *
 * Architecture (per token):
 *   Input: past_hidden (from Talker) + last_id_hidden (codec embedding)
 *   -> 5x Transformer layers (MHA + SwiGLU FFN)
 *   -> 15x LM head projections (one per codebook 1-15)
 *   Output: 15 codebook tokens
 */

#ifndef MTMD_TTS_GPU_H
#define MTMD_TTS_GPU_H

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Code Predictor constants
#define CP_N_EMBD     1024
#define CP_N_HEAD     16
#define CP_N_HEAD_KV  8
#define CP_HEAD_DIM   128
#define CP_N_FF       3072
#define CP_N_LAYER    5
#define CP_VOCAB      2048
#define CP_N_CODEBOOKS 15
#define CP_ROPE_THETA 1000000.0f
#define CP_MAX_SEQ    32  // max sequence length (2 init + 15 codebooks + buffer)

// Opaque GPU Code Predictor context
struct mtmd_cp_gpu_context;
typedef struct mtmd_cp_gpu_context mtmd_cp_gpu_context;

/**
 * Initialize GPU Code Predictor context
 *
 * @param model       Talker model containing Code Predictor weights
 * @param n_threads   Number of CPU threads for fallback operations
 * @return            Context, or NULL on failure (falls back to CPU)
 */
mtmd_cp_gpu_context * mtmd_cp_gpu_init(
    const struct llama_model * model,
    int n_threads);

/**
 * Free GPU Code Predictor context
 */
void mtmd_cp_gpu_free(mtmd_cp_gpu_context * ctx);

/**
 * Run Code Predictor on GPU
 *
 * Generates 15 codebook tokens (codebooks 1-15) from Talker hidden states.
 *
 * @param ctx              GPU Code Predictor context
 * @param past_hidden      Hidden state from previous Talker step [CP_N_EMBD floats]
 * @param last_id_hidden   Embedding of last codec token [CP_N_EMBD floats]
 * @param codebook_tokens  Output: 15 generated tokens [CP_N_CODEBOOKS ints]
 * @param codec_embeddings Output: 16 embeddings for Code2Wav [16 * CP_N_EMBD floats]
 * @param rng_seed         RNG seed for sampling (0 = use internal)
 * @param temperature      Sampling temperature (0 = greedy)
 * @param verbose          Enable verbose logging
 * @return                 true on success
 */
bool mtmd_cp_gpu_generate(
    mtmd_cp_gpu_context * ctx,
    const float * past_hidden,
    const float * last_id_hidden,
    int * codebook_tokens,
    float * codec_embeddings,
    uint64_t rng_seed,
    float temperature,
    bool verbose);

/**
 * Check if GPU Code Predictor is available
 *
 * Returns true if the context was initialized with GPU backend.
 * If false, caller should use CPU fallback.
 */
bool mtmd_cp_gpu_available(const mtmd_cp_gpu_context * ctx);

/**
 * Reset KV cache for new sequence
 *
 * Call this before starting a new Code Predictor generation.
 */
void mtmd_cp_gpu_reset(mtmd_cp_gpu_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // MTMD_TTS_GPU_H
