#ifndef LLAMA_KV_CODEC_H
#define LLAMA_KV_CODEC_H

#include "ggml.h"
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// KV Cache Codec Interface for TurboQuant-style compression
//
// This interface allows external code to intercept and transform K/V vectors
// during the forward pass, enabling custom compression algorithms like TurboQuant
// to be used with llama.cpp without modifying core inference logic.
//
// Paper: "TurboQuant: A Fast and Efficient Post-Training KV Cache Compression"
//        https://arxiv.org/abs/2504.19874
//

// Opaque handle for a KV codec instance
typedef struct llama_kv_codec llama_kv_codec;

// Callback: Compress a key vector after K projection
// Parameters:
//   codec     - codec instance
//   layer     - layer index (0-based)
//   head      - KV head index (0-based)
//   pos       - token position in sequence
//   k_data    - input key vector (head_dim f32 values)
//   head_dim  - dimension of each head
//   output    - output buffer (head_dim f32 values, pre-allocated)
// Returns: 0 on success, non-zero on error
typedef int (*llama_kv_codec_compress_k_fn)(
    llama_kv_codec * codec,
    uint32_t layer,
    uint32_t head,
    uint32_t pos,
    const float * k_data,
    uint32_t head_dim,
    float * output
);

// Callback: Compress a value vector after V projection
// Same signature as compress_k
typedef int (*llama_kv_codec_compress_v_fn)(
    llama_kv_codec * codec,
    uint32_t layer,
    uint32_t head,
    uint32_t pos,
    const float * v_data,
    uint32_t head_dim,
    float * output
);

// Callback: Compute attention scores using compressed K representations
// Parameters:
//   codec      - codec instance
//   layer      - layer index
//   head       - KV head index
//   query      - query vector (head_dim f32 values)
//   head_dim   - dimension of each head
//   pos_start  - start position in sequence
//   pos_end    - end position in sequence (exclusive)
//   scores     - output scores buffer (pos_end - pos_start f32 values)
// Returns: 0 on success, non-zero on error
typedef int (*llama_kv_codec_score_k_fn)(
    const llama_kv_codec * codec,
    uint32_t layer,
    uint32_t head,
    const float * query,
    uint32_t head_dim,
    uint32_t pos_start,
    uint32_t pos_end,
    float * scores
);

// Callback: Read V vectors for attention output computation
// Parameters:
//   codec      - codec instance
//   layer      - layer index
//   head       - KV head index
//   head_dim   - dimension of each head
//   pos_start  - start position
//   pos_end    - end position (exclusive)
//   v_buffer   - output buffer ((pos_end - pos_start) * head_dim f32 values)
// Returns: 0 on success, non-zero on error
typedef int (*llama_kv_codec_read_v_fn)(
    const llama_kv_codec * codec,
    uint32_t layer,
    uint32_t head,
    uint32_t head_dim,
    uint32_t pos_start,
    uint32_t pos_end,
    float * v_buffer
);

// Callback: Reset codec state for a new sequence
typedef void (*llama_kv_codec_reset_fn)(llama_kv_codec * codec);

// Callback: Free codec resources
typedef void (*llama_kv_codec_free_fn)(llama_kv_codec * codec);

// KV Codec configuration
struct llama_kv_codec_params {
    // Codec instance (NULL to disable compression)
    llama_kv_codec * codec;

    // Callbacks
    llama_kv_codec_compress_k_fn compress_k;
    llama_kv_codec_compress_v_fn compress_v;
    llama_kv_codec_score_k_fn    score_k;
    llama_kv_codec_read_v_fn     read_v;
    llama_kv_codec_reset_fn      reset;
    llama_kv_codec_free_fn       free_codec;

    // Model dimensions
    uint32_t n_layer;
    uint32_t n_kv_head;
    uint32_t head_dim;

    // Flags
    bool compress_keys;     // apply key compression
    bool compress_values;   // apply value compression
    bool use_codec_score;   // use codec for scoring instead of raw K
    bool use_codec_readv;   // use codec for V reading instead of raw V
};

// Create default params (compression disabled)
LLAMA_API struct llama_kv_codec_params llama_kv_codec_default_params(void);

// Set KV codec for a context (call before first decode)
// Pass NULL codec to disable compression
// The codec must remain valid for the lifetime of the context
LLAMA_API void llama_set_kv_codec(
    struct llama_context * ctx,
    const struct llama_kv_codec_params * params
);

// Get current codec info (returns false if no codec is active)
LLAMA_API bool llama_get_kv_codec_info(
    const struct llama_context * ctx,
    struct llama_kv_codec_params * params
);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_KV_CODEC_H
