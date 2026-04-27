#pragma once
#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <cstdint>
#include <cstddef>
#include <stdbool.h>

#ifdef _WIN32
#  if defined(TQ_BUILD_DLL) || defined(ggml_hip_EXPORTS)
#    define TQ_API __declspec(dllexport)
#  else
#    define TQ_API __declspec(dllimport)
#  endif
#else
#  define TQ_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Types & Config
// ─────────────────────────────────────────────────────────────────────────────

typedef enum {
    TQ_BITS_2 = 2,
    TQ_BITS_3 = 3,
    TQ_BITS_4 = 4
} tq_bits_t;

typedef struct {
    int       head_dim;
    int       n_heads_kv;
    int       max_seq_len;
    tq_bits_t key_bits;
    tq_bits_t val_bits;
    bool      use_qjl;
} tq_config_t;

typedef struct {
    uint8_t  *levels;      // [2^bits]
    float    *thresholds;  // [2^bits + 1]
    int       n_levels;
} tq_codebook_t;

typedef struct {
    tq_config_t cfg;
    int         n_tokens;

    // Compressed KV buffers - 128-byte aligned for RDNA4 Infinity Cache
    uint8_t *d_keys_quant; // [max_seq_len, n_heads_kv, ALIGNED(bytes_per_key)]
    uint8_t *d_vals_quant; // [max_seq_len, n_heads_kv, ALIGNED(bytes_per_val)]
    uint8_t *d_keys_qjl;   // [max_seq_len, n_heads_kv, ALIGNED(head_dim/8)]
    uint8_t *d_vals_qjl;   // [max_seq_len, n_heads_kv, ALIGNED(head_dim/8)]

    // アライメント情報を保存
    size_t aligned_bytes_per_key;
    size_t aligned_bytes_per_val;
    size_t aligned_bytes_per_qjl;

    // Random rotation and projection vectors (device)
    float   *d_sign_k1, *d_sign_k2;
    float   *d_sign_v1, *d_sign_v2;
    float   *d_qjl_proj_k, *d_qjl_proj_v;

    // Temporary scratch (pre-allocated)
    float   *d_tmp_kf, *d_tmp_vf;
    float   *d_tmp_kf_orig, *d_tmp_vf_orig;
    float   *d_tmp_recon_k, *d_tmp_recon_v;
} tq_kv_cache_t;

typedef struct ihipStream_t *tq_hip_stream_t;

#define TQ_SEED_ROTATION 42
#define TQ_SEED_QJL      1337

// ─────────────────────────────────────────────────────────────────────────────
// Core API
// ─────────────────────────────────────────────────────────────────────────────

TQ_API int  tq_init(void);
TQ_API int  tq_kv_cache_alloc(tq_kv_cache_t *cache, const tq_config_t *cfg);
TQ_API void tq_kv_cache_free(tq_kv_cache_t *cache);

TQ_API int  tq_store_k(tq_kv_cache_t *cache, const void *d_keys_f32, const int *d_indices, int n_tokens);
TQ_API int  tq_store_v(tq_kv_cache_t *cache, const void *d_vals_f32, const int *d_indices, int n_tokens);

TQ_API int  tq_attn_logits(const tq_kv_cache_t *cache, const float *d_query, float *d_logits, int n_queries, int n_heads_q);
TQ_API int  tq_attn_output(const tq_kv_cache_t *cache, const float *d_logits, float *d_output, int n_queries, int n_heads_q);

TQ_API const tq_codebook_t *tq_get_codebook(tq_bits_t bits);
TQ_API size_t tq_bytes_per_token(const tq_config_t *cfg);

// ─────────────────────────────────────────────────────────────────────────────
// Low-level Launchers
// ─────────────────────────────────────────────────────────────────────────────

TQ_API int tq_launch_rotate(float *d_x, const float *d_sign1, const float *d_sign2, int n_tokens, int n_heads, int head_dim, tq_hip_stream_t stream);
TQ_API int tq_launch_quantize_scatter(const float *d_x_rot, uint8_t *d_q_out, const int *d_indices, int n_tokens, int n_heads, int head_dim, int bits, tq_hip_stream_t stream);
TQ_API int tq_launch_attn_logits(const float *d_query, const uint8_t *d_keys_q, const float *d_sign1, const float *d_sign2, float *d_logits, int n_heads_q, int n_heads_kv, int head_dim, int n_tokens_kv, int n_queries, int key_bits, tq_hip_stream_t stream);
TQ_API int tq_launch_attn_output(const float *d_logits, const uint8_t *d_vals_q, float *d_output, int n_heads_q, int n_heads_kv, int head_dim, int n_tokens_kv, int n_queries, int val_bits, tq_hip_stream_t stream);
TQ_API int tq_launch_fp16_to_float(const void *d_src, float *d_dst, int n, tq_hip_stream_t stream);

// ─────────────────────────────────────────────────────────────────────────────
// llama.cpp Integration API
// ─────────────────────────────────────────────────────────────────────────────

struct llama_tq_context;

TQ_API struct llama_tq_context *llama_tq_create(int n_layers, int n_heads_kv, int head_dim, int max_seq_len, int key_bits, int val_bits);
TQ_API void llama_tq_free  (struct llama_tq_context *ctx);
TQ_API void llama_tq_reset (struct llama_tq_context *ctx);
TQ_API void llama_tq_set_global(struct llama_tq_context *ctx);
TQ_API struct llama_tq_context *llama_tq_get_global(void);

TQ_API int  llama_tq_store_k(struct llama_tq_context *ctx, int layer, int start_token, const void *d_keys_f32, int n_tokens, const int *d_indices);
TQ_API int  llama_tq_store_v(struct llama_tq_context *ctx, int layer, int start_token, const void *d_vals_f32, int n_tokens, const int *d_indices);
TQ_API int  llama_tq_attn(struct llama_tq_context *ctx, int layer, const void *d_query_fp16, float *d_logits_out, float *d_output, int n_heads_q, int n_queries, int n_tokens_kv);
TQ_API void llama_tq_print_stats(const struct llama_tq_context *ctx);

// GLOBAL CONTEXT POINTER
#ifdef __cplusplus
extern "C" {
#endif
TQ_API extern struct llama_tq_context * g_tq_ctx;
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
}
#endif

#endif // TURBOQUANT_H
