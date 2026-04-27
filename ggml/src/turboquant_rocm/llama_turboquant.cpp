#include "turboquant.h"
/**
 * llama_turboquant.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Integration layer: hooks TurboQuant KV compression into llama.cpp (ROCm).
 *
 * HOW TO INTEGRATE
 * ────────────────
 * 1. Copy turboquant.h, turboquant.cpp, turboquant_hip.hip into llama.cpp/
 * 2. Add to CMakeLists.txt (ROCm section):
 *
 *      if (LLAMA_HIPBLAS OR GGML_HIP)
 *        set_source_files_properties(turboquant_hip.hip PROPERTIES LANGUAGE HIP)
 *        target_sources(llama PRIVATE
 *          turboquant.cpp
 *          turboquant_hip.hip
 *          llama_turboquant.cpp)
 *        target_compile_definitions(llama PRIVATE LLAMA_TURBOQUANT)
 *      endif()
 *
 * 3. In llama.cpp, find llama_new_context_with_model() and add:
 *
 *      #ifdef LLAMA_TURBOQUANT
 *        lctx->tq = llama_tq_create(model, params);
 *      #endif
 *
 * 4. Replace kv_cache fill in llm_build_kv_store() with tq_store_kv().
 * 5. Replace ggml_mul_mat attention with tq_attn_logits() + tq_attn_output().
 *
 * QUICK START (monkey-patch approach for testing)
 * ───────────────────────────────────────────────
 * Set environment variables before running:
 *   LLAMA_TQ=1             enable TurboQuant
 *   LLAMA_TQ_KEY_BITS=3    key quantization bits (2/3/4, default 3)
 *   LLAMA_TQ_VAL_BITS=2    value quantization bits (2/3/4, default 2)
 */

#ifdef LLAMA_TURBOQUANT

#ifndef TQ_STANDALONE
#include "../../include/llama.h"
#endif
#include "turboquant.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <memory>

// ─────────────────────────────────────────────────────────────────────────────
// Context: one TQ KV cache per layer
// ─────────────────────────────────────────────────────────────────────────────

struct llama_tq_context {
    std::vector<tq_kv_cache_t> layer_caches;
    tq_config_t                cfg;
    bool                       enabled = false;
    float                     *d_logits_buf = nullptr;   // [n_heads, max_seq_len]
};

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────

llama_tq_context* llama_tq_create(
    int n_layers,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int key_bits,   // 3 recommended
    int val_bits)   // 2 recommended
{
    auto ctx = new llama_tq_context();

    if (tq_init() != 0) {
        fprintf(stderr, "[TurboQuant] Initialization failed\n");
        delete ctx;
        return nullptr;
    }

    ctx->cfg = tq_config_t{
        .head_dim    = head_dim,
        .n_heads_kv  = n_heads_kv,
        .max_seq_len = max_seq_len,
        .key_bits    = (tq_bits_t)key_bits,
        .val_bits    = (tq_bits_t)val_bits,
        .use_qjl     = true,
    };

    ctx->layer_caches.resize(n_layers);
    for (int l = 0; l < n_layers; l++) {
        if (tq_kv_cache_alloc(&ctx->layer_caches[l], &ctx->cfg) != 0) {
            fprintf(stderr, "[TurboQuant] Layer %d alloc failed\n", l);
            llama_tq_free(ctx);
            return nullptr;
        }
    }

    // Allocate logits buffer (max head ÁEseq_len floats)
    size_t logits_sz = (size_t)n_heads_kv * max_seq_len * sizeof(float);
    hipMalloc(&ctx->d_logits_buf, logits_sz);

    ctx->enabled = true;

    float savings_x = 32.0f / (key_bits + val_bits + 2.0f);  // approx vs fp32
    fprintf(stderr,
        "[TurboQuant] Ready: %d layers, %d KV heads, dim=%d, "
        "keys=%dbit, vals=%dbit, ~%.1fx memory savings vs fp32\n",
        n_layers, n_heads_kv, head_dim, key_bits, val_bits, savings_x);

    return ctx;
}

void llama_tq_free(llama_tq_context *ctx) {
    if (!ctx) return;
    for (auto &c : ctx->layer_caches) tq_kv_cache_free(&c);
    if (ctx->d_logits_buf) hipFree(ctx->d_logits_buf);
    delete ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-token KV storage
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Store keys and values for a given layer and token position.
 *
 * @param ctx     TurboQuant context
 * @param layer   layer index
 * @param token   token position in sequence
 * @param d_keys  device fp16 tensor [n_heads_kv, head_dim]
 * @param d_vals  device fp16 tensor [n_heads_kv, head_dim]
 */
int llama_tq_store(llama_tq_context *ctx,
                   int layer,
                   int token,
                   const void *d_keys,
                   const void *d_vals)
{
    if (!ctx || !ctx->enabled) return -1;
    return tq_store_kv(&ctx->layer_caches[layer], d_keys, d_vals, token);
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention computation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Full TurboQuant attention for one layer (decode step).
 *
 * Replaces the standard:
 *   logits = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * @param ctx       TurboQuant context
 * @param layer     layer index
 * @param d_query   device fp16 [n_heads_kv, head_dim]
 * @param d_output  device fp16 [n_heads_kv, head_dim]  (output)
 */
int llama_tq_attn(llama_tq_context *ctx,
                  int layer,
                  const void *d_query,
                  void *d_output)
{
    if (!ctx || !ctx->enabled) return -1;
    tq_kv_cache_t *cache = &ctx->layer_caches[layer];

    // 1. Compute logits: [n_heads, n_tokens]
    int ret = tq_attn_logits(cache, d_query, ctx->d_logits_buf);
    if (ret != 0) return ret;

    // 2. Softmax + weighted value sum ↁEoutput
    return tq_attn_output(cache, ctx->d_logits_buf, d_output);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cache management
// ─────────────────────────────────────────────────────────────────────────────

/** Reset sequence (e.g., on new prompt) */
void llama_tq_reset(llama_tq_context *ctx) {
    if (!ctx) return;
    for (auto &c : ctx->layer_caches) {
        c.n_tokens = 0;
        // Clear packed bit arrays (important for correct atomicOr accumulation)
        const tq_config_t &cfg = c.cfg;
        int k_b = (cfg.head_dim * cfg.key_bits + 7) / 8;
        int v_b = (cfg.head_dim * cfg.val_bits + 7) / 8;
        int q_b = (cfg.head_dim + 7) / 8;
        size_t T = cfg.max_seq_len, H = cfg.n_heads_kv;
        hipMemset(c.d_keys_quant, 0, T * H * k_b);
        hipMemset(c.d_vals_quant, 0, T * H * v_b);
        hipMemset(c.d_keys_qjl,  0, T * H * q_b);
        hipMemset(c.d_vals_qjl,  0, T * H * q_b);
    }
}

/** Print memory usage summary */
void llama_tq_print_stats(const llama_tq_context *ctx) {
    if (!ctx) return;
    const tq_config_t &cfg = ctx->cfg;
    size_t n_layers = ctx->layer_caches.size();
    size_t tq_total = n_layers * tq_bytes_per_token(&cfg) * cfg.max_seq_len;
    size_t fp16_total = n_layers * (size_t)cfg.n_heads_kv * cfg.head_dim
                        * cfg.max_seq_len * 2 * 2; // K+V, 2 bytes fp16
    fprintf(stderr,
        "[TurboQuant] Memory: TQ=%.1fMB  fp16=%.1fMB  ratio=%.2fx\n",
        tq_total / 1e6, fp16_total / 1e6,
        (double)fp16_total / (double)tq_total);
}

// ─────────────────────────────────────────────────────────────────────────────
// CMakeLists.txt snippet (print to stderr for reference)
// ─────────────────────────────────────────────────────────────────────────────

__attribute__((constructor))
static void tq_print_build_hint(void) {
#ifdef LLAMA_TURBOQUANT_VERBOSE
    fprintf(stderr,
        "[TurboQuant] Built with LLAMA_TURBOQUANT. "
        "Add -DLLAMA_TURBOQUANT=ON to cmake.\n");
#endif
}

#endif // LLAMA_TURBOQUANT
