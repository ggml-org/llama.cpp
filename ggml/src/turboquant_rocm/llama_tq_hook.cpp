#include "turboquant.h"
#include <vector>
#include <mutex>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifdef LLAMA_TURBOQUANT

struct llama_tq_layer {
    tq_kv_cache_t cache;
    bool allocated = false;
};

struct llama_tq_context {
    tq_config_t cfg;
    std::vector<llama_tq_layer> layers;
    std::mutex mtx;
    
    float * d_q_float = nullptr;
    size_t q_float_size = 0;
};

struct llama_tq_context * g_tq_ctx = nullptr;

extern "C" struct llama_tq_context *llama_tq_create(int n_layers, int n_heads_kv, int head_dim, int max_seq_len, int key_bits, int val_bits) {
    try {
        if (g_tq_ctx) {
            llama_tq_free(g_tq_ctx);
        }

        tq_init();
        auto *ctx = new llama_tq_context();
        ctx->cfg = tq_config_t{ head_dim, n_heads_kv, max_seq_len, (tq_bits_t)key_bits, (tq_bits_t)val_bits, false };
        ctx->layers.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            ctx->layers[i].allocated = false;
        }
        ctx->d_q_float = nullptr;
        ctx->q_float_size = 0;
        
        g_tq_ctx = ctx;
        return ctx;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_tq_free(struct llama_tq_context *ctx) {
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        for (auto &l : ctx->layers) {
            if (l.allocated) tq_kv_cache_free(&l.cache);
        }
        if (ctx->d_q_float) {
            hipFree(ctx->d_q_float);
        }
    }
    if (g_tq_ctx == ctx) g_tq_ctx = nullptr;
    delete ctx;
}

extern "C" void llama_tq_set_global(struct llama_tq_context *ctx) {
    g_tq_ctx = ctx;
}

extern "C" struct llama_tq_context *llama_tq_get_global(void) {
    return g_tq_ctx;
}

extern "C" int llama_tq_store_k(struct llama_tq_context *ctx, int layer, int start_token, const void *d_keys, int n_tokens, const int *d_indices) {
    if (!ctx || n_tokens <= 0) return 0; 
    std::lock_guard<std::mutex> lock(ctx->mtx);
    if (layer < 0 || layer >= (int)ctx->layers.size()) return -1;
    auto &l = ctx->layers[layer];
    if (!l.allocated) {
        if (tq_kv_cache_alloc(&l.cache, &ctx->cfg) != 0) return -1;
        l.allocated = true;
    }
    return tq_store_k(&l.cache, d_keys, d_indices, n_tokens);
}

extern "C" int llama_tq_store_v(struct llama_tq_context *ctx, int layer, int start_token, const void *d_vals, int n_tokens, const int *d_indices) {
    if (!ctx || n_tokens <= 0) return 0; 
    std::lock_guard<std::mutex> lock(ctx->mtx);
    if (layer < 0 || layer >= (int)ctx->layers.size()) return -1;
    auto &l = ctx->layers[layer];
    if (!l.allocated) {
        if (tq_kv_cache_alloc(&l.cache, &ctx->cfg) != 0) return -1;
        l.allocated = true;
    }
    return tq_store_v(&l.cache, d_vals, d_indices, n_tokens);
}

extern "C" int llama_tq_attn(struct llama_tq_context *ctx, int layer, const void *d_query_fp16, float *d_logits_out, float *d_output, int n_heads_q, int n_queries, int n_tokens_kv) {
    if (!ctx || layer < 0 || layer >= (int)ctx->layers.size()) return -1;
    if (n_queries <= 0 || n_tokens_kv <= 0) return 0;

    auto &l = ctx->layers[layer];
    if (!l.allocated) return -1;

    int n_elements = n_queries * n_heads_q * ctx->cfg.head_dim;
    
    ctx->mtx.lock();
    if (ctx->q_float_size < (size_t)n_elements) {
        if (ctx->d_q_float) hipFree(ctx->d_q_float);
        hipMalloc(&ctx->d_q_float, n_elements * sizeof(float));
        ctx->q_float_size = n_elements;
    }
    ctx->mtx.unlock();

    tq_launch_fp16_to_float(d_query_fp16, ctx->d_q_float, n_elements, nullptr);

    l.cache.n_tokens = n_tokens_kv; 
    tq_attn_logits(&l.cache, ctx->d_q_float, d_logits_out, n_queries, n_heads_q);
    return tq_attn_output(&l.cache, d_logits_out, d_output, n_queries, n_heads_q);
}

extern "C" void llama_tq_print_stats(const struct llama_tq_context *ctx) {
    (void)ctx;
}

#endif