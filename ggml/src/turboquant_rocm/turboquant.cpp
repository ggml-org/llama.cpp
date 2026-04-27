/**
 * turboquant.cpp  —  Host-side orchestration (Scatter Store & Batch Support)
 */

#include "turboquant.h"
#include "turboquant_hip.cuh" // 実装を直接注入
#include <hip/hip_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>
#include <cstdio>

#define HIP_CHECK(ans) { if ((ans) != hipSuccess) return -1; }

static tq_codebook_t g_codebooks[3]; // 2, 3, 4 bit

static void build_codebook_2bit(tq_codebook_t *cb) {
    cb->n_levels = 4;
    cb->levels[0] = -0.7071f; cb->levels[1] = -0.2357f;
    cb->levels[2] =  0.2357f; cb->levels[3] =  0.7071f;
    cb->thresholds[0] = -1.0f; cb->thresholds[1] = -0.5f; cb->thresholds[2] = 0.0f;
    cb->thresholds[3] = 0.5f;  cb->thresholds[4] = 1.0f;
}

static void build_codebook_3bit(tq_codebook_t *cb) {
    cb->n_levels = 8;
    float l[] = {-0.9009f,-0.6235f,-0.3612f,-0.1205f, 0.1205f, 0.3612f, 0.6235f, 0.9009f};
    float t[] = {-1.0f,-0.7654f,-0.5000f,-0.2393f,0.0f, 0.2393f,0.5000f,0.7654f,1.0f};
    for(int i=0; i<8; i++) cb->levels[i]=l[i];
    for(int i=0; i<9; i++) cb->thresholds[i]=t[i];
}

static void build_codebook_4bit(tq_codebook_t *cb) {
    cb->n_levels = 16;
    float l[] = {-0.9613f,-0.8315f,-0.6895f,-0.5345f, -0.3681f,-0.1951f,-0.0980f, 0.0000f,
                  0.0980f, 0.1951f, 0.3681f, 0.5345f,  0.6895f, 0.8315f, 0.9613f, 0.9903f};
    float t[] = {-1.0f, -0.9009f,-0.7654f,-0.6124f, -0.4540f,-0.2820f,-0.1464f,-0.0490f,
                  0.0490f, 0.1464f, 0.2820f, 0.4540f,  0.6124f, 0.7654f, 0.9009f, 0.9808f, 1.0f};
    for(int i=0; i<16; i++) cb->levels[i]=l[i];
    for(int i=0; i<17; i++) cb->thresholds[i]=t[i];
}

static void generate_random_ortho_matrix(float *out, int rows, int cols, uint64_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int r = 0; r < rows; r++) {
        float norm2 = 0.0f;
        for (int c = 0; c < cols; c++) {
            float v = dist(rng);
            out[r * cols + c] = v;
            norm2 += v * v;
        }
        float inv = 1.0f / sqrtf(norm2);
        for (int c = 0; c < cols; c++)
            out[r * cols + c] *= inv;
    }
}

extern "C" {

int tq_init(void) {
    build_codebook_2bit(&g_codebooks[0]);
    build_codebook_3bit(&g_codebooks[1]);
    build_codebook_4bit(&g_codebooks[2]);
    fprintf(stderr, "[TQ] Initialized  (ROCm/HIP backend)\n");
    return 0;
}

const tq_codebook_t *tq_get_codebook(tq_bits_t bits) {
    if (bits < 2 || bits > 4) return nullptr;
    return &g_codebooks[bits - 2];
}

size_t tq_bytes_per_token(const tq_config_t *cfg) {
    size_t k = (size_t)((cfg->key_bits) * cfg->head_dim + 7) / 8;
    size_t v = (size_t)((cfg->val_bits) * cfg->head_dim + 7) / 8;
    size_t qj = (size_t)(cfg->head_dim + 7) / 8;
    return (k + v + 2*qj) * cfg->n_heads_kv;
}

int tq_kv_cache_alloc(tq_kv_cache_t *cache, const tq_config_t *cfg) {
    assert(cache && cfg);
    memset(cache, 0, sizeof(*cache));
    cache->cfg = *cfg;

    int    d  = cfg->head_dim;
    int    H  = cfg->n_heads_kv;
    int    T  = cfg->max_seq_len;

    // 128-byte aligned sizes for RDNA4 Infinity Cache optimization
    int raw_k_bpv = (d * cfg->key_bits + 7) / 8;
    int raw_v_bpv = (d * cfg->val_bits + 7) / 8;
    int raw_qjl_bpv = (d + 7) / 8;
    
    // アラインされたサイズを計算
    cache->aligned_bytes_per_key = ((raw_k_bpv + 127) / 128) * 128;
    cache->aligned_bytes_per_val = ((raw_v_bpv + 127) / 128) * 128;
    cache->aligned_bytes_per_qjl = ((raw_qjl_bpv + 127) / 128) * 128;
    
    int k_bpv = (int)cache->aligned_bytes_per_key;
    int v_bpv = (int)cache->aligned_bytes_per_val;
    size_t total = (size_t)T * H;

    // hipMallocは一般的に128-byteアラインされているが、明示的に確認
    if (hipMalloc(&cache->d_keys_quant, total * k_bpv) != hipSuccess) return -1;
    if (hipMalloc(&cache->d_vals_quant, total * v_bpv) != hipSuccess) return -1;

    if (hipMalloc(&cache->d_sign_k1, d * sizeof(float)) != hipSuccess) return -1;
    if (hipMalloc(&cache->d_sign_k2, d * sizeof(float)) != hipSuccess) return -1;
    if (hipMalloc(&cache->d_sign_v1, d * sizeof(float)) != hipSuccess) return -1;
    if (hipMalloc(&cache->d_sign_v2, d * sizeof(float)) != hipSuccess) return -1;

    // Allocate scratch for batch processing (1024 tokens)
    size_t scratch_tokens = 1024;
    size_t scratch_sz = scratch_tokens * H * d * sizeof(float);
    if (hipMalloc(&cache->d_tmp_kf, scratch_sz) != hipSuccess) return -1;
    if (hipMalloc(&cache->d_tmp_vf, scratch_sz) != hipSuccess) return -1;

    std::vector<float> h_sign(d);
    std::mt19937 rng(TQ_SEED_ROTATION);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto fill_sign = [&](float *d_ptr) {
        for(int i=0; i<d; i++) h_sign[i] = (dist(rng) > 0) ? 1.0f : -1.0f;
        hipMemcpy(d_ptr, h_sign.data(), d*sizeof(float), hipMemcpyHostToDevice);
    };

    fill_sign(cache->d_sign_k1); fill_sign(cache->d_sign_k2);
    fill_sign(cache->d_sign_v1); fill_sign(cache->d_sign_v2);

    cache->n_tokens = 0;
    return 0;
}

void tq_kv_cache_free(tq_kv_cache_t *cache) {
    if (!cache) return;
    if (cache->d_keys_quant) hipFree(cache->d_keys_quant);
    if (cache->d_vals_quant) hipFree(cache->d_vals_quant);
    if (cache->d_sign_k1)    hipFree(cache->d_sign_k1);
    if (cache->d_sign_k2)    hipFree(cache->d_sign_k2);
    if (cache->d_sign_v1)    hipFree(cache->d_sign_v1);
    if (cache->d_sign_v2)    hipFree(cache->d_sign_v2);
    if (cache->d_tmp_kf)     hipFree(cache->d_tmp_kf);
    if (cache->d_tmp_vf)     hipFree(cache->d_tmp_vf);
    memset(cache, 0, sizeof(*cache));
}

int tq_store_k(tq_kv_cache_t *cache, const void *d_keys_f32, const int *d_indices, int n_tokens) {
    const tq_config_t *cfg = &cache->cfg; int d = cfg->head_dim; int H = cfg->n_heads_kv; hipStream_t stream = nullptr;
    float *d_kf = cache->d_tmp_kf;
    
    HIP_CHECK(hipMemcpyAsync(d_kf, d_keys_f32, (size_t)n_tokens * H * d * sizeof(float), hipMemcpyDeviceToDevice, stream));
    
    tq_launch_rotate(d_kf, cache->d_sign_k1, cache->d_sign_k2, n_tokens, H, d, (tq_hip_stream_t)stream);
    tq_launch_quantize_scatter(d_kf, cache->d_keys_quant, d_indices, n_tokens, H, d, cfg->key_bits, (tq_hip_stream_t)stream);
    return 0;
}

int tq_store_v(tq_kv_cache_t *cache, const void *d_vals_f32, const int *d_indices, int n_tokens) {
    const tq_config_t *cfg = &cache->cfg; int d = cfg->head_dim; int H = cfg->n_heads_kv; hipStream_t stream = nullptr;
    float *d_vf = cache->d_tmp_vf;
    
    HIP_CHECK(hipMemcpyAsync(d_vf, d_vals_f32, (size_t)n_tokens * H * d * sizeof(float), hipMemcpyDeviceToDevice, stream));
    
    tq_launch_rotate(d_vf, cache->d_sign_v1, cache->d_sign_v2, n_tokens, H, d, (tq_hip_stream_t)stream);
    tq_launch_quantize_scatter(d_vf, cache->d_vals_quant, d_indices, n_tokens, H, d, cfg->val_bits, (tq_hip_stream_t)stream);
    return 0;
}

extern "C" int tq_attn_logits(const tq_kv_cache_t *cache, const float *d_query, float *d_logits, int n_queries, int n_heads_q) {
    const tq_config_t *c = &cache->cfg;
    return tq_launch_attn_logits(
        d_query, cache->d_keys_quant, cache->d_sign_k1, cache->d_sign_k2,
        d_logits,
        n_heads_q, c->n_heads_kv, c->head_dim, cache->n_tokens, n_queries, c->key_bits,
        nullptr);
}

extern "C" int tq_attn_output(const tq_kv_cache_t *cache, const float *d_logits, float *d_output, int n_queries, int n_heads_q) {
    const tq_config_t *c = &cache->cfg;
    int r = tq_launch_attn_output(
        d_logits, cache->d_vals_quant,
        d_output,
        n_heads_q, c->n_heads_kv, c->head_dim, cache->n_tokens, n_queries, c->val_bits,
        nullptr);
    if (r != 0) return r;
    return tq_launch_rotate(d_output, cache->d_sign_v2, cache->d_sign_v1, n_queries, n_heads_q, c->head_dim, nullptr);
}

} // extern "C"
