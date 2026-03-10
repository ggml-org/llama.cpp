//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Applies rotary position encoding:
//   f32[head_dim, heads, seq_len] x i32 -> f32[head_dim, heads, seq_len]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// ROPE constants (matching GGML definitions)
#define GGML_ROPE_TYPE_NEOX 2
#define MAX_ROPE_HALF_DIMS 256  // supports up to n_dims=512

// ROPE operation parameters structure (matches ggml-et-ops.h)
typedef struct {
    int32_t n_past;
    int32_t n_dims;        // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;          // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    float   freq_base;     // Base frequency (usually 10000.0f)
    float   freq_scale;    // Frequency scaling factor
    float   ext_factor;    // Extension factor for YaRN
    float   attn_factor;   // Attention factor for YaRN
    float   beta_fast;     // Fast beta for YaRN
    float   beta_slow;     // Slow beta for YaRN
    int32_t sections[4];   // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
struct ggml_et_rope_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor src1;  // I32 position tensor
    struct ggml_tensor src2;  // F32 frequency factors (optional)
    struct ggml_tensor dst;   // F32 output tensor
    rope_params_t rope_params;
};

// YaRN helper functions
static inline float rope_yarn_ramp(const float low, const float high, const int i0) {
    float denom = high - low;
    if (denom < 0.001f) denom = 0.001f;  // MAX(0.001f, high - low)

    const float y = et_fdiv((float)(i0 / 2) - low, denom);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);  // MIN(1, MAX(0, y))
    return 1.0f - clamped;
}

static inline float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float beta, float freq_base) {
    return n_dims * et_fdiv(et_logf(et_fdiv((float)n_ctx_orig, freq_base)), et_logf(beta) * 2.0f);
}

static inline void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                                       float beta_fast, float beta_slow, float dims[2]) {
    float start = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base);
    float end = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base);

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (float)(n_dims - 1) ? end : (float)(n_dims - 1);
}

// YaRN algorithm (MIT licensed, Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng)
static inline void rope_yarn(float theta_extrap, float freq_scale, const float corr_dims[2],
                             int64_t i0, float ext_factor, float mscale,
                             float* cos_theta, float* sin_theta) {
    // theta_interp uses frequency scaling
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    if (ext_factor != 0.0f) {
        // Mix between interpolated and extrapolated based on dimension
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;

        // Magnitude scaling correction for interpolation
        mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
    }

    *cos_theta = et_cosf(theta) * mscale;
    *sin_theta = et_sinf(theta) * mscale;
}

// Populate cos/sin cache for a given position using running theta product
static inline void compute_rope_cache(
    float* cos_cache, float* sin_cache,
    int32_t n_dims, float theta_scale, int32_t pos,
    const float* freq_factors, float freq_scale,
    const float corr_dims[2], float ext_factor, float attn_factor) {

    const int32_t half_dims = n_dims / 2;
    float theta = 1.0f;

    for (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx++) {
        const float ff = freq_factors ? freq_factors[dim_idx] : 1.0f;
        const float theta_base = (float)pos * theta;

        rope_yarn(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_idx * 2,
                  ext_factor, attn_factor,
                  &cos_cache[dim_idx], &sin_cache[dim_idx]);

        theta *= theta_scale;
    }
}

int entry_point(struct ggml_et_rope_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return -1;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    struct ggml_tensor* src0 = &params->src0; // F32 input tensor [head_dim, heads, seq_len, batch]
    struct ggml_tensor* src1 = &params->src1; // I32 position tensor [seq_len]
    struct ggml_tensor* src2 = &params->src2; // F32 frequency factors (optional)
    struct ggml_tensor* dst = &params->dst;   // F32 output tensor [head_dim, heads, seq_len, batch]

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination
    }

    const float* src0_data = (const float*)src0->data;    // F32 input activations
    const int32_t* src1_data = (const int32_t*)src1->data; // I32 positions
    const float* freq_factors = NULL;                      // Optional frequency factors
    if (src2 && src2->data) {
        freq_factors = (const float*)src2->data;
    }
    float* dst_data = (float*)dst->data;                  // F32 output

    if (!src0_data || !src1_data || !dst_data) {
        return -1; // Null data pointer
    }

    // Get tensor dimensions
    // src0: [head_dim, heads, seq_len, batch]
    // src1: [seq_len] (positions)
    // dst:  [head_dim, heads, seq_len, batch]
    const int64_t head_dim = src0->ne[0];   // Head dimension (e.g., 128)
    const int64_t heads = src0->ne[1];      // Number of heads (e.g., 32 for Q, 8 for K/V)
    const int64_t seq_len = src0->ne[2];    // Sequence length (e.g., 512)
    const int64_t batch = src0->ne[3];      // Batch size

    const rope_params_t* rope_params = &params->rope_params;
    const int32_t n_dims = rope_params->n_dims;           // Dimensions to apply ROPE to
    const float freq_base = rope_params->freq_base;       // Base frequency (10000.0)
    const float freq_scale = rope_params->freq_scale;     // Frequency scaling
    const int32_t mode = rope_params->mode;               // ROPE mode

    if (n_dims <= 0 || n_dims > head_dim || n_dims % 2 != 0) {
        return -1; // Invalid ROPE dimensions
    }

    if (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        return -1; // n_dims exceeds cache capacity
    }

    float cos_cache[MAX_ROPE_HALF_DIMS];
    float sin_cache[MAX_ROPE_HALF_DIMS];

    // Calculate YaRN correction dimensions
    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base,
                       rope_params->beta_fast, rope_params->beta_slow, corr_dims);

    const int64_t total_positions = batch * seq_len;

    // Distribute positions across threads
    int64_t pos_per_thread = total_positions / num_threads;
    int64_t start_pos = thread_id * pos_per_thread;
    int64_t end_pos = (thread_id == num_threads - 1) ?
                      total_positions :
                      start_pos + pos_per_thread;

    const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));
    const int32_t half_dims = n_dims / 2;

    const int is_neox = (mode & GGML_ROPE_TYPE_NEOX) != 0;

    for (int64_t bs = start_pos; bs < end_pos; bs++) {
        int64_t s = bs % seq_len;
        int64_t b = bs / seq_len;

        const int32_t pos = src1_data[s] + rope_params->n_past;

        // Compute trig cache once for this position
        compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos,
                           freq_factors, freq_scale, corr_dims,
                           rope_params->ext_factor, rope_params->attn_factor);

        // Apply cached values to all heads
        for (int64_t h = 0; h < heads; h++) {
            const float* head_src = (const float*)((const char*)src0_data +
                b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

            float* head_dst = (float*)((char*)dst_data +
                b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

            // Copy entire head (including dims beyond n_dims)
            for (int64_t d = 0; d < head_dim; d++) {
                head_dst[d] = head_src[d];
            }

            if (is_neox) {
                // Apply rotations using cached cos/sin
                uint64_t temp_mask;
                __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
                __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

                int32_t dim_idx = 0;
                for (; dim_idx < half_dims; dim_idx+=8) {
                    // (scalar paths says dim_idx+=1)
                    // float x0 = head_src[dim_idx];
                    // float x1 = head_src[dim_idx + half_dims];

                    // head_dst[dim_idx]             = x0 * cos_cache[dim_idx] - x1 * sin_cache[dim_idx];
                    // head_dst[dim_idx + half_dims] = x0 * sin_cache[dim_idx] + x1 * cos_cache[dim_idx];
                    __asm__ volatile(
                        "flw.ps f0, %[x0_src]       \n\t"
                        "flw.ps f1, %[x1_src]       \n\t"
                        "flw.ps f2, %[sin_cache]    \n\t"
                        "flw.ps f3, %[cos_cache]    \n\t"
                        "fmul.ps f4, f0, f3         \n\t"
                        "fmul.ps f5, f0, f2         \n\t"
                        "fnmsub.ps f4, f1, f2, f4   \n\t"
                        "fmadd.ps f5, f1, f3, f5    \n\t"
                        "fsw.ps f4, %[x0_dst]       \n\t"
                        "fsw.ps f5, %[x1_dst]       \n\t"
                        : [x0_dst] "=m"(*(float(*)[8])&head_dst[dim_idx]),
                            [x1_dst] "=m"(*(float(*)[8])&head_dst[dim_idx + half_dims])
                        : [x0_src] "m"(*(const float(*)[8])&head_src[dim_idx]),
                            [x1_src] "m"(*(const float(*)[8])&head_src[dim_idx + half_dims]),
                            [sin_cache] "m"(*(const float(*)[8])&sin_cache[dim_idx]),
                            [cos_cache] "m"(*(const float(*)[8])&cos_cache[dim_idx])
                        : "f0", "f1", "f2", "f3", "f4", "f5", "memory"
                    );
                }
                __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
            } else {
                // Apply rotations using cached cos/sin (standard interleaved pairs)
                for (int32_t pair_idx = 0; pair_idx < half_dims; pair_idx++) {
                    int32_t dim_in_head = pair_idx * 2;
                    float x0 = head_src[dim_in_head];
                    float x1 = head_src[dim_in_head + 1];

                    head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                    head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
                }
            }
        }
    }

    return 0;
}
