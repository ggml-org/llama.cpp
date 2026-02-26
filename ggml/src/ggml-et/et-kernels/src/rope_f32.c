//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Applies rotary position encoding:
//   f32[head_dim, heads, seq_len] x i32 -> f32[head_dim, heads, seq_len]
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// ROPE constants (matching GGML definitions)
#define GGML_ROPE_TYPE_NEOX 2
#define CACHE_LINE_SIZE_F32 16

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

    const int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float

    // Calculate YaRN correction dimensions
    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base,
                       rope_params->beta_fast, rope_params->beta_slow, corr_dims);

    if (mode & GGML_ROPE_TYPE_NEOX) {
        // NeoX Mode: Work on complete heads to handle split pattern
        const int64_t total_work_units = batch * seq_len * heads;

        // Distribute work units across threads
        int64_t units_per_thread = total_work_units / num_threads;
        int64_t start_unit = thread_id * units_per_thread;
        int64_t end_unit = (thread_id == num_threads - 1) ?
                           total_work_units :
                           start_unit + units_per_thread;

        const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));

        const bool is_inplace = (src0_data == dst_data);

        for (int64_t unit = start_unit; unit < end_unit; unit++) {
            // Map work unit back to coordinates
            int64_t h = unit % heads;
            int64_t s = (unit / heads) % seq_len;
            int64_t b = unit / (heads * seq_len);

            const float* head_src = (const float*)((char*)src0_data +
                b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

            float* head_dst = (float*)((char*)dst_data +
                b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

            const int32_t pos = src1_data[s] + rope_params->n_past;

            // Process cache lines within this head
            int64_t cachelines_in_head = (head_dim + elements_per_cacheline - 1) / elements_per_cacheline;

            if (is_inplace) {
                // Inplace: Process each cacheline, writing both halves together
                for (int64_t cl = 0; cl < cachelines_in_head; cl++) {
                    float* cacheline = head_dst + cl * elements_per_cacheline;

                    // Copy source to destination first (for unmodified elements)
                    for (int64_t elem = 0; elem < elements_per_cacheline && (cl * elements_per_cacheline + elem) < head_dim; elem++) {
                        cacheline[elem] = head_src[cl * elements_per_cacheline + elem];
                    }

                    // Process elements in this cache line
                    for (int64_t elem = 0; elem < elements_per_cacheline && (cl * elements_per_cacheline + elem) < head_dim; elem++) {
                        int64_t dim_idx = cl * elements_per_cacheline + elem;

                        if (dim_idx < n_dims / 2) {
                            // First half - read before any writes, then write both halves
                            float x0 = head_src[dim_idx];
                            float x1 = head_src[dim_idx + n_dims/2];

                            // Calculate theta for this dimension pair
                            float theta = 1.0f;
                            for (int64_t j = 0; j < dim_idx; j++) {
                                theta *= theta_scale;
                            }

                            const float ff = freq_factors ? freq_factors[dim_idx] : 1.0f;
                            const float theta_base = (float)pos * theta;

                            float cos_theta, sin_theta;
                            rope_yarn(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_idx * 2,
                                     rope_params->ext_factor, rope_params->attn_factor,
                                     &cos_theta, &sin_theta);

                            // Write both halves of the pair (safe because we read both first)
                            head_dst[dim_idx] = x0 * cos_theta - x1 * sin_theta;
                            head_dst[dim_idx + n_dims/2] = x0 * sin_theta + x1 * cos_theta;
                        }
                    }
                }
            } else {
                // Non-inplace: Copy all cachelines first, then process rotations
                for (int64_t cl = 0; cl < cachelines_in_head; cl++) {
                    float* cacheline = head_dst + cl * elements_per_cacheline;
                    for (int64_t elem = 0; elem < elements_per_cacheline && (cl * elements_per_cacheline + elem) < head_dim; elem++) {
                        cacheline[elem] = head_src[cl * elements_per_cacheline + elem];
                    }
                }

                // Now process rotations, writing both halves together
                for (int64_t dim_idx = 0; dim_idx < n_dims / 2; dim_idx++) {
                    float x0 = head_dst[dim_idx];
                    float x1 = head_dst[dim_idx + n_dims/2];

                    // Calculate theta for this dimension pair
                    float theta = 1.0f;
                    for (int64_t j = 0; j < dim_idx; j++) {
                        theta *= theta_scale;
                    }

                    const float ff = freq_factors ? freq_factors[dim_idx] : 1.0f;
                    const float theta_base = (float)pos * theta;

                    float cos_theta, sin_theta;
                    rope_yarn(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_idx * 2,
                             rope_params->ext_factor, rope_params->attn_factor,
                             &cos_theta, &sin_theta);

                    // Write both halves of the pair
                    head_dst[dim_idx] = x0 * cos_theta - x1 * sin_theta;
                    head_dst[dim_idx + n_dims/2] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    } else {
        // Standard Mode: Process cache lines directly across all data
        const int64_t total_elements = batch * seq_len * heads * head_dim;
        const int64_t total_cachelines = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

        // Distribute cache lines across threads
        int64_t cachelines_per_thread = total_cachelines / num_threads;
        int64_t start_cacheline = thread_id * cachelines_per_thread;
        int64_t end_cacheline = (thread_id == num_threads - 1) ?
                                total_cachelines :
                                start_cacheline + cachelines_per_thread;

        // Pre-calculate theta_scale
        const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));

        for (int64_t cl = start_cacheline; cl < end_cacheline; cl++) {
            float* cacheline_dst = dst_data + cl * elements_per_cacheline;
            const float* cacheline_src = src0_data + cl * elements_per_cacheline;

            // Copy source to destination first
            for (int64_t elem = 0; elem < elements_per_cacheline && (cl * elements_per_cacheline + elem) < total_elements; elem++) {
                cacheline_dst[elem] = cacheline_src[elem];
            }

            // Process pairs in this cache line
            const int64_t pairs_per_cacheline = elements_per_cacheline / 2;

            for (int64_t local_pair = 0; local_pair < pairs_per_cacheline; local_pair++) {
                int64_t global_element_idx = cl * elements_per_cacheline + local_pair * 2;

                if (global_element_idx + 1 >= total_elements) break;

                // Map back to [batch, seq, head, dim] coordinates
                int64_t dim_in_head = global_element_idx % head_dim;
                int64_t h = (global_element_idx / head_dim) % heads;
                int64_t s = (global_element_idx / (head_dim * heads)) % seq_len;
                int64_t b = global_element_idx / (head_dim * heads * seq_len);

                if (dim_in_head < n_dims && dim_in_head % 2 == 0) {
                    // Calculate position and theta
                    const int32_t pos = src1_data[s] + rope_params->n_past;
                    const int64_t pair_idx = dim_in_head / 2;

                    float theta = 1.0f;
                    for (int64_t j = 0; j < pair_idx; j++) {
                        theta *= theta_scale;
                    }

                    const float ff = freq_factors ? freq_factors[pair_idx] : 1.0f;
                    const float theta_base = (float)pos * theta;

                    // Apply rotation to this pair using YaRN
                    float cos_theta, sin_theta;
                    rope_yarn(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_in_head,
                             rope_params->ext_factor, rope_params->attn_factor,
                             &cos_theta, &sin_theta);

                    float x0 = cacheline_dst[local_pair * 2];
                    float x1 = cacheline_dst[local_pair * 2 + 1];

                    cacheline_dst[local_pair * 2]     = x0 * cos_theta - x1 * sin_theta;
                    cacheline_dst[local_pair * 2 + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }

    return 0;
}
