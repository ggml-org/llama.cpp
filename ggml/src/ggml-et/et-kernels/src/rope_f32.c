//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <etsoc/common/utils.h>
#include <stdint.h>

// ROPE constants (matching GGML definitions)
#define GGML_ROPE_TYPE_NEOX   2
#define GGML_ROPE_TYPE_MROPE  8
#define GGML_ROPE_TYPE_IMROPE 40
#define MAX_ROPE_HALF_DIMS    256  // supports up to n_dims=512

#define ROPE_VEC_WIDTH 8

#define ROPE_PI         3.14159265358979323846f
#define ROPE_TWO_PI     6.28318530717958647693f
#define ROPE_PI_OVER_2  1.57079632679489661923f
#define ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
typedef struct {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    float   freq_base;    // Base frequency (usually 10000.0f)
    float   freq_scale;   // Frequency scaling factor
    float   ext_factor;   // Extension factor for YaRN
    float   attn_factor;  // Attention factor for YaRN
    float   beta_fast;    // Fast beta for YaRN
    float   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
struct ggml_et_rope_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor src1;  // I32 position tensor
    struct ggml_tensor src2;  // F32 frequency factors (optional)
    struct ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
static inline float rope_floorf(float x) {
    union {
        float    f;
        uint32_t u;
    } v = { .f = x };

    const uint32_t expo = (v.u >> 23) & 0xFF;
    if (expo == 0xFF) {
        return x;  // inf or NaN
    }
    if (expo >= 23 + 127) {
        return x;  // already integer-valued
    }
    int i = (int) x;
    return (x < 0.0f && (float) i != x) ? (float) (i - 1) : (float) i;
}

static inline float rope_ceilf(float x) {
    union {
        float    f;
        uint32_t u;
    } v = { .f = x };

    const uint32_t expo = (v.u >> 23) & 0xFF;
    if (expo == 0xFF) {
        return x;  // inf or NaN
    }
    if (expo >= 23 + 127) {
        return x;  // already integer-valued
    }
    int i = (int) x;
    return (x > 0.0f && (float) i != x) ? (float) (i + 1) : (float) i;
}

static inline float rope_yarn_ramp(const float low, const float high, const int i0) {
    float denom = high - low;
    if (denom < 0.001f) {
        denom = 0.001f;
    }

    const float y       = et_fdiv((float) (i0 / 2) - low, denom);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
static inline float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float beta, float freq_base) {
    return (float) n_dims *
           et_fdiv(et_logf(et_fdiv((float) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

static inline void rope_yarn_corr_dims(int   n_dims,
                                       int   n_ctx_orig,
                                       float freq_base,
                                       float beta_fast,
                                       float beta_slow,
                                       float dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    float start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (float) (n_dims - 1) ? end : (float) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

static const float rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
static const float rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
static const float rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
static const float rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
static const float rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
static const float rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

static inline uint64_t rope_ps_enter_fullmask(void) {
    uint64_t old_mask;
    __asm__ volatile(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    return old_mask;
}

static inline void rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ volatile("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

static inline void rope_poly_sin_block8(float * out, const float * x) {
    __asm__ volatile(
        "flw.ps    f0,  %[x]           \n\t"
        "fmul.ps   f1,  f0,  f0        \n\t"

        "flw.ps    f2,  %[c11]         \n\t"
        "flw.ps    f3,  %[c9]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c7]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c5]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c3]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[one]         \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "fmul.ps   f4,  f0,  f2        \n\t"
        "fsw.ps    f4,  %[out]         \n\t"
        : [out] "=m"(*(float (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(const float (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(const float (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(const float (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(const float (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(const float (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(const float (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(const float (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

static inline void rope_sincos_block8(float * sin8, float * cos8, const float * theta8) {
    float sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    float cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    float sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    float cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
        float x = theta8[i];

        if (x > ROPE_PI || x < -ROPE_PI) {
            float cycles = x * ROPE_INV_TWO_PI;
            int   n      = (int) cycles;
            if (x < 0.0f) {
                n--;
            }
            x = x - (float) n * ROPE_TWO_PI;
        }

        {
            float y = x;
            float s = 1.0f;
            if (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } else if (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            float y = x + ROPE_PI_OVER_2;
            if (y > ROPE_PI || y < -ROPE_PI) {
                float cycles = y * ROPE_INV_TWO_PI;
                int   n      = (int) cycles;
                if (y < 0.0f) {
                    n--;
                }
                y = y - (float) n * ROPE_TWO_PI;
            }

            float s = 1.0f;
            if (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } else if (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        const uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ volatile(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(float (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(float (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(const float (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(const float (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(const float (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(const float (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
static inline void rope_yarn_scalar(float       theta_extrap,
                                    float       freq_scale,
                                    const float corr_dims[2],
                                    int64_t     i0,
                                    float       ext_factor,
                                    float       mscale,
                                    float *     cos_theta,
                                    float *     sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta        = theta_interp;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int) i0) * ext_factor;
        theta          = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
    }

    *cos_theta = et_cosf(theta) * mscale;
    *sin_theta = et_sinf(theta) * mscale;
}

// Populate cos/sin cache for a given position using running theta product
// Experiment 1:
//   - theta construction and YaRN mixing stay scalar
//   - actual sin/cos approximation is done in vec8 blocks
static inline void compute_rope_cache(float *       cos_cache,
                                      float *       sin_cache,
                                      int32_t       n_dims,
                                      float         theta_scale,
                                      int32_t       pos,
                                      const float * freq_factors,
                                      float         freq_scale,
                                      const float   corr_dims[2],
                                      float         ext_factor,
                                      float         attn_factor) {
    const int32_t half_dims = n_dims / 2;
    float         theta     = 1.0f;

    int32_t dim_idx = 0;

    for (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        float theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        float theta_local = theta;
        float mscale      = attn_factor;

        if (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
            const int32_t pair_idx     = dim_idx + i;
            const float   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            const float   theta_base   = (float) pos * theta_local;
            const float   theta_extrap = et_fdiv(theta_base, ff);

            float theta_interp = freq_scale * theta_extrap;
            float theta_mix    = theta_interp;

            if (ext_factor != 0.0f) {
                float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    for (; dim_idx < half_dims; ++dim_idx) {
        const float ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        const float theta_base = (float) pos * theta;

        rope_yarn_scalar(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_idx * 2, ext_factor, attn_factor,
                         &cos_cache[dim_idx], &sin_cache[dim_idx]);

        theta *= theta_scale;
    }
}

//------------------------------------------------------------------------------
// IMROPE cache build (interleaved multi-modal RoPE for Qwen3VL)
//------------------------------------------------------------------------------

// Builds cos/sin cache with 4 interleaved position channels.
// Each dimension pair selects from {theta_t, theta_h, theta_w, theta_e}
// using a mod-3 sector pattern, matching the CPU reference exactly.
static inline void compute_imrope_cache(float *       cos_cache,
                                        float *       sin_cache,
                                        int32_t       n_dims,
                                        float         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        const int32_t sections[4],
                                        const float * freq_factors,
                                        float         freq_scale,
                                        const float   corr_dims[2],
                                        float         ext_factor,
                                        float         attn_factor) {
    const int32_t half_dims = n_dims / 2;
    const int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    float theta_t = (float) pos_t;
    float theta_h = (float) pos_h;
    float theta_w = (float) pos_w;
    float theta_e = (float) pos_e;

    int32_t dim_idx = 0;

    for (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        float theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        float mscale = attn_factor;

        if (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
            const int32_t pair_idx = dim_idx + i;
            const int32_t sector   = pair_idx % sect_dims;
            const float   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            float theta;
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }

            const float theta_extrap = et_fdiv(theta, ff);
            float       theta_interp = freq_scale * theta_extrap;
            float       theta_mix    = theta_interp;

            if (ext_factor != 0.0f) {
                float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;

            // All 4 thetas advance every iteration
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    for (; dim_idx < half_dims; ++dim_idx) {
        const int32_t sector = dim_idx % sect_dims;
        const float   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        float theta;
        if (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } else {
            theta = theta_e;
        }

        rope_yarn_scalar(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2, ext_factor, attn_factor,
                         &cos_cache[dim_idx], &sin_cache[dim_idx]);

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }
}

//------------------------------------------------------------------------------
// Entry point
//------------------------------------------------------------------------------

int entry_point(struct ggml_et_rope_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return -1;
    }

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * src1 = &params->src1;
    struct ggml_tensor * src2 = &params->src2;
    struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float *   src0_data    = (const float *) src0->data;
    const int32_t * src1_data    = (const int32_t *) src1->data;
    const float *   freq_factors = (src2 && src2->data) ? (const float *) src2->data : NULL;
    float *         dst_data     = (float *) dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1;
    }
#ifdef ET_UBERKERNEL
    const size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    const size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    const int64_t head_dim = src0->ne[0];
    const int64_t heads    = src0->ne[1];
    const int64_t seq_len  = src0->ne[2];
    const int64_t batch    = src0->ne[3];

    const rope_params_t * rope_params = &params->rope_params;
    const int32_t         n_dims      = rope_params->n_dims;
    const float           freq_base   = rope_params->freq_base;
    const float           freq_scale  = rope_params->freq_scale;
    const int32_t         mode        = rope_params->mode;

    if (n_dims <= 0 || n_dims > head_dim || (n_dims & 1) != 0) {
        return -1;
    }

    if (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        return -1;
    }

    float cos_cache[MAX_ROPE_HALF_DIMS];
    float sin_cache[MAX_ROPE_HALF_DIMS];

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    const int64_t total_heads = batch * seq_len * heads;
    const int64_t start_wu    = (total_heads * thread_id) / num_threads;
    const int64_t end_wu      = (total_heads * (thread_id + 1)) / num_threads;

    if (start_wu >= end_wu) {
        return 0;
    }

    const float   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (float) n_dims));
    const int32_t half_dims         = n_dims / 2;
    const int     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    const int     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    const int     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    for (int64_t wu = start_wu; wu < end_wu; ++wu) {
        const int64_t h = wu % heads;
        const int64_t s = (wu / heads) % seq_len;
        const int64_t b = wu / (heads * seq_len);

        if (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            const int32_t pt = src1_data[s] + rope_params->n_past;
            const int32_t ph = src1_data[s + seq_len] + rope_params->n_past;
            const int32_t pw = src1_data[s + seq_len * 2] + rope_params->n_past;
            const int32_t pe = src1_data[s + seq_len * 3] + rope_params->n_past;

            if (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_factor,
                                     rope_params->attn_factor);
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } else {
            const int32_t pos = src1_data[s] + rope_params->n_past;

            if (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        const float * head_src =
            (const float *) ((const char *) src0_data + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        float * head_dst = (float *) ((char *) dst_data + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        for (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        if (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t temp_mask;
            __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
            __asm__ volatile("mov.m.x m0, x0, 0xFF");

            for (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
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
                    : [x0_dst] "=m"(*(float (*)[8]) & head_dst[dim_idx]), [x1_dst] "=m"(*(float (*)[8]) &
                                                                                        head_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(const float (*)[8]) & head_src[dim_idx]),
                      [x1_src] "m"(*(const float (*)[8]) & head_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(const float (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(const float (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));
        } else {
            // Standard: adjacent pairs (2i, 2i+1)
            for (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                const int32_t dim_in_head = pair_idx * 2;
                const float   x0          = head_src[dim_in_head];
                const float   x1          = head_src[dim_in_head + 1];

                head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    return 0;
}
