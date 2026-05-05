#include "unary-ops.h"

static inline float op_abs(float x) {
    return fabsf(x);
}

static inline float op_sgn(float x) {
    return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

static inline float op_neg(float x) {
    return -x;
}

static inline float op_step(float x) {
    return (x > 0.f) ? 1.f : 0.f;
}

static inline float op_tanh(float x) {
    return tanhf(x);
}

static inline float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static inline float op_relu(float x) {
    return (x > 0.f) ? x : 0.f;
}

static inline float op_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

static inline float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_exp(float x) {
    return expf(x);
}

static inline float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_sqr(float x) {
    return x * x;
}

static inline float op_sqrt(float x) {
    return sqrtf(x);
}

static inline float op_xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
    if (x > 0.0f) {
        return alpha_p * x * x + beta * x;
    } else {
        const float min_x_eps = fminf(x, eps);
        return (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

static inline float op_sin(float x) {
    return sinf(x);
}

static inline float op_cos(float x) {
    return cosf(x);
}

static inline float op_log(float x) {
    return logf(x);
}

static inline float op_expm1(float x) {
    return expf(x) - 1.0f;
}

static inline float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static inline float op_floor(float x) {
    return floorf(x);
}

static inline float op_ceil(float x) {
    return ceilf(x);
}

static inline float op_round(float x) {
    return roundf(x);
}

static inline float op_trunc(float x) {
    return truncf(x);
}

static inline float act_quant_pow2_scale(float amax, float max_inv, float min_amax) {
    const float scaled = fmaxf(amax, min_amax) * max_inv;
    return exp2f(ceilf(log2f(scaled)));
}

static inline uint8_t fp32_to_fp8_e4m3fn(float x) {
    if (isnan(x)) {
        return 0x7F;
    }

    const uint8_t sign = signbit(x) ? 0x80 : 0x00;
    const float ax = fabsf(x);

    if (ax == 0.0f) {
        return sign;
    }

    if (ax < 0x1p-6f) {
        const int man = (int) roundf(ax * 512.0f);
        if (man <= 0) {
            return sign;
        }
        if (man >= 8) {
            return sign | 0x08;
        }
        return sign | (uint8_t) man;
    }

    int exp_unbiased;
    const float fr = frexpf(ax, &exp_unbiased);
    exp_unbiased -= 1;

    int exp = exp_unbiased + 7;
    int man = (int) roundf((2.0f * fr - 1.0f) * 8.0f);
    if (man == 8) {
        man = 0;
        exp++;
    }

    if (exp > 15 || (exp == 15 && man > 6)) {
        return sign | 0x7E;
    }

    return sign | (uint8_t) ((exp << 3) | man);
}

static inline float fp8_e4m3fn_to_fp32(uint8_t x) {
    if ((x & 0x7F) == 0) {
        return 0.0f;
    }
    if ((x & 0x7F) == 0x7F) {
        return NAN;
    }

    const int sign = x >> 7;
    const int exp  = (x >> 3) & 0x0F;
    const int man  = x & 0x07;
    const float val = exp == 0 ? ldexpf((float) man, -9) : ldexpf(1.0f + (float) man * 0.125f, exp - 7);

    return sign ? -val : val;
}

static inline float quant_dequant_fp8_e4m3(float x) {
    return fp8_e4m3fn_to_fp32(fp32_to_fp8_e4m3fn(fminf(fmaxf(x, -448.0f), 448.0f)));
}

static inline float quant_dequant_fp4_e2m1(float x) {
    static const float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
    };

    const float xc = fminf(fmaxf(x, -6.0f), 6.0f);
    int best = 0;
    float best_err = fabsf(values[0] - xc);
    for (int i = 1; i < 16; ++i) {
        const float err = fabsf(values[i] - xc);
        if (err < best_err) {
            best = i;
            best_err = err;
        }
    }

    return values[best];
}

template <int mode>
static inline float act_quant_max_value() {
    if constexpr (mode == 4) {
        return 6.0f;
    } else {
        return 448.0f;
    }
}

template <int mode>
static inline float act_quant_min_amax() {
    if constexpr (mode == 4) {
        return 0x1.8p-124f;
    } else {
        return 1.0e-4f;
    }
}

template <int mode>
static inline float act_quant_dequant(float x) {
    if constexpr (mode == 4) {
        return quant_dequant_fp4_e2m1(x);
    } else {
        return quant_dequant_fp8_e4m3(x);
    }
}

template <float (*op)(float), typename src0_t, typename dst_t>
static inline void vec_unary_op(int64_t n, dst_t * y, const src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <int block_size, int mode, typename src0_t, typename dst_t>
static inline void vec_act_quant_op(int64_t n, dst_t * y, const src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    GGML_ASSERT(n % block_size == 0);

    for (int64_t ib = 0; ib < n; ib += block_size) {
        float amax = 0.0f;
        for (int64_t i = 0; i < block_size; ++i) {
            const float v = fabsf(src0_to_f32(x[ib + i]));
            if (isfinite(v)) {
                amax = fmaxf(amax, v);
            }
        }

        const float scale = act_quant_pow2_scale(amax, 1.0f / act_quant_max_value<mode>(), act_quant_min_amax<mode>());
        const float iscale = 1.0f / scale;

        for (int64_t i = 0; i < block_size; ++i) {
            y[ib + i] = f32_to_dst(act_quant_dequant<mode>(src0_to_f32(x[ib + i]) * iscale) * scale);
        }
    }
}

template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

template <int block_size, int mode, typename src0_t, typename dst_t>
static void apply_act_quant_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->ne[0] % block_size == 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_act_quant_op<block_size, mode>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float)>
static void unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <int block_size, int mode>
static void act_quant_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        apply_act_quant_op<block_size, mode, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        apply_act_quant_op<block_size, mode, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <float (*op)(float, ggml_tensor *)>
static void unary_op_params(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static inline void vec_unary_op_functor(int64_t n, dst_t * y, const src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static void apply_unary_op_functor(const ggml_compute_params * params, ggml_tensor * dst, Op op) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
static void unary_op_functor(const ggml_compute_params * params, ggml_tensor * dst, Op op) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, float, float>(params, dst, op);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, float>(params, dst, op);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, float>(params, dst, op);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_compute_forward_abs(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

void ggml_compute_forward_sgn(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

void ggml_compute_forward_neg(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

void ggml_compute_forward_step(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

void ggml_compute_forward_tanh(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

void ggml_compute_forward_elu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

void ggml_compute_forward_relu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

void ggml_compute_forward_sigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

void ggml_compute_forward_hardsigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

void ggml_compute_forward_exp(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

void ggml_compute_forward_hardswish(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

void ggml_compute_forward_sqr(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

void ggml_compute_forward_sqrt(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

void ggml_compute_forward_sin(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

void ggml_compute_forward_cos(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

void ggml_compute_forward_log(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

void ggml_compute_forward_expm1(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

void ggml_compute_forward_softplus(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

void ggml_compute_forward_floor(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

void ggml_compute_forward_ceil(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

void ggml_compute_forward_round(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

void ggml_compute_forward_trunc(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

void ggml_compute_forward_fp4_act_quant(const ggml_compute_params * params, ggml_tensor * dst) {
    act_quant_op<32, 4>(params, dst);
}

void ggml_compute_forward_fp8_act_quant(const ggml_compute_params * params, ggml_tensor * dst) {
    act_quant_op<64, 8>(params, dst);
}

void ggml_compute_forward_sinkhorn_4x4(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->ne[0] == 4 && src0->ne[1] == 4);
    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));

    // Distribute the per-batch 4x4 problems across worker threads. Each
    // thread handles a slice of the batch dimension (src0->ne[2]).
    const int64_t n_batch = src0->ne[2];
    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t b0 = (n_batch * ith) / nth;
    const int64_t b1 = (n_batch * (ith + 1)) / nth;

    for (int64_t b = b0; b < b1; ++b) {
        const float * src = (const float *) ((const char *) src0->data + b * src0->nb[2]);
        float * out = (float *) ((char *) dst->data + b * dst->nb[2]);
        float x[4][4];

        for (int r = 0; r < 4; ++r) {
            float maxv = src[4*r + 0];
            for (int c = 1; c < 4; ++c) {
                maxv = fmaxf(maxv, src[4*r + c]);
            }

            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) {
                x[r][c] = expf(src[4*r + c] - maxv);
                sum += x[r][c];
            }

            const float inv_sum = 1.0f / sum;
            for (int c = 0; c < 4; ++c) {
                x[r][c] = fmaxf(x[r][c] * inv_sum, 1e-6f);
            }
        }

        for (int c = 0; c < 4; ++c) {
            float sum = 0.0f;
            for (int r = 0; r < 4; ++r) {
                sum += x[r][c];
            }
            const float inv_sum = 1.0f / fmaxf(sum, 1e-6f);
            for (int r = 0; r < 4; ++r) {
                x[r][c] *= inv_sum;
            }
        }

        for (int it = 1; it < 20; ++it) {
            for (int r = 0; r < 4; ++r) {
                float sum = 0.0f;
                for (int c = 0; c < 4; ++c) {
                    sum += x[r][c];
                }
                const float inv_sum = 1.0f / fmaxf(sum, 1e-6f);
                for (int c = 0; c < 4; ++c) {
                    x[r][c] *= inv_sum;
                }
            }

            for (int c = 0; c < 4; ++c) {
                float sum = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    sum += x[r][c];
                }
                const float inv_sum = 1.0f / fmaxf(sum, 1e-6f);
                for (int r = 0; r < 4; ++r) {
                    x[r][c] *= inv_sum;
                }
            }
        }

        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[4*r + c] = x[r][c];
            }
        }
    }
}

void ggml_compute_forward_xielu(const ggml_compute_params * params, ggml_tensor * dst) {
    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta = ggml_get_op_params_f32(dst, 3);
    const float eps = ggml_get_op_params_f32(dst, 4);

    const auto xielu_op_params = [alpha_n, alpha_p, beta, eps](float f) {
        return op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}
