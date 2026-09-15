#include <hexagon_protos.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>

#include "groupnorm-ops.h"
#include "htp-ctx.h"
#include "hvx-reduce.h"
#include "hvx-sigmoid.h"
#include "hvx-utils.h"

struct group_norm_state {
    struct htp_context * ctx;
    const void * src;
    const float * weight;
    const float * bias;
    void * dst;
    uint32_t spatial;
    uint32_t width;
    uint32_t channels;
    uint32_t groups;
    uint32_t channels_per_group;
    uint32_t batches;
    uint32_t element_size;
    float epsilon;
    uint32_t flags;
    uint8_t * vtcm;
    uint32_t vtcm_size;
    atomic_uint next_job;
};

struct group_norm_accum {
    HVX_Vector sum[4];
    HVX_Vector square[4];
};

typedef void (*group_norm_accumulate_fn)(struct group_norm_accum *, const void *, uint32_t);
typedef void (*group_norm_transform_fn)(const struct group_norm_state *, uint32_t, uint32_t, const void *, void *, uint32_t, HVX_Vector, HVX_Vector);

static inline void group_norm_accum_init(struct group_norm_accum * acc) {
    const HVX_Vector zero = Q6_V_vzero();
    for (int i = 0; i < 4; ++i) {
        acc->sum[i] = zero;
        acc->square[i] = zero;
    }
}

static inline void group_norm_accumulate_vector(struct group_norm_accum * acc, int lane, HVX_Vector x) {
#if __HVX_ARCH__ >= 79
    acc->sum[lane] = Q6_Vsf_vadd_VsfVsf(acc->sum[lane], x);
    acc->square[lane] = Q6_Vsf_vadd_VsfVsf(acc->square[lane], Q6_Vsf_vmpy_VsfVsf(x, x));
#else
    const HVX_Vector zero = Q6_V_vzero();
    acc->sum[lane] = Q6_Vqf32_vadd_Vqf32Vqf32(acc->sum[lane], Q6_Vqf32_vadd_VsfVsf(x, zero));
    acc->square[lane] = Q6_Vqf32_vadd_Vqf32Vqf32(acc->square[lane], Q6_Vqf32_vmpy_VsfVsf(x, x));
#endif
}

static inline __attribute__((always_inline)) void group_norm_accumulate_f32(struct group_norm_accum * acc, const void * data, uint32_t n) {
    const HVX_Vector * src = (const HVX_Vector *) data;
    const uint32_t nvec = n / 32u;
    uint32_t i = 0;

    for (; i + 3u < nvec; i += 4u) {
#pragma unroll(4)
        for (int lane = 0; lane < 4; ++lane) {
            group_norm_accumulate_vector(acc, lane, src[i + lane]);
        }
    }
    for (; i < nvec; ++i) {
        group_norm_accumulate_vector(acc, 0, src[i]);
    }
}

static inline __attribute__((always_inline)) void group_norm_accumulate_f16(struct group_norm_accum * acc, const void * data, uint32_t n) {
    const __fp16 * src = (const __fp16 *) data;
    const uint32_t nvec = n / 64u;
    uint32_t i = 0;

    for (; i + 1u < nvec; i += 2u) {
        const HVX_VectorPair p0 = hvx_vec_f16_to_f32(hvx_vmemu(src + (size_t) i * 64u));
        const HVX_VectorPair p1 = hvx_vec_f16_to_f32(hvx_vmemu(src + (size_t) (i + 1u) * 64u));
        const HVX_Vector x0 = Q6_V_lo_W(p0);
        const HVX_Vector x1 = Q6_V_hi_W(p0);
        const HVX_Vector x2 = Q6_V_lo_W(p1);
        const HVX_Vector x3 = Q6_V_hi_W(p1);
        acc->sum[0] = hvx_vec_add_f32_f32(acc->sum[0], x0);
        acc->sum[1] = hvx_vec_add_f32_f32(acc->sum[1], x1);
        acc->sum[2] = hvx_vec_add_f32_f32(acc->sum[2], x2);
        acc->sum[3] = hvx_vec_add_f32_f32(acc->sum[3], x3);
        acc->square[0] = hvx_vec_add_f32_f32(acc->square[0], hvx_vec_mul_f32_f32(x0, x0));
        acc->square[1] = hvx_vec_add_f32_f32(acc->square[1], hvx_vec_mul_f32_f32(x1, x1));
        acc->square[2] = hvx_vec_add_f32_f32(acc->square[2], hvx_vec_mul_f32_f32(x2, x2));
        acc->square[3] = hvx_vec_add_f32_f32(acc->square[3], hvx_vec_mul_f32_f32(x3, x3));
    }
    if (i < nvec) {
        const HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vmemu(src + (size_t) i * 64u));
        const HVX_Vector x0 = Q6_V_lo_W(p);
        const HVX_Vector x1 = Q6_V_hi_W(p);
        acc->sum[0] = hvx_vec_add_f32_f32(acc->sum[0], x0);
        acc->sum[1] = hvx_vec_add_f32_f32(acc->sum[1], x1);
        acc->square[0] = hvx_vec_add_f32_f32(acc->square[0], hvx_vec_mul_f32_f32(x0, x0));
        acc->square[1] = hvx_vec_add_f32_f32(acc->square[1], hvx_vec_mul_f32_f32(x1, x1));
    }

    const uint32_t tail = n % 64u;
    if (tail) {
        HVX_Vector x = hvx_vmemu(src + (size_t) nvec * 64u);
        x = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(tail * sizeof(__fp16)), x, Q6_V_vzero());
        const HVX_VectorPair p = hvx_vec_f16_to_f32(x);
        const HVX_Vector x0 = Q6_V_lo_W(p);
        const HVX_Vector x1 = Q6_V_hi_W(p);
        acc->sum[0] = hvx_vec_add_f32_f32(acc->sum[0], x0);
        acc->sum[1] = hvx_vec_add_f32_f32(acc->sum[1], x1);
        acc->square[0] = hvx_vec_add_f32_f32(acc->square[0], hvx_vec_mul_f32_f32(x0, x0));
        acc->square[1] = hvx_vec_add_f32_f32(acc->square[1], hvx_vec_mul_f32_f32(x1, x1));
    }
}

static inline void group_norm_accum_finish(const struct group_norm_accum * acc, uint32_t n, float * mean, float * mean_square) {
#if __HVX_ARCH__ >= 79
    const HVX_Vector sum = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(acc->sum[0], acc->sum[1]), Q6_Vsf_vadd_VsfVsf(acc->sum[2], acc->sum[3]));
    const HVX_Vector square = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(acc->square[0], acc->square[1]), Q6_Vsf_vadd_VsfVsf(acc->square[2], acc->square[3]));
    *mean = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(sum)) / (float) n;
    *mean_square = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(square)) / (float) n;
#else
    const HVX_Vector sum = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(acc->sum[0], acc->sum[1]), Q6_Vqf32_vadd_Vqf32Vqf32(acc->sum[2], acc->sum[3]));
    const HVX_Vector square = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(acc->square[0], acc->square[1]), Q6_Vqf32_vadd_Vqf32Vqf32(acc->square[2], acc->square[3]));
    *mean = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum))) / (float) n;
    *mean_square = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(square))) / (float) n;
#endif
}

static inline void group_norm_dma_push(dma_queue * queue, void * dst, const void * src, uint32_t bytes) {
    while (!dma_queue_push(queue, dma_make_ptr(dst, src), bytes, bytes, bytes, 1)) {
        dma_queue_flush(queue);
    }
}

static inline __attribute__((always_inline)) void group_norm_stats_dma(const struct group_norm_state * st, dma_queue * queue, const void * src, uint32_t n, void * slots[2], uint32_t chunk, group_norm_accumulate_fn accumulate, float * mean, float * mean_square) {
    const uint32_t chunks = (n + chunk - 1u) / chunk;
    for (uint32_t i = 0; i < chunks && i < 2u; ++i) {
        const uint32_t count = hex_smin(n - i * chunk, chunk);
        group_norm_dma_push(queue, slots[i], (const uint8_t *) src + (size_t) i * chunk * st->element_size, count * st->element_size);
    }

    struct group_norm_accum acc;
    group_norm_accum_init(&acc);
    for (uint32_t i = 0; i < chunks; ++i) {
        const uint32_t count = hex_smin(n - i * chunk, chunk);
        const dma_ptr ready = dma_queue_pop(queue);
        accumulate(&acc, (const void *) ready.dst, count);
        if (i + 2u < chunks) {
            const uint32_t next = i + 2u;
            const uint32_t next_count = hex_smin(n - next * chunk, chunk);
            group_norm_dma_push(queue, slots[i & 1u], (const uint8_t *) src + (size_t) next * chunk * st->element_size, next_count * st->element_size);
        }
    }
    dma_queue_flush(queue);
    group_norm_accum_finish(&acc, n, mean, mean_square);
}

static inline __attribute__((always_inline)) void group_norm_transform_f32(const struct group_norm_state * st, uint32_t channel0, uint32_t group_offset, const void * src_data, void * dst_data, uint32_t n, HVX_Vector mean, HVX_Vector inv_std) {
    const float * src = (const float *) src_data;
    float * dst = (float *) dst_data;
    const bool affine = (st->flags & HTP_GROUP_NORM_AFFINE) != 0;
    const bool silu = (st->flags & HTP_GROUP_NORM_SILU) != 0;
    uint32_t done = 0;

    while (done < n) {
        const uint32_t offset = group_offset + done;
        const uint32_t channel = channel0 + offset / st->spatial;
        const uint32_t in_channel = offset % st->spatial;
        const uint32_t count = hex_smin(st->spatial - in_channel, n - done);
        const HVX_Vector weight = hvx_vec_splat_f32(affine ? st->weight[channel] : 1.0f);
        const HVX_Vector bias = hvx_vec_splat_f32(affine ? st->bias[channel] : 0.0f);
        const HVX_Vector scale = hvx_vec_mul_f32_f32(inv_std, weight);
        const HVX_Vector offset_vec = hvx_vec_sub_f32_f32(bias, hvx_vec_mul_f32_f32(mean, scale));
        const HVX_Vector * input = (const HVX_Vector *) (src + done);
        HVX_Vector * output = (HVX_Vector *) (dst + done);
        const uint32_t nvec = count / 32u;

#pragma unroll(4)
        for (uint32_t i = 0; i < nvec; ++i) {
            output[i] = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(input[i], scale), offset_vec);
        }
        done += count;
    }

    if (silu) {
        for (uint32_t i = 0; i < n; i += st->width) {
            hvx_sigmoid_f32_aa((uint8_t *) (dst + i), (const uint8_t *) (dst + i), st->width);
            hvx_mul_f32_aaa((uint8_t *) (dst + i), (const uint8_t *) (src + i), (const uint8_t *) (dst + i), st->width);
        }
    }
}

static const uint16_t group_norm_silu_lut_a[64] __attribute__((aligned(128))) = {
    0x997f, 0x9c3f, 0x9e88, 0xa0ff, 0xa399, 0xa5bb, 0xa849, 0xaa54,
    0xac9b, 0xae91, 0xb08e, 0xb212, 0xb3a1, 0xb461, 0xb44e, 0xb20a,
    0x0000, 0x34fb, 0x39d9, 0x3ce8, 0x3f0c, 0x409f, 0x41b7, 0x42cb,
    0x43db, 0x4473, 0x44f7, 0x457a, 0x45fc, 0x467e, 0x46fe, 0x477f,
};

static const uint16_t group_norm_silu_lut_b[64] __attribute__((aligned(128))) = {
    0x94b9, 0x973b, 0x9981, 0x9c28, 0x9e3a, 0xa09c, 0xa2ba, 0xa4d1,
    0xa6b7, 0xa881, 0xa9b1, 0xaa81, 0xaa14, 0xa60e, 0x283e, 0x301d,
    0x3404, 0x35f7, 0x3779, 0x382f, 0x3860, 0x3867, 0x385a, 0x3848,
    0x3836, 0x3827, 0x381b, 0x3812, 0x380c, 0x3808, 0x3806, 0x3804,
};

static const uint16_t group_norm_silu_lut_c[64] __attribute__((aligned(128))) = {
    0x8d14, 0x8fa2, 0x91ae, 0x942c, 0x9607, 0x9841, 0x99d0, 0x9b8f,
    0x9c85, 0x9c99, 0x99ed, 0x182f, 0x2245, 0x2741, 0x2a05, 0x2bb7,
    0x2bb7, 0x2a05, 0x2741, 0x2245, 0x182f, 0x99ed, 0x9c99, 0x9c85,
    0x9b8f, 0x99d0, 0x9841, 0x9607, 0x942c, 0x91ae, 0x8fa2, 0x8d14,
};

static inline HVX_Vector group_norm_lut32_f16(HVX_Vector index, HVX_Vector table) {
    HVX_VectorPair values = Q6_Wh_vlut16_VbVhR(index, table, 0);
    values = Q6_Wh_vlut16or_WhVbVhR(values, index, table, 1);
    return Q6_V_lo_W(values);
}

static inline HVX_Vector group_norm_silu_f16(HVX_Vector x, HVX_Vector lut_a, HVX_Vector lut_b, HVX_Vector lut_c) {
    const HVX_Vector scaled = hvx_vec_mul_f16_f16(hvx_vec_add_f16_f16(x, Q6_Vh_vsplat_R(0x4800)), Q6_Vh_vsplat_R(0x4000));
    const HVX_Vector index_h = Q6_Vh_equals_Vhf(scaled);
    const HVX_Vector fraction = hvx_vec_sub_f16_f16(scaled, Q6_Vhf_equals_Vh(index_h));
    const HVX_Vector index = Q6_Vb_vshuffe_VbVb(index_h, index_h);
    const HVX_Vector a = group_norm_lut32_f16(index, lut_a);
    const HVX_Vector b = group_norm_lut32_f16(index, lut_b);
    const HVX_Vector c = group_norm_lut32_f16(index, lut_c);
    HVX_Vector y = hvx_vec_add_f16_f16(a, hvx_vec_mul_f16_f16(fraction, hvx_vec_add_f16_f16(b, hvx_vec_mul_f16_f16(fraction, c))));
    y = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(Q6_Vh_vsplat_R(0x4800), x), y, x);
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(Q6_Vh_vsplat_R(0xc800), x), Q6_V_vzero(), y);
}

static inline __attribute__((always_inline)) void group_norm_transform_f16(const struct group_norm_state * st, uint32_t channel0, uint32_t group_offset, const void * src_data, void * dst_data, uint32_t n, HVX_Vector mean, HVX_Vector inv_std) {
    const __fp16 * src = (const __fp16 *) src_data;
    __fp16 * dst = (__fp16 *) dst_data;
    const bool affine = (st->flags & HTP_GROUP_NORM_AFFINE) != 0;
    const bool silu = (st->flags & HTP_GROUP_NORM_SILU) != 0;
    const HVX_Vector lut_a = Q6_Vh_vshuff_Vh(hvx_vmem(group_norm_silu_lut_a));
    const HVX_Vector lut_b = Q6_Vh_vshuff_Vh(hvx_vmem(group_norm_silu_lut_b));
    const HVX_Vector lut_c = Q6_Vh_vshuff_Vh(hvx_vmem(group_norm_silu_lut_c));
    const float mean_f32 = hvx_vec_get_f32(mean);
    const float inv_std_f32 = hvx_vec_get_f32(inv_std);
    uint32_t done = 0;

    while (done < n) {
        const uint32_t offset = group_offset + done;
        const uint32_t channel = channel0 + offset / st->spatial;
        const uint32_t in_channel = offset % st->spatial;
        const uint32_t count = hex_smin(st->spatial - in_channel, n - done);
        const float weight = affine ? st->weight[channel] : 1.0f;
        const HVX_Vector scale = hvx_vec_splat_f32(weight * inv_std_f32);
        const HVX_Vector offset_vec = hvx_vec_splat_f32((affine ? st->bias[channel] : 0.0f) - mean_f32 * weight * inv_std_f32);
        const uint32_t nvec = count / 64u;

#pragma unroll(4)
        for (uint32_t i = 0; i < nvec; ++i) {
            const HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vmemu(src + done + (size_t) i * 64u));
            const HVX_Vector y0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(Q6_V_lo_W(p), scale), offset_vec);
            const HVX_Vector y1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(Q6_V_hi_W(p), scale), offset_vec);
            HVX_Vector y = hvx_vec_f32_to_f16(y0, y1);
            if (silu) {
                y = group_norm_silu_f16(y, lut_a, lut_b, lut_c);
            }
            hvx_vmemu(dst + done + (size_t) i * 64u) = y;
        }

        const uint32_t tail = count % 64u;
        if (tail) {
            const HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vmemu(src + done + (size_t) nvec * 64u));
            const HVX_Vector y0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(Q6_V_lo_W(p), scale), offset_vec);
            const HVX_Vector y1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(Q6_V_hi_W(p), scale), offset_vec);
            HVX_Vector y = hvx_vec_f32_to_f16(y0, y1);
            if (silu) {
                y = group_norm_silu_f16(y, lut_a, lut_b, lut_c);
            }
            hvx_vec_store_u(dst + done + (size_t) nvec * 64u, tail * sizeof(__fp16), y);
        }
        done += count;
    }
}

static inline __attribute__((always_inline)) void group_norm_output_dma(const struct group_norm_state * st, dma_queue * queue, const void * src, void * dst, uint32_t channel0, uint32_t n, void * slots[2], uint32_t chunk, group_norm_transform_fn transform, HVX_Vector mean, HVX_Vector inv_std) {
    const uint32_t chunks = (n + chunk - 1u) / chunk;
    for (uint32_t i = 0; i < chunks && i < 2u; ++i) {
        const uint32_t count = hex_smin(n - i * chunk, chunk);
        group_norm_dma_push(queue, slots[i], (const uint8_t *) src + (size_t) i * chunk * st->element_size, count * st->element_size);
    }

    for (uint32_t i = 0; i < chunks; ++i) {
        if (i >= 2u) {
            (void) dma_queue_pop(queue);
        }
        const dma_ptr ready = dma_queue_pop(queue);
        const uint32_t count = hex_smin(n - i * chunk, chunk);
        transform(st, channel0, i * chunk, (const void *) ready.dst, (void *) ready.dst, count, mean, inv_std);
        group_norm_dma_push(queue, (uint8_t *) dst + (size_t) i * chunk * st->element_size, (const void *) ready.dst, count * st->element_size);
        if (i + 2u < chunks) {
            const uint32_t next = i + 2u;
            const uint32_t next_count = hex_smin(n - next * chunk, chunk);
            group_norm_dma_push(queue, slots[i & 1u], (const uint8_t *) src + (size_t) next * chunk * st->element_size, next_count * st->element_size);
        }
    }
    dma_queue_flush(queue);
}

#define GROUP_NORM_WORKER(name, type, chunk_expr, accumulate, transform) \
static void name(unsigned int nth, unsigned int ith, void * data) { \
    struct group_norm_state * st = (struct group_norm_state *) data; \
    const uint32_t per_thread = (st->vtcm_size / nth) & ~2047u; \
    const uint32_t slot_bytes = (per_thread / 2u) & ~2047u; \
    const uint32_t chunk = (chunk_expr); \
    uint8_t * thread_vtcm = st->vtcm + (size_t) ith * per_thread; \
    void * slots[2] = { thread_vtcm, thread_vtcm + slot_bytes }; \
    dma_queue * queue = st->ctx->dma[ith]; \
    const uint32_t jobs = st->groups * st->batches; \
    for (;;) { \
        const uint32_t job = atomic_fetch_add_explicit(&st->next_job, 1u, memory_order_relaxed); \
        if (job >= jobs) { \
            break; \
        } \
        const uint32_t batch = job / st->groups; \
        const uint32_t group = job - batch * st->groups; \
        const uint32_t channel0 = group * st->channels_per_group; \
        const uint32_t group_elems = st->channels_per_group * st->spatial; \
        const size_t element_offset = (size_t) batch * st->channels * st->spatial + (size_t) channel0 * st->spatial; \
        const type * group_src = (const type *) st->src + element_offset; \
        type * group_dst = (type *) st->dst + element_offset; \
        float mean; \
        float mean_square; \
        group_norm_stats_dma(st, queue, group_src, group_elems, slots, chunk, accumulate, &mean, &mean_square); \
        const float variance = fmaxf(mean_square - mean * mean, 0.0f); \
        const float inv_std = 1.0f / sqrtf(variance + st->epsilon); \
        group_norm_output_dma(st, queue, group_src, group_dst, channel0, group_elems, slots, chunk, transform, hvx_vec_splat_f32(mean), hvx_vec_splat_f32(inv_std)); \
    } \
}

GROUP_NORM_WORKER(group_norm_worker_f32, float, ((slot_bytes / sizeof(float)) / st->width) * st->width, group_norm_accumulate_f32, group_norm_transform_f32)
GROUP_NORM_WORKER(group_norm_worker_f16, __fp16, (slot_bytes / sizeof(__fp16)) & ~63u, group_norm_accumulate_f16, group_norm_transform_f16)

int op_group_norm(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    const struct htp_group_norm_kernel_params * params = (const struct htp_group_norm_kernel_params *) octx->kernel_params;
    const bool affine = (params->flags & HTP_GROUP_NORM_AFFINE) != 0;
    const struct htp_tensor * weight = affine ? octx->src[1] : NULL;
    const struct htp_tensor * bias = affine ? octx->src[2] : NULL;
    const uint32_t groups = (uint32_t) octx->op_params[0];
    const bool f16 = src && src->type == HTP_TYPE_F16;
    float epsilon;
    __builtin_memcpy(&epsilon, &octx->op_params[1], sizeof(epsilon));

    if (!src || !dst || (!f16 && (params->flags & HTP_GROUP_NORM_SILU)) ||
        !((src->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F32) || (f16 && dst->type == HTP_TYPE_F16)) ||
        groups == 0 || src->ne[2] % groups != 0 || (!f16 && (src->ne[0] * src->ne[1]) % 32u != 0) ||
        src->ne[0] != dst->ne[0] || src->ne[1] != dst->ne[1] ||
        src->ne[2] != dst->ne[2] || src->ne[3] != dst->ne[3]) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint64_t weight_elems = weight ? (uint64_t) weight->ne[0] * weight->ne[1] * weight->ne[2] * weight->ne[3] : 0;
    const uint64_t bias_elems = bias ? (uint64_t) bias->ne[0] * bias->ne[1] * bias->ne[2] * bias->ne[3] : 0;
    if (affine && (!weight || !bias || weight->type != HTP_TYPE_F32 || bias->type != HTP_TYPE_F32 || weight_elems != src->ne[2] || bias_elems != src->ne[2])) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    struct group_norm_state state = {
        .ctx = octx->ctx,
        .src = (const void *) src->data,
        .weight = affine ? (const float *) weight->data : NULL,
        .bias = affine ? (const float *) bias->data : NULL,
        .dst = (void *) dst->data,
        .spatial = src->ne[0] * src->ne[1],
        .width = src->ne[0],
        .channels = src->ne[2],
        .groups = groups,
        .channels_per_group = src->ne[2] / groups,
        .batches = src->ne[3],
        .element_size = f16 ? sizeof(__fp16) : sizeof(float),
        .epsilon = epsilon,
        .flags = params->flags,
        .vtcm = (uint8_t *) octx->ctx->vtcm_base,
        .vtcm_size = octx->ctx->vtcm_size,
        .next_job = 0,
    };
    work_queue_run(octx->ctx->worker_pool, f16 ? group_norm_worker_f16 : group_norm_worker_f32, &state, octx->n_threads);
    asm volatile("syncht" ::: "memory");
    return HTP_STATUS_OK;
}
