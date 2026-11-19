#include "common.cuh"
#include "dsv4-hc.cuh"

#include <cfloat>

static constexpr int DSV4_HC = 4;

static __device__ float dsv4_hc_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static __device__ float4 dsv4_hc_comb_two(const float * comb, int32_t n_iter) {
    const float d0 = comb[0] - comb[1];
    const float d1 = comb[2] - comb[3];
    if (fabsf(d0) > FLT_MAX/8 || fabsf(d1) > FLT_MAX/8) {
        // Keep finite inputs valid when their differences exceed the F32 range.
        double logs[] = {comb[0], comb[1], comb[2], comb[3]};
        for (int32_t n = 0; n < n_iter; ++n) {
            for (int axis = 0; axis < 2; ++axis) {
                for (int row = 0; row < 2; ++row) {
                    const int i = axis == 0 ? 2*row : row;
                    const int j = i + (axis == 0 ? 1 : 2);
                    const double diff = logs[i] - logs[j];
                    const double norm = log1p(exp(-fabs(diff)));
                    logs[i] = -fmax(-diff, 0.0) - norm;
                    logs[j] = -fmax( diff, 0.0) - norm;
                }
            }
        }
        return make_float4(exp(logs[0]), exp(logs[1]), exp(logs[2]), exp(logs[3]));
    }
    float a = dsv4_hc_softplus(-d1) - dsv4_hc_softplus(-d0);
    float b = dsv4_hc_softplus(d1) - dsv4_hc_softplus(d0);
    if (n_iter > 1) {
        const float hi = fmaxf(a, 0.0f) + fmaxf(b, 0.0f);
        const float lo = -(fmaxf(-a, 0.0f) + fmaxf(-b, 0.0f));
        const float range = hi - lo;
        const float log_two = logf(2.0f);
        float power = -dsv4_hc_softplus(-range) + log_two;
        float sum = 0.0f;
        bool have_sum = false;
        // Power the normalization map for the requested finite iteration count.
        for (uint32_t n = n_iter - 1; n; n >>= 1) {
            if (n & 1) {
                sum = have_sum ? (sum + dsv4_hc_softplus(power - sum)) - dsv4_hc_softplus((sum + power) - range) : power;
                have_sum = true;
            }
            if (n > 1) {
                power = (power - dsv4_hc_softplus(2.0f*power - range)) + log_two;
            }
        }
        const float shift = dsv4_hc_softplus(sum - hi) - dsv4_hc_softplus(sum + lo);
        a += shift;
        b += shift;
    }
    return make_float4(1.0f / (1.0f + expf(-a)), 1.0f / (1.0f + expf(-b)),
                       1.0f / (1.0f + expf( a)), 1.0f / (1.0f + expf( b)));
}


template <bool gates>
static __global__ void dsv4_hc_comb_two_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float,
        int32_t n_iter,
        float * pre,
        float * post,
        int64_t sp,
        int64_t so) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_tokens) {
        return;
    }
    ggml_cuda_pdl_sync();

    if constexpr (gates) {
        for (int i = 0; i < 2; ++i) {
            const float p = mixes[i*sm0 + it*sm1] + base[i*sb0];
            const float q = mixes[(2 + i)*sm0 + it*sm1] + base[(2 + i)*sb0];
            pre[i + it*sp] = 1.0f / (1.0f + expf(-p));
            post[i + it*so] = __fmul_rn(2.0f, 1.0f / (1.0f + expf(-q)));
        }
    }

    float comb[4];
    for (int i = 0; i < 4; ++i) {
        comb[i] = mixes[(4 + i)*sm0 + it*sm1] * scale[2*ss0] + base[(4 + i)*sb0];
    }
    const float4 c = dsv4_hc_comb_two(comb, n_iter);
    dst[it*sd2]             = c.x;
    dst[sd0 + it*sd2]       = c.y;
    dst[sd1 + it*sd2]       = c.z;
    dst[sd0 + sd1 + it*sd2] = c.w;
}

template <bool vectorized>
static __global__ void dsv4_hc_route_pre_f32(
        const float * mixes, const float * bias, const float * residual,
        float * post, float * comb, float * dst,
        int64_t n_embd, int64_t n_tokens, int64_t sm0, int64_t sm1, int64_t sb0,
        int64_t sr0, int64_t sr1, int64_t sr2, int64_t so,
        int64_t sc0, int64_t sc1, int64_t sc2, int64_t sd0, int64_t sd1, int32_t n_iter) {
    ggml_cuda_pdl_lc();
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) {
        return;
    }
    ggml_cuda_pdl_sync();
    for (int64_t it = blockIdx.y; it < n_tokens; it += gridDim.y) {
        const float p0 = mixes[it*sm1] + bias[0];
        const float p1 = mixes[sm0 + it*sm1] + bias[sb0];
        const float w0 = 1.0f / (1.0f + expf(-p0));
        const float w1 = 1.0f / (1.0f + expf(-p1));
        if constexpr (vectorized) {
            const float4 a = ((const float4 *) residual)[j + it*sr2];
            const float4 b = ((const float4 *) residual)[j + sr1 + it*sr2];
            ((float4 *) dst)[j + it*sd1] = make_float4(
                __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a.x, w0)), __fmul_rn(b.x, w1)),
                __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a.y, w0)), __fmul_rn(b.y, w1)),
                __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a.z, w0)), __fmul_rn(b.z, w1)),
                __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a.w, w0)), __fmul_rn(b.w, w1)));
        } else {
            const float a = __fmul_rn(residual[j*sr0 + it*sr2], w0);
            const float b = __fmul_rn(residual[j*sr0 + sr1 + it*sr2], w1);
            dst[j*sd0 + it*sd1] = __fadd_rn(__fadd_rn(0.0f, a), b);
        }
        if (j == 0) {
            for (int k = 0; k < 2; ++k) {
                const float q = mixes[(2 + k)*sm0 + it*sm1] + bias[(2 + k)*sb0];
                post[k + it*so] = __fmul_rn(2.0f, 1.0f / (1.0f + expf(-q)));
            }
            float logits[4];
            for (int k = 0; k < 4; ++k) {
                logits[k] = mixes[(4 + k)*sm0 + it*sm1] + bias[(4 + k)*sb0];
            }
            const float4 c = dsv4_hc_comb_two(logits, n_iter);
            comb[it*sc2] = c.x;
            comb[sc0 + it*sc2] = c.y;
            comb[sc1 + it*sc2] = c.z;
            comb[sc0 + sc1 + it*sc2] = c.w;
        }
    }
}


static __global__ void __launch_bounds__(1024) dsv4_hc_route_pre_norm_f32(
        const float * mixes, const float * bias, const float4 * residual, const float4 * weights,
        float * post, float * comb, float4 * dst,
        int64_t sm0, int64_t sm1, int64_t sb0, int64_t so,
        int64_t sc0, int64_t sc1, int64_t sc2, int32_t n_iter, float eps) {
    const int j = threadIdx.x;
    const int64_t it = blockIdx.x;
    __shared__ float sums[32];
    ggml_cuda_pdl_sync();
    const float p0 = mixes[it*sm1] + bias[0];
    const float p1 = mixes[sm0 + it*sm1] + bias[sb0];
    const float w0 = 1.0f / (1.0f + expf(-p0));
    const float w1 = 1.0f / (1.0f + expf(-p1));
    const float4 a = residual[it*2048 + j];
    const float4 b = residual[it*2048 + 1024 + j];
    const float4 v = make_float4(
            __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a.x, w0)), __fmul_rn(b.x, w1)),
            __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a.y, w0)), __fmul_rn(b.y, w1)),
            __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a.z, w0)), __fmul_rn(b.z, w1)),
            __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a.w, w0)), __fmul_rn(b.w, w1)));
    float sum = 0.0f;
    sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    sum = block_reduce<block_reduce_method::SUM, 1024>(sum, sums);
    const float scale = rsqrtf(sum/4096 + eps);
    const float4 w = weights[j];
    dst[it*1024 + j] = make_float4(
            __fmul_rn(__fmul_rn(scale, v.x), w.x),
            __fmul_rn(__fmul_rn(scale, v.y), w.y),
            __fmul_rn(__fmul_rn(scale, v.z), w.z),
            __fmul_rn(__fmul_rn(scale, v.w), w.w));
    if (j == 0) {
        for (int k = 0; k < 2; ++k) {
            const float q = mixes[(2 + k)*sm0 + it*sm1] + bias[(2 + k)*sb0];
            post[k + it*so] = __fmul_rn(2.0f, 1.0f / (1.0f + expf(-q)));
        }
        float logits[4];
        for (int k = 0; k < 4; ++k) {
            logits[k] = mixes[(4 + k)*sm0 + it*sm1] + bias[(4 + k)*sb0];
        }
        const float4 c = dsv4_hc_comb_two(logits, n_iter);
        comb[it*sc2] = c.x;
        comb[sc0 + it*sc2] = c.y;
        comb[sc1 + it*sc2] = c.z;
        comb[sc0 + sc1 + it*sc2] = c.w;
    }
}

static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();
    for (int64_t it = blockIdx.y; it < n_tokens; it += gridDim.y) {
        float sum = 0.0f;
        for (int64_t ih = 0; ih < hc; ++ih) {
            const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
            float wv;
            if constexpr (gated) {
                wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
            } else {
                wv = weights[ih*sw0 + it*sw1];
            }
            // Preserve the two-stream pointwise products before their sum.
            sum = hc == 2 ? __fadd_rn(sum, __fmul_rn(xv, wv)) : sum + xv * wv;
        }

        dst[i0*sd0 + it*sd1] = scale * sum;
    }
}

// gated: post = gate_scale_out*sigmoid(gate_scale_in*post), the fused SCALE -> SIGMOID -> SCALE chain of qwen4exp
template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float gate_scale_in,
        float gate_scale_out) {
    ggml_cuda_pdl_lc();
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();
    for (int64_t idst = blockIdx.y; idst < hc; idst += gridDim.y) {
        for (int64_t it = blockIdx.z; it < n_tokens; it += gridDim.z) {
            float p = post[idst*sp0 + it*sp1];
            if constexpr (gated) {
                p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));
            }

            if (hc == 2) {
                float mixed;
                if constexpr (has_comb) {
                    const float a = __fmul_rn(residual[i0*sr0 + it*sr2], comb[idst*sc0 + it*sc2]);
                    const float b = __fmul_rn(residual[i0*sr0 + sr1 + it*sr2], comb[idst*sc0 + sc1 + it*sc2]);
                    mixed = __fadd_rn(a, b);
                } else {
                    mixed = residual[i0*sr0 + idst*sr1 + it*sr2];
                }
                dst[i0*sd0 + idst*sd1 + it*sd2] = __fadd_rn(mixed, __fmul_rn(x[i0*sx0 + it*sx1], p));
                continue;
            }

            float sum = x[i0*sx0 + it*sx1] * p;
            if constexpr (has_comb) {
                for (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } else {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        }
    }
}

template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_two_vec_f32(
        const float4 * x, const float4 * residual, const float * post, const float * comb, float4 * dst,
        int64_t n_embd4, int64_t n_tokens, int64_t sx1, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1, int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd1, int64_t sd2, float gate_in, float gate_out) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_embd4) {
        return;
    }
    ggml_cuda_pdl_sync();
    for (int64_t t = blockIdx.y; t < n_tokens; t += gridDim.y) {
        float p0 = post[t*sp1];
        float p1 = post[sp0 + t*sp1];
        if constexpr (gated) {
            p0 = gate_out * (1.0f / (1.0f + expf(-(gate_in * p0))));
            p1 = gate_out * (1.0f / (1.0f + expf(-(gate_in * p1))));
        }
        const float4 a = residual[i + t*sr2];
        const float4 b = residual[i + sr1 + t*sr2];
        const float4 v = x[i + t*sx1];
        float4 ra, rb;
        float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
        if constexpr (has_comb) {
            c00 = comb[t*sc2];
            c01 = comb[sc1 + t*sc2];
            c10 = comb[sc0 + t*sc2];
            c11 = comb[sc0 + sc1 + t*sc2];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float av = reinterpret_cast<const float *>(&a)[j];
            const float bv = reinterpret_cast<const float *>(&b)[j];
            const float vv = reinterpret_cast<const float *>(&v)[j];
            float ma = av, mb = bv;
            if constexpr (has_comb) {
                ma = __fadd_rn(__fmul_rn(av, c00), __fmul_rn(bv, c01));
                mb = __fadd_rn(__fmul_rn(av, c10), __fmul_rn(bv, c11));
            }
            reinterpret_cast<float *>(&ra)[j] = __fadd_rn(ma, __fmul_rn(vv, p0));
            reinterpret_cast<float *>(&rb)[j] = __fadd_rn(mb, __fmul_rn(vv, p1));
        }
        dst[i + t*sd2] = ra;
        dst[i + sd1 + t*sd2] = rb;
    }
}

static __global__ void __launch_bounds__(1024) dsv4_hc_post_norm_f32(
        const float4 * x, const float4 * residual, const float * post, const float * comb,
        float4 * raw, float4 * dst, int64_t sc0, int64_t sc1, int64_t sc2, float eps) {
    const int j = threadIdx.x;
    const int64_t it = blockIdx.x;
    __shared__ float sums[32];
    ggml_cuda_pdl_sync();
    const float4 a = residual[it*2048 + j];
    const float4 b = residual[it*2048 + 1024 + j];
    const float4 v = x[it*1024 + j];
    const float p0 = post[it*2];
    const float p1 = post[it*2 + 1];
    const float c00 = comb[it*sc2];
    const float c01 = comb[sc1 + it*sc2];
    const float c10 = comb[sc0 + it*sc2];
    const float c11 = comb[sc0 + sc1 + it*sc2];
    float4 ra, rb;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float av = reinterpret_cast<const float *>(&a)[k];
        const float bv = reinterpret_cast<const float *>(&b)[k];
        const float vv = reinterpret_cast<const float *>(&v)[k];
        const float ma = __fadd_rn(__fmul_rn(av, c00), __fmul_rn(bv, c01));
        const float mb = __fadd_rn(__fmul_rn(av, c10), __fmul_rn(bv, c11));
        reinterpret_cast<float *>(&ra)[k] = __fadd_rn(ma, __fmul_rn(vv, p0));
        reinterpret_cast<float *>(&rb)[k] = __fadd_rn(mb, __fmul_rn(vv, p1));
    }
    raw[it*2048 + j] = ra;
    raw[it*2048 + 1024 + j] = rb;
    float sum = 0.0f;
    sum += ra.x*ra.x + ra.y*ra.y + ra.z*ra.z + ra.w*ra.w;
    sum += rb.x*rb.x + rb.y*rb.y + rb.z*rb.z + rb.w*rb.w;
    sum = block_reduce<block_reduce_method::SUM, 1024>(sum, sums);
    const float scale = rsqrtf(sum/8192 + eps);
    dst[it*2048 + j] = make_float4(scale*ra.x, scale*ra.y, scale*ra.z, scale*ra.w);
    dst[it*2048 + 1024 + j] = make_float4(scale*rb.x, scale*rb.y, scale*rb.z, scale*rb.w);
}

void ggml_cuda_op_dsv4_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm) {
    const ggml_tensor * comb = post->src[3];
    const ggml_cuda_kernel_launch_params params(dim3(post->ne[2]), 1024, 0, ctx.stream());
    ggml_cuda_kernel_launch(dsv4_hc_post_norm_f32, params,
            (const float4 *) post->src[0]->data, (const float4 *) post->src[1]->data,
            (const float *) post->src[2]->data, (const float *) comb->data,
            (float4 *) post->data, (float4 *) norm->data,
            comb->nb[0]/sizeof(float), comb->nb[1]/sizeof(float), comb->nb[2]/sizeof(float),
            ggml_get_op_params_f32(norm, 0));
}

static void dsv4_hc_comb_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        ggml_tensor * pre, ggml_tensor * post) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t hc = dst->ne[0];
    const int64_t hc_mix_dim = (2 + hc)*hc;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(hc == DSV4_HC || hc == 2);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(hc != 2 || eps == 0.0f);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    if (hc == 2) {
        auto kernel = pre ? dsv4_hc_comb_two_f32<true> : dsv4_hc_comb_two_f32<false>;
        ggml_cuda_kernel_launch(kernel, launch_params,
                (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
                n_tokens, nbm0/sizeof(float), nbm1/sizeof(float), nbs0/sizeof(float), nbb0/sizeof(float),
                nbd0/sizeof(float), nbd1/sizeof(float), nbd2/sizeof(float), eps, n_iter,
                pre ? (float *) pre->data : nullptr, post ? (float *) post->data : nullptr,
                pre ? pre->nb[1]/sizeof(float) : 0, post ? post->nb[1]/sizeof(float) : 0);
        return;
    }
    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    dsv4_hc_comb_launch(ctx, dst, nullptr, nullptr);
}

void ggml_cuda_op_dsv4_hc_route(ggml_backend_cuda_context & ctx, ggml_tensor * pre, ggml_tensor * post, ggml_tensor * comb) {
    GGML_ASSERT(comb->ne[0] == 2 && pre->ne[0] == 2 && post->ne[0] == 2);
    GGML_ASSERT(pre->type == GGML_TYPE_F32 && post->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(pre) && ggml_is_contiguous(post));
    dsv4_hc_comb_launch(ctx, comb, pre, post);
}

void ggml_cuda_op_dsv4_hc_route_pre(ggml_backend_cuda_context & ctx, ggml_tensor * pre, ggml_tensor * post, ggml_tensor * comb) {
    const ggml_tensor * residual = pre->src[0];
    const ggml_tensor * mixes = comb->src[0];
    const ggml_tensor * bias = comb->src[2];
    const bool vectorized = residual->ne[0] % 4 == 0 && residual->nb[0] == sizeof(float) &&
            ggml_cuda_is_aligned(residual, 16) && ggml_cuda_is_aligned(pre, 16);
    const int width = vectorized ? 4 : 1;
    const int stride = vectorized ? sizeof(float4) : sizeof(float);
    const int block_size = mixes->ne[1] < 32 ? 64 : 256;
    const dim3 blocks((residual->ne[0]/width + block_size - 1)/block_size, std::min<int64_t>(mixes->ne[1], 65535));
    const ggml_cuda_kernel_launch_params params(blocks, dim3(block_size), 0, ctx.stream());
    auto kernel = vectorized ? dsv4_hc_route_pre_f32<true> : dsv4_hc_route_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, params,
            (const float *) mixes->data, (const float *) bias->data, (const float *) residual->data,
            (float *) post->data, (float *) comb->data, (float *) pre->data,
            residual->ne[0]/width, mixes->ne[1], mixes->nb[0]/sizeof(float), mixes->nb[1]/sizeof(float), bias->nb[0]/sizeof(float),
            residual->nb[0]/stride, residual->nb[1]/stride, residual->nb[2]/stride, post->nb[1]/sizeof(float),
            comb->nb[0]/sizeof(float), comb->nb[1]/sizeof(float), comb->nb[2]/sizeof(float),
            pre->nb[0]/stride, pre->nb[1]/stride, ggml_get_op_params_i32(comb, 1));
}

void ggml_cuda_op_dsv4_hc_route_pre_norm(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        const ggml_tensor * weights, ggml_tensor * dst, ggml_tensor * post, ggml_tensor * comb) {
    const ggml_tensor * residual = norm->src[0]->src[0];
    const ggml_tensor * mixes = comb->src[0];
    const ggml_tensor * bias = comb->src[2];
    const ggml_cuda_kernel_launch_params params(dim3(mixes->ne[1]), dim3(1024), 0, ctx.stream());
    ggml_cuda_kernel_launch(dsv4_hc_route_pre_norm_f32, params,
            (const float *) mixes->data, (const float *) bias->data, (const float4 *) residual->data,
            (const float4 *) weights->data, (float *) post->data, (float *) comb->data, (float4 *) dst->data,
            mixes->nb[0]/sizeof(float), mixes->nb[1]/sizeof(float), bias->nb[0]/sizeof(float), post->nb[1]/sizeof(float),
            comb->nb[0]/sizeof(float), comb->nb[1]/sizeof(float), comb->nb[2]/sizeof(float),
            ggml_get_op_params_i32(comb, 1), ggml_get_op_params_f32(norm, 0));
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, std::min<int64_t>(n_tokens, 65535), 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    // when fused, read the raw post weights from before the SCALE -> SIGMOID -> SCALE chain
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];

    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = n_tokens < 32 ? 64 : 256;
    if (hc == 2 && n_embd % 4 == 0 && nbx0 == sizeof(float) && nbr0 == sizeof(float) &&
            ggml_cuda_is_aligned(x, 16) && ggml_cuda_is_aligned(residual, 16) && ggml_cuda_is_aligned(dst, 16)) {
        const dim3 blocks((n_embd/4 + block_size - 1) / block_size, std::min<int64_t>(n_tokens, 65535));
        const ggml_cuda_kernel_launch_params params(blocks, dim3(block_size), 0, ctx.stream());
        auto kernel = comb ? (gated ? dsv4_hc_post_two_vec_f32<true, true> : dsv4_hc_post_two_vec_f32<true, false>)
                           : (gated ? dsv4_hc_post_two_vec_f32<false, true> : dsv4_hc_post_two_vec_f32<false, false>);
        ggml_cuda_kernel_launch(kernel, params,
                (const float4 *) x->data, (const float4 *) residual->data, (const float *) post->data,
                comb ? (const float *) comb->data : nullptr, (float4 *) dst->data,
                n_embd/4, n_tokens, nbx1/sizeof(float4), nbr1/sizeof(float4), nbr2/sizeof(float4),
                nbp0/sizeof(float), nbp1/sizeof(float), nbc0/sizeof(float), nbc1/sizeof(float), nbc2/sizeof(float),
                nbd1/sizeof(float4), nbd2/sizeof(float4), gate_in, gate_out);
        return;
    }
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, std::min<int64_t>(hc, 65535), std::min<int64_t>(n_tokens, 65535));
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? (gated ? dsv4_hc_post_f32<true,  true> : dsv4_hc_post_f32<true,  false>)
                       : (gated ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
}
