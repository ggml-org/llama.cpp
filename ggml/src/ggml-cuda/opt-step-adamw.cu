#include "ggml-impl.h"
#include "opt-step-adamw.cuh"

#include <cstdint>

static __global__ void opt_step_adamw_f32(
    float * __restrict__ x, const float * __restrict__ g, float * __restrict__ g_m, float * __restrict__ g_v,
    const float * __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];
    const float gclip  = pars[7]; // element-wise gradient clip (0 = disabled)

    const float gi = (gclip > 0.0f) ? fmaxf(-gclip, fminf(gclip, g[i])) : g[i];
    const float gmi = g_m[i]*beta1 +    gi*(1.0f - beta1);
    const float gvi = g_v[i]*beta2 + gi*gi*(1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    const float mh =       gmi*beta1h;
    const float vh = sqrtf(gvi*beta2h) + eps;

    x[i] = x[i]*(1.0f - alpha*wd) - alpha*mh/vh;
}

static void opt_step_adamw_f32_cuda(
    float * x, const float * g, float * g_m, float * g_v, const float * pars, const int64_t k, cudaStream_t stream) {

    const dim3 block_dims(CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_ADAMW_BLOCK_SIZE - 1) / CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    opt_step_adamw_f32<<<block_nums, block_dims, 0, stream>>>(x, g, g_m, g_v, pars, k);
}

static __global__ void opt_step_adamw_q8_0(
        float * __restrict__ x, const float * __restrict__ g,
        block_q8_0 * __restrict__ g_m, block_q8_0 * __restrict__ g_v,
        const float * __restrict__ pars, const int64_t k) {
    const int ib = blockIdx.x;
    const int iq = threadIdx.x;
    const int64_t i = (int64_t) ib*QK8_0 + iq;

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];
    const float gclip  = pars[7];

    const float m_old = __half2float(g_m[ib].d)*g_m[ib].qs[iq];
    const float v_old = __half2float(g_v[ib].d)*g_v[ib].qs[iq];
    float gi = i < k ? g[i] : 0.0f;
    gi = isfinite(gi) ? gi : 0.0f;
    if (gclip > 0.0f) {
        gi = fmaxf(-gclip, fminf(gclip, gi));
    }

    const float m_new = m_old*beta1 + gi*(1.0f - beta1);
    const float v_new = fmaxf(8.0e-6f, v_old*beta2 + gi*gi*(1.0f - beta2));

    if (i < k) {
        const float update = alpha*(m_new*beta1h)/(sqrtf(v_new*beta2h) + eps);
        if (isfinite(update) && isfinite(x[i])) {
            x[i] = x[i]*(1.0f - alpha*wd) - update;
        }
    }

    const float m_d = warp_reduce_max(fabsf(m_new))/127.0f;
    const float v_d = warp_reduce_max(fabsf(v_new))/127.0f;
    g_m[ib].d = __float2half(m_d);
    g_v[ib].d = __float2half(v_d);
    const int m_q = m_d ? (int) roundf(m_new/m_d) : 0;
    const int v_q = v_d ? (int) roundf(v_new/v_d) : 0;
    g_m[ib].qs[iq] = (int8_t) fmaxf(-127.0f, fminf(127.0f, (float) m_q));
    g_v[ib].qs[iq] = (int8_t) fmaxf(-127.0f, fminf(127.0f, (float) v_q));
}

static void opt_step_adamw_q8_0_cuda(
        float * x, const float * g, block_q8_0 * g_m, block_q8_0 * g_v,
        const float * pars, const int64_t k, cudaStream_t stream) {
    const dim3 block_dims(QK8_0, 1, 1);
    const dim3 block_nums((k + QK8_0 - 1)/QK8_0, 1, 1);
    opt_step_adamw_q8_0<<<block_nums, block_dims, 0, stream>>>(x, g, g_m, g_v, pars, k);
}

void ggml_cuda_opt_step_adamw(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * src0_grad    = dst->src[1];
    const ggml_tensor * src0_grad_m  = dst->src[2];
    const ggml_tensor * src0_grad_v  = dst->src[3];
    const ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(src0->type         == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type    == GGML_TYPE_F32);
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_m));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_v));
    GGML_ASSERT(ggml_is_contiguous(adamw_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(adamw_params) == 8);

    float       * src0_d         = (float       *) src0->data;
    const float * src0_grad_d    = (const float *) src0_grad->data;
    const float * adamw_params_d = (const float *) adamw_params->data;

    cudaStream_t stream = ctx.stream();

    const int64_t ne = ggml_nelements(src0);

    if (src0_grad_m->type == GGML_TYPE_F32 && src0_grad_v->type == GGML_TYPE_F32) {
        GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
        GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
        opt_step_adamw_f32_cuda(src0_d, src0_grad_d,
            (float *) src0_grad_m->data, (float *) src0_grad_v->data, adamw_params_d, ne, stream);
        return;
    }

    GGML_ASSERT(src0_grad_m->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0_grad_v->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(ggml_nelements(src0_grad_m) >= ne);
    GGML_ASSERT(ggml_nelements(src0_grad_v) >= ne);
    opt_step_adamw_q8_0_cuda(src0_d, src0_grad_d,
        (block_q8_0 *) src0_grad_m->data, (block_q8_0 *) src0_grad_v->data, adamw_params_d, ne, stream);
}
