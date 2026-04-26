#include "hc-weighted-sum.cuh"

static __global__ void hc_weighted_sum_h4_f32(
        const char * __restrict__ x,
        const char * __restrict__ w,
        float      * __restrict__ dst,
        const int64_t n_embd,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbw0) {
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;

    const float w0 = *(const float *) (w + 0*nbw0);
    const float w1 = *(const float *) (w + 1*nbw0);
    const float w2 = *(const float *) (w + 2*nbw0);
    const float w3 = *(const float *) (w + 3*nbw0);

    for (int64_t e = tid; e < n_embd; e += stride) {
        const char * xe = x + e*nbx0;
        dst[e] = *(const float *) (xe + 0*nbx1) * w0
               + *(const float *) (xe + 1*nbx1) * w1
               + *(const float *) (xe + 2*nbx1) * w2
               + *(const float *) (xe + 3*nbx1) * w3;
    }
}

static __global__ void hc_weighted_sum_f32(
        const char * __restrict__ x,
        const char * __restrict__ w,
        float      * __restrict__ dst,
        const int64_t n_embd,
        const int64_t hc_mult,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbw0) {
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;

    for (int64_t e = tid; e < n_embd; e += stride) {
        const char * xe = x + e*nbx0;
        float sum = 0.0f;
        for (int64_t h = 0; h < hc_mult; ++h) {
            sum += *(const float *) (xe + h*nbx1) * *(const float *) (w + h*nbw0);
        }
        dst[e] = sum;
    }
}

void ggml_cuda_op_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[1] == src1->ne[0]);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == src0->ne[0]);

    const int64_t n_embd  = src0->ne[0];
    const int64_t hc_mult = src0->ne[1];

    const int64_t num_blocks = (n_embd + CUDA_HC_WEIGHTED_SUM_BLOCK_SIZE - 1) / CUDA_HC_WEIGHTED_SUM_BLOCK_SIZE;
    const dim3 block_nums(num_blocks, 1, 1);
    const dim3 block_dims(CUDA_HC_WEIGHTED_SUM_BLOCK_SIZE, 1, 1);

    const char * src0_d = (const char *) src0->data;
    const char * src1_d = (const char *) src1->data;
    float * dst_d = (float *) dst->data;

    if (hc_mult == 4) {
        hc_weighted_sum_h4_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
                src0_d, src1_d, dst_d, n_embd, src0->nb[0], src0->nb[1], src1->nb[0]);
    } else {
        hc_weighted_sum_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
                src0_d, src1_d, dst_d, n_embd, hc_mult, src0->nb[0], src0->nb[1], src1->nb[0]);
    }
}
