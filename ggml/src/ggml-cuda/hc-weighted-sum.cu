#include "hc-weighted-sum.cuh"

// Per-batch n_embd-major layout. Each (block.y, thread block on x) pair
// owns one batch and a slice of n_embd. The h4 specialization keeps the
// 4 weights in registers.
static __global__ void hc_weighted_sum_h4_f32(
        const char * __restrict__ x,
        const char * __restrict__ w,
        float      * __restrict__ dst,
        const int64_t n_embd,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbw0,
        const int64_t nbw1,
        const int64_t nbd0,
        const int64_t nbd1) {
    const int64_t b      = blockIdx.y;
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;

    const char * xb = x + b*nbx2;
    const char * wb = w + b*nbw1;
    char       * db = ((char *) dst) + b*nbd1;

    const float w0 = *(const float *) (wb + 0*nbw0);
    const float w1 = *(const float *) (wb + 1*nbw0);
    const float w2 = *(const float *) (wb + 2*nbw0);
    const float w3 = *(const float *) (wb + 3*nbw0);

    for (int64_t e = tid; e < n_embd; e += stride) {
        const char * xe = xb + e*nbx0;
        const float v = *(const float *) (xe + 0*nbx1) * w0
                      + *(const float *) (xe + 1*nbx1) * w1
                      + *(const float *) (xe + 2*nbx1) * w2
                      + *(const float *) (xe + 3*nbx1) * w3;
        *(float *) (db + e*nbd0) = v;
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
        const int64_t nbx2,
        const int64_t nbw0,
        const int64_t nbw1,
        const int64_t nbd0,
        const int64_t nbd1) {
    const int64_t b      = blockIdx.y;
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;

    const char * xb = x + b*nbx2;
    const char * wb = w + b*nbw1;
    char       * db = ((char *) dst) + b*nbd1;

    for (int64_t e = tid; e < n_embd; e += stride) {
        const char * xe = xb + e*nbx0;
        float sum = 0.0f;
        for (int64_t h = 0; h < hc_mult; ++h) {
            sum += *(const float *) (xe + h*nbx1) * *(const float *) (wb + h*nbw0);
        }
        *(float *) (db + e*nbd0) = sum;
    }
}

void ggml_cuda_op_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    // src0: [n_embd, hc_mult, n_batch]; src1: [hc_mult, n_batch];
    // dst:  [n_embd, n_batch]; src0->ne[3]/src1->ne[2..3] all == 1.
    GGML_ASSERT(src0->ne[1] == src1->ne[0]);
    GGML_ASSERT(src0->ne[2] == src1->ne[1]);
    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == src0->ne[0]);
    GGML_ASSERT(dst->ne[1] == src0->ne[2]);
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1);

    const int64_t n_embd  = src0->ne[0];
    const int64_t hc_mult = src0->ne[1];
    const int64_t n_batch = src0->ne[2];

    const int64_t num_blocks_x = (n_embd + CUDA_HC_WEIGHTED_SUM_BLOCK_SIZE - 1) / CUDA_HC_WEIGHTED_SUM_BLOCK_SIZE;
    const dim3 block_nums((unsigned int) num_blocks_x, (unsigned int) n_batch, 1);
    const dim3 block_dims(CUDA_HC_WEIGHTED_SUM_BLOCK_SIZE, 1, 1);

    const char * src0_d = (const char *) src0->data;
    const char * src1_d = (const char *) src1->data;
    float * dst_d = (float *) dst->data;

    if (hc_mult == 4) {
        hc_weighted_sum_h4_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
                src0_d, src1_d, dst_d, n_embd,
                src0->nb[0], src0->nb[1], src0->nb[2],
                src1->nb[0], src1->nb[1],
                dst->nb[0], dst->nb[1]);
    } else {
        hc_weighted_sum_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
                src0_d, src1_d, dst_d, n_embd, hc_mult,
                src0->nb[0], src0->nb[1], src0->nb[2],
                src1->nb[0], src1->nb[1],
                dst->nb[0], dst->nb[1]);
    }
}
