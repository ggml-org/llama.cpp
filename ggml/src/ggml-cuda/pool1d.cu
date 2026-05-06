#include "pool1d.cuh"

static __global__ void pool1d_kernel_f32(
        const int iw, const int ow,
        const int k, const int s, const int p,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int nr   = idx / ow;
    const int cur_ow = idx % ow;

    const float * i_ptr = src + nr * iw;
    float       * o_ptr = dst + nr * ow;

    const int start_w = cur_ow * s - p;
    const int bw = max(0, start_w);
    const int ew = min(iw, start_w + k);

    float res;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0.0f;     break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX;  break;
        default: return;
    }

    int count = 0;
    for (int j = bw; j < ew; ++j) {
        const float cur = i_ptr[j];
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur; break;
            case GGML_OP_POOL_MAX: res = max(res, cur); break;
            default: break;
        }
        ++count;
    }

    if (op == GGML_OP_POOL_AVG && count > 0) {
        res /= count;
    }

    o_ptr[cur_ow] = res;
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float       * dst_d  = (float *)dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const int parallel_elements = nr * OW;

    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    pool1d_kernel_f32<<<num_blocks, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(
        IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, op);
}
