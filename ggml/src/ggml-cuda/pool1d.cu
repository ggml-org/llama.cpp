#include "pool1d.cuh"

static __global__ void pool1d_kernel(
        const int iw, const int ow,
        const int k0, const int s0, const int p0,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int cur_row = idx / ow;
    const int cur_ow  = idx - cur_row * ow;

    const float * i_ptr = src + (int64_t) cur_row * iw;
    float       * o_ptr = dst + (int64_t) cur_row * ow;

    const int start = cur_ow * s0 - p0;
    const int bw    = max(0,  start);
    const int ew    = min(iw, start + k0);

    float res = 0.0f;

    switch (op) {
        case GGML_OP_POOL_AVG: res = 0.0f;     break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: assert(false);
    }

    // Padded positions are outside [bw, ew) and are excluded from the average,
    // matching the CPU implementation which divides by the in-bounds count.
    int count = 0;

    for (int j = bw; j < ew; ++j) {
        const float cur = i_ptr[j];

        switch (op) {
            case GGML_OP_POOL_AVG: res += cur;             break;
            case GGML_OP_POOL_MAX: res  = max(res, cur);   break;
            default: assert(false);
        }

        ++count;
    }

    if (op == GGML_OP_POOL_AVG) {
        res = count > 0 ? res / count : 0.0f;
    }

    o_ptr[cur_ow] = res;
}

static void pool1d_f32_cuda(
        const int iw, const int ow,
        const int k0, const int s0, const int p0,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {

    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    pool1d_kernel<<<num_blocks, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(iw, ow, k0, s0, p0, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int32_t * opts = (const int32_t *) dst->op_params;
    const enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t OW = dst->ne[0];

    const int64_t parallel_elements = ggml_nrows(dst) * OW;

    pool1d_f32_cuda(IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, op, stream);
}
