#include "pool1d.cuh"

static __global__ void pool1d_nchw_kernel(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int nc = idx / ow;
    const int cur_ow = idx % ow;

    const float * i_ptr = src + nc * iw;
    float * o_ptr = dst + nc * ow;

    const int start = cur_ow * sw - pw;
    const int b = max(0, start);
    const int e = min(iw, start + kw);

    float res;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0.0f;     break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: return;
    }

    int count = 0;
    for (int i = b; i < e; i++) {
#if __CUDA_ARCH__ >= 350
        float cur = __ldg(i_ptr + i);
#else
        float cur = i_ptr[i];
#endif
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur;                break;
            case GGML_OP_POOL_MAX: res = max(res, cur);       break;
            default: break;
        }
        count++;
    }

    if (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : 0.0f;
    }

    o_ptr[cur_ow] = res;
}

// a window covering the whole row gives one output per row: reduce it with a block instead of one thread
template <bool is_max>
static __global__ void pool1d_row_kernel(const int iw, const float * src, float * dst) {
    const float * row = src + (int64_t) blockIdx.x*iw;

    float res = is_max ? -FLT_MAX : 0.0f;
    for (int i = threadIdx.x; i < iw; i += blockDim.x) {
        res = is_max ? max(res, row[i]) : res + row[i];
    }

    __shared__ float buf[WARP_SIZE];
    res = block_reduce<is_max ? block_reduce_method::MAX : block_reduce_method::SUM>(res, buf);

    if (threadIdx.x == 0) {
        dst[blockIdx.x] = is_max ? res : res/iw;
    }
}

static void pool1d_row_f32_cuda(const int iw, const int64_t nr, const float * src, float * dst, const enum ggml_op_pool op, cudaStream_t stream) {
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const dim3 block_dims(nr/nsm < 2 ? 512 : (iw < 1024 ? 32 : 128));
    if (op == GGML_OP_POOL_MAX) {
        pool1d_row_kernel<true><<<nr, block_dims, 0, stream>>>(iw, src, dst);
    } else {
        pool1d_row_kernel<false><<<nr, block_dims, 0, stream>>>(iw, src, dst);
    }
}

static void pool1d_nchw_kernel_f32_f32_cuda(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {
    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool1d_nchw_kernel<<<block_nums, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(iw, ow, kw, sw, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);

    // short rows: one thread per row is faster
    if (OW == 1 && p0 == 0 && k0 == IW && IW >= CUDA_POOL1D_ROW_MIN_IW && (op == GGML_OP_POOL_MAX || op == GGML_OP_POOL_AVG)) {
        pool1d_row_f32_cuda(IW, nr, src0_d, dst_d, op, stream);
        return;
    }

    const int parallel_elements = (int)(nr * OW);

    pool1d_nchw_kernel_f32_f32_cuda(IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, op, stream);
}
