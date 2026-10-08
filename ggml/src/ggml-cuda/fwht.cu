#include "common.cuh"
#include "convert.cuh"
#include "fwht.cuh"

// wide FWHT blocks use one row per thread block with this many threads
#define GGML_CUDA_FWHT_BLOCK_NT 256

template <int N, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Wide blocks: one row per thread block instead of per warp, so each thread keeps N/NT
// values rather than N/32. Stages below the warp width still shuffle, those up to the
// block width go through shared memory, and the rest stay in registers.
template <int N, int NT, typename T>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * NT + tid]) * scale;
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const T *     src_d = (const T *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            break;
    }

    // wide blocks: one row per thread block
    {
        constexpr int nt = GGML_CUDA_FWHT_BLOCK_NT;

        dim3 grid_dims_w(rows, 1, 1);
        dim3 block_dims_w(nt, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params_w =
            ggml_cuda_kernel_launch_params(grid_dims_w, block_dims_w, 0, stream);

        switch (n) {
            case 1024:
                ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 2048:
                ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 4096:
                ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 8192:
                ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            default:
                return false;
        }
    }
}

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            return ggml_cuda_op_fwht_impl<float>(ctx, src, dst);
        case GGML_TYPE_F16:
            return ggml_cuda_op_fwht_impl<half>(ctx, src, dst);
        default:
            return false;
    }
}
