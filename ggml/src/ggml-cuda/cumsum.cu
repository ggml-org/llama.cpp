#include "cumsum.cuh"

// Kernel to compute cumulative sum along the innermost dimension (ne[0])
// Each block processes one row (ne[0] elements)
// Algorithm matches Metal implementation:
// 1. Each warp computes prefix sum within itself
// 2. Last thread of each warp stores result in shared memory
// 3. All warps sync
// 4. Each element adds the sum of all preceding warps

template<typename T>
static __global__ void cumsum_kernel(
    const T * src, T * dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3) {

    // Shared memory to store warp sums (always use float for accumulation)
    extern __shared__ float shmem[];

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const T * src_row = (const T *) ((const char *) src + i1*nb01 + i2*nb02 + i3*nb03);
    T       * dst_row = (T       *) ((      char *) dst + i1*nb1  + i2*nb2  + i3*nb3);

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;

    if (tid >= ne00) {
        return;
    }

    // Phase 1: Each thread processes elements at stride blockDim.x
    // Compute warp-level prefix sums
    for (int64_t i0 = tid; i0 < ne00; i0 += blockDim.x) {
        // Load value and compute prefix sum within warp
        float val = static_cast<float>(src_row[i0]);
        val = warp_prefix_inclusive_sum(val);
        dst_row[i0] = static_cast<T>(val);

        // Last thread of warp stores its sum to shared memory at position based on data index
        if (lane_id == WARP_SIZE - 1 || i0 == ne00 - 1) {
            const int shmem_idx = i0 / WARP_SIZE;
            shmem[shmem_idx] = val;
        }
    }

    // Sync once after all warp prefix sums are computed
    __syncthreads();

    // Phase 2: Add the sum of all preceding warp groups to each element
    for (int64_t i0 = tid; i0 < ne00; i0 += blockDim.x) {
        const int shmem_idx = i0 / WARP_SIZE;
        float sum = 0.0f;
        for (int j = 0; j < shmem_idx; ++j) {
            sum += shmem[j];
        }
        dst_row[i0] = static_cast<T>(static_cast<float>(dst_row[i0]) + sum);
    }
}

template<typename T>
static void cumsum_cuda(
    const T * src, T * dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3,
    cudaStream_t stream) {

    dim3 grid_dims(ne01, ne02, ne03);

    // Shared memory size: one float per warp
    const int num_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    const size_t shmem_size = num_warps * sizeof(float);

    int block_size = num_warps * WARP_SIZE;
    if (block_size > CUDA_CUMSUM_BLOCK_SIZE) {
        block_size = CUDA_CUMSUM_BLOCK_SIZE;
    }
    dim3 block_dims(block_size, 1, 1);

    cumsum_kernel<<<grid_dims, block_dims, shmem_size, stream>>>(
        src, dst,
        ne00, ne01, ne02, ne03,
        nb00, nb01, nb02, nb03,
        nb0, nb1, nb2, nb3
    );
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);
    switch(src0->type) {
        case GGML_TYPE_F32:
            {
                cumsum_cuda(
                    (const float *)src0->data, (float *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    stream
                );
            } break;
        case GGML_TYPE_F16:
            {
                cumsum_cuda(
                    (const half *)src0->data, (half *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    stream
                );
            } break;
        case GGML_TYPE_BF16:
            {
                cumsum_cuda(
                    (const nv_bfloat16 *)src0->data, (nv_bfloat16 *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    stream
                );
            } break;
        default:
            GGML_ABORT("fatal error");
    }
}
