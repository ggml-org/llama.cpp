#include <algorithm>

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

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    extern __shared__ float smem[];
    float* s_vals = smem;
    float* s_warp_sums = smem + blockDim.x;
    float* s_carry = smem + blockDim.x + warps_per_block;
    float* s_chunk_total = s_carry + 1;

    // Initialize carry
    if (tid == 0) {
        *s_carry = 0.0f;
    }
    __syncthreads();

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const T * src_row = src + i1 * nb01 + i2 * nb02 + i3 * nb03;
    T       * dst_row = dst + i1 * nb1  + i2 * nb2  + i3 * nb3;

    for (int64_t start = 0; start < ne00; start += blockDim.x) {
        int64_t idx = start + tid;
        float val = (idx < ne00) ? static_cast<float>(src_row[idx]) : 0.0f;

        // 1. Warp inclusive scan
        val = warp_prefix_inclusive_sum(val);
        s_vals[tid] = val;

        // Store warp total
        if (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        __syncthreads();

        // 2. Exclusive scan of warp sums (warp 0 only)
        if (warp == 0) {
            float w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            float inc = warp_prefix_inclusive_sum(w);
            if (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;   // exclusive sum
            }
            if (tid == warps_per_block - 1) {
                *s_chunk_total = inc;          // total sum of this chunk
            }
        }
        __syncthreads();

        float carry = *s_carry;
        float final_val = s_vals[tid] + s_warp_sums[warp] + carry;
        if (idx < ne00) {
            dst_row[idx] = static_cast<T>(final_val);
        }
        __syncthreads();

        // Update carry for next chunk
        if (tid == 0) {
            *s_carry += *s_chunk_total;
        }
        __syncthreads();
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
    const int num_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    int block_size = num_warps * WARP_SIZE;
    block_size = std::min(block_size, CUDA_CUMSUM_BLOCK_SIZE);
    dim3 block_dims(block_size, 1, 1);
    const int warps_per_block = block_size / WARP_SIZE;
    const size_t shmem_size = (block_size + warps_per_block + 2) * sizeof(float);
    const size_t type_size = sizeof(T);

    cumsum_kernel<<<grid_dims, block_dims, shmem_size, stream>>>(
        src, dst,
        ne00, ne01, ne02, ne03,
        nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
        nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
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
