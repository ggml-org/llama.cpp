#include "cumsum.cuh"

// Kernel to compute cumulative sum along an arbitrary dimension
// Each block processes one position in the non-cumsum dimensions
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
    const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3,
    const int dim) {

    // Shared memory to store warp sums (always use float for accumulation)
    extern __shared__ float shmem[];

    // Map block indices to actual tensor dimensions
    // blockIdx.x, blockIdx.y, blockIdx.z represent the 3 non-cumsum dimensions
    // threadIdx.x represents position in the cumsum dimension
    int64_t grid_indices[3] = {blockIdx.x, blockIdx.y, blockIdx.z};
    int64_t i_vals[4];

    int grid_idx = 0;
    for (int d = 0; d < 4; ++d) {
        if (d == dim) {
            i_vals[d] = 0; // Will be set in the loop below
        } else {
            i_vals[d] = grid_indices[grid_idx++];
        }
    }

    const int64_t i0 = i_vals[0];
    const int64_t i1 = i_vals[1];
    const int64_t i2 = i_vals[2];
    const int64_t i3 = i_vals[3];

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01 || i0 >= ne00) {
        return;
    }

    const int64_t ne_dim = (dim == 0) ? ne00 : (dim == 1) ? ne01 : (dim == 2) ? ne02 : ne03;
    const int64_t nb_dim_src = (dim == 0) ? nb00 : (dim == 1) ? nb01 : (dim == 2) ? nb02 : nb03;
    const int64_t nb_dim_dst = (dim == 0) ? nb0  : (dim == 1) ? nb1  : (dim == 2) ? nb2  : nb3;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;

    // Phase 1: Each thread processes elements at stride blockDim.x
    // Compute warp-level prefix sums
    for (int64_t i_dim = tid; i_dim < ne_dim; i_dim += blockDim.x) {
        const int64_t offset_src = i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03 + i_dim*nb_dim_src;
        const int64_t offset_dst = i0*nb0  + i1*nb1  + i2*nb2  + i3*nb3  + i_dim*nb_dim_dst;

        const T * src_ptr = (const T *) ((const char *) src + offset_src);
        T       * dst_ptr = (T       *) ((      char *) dst + offset_dst);

        // Load value and compute prefix sum within warp
        float val = static_cast<float>(src_ptr[0]);
        val = warp_prefix_inclusive_sum(val);
        dst_ptr[0] = static_cast<T>(val);

        // Last thread of warp stores its sum to shared memory at position based on data index
        if (lane_id == WARP_SIZE - 1 || i_dim == ne_dim - 1) {
            const int shmem_idx = i_dim / WARP_SIZE;
            shmem[shmem_idx] = val;
        }
    }

    // Sync once after all warp prefix sums are computed
    __syncthreads();

    // Phase 2: Add the sum of all preceding warp groups to each element
    for (int64_t i_dim = tid; i_dim < ne_dim; i_dim += blockDim.x) {
        const int64_t offset_dst = i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3 + i_dim*nb_dim_dst;
        T * dst_ptr = (T *) ((char *) dst + offset_dst);

        const int shmem_idx = i_dim / WARP_SIZE;
        float sum = 0.0f;
        for (int j = 0; j < shmem_idx; ++j) {
            sum += shmem[j];
        }
        dst_ptr[0] = static_cast<T>(static_cast<float>(dst_ptr[0]) + sum);
    }
}

template<typename T>
static void cumsum_cuda(
    const T * src, T * dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3,
    const int dim,
    cudaStream_t stream) {

    // Dimension being accumulated
    const int64_t ne_dims[4] = {ne00, ne01, ne02, ne03};
    const int64_t ne_dim = ne_dims[dim];

    // Grid dimensions: the GGML_MAX_DIMS-1 non-cumsum dimensions
    int64_t grid_dims_arr[GGML_MAX_DIMS - 1];
    int grid_idx = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d != dim) {
            grid_dims_arr[grid_idx++] = ne_dims[d];
        }
    }

    dim3 block_dims(CUDA_CUMSUM_BLOCK_SIZE, 1, 1);
    dim3 grid_dims(grid_dims_arr[0], grid_dims_arr[1], grid_dims_arr[2]);

    // Shared memory size: one float per warp
    const int num_warps = (ne_dim + WARP_SIZE - 1) / WARP_SIZE;
    const size_t shmem_size = num_warps * sizeof(float);

    cumsum_kernel<<<grid_dims, block_dims, shmem_size, stream>>>(
        src, dst,
        ne00, ne01, ne02, ne03,
        nb00, nb01, nb02, nb03,
        nb0, nb1, nb2, nb3,
        dim
    );
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    const int dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(src0->type == dst->type);
    switch(src0->type) {
        case GGML_TYPE_F32:
            {
                cumsum_cuda(
                    (const float *)src0->data, (float *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    dim,
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
                    dim,
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
                    dim,
                    stream
                );
            } break;
        default:
            GGML_ABORT("fatal error");
    }
}
