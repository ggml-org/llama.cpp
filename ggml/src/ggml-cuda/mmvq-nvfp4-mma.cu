#include "mmvq-nvfp4-mma.cuh"

#include "mmq.cuh"
#include "vecdotq.cuh"

#include <cstdint>

constexpr int NVFP4_MMVQ_NWARPS = 4;
constexpr int NVFP4_MMVQ_TILE_BYTES = (int) sizeof(block_nvfp4_mmq);

template <int preload>
static __device__ __forceinline__ void nvfp4_mmvq_prefetch_tile(
        uint8_t * dst_smem, const block_nvfp4_mmq * src_gmem) {
#ifdef CP_ASYNC_AVAILABLE
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const char * src = reinterpret_cast<const char *>(src_gmem);
    for (int off = lane * 16; off < NVFP4_MMVQ_TILE_BYTES; off += WARP_SIZE * 16) {
        cp_async_fast_16<preload>(ggml_cuda_cvta_generic_to_shared(dst_smem + off), src + off);
    }
#else
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int * src = reinterpret_cast<const int *>(src_gmem);
    int * dst = reinterpret_cast<int *>(dst_smem);
    constexpr int n_i32 = NVFP4_MMVQ_TILE_BYTES / (int) sizeof(int);
    for (int i = lane; i < n_i32; i += WARP_SIZE) {
        dst[i] = src[i];
    }
#endif
}

template <int nwarps>
__launch_bounds__(nwarps * WARP_SIZE, 1)
static __global__ void mul_mat_vec_nvfp4_mmvq_mma_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const float alpha,
        const int ncols_x,
        const int nrows_x,
        const int stride_row_x,
        const int stride_channel_x,
        const int stride_channel_y,
        const int stride_channel_dst,
        const int stride_sample_x,
        const int stride_sample_y,
        const int stride_sample_dst) {
    const int row = blockIdx.x;
    if (row >= nrows_x) {
        return;
    }

    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;

    const int lane    = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;

    const int kbx_row_offset =
        sample * stride_sample_x +
        channel * stride_channel_x +
        row * stride_row_x;

    const block_nvfp4_mmq * __restrict__ y_base =
        reinterpret_cast<const block_nvfp4_mmq *>(vy) +
        sample * stride_sample_y +
        channel * stride_channel_y;

    const int nblocks = ncols_x / QK_K;

    extern __shared__ uint8_t smem[];
    uint8_t * warp_smem = smem + warp_id * (2 * NVFP4_MMVQ_TILE_BYTES);

    float warp_acc = 0.0f;

    int kb = warp_id;
    int buf = 0;

    if (kb < nblocks) {
        nvfp4_mmvq_prefetch_tile<128>(warp_smem + buf * NVFP4_MMVQ_TILE_BYTES, y_base + kb);
        cp_async_commit_group();
    }

    while (kb < nblocks) {
        const int kb_next = kb + nwarps;
        if (kb_next < nblocks) {
            nvfp4_mmvq_prefetch_tile<128>(warp_smem + (buf ^ 1) * NVFP4_MMVQ_TILE_BYTES, y_base + kb_next);
            cp_async_commit_group();
        }

#ifdef CP_ASYNC_AVAILABLE
        if (kb_next < nblocks) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
#endif
        __syncwarp();

        const block_nvfp4_mmq * __restrict__ y_blk =
            reinterpret_cast<const block_nvfp4_mmq *>(warp_smem + buf * NVFP4_MMVQ_TILE_BYTES);

        float partial = 0.0f;
        if (lane < (QI_NVFP4 / VDR_NVFP4_NVFP4_MMVQ)) {
            const int iqs = lane * VDR_NVFP4_NVFP4_MMVQ;
            partial = vec_dot_nvfp4_nvfp4_mmvq_mma(vx, (const void *) y_blk, kbx_row_offset + kb, iqs);
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, mask);
        }

        if (lane == 0) {
            warp_acc += partial;
        }

        kb = kb_next;
        buf ^= 1;
    }

    __shared__ float warp_sums[nwarps];
    if (lane == 0) {
        warp_sums[warp_id] = warp_acc;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            sum += warp_sums[w];
        }
        if (alpha != 1.0f) {
            sum *= alpha;
        }
        dst[sample * stride_sample_dst + channel * stride_channel_dst + row] = sum;
    }
}

bool ggml_cuda_mul_mat_vec_nvfp4_mma(
        const void * vx, const void * vy, float * dst, float alpha,
        int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst,
        int stride_channel_x, int stride_channel_y, int stride_channel_dst,
        int nsamples_x, int nsamples_dst, int stride_sample_x, int stride_sample_y, int stride_sample_dst,
        cudaStream_t stream) {
    constexpr bool k_nvfp4_mmvq_native_enabled = true;
    if (!k_nvfp4_mmvq_native_enabled) {
        return false;
    }

    GGML_UNUSED(stride_col_y);
    GGML_UNUSED(stride_col_dst);

    if (ncols_dst != 1) {
        return false;
    }
    if (ncols_x % QK_K != 0) {
        return false;
    }
    if (nchannels_x != nchannels_y || nchannels_y != nchannels_dst) {
        return false;
    }
    if (nsamples_x != nsamples_dst) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!blackwell_mma_available(cc)) {
        return false;
    }

    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    const dim3 block(NVFP4_MMVQ_NWARPS * WARP_SIZE, 1, 1);
    const size_t smem = (size_t) NVFP4_MMVQ_NWARPS * 2 * NVFP4_MMVQ_TILE_BYTES;

    mul_mat_vec_nvfp4_mmvq_mma_kernel<NVFP4_MMVQ_NWARPS><<<grid, block, smem, stream>>>(
        vx, vy, dst, alpha, ncols_x, nrows_x,
        stride_row_x, stride_channel_x, stride_channel_y, stride_channel_dst,
        stride_sample_x, stride_sample_y, stride_sample_dst);
    return true;
}
