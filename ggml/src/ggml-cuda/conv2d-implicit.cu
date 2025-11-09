// #include <cuda_runtime.h>
#include "ggml.h"
#include "common.cuh"
#include "convert.cuh"
#include "conv2d-implicit.cuh"


typedef unsigned int uint;
constexpr uint WARPSIZE = 32;
#define CUDA_NCHW_2_NHWC_TILE_DIM 32
#define CUDA_NCHW_2_NHWC_BLOCK_NM 8
#define CUDA_NCHW_2_NHWC_BLOCK_ROWS 8
#define CUDA_NCHW_2_NHWC_BLOCK_C 64


//currently not use; in future for split-k kernels
template <typename src_T, typename dst_T>
static __global__ void reduce_f32(const src_T * __restrict__ x, dst_T * __restrict__ dst, const int ncols, const int nrows) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    if (row * blockDim.x + col < ncols) {
        for (int i = 0; i < nrows; ++i){
            sum += ggml_cuda_cast<float>(x[i * ncols + row * blockDim.x + col]);
        }
        dst[row * blockDim.x + col] = ggml_cuda_cast<dst_T>(sum);
    }
}

constexpr uint32_t filter_swizzle_mask(uint32_t n, uint32_t m) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    int count = 0;
    while ((m >>= 1) != 0)
        ++count;
    return n << count;
}

template <typename src_T, typename dst_T>
static __global__ void NCHW2NHWC(const src_T *src, dst_T * dst, const int ne, const int ne00, const int ne01){

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n = ne00 * ne01;

    int x  = blockIdx.x * CUDA_NCHW_2_NHWC_TILE_DIM + threadIdx.x;
    int y  = blockIdx.y * CUDA_NCHW_2_NHWC_TILE_DIM + threadIdx.y;
    int tx = blockIdx.y * CUDA_NCHW_2_NHWC_TILE_DIM + threadIdx.x;  // transpose block offset
    int ty = blockIdx.x * CUDA_NCHW_2_NHWC_TILE_DIM + threadIdx.y;

    __shared__ src_T tile[CUDA_NCHW_2_NHWC_TILE_DIM][CUDA_NCHW_2_NHWC_TILE_DIM];

    for(int i = 0; i < CUDA_NCHW_2_NHWC_BLOCK_NM; ++i){

        const unsigned int imat = blockIdx.z * CUDA_NCHW_2_NHWC_BLOCK_NM + i;
        if(imat >= nmat)
            break;
        for (int j = 0; j < CUDA_NCHW_2_NHWC_TILE_DIM; j += CUDA_NCHW_2_NHWC_BLOCK_ROWS){
            if(x < ne01 && y + j < ne00){
                const int row = threadIdx.y+j;
                const int col = threadIdx.x ^ row;
                tile[row][col] = src[imat*n + (y+j)*ne01 + x];
            }
        }
        __syncthreads();

        for (int j = 0; j < CUDA_NCHW_2_NHWC_TILE_DIM; j += CUDA_NCHW_2_NHWC_BLOCK_ROWS){
            if(ty + j < ne01 && tx < ne00){
                const int col = (threadIdx.y+j) ^ threadIdx.x;
                dst[imat*n + (ty+j)*ne00 + tx] = ggml_cuda_cast<dst_T>(tile[threadIdx.x][col]);
            }
        }
    }
}

template <typename src_T, typename dst_T, const unsigned int mask, const int rs, const unsigned int blk_c>
static __global__ void NCHW2NHWC(const src_T *src, dst_T * dst, const int ne, const int ne00, const int ne01){

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n = ne00 * ne01;

    const unsigned int tx   = threadIdx.x;
    const unsigned int bx   = blockIdx.x;
    const unsigned int by   = blockIdx.y;

    __shared__ src_T tile[rs*blk_c];

#pragma unroll
    for(int i = 0; i < CUDA_NCHW_2_NHWC_BLOCK_NM; ++i){

        const unsigned int imat = by * CUDA_NCHW_2_NHWC_BLOCK_NM + i;
        if(imat >= nmat)
            break;
#pragma unroll
        for (unsigned int j = 0; j < rs; j++){
            const unsigned int row = (j * blk_c + tx) % rs;
            const unsigned int col = (j * blk_c + tx) / rs;
            const unsigned int src_index = imat*n + bx * blk_c * rs + j * blk_c + tx;
            unsigned int idx = row * blk_c + col;
            idx =  idx ^ ((idx & mask) >> 4);
            if (src_index < ne) {
              tile[idx] = src[src_index];
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned int j = 0; j < rs; j++){
            const unsigned int dst_index = imat*n + j*ne00 + bx*blk_c + tx;
            if(dst_index < ne){
              unsigned int idx = j*blk_c + tx;
              idx =  idx ^ ((idx & mask) >> 4);
               dst[dst_index] = ggml_cuda_cast<dst_T>(tile[idx]);
            }
        }
    }
}



template<typename T, const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          // layout: 0, NHWC; 1, NCHW
          const int layout, const bool vec_load, const int ksplit, const int PAD=4>
static __global__ void conv2d_implicit_kernel(const float * __restrict__ input,
                                              const T * __restrict__ kernel,
                                              float * __restrict__ output,
                                              const param_t param) {

    __shared__ char smem[sizeof(float) * (TM*TN*NUM_THREADS) <= sizeof(float) * 2 * (BM+PAD) * BK +  sizeof(T)*2*BK * (BN+PAD) ?
                         sizeof(float)*2*(BM+PAD)*BK + sizeof(T)*2*BK*(BN+PAD) : sizeof(float) * (TM*TN*NUM_THREADS)];
    T *smemweight = reinterpret_cast<T *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 2 * BK * (BN+PAD) * sizeof(T));

    const uint tx = threadIdx.x;
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;

    const uint PQ = param.Oh * param.Ow;
    const uint CHW = param.c * param.h * param.w;

    // Warp tile
    const uint lane_id = tx % WARPSIZE;
    const uint warp_id = tx / WARPSIZE;
    const int mma_tid_x = warp_id / (BN / WN);
    const int mma_tid_y = warp_id % (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/2=32
    constexpr uint WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadColInWarp = lane_id % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = lane_id / (WSUBN / TN); // i/4

    int z = blockIdx.z;

    int inChannelOffset = layout == 0 ? param.c * param.w : param.h * param.w;
    int weightKOffset = param.c * param.r * param.s;

    const uint ks =  (ksplit > 0) ? (weightKOffset + ksplit - 1) / ksplit : weightKOffset;
    const uint start_k = (ksplit > 0)? z * ks: 0;
    const uint end_k = min(start_k + ks, weightKOffset);

    int write_flag = 1;
    T weight_frag[2][WNITER * TN];
    float input_frag[2][WMITER * TM] = {0.f};
    float output_frag[WMITER * TM * WNITER * TN] = {0.f};

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = tx / (BK / 4);
    const uint innerColA = tx % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

// ldg
    loadFilter<T, BN, rowStrideA, layout, vec_load, ksplit, PAD>
       (kernel, smemweight, by, innerRowA, innerColA, weightKOffset,
        start_k, end_k, param);

    loadInput<BM, rowStrideA, layout, vec_load, ksplit, PAD>
       (input,  smeminput, bx,  innerRowA, innerColA,
        start_k, end_k, PQ, CHW, inChannelOffset, param);

    __syncthreads();

    // lds
    const uint input_lds_addr =  mma_tid_x * WM;
#pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
      for (uint i = 0; i < TM; ++i)
        input_frag[0][wSubRowIdx * TM + i] = smeminput[input_lds_addr + wSubRowIdx * WSUBM +
                               threadRowInWarp * TM + i];

    const uint weight_lds_addr = mma_tid_y * WN;
#pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
      for (uint i = 0; i < TN; ++i)
        weight_frag[0][wSubColIdx * TN + i] = smemweight[weight_lds_addr + wSubColIdx * WSUBN +
                             threadColInWarp * TN + i];

    for (int crs = start_k; crs < end_k; crs += BK) {

        int load_flag = write_flag ^ 1;
#pragma unroll
        for (int subcrs = 0; subcrs < BK - 1; ++subcrs)
        {

#pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
                for (uint i = 0; i < TN; ++i)
                    weight_frag[(subcrs + 1) % 2][wSubColIdx * TN + i] = smemweight[load_flag * (BN+PAD) * BK +
                        (subcrs + 1) * (BN+PAD) + weight_lds_addr + wSubColIdx * WSUBN + threadColInWarp * TN + i];
#pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
                for (uint i = 0; i < TM; ++i)
                    input_frag[(subcrs + 1) % 2][wSubRowIdx * TM + i] = smeminput[load_flag * (BM+PAD) * BK +
                        (subcrs + 1) * (BM+PAD) + input_lds_addr + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];

            // execute warptile matmul
#pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
#pragma unroll
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                        (wSubColIdx * TN) + resIdxN] +=
                                input_frag[subcrs % 2][wSubRowIdx * TM + resIdxM] *
                                ggml_cuda_cast<float>(weight_frag[subcrs % 2][wSubColIdx * TN + resIdxN]);
                        }
                    }
                }
            }
        }
        // ldg

        loadFilter<T, BN, rowStrideA, layout, vec_load, ksplit, PAD>
            (kernel, &smemweight[write_flag * (BN+PAD) * BK], by, innerRowA, innerColA, weightKOffset,
                crs+BK, end_k, param);

        loadInput<BM, rowStrideA, layout, vec_load, ksplit, PAD>
            (input,  &smeminput[write_flag * (BM+PAD) * BK], bx,  innerRowA, innerColA,
                crs + BK, end_k, PQ, CHW, inChannelOffset, param);

        __syncthreads();

        write_flag ^= 1;

#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
            for (uint i = 0; i < TM; ++i)
                input_frag[0][wSubRowIdx * TM + i] = smeminput[(load_flag ^ 1) * (BM+PAD) * BK +
                    input_lds_addr + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
            for (uint i = 0; i < TN; ++i)
                weight_frag[0][wSubColIdx * TN + i] = smemweight[(load_flag ^ 1) * (BN+PAD) * BK +
                    weight_lds_addr + wSubColIdx * WSUBN + threadColInWarp * TN + i];
#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // calculate per-thread results
#pragma unroll
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                    (wSubColIdx * TN) + resIdxN] +=
                            input_frag[1][wSubRowIdx * TM + resIdxM] *
                            ggml_cuda_cast<float>(weight_frag[1][wSubColIdx * TN + resIdxN]);
                    }
                }
            }
        }
    }

    // reuse smem
    float *smemoutput = reinterpret_cast<float *>(smem);

    const uint output_lds_addr = warp_id * WSUBM * WSUBN + lane_id;
    const uint output_sts_addr = mma_tid_x * BN / WN * TM * TN * WARPSIZE + mma_tid_y * TM * TN * WARPSIZE +
                         threadColInWarp * TN * WSUBM + threadRowInWarp * TM;
    const uint m_idx = by * BN + mma_tid_y * WN;
    const uint n_idx = bx * BM + mma_tid_x * WM;

#pragma unroll
    for (int i = 0; i < WMITER; ++i)
    {
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
        {
            __syncthreads();

#pragma unroll
            for (int subi = 0; subi < TM; ++subi)
            {
#pragma unroll
                for (int subj = 0; subj < TN; ++subj)
                {
                    // output sts
                    smemoutput[output_sts_addr + subj * WSUBM + subi] =
                        output_frag[(i * TM + subi) * (WNITER * TN) + j * TN + subj];
                }
            }
            __syncthreads();
#pragma unroll
            for (int subk = 0; subk < TM * TN; ++subk){
                const uint row =  m_idx + j * WSUBN + (lane_id + subk * WARPSIZE) / WSUBM;
                const uint gemm_i =  n_idx + i * WSUBM + (lane_id + subk * WARPSIZE) % WSUBM;
                const int n = (ksplit > 0) ? gemm_i / PQ : z;
                const int col = (ksplit > 0) ? gemm_i % PQ : gemm_i;
                if (n < param.n && row < param.k && col < param.Oh * param.Ow){
                    const uint outOffset = ksplit > 0 ?
                                z * param.n * param.k * param.Oh * param.Ow + n * param.k * param.Oh * param.Ow +
                                row * param.Oh * param.Ow + col :
                                z * param.k * param.Oh * param.Ow + row * param.Oh * param.Ow + col;
                    output[outOffset] = smemoutput[output_lds_addr + subk * WARPSIZE];
                }
            }
        }
    }
}



template <unsigned int mma_tiles_per_warp_m, unsigned int mma_tiles_per_warp_k, unsigned int smem_stride>
__device__ __forceinline__ void ldmatrix_a(
  const half* src,
  half (&reg)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4]
){
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
  static_assert(mma_tiles_per_warp_m == 8, "mma_tiles_per_warp_m must be 4");
  static_assert(mma_tiles_per_warp_k == 4, "mma_tiles_per_warp_k must be 4");

  uint32_t (&reg_) [mma_tiles_per_warp_m][mma_tiles_per_warp_k][2] = reinterpret_cast<uint32_t(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2]>(reg);
  unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
  unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4);
  swizzled_offset = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2);
  uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
  constexpr unsigned int smem_stride_ = smem_stride * sizeof(half); // convert stride to bytes

    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][0][0]), "=r"(reg_[0][0][1]), "=r"(reg_[1][0][0]), "=r"(reg_[1][0][1])
      : "r"(src_addr)
    );

    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][0][0]), "=r"(reg_[2][0][1]), "=r"(reg_[3][0][0]), "=r"(reg_[3][0][1])
      : "r"(src_addr + 32 * smem_stride_)
    );

    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[4][0][0]), "=r"(reg_[4][0][1]), "=r"(reg_[5][0][0]), "=r"(reg_[5][0][1])
      : "r"(src_addr + 64 * smem_stride_)
    );

    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[6][0][0]), "=r"(reg_[6][0][1]), "=r"(reg_[7][0][0]), "=r"(reg_[7][0][1])
      : "r"(src_addr + 96 * smem_stride_)
    );

    src_addr ^= 0b10000;

    // 1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][1][0]), "=r"(reg_[0][1][1]), "=r"(reg_[1][1][0]), "=r"(reg_[1][1][1])
      : "r"(src_addr)
    );

    // 1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][1][0]), "=r"(reg_[2][1][1]), "=r"(reg_[3][1][0]), "=r"(reg_[3][1][1])
      : "r"(src_addr + 32 * smem_stride_)
    );

    // 1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[4][1][0]), "=r"(reg_[4][1][1]), "=r"(reg_[5][1][0]), "=r"(reg_[5][1][1])
      : "r"(src_addr + 64 * smem_stride_)
    );

    // 1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[6][1][0]), "=r"(reg_[6][1][1]), "=r"(reg_[7][1][0]), "=r"(reg_[7][1][1])
      : "r"(src_addr + 96 * smem_stride_)
    );

    src_addr ^= 0b110000;

    // 2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][2][0]), "=r"(reg_[0][2][1]), "=r"(reg_[1][2][0]), "=r"(reg_[1][2][1])
      : "r"(src_addr)
    );

    // 2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][2][0]), "=r"(reg_[2][2][1]), "=r"(reg_[3][2][0]), "=r"(reg_[3][2][1])
      : "r"(src_addr + 32 * smem_stride_)
    );

    // 2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[4][2][0]), "=r"(reg_[4][2][1]), "=r"(reg_[5][2][0]), "=r"(reg_[5][2][1])
      : "r"(src_addr + 64 * smem_stride_)
    );

    // 2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[6][2][0]), "=r"(reg_[6][2][1]), "=r"(reg_[7][2][0]), "=r"(reg_[7][2][1])
      : "r"(src_addr + 96 * smem_stride_)
    );
    src_addr ^= 0b10000;

    // 3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][3][0]), "=r"(reg_[0][3][1]), "=r"(reg_[1][3][0]), "=r"(reg_[1][3][1])
      : "r"(src_addr)
    );

    // 3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][3][0]), "=r"(reg_[2][3][1]), "=r"(reg_[3][3][0]), "=r"(reg_[3][3][1])
      : "r"(src_addr + 32 * smem_stride_)
    );

    // 3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[4][3][0]), "=r"(reg_[4][3][1]), "=r"(reg_[5][3][0]), "=r"(reg_[5][3][1])
      : "r"(src_addr + 64 * smem_stride_)
    );

    // 3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[6][3][0]), "=r"(reg_[6][3][1]), "=r"(reg_[7][3][0]), "=r"(reg_[7][3][1])
      : "r"(src_addr + 96 * smem_stride_)
    );
#else
    GGML_UNUSED(src);
    GGML_UNUSED(reg);
    NO_DEVICE_CODE;
#endif
}

template <unsigned int mma_tiles_per_warp_k, unsigned int mma_tiles_per_warp_n, unsigned int smem_stride>
__device__ __forceinline__ void ldmatrix_b(
  const half* src,
  half (&reg)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2]
){
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

  static_assert(mma_tiles_per_warp_k == 4, "mma_tiles_per_warp_k must be 4");
  static_assert(mma_tiles_per_warp_n == 8, "mma_tiles_per_warp_n must be 8");

  uint32_t (&reg_) [4][8] = reinterpret_cast<uint32_t(&)[4][8]>(reg);
  unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
  unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4);
  swizzled_offset = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2);
  uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
  constexpr unsigned int smem_stride_ = smem_stride * sizeof(half); // convert stride to bytes

    // 0
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][0]), "=r"(reg_[0][1]), "=r"(reg_[0][2]), "=r"(reg_[0][3])
      : "r"(src_addr)
    );


  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[0][4]), "=r"(reg_[0][5]), "=r"(reg_[0][6]), "=r"(reg_[0][7])
    : "r"(src_addr + 32 * smem_stride_)
  );

  src_addr ^= 0b10000;

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[1][0]), "=r"(reg_[1][1]), "=r"(reg_[1][2]), "=r"(reg_[1][3])
    : "r"(src_addr)
  );

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[1][4]), "=r"(reg_[1][5]), "=r"(reg_[1][6]), "=r"(reg_[1][7])
    : "r"(src_addr + 32 * smem_stride_)
  );

  src_addr ^= 0b110000;

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[2][0]), "=r"(reg_[2][1]), "=r"(reg_[2][2]), "=r"(reg_[2][3])
    : "r"(src_addr)
  );

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[2][4]), "=r"(reg_[2][5]), "=r"(reg_[2][6]), "=r"(reg_[2][7])
    : "r"(src_addr + 32 * smem_stride_)
  );

  src_addr ^= 0b10000;

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[3][0]), "=r"(reg_[3][1]), "=r"(reg_[3][2]), "=r"(reg_[3][3])
    : "r"(src_addr)
  );

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[3][4]), "=r"(reg_[3][5]), "=r"(reg_[3][6]), "=r"(reg_[3][7])
    : "r"(src_addr + 32 * smem_stride_)
  );
#else
    GGML_UNUSED(src);
    GGML_UNUSED(reg);
    NO_DEVICE_CODE;
#endif
}

template<const int BM, const int BN, const int BK, const int WM, const int WN,
        const int WK,  const int ksplit, const int NUM_THREADS>
static __global__ void conv2d_implicit_kernel(const half * __restrict__ input,
                                              const half * __restrict__ kernel,
                                              half * __restrict__ output,
                                              const param_t param) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

  constexpr unsigned int MMA_M = 16;
  constexpr unsigned int MMA_N = 8;

  const unsigned int K = param.c * param.r * param.s;
  const uint inChannelOffset = param.c * param.w;
  const uint weightKOffset = K;

  const unsigned int PQ   = param.Ow * param.Oh;
  const unsigned int KPQ  = param.k * PQ;
  const unsigned int NKPQ = param.n * KPQ;

  // loop bounds, constexpr where possible allows for loop unrolling
  constexpr unsigned int mma_tiles_per_warp_k = 4;
  constexpr unsigned int mma_tiles_per_warp_m = WM / MMA_M;
  constexpr unsigned int mma_tiles_per_warp_n = WN / MMA_N;
  const unsigned int z = blockIdx.z;

  const unsigned int ks =  (ksplit > 0) ? (weightKOffset + ksplit - 1) / ksplit : weightKOffset;
  const unsigned int start_k = (ksplit > 0) ? z * ks : 0;
  const unsigned int end_k = min(start_k + ks, weightKOffset);
  const unsigned int num_block_tiles_k = (ks + (BK-1)) / BK;



  // calculate block/warp indices
  const unsigned int block_m = blockIdx.y;
  const unsigned int block_n = blockIdx.x;
  const unsigned int warp_m = threadIdx.y;
  const unsigned int warp_n = threadIdx.x / 32;

  // double buffering
  extern __shared__ half shmem[];
  half* A_block_smem = shmem;
  half* B_block_smem = &shmem[BM * BK];
  constexpr int BUFFER_SIZE = BM * BK + BK * BN;

  // declare register storage
  // ptx instructions expect uint32_t registers, where each uint32_t is 2 halfs packed together
  uint32_t acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];
  uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2];
  uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n];

  // convenience cast to half for register storage
  half (&acc_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_n][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4]>(acc_register);
  half (&A_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_k][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4]>(A_register);
  half (&B_register_) [mma_tiles_per_warp_k][mma_tiles_per_warp_n][2] = reinterpret_cast<half(&)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2]>(B_register);

  // accumulators start at 0
  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++){
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++){
        acc_register_[mma_m][mma_n][0] = 0;
        acc_register_[mma_m][mma_n][1] = 0;
        acc_register_[mma_m][mma_n][2] = 0;
        acc_register_[mma_m][mma_n][3] = 0;
      }
  }

  static_assert(BM == 256);
  static_assert(BN == 256);
  static_assert(BK == 32);
  static_assert(NUM_THREADS == 256);
  float4 A_gmem_cache_reg[4];
  float4 B_gmem_cache_reg[4];

  // prefetch the first block tile of A,B into shared memory

  const half* A_block_gmem = input;
  const half* B_block_gmem = kernel + block_n * BN * weightKOffset;
  tileMemcpySwizzleA<BM, NUM_THREADS>(A_block_gmem, A_block_smem, start_k, end_k, inChannelOffset, param);
  tileMemcpySwizzleB<BN, NUM_THREADS>(B_block_gmem, B_block_smem, start_k, end_k, weightKOffset, param);

  int offset_direction = 1;

  for (unsigned int block_k = 1; block_k <= num_block_tiles_k; block_k++){
    __syncthreads();

    if (block_k != num_block_tiles_k){
      const half* A_block_gmem = input;
      const half* B_block_gmem = kernel + (block_n * BN * weightKOffset);
      tileMemcpyLoadA<BM, BK, NUM_THREADS, 4>(A_block_gmem, A_gmem_cache_reg, block_k * BK, start_k, end_k, inChannelOffset, param);
      tileMemcpyLoadB<BN, BK, NUM_THREADS, 4>(B_block_gmem, B_gmem_cache_reg, block_k * BK, start_k, end_k, weightKOffset, param);
    }
    half* A_warp_tile = A_block_smem + (warp_m * WM * BK);
    half* B_warp_tile = B_block_smem + (warp_n * WN * BK);

    ldmatrix_a<mma_tiles_per_warp_m, mma_tiles_per_warp_k, BK>(A_warp_tile, A_register_);
    ldmatrix_b<mma_tiles_per_warp_k, mma_tiles_per_warp_n, BK>(B_warp_tile, B_register_);

    // outer product between mma tiles
#pragma unroll
    for (unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++){
#pragma unroll
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++){
#pragma unroll
        for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++){
          asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3}, "
            "{%4}, "
            "{%5, %6};"
            : "=r"(acc_register[mma_m][mma_n][0]), "=r"(acc_register[mma_m][mma_n][1])
            : "r"(A_register[mma_m][mma_k][0]), "r"(A_register[mma_m][mma_k][1]),
              "r"(B_register[mma_k][mma_n])
              "r"(acc_register[mma_m][mma_n][0]), "r"(acc_register[mma_m][mma_n][1])
          );
        }
      }
    }

    if (block_k != num_block_tiles_k)
    {
      // switch smem buffers each iteration
      A_block_smem = A_block_smem + BUFFER_SIZE * offset_direction;
      B_block_smem = B_block_smem + BUFFER_SIZE * offset_direction;
      offset_direction = -1 * offset_direction;

      tileMemcpySwizzleStore<BM, NUM_THREADS, 4>(A_gmem_cache_reg, A_block_smem);
      tileMemcpySwizzleStore<BN, NUM_THREADS, 4>(B_gmem_cache_reg, B_block_smem);
    }
  }

    // reuse smem
    half *smemoutput = shmem;
    const uint lane_id = threadIdx.x % WARPSIZE;
    const uint mma_row = lane_id / 4;
    const uint mma_col = lane_id % 4;
    const uint output_lds_addr = warp_m * WM * BN/2 + lane_id * BN/2 + warp_n * WN/2;
    const uint output_sts_addr = warp_m * WM * BN/2 + mma_row * BN/2 + warp_n * WN/2  + mma_col * 2;
    const uint m_idx = block_n * BN + warp_n * WN;
    const uint n_idx = block_m * BM + warp_m * WM + lane_id;

#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        __syncthreads();
#pragma unroll
        for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
        {
            for (unsigned int mma_n = i * mma_tiles_per_warp_n/2; mma_n < (i+1)*mma_tiles_per_warp_n/2; mma_n++)
            {
                uint32_t (&reg_)[2] = reinterpret_cast<uint32_t(&)[2]>(acc_register_[mma_m][mma_n]);
                uint idx = output_sts_addr +
                            mma_m * MMA_M * BN / 2 + (mma_n - i * mma_tiles_per_warp_n/2) * MMA_N;
                idx = idx ^ ((idx & 0b110000000000) >> 9);
                idx = idx ^ ((idx & 0b1110000000) >> 4);
                uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(&smemoutput[idx]);
                dst_ptr[0] = reg_[0];
                idx = (idx + 8 * BN / 2 ) ^ 0b010;
                dst_ptr = reinterpret_cast<uint32_t*>(&smemoutput[idx]);
                dst_ptr[0] = reg_[1];
            }
        }
        __syncthreads();
        const unsigned int  m_i_wn = m_idx + i * WN / 2;
#pragma unroll
        for (int subk = 0; subk < WN / 4; ++subk){
            const uint row =  m_i_wn + subk*2;
            uint idx = output_lds_addr + subk*2;
            idx = idx ^ ((idx & 0b110000000000) >> 9);
            idx = idx ^ ((idx & 0b1110000000) >> 4);
#pragma unroll
            for (int j = 0; j < 4; ++j){
                const uint gemm_i =  n_idx + j*32;
                const int n = fastdiv(gemm_i, param.OHOW_fastdiv);
                const int col = fastmodulo(gemm_i, param.OHOW_fastdiv);
                uint32_t dst_ptr = *(reinterpret_cast<uint32_t*>(&smemoutput[idx+j*32*BN/2]));
                half (&res_)[2] = reinterpret_cast<half(&)[2]>(dst_ptr);
                if (n < param.n && row < param.k && col < PQ) {
                    if constexpr (ksplit > 0) {
                        const uint outOffset = z * NKPQ +
                                               n * KPQ +
                                               row * PQ + col;
                        output[outOffset] = res_[0];
                    } else {
                        const uint outOffset = n * KPQ + row * PQ + col;
                        output[outOffset] = res_[0];
                    }
                }
                if (n < param.n && row+1 < param.k && col < PQ) {
                    if constexpr (ksplit > 0) {
                        const uint outOffset = z * NKPQ +
                                               n * KPQ +
                                               (row+1) * PQ + col;
                        output[outOffset] = res_[1];
                    } else {
                        const uint outOffset = n * KPQ + (row+1) * PQ + col;
                        output[outOffset] = res_[1];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED(input);
    GGML_UNUSED(kernel);
    GGML_UNUSED(output);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}


#define NUM_VARIANTS 4

/*
  conv_shapes[][0]: ne_input=[384,512,256,1],ne_kernel=[3,3,256,256]
  conv_shapes[][1]: ne_input=[96,128,512,1],ne_kernel=[3,3,512,512]
  conv_shapes[][2]: ne_input=[192,256,512,1git diff],ne_kernel=[3,3,512,512]
*/
constexpr static int conv_shapes[][NUM_VARIANTS] = {
    { 128, 128,  128, 256 }, // BM
    { 256,  128,  256, 128 }, // BN
    { 8, 8, 8, 8 }, // BK
    { 128, 64,  32, 128   }, // WM
    { 32,  32 ,  256, 32   }, // WN
    { 2,   2,  1, 1   }, // WNITER
    { 8,   4,  4, 4  }, // TM
    { 8,   4,  8, 8   }, // TN
    { 256,  256, 128, 256}	    //  NUM_THREADS
};

template <typename T, unsigned int CONV_SHAPE>
static void conv2d_implicit_cuda(const float * X_D, const T * K_D, float * Y_D, const param_t P, cudaStream_t st) {

    const uint BM = conv_shapes[0][CONV_SHAPE];
    const uint BN = conv_shapes[1][CONV_SHAPE];
    const uint BK = conv_shapes[2][CONV_SHAPE];
    const uint WM = conv_shapes[3][CONV_SHAPE];
    const uint WN = conv_shapes[4][CONV_SHAPE];
    const uint WNITER = conv_shapes[5][CONV_SHAPE];
    const uint TM = conv_shapes[6][CONV_SHAPE];
    const uint TN = conv_shapes[7][CONV_SHAPE];
    const uint NUM_THREADS = conv_shapes[8][CONV_SHAPE];
    int blockx = ((P.Oh * P.Ow + BM - 1) / BM); // blockx  number
    int blocky = (P.k + BN-1) / BN;             // blocky  number
    int blockz = P.n;                           // blockz  number
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 thblock(NUM_THREADS, thready, threadz);
    dim3 grid(blockx, blocky, blockz);

    conv2d_implicit_kernel<T, BM, BN, BK, WM, WN,
          WNITER, TM, TN, NUM_THREADS, 1, false, 0><<<grid, thblock, 0, st>>>(X_D, K_D, Y_D, P);
}

template<const int BM, const int BN, const int BK,
        const int WM, const int WN, const int WK,  const int ksplit,
        const unsigned int ThreadsM, const unsigned int ThreadsN,
        const int NUM_THREADS>
static void launch_conv2d_implicit_split_kernel(ggml_backend_cuda_context & ctx, const half *X_H, const half *K_H, float *Y_D,
                    const unsigned int BlocksM, const unsigned int BlocksN,
                    const unsigned int shmem_bytes,
                    const param_t P, cudaStream_t st){

        int id = ggml_cuda_get_device();

        ggml_cuda_pool_alloc<half> Y_H(ctx.pool(id), ksplit * P.k * P.Oh * P.Ow * P.n);
        cudaFuncSetAttribute(conv2d_implicit_kernel<BM, BN, BK, WM, WN, WK, ksplit, NUM_THREADS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,    65536); // set shared memory limit to 64KB which is maximum for sm_75
        dim3 gridDim(BlocksN, BlocksM, ksplit);
        dim3 blockDim(ThreadsN, ThreadsM);

        conv2d_implicit_kernel<BM, BN, BK,
            WM, WN, WK, ksplit, NUM_THREADS><<<gridDim, blockDim, shmem_bytes, st>>>(X_H, K_H, Y_H.get(), P);

        const unsigned int nrows = P.n * P.k * P.Oh * P.Ow;
        const unsigned int blockx = (nrows + 511) / 512;
        const dim3 block_nums(blockx, 1, 1);
        const dim3 block_dims(512, 1, 1);
        reduce_f32<half, float><<<block_nums, block_dims, 0, st>>>(Y_H.get(), Y_D, nrows, ksplit);
}

static void conv2d_implicit_cuda_f16(ggml_backend_cuda_context & ctx, const float * X_D, const half * K_D, float * Y_D, int cc, const param_t P, cudaStream_t st) {

    // if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && P.c % 8 == 0 && (P.r > 1 || P.s > 1)) {
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && P.c % 8 == 0 && (P.r > 1 || P.s > 1)) {

        int id = ggml_cuda_get_device();

        int64_t ne = P.c * P.h * P.w * P.n;
        int64_t ne00 = P.c;
        int64_t ne01 = P.h * P.w;
        ggml_cuda_pool_alloc<half> input_f16(ctx.pool(id), ne);

        dim3 dimGrid( (ne01 + CUDA_NCHW_2_NHWC_TILE_DIM - 1) / CUDA_NCHW_2_NHWC_TILE_DIM,
                      (ne00 + CUDA_NCHW_2_NHWC_TILE_DIM - 1) / CUDA_NCHW_2_NHWC_TILE_DIM,
                      (ne/(ne00*ne01) + CUDA_NCHW_2_NHWC_BLOCK_NM - 1) / CUDA_NCHW_2_NHWC_BLOCK_NM) ;
        dim3 dimBlock(CUDA_NCHW_2_NHWC_TILE_DIM,CUDA_NCHW_2_NHWC_BLOCK_ROWS, 1);
        NCHW2NHWC<float, half><<<dimGrid, dimBlock, 0, st>>>(X_D, input_f16.get(), ne, ne00, ne01);

        ne = P.c * P.r * P.s * P.k;
        ne01 = P.r * P.s;
        ggml_cuda_pool_alloc<half> kernel_f16(ctx.pool(id), ne);
        if (ne01 > 1){
          dim3 dimGrid1((ne00 + CUDA_NCHW_2_NHWC_BLOCK_C - 1) / CUDA_NCHW_2_NHWC_BLOCK_C,
                          (ne/(ne00*ne01) + CUDA_NCHW_2_NHWC_BLOCK_NM - 1) / CUDA_NCHW_2_NHWC_BLOCK_NM,
                          1) ;
          if (ne01 == 25) {
            constexpr unsigned int mask = filter_swizzle_mask(25, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 25, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          }else if (ne01 == 16) {
            constexpr unsigned int mask = filter_swizzle_mask(16, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 16, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          }else if (ne01 == 9) {
            constexpr unsigned int mask = filter_swizzle_mask(9, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 9, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          } else if (ne01 == 8) {
            constexpr unsigned int mask = filter_swizzle_mask(8, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 8, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          } else if (ne01 == 7) {
            constexpr unsigned int mask = filter_swizzle_mask(7, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 7, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          } else if (ne01 == 6) {
            constexpr unsigned int mask = filter_swizzle_mask(6, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 6, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          } else if (ne01 == 5) {
            constexpr unsigned int mask = filter_swizzle_mask(5, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 5, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          } else if (ne01 == 4) {
            constexpr unsigned int mask = filter_swizzle_mask(4, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 4, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          } else if (ne01 == 3) {
            constexpr unsigned int mask = filter_swizzle_mask(3, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 3, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          } else if (ne01 == 2) {
            constexpr unsigned int mask = filter_swizzle_mask(2, CUDA_NCHW_2_NHWC_BLOCK_C);
            NCHW2NHWC<half, half, mask, 2, CUDA_NCHW_2_NHWC_BLOCK_C><<<dimGrid1, CUDA_NCHW_2_NHWC_BLOCK_C, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          } else {
            dim3 dimGrid2((ne01 + CUDA_NCHW_2_NHWC_TILE_DIM - 1) / CUDA_NCHW_2_NHWC_TILE_DIM,
                          (ne00 + CUDA_NCHW_2_NHWC_TILE_DIM - 1) / CUDA_NCHW_2_NHWC_TILE_DIM,
                          (ne/(ne00*ne01) + CUDA_NCHW_2_NHWC_BLOCK_NM - 1) / CUDA_NCHW_2_NHWC_BLOCK_NM) ;
            NCHW2NHWC<half, half><<<dimGrid2, dimBlock, 0, st>>>(K_D, kernel_f16.get(), ne, ne00, ne01);
          }
        }

        const half *X_H = input_f16.get();
        const half *K_H = ne01 == 1 ? K_D : kernel_f16.get();

        constexpr unsigned int BM_dim = 256;
        constexpr unsigned int BN_dim = 256;
        constexpr unsigned int BK_dim = 32;

        constexpr unsigned int WARPS_PER_BLOCK_M = 2;
        constexpr unsigned int WARPS_PER_BLOCK_N = 4;
        constexpr unsigned int WARPS_PER_BLOCK_K = 4;

        constexpr unsigned int WM_dim = BM_dim / WARPS_PER_BLOCK_M;
        constexpr unsigned int WN_dim = BN_dim / WARPS_PER_BLOCK_N;
        constexpr unsigned int WK_dim = BK_dim / WARPS_PER_BLOCK_K;

        static_assert(WN_dim % 4 == 0,  "final output requires this to be bank conflicts free");

        const unsigned int BlocksM =  (P.n * P.Oh * P.Ow + BM_dim - 1) / BM_dim;
        const unsigned int BlocksN =  (P.k + BN_dim - 1) / BN_dim;
        constexpr unsigned int ThreadsM = WARPS_PER_BLOCK_M;
        constexpr unsigned int ThreadsN = WARPSIZE * WARPS_PER_BLOCK_N;
        constexpr unsigned int NumThreads = ThreadsM * ThreadsN;
        const unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * 2 * sizeof(half);

        const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        // if (BlocksM * BlocksN < nsm && P.c >= 8 * ksplit && (P.c * P.r * P.s) % (8*ksplit) == 0) {
        if (BlocksM * BlocksN < (unsigned int)nsm){

            int ks = min(12, nsm / (BlocksM * BlocksN));
            int j;
            bool can_split = false;
            for (j = ks; j >= 2; j--){
               if ((P.c * P.r * P.s) % (8*j) == 0){
                 can_split = true;
                 break;
               }
            }
            if(can_split){
              if (j == 2) {
                const unsigned int ksplit = 2;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 3) {
                const unsigned int ksplit = 3;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 4) {
                const unsigned int ksplit = 4;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 5) {
                const unsigned int ksplit = 5;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 6) {
                const unsigned int ksplit = 6;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 7) {
                const unsigned int ksplit = 7;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 8) {
                const unsigned int ksplit = 8;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 9) {
                const unsigned int ksplit = 9;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 10) {
                const unsigned int ksplit = 10;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if (j == 11) {
                const unsigned int ksplit = 11;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              } else if(j == 12) {
                const unsigned int ksplit = 12;
                launch_conv2d_implicit_split_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, ksplit,
                ThreadsM, ThreadsN, NumThreads>(ctx, X_H, K_H, Y_D, BlocksM, BlocksN, shmem_bytes, P, st);
              }
              return;
            }
        }
        ggml_cuda_pool_alloc<half> Y_H(ctx.pool(id), P.k * P.Oh * P.Ow * P.n);

        cudaFuncSetAttribute(conv2d_implicit_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, 0, NumThreads>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,    65536); // set shared memory limit to 64KB which is maximum for sm_75
        dim3 gridDim(BlocksN, BlocksM);
        dim3 blockDim(ThreadsN, ThreadsM);

        conv2d_implicit_kernel<BM_dim, BN_dim, BK_dim,
            WM_dim, WN_dim, WK_dim, 0, NumThreads>
            <<<gridDim, blockDim, shmem_bytes, st>>>(X_H, K_H, Y_H.get(), P);
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(Y_H.get(), Y_D, P.k * P.Oh * P.Ow * P.n, st);
    } else{
       conv2d_implicit_cuda<half, 1>(X_D, K_D, Y_D, P, st);
    }

}

static void conv2d_implicit_cuda_f32(ggml_backend_cuda_context & ctx, const float * X_D, const float * K_D, float * Y_D, int cc, const param_t P, cudaStream_t st) {
    conv2d_implicit_cuda<float, 1>(X_D, K_D, Y_D, P, st);
    GGML_UNUSED(ctx);
    GGML_UNUSED(cc);
}

void ggml_cuda_op_conv2d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);


    cudaStream_t st = ctx.stream();
    const int cc            = ggml_cuda_info().devices[ctx.device].cc;

    const int32_t * p    = (const int32_t *) dst->op_params;
    const uint       ST_X = p[0];  // stride_x
    const uint       ST_Y = p[1];  // stride_y
    const uint       PD_X = p[2];  // padding_x
    const uint       PD_Y = p[3];  // padding_y
    const uint       DL_X = p[4];  // dilation_x
    const uint       DL_Y = p[5];  // dilation_y
    // const int       LT   = p[6];  // layout

    // GGML_ASSERT(LT == 0 || LT == 1);

    // same number of input channels
    // GGML_ASSERT(LT == 0 ? input->ne[0] == kernel->ne[0] : input->ne[2] == kernel->ne[2]);
    // No cwhn
    GGML_ASSERT(p[6] == false);

    const uint IW = input->ne[0];   // input_w
    const uint IH = input->ne[1];   // input_h
    const uint OW = dst->ne[0];     // output_w
    const uint OH = dst->ne[1];     // output_h
    const uint KW = kernel->ne[0];  // kernel_w
    const uint KH = kernel->ne[1];  // kernel_h
    const uint IC = input->ne[2];   // input_channels

    const uint OC = kernel->ne[3];  // ouptut_chanles
    const uint B  = input->ne[3];   // n_batches

    param_t params = { B, IC, IH, IW, OC, KH, KW, ST_Y, ST_X, PD_Y, PD_X, DL_Y, DL_X, OH, OW,
                      init_fastdiv_values(KW*IC),
                      init_fastdiv_values(OW),
                      init_fastdiv_values(IC),
                      init_fastdiv_values(KW*KH),
                      init_fastdiv_values(KW),
                      init_fastdiv_values(OW*OH)};

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_implicit_cuda_f16(ctx, X_D, (half *) K_D, Y_D, cc, params, st);
    } else {
        conv2d_implicit_cuda_f32(ctx, X_D, K_D, Y_D, cc, params, st);
    }
}
