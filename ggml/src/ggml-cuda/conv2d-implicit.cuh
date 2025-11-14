#pragma once
#include "common.cuh"

typedef struct{
    unsigned int      n;                              //batch size
    unsigned int      c;                              //number if channels
    unsigned int      h;                              //height
    unsigned int      w;                              //width
    unsigned int      k;                              //number of filters
    unsigned int      r;                              //filter height
    unsigned int      s;                              //filter width
    unsigned int      u;                              //stride height
    unsigned int      v;                              //stride width
    unsigned int      p;                              //padding height
    unsigned int      q;                              //padding width
    unsigned int      d_h;                            //dilation height
    unsigned int      d_w;                            //dilation width
    unsigned int      Oh;                             //output height
    unsigned int      Ow;                             //output width
    uint3 SC_fastdiv;
    uint3 OW_fastdiv;
    uint3 C_fastdiv;
    uint3 RS_fastdiv;
    uint3 S_fastdiv;
    uint3 OHOW_fastdiv;
    int64_t inc_next[3];
} param_t;


/// Clears the predicates

template<const unsigned int K_STRID>
__host__ __device__ void clear_mask(unsigned int masks_[][2], bool clear = true) {

#pragma unroll
    for (int s = 0; s < K_STRID; ++s) {
        masks_[s][0] = clear ? 0 : masks_[s][0];
        masks_[s][1] = clear ? 0 : masks_[s][1];
    }
}

template<const unsigned int K_STRID>
__host__ __device__ void add_byte_offset(int64_t element_offset[], const int64_t offset){
#pragma unroll
    for (int s = 0; s < K_STRID; ++s) {
       element_offset[s] += offset;
    }
}

template<const unsigned int TILE_ROWS,
         const unsigned int TILE_COLS,
         const unsigned int A_K_STRID,
         const unsigned int ROW_STEP>
__device__ void prepareIteratorA(const int thread_idx,
                                 unsigned int masks[][2],
                                 int64_t element_offset[],
                                 const param_t param){
    int offset_n[A_K_STRID];
    int offset_p[A_K_STRID];
    int offset_q[A_K_STRID];

    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int chw = param.c * param.h * param.w;

#pragma unroll
    for (int s = 0; s < A_K_STRID; ++s) {

    //   pointer_[s] = reinterpret_cast<char const *>(ptr);

    //   int offset_npq = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;
        const unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
        offset_n[s]  = fastdiv(gemm_i, param.OHOW_fastdiv);
        unsigned int npq_res = fastmodulo(gemm_i, param.OHOW_fastdiv);
        offset_p[s] = fastdiv(npq_res, param.OW_fastdiv); //* param.u - param.p;
        offset_q[s] = fastmodulo(npq_res, param.OW_fastdiv); // * param.v - param.q;
        const int h = offset_p[s] * (int)param.u - (int) param.p;
        const int w = offset_q[s] * (int)param.v - (int) param.q;

    //   if(threadIdx.x < 32 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    //    printf("%d, %d : %d, %d, %d, %d offset (%d, %d, %d), kele %llu Kcont %d\n ", thread_idx, s,
    //     // printf("[%s - %d] %d, %d : %d, %d, %d, %d\n ", __FUNCTION__, __LINE__, thread_idx, s,
    //               threadblock_offset.row(), thread_coord.strided(), ThreadMap::Delta::kStrided,
    //             offset_npq, offset_n[s], offset_p[s], offset_q[s], AccessType::kElements, 
    //             ThreadMap::Iterations::kContiguous);

        element_offset[s] =  offset_n[s] * (int64_t)chw + h * (int64_t)(param.c * param.w) + w * (int64_t)param.c;

        // if(element_offset[s] >= 327680)
        //     printf("(%d, %d, %d, %d, %d), %d, %lld, %d, %d, %d, %d, %d, %u, %u, %u \n",
        //     threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z,
        //            s, element_offset[s], offset_n[s], offset_p[s], offset_q[s], h, w, chw, param.c * param.w, param.c);
        thread_row += ROW_STEP;
    }

    clear_mask<A_K_STRID>(masks);

    for (int r = 0; r < param.r; ++r) {
#pragma unroll
      for (int s_idx = 0; s_idx < A_K_STRID; ++s_idx) {
        const int h = offset_p[s_idx] * param.u - param.p + r * param.d_h;

        bool pred = (offset_n[s_idx] < param.n && h >= 0 && h < param.h);
        masks[s_idx][0] |= (pred << r);
      }
    }

    for (int s = 0; s < param.s; ++s) {
#pragma unroll
      for (int s_idx = 0; s_idx < A_K_STRID; ++s_idx) {
        const int w = offset_q[s_idx] * param.v - param.q + s * param.d_w;
        bool pred = (w >= 0 && w < param.w);
        masks[s_idx][1] |= (pred << s);
      }
    }
}

// same as above, but writes are swizzled to avoid bank conflicts when shared memory is read later in the kernel
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpySwizzleB(
    const half* src,
    half* dst,
    const unsigned int curR,
    const unsigned int curS,
    const unsigned int start_k,
    const unsigned int end_k,
    const unsigned int src_stride,
    param_t param
){
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
    constexpr unsigned int SWIZZLE_BITS_1 = 4;
    constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
    constexpr unsigned int SWIZZLE_BITS_2 = 2;
    constexpr unsigned int TILE_COLS = 32;

    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

    // const unsigned int ki = (curR*param.s+curS)*param.c + start_k+thread_col*8;
    // const unsigned int curR = fastdiv(ki,                                 param.SC_fastdiv); // channel offset
    // const unsigned int curS = fastdiv(fastmodulo(ki,    param.SC_fastdiv), param.C_fastdiv); // kernel r offset
    // const unsigned int curC = fastmodulo(fastmodulo(ki, param.SC_fastdiv), param.C_fastdiv); //
    const unsigned int curC = start_k+thread_col*8;
    const unsigned int ki = (curR*param.s+curS)*param.c + curC;

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++){
        // apply swizzle to the dst index
        const unsigned int src_index = thread_row * src_stride + ki;
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        if (thread_row + blockIdx.x * TILE_ROWS < param.k && curC < end_k){
            dst_float4[dst_index] = reinterpret_cast<const float4 *>(&src[src_index])[0];
        }else{ // read 4 halves
            dst_float4[dst_index] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(src_stride);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}


// this is a special case of the above for when TILE_COLS == 32
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS>
__device__ __forceinline__ unsigned int tileMemcpySwizzleA(
    const half* src,
    half* dst,
    const unsigned int curR,
    const unsigned int curS,
    unsigned int masks[][2],
    const int64_t element_offset[],
    const unsigned int thread_idx,
    const unsigned int start_k,
    const unsigned int end_k,
    const unsigned int inChannelOffset,
    param_t param
)
{
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
    constexpr unsigned int SWIZZLE_BITS_1 = 4;
    constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
    constexpr unsigned int SWIZZLE_BITS_2 = 2;
    constexpr unsigned int TILE_COLS = 32;

    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    // const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

    // const unsigned int ki = start_k+thread_col*8;
    // const unsigned int chw = param.c * param.h * param.w;
    // const unsigned int curR = fastdiv(ki,                                 param.SC_fastdiv); // channel offset
    // const unsigned int curS = fastdiv(fastmodulo(ki,    param.SC_fastdiv), param.C_fastdiv); // kernel r offset
    // const unsigned int curC = fastmodulo(fastmodulo(ki, param.SC_fastdiv), param.C_fastdiv); // kernel r offset
    const unsigned int curC = start_k+thread_col*8;
    clear_mask<NUM_ITERS>(masks, curC >= end_k);

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++){
        bool valid = (masks[i][0] & (1u << curR)) && (masks[i][1] & (1u << curS));
        // apply swizzle to the dst index
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        // if(threadIdx.x == 3 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 1){
        //    printf(" %u, %u, %u, %u, %lld, %d\n", i, curR, curS,  curC, element_offset[i], valid?1:0);
        //  }
        // if (valid && curC < end_k){
        if (valid){
        //     if(element_offset[i] >= 327680 || element_offset[i] < 0)
        //        printf("%d, %d, %d, %d, %d, %d, %d, %d, %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
        //            i, element_offset[i], curR, curS, curC);
            dst_float4[dst_index] = reinterpret_cast<const float4 *>(&src[element_offset[i]+curC])[0];
        } else{
            dst_float4[dst_index] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
    // #pragma unroll
    // for (unsigned int i = 0; i < NUM_ITERS; i++){
    //     unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
    //     unsigned int n = fastdiv(gemm_i, param.OHOW_fastdiv);
    //     unsigned int npq_res = fastmodulo(gemm_i, param.OHOW_fastdiv);
    //     int posh_ori = fastdiv(npq_res, param.OW_fastdiv) * param.u - param.p;
    //     int posw_ori = fastmodulo(npq_res, param.OW_fastdiv) * param.v - param.q;
    //     // unsigned int inOffset = n * param.c * param.h * param.w;
    //     int curH = posh_ori + curR * param.d_h; // input h
    //     int curW = posw_ori + curS * param.d_w; // input w
    //     // apply swizzle to the dst index
    //     unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
    //     dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
    //     dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
    //     if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h &&
    //         curR < param.r && curS < param.s && curC < param.c && n < param.n && ki < end_k){
    //         const unsigned int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
    //         dst_float4[dst_index] = reinterpret_cast<const float4 *>(&src[n * chw + inOffsetTmp])[0];
    //     } else{
    //         dst_float4[dst_index] = make_float4(0.f, 0.f, 0.f, 0.f);
    //     }
    //     thread_row += ROW_STEP;
    // }
    return curC;
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(inChannelOffset);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}

template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ unsigned int tileMemcpyLoadA(
    const half* src,
    float4 (&dst_reg)[ELEMENTS_PER_THREAD],
    const unsigned int curR,
    const unsigned int curS,
    unsigned int masks[][2],
    const int64_t element_offset[],
    const unsigned int thread_idx,
    const unsigned int block_k,
    const unsigned int start_k,
    const unsigned int end_k,
    unsigned int oldC,
    const unsigned int inChannelOffset,
    param_t param
){
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);

    // const unsigned int ki = start_k+block_k+thread_col*8;
    // const unsigned int chw = param.c * param.h * param.w;

    // const unsigned int curR = fastdiv(ki,                                 param.SC_fastdiv); // channel offset
    // const unsigned int curS = fastdiv(fastmodulo(ki,    param.SC_fastdiv), param.C_fastdiv); // kernel r offset
    // const unsigned int curC = fastmodulo(fastmodulo(ki, param.SC_fastdiv), param.C_fastdiv); // kernel r offset
    const unsigned int curC = start_k+block_k+thread_col*8;
    if (curC > oldC)
        clear_mask<NUM_ITERS>(masks, curC >= end_k);

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++){
         bool valid = (masks[i][0] & (1u << curR)) && (masks[i][1] & (1u << curS));
        //  if(threadIdx.x == 3 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 1){
        //    printf(" %u, %u, %u, %u, %u, %lld, %d\n", i, curR, curS, oldC, curC, element_offset[i], valid?1:0);
        //  }
        if (valid) {
            dst_reg[i] = reinterpret_cast<const float4 *>(&src[element_offset[i]+curC])[0];
        } else{
            dst_reg[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    // #pragma unroll
    // for (unsigned int i = 0; i < NUM_ITERS; i++){
    //     unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
    //     unsigned int n = fastdiv(gemm_i, param.OHOW_fastdiv);
    //     unsigned int npq_res = fastmodulo(gemm_i, param.OHOW_fastdiv);
    //     int posh_ori = fastdiv(npq_res, param.OW_fastdiv) * param.u - param.p;
    //     int posw_ori = fastmodulo(npq_res, param.OW_fastdiv) * param.v - param.q;
    //     // unsigned int inOffset = n * param.c * param.h * param.w;
    //     int curH = posh_ori + curR * param.d_h; // input h
    //     int curW = posw_ori + curS * param.d_w; // input w
    //     bool valid = (masks[i][0] & (1u << curR)) && (masks[i][1] & (1u << curS));
    //     bool ovl = curH >= 0 && curW >= 0 && curW < param.w && curH < param.h &&
    //         curR < param.r && curS < param.s && curC < param.c && n < param.n && ki < end_k;
    //     const int txx = curH * (int) inChannelOffset + curW * (int)param.c + (int)curC;

    //     if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 1){
    //          printf(" %u, %u, %u, %u, %u, %lld, %lld, %d, %d, %d\n", i, curR, curS, oldC, curC, 
    //             element_offset[i], element_offset[i]+(int64_t)curC, n * (int)chw + txx,
    //             valid?1:0, ovl?1:0);
    //     }

    //     if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h &&
    //         curR < param.r && curS < param.s && curC < param.c && n < param.n && ki < end_k){
    //         const unsigned int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
    //         dst_reg[i] = reinterpret_cast<const float4 *>(&src[n * chw + inOffsetTmp])[0];
    //     } else{
    //         dst_reg[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    //     }
    //     thread_row += ROW_STEP;
    // }
    return curC;
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst_reg);
    GGML_UNUSED(block_k);
    GGML_UNUSED(inChannelOffset);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}


template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpyLoadB(
    const half* src,
    float4 (&dst_reg)[ELEMENTS_PER_THREAD],
    const unsigned int curR,
    const unsigned int curS,
    const unsigned int block_k,
    const unsigned int start_k,
    const unsigned int end_k,
    const unsigned int src_stride,
    param_t param
){
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);

    // const unsigned int curR = fastdiv(ki,                                 param.SC_fastdiv); // channel offset
    // const unsigned int curS = fastdiv(fastmodulo(ki,    param.SC_fastdiv), param.C_fastdiv); // kernel r offset
    // const unsigned int curC = fastmodulo(fastmodulo(ki, param.SC_fastdiv), param.C_fastdiv); //
    const unsigned int curC = start_k+block_k+thread_col*8;
    const unsigned int ki = (curR*param.s+curS)*param.c + curC;

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++){
        const unsigned int src_index = thread_row * src_stride + ki;
        if (thread_row + blockIdx.x * TILE_ROWS < param.k && curC < end_k){
            dst_reg[i] = reinterpret_cast<const float4 *>(&src[src_index])[0];
        }else{ // read 4 halves
            dst_reg[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst_reg);
    GGML_UNUSED(block_k);
    GGML_UNUSED(src_stride);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}


// same as above but without the swizzle

// this is a special case of the above for when TILE_COLS == 32
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpySwizzleStore(
    const float4 (&src_reg)[ELEMENTS_PER_THREAD],
    half* dst
)
{
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
    constexpr unsigned int SWIZZLE_BITS_1 = 4;
    constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
    constexpr unsigned int SWIZZLE_BITS_2 = 2;
    constexpr unsigned int TILE_COLS = 32;

    // reinterpret input/output as float4
    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        // apply swizzle to the dst index
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        dst_float4[dst_index] =  src_reg[i];
        thread_row += ROW_STEP;
    }
#else
    GGML_UNUSED(src_reg);
    GGML_UNUSED(dst);
    NO_DEVICE_CODE;
#endif
}

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
}

template<typename T, const int BN, const int rowStrideA, const int layout,
         const bool vec_load, const int ksplit, const int PAD>
__device__ __forceinline__ void loadFilter(const T * __restrict__ kernel,
                                                 T * __restrict__ smemweight,
                                           const unsigned int by,
                                           const unsigned int innerRowA,
                                           const unsigned int innerColA,
                                           const unsigned int weightKOffset,
                                           const unsigned int start_k,
                                           const unsigned int end_k,
                                           const param_t param){

    const unsigned int weight_sts_addr = innerRowA + innerColA * (BN+PAD) * 4;
    const unsigned int kidx = start_k + innerColA * 4;
#pragma unroll
    for (int offset = 0; offset + rowStrideA <= BN; offset += rowStrideA) {
        const unsigned int nidx = by * BN + innerRowA + offset;
        if (vec_load) {
            if (nidx < param.k && kidx < end_k) {
                if constexpr (std::is_same_v<T, float>){
                    float4 tmp = reinterpret_cast<const float4 *>(&kernel[nidx * weightKOffset + kidx])[0];
                    smemweight[weight_sts_addr + offset +          0] = tmp.x;
                    smemweight[weight_sts_addr + offset +   (BN+PAD)] = tmp.y;
                    smemweight[weight_sts_addr + offset + 2*(BN+PAD)] = tmp.z;
                    smemweight[weight_sts_addr + offset + 3*(BN+PAD)] = tmp.w;
                } else { // read 4 halves
                    float2 tmp = reinterpret_cast<const float2 *>(&kernel[nidx * weightKOffset + kidx])[0];
                    const half *val = reinterpret_cast<const half *>(&tmp);
                    smemweight[weight_sts_addr + offset +          0] = val[0];
                    smemweight[weight_sts_addr + offset +   (BN+PAD)] = val[1];
                    smemweight[weight_sts_addr + offset + 2*(BN+PAD)] = val[2];
                    smemweight[weight_sts_addr + offset + 3*(BN+PAD)] = val[3];
                }
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = (T)0.f;
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (nidx < param.k && kidx + i < end_k) {
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = kernel[nidx * weightKOffset + kidx + i];
                } else {
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = (T)0.f;
                }
            }
        }
    }
}


template<const int BM, const int rowStrideA, const int layout,
         const bool vec_load, const int ksplit, const int PAD>
__device__ __forceinline__ void loadInput(const float * __restrict__ input,
                                                float * __restrict__ smeminput,
                                                const unsigned int bx,
                                                const unsigned int innerRowA,
                                                const unsigned int innerColA,
                                                const unsigned int start_k,
                                                const unsigned int end_k,
                                                const unsigned int PQ,
                                                const unsigned int CHW,
                                                const unsigned int inChannelOffset,
                                                const param_t param) {
    const unsigned int input_sts_addr = innerRowA + innerColA * (BM+PAD) * 4;
    const unsigned int kidx = start_k + innerColA * 4;
#pragma unroll
    for (unsigned int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        const unsigned int midx = bx * BM + innerRowA + offset;
        int n = (ksplit > 0) ? midx / PQ : blockIdx.z;
        const unsigned int npq_res = midx % PQ;
        const int posh_ori = fastdiv((ksplit > 0) ? npq_res: midx, param.OW_fastdiv) * param.u - param.p;
        const int posw_ori = fastmodulo((ksplit > 0) ? npq_res: midx, param.OW_fastdiv) * param.v - param.q;
        const unsigned int inOffset = n * CHW;
        if (vec_load) {
            const unsigned int cur0 = fastdiv(kidx,
                   layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
            const unsigned int cur1 = fastdiv(fastmodulo(kidx,
                layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
            const unsigned int cur2 = fastmodulo(fastmodulo(kidx,
                layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
            const unsigned int curC = layout == 0 ? cur2 : cur0;
            const unsigned int curR = layout == 0 ? cur0 : cur1;
            const unsigned int curS = layout == 0 ? cur1 : cur2;
            const int curH = posh_ori + curR * param.d_h; // input h
            const int curW = posw_ori + curS * param.d_w; // input w
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && kidx < end_k) {
                int inOffsetTmp = layout == 0 ?
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                float4 tmp = reinterpret_cast<const float4 *>(&input[inOffset + inOffsetTmp])[0];
                smeminput[input_sts_addr + offset +           0] = tmp.x;
                smeminput[input_sts_addr + offset +      BM+PAD] = tmp.y;
                smeminput[input_sts_addr + offset +  2*(BM+PAD)] = tmp.z;
                smeminput[input_sts_addr + offset +  3*(BM+PAD)] = tmp.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = 0.f;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int cur0 = fastdiv(kidx + i,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
                const unsigned int cur1 = fastdiv(fastmodulo(kidx + i,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const unsigned int cur2 = fastmodulo(fastmodulo(kidx + i,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const unsigned int curC = layout == 0 ? cur2 : cur0;
                const unsigned int curR = layout == 0 ? cur0 : cur1;
                const unsigned int curS = layout == 0 ? cur1 : cur2;
                const int curH = posh_ori + curR * param.d_h; // input h
                const int curW = posw_ori + curS * param.d_w; // input w
                if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && kidx + i < end_k) {
                    int inOffsetTmp = layout == 0 ?
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = input[inOffset + inOffsetTmp];
                } else {
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = 0.f;
                }
            }
        }
    }
}


#define CUDA_CONV2D_IMPLICT_BLOCK_SIZE 256
void ggml_cuda_op_conv2d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
