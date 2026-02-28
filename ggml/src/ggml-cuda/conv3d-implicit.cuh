#pragma once
#include "common.cuh"

typedef struct{
    unsigned int      n;                              //batch size
    unsigned int      c;                              //number if channels
    unsigned int      h;                              //height
    unsigned int      w;                              //width
    unsigned int      d;                              //depth
    unsigned int      k;                              //number of filters
    unsigned int      r;                              //filter height
    unsigned int      s;                              //filter width
    unsigned int      t;                              //filter depth
    unsigned int      stride0;                        //stride width
    unsigned int      stride1;                        //stride height
    unsigned int      stride2;                        //stride depth
    unsigned int      padding0;                       //padding width
    unsigned int      padding1;                       //padding height
    unsigned int      padding2;                       //padding depth
    unsigned int      dilation0;                      //dilation width
    unsigned int      dilation1;                      //dilation height
    unsigned int      dilation2;                      //dilation depth
    unsigned int      Oh;                             //output height
    unsigned int      Ow;                             //output width
    unsigned int      Od;                             //output depth
    uint3 SC_fastdiv;
    uint3 OW_fastdiv;
    uint3 C_fastdiv;
    uint3 RS_fastdiv;
    uint3 S_fastdiv;
    uint3 OHOW_fastdiv;
    uint3 PQZ_fastdiv;
    uint3 RSC_fastdiv;
    uint3 TRS_fastdiv;
} param_t;


template<const int layout>
__device__ __forceinline__ int4 inputIndices(const unsigned int kidx, param_t param) {

    const unsigned int cur0 = fastdiv(kidx,
        layout == 0 ? param.RSC_fastdiv : param.TRS_fastdiv);
    const unsigned int cur0_res = fastmodulo(kidx,
        layout == 0 ? param.RSC_fastdiv : param.TRS_fastdiv);
    const unsigned int cur1 = fastdiv(cur0_res,
        layout == 0 ? param.SC_fastdiv  : param.RS_fastdiv);
    const unsigned int cur1_res = fastmodulo(cur0_res,
        layout == 0 ? param.SC_fastdiv  : param.RS_fastdiv);
    const unsigned int cur2 = fastdiv(cur1_res,
        layout == 0 ? param.C_fastdiv  : param.S_fastdiv);
    const unsigned int cur3 = fastmodulo(cur1_res,
        layout == 0 ? param.C_fastdiv  : param.S_fastdiv);
    const unsigned int curC = layout == 0 ? cur3 : cur0;
    const unsigned int curT = layout == 0 ? cur0 : cur1;
    const unsigned int curR = layout == 0 ? cur1 : cur2;
    const unsigned int curS = layout == 0 ? cur2 : cur3;
    return make_int4(curC, curT, curR, curS);

}


// same as above, but writes are swizzled to avoid bank conflicts when shared memory is read later in the kernel
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpySwizzleB(
    const half* src,
    half* dst,
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
    const unsigned int kidx = start_k + thread_col*8;
    const int4 curIdx = inputIndices<0>(kidx, param);
    const int curC = curIdx.x;
    const int curT = curIdx.y;
    const int curR = curIdx.z;
    const int curS = curIdx.w;
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++){
        // apply swizzle to the dst index
        const unsigned int src_index = thread_row * src_stride + kidx;
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        // TODO: move some checks outside of loop?
        if (thread_row + blockIdx.x * TILE_ROWS < param.k && curR < param.r && curS < param.s && curT < param.t && curC < param.c && kidx < end_k){
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
__device__ __forceinline__ void tileMemcpySwizzleA(
    const half* src,
    half* dst,
    const unsigned int start_k,
    const unsigned int end_k,
    const unsigned int inNOffset,
    const unsigned int inDepthOffset,
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
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

    const unsigned int kidx = start_k+thread_col*8;
    const int4 curIdx = inputIndices<0>(kidx, param);

#pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++){
        unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
        unsigned int n = fastdiv(gemm_i, param.PQZ_fastdiv);
        const unsigned int npqz_res = fastmodulo(gemm_i, param.PQZ_fastdiv);
        const int posd_ori = fastdiv(npqz_res, param.OHOW_fastdiv) * param.stride2 - param.padding2;
        const int ohow_res = fastmodulo(npqz_res, param.OHOW_fastdiv);
        const int posh_ori = fastdiv(ohow_res, param.OW_fastdiv) * param.stride1 - param.padding1;
        const int posw_ori = fastmodulo(ohow_res, param.OW_fastdiv) * param.stride0 - param.padding0;
        const int curD = posd_ori + curIdx.y * param.dilation2; // input d
        const int curH = posh_ori + curIdx.z * param.dilation1; // input h
        const int curW = posw_ori + curIdx.w * param.dilation0; // input w
        const int curC = curIdx.x;
        // apply swizzle to the dst index
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        if (curH >= 0 && curW >= 0 && curD >= 0 && curW < param.w && curH < param.h && curD < param.d &&
            n < param.n && curC < param.c && kidx < end_k){
            int inOffsetTmp = curD * inDepthOffset + curH * inChannelOffset + curW * param.c + curC;
            dst_float4[dst_index] = reinterpret_cast<const float4 *>(&src[n * inNOffset + inOffsetTmp])[0];
        } else{
            dst_float4[dst_index] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
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
__device__ __forceinline__ void tileMemcpyLoadA(
    const half* src,
    float4 (&dst_reg)[ELEMENTS_PER_THREAD],
    // const unsigned int src_stride,
    const unsigned int block_k,
    const unsigned int start_k,
    const unsigned int end_k,
    const unsigned int inNOffset,
    const unsigned int inDepthOffset,
    const unsigned int inChannelOffset,
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

    const unsigned int kidx = start_k + block_k + thread_col*8;
    const int4 curIdx = inputIndices<0>(kidx, param);

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++){
        unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
        unsigned int n = fastdiv(gemm_i, param.PQZ_fastdiv);
        const unsigned int npqz_res = fastmodulo(gemm_i, param.PQZ_fastdiv);
        const int posd_ori = fastdiv(npqz_res, param.OHOW_fastdiv) * param.stride2 - param.padding2;
        const int ohow_res = fastmodulo(npqz_res, param.OHOW_fastdiv);
        const int posh_ori = fastdiv(ohow_res, param.OW_fastdiv) * param.stride1 - param.padding1;
        const int posw_ori = fastmodulo(ohow_res, param.OW_fastdiv) * param.stride0 - param.padding0;
        const int curD = posd_ori + curIdx.y * param.dilation2; // input d
        const int curH = posh_ori + curIdx.z * param.dilation1; // input h
        const int curW = posw_ori + curIdx.w * param.dilation0; // input w
        const int curC = curIdx.x;
        if (curH >= 0 && curW >= 0 && curD >= 0 && curW < param.w && curH < param.h && curD < param.d
            && n < param.n && curC < param.c && kidx < end_k){
            int inOffsetTmp = curD * inDepthOffset + curH * inChannelOffset + curW * param.c + curC;
            dst_reg[i] = reinterpret_cast<const float4 *>(&src[n * inNOffset + inOffsetTmp])[0];
        } else{
            dst_reg[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
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

    const unsigned int kidx = start_k + block_k + thread_col*8;
    const int4 curIdx = inputIndices<0>(kidx, param);
    const int curC = curIdx.x;
    const int curT = curIdx.y;
    const int curR = curIdx.z;
    const int curS = curIdx.w;
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++){
        const unsigned int src_index = thread_row * src_stride + kidx;
        // TODO : move some checks outside of the loop
        if (thread_row + blockIdx.x * TILE_ROWS < param.k && curR < param.r && curS < param.s && curT < param.t
            && curC < param.c && kidx < end_k){
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

#define CUDA_CONV3D_IMPLICT_BLOCK_SIZE 256
void ggml_cuda_op_conv3d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
