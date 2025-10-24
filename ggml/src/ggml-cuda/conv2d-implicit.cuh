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
    unsigned int      layout;
    uint3 SC_fastdiv;
    uint3 OW_fastdiv;
    uint3 C_fastdiv;
    uint3 RS_fastdiv;
    uint3 S_fastdiv;
    uint3 OHOW_fastdiv;
} param_t;

// #if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
// same as above, but writes are swizzled to avoid bank conflicts when shared memory is read later in the kernel
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpySwizzleB(
    const half* src,
    half* dst,
    const unsigned int src_stride,
    param_t param
)
{
    // constexpr unsigned int SWIZZLE_MASK = 0b111 << SWIZZLE_BITS;

    // // reinterpret input/output as float4
    // float4* src_float4 = reinterpret_cast<float4*>(src);
    // float4* dst_float4 = reinterpret_cast<float4*>(dst);
    // const unsigned int src_stride_vectorized = src_stride / 8;

    // // # of threads is multiple of # of columns in the tile
    // constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    // static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // // flatten out 2d grid of threads into in order of increasing threadIdx.x
    // const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // // assign each thread a row/column in the tile, calculate how many iterations we need
    // // to cover the whole tile
    // constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    // constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    // unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    // const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    
    // #pragma unroll
    // for (unsigned int i = 0; i < NUM_ITERS; i++)
    // {
    //     // apply swizzle to the dst index
    //     const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
    //     unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
    //     dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);
    //     if (thread_col * 8 < param.k && start_k + innerColA * 4 < end_k){
    //                 float4 tmp = reinterpret_cast<const float4 *>(&src[thread_row * src_stride_vectorized  + thread_col*8)[0];
    //                 dst_float4[dst_index] =  src_float4[src_index];
    //         }else{ // read 4 halves
    //         dst_float4[dst_index] =  make_float4(0.f, 0.f, 0.f, 0.f);
    //     }
    //     thread_row += ROW_STEP;
    // }

    constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
    constexpr unsigned int SWIZZLE_BITS_1 = 4;
    constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
    constexpr unsigned int SWIZZLE_BITS_2 = 2;
    constexpr unsigned int TILE_COLS = 32;

    // reinterpret input/output as float4
    // float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    // const unsigned int src_stride_vectorized = src_stride / 8;

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
  // TODO: next block_k loop
    const unsigned int curR = fastdiv(thread_col*8,                                 param.SC_fastdiv); // channel offset
    const unsigned int curS = fastdiv(fastmodulo(thread_col*8,    param.SC_fastdiv), param.C_fastdiv); // kernel r offset
    const unsigned int curC = fastmodulo(fastmodulo(thread_col*8, param.SC_fastdiv), param.C_fastdiv); //

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        // apply swizzle to the dst index
        const unsigned int src_index = thread_row * src_stride + thread_col * 8;
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        if (thread_row < param.k && curR < param.r && curS < param.s && curC < param.c){
            dst_float4[dst_index] = reinterpret_cast<const float4 *>(&src[src_index])[0];
        }else{ // read 4 halves
            dst_float4[dst_index] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
}


// this is a special case of the above for when TILE_COLS == 32
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpySwizzleA(
    const half* src,
    half* dst,
    // const unsigned int src_stride,
    const unsigned int inChannelOffset,
    param_t param
)
{
    constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
    constexpr unsigned int SWIZZLE_BITS_1 = 4;
    constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
    constexpr unsigned int SWIZZLE_BITS_2 = 2;
    constexpr unsigned int TILE_COLS = 32;

    // reinterpret input/output as float4
    // float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    // const unsigned int src_stride_vectorized = src_stride / 8;

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


    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        // unsigned int gemm_i = blockDim.y * TILE_ROWS + thread_row;
        unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
        unsigned int n = fastdiv(gemm_i, param.OHOW_fastdiv);
        unsigned int npq_res = fastmodulo(gemm_i, param.OHOW_fastdiv);
        int posh_ori = fastdiv(npq_res, param.OW_fastdiv) * param.u - param.p;
        int posw_ori = fastmodulo(npq_res, param.OW_fastdiv) * param.v - param.q;
        unsigned int inOffset = n * param.c * param.h * param.w;
        // TODO: next block_k loop
        const unsigned int curR = fastdiv(thread_col*8,                                 param.SC_fastdiv); // channel offset
        const unsigned int curS = fastdiv(fastmodulo(thread_col*8,    param.SC_fastdiv), param.C_fastdiv); // kernel r offset
        const unsigned int curC = fastmodulo(fastmodulo(thread_col*8, param.SC_fastdiv), param.C_fastdiv); // kernel r offset
        int curH = posh_ori + curR * param.d_h; // input h
        int curW = posw_ori + curS * param.d_w; // input w
        // apply swizzle to the dst index
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && 
            curR < param.r && curS < param.s && curC < param.c){
            // const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
            const unsigned int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
            dst_float4[dst_index] = reinterpret_cast<const float4 *>(&src[inOffset + inOffsetTmp])[0];
        } else{
            dst_float4[dst_index] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
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
    const unsigned int inChannelOffset,
    param_t param
)
{
    // reinterpret input/output as float4
    // const float4* src_float4 = reinterpret_cast<const float4*>(src);
    // const unsigned int src_stride_vectorized = src_stride / 8;

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
        // const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
        // dst_reg[i] = src_float4[src_index];
        // thread_row += ROW_STEP;
        // unsigned int gemm_i = blockDim.y * TILE_ROWS + thread_row;
        unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
        unsigned int n = fastdiv(gemm_i, param.OHOW_fastdiv);
        unsigned int npq_res = fastmodulo(gemm_i, param.OHOW_fastdiv);
        int posh_ori = fastdiv(npq_res, param.OW_fastdiv) * param.u - param.p;
        int posw_ori = fastmodulo(npq_res, param.OW_fastdiv) * param.v - param.q;
        unsigned int inOffset = n * param.c * param.h * param.w;
        // TODO: next block_k loop
        const unsigned int curR = fastdiv(block_k+thread_col*8,                                 param.SC_fastdiv); // channel offset
        const unsigned int curS = fastdiv(fastmodulo(block_k+thread_col*8,    param.SC_fastdiv), param.C_fastdiv); // kernel r offset
        const unsigned int curC = fastmodulo(fastmodulo(block_k+thread_col*8, param.SC_fastdiv), param.C_fastdiv); // kernel r offset
        int curH = posh_ori + curR * param.d_h; // input h
        int curW = posw_ori + curS * param.d_w; // input w
        if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && 
            curR < param.r && curS < param.s && curC < param.c){
            // const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
            const unsigned int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
            dst_reg[i] = reinterpret_cast<const float4 *>(&src[inOffset + inOffsetTmp])[0];
        } else{
            dst_reg[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
}


template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpyLoadB(
    const half* src,
    float4 (&dst_reg)[ELEMENTS_PER_THREAD],
    const unsigned int block_k,
    const unsigned int src_stride,
    param_t param
)
{
    // reinterpret input/output as float4
    // const float4* src_float4 = reinterpret_cast<const float4*>(src);
    // const unsigned int src_stride_vectorized = src_stride / 8;

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

    const unsigned int curR = fastdiv(block_k+thread_col*8,                                 param.SC_fastdiv); // channel offset
    const unsigned int curS = fastdiv(fastmodulo(block_k+thread_col*8,    param.SC_fastdiv), param.C_fastdiv); // kernel r offset
    const unsigned int curC = fastmodulo(fastmodulo(block_k+thread_col*8, param.SC_fastdiv), param.C_fastdiv); //

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        // const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
        // dst_reg[i] = src_float4[src_index];
        // thread_row += ROW_STEP;
        const unsigned int src_index = thread_row * src_stride + block_k + thread_col * 8;
        if (thread_row < param.k && curR < param.r && curS < param.s && curC < param.c){
            dst_reg[i] = reinterpret_cast<const float4 *>(&src[src_index])[0];
        }else{ // read 4 halves
            dst_reg[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        thread_row += ROW_STEP;
    }
}

// template<unsigned int TILE_ROWS,
// unsigned int TILE_COLS,
// unsigned int NUM_THREADS,
// unsigned int SWIZZLE_BITS,
// unsigned int ELEMENTS_PER_THREAD>
// __device__ __forceinline__ void tileMemcpySwizzleStoreB(
//     float4 src_reg[ELEMENTS_PER_THREAD],
//     half* dst
// )
// {
//     constexpr unsigned int SWIZZLE_MASK = 0b111 << SWIZZLE_BITS;

//     // reinterpret input/output as float4
//     float4* dst_float4 = reinterpret_cast<float4*>(dst);

//     // # of threads is multiple of # of columns in the tile
//     constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
//     static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
//     // flatten out 2d grid of threads into in order of increasing threadIdx.x
//     const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

//     // assign each thread a row/column in the tile, calculate how many iterations we need
//     // to cover the whole tile
//     constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
//     constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
//     unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
//     const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    
//     // compile time check that we provided the right amount of registers for storage
//     static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);
    
//     #pragma unroll
//     for (unsigned int i = 0; i < NUM_ITERS; i++)
//     {
//         // apply swizzle to the dst index
//         unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
//         dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);
//         dst_float4[dst_index] = src_reg[i];
//         thread_row += ROW_STEP;
//     }
// }

// same as above but without the swizzle
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpyStore(
    float4 src_reg[ELEMENTS_PER_THREAD],
    half* dst,
    unsigned int dst_stride_float4
)
{
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
        unsigned int dst_index = thread_row * dst_stride_float4 + thread_col;
        dst_float4[dst_index] = src_reg[i];
        thread_row += ROW_STEP;
    }
}

// this is a special case of the above for when TILE_COLS == 32
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpySwizzleStore(
    const float4 (&src_reg)[ELEMENTS_PER_THREAD],
    half* dst
)
{
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

// #endif

// constexpr unsigned int int_log2(unsigned int x)
// {
//     unsigned int result = 0;
//     while (x >>= 1)
//     {
//         result++;
//     }
//     return result;
// }

#define CUDA_CONV2D_IMPLICT_BLOCK_SIZE 256
void ggml_cuda_op_conv2d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
