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
} param_t;

#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
// same as above, but writes are swizzled to avoid bank conflicts when shared memory is read later in the kernel
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int SWIZZLE_BITS>
__device__ __forceinline__ void tileMemcpySwizzle(
    half* src,
    half* dst,
    const unsigned int src_stride
)
{
    constexpr unsigned int SWIZZLE_MASK = 0b111 << SWIZZLE_BITS;

    // reinterpret input/output as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    const unsigned int src_stride_vectorized = src_stride / 8;

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
        // apply swizzle to the dst index
        const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);
        dst_float4[dst_index] =  src_float4[src_index];
        thread_row += ROW_STEP;
    }
}


// this is a special case of the above for when TILE_COLS == 32
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpySwizzleA(
    half* src,
    half* dst,
    const unsigned int src_stride
)
{
    constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
    constexpr unsigned int SWIZZLE_BITS_1 = 4;
    constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
    constexpr unsigned int SWIZZLE_BITS_2 = 2;
    constexpr unsigned int TILE_COLS = 32;

    // reinterpret input/output as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    const unsigned int src_stride_vectorized = src_stride / 8;

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
        // apply swizzle to the dst index
        const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        dst_float4[dst_index] =  src_float4[src_index];
        thread_row += ROW_STEP;
    }
}

template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpyLoad(
    half* src,
    float4 (&dst_reg)[ELEMENTS_PER_THREAD],
    const unsigned int src_stride
)
{
    // reinterpret input/output as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    const unsigned int src_stride_vectorized = src_stride / 8;

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
        const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
        dst_reg[i] = src_float4[src_index];
        thread_row += ROW_STEP;
    }
}
#endif

#define CUDA_CONV2D_IMPLICT_BLOCK_SIZE 256
void ggml_cuda_op_conv2d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
