#pragma once
#include "common.cuh"

constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
constexpr unsigned int SWIZZLE_BITS_1 = 4;
constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
constexpr unsigned int SWIZZLE_BITS_2 = 2;

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
    unsigned int inChannelOffset;
    unsigned int weightKOffset;
    unsigned int PQ;
    unsigned int KPQ;
    unsigned int NKPQ;
    unsigned int CHW;
} param_t;


/// Clears the predicates

template<const unsigned int K_STRID>
__device__ void clear_mask(unsigned int masks_[][2], bool clear = true) {

#pragma unroll
    for (int s = 0; s < K_STRID; ++s) {
        masks_[s][0] = clear ? 0 : masks_[s][0];
        masks_[s][1] = clear ? 0 : masks_[s][1];
    }
}

template<const unsigned int K_STRID>
__device__ void add_byte_offset(int64_t element_offset[], const int64_t offset) {
#pragma unroll
    for (int s = 0; s < K_STRID; ++s) {
       element_offset[s] += offset;
    }
}

template<const unsigned int TILE_ROWS,
         const unsigned int TILE_COLS,
         const unsigned int A_K_STRID,
         const unsigned int ROW_STEP>
__device__ void prepareIteratorA(unsigned int thread_row,
                                 unsigned int masks[][2],
                                 int64_t element_offset[],
                                 const param_t param) {
    int offset_n[A_K_STRID];
    int offset_p[A_K_STRID];
    int offset_q[A_K_STRID];

#pragma unroll
    for (int s = 0; s < A_K_STRID; ++s) {

        const unsigned int gemm_i = blockIdx.y * TILE_ROWS + thread_row;
        offset_n[s]  = fastdiv(gemm_i, param.OHOW_fastdiv);
        unsigned int npq_res = fastmodulo(gemm_i, param.OHOW_fastdiv);
        offset_p[s] = fastdiv(npq_res, param.OW_fastdiv); //* param.u - param.p;
        offset_q[s] = fastmodulo(npq_res, param.OW_fastdiv); // * param.v - param.q;
        const int h = offset_p[s] * (int)param.u - (int) param.p;
        const int w = offset_q[s] * (int)param.v - (int) param.q;

        element_offset[s] =  offset_n[s] * (int64_t)param.CHW + h * (int64_t)(param.inChannelOffset) + w * (int64_t)param.c;

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

template <int preload=16>
__device__ void cp_async_zfill(void *ptr, void const *global_ptr, bool pred_guard = true) {
#ifdef CP_ASYNC_AVAILABLE
    unsigned int smem_ptr;
    int src_in_bytes = pred_guard ? preload : 0;

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
    "%0, smem_ptr; }\n"
    : "=r"(smem_ptr)
    : "l"(ptr));

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_ptr),
            "l"(global_ptr),
            "n"(preload), "r"(src_in_bytes));
#else
    GGML_UNUSED(ptr);
    GGML_UNUSED(global_ptr);
    GGML_UNUSED(pred_guard);
#endif
}

// same as above, but writes are swizzled to avoid bank conflicts when shared memory is read later in the kernel
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpySwizzleB(
    const half* __restrict__ src,
    half* __restrict__ dst,
    const unsigned int curR,
    const unsigned int curS,
    const unsigned int curC,
    const int64_t ki,
    const unsigned int start_k,
    const unsigned int end_k,
    unsigned int thread_row,
    const unsigned int thread_col,
    param_t param
) {
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    constexpr unsigned int TILE_COLS = 32;

    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        // apply swizzle to the dst index
        const unsigned int src_index = thread_row * param.weightKOffset  + ki;
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
#ifdef CP_ASYNC_AVAILABLE
        cp_async_zfill((void *)(&dst_float4[dst_index]), (void const *)(&src[src_index]),
                       thread_row + blockIdx.x * TILE_ROWS < param.k && curC < end_k);

#else
        if (thread_row + blockIdx.x * TILE_ROWS < param.k && curC < end_k) {
            dst_float4[dst_index] = reinterpret_cast<const float4 *>(&src[src_index])[0];
        } else { // read 4 halves
            dst_float4[dst_index] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
#endif
        thread_row += ROW_STEP;
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(curR);
    GGML_UNUSED(curS);
    GGML_UNUSED(ki);
    GGML_UNUSED(start_k);
    GGML_UNUSED(end_k);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}


// this is a special case of the above for when TILE_COLS == 32
template<unsigned int TILE_ROWS,
unsigned int NUM_THREADS>
__device__ __forceinline__ unsigned int tileMemcpySwizzleA(
    const half* __restrict__ src,
    half* __restrict__ dst,
    const unsigned int curR,
    const unsigned int curS,
    unsigned int masks[][2],
    const int64_t element_offset[],
    unsigned int thread_row,
    const unsigned int thread_col,
    const unsigned int start_k,
    const unsigned int end_k,
    param_t param
) {
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    constexpr unsigned int TILE_COLS = 32;

    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;

    const unsigned int curC = start_k+thread_col*8;
    clear_mask<NUM_ITERS>(masks, curC >= end_k);

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        bool valid = (masks[i][0] & (1u << curR)) && (masks[i][1] & (1u << curS));
        // apply swizzle to the dst index
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
#ifdef CP_ASYNC_AVAILABLE
        cp_async_zfill((void *)(&dst_float4[dst_index]), (void const *)(&src[element_offset[i]+curC]), valid);
#else
        if (valid) {
            dst_float4[dst_index] = reinterpret_cast<const float4 *>(&src[element_offset[i]+curC])[0];
        } else {
            dst_float4[dst_index] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
#endif
        thread_row += ROW_STEP;
    }
    return curC;
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(curR);
    GGML_UNUSED(curS);
    GGML_UNUSED(start_k);
    GGML_UNUSED(end_k);
    GGML_UNUSED(masks);
    GGML_UNUSED(element_offset);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}

template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ unsigned int tileMemcpyLoadA(
    const half* __restrict__ src,
    float4 (&dst_reg)[ELEMENTS_PER_THREAD],
    const unsigned int curR,
    const unsigned int curS,
    unsigned int masks[][2],
    const int64_t element_offset[],
    unsigned int thread_row,
    const unsigned int thread_col,
    const unsigned int block_k,
    const unsigned int start_k,
    const unsigned int end_k,
    unsigned int oldC,
    param_t param
) {
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);

    const unsigned int curC = start_k+block_k+thread_col*8;
    if (curC > oldC)
        clear_mask<NUM_ITERS>(masks, curC >= end_k);

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
         bool valid = (masks[i][0] & (1u << curR)) && (masks[i][1] & (1u << curS));
        if (valid) {
            dst_reg[i] = reinterpret_cast<const float4 *>(&src[element_offset[i]+curC])[0];
        } else{
            dst_reg[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    return curC;
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst_reg);
    GGML_UNUSED(block_k);
    GGML_UNUSED(curR);
    GGML_UNUSED(curS);
    GGML_UNUSED(start_k);
    GGML_UNUSED(end_k);
    GGML_UNUSED(masks);
    GGML_UNUSED(element_offset);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(oldC);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}

template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ unsigned int tileMemcpyAsyncLoadA(
    const half* __restrict__ src,
    half* __restrict__ dst,
    const unsigned int curR,
    const unsigned int curS,
    unsigned int masks[][2],
    const int64_t element_offset[],
    unsigned int thread_row,
    const unsigned int thread_col,
    unsigned int iter_idx,
    const unsigned int block_k,
    const unsigned int start_k,
    const unsigned int end_k,
    unsigned int oldC,
    param_t param
) {
#ifdef CP_ASYNC_AVAILABLE

    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    constexpr unsigned int ITER_STEPS = ROW_STEP * TILE_COLS_VECTORIZED;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);

    const unsigned int curC = start_k+block_k+thread_col*8;
    if (curC > oldC)
        clear_mask<NUM_ITERS>(masks, curC >= end_k);

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        bool valid = (masks[i][0] & (1u << curR)) && (masks[i][1] & (1u << curS));
        unsigned int dst_index = iter_idx;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);

        cp_async_zfill((void *)(&dst_float4[dst_index]), (void const *)(&src[element_offset[i]+curC]), valid);
        iter_idx += ITER_STEPS;
    }
    return curC;
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(block_k);
    GGML_UNUSED(curR);
    GGML_UNUSED(curS);
    GGML_UNUSED(start_k);
    GGML_UNUSED(end_k);
    GGML_UNUSED(masks);
    GGML_UNUSED(element_offset);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(iter_idx);
    GGML_UNUSED(oldC);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}


template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpyLoadB(
    const half* __restrict__ src,
    float4 (&dst_reg)[ELEMENTS_PER_THREAD],
    const unsigned int curR,
    const unsigned int curS,
    const unsigned int curC,
    const int64_t ki,
    const unsigned int block_k,
    const unsigned int start_k,
    const unsigned int end_k,
    unsigned int thread_row,
    const unsigned int thread_col,
    param_t param
) {
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);


    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);

    unsigned int iter_idx = thread_row * param.weightKOffset + ki;
    unsigned int krow_idx = thread_row + blockIdx.x * TILE_ROWS;
    const int ITER_STEPS = ROW_STEP * param.weightKOffset;

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        const unsigned int src_index = iter_idx;
        if (krow_idx < param.k && curC < end_k) {
            dst_reg[i] = reinterpret_cast<const float4 *>(&src[src_index])[0];
        } else { // read 4 halves
            dst_reg[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        krow_idx += ROW_STEP;
        iter_idx += ITER_STEPS;
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst_reg);
    GGML_UNUSED(block_k);
    GGML_UNUSED(curR);
    GGML_UNUSED(curS);
    GGML_UNUSED(ki);
    GGML_UNUSED(start_k);
    GGML_UNUSED(end_k);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}

template<unsigned int TILE_ROWS,
         unsigned int TILE_COLS,
         unsigned int NUM_THREADS,
         unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpyAsyncLoadB(
    const half *src,
    half *dst,
    const unsigned int curR,
    const unsigned int curS,
    const unsigned int curC,
    const int64_t ki,
    const unsigned int block_k,
    const unsigned int start_k,
    const unsigned int end_k,
    unsigned int thread_row,
    const unsigned int thread_col,
    unsigned int iter_src_idx,
    unsigned int iter_dst_idx,
    unsigned int krow_idx,
    const int ITER_SRC_STEPS,
    param_t param
) {

#ifdef CP_ASYNC_AVAILABLE

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    constexpr unsigned int ITER_DST_STEPS = ROW_STEP * TILE_COLS_VECTORIZED;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);

    iter_src_idx += ki;

    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        const unsigned int src_index = iter_src_idx;
        unsigned int dst_index = iter_dst_idx;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);

        cp_async_zfill((void *)(&dst_float4[dst_index]), (void const *)(&src[src_index]), krow_idx < param.k && curC < end_k);

        iter_src_idx += ITER_SRC_STEPS;
        krow_idx += ROW_STEP;
        iter_dst_idx += ITER_DST_STEPS;
    }
#else
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(block_k);
    GGML_UNUSED(curR);
    GGML_UNUSED(curS);
    GGML_UNUSED(ki);
    GGML_UNUSED(start_k);
    GGML_UNUSED(end_k);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    GGML_UNUSED(iter_src_idx);
    GGML_UNUSED(iter_dst_idx);
    GGML_UNUSED(krow_idx);
    GGML_UNUSED(ITER_SRC_STEPS);
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
    half* __restrict__ dst,
    unsigned int thread_row,
    const unsigned int thread_col
) {
#if __CUDA_ARCH__ >= GGML_CUDA_TURING

    constexpr unsigned int TILE_COLS = 32;

    // reinterpret input/output as float4
    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    constexpr unsigned int ITER_STEPS = ROW_STEP * TILE_COLS_VECTORIZED;

    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);

    unsigned int iter_idx = thread_row * TILE_COLS_VECTORIZED + thread_col;
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++) {
        // apply swizzle to the dst index
        unsigned int dst_index = iter_idx;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
        dst_float4[dst_index] =  src_reg[i];
        iter_idx += ITER_STEPS;
    }
#else
    GGML_UNUSED(src_reg);
    GGML_UNUSED(dst);
    GGML_UNUSED(thread_row);
    GGML_UNUSED(thread_col);
    NO_DEVICE_CODE;
#endif
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
