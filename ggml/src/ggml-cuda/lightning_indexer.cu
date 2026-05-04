#include "lightning_indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int64_t n_embd, int64_t n_head, ggml_type type_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * src0, const char * src1, const float * src2, float * dst,
        const float scale_embd, const float scale_heads,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb21, size_t nb22, size_t nb23
    ) {

    constexpr int K_VECS_PER_BLOCK = 16;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 16;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int n_embd_padded = n_embd + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 src0 + i_batch*nb02 + i_stream*nb03;
    const float * w_base = (const float *) ((const char *) src2 + i_batch*nb21 + i_stream*nb23);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[n_head];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][n_embd_padded / 4];

    if (tid < n_head) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x n_embd Q tile
    constexpr int n_q_tile = HEADS_PER_INNER_LOOP * (n_embd / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int n_q_next = (n_q_tile + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    #pragma unroll
    for (int i_q = tid; i_q < n_q_tile; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (n_embd / 4);
        const int i_embd = i_q % (n_embd / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nb01 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][n_embd_padded / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (n_embd / 4);

    if constexpr (type_K == GGML_TYPE_F16) {
        #pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (n_embd / 4);
            const int i_embd = i_k % (n_embd / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<type_K, half, 4>();
        #pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (n_embd / 4);
            const int i_embd = i_k % (n_embd / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], n_embd_padded);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < n_head; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], n_embd_padded);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[n_q_next];

        if (i_head_next < n_head) {
            #pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < n_q_tile; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (n_embd / 4);
                const int i_embd =               i_q % (n_embd / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nb01 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < n_head) {
            #pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < n_q_tile; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (n_embd / 4);
                const int i_embd = i_q % (n_embd / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // scale_embd, ReLU, weight
        sum *= scale_embd;
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
            #pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k * scale_heads;
        }
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 or half4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (n_embd)

template <int64_t n_embd, int64_t n_head, ggml_type type_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * src0, const char * src1, const float * src2, float * dst,
        const float scale_embd, const float scale_heads,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb21, size_t nb22, size_t nb23
    ) {

    constexpr int K_VECS_PER_WARP = 8;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * (WARPS_PER_BLOCK * K_VECS_PER_WARP);
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 src0 + i_batch*nb02 + i_stream*nb03;
    const float * w_base = (const float *) ((const char *) src2 + i_batch*nb21 + i_stream*nb23);

    // phase 1 - load weights and Q to shared memory

    __shared__ float  w_shared[n_head];
    // Q are loaded to shared memory either as float4 or half4 (stored as int2) depending on K type
    __shared__ float4 q_shared_f[n_head][n_embd / 4];
    __shared__ int2   q_shared_h[n_head][n_embd / 4];

    if (tid < n_head) {
        w_shared[tid] = w_base[tid];
    }

    constexpr int n_q = n_head * (n_embd / 4);
    #pragma unroll
    for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (n_embd / 4);
        const int i_embd = i_q % (n_embd / 4);
        if constexpr (type_K == GGML_TYPE_F32 || type_K == GGML_TYPE_BF16) {
            q_shared_f[i_head][i_embd] = *(const float4 *) (q_base + i_head*nb01 + i_embd*sizeof(float4));
        } else {
            const float4 q = *(const float4 *) (q_base + i_head*nb01 + i_embd*sizeof(float4));
            half4 q_packed;
            q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
            q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
            q_shared_h[i_head][i_embd] = q_packed.i2;
        }
    }

    __syncthreads();

    // phase 2 - load (and dequantize if needed) K to registers

    // K are loaded to registers either as float4 or half2 depending on K type
    float4 k_reg_f[K_VECS_PER_WARP];
    half2  k_reg_h[K_VECS_PER_WARP][2];

    if constexpr (type_K == GGML_TYPE_F32) {
        // direct copy of float4
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        // dequantize bf16 to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<type_K, float, 4>();
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else if constexpr (type_K == GGML_TYPE_F16) {
        // direct copy of halfs - separate case for some extra performance
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                *(int2*) &k_reg_h[k] = k_base[i_lane];
            } else {
                *(int2*) &k_reg_h[k] = make_int2(0, 0);
            }
        }
    } else {
        // dequantize remaining types to half
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<type_K, half, 4>();
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                dequantize_k(k_base, &k_reg_h[k][0], i_lane * 4);
            } else {
                *(int2*) &k_reg_h[k] = make_int2(0, 0);
            }
        }
    }

    // phase 3 - calculate lightning indexer scores

    float score_k[K_VECS_PER_WARP] = { 0.0f };
    for (int i_head = 0; i_head < n_head; ++i_head) {
        const float w_val = w_shared[i_head];
        float qk[K_VECS_PER_WARP] = { 0.0f };

        if constexpr (type_K == GGML_TYPE_F32 || type_K == GGML_TYPE_BF16) {
            // dot product of floats for f32 and bf16
            const float4 q_vec = q_shared_f[i_head][i_lane];

            #pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }
        } else {
            // dot product of halfs for remaining types
            half4 q_vec;
            q_vec.i2 = q_shared_h[i_head][i_lane];
            const half2 q_h0 = q_vec.h2[0];
            const half2 q_h1 = q_vec.h2[1];

            #pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_h0.x, k_reg_h[k][0].x);
                ggml_cuda_mad(qk[k], q_h0.y, k_reg_h[k][0].y);
                ggml_cuda_mad(qk[k], q_h1.x, k_reg_h[k][1].x);
                ggml_cuda_mad(qk[k], q_h1.y, k_reg_h[k][1].y);
            }
        }

        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            float sum = warp_reduce_sum(qk[k]);

            // scale_embd, ReLU, weight
            if (i_lane == 0) {
                sum *= scale_embd;
                sum = (sum > 0.0f) ? sum : 0.0f;
                score_k[k] += sum * w_val;
            }
        }
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[WARPS_PER_BLOCK * K_VECS_PER_WARP];

    if (i_lane == 0) {
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k] * scale_heads;
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < WARPS_PER_BLOCK * K_VECS_PER_WARP) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid];
        }
    }
}

#define DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, type_K) \
    template __global__ void lightning_indexer_kernel<n_embd, n_head, type_K>(        \
        const float * src0, const char * src1, const float * src2, float * dst,       \
        const float scale_embd, const float scale_heads,                              \
        int64_t n_stream, int64_t n_batch, int64_t n_kv,                              \
        size_t nb1, size_t nb2, size_t nb3,                                           \
        size_t nb01, size_t nb02, size_t nb03,                                        \
        size_t nb11, size_t nb12, size_t nb13,                                        \
        size_t nb21, size_t nb22, size_t nb23);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, GGML_TYPE_F16)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, GGML_TYPE_Q4_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, GGML_TYPE_Q4_1)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, GGML_TYPE_Q5_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, GGML_TYPE_Q5_1)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, GGML_TYPE_Q8_0)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_F16)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q4_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q4_1)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q5_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q5_1)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q8_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_BF16)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_F32)

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<n_embd, n_head, type_K><<<grid, block, 0, ctx.stream()>>>( \
            src0_d, src1_d, src2_d, dst_d, scale_embd, scale_heads,                         \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nb01, nb02, nb03,                                                               \
            nb11, nb12, nb13,                                                               \
            nb21, nb22, nb23                                                                \
        );                                                                                  \
    } else

void ggml_cuda_op_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float scale_embd = ggml_get_op_params_f32(dst, 0);
    const float scale_heads = ggml_get_op_params_f32(dst, 1);

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);

    GGML_TENSOR_TERNARY_OP_LOCALS

    // input tensor rows must be contiguous
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb20 == ggml_type_size(src2->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = src0->ne[0];
    const int n_head   = src0->ne[1];
    const int n_batch  = src0->ne[2];
    const int n_stream = src0->ne[3];
    const int n_kv     = src1->ne[2];

    const float * src0_d = (const float *) src0->data;
    const char  * src1_d = (const char  *) src1->data;
    const float * src2_d = (const float *) src2->data;
    float       * dst_d  = (float *)       dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && ampere_mma_available(cc) && src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 16;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else {
        GGML_ABORT("fatal error");
    }
}
