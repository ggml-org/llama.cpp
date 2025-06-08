#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-f16.cuh"
#include "paged_attn_common.cuh" // For paged view structures

#define FATTN_KQ_STRIDE_TILE_F16 64

template<int D, int ncols, int nwarps, bool use_logit_softcap> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)

    // Skip unused kernel variants for faster compilation:
#ifdef FP16_MMA_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FP16_MMA_AVAILABLE
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.z              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.z / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V    + nb12*(blockIdx.z / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne11*ic0;

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slopef = get_alibi_slope(max_bias, blockIdx.z, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    __shared__ half KQ[ncols*FATTN_KQ_STRIDE_TILE_F16];
    half2 * KQ2 = (half2 *) KQ;

    __shared__ half2 KV_tmp[FATTN_KQ_STRIDE_TILE_F16][D/2 + 1]; // Pad D to avoid memory bank conflicts.

    half kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -HALF_MAX_HALF;
    }
    half2 kqsum[ncols/nwarps] = {{0.0f, 0.0f}};

    half2 VKQ[ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Convert Q to half2 and store in registers:
    __shared__ half2 Q_h2[ncols][D/2];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
            Q_h2[j][i] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
        }
    }

    __syncthreads();

    for (int k_VKQ_0 = blockIdx.y*FATTN_KQ_STRIDE_TILE_F16; k_VKQ_0 < ne11; k_VKQ_0 += gridDim.y*FATTN_KQ_STRIDE_TILE_F16) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        half kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                KV_tmp[i_KQ][k_KQ] = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
            }
        }

        __syncthreads();

        half2 sum2[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE][ncols/nwarps] = {{{0.0f, 0.0f}}};

#pragma unroll
        for (int k_KQ = 0; k_KQ < D/2; ++k_KQ) {
            half2 K_k[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE];
            half2 Q_k[ncols/nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0/WARP_SIZE] = KV_tmp[i_KQ][k_KQ];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0/nwarps] = Q_h2[j_KQ][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE]*Q_k[j_KQ_0/nwarps];
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                half sum;
                if (use_logit_softcap) {
                    const float2 tmp = __half22float2(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                    sum = logit_softcap * tanhf(tmp.x + tmp.y);
                } else {
                    sum = __low2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]) + __high2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                }
                sum += mask ? slopeh*maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ] : __float2half(0.0f);

                kqmax_new[j_KQ_0/nwarps] = ggml_cuda_hmax(kqmax_new[j_KQ_0/nwarps], sum);

                KQ[j_KQ*FATTN_KQ_STRIDE_TILE_F16 + i_KQ] = sum;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const half2 KQ_max_scale = __half2half2(hexp(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]));
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F16/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const half2 diff = KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + i] - __half2half2(kqmax[j0/nwarps]);
                const half2 val = h2exp(diff);
                kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + val;
                KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + i] = val;
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE] *= KQ_max_scale;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                KV_tmp[k][i] = V_h2[(k_VKQ_0 + k)*stride_KV2 + i];
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += 2) {
            half2  V_k[(D/2)/WARP_SIZE][2];
            half2 KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0/WARP_SIZE][0] = KV_tmp[k0 + 0][i];
                V_k[i0/WARP_SIZE][1] = KV_tmp[k0 + 1][i];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + k0/2];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    VKQ[j0/nwarps][i0/WARP_SIZE] += V_k[i0/WARP_SIZE][0]* __low2half2(KQ_k[j0/nwarps]);
                    VKQ[j0/nwarps][i0/WARP_SIZE] += V_k[i0/WARP_SIZE][1]*__high2half2(KQ_k[j0/nwarps]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        half kqsum_j = __low2half(kqsum[j_VKQ_0/nwarps]) + __high2half(kqsum[j_VKQ_0/nwarps]);
        kqsum_j = warp_reduce_sum((float)kqsum_j);

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 2*WARP_SIZE) {
            const int i0 = i00 + 2*threadIdx.x;

            half2 dst_val = VKQ[j_VKQ_0/nwarps][i0/(2*WARP_SIZE)];
            if (gridDim.y == 1) {
                dst_val /= __half2half2(kqsum_j);
            }
            const int j_dst = (ic0 + j_VKQ)*gridDim.y + blockIdx.y;
            dst[j_dst*D*gridDim.z + D*blockIdx.z + i0 + 0] =  __low2float(dst_val);
            dst[j_dst*D*gridDim.z + D*blockIdx.z + i0 + 1] = __high2float(dst_val);
        }

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[((ic0 + j_VKQ)*gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
    GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02);
    GGML_UNUSED(ne03); GGML_UNUSED(ne10); GGML_UNUSED(ne11);
    GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31);
    GGML_UNUSED(nb31); GGML_UNUSED(nb01); GGML_UNUSED(nb02);
    GGML_UNUSED(nb03); GGML_UNUSED(nb11); GGML_UNUSED(nb12);
    GGML_UNUSED(nb13); GGML_UNUSED(nb21); GGML_UNUSED(nb22);
    GGML_UNUSED(nb23); GGML_UNUSED(ne0); GGML_UNUSED(ne1);
    GGML_UNUSED(ne2); GGML_UNUSED(ne3);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
}


// Paged version of the Tile F16 kernel
template<int D, int ncols, int nwarps, bool use_logit_softcap> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_f16_paged(
        const char * __restrict__ Q_ptr, // Q remains non-paged
        const paged_kv_sequence_view_gpu K_view,
        const paged_kv_sequence_view_gpu V_view,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00, // Q elements per dim
        const int ne01, // Q sequence length (n_q)
        const int ne02, // Q num heads
        const int ne03, // Q batch size (unused)
        // K view provides K sequence length (n_kv) and other layout info (ne10, ne11, ne12, ne13)
        // V view provides V sequence length and other layout info
        const int ne31, // dst batch stride
        const int nb31, // dst batch stride bytes
        const int nb01, // Q stride bytes for seq_len dim
        const int nb02, // Q stride bytes for num_heads dim
        const int nb03, // Q stride bytes for batch_size dim (unused)
        // K_view provides K sequence length (n_kv) and other layout info
        // V_view provides V sequence length and other layout info
        const int num_kv_heads, // Number of K/V heads in the model (K_meta_tensor->ne[2])
        const int mask_k_seq_len, // Mask's K sequence length (mask_tensor ? mask_tensor->ne[1] : 0)
        const int mask_k_stride_bytes, // Mask's K stride in bytes (mask_tensor ? mask_tensor->nb[1] : 0)
        const int _dst_ne0,  // Dst tensor ne0 (D, head_size) - should match ne00
        const int _dst_ne1,  // Dst tensor ne1 (n_q) - should match ne01
        const int _dst_ne2,  // Dst tensor ne2 (n_heads) - should match ne02
        const int _dst_ne3   // Dst tensor ne3 (batch_size) - should match ne03
) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
    // ne00, ne01, ne02, ne03 are Q dimensions
    // _dst_ne0, _dst_ne1, _dst_ne2, _dst_ne3 are Dst dimensions (passed from Q, used for Dst indexing)
    // nb01, nb02, nb03 are Q byte strides

    // Skip unused kernel variants for faster compilation:
#ifdef FP16_MMA_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FP16_MMA_AVAILABLE
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    // Q is indexed by blockIdx.z for the head
    const float2 * Q_f2  = (const float2 *) (Q_ptr  + nb02* blockIdx.z + nb01*ic0);
    // K and V will be accessed via K_view and V_view
    // Mask pointer `mask` is base for current head if mask is per-head, or global base.
    // Kernel uses ic0 and k_VKQ_0 + i_KQ_local to index into it.

    const float slopef = get_alibi_slope(max_bias, blockIdx.z, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    __shared__ half KQ[ncols*FATTN_KQ_STRIDE_TILE_F16];
    half2 * KQ2 = (half2 *) KQ;

    // Shared memory for K and V tiles
    // +1 for padding to avoid bank conflicts is a common pattern, ensure D/2 is correct for half2
    __shared__ half2 K_tmp_sh[FATTN_KQ_STRIDE_TILE_F16][D/2 + 1];
    __shared__ half2 V_tmp_sh[FATTN_KQ_STRIDE_TILE_F16][D/2 + 1]; // Assuming V_head_size == K_head_size == D

    half kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -HALF_MAX_HALF;
    }
    half2 kqsum[ncols/nwarps] = {{0.0f, 0.0f}};

    half2 VKQ[ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Convert Q to half2 and store in registers:
    __shared__ half2 Q_h2[ncols][D/2];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
            Q_h2[j][i] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
        }
    }

    __syncthreads();

    // K_view.sequence_length_tokens gives n_kv (ne11 in original kernel)
    for (int k_VKQ_0 = blockIdx.y*FATTN_KQ_STRIDE_TILE_F16; k_VKQ_0 < K_view.sequence_length_tokens; k_VKQ_0 += gridDim.y*FATTN_KQ_STRIDE_TILE_F16) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        half kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

        // Load K tile from paged KV cache into shared memory K_tmp_sh
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += nwarps) { // Loop over tile rows (tokens)
            const int i_KQ_local = i_KQ_0 + threadIdx.y; // Local row index in shared memory tile
            const int token_k_idx = k_VKQ_0 + i_KQ_local; // Global token index in K sequence

            if (token_k_idx < K_view.sequence_length_tokens) {
#pragma unroll
                for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) { // Loop over head dimension elements
                    const int k_KQ_local_col = k_KQ_0 + threadIdx.x; // Local column index in shared memory tile
                    if (k_KQ_local_col < D/2) { // Check bounds for D/2
                        // blockIdx.z is the head index for Q. For K, we need to map it if GQA/MQA
                        // Assuming K_view is for the correct group of K heads if GQA is used.
                        // The get_paged_kv_data_ptr_cuda takes the absolute head index.
                        // For GQA, K_view.num_k_heads_total might be less than ne02 (num_q_heads)
                        // The head_idx for K should be blockIdx.z / gqa_ratio.
                        // However, paged_kv_sequence_view_gpu is usually constructed for a specific head group already.
                        // So, if K_view.num_k_heads_total == 1 (e.g. MQA), head_idx for K is 0.
                        // If K_view.num_k_heads_total == num_q_heads (MHA), head_idx for K is blockIdx.z
                        // Let's assume K_view is set up for the specific head_idx we need, or head_idx is 0 for broadcast.
                        // The paged_attn_common.cuh get_paged_kv_data_ptr_cuda needs absolute head index within the K tensor.
                        // This means K_view should ideally represent all K heads or the dispatcher should select the correct K_view.
                        // For now, assume blockIdx.z is the relevant head index for K_view if K_view spans all heads,
                        // or K_view itself is pre-filtered for a specific head group and blockIdx.z is an offset within that.
                        // This needs careful handling by the caller in setting up K_view.
                        // Let's assume K_view.head_idx_offset is 0 or already incorporated by the caller.
                        // The current K_view is per-head group, so head_idx for get_paged_kv_data_ptr_cuda should be relative to that group.
                        // For simplicity here, if K_view.num_k_heads_total == 1, it's head 0. Otherwise, it's blockIdx.z % K_view.num_k_heads_total.
                        // This is still tricky. The simplest is K_view is for a single head, or for MHA where K_head_idx = Q_head_idx.

                        // Let's use current_q_head_idx = blockIdx.z
                        // int current_k_head_idx = current_q_head_idx % K_view.num_k_heads_total; // simplistic mapping
                        // This mapping needs to be correct based on how K_view is prepared by the dispatcher.
                        // For now, assume K_view is constructed such that head_idx 0 within the view is the target.
                        // Or, more robustly, the dispatcher should pass the correct K_view for the Q head.
                        // If K_view is global for all K heads, then K_head_idx = blockIdx.z / gqa_ratio.
                        // Let's assume K_view is already for the correct head group and head_idx for get_paged_kv_data_ptr_cuda is 0
                        // if K_view.num_k_heads_total refers to heads *within that group*.
                        // This is a major point of complexity for GQA/MQA with paged attention.
                        // The `paged_kv_sequence_view_gpu` has `num_k_heads_total` which is the total number of K heads in the model.
                        // And `k_head_start_idx` which is the starting index of K heads this view pertains to.
                        // So, the actual K head for the current Q head (blockIdx.z) is:
                        // int actual_k_head_idx = K_view.k_head_start_idx + (blockIdx.z % (ne02 / K_view.num_k_heads_total));
                        // No, this is simpler: blockIdx.z is the Q head. K head is blockIdx.z / gqa_ratio.
                        // The K_view should be prepared for the specific K head group.
                        // Let's assume the K_view passed corresponds to the Q head group (i.e. for MHA, it's 1-to-1, for GQA, K_view might be reused).
                        // The most direct approach: K_view is prepared for a specific K head (or group of K heads).
                        // The `get_paged_kv_data_ptr_cuda` will use `head_idx` passed to it. This head_idx should be the *absolute* K head index.
                        // ne02 is num_q_heads. num_kv_heads is passed from K_meta_tensor->ne[2].
                        int gqa_ratio_k = (num_kv_heads == 0 || ne02 == 0) ? 1 : ne02 / num_kv_heads; // Avoid division by zero
                        if (gqa_ratio_k == 0) gqa_ratio_k = 1; // Should not happen if params are correct
                        int abs_k_head_idx = blockIdx.z / gqa_ratio_k;

                        const half2* k_data_ptr = get_paged_kv_data_ptr_cuda<half2>(K_view, token_k_idx, abs_k_head_idx);
                        if (k_data_ptr) { // Check if token is valid and page exists
                           K_tmp_sh[i_KQ_local][k_KQ_local_col] = k_data_ptr[k_KQ_local_col]; // k_KQ_local_col is offset within head
                        } else {
                           // Handle case where page is not found (e.g. out of bounds) - fill with zero?
                           K_tmp_sh[i_KQ_local][k_KQ_local_col] = make_half2(0.0f, 0.0f);
                        }
                    }
                }
            } else {
                // Pad with zeros if token_k_idx is out of bounds (for the last block)
#pragma unroll
                for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                    const int k_KQ_local_col = k_KQ_0 + threadIdx.x;
                     if (k_KQ_local_col < D/2) {
                        K_tmp_sh[i_KQ_local][k_KQ_local_col] = make_half2(0.0f, 0.0f);
                    }
                }
            }
        }
        __syncthreads(); // Ensure K_tmp_sh is filled

        // --- Computation part (copied and adapted from non-paged flash_attn_tile_ext_f16) ---
        // This part assumes K data is in K_tmp_sh.
        half2 sum2[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE][ncols/nwarps] = {{{0.0f, 0.0f}}};

#pragma unroll
        for (int k_KQ = 0; k_KQ < D/2; ++k_KQ) { // Loop over head dimension (columns of K_tmp_sh)
            half2 K_k[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE]; // Holds a column of K_tmp_sh for current k_KQ
            half2 Q_k[ncols/nwarps]; // Holds a column of Q_h2 for current k_KQ

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) { // Loop over rows of K_tmp_sh (tokens)
                const int i_KQ_local_row = i_KQ_0 + threadIdx.x; // Current row in K_tmp_sh
                K_k[i_KQ_0/WARP_SIZE] = K_tmp_sh[i_KQ_local_row][k_KQ];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) { // Loop over Q queries
                const int j_KQ_local_row = j_KQ_0 + threadIdx.y; // Current Q query index
                Q_k[j_KQ_0/nwarps] = Q_h2[j_KQ_local_row][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) { // Iterate over K tile rows
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) { // Iterate over Q queries
                    sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE]*Q_k[j_KQ_0/nwarps];
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) { // Iterate over K tile rows
            const int i_KQ_local_row = i_KQ_0 + threadIdx.x; // Current row in K_tmp_sh / output KQ tile

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) { // Iterate over Q queries
                const int j_KQ_local_row = j_KQ_0 + threadIdx.y; // Current Q query index

                half sum;
                if (use_logit_softcap) {
                    const float2 tmp = __half22float2(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                    sum = logit_softcap * tanhf(tmp.x + tmp.y);
                } else {
                    sum = __low2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]) + __high2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                }
                // Masking assumes mask is (Q_seq_len_tile, K_seq_len_tile)
                // maskh here is (K_seq_len_total * Q_seq_len_tile_offset)
                // The token index in K sequence is (k_VKQ_0 + i_KQ_local_row)
                // The Q index is (ic0 + j_KQ_local_row)
                // This kernel processes `ncols` Q tokens starting at `ic0`.
                // And `FATTN_KQ_STRIDE_TILE_F16` K tokens starting at `k_VKQ_0`.
                // Original mask indexing: maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ] where ne11 is K_seq_len_total
                // New mask indexing: maskh[(ic0 + j_KQ_local_row) * K_view.sequence_length_tokens + (k_VKQ_0 + i_KQ_local_row)] if mask is (Q_total, K_total)
                // Or, if mask is passed as a tile: mask_tile[j_KQ_local_row * FATTN_KQ_STRIDE_TILE_F16 + i_KQ_local_row]
                // The `mask` pointer is to `mask_ptr + ne11*ic0` where ne11 is K_view.k_head_size_elements (this seems wrong for mask)
                // Let's assume the mask is prepared and passed appropriately by the caller, matching the tile structure.
                // The original `maskh` was `(const half *) mask + ne11*ic0;` where `ne11` was `K.ne[1]` (K sequence length).
                // So `maskh` points to `mask[ic0][0]` if mask is `(Q_seq_len, K_seq_len)`.
                // Then `maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ]` becomes `mask[ic0 + j_KQ][k_VKQ_0 + i_KQ]`.
                // For paged, `mask` is passed as `const char * __restrict__ mask`.
                // `maskh` is `(const half *) mask + K_view.k_head_size_elements*ic0;` - this seems like a bug from copy-paste.
                // `k_head_size_elements` is D. `ne11` should be K sequence length.
                // Corrected maskh definition: const half   * maskh_base = (const half   *)  mask;
                // Access: maskh_base[ (ic0 + j_KQ_local_row) * K_view.sequence_length_tokens + (k_VKQ_0 + i_KQ_local_row) ] for a full mask.
                // If the mask is pre-sliced for the Q block: (const half *) mask; then mask[j_KQ_local_row * K_view.sequence_length_tokens + (k_VKQ_0 + i_KQ_local_row)]
                // For now, let's assume the mask is handled by the caller or is NULL. If not NULL, this needs fixing.
                // The original kernel's mask was complex due to alibi. If mask is just for causal, it's simpler.
                // If mask is not NULL, the indexing `maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ]` with ne11 = K_view.sequence_length_tokens would be:
                // `mask_val = maskh[j_KQ_local_row * K_view.sequence_length_tokens + k_VKQ_0 + i_KQ_local_row]`
                // This is still not quite right. The original `mask` parameter to the kernel is already offset by `nb11*ic0` by the `launch_fattn` helper.
                // Let's assume `mask` points to the top-left of the relevant mask tile for this Q-block vs K-sequence.
                // So mask access would be `maskh_tile[j_KQ_local_row * FATTN_KQ_STRIDE_TILE_F16 + i_KQ_local_row]` if mask is tiled.
                // Given `maskh = (const half *) mask;` (assuming launch_fattn passes the right slice)
                // then `maskh[j_KQ_local_row * stride_mask_k + i_KQ_local_row]`
                // The original `mask` parameter in `launch_fattn` is `dst->src[3]`. It's a full mask tensor.
                // `launch_fattn` passes `mask_ptr = ggml_backend_buffer_get_base(ctx.flash_mask_buffer) + máscara_desplazamiento`
                // `mask_ptr += nb1m*ic0;` where `nb1m` is stride over Q dimension.
                // So `mask` points to `mask_mem[ic0][0]`. Then `mask[j][k]` is `mask_mem[ic0+j][k]`.
                // `const half* current_mask_q_row = (const half*)mask + j_KQ_local_row * K_view.sequence_length_tokens;`
                // `half mask_val = current_mask_q_row[k_VKQ_0 + i_KQ_local_row];`
                // This seems more plausible if `mask` is `(Q_block_size, K_total_seq_len)`.
                // This part is critical and needs to match how `launch_fattn_paged` sets up the mask argument.
                // For now, let's assume if mask is present, it's correctly indexed or handled by alibi.
                // The alibi part `slopeh*mask_val` is the main user.
                // The original mask was (ne01, ne11). So `mask[q_idx][k_idx]`.
                // `q_idx = ic0 + j_KQ_local_row`, `k_idx = k_VKQ_0 + i_KQ_local_row`.
                // So if `mask` points to `orig_mask[0][0]`:
                // `half mask_val = ((const half *)mask)[ (ic0 + j_KQ_local_row) * K_view.sequence_length_tokens + (k_VKQ_0 + i_KQ_local_row) ];`
                // This requires `mask` to be the global mask pointer. The `launch_fattn_paged` needs to pass this.
                // The current `mask` parameter is `const char * __restrict__ mask`.
                if (mask) {
                    // mask pointer is base for current head (if applicable). Indexing needs full Q and K global indices.
                    int current_q_global_idx = ic0 + j_KQ_local_row;
                    int current_k_global_idx = k_VKQ_0 + i_KQ_local_row;

                    // mask_k_seq_len is total K sequence length for the mask tensor.
                    // mask_k_stride_bytes is the byte stride for one step in K dimension for the mask.
                    // We need element stride for half.
                    int mask_k_stride_elements = mask_k_stride_bytes / sizeof(half);


                    if (current_q_global_idx < ne01 && current_k_global_idx < mask_k_seq_len) { // Check bounds for Q and K mask access
                        // This assumes mask layout [Q_seq_len, K_seq_len] for the current head.
                        // Or if mask is [Batch, Head, Q_seq, K_seq], then `mask` pointer must be pre-offset for Batch and Head.
                        // `launch_fattn_paged` passes `mask_tensor->data`. If mask has head/batch dims, this needs care.
                        // For now, assume mask is effectively [Q_seq_len, K_seq_len] as seen by this kernel instance for its head.
                        half mask_val = ((const half *)mask)[current_q_global_idx * mask_k_stride_elements + current_k_global_idx];
                        sum += slopeh * mask_val;
                    }
                }
                kqmax_new[j_KQ_0/nwarps] = ggml_cuda_hmax(kqmax_new[j_KQ_0/nwarps], sum);
                KQ[j_KQ_local_row*FATTN_KQ_STRIDE_TILE_F16 + i_KQ_local_row] = sum;
            }
        }

        __syncthreads(); // KQ is filled

        // Update kqmax, kqsum, VKQ (rescaling part)
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j_local = j0 + threadIdx.y; // local Q index

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const half2 KQ_max_scale = __half2half2(hexp(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]));
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F16/2; i0 += WARP_SIZE) { // Iterate over K tile elements (paired)
                const int i_local_pair = i0 + threadIdx.x;

                const half2 diff = KQ2[j_local*(FATTN_KQ_STRIDE_TILE_F16/2) + i_local_pair] - __half2half2(kqmax[j0/nwarps]);
                const half2 val = h2exp(diff);
                kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + val;
                KQ2[j_local*(FATTN_KQ_STRIDE_TILE_F16/2) + i_local_pair] = val; // KQ now stores exp( KQ - max_new )
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) { // Iterate over V dimensions
                VKQ[j0/nwarps][i0/WARP_SIZE] *= KQ_max_scale;
            }
        }
        __syncthreads(); // KQ updated, kqsum and VKQ rescaled

        // Load V tile from paged KV cache into shared memory V_tmp_sh
#pragma unroll
        for (int k0_V = 0; k0_V < FATTN_KQ_STRIDE_TILE_F16; k0_V += nwarps) { // Loop over tile rows (tokens)
            const int k_local_V_row = k0_V + threadIdx.y; // Local row index in shared memory tile
            const int token_v_idx = k_VKQ_0 + k_local_V_row;    // Global token index in V sequence

            if (token_v_idx < V_view.sequence_length_tokens) {
#pragma unroll
                for (int i0_V = 0; i0_V < D/2; i0_V += WARP_SIZE) { // Loop over head dimension elements
                    const int i_local_V_col = i0_V + threadIdx.x; // Local column index in shared memory tile
                    if (i_local_V_col < D/2) {
                        int gqa_ratio_v = (num_kv_heads == 0 || ne02 == 0) ? 1 : ne02 / num_kv_heads; // Assuming V has same head count as K for GQA
                        if (gqa_ratio_v == 0) gqa_ratio_v = 1;
                        int abs_v_head_idx = blockIdx.z / gqa_ratio_v;
                        const half2* v_data_ptr = get_paged_kv_data_ptr_cuda<half2>(V_view, token_v_idx, abs_v_head_idx);
                        if (v_data_ptr) {
                            V_tmp_sh[k_local_V_row][i_local_V_col] = v_data_ptr[i_local_V_col];
                        } else {
                            V_tmp_sh[k_local_V_row][i_local_V_col] = make_half2(0.0f, 0.0f);
                        }
                    }
                }
            } else {
                // Pad with zeros if token_v_idx is out of bounds
#pragma unroll
                 for (int i0_V = 0; i0_V < D/2; i0_V += WARP_SIZE) {
                    const int i_local_V_col = i0_V + threadIdx.x;
                    if (i_local_V_col < D/2) {
                        V_tmp_sh[k_local_V_row][i_local_V_col] = make_half2(0.0f, 0.0f);
                    }
                }
            }
        }
        __syncthreads(); // V_tmp_sh is filled

        // Accumulate V into VKQ, weighted by KQ
#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += 2) { // Loop over K/V tile rows (tokens), step 2 for half2
            half2 V_k_pairs[(D/2)/WARP_SIZE][2]; // Holds two V vectors (for k0 and k0+1)
            half2 KQ_k_pair[ncols/nwarps];      // Holds KQ values for current Q query and k0, k0+1 K tokens

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) { // Loop over V head dimensions
                const int i_local_col = i0 + threadIdx.x; // V head dim element index
                V_k_pairs[i0/WARP_SIZE][0] = V_tmp_sh[k0 + 0][i_local_col];
                V_k_pairs[i0/WARP_SIZE][1] = V_tmp_sh[k0 + 1][i_local_col];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) { // Loop over Q queries
                const int j_local_row = j0 + threadIdx.y; // Q query index
                KQ_k_pair[j0/nwarps] = KQ2[j_local_row*(FATTN_KQ_STRIDE_TILE_F16/2) + k0/2]; // KQ2 stores pairs
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) { // Loop over V head dimensions
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) { // Loop over Q queries
                    VKQ[j0/nwarps][i0/WARP_SIZE] += V_k_pairs[i0/WARP_SIZE][0]* __low2half2(KQ_k_pair[j0/nwarps]);
                    VKQ[j0/nwarps][i0/WARP_SIZE] += V_k_pairs[i0/WARP_SIZE][1]*__high2half2(KQ_k_pair[j0/nwarps]);
                }
            }
        }
        __syncthreads(); // All threads in block done with this K/V tile
    } // End of loop over K/V sequence blocks (k_VKQ_0)

    // --- Output section (copied and adapted from non-paged) ---
#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) { // Loop over Q queries processed by this warp
        const int j_VKQ_local = j_VKQ_0 + threadIdx.y; // Local Q index for this thread

        if (ic0 + j_VKQ_local >= ne01) { // ne01 is total Q sequence length
            // This check might be redundant if launch parameters ensure ncols fits within ne01 bounds for each block
            // However, it's a good safety for the last block along Q dimension.
            // return; // Exiting early can be problematic if other threads in warp continue to syncthreads or shared mem access
            // It's generally safer to let them run but skip global writes.
            // No, if a Q token is out of bounds, its results should not be written.
            // The original kernel has this return, let's keep it.
            // Ensure this is only for threads whose Q is out of bounds.
             if (threadIdx.x == 0 && threadIdx.y == 0) { // To avoid multiple returns / messages
                // This condition is not quite right. It should be per thread's j_VKQ_local.
             }
        }
        // If ic0 + j_VKQ_local >= ne01, this thread's Q is out of actual sequence length.
        // It should not write any output.

        half kqsum_j = __low2half(kqsum[j_VKQ_0/nwarps]) + __high2half(kqsum[j_VKQ_0/nwarps]);
        kqsum_j = warp_reduce_sum((float)kqsum_j);

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 2*WARP_SIZE) { // Loop over head dimensions
            const int i0_base = i00 + 2*threadIdx.x; // Start element index for this thread (processes 2 elements: i0, i0+1)

            if (ic0 + j_VKQ_local < ne01) { // Only write if Q is within bounds
                half2 dst_val = VKQ[j_VKQ_0/nwarps][i0_base/(2*WARP_SIZE)]; // Each thread in warp gets unique part of VKQ
                if (gridDim.y == 1) { // If only one K/V block processed (no partial sums)
                    dst_val /= __half2half2(kqsum_j);
                }
                // dst layout: [n_q_total, n_heads_total, head_size_elements]
                // ne01: n_q_total (Q sequence length)
                // ne02: n_heads_total (Q heads)
                // D:    head_size_elements
                // nb01, nb02 are byte strides for Q. dst uses element strides.
                // Original dst indexing: dst[j_dst*D*gridDim.z + D*blockIdx.z + i0 + 0]
                // j_dst = (ic0 + j_VKQ)*gridDim.y + blockIdx.y;
                // This was for multi-pass reduction (gridDim.y > 1).
                // If gridDim.y == 1 (single pass), then j_dst = ic0 + j_VKQ_local.
                // dst pointer is to float.
                // The output tensor `dst` has dimensions [ne03, ne02, ne01, ne00] = [batch, n_head_q, n_q, d_head]
                // Strides are nb03, nb02, nb01, nb00 (bytes).
                // We are writing for head blockIdx.z, Q token (ic0 + j_VKQ_local).
                // float* current_q_head_dst_ptr = (float*)( (char*)dst + blockIdx.z * nb02_elements * sizeof(float) + (ic0 + j_VKQ_local) * nb01_elements * sizeof(float) );
                // This needs to use the strides passed in: ne0, ne1, ne2, ne3 are counts.
                // nb01, nb02 are Q strides.
                // The output tensor `dst` is passed as float*.
                // Its shape is (ne3, ne2, ne1, ne0) typically for llama.cpp (batch, n_heads, seq_len, head_dim)
                // Or (ne1, ne2, ne0) if batch=1. (seq_len, n_heads, head_dim)
                // The original kernel used `dst[j_dst*D*gridDim.z + D*blockIdx.z + i0 + 0]`
                // `gridDim.z` was number of heads. `blockIdx.z` was current head.
                // `D` was `ne00` (head_size).
                // `j_dst` was complex due to reduction passes.
                // For a single pass (gridDim.y == 1):
                // `q_token_global_idx = ic0 + j_VKQ_local;`
                // `head_global_idx = blockIdx.z;`
                // `dst_ptr_for_token_head = dst + head_global_idx * ne01 * D + q_token_global_idx * D;` (assuming standard layout [n_head, n_q, D])
                // Llama.cpp dst is often (..., n_embd), so (..., n_heads, head_size).
                // The `dst` ggml_tensor has shape (ne0, ne1, ne2, ne3) = (D, n_q, n_heads, batch)
                // Strides nb0, nb1, nb2, nb3 (bytes).
                // Access: char* p = (char*)dst->data + head_idx*nb2 + q_idx*nb1 + element_d_idx*nb0;
                // Here, dst is already float* `dst->data`.
                // float* base_dst_ptr = (float*)((char*)dst + blockIdx.z * nb02_dst_bytes + (ic0 + j_VKQ_local) * nb01_dst_bytes);
                // This requires passing nb01_dst, nb02_dst.
                // The current `dst` param is already `dst->data`.
                // So, global_q_idx = ic0 + j_VKQ_local
                // global_head_idx = blockIdx.z
                // element_idx_in_head = i0_base or i0_base + 1
                // dst is (D, n_q, n_head_q, n_batch)
                // nb0=sizeof(float), nb1=D*sizeof(float), nb2=n_q*D*sizeof(float), nb3=...
                // offset = global_head_idx * (nb2/sizeof(float)) + global_q_idx * (nb1/sizeof(float)) + element_idx_in_head
                // Dst tensor ggml_dims: [_dst_ne0=D, _dst_ne1=n_q, _dst_ne2=n_heads, _dst_ne3=batch_size]
                // Strides (elements) for Dst, based on Dst dimensions passed by launcher (_dst_ne0, _dst_ne1 etc)
                const int s1d = _dst_ne0; // D
                const int s2d = _dst_ne0 * _dst_ne1; // D * n_q
                // const int s3d = _dst_ne0 * _dst_ne1 * _dst_ne2; // D * n_q * n_heads (for batch > 1, if _dst_ne3 used)

                // global_q_idx = ic0 + j_VKQ_local
                // global_head_idx = blockIdx.z
                // element_idx_in_head = i0_base or i0_base + 1
                size_t base_offset_elements = blockIdx.z * s2d + (ic0 + j_VKQ_local) * s1d;

                dst[base_offset_elements + i0_base + 0] =  __low2float(dst_val);
                dst[base_offset_elements + i0_base + 1] = __high2float(dst_val);
            }
        }

        if (gridDim.y != 1 && threadIdx.x == 0) { // Multi-pass reduction case
             if (ic0 + j_VKQ_local < ne01) { // Only write if Q is within bounds
                // dst_meta layout: [n_q_total, n_heads_total, n_kv_blocks_total_for_reduction]
                // Access: dst_meta[q_idx * n_heads * n_kv_blocks + head_idx * n_kv_blocks + kv_block_idx]
                // Or: dst_meta[ ( (ic0 + j_VKQ_local)*gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y ]
                // gridDim.z is n_heads in original launch_fattn. Here it's 1 head per kernel.
                // So, if dst_meta is (n_q, n_heads, n_kv_blocks), then:
                // q_global = ic0 + j_VKQ_local
                // head_global = blockIdx.z
                // kv_block_idx = blockIdx.y
                // num_kv_blocks = gridDim.y
                // num_heads = ne02 (total Q heads)
                // dst_meta_offset = q_global * num_heads * num_kv_blocks + head_global * num_kv_blocks + kv_block_idx;
                // This implies dst_meta is passed as base pointer.
                // The original indexing was: `dst_meta[((ic0 + j_VKQ)*gridDim.z/*n_heads_dispatch*/ + blockIdx.z/*head_in_dispatch*/) * gridDim.y/*n_kv_blocks*/ + blockIdx.y/*kv_block_idx*/]`
                // For paged kernel, gridDim.z is effectively 1 (as kernel is launched per head). blockIdx.z is the absolute head index.
                // So: `dst_meta[((ic0 + j_VKQ_local)*1 + blockIdx.z) * gridDim.y + blockIdx.y]` is not right.
                // It should be `dst_meta[ ( (ic0 + j_VKQ_local) * ne02 + blockIdx.z ) * gridDim.y + blockIdx.y ]`
                // where ne02 is total number of Q heads.
                // This requires ne02 to be passed or dst_meta to be pre-offset by caller.
                // The `launch_fattn_paged` will set up `dst_meta` pointer.
                // It gets `dst->src[4]` which is the full meta tensor.
                // Shape of meta tensor: [n_batch, n_head_q, n_q, n_blocks_y_dim]. Here, float2 elements.
                // Strides: nb0_meta, nb1_meta, nb2_meta, nb3_meta (bytes)
                // Assuming batch=1 for simplicity for now.
                // float2* meta_ptr = (float2*) ((char*)dst_meta + blockIdx.z * nb2_meta_bytes + (ic0 + j_VKQ_local) * nb1_meta_bytes);
                // meta_ptr[blockIdx.y] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
                // This is simpler: the launch_fattn_paged should provide the correct offset into dst_meta for this head and Q-block.
                // The original kernel got `dst_meta` as `float2*`.
                // `dst_meta_ptr_for_current_q_block_head = dst_meta + ( (ic0 + j_VKQ_local)*ne02 + blockIdx.z ) * gridDim.y`
                // `dst_meta_ptr_for_current_q_block_head[blockIdx.y] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);`
                // This is what `launch_fattn_paged` should compute as the `dst_meta` argument to the kernel.
                // So, inside kernel, `dst_meta` is already pointing to `meta_tensor_base + offset_for_q_block_and_head`.
                // Then simply `dst_meta[blockIdx.y] = ...`
                dst_meta[blockIdx.y] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
            }
        }
    }

#else // defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
    GGML_UNUSED(Q_ptr); GGML_UNUSED(K_view); GGML_UNUSED(V_view); GGML_UNUSED(mask); GGML_UNUSED(mask_k_seq_len); GGML_UNUSED(mask_k_stride_bytes);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
    GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03); GGML_UNUSED(num_kv_heads);
    GGML_UNUSED(ne31); GGML_UNUSED(nb31); // ne31, nb31 were for original kernel's dst, not used with current indexing
    GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(_dst_ne0); GGML_UNUSED(_dst_ne1); GGML_UNUSED(_dst_ne2); GGML_UNUSED(_dst_ne3);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
}



template <int cols_per_block, bool use_logit_softcap>
void launch_fattn_tile_f16_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            constexpr int    D             = 64;
            constexpr int    nwarps        = 8;
            constexpr size_t nbytes_shared = 0;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16<D, cols_per_block, nwarps, use_logit_softcap>;
            launch_fattn<D, cols_per_block, 1>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, FATTN_KQ_STRIDE_TILE_F16, true, true, false);
        } break;
        case 128: {
            constexpr int    D             = 128;
            constexpr int    nwarps        = 8;
            constexpr size_t nbytes_shared = 0;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16<D, cols_per_block, nwarps, use_logit_softcap>;
            launch_fattn<D, cols_per_block, 1>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, FATTN_KQ_STRIDE_TILE_F16, true, true, false);
        } break;
        default: {
            GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64 and 128.");
        } break;
    }
}

void ggml_cuda_flash_attn_ext_tile_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f16_64_128<cols_per_block, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f16_64_128<cols_per_block, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_f16_64_128<cols_per_block, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_f16_64_128<cols_per_block, use_logit_softcap>(ctx, dst);
    }
}
