#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

typedef tile<16, 8, half2> tile_A;
typedef tile< 8, 8, half2> tile_B;
typedef tile<16, 8, float> tile_C_KQ;
typedef tile<16, 4, half2> tile_C_VKQ;

template<int D, int nwarps, int KQ_stride>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV) {
    constexpr int D2_padded = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    // If cp.async is available, load up to the highest power of 2 in D asynchronously:
#ifdef CP_ASYNC_AVAILABLE
    static_assert(D >= 64 && D < 512, "bad D");
    constexpr int k0_sync_start = D/2 < 64 ? 32 : (D/2 < 128 ? 64 : 128);

    const unsigned int tile_KV_32 = __cvta_generic_to_shared(tile_KV);

    constexpr int preload = 64;
    constexpr int h2_per_chunk = 16/sizeof(half2);
    constexpr int chunks_per_row = k0_sync_start / h2_per_chunk;
    constexpr int stride_i = WARP_SIZE / chunks_per_row;
#pragma unroll
    for (int i0 = 0; i0 < KQ_stride; i0 += nwarps*stride_i) {
        const int i = i0 + threadIdx.y*stride_i + (chunks_per_row == WARP_SIZE ? 0 : threadIdx.x / chunks_per_row);
        const int k = (chunks_per_row == WARP_SIZE ? threadIdx.x : threadIdx.x % chunks_per_row)*h2_per_chunk;

        cp_async_cg_16<preload>(tile_KV_32 + (i*D2_padded + k)*sizeof(half2), KV + i*stride_KV + k);
    }
#else
    constexpr int k0_sync_start = 0;
#endif // CP_ASYNC_AVAILABLE
    static_assert(k0_sync_start % WARP_SIZE == 0, "bad k0_sync_start");

    // If D is not a power of 2, the rest is loaded synchronously.
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    static_assert(KQ_stride % (4*nwarps) == 0, "out of bounds");
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start = stride_k == WARP_SIZE ? k0_sync_start : D/2 - (D/2) % (2*stride_k);
        const int k0_stop  =                                         D/2 - (D/2) % (1*stride_k);
        const int stride_i = WARP_SIZE / stride_k;

        if (k0_start == k0_stop || k0_stop <= k0_sync_start) {
            continue;
        }

#pragma unroll
        for (int i0 = 0; i0 < KQ_stride; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                tile_KV[i*D2_padded + k] = KV[i*stride_KV + k];
            }
        }
    }
}

template<int D, int ncols, int nwarps, int KQ_stride, bool use_logit_softcap, bool needs_fixup, bool is_fixup, bool last_iter>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ maskh,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const int ne01,
        const int ne02,
        const int stride_Q,
        const int stride_KV,
        const int stride_mask,
        const int jt,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        const tile_B * const __restrict__ Q_B,
        tile_C_VKQ   * const __restrict__ VKQ_C,
        float2 & KQ_max,
        float2 & KQ_rowsum,
        const int kb0) {
#ifdef NEW_MMA_AVAILABLE
    constexpr int np = nwarps*tile_B::I / ncols; // Number of parallel CUDA warps per Q column.
    constexpr int D2_padded = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    const int k_VKQ_0 = kb0*KQ_stride;
    tile_C_KQ KQ_C[KQ_stride/(np*tile_C_KQ::I)];

#ifdef CP_ASYNC_AVAILABLE
    cp_async_wait_all();
    __syncthreads();
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_stride>(V_h2 + k_VKQ_0*stride_KV, tile_V, stride_KV);
#else
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_stride>(K_h2 + k_VKQ_0*stride_KV, tile_K, stride_KV);
    __syncthreads();
#endif // CP_ASYNC_AVAILABLE

    // Calculate tile of KQ:
#pragma unroll
    for (int i_KQ_00 = 0; i_KQ_00 < KQ_stride; i_KQ_00 += np*tile_A::I) {
        const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*tile_A::I;
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += tile_A::J) {
            tile_A K_A;
            load_ldmatrix(K_A, tile_K + i_KQ_0*D2_padded + k_KQ_0, D2_padded);
            mma(KQ_C[i_KQ_00/(np*tile_A::I)], K_A, ((tile_B *) Q_B)[k_KQ_0/tile_A::J]);
        }
    }

#ifndef CP_ASYNC_AVAILABLE
    __syncthreads(); // Only needed if tile_K == tile_V.
#endif // CP_ASYNC_AVAILABLE

    if (use_logit_softcap) {
        static_assert(KQ_stride % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < KQ_stride/(np*tile_C_KQ::I); ++i) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    if (maskh) {
        static_assert(KQ_stride % (np       *tile_C_KQ::I) == 0, "bad loop size");
        static_assert(ncols     % (nwarps/np*tile_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int i00 = 0; i00 < KQ_stride; i00 += np*tile_C_KQ::I) {
            const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ::I;
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                const int i = i0 + tile_C_KQ::get_i(l);
                const int j = (threadIdx.y / np)*tile_C_KQ::J + tile_C_KQ::get_j(l);

                KQ_C[i00/(np*tile_C_KQ::I)].x[l] += slope*__half2float(maskh[j*stride_mask + k_VKQ_0 + i]);
            }
        }
    }

    // Calculate softmax for each KQ column using the current max. value.
    // The divisor is stored in KQ_rowsum and will be applied at the end.
    float2 KQ_max_new = KQ_max;
    static_assert(KQ_stride % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
    for (int k = 0; k < KQ_stride/(np*tile_C_KQ::I); ++k) {
#pragma unroll
        for (int l0 = 0; l0 < tile_C_KQ::ne; l0 += 2) {
            KQ_max_new.x = fmaxf(KQ_max_new.x, KQ_C[k].x[l0 + 0]);
            KQ_max_new.y = fmaxf(KQ_max_new.y, KQ_C[k].x[l0 + 1]);
        }
    }

    // Values per KQ column are spread across 8 threads, does not need full warp reduce:
#pragma unroll
    for (int offset = 16; offset > 2; offset >>= 1) {
        KQ_max_new.x = fmaxf(KQ_max_new.x, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new.x, offset, WARP_SIZE));
        KQ_max_new.y = fmaxf(KQ_max_new.y, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new.y, offset, WARP_SIZE));
    }

    float2 KQ_rowsum_add = make_float2(0.0f, 0.0f);
    static_assert(KQ_stride % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
    for (int k = 0; k < KQ_stride/(np*tile_C_KQ::I); ++k) {
#pragma unroll
        for (int l = 0; l < tile_C_KQ::ne; ++l) {
            const float KQ_max_l = l % 2 == 0 ? KQ_max_new.x : KQ_max_new.y;
            const float diff = KQ_C[k].x[l] - KQ_max_l;
            KQ_C[k].x[l] = expf(diff);

            if (l % 2 == 0) {
                KQ_rowsum_add.x += KQ_C[k].x[l];
            } else {
                KQ_rowsum_add.y += KQ_C[k].x[l];
            }
        }
    }

    {
        const float2 diff = make_float2(KQ_max.x - KQ_max_new.x, KQ_max.y - KQ_max_new.y);
        const float2 KQ_max_scale = make_float2(expf(diff.x), expf(diff.y));
        KQ_max = KQ_max_new;

        // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
        KQ_rowsum.x = KQ_max_scale.x*KQ_rowsum.x + KQ_rowsum_add.x;
        KQ_rowsum.y = KQ_max_scale.y*KQ_rowsum.y + KQ_rowsum_add.y;

        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale.x, KQ_max_scale.y);
#pragma unroll
        for (int i = 0; i < D/tile_C_VKQ::I; ++i) {
#pragma unroll
            for (int l = 0; l < tile_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    tile_B B[KQ_stride/(np*2*tile_B::J)];
    static_assert(KQ_stride % (np*2*tile_B::J) == 0, "bad loop size");
#pragma unroll
    for (int k = 0; k < KQ_stride/(np*2*tile_B::J); ++k) {
        B[k] = get_transposed(get_half2(KQ_C[k]));
    }

#ifdef CP_ASYNC_AVAILABLE
    cp_async_wait_all();
    __syncthreads();
    if (!last_iter) {
        flash_attn_ext_f16_load_tile<D, nwarps, KQ_stride>(K_h2 + (k_VKQ_0 + KQ_stride)*stride_KV, tile_K, stride_KV);
    }
#else
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_stride>(V_h2 + k_VKQ_0*stride_KV, tile_V, stride_KV);
    __syncthreads();
#endif // CP_ASYNC_AVAILABLE

    // Calculate VKQ tile:
#pragma unroll
    for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += tile_C_VKQ::I) {
        static_assert((KQ_stride/2) % (np*tile_A::J) == 0, "bad loop size");
#pragma unroll
        for (int k00 = 0; k00 < KQ_stride/2; k00 += np*tile_A::J) {
            const int k0 = k00 + (threadIdx.y % np)*tile_A::J;

            tile_A A;
            load_ldmatrix_trans(A, tile_V + 2*k0*D2_padded + i_VKQ_0/2, D2_padded);
            mma(VKQ_C[i_VKQ_0/tile_C_VKQ::I], A, B[k00/(np*tile_A::J)]);
        }
    }

#ifndef CP_ASYNC_AVAILABLE
    __syncthreads(); // Only needed if tile_K == tile_V.
#endif // CP_ASYNC_AVAILABLE

#else
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template<int D, int ncols, int nwarps, int KQ_stride, bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ maskh,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const int ne01,
        const int ne02,
        const int stride_Q,
        const int stride_KV,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
#ifdef NEW_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    static_assert(nwarps*tile_B::I % ncols == 0, "bad nwarps");
    constexpr int np = nwarps*tile_B::I / ncols; // Number of parallel CUDA warps per Q column.

    static_assert(D         % nwarps == 0, "bad D");
    static_assert(KQ_stride % nwarps == 0, "bad KQ_stride");

    constexpr int D2_padded = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    // Temporary shared buffer for loading K/V data with KQ_stride*D logical elements:
    extern __shared__ half2 tile_K[];
#ifdef CP_ASYNC_AVAILABLE
    half2 * tile_V = tile_K + KQ_stride*D2_padded;
#else
    half2 * tile_V = tile_K;
#endif // CP_ASYNC_AVAILABLE

    tile_B Q_B[D/(2*tile_B::J)];
    tile_C_VKQ VKQ_C[D/tile_C_VKQ::I];

    float2 KQ_rowsum = {0.0f, 0.0f};
    float2    KQ_max = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};

    // Temporarily load Q data into tile_K, will be loaded into registers afterwards.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
        const int k0_stop  =                             D/2 - (D/2) % (1*stride_k);
        const int stride_j = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

        if (nwarps*stride_j > ncols && threadIdx.y*stride_j >= ncols) {
            break;
        }

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps*stride_j) {
            const int j = j0 + threadIdx.y*stride_j + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jt*ncols + j < ne01) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols + j)*stride_Q + k];
                    tile_K[j*D2_padded + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_K[j*D2_padded + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    {
        const int j0 = (threadIdx.y / np) * tile_B::I;

#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += tile_B::J) {
            load_ldmatrix(Q_B[k0/tile_B::J], tile_K + j0*D2_padded + k0, D2_padded);
        }
    }

    __syncthreads();

    // Preload K data for first iteration when using cp_async:
#ifdef CP_ASYNC_AVAILABLE
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_stride>(K_h2 + kb0_start*KQ_stride*stride_KV, tile_K, stride_KV);
#endif // CP_ASYNC_AVAILABLE

    // Iterate over ne11 == previous tokens:
    for (int kb0 = kb0_start; kb0 < kb0_stop-1; ++kb0) {
        constexpr bool last_iter = false;
        flash_attn_ext_f16_iter<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, maskh, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_Q, stride_KV, stride_mask, jt, tile_K, tile_V, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    }
    { // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
        constexpr bool last_iter = true;
        flash_attn_ext_f16_iter<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, maskh, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_Q, stride_KV, stride_mask, jt, tile_K, tile_V, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0_stop-1);
    }

    // With cp_async there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
#ifdef CP_ASYNC_AVAILABLE
    if (nwarps*tile_B::I > KQ_stride) {
        __syncthreads();
    }
#endif // CP_ASYNC_AVAILABLE

    // Finally, sum up partial KQ rowsums.
    // The partial sums are spread across 8 threads each, does not need full reduce.
#pragma unroll
    for (int offset = 16; offset > 2; offset >>= 1) {
        KQ_rowsum.x += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum.x, offset, WARP_SIZE);
        KQ_rowsum.y += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum.y, offset, WARP_SIZE);
    }

    // Write VKQ accumulators to shared memory in column-major format.
    // It's faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // Also for np > 1 the combination is done via these values in shared memory.
    const int j_cwd = threadIdx.y*tile_B::I + tile_B::get_i(-1); // j combine write data
#pragma unroll
    for (int k0 = 0; k0 < D/2; k0 += tile_B::J) {
        const tile_B B = get_transposed(VKQ_C[k0/tile_B::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
        for (int l = 0; l < tile_B::ne; ++l) {
            const int k = k0 + tile_B::get_j(l);

            tile_K[j_cwd*D2_padded + k] = B.x[l];
        }
    }

    const int j_cwmo = (threadIdx.x % (2*tile_C_VKQ::J)) / tile_C_VKQ::J; // j combine write meta offset
    const int j_cwm = threadIdx.y*(2*tile_C_VKQ::J) + 2*tile_C_VKQ::get_j(-1) + j_cwmo; // j combine write meta
    const float2 KQ_cmr = make_float2(((const float *) &KQ_max)[j_cwmo], ((const float *) &KQ_rowsum)[j_cwmo]); // KQ combine max rowsum

    if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*tile_C_VKQ::J) {
        // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
        ((float2 *) tile_K)[j_cwm*(D2_padded/2) + D/4] = KQ_cmr;
    }

    __syncthreads();

    static_assert(np == 1 || np == 2 || np == 4, "bad np");
    if (np == 1) {
        // No combination is needed, the meta data can be directly written from registers to VRAM.
        if (needs_fixup && threadIdx.x < tile_B::I) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[j_cwm] = KQ_cmr;
        }
        if (is_fixup && threadIdx.x < tile_B::I) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[j_cwm] = KQ_cmr;
        }
    } else if (threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        float * meta_j = (float *) tile_K + (threadIdx.y*tile_B::I + threadIdx.x)*D2_padded + D/2;

        float KQ_cm = -FLT_MAX/2; // KQ combine max per parallel warp.
        if (np*tile_B::I == WARP_SIZE || threadIdx.x < np*tile_B::I) {
            KQ_cm = meta_j[0];
        }

        float KQ_cmn = KQ_cm; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int offset = np*tile_B::I/2; offset >= tile_B::I; offset >>= 1) {
            KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
        }

        const float KQ_cms = expf(KQ_cm - KQ_cmn); // KQ combine max scale per warp.
        float KQ_crs = 0.0f; // KQ combine rowsum, scaled sum of all parallel warps.
        if (np*tile_B::I == WARP_SIZE || threadIdx.x < np*tile_B::I) {
            KQ_crs = KQ_cms*meta_j[1];
        }
#pragma unroll
        for (int offset = np*tile_B::I/2; offset >= tile_B::I; offset >>= 1) {
            KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
        }

        // Write back combined meta data:
        if (np*tile_B::I == WARP_SIZE || threadIdx.x < np*tile_B::I) {
            *((float2 *) meta_j) = make_float2(KQ_cms, KQ_crs); // Combined KQ max scale + rowsum.
        }
        if (needs_fixup && threadIdx.x < tile_B::I) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*tile_B::I + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && threadIdx.x < tile_B::I) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*tile_B::I + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    }

    if (np > 1) {
        __syncthreads();
    }

    if (np == 1 || threadIdx.y % np == 0) {
        // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
        // The values after that are for the partial results of the individual blocks.
        float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(D/2));

#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int k0_start = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
            const int k0_stop  =                             D/2 - (D/2) % (1*stride_k);
            const int stride_j = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                continue;
            }

            if (nwarps*stride_j > ncols && threadIdx.y*stride_j >= ncols) {
                break;
            }

#pragma unroll
            for (int j0_dst = 0; j0_dst < ncols; j0_dst += (nwarps/np)*stride_j) {
                const int j_dst = j0_dst + (threadIdx.y/np)*stride_j + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);
                const int j_tile_K = (j_dst/tile_B::I)*(np*tile_B::I) + j_dst % tile_B::I;

                if (!is_fixup && jt*ncols + j_dst >= ne01) {
                    continue;
                }
                const float * meta_j = (const float *) tile_K + j_tile_K*D2_padded + D/2;
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                    for (int ip = 0; ip < np; ++ip) {
                        const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*tile_B::I*D2_padded + 0];
                        const float2 dstk_val_add = __half22float2(tile_K[(j_tile_K + ip*tile_B::I)*D2_padded + k]);
                        dstk_val.x += dstk_val_add.x*KQ_crs;
                        dstk_val.y += dstk_val_add.y*KQ_crs;
                    }

                    if (!needs_fixup && !is_fixup) {
                        const float KQ_rowsum_j = meta_j[1];
                        dstk_val.x /= KQ_rowsum_j;
                        dstk_val.y /= KQ_rowsum_j;
                    }

                    if (is_fixup) {
                        dstk_fixup_data[j_dst*(D/2) + k] = dstk_val;
                    } else {
                        dstk[(jt*ncols + j_dst)*ne02*(D/2) + k] = dstk_val;
                    }
                }
            }
        }
    }

    if (np > 1) {
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template<int D, int ncols, int nwarps, int KQ_stride, bool use_logit_softcap>
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 2)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_ext_f16(
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
#ifndef NEW_MMA_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // NEW_MMA_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(FATTN_KQ_STRIDE % KQ_stride == 0, "bad KQ_stride");

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q    = nb01 / sizeof(float2);
    const int stride_KV   = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int iter_k = ne11 / KQ_stride;
    const int iter_j = (ne01 + (ncols - 1)) / ncols;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = (blockIdx.x + 0)*iter_k*iter_j*ne02 / gridDim.x;
    const int kbc_stop = (blockIdx.x + 1)*iter_k*iter_j*ne02 / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int channel = kbc / (iter_k*iter_j);
        const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

        const float2 * Q_f2  = (const float2 *) (Q + nb02* channel);
        const half2  * K_h2  = (const half2  *) (K + nb12*(channel / gqa_ratio));
        const half2  * V_h2  = (const half2  *) (V + nb12*(channel / gqa_ratio)); // K and V have same shape
        const half   * maskh = mask ? (const half  *) mask + (nb31/sizeof(half))*jt*ncols : nullptr;
        float2       * dstk  = ((float2 *) dst) + channel*(D/2);

        const float slope = get_alibi_slope(max_bias, channel, n_head_log2, m0, m1);

        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, stride_Q, stride_KV, stride_mask, jt, kb0_start, kb0_stop);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is working on the beginning of a tile.
            flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, stride_Q, stride_KV, stride_mask, jt, kb0_start, kb0_stop);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int channel = kbc / (iter_k*iter_j);
    const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

    const float2 * Q_f2  = (const float2 *) (Q + nb02* channel);
    const half2  * K_h2  = (const half2  *) (K + nb12*(channel / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V + nb12*(channel / gqa_ratio)); // K and V have same shape
    const half   * maskh = mask ? (const half  *) mask + (nb31/sizeof(half))*jt*ncols : nullptr;
    float2       * dstk  = ((float2 *) dst) + channel*(D/2);

    const float slope = get_alibi_slope(max_bias, channel, n_head_log2, m0, m1);

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<D, ncols, nwarps, KQ_stride, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, maskh, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, stride_Q, stride_KV, stride_mask, jt, kb0_start, kb0_stop);
}

template <int D, int cols_per_block>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    typedef tile<16, 8, half2> tile_A;
    typedef tile< 8, 8, half2> tile_B;

    static_assert(D              % tile_B::J == 0, "bad D");
    static_assert(cols_per_block % tile_B::I == 0, "bad cols_per_block");

    const ggml_tensor * KQV = dst;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    constexpr int KQ_stride = D <= 128 ? 64 : 32;
    constexpr int nwarps    = (KQ_stride == 32 && cols_per_block <= 16) ?
                              cols_per_block/tile_B::J * KQ_stride/tile_A::I : (cols_per_block <= 8 ? 4 : 8);

    const int    nrows_KQ      = cp_async_available(cc) ? 2*KQ_stride : KQ_stride;
    const int    nrows_combine = nwarps*tile_B::J;
    const size_t nbytes_shared = std::max(nrows_KQ, nrows_combine) * (D + 8) * sizeof(half);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<D, cols_per_block, nwarps, KQ_stride, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<D, cols_per_block, nwarps, KQ_stride, use_logit_softcap>;
    }
    launch_fattn<D, cols_per_block, 0, KQ_stride>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, true, true);
}

#define DECL_FATTN_MMA_F16_CASE(D, cols_per_block)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                     \
    <D, cols_per_block>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_MMA_F16_CASE( 64,  8);
extern DECL_FATTN_MMA_F16_CASE( 80,  8);
extern DECL_FATTN_MMA_F16_CASE( 96,  8);
extern DECL_FATTN_MMA_F16_CASE(112,  8);
extern DECL_FATTN_MMA_F16_CASE(128,  8);
extern DECL_FATTN_MMA_F16_CASE(256,  8);

extern DECL_FATTN_MMA_F16_CASE( 64, 16);
extern DECL_FATTN_MMA_F16_CASE( 80, 16);
extern DECL_FATTN_MMA_F16_CASE( 96, 16);
extern DECL_FATTN_MMA_F16_CASE(112, 16);
extern DECL_FATTN_MMA_F16_CASE(128, 16);
extern DECL_FATTN_MMA_F16_CASE(256, 16);

extern DECL_FATTN_MMA_F16_CASE( 64, 32);
extern DECL_FATTN_MMA_F16_CASE( 80, 32);
extern DECL_FATTN_MMA_F16_CASE( 96, 32);
extern DECL_FATTN_MMA_F16_CASE(112, 32);
extern DECL_FATTN_MMA_F16_CASE(128, 32);
extern DECL_FATTN_MMA_F16_CASE(256, 32);

extern DECL_FATTN_MMA_F16_CASE( 64, 64);
extern DECL_FATTN_MMA_F16_CASE( 80, 64);
extern DECL_FATTN_MMA_F16_CASE( 96, 64);
extern DECL_FATTN_MMA_F16_CASE(112, 64);
extern DECL_FATTN_MMA_F16_CASE(128, 64);
extern DECL_FATTN_MMA_F16_CASE(256, 64);
