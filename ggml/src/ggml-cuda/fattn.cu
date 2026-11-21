#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#if !defined(GGML_USE_MUSA)

// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/warp_size];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        ggml_cuda_thread_mask_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = ggml_cuda_ballot((int) selected);
            warp_count += ggml_cuda_popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/warp_size; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const ggml_cuda_thread_mask_t lane_mask = lane == 0 ? 0 : ((ggml_cuda_thread_mask_t)1 << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + ggml_cuda_popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & ((ggml_cuda_thread_mask_t)1 << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += ggml_cuda_popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}

// Build sparse index lists directly from top_k_mask inputs without materializing the dense mask.
// Processes columns in 32 KiB shared-memory chunks (262144 cols/pass for ncols1==1, 32768 for ncols1==8).
template <int ncols1>
__launch_bounds__(256, 1)
static __global__ void flash_attn_top_k_mask_to_sparse_indices(
        const half * base_mask_ptr, const int32_t * top_k_ptr, const float * score_ptr,
        const int32_t * blocks_ptr, const int32_t * extra_ptr,
        int32_t * indices_ptr, int32_t * counts_ptr, uint8_t * sparse_qmask_ptr,
        const int64_t n_kv, const int32_t n_queries, const int32_t n_kv_max,
        const int64_t fa_ne1, const int64_t tkm_ne1,
        const int64_t m1, const int64_t m3,
        const int64_t n_top, const int64_t n_pick, const int64_t n_blk, const int64_t n_extra,
        const int64_t sm1, const int64_t sm3,
        const int64_t sk0, const int64_t sk1, const int64_t sk3,
        const int64_t ss0, const int64_t ss1, const int64_t ss3,
        const int64_t sb0, const int64_t sb1,
        const int64_t se0, const int64_t se1, const int64_t se3) {
    ggml_cuda_pdl_sync();

    static_assert(ncols1 == 1 || ncols1 == 8, "unsupported ncols1");
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps       = 256 / warp_size;
    constexpr int smem_words    = 8192; // 32 KiB
    constexpr int cols_per_word = ncols1 == 1 ? 32 : 4;
    constexpr int chunk_cols    = smem_words * cols_per_word;

    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0      = group * ncols1;
    const int q_count = min(ncols1, n_queries - q0);

    const int64_t list_idx = int64_t(sequence) * gridDim.x + group;
    int32_t * indices      = indices_ptr + list_idx * n_kv_max;
    uint8_t * sparse_qmask = ncols1 > 1 ? sparse_qmask_ptr + list_idx * n_kv_max : nullptr;

    __shared__ uint4           smem_v4[smem_words / 4];
    __shared__ int             warp_offsets[n_warps];
    __shared__ int             iter_count;
    __shared__ const half    * mask_rows[ncols1];
    __shared__ const int32_t * top_k_rows[ncols1];
    __shared__ const float   * score_rows[ncols1];
    __shared__ const int32_t * extra_rows[ncols1];
    uint32_t * smem = (uint32_t *) smem_v4;

    if (tid < q_count) {
        const int64_t ir = int64_t(sequence) * fa_ne1 + (q0 + tid);
        const int64_t i1 = ir % tkm_ne1;
        const int64_t i3 = ir / tkm_ne1;
        mask_rows[tid]  = base_mask_ptr + (i1 % m1)*sm1 + (i3 % m3)*sm3;
        top_k_rows[tid] = top_k_ptr + i1*sk1 + i3*sk3;
        score_rows[tid] = score_ptr != nullptr ? score_ptr + i1*ss1 + i3*ss3 : nullptr;
        extra_rows[tid] = extra_ptr != nullptr ? extra_ptr + i1*se1 + i3*se3 : nullptr;
    }
    __syncthreads();

    int row_count = 0;

    for (int64_t chunk_start = 0; chunk_start < n_kv && row_count < n_kv_max; chunk_start += chunk_cols) {
        const int     chunk_len = int(n_kv - chunk_start < chunk_cols ? n_kv - chunk_start : chunk_cols);
        const int64_t chunk_end = chunk_start + chunk_len;
        const int     n_words   = (chunk_len + cols_per_word - 1) / cols_per_word;
        const int     n_vec4    = (n_words + 3) / 4;

        for (int v = tid; v < n_vec4; v += blockDim.x) {
            smem_v4[v] = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

        auto mark_cell = [&](const int32_t c, const int q, const half * mask_row) {
            if (c >= chunk_start && c < chunk_end && isfinite(__half2float(mask_row[c]))) {
                const int local_c = int(c - chunk_start);
                if constexpr (ncols1 == 1) {
                    atomicOr(&smem[local_c >> 5], 1u << (local_c & 31));
                } else {
                    atomicOr(&smem[local_c >> 2], (1u << q) << (8 * (local_c & 3)));
                }
            }
        };

        for (int q = 0; q < q_count; ++q) {
            const half    * mask_row  = mask_rows[q];
            const int32_t * top_k_row = top_k_rows[q];
            const float   * score_row = score_rows[q];
            const int32_t * extra_row = extra_rows[q];

            if (blocks_ptr == nullptr) {
                for (int64_t i = tid; i < n_top; i += blockDim.x) {
                    const int32_t p = top_k_row[i*sk0];
                    if (p >= 0 && p < n_pick && (score_row == nullptr || score_row[p*ss0] > -INFINITY)) {
                        mark_cell(p, q, mask_row);
                    }
                }
            } else if (n_blk < 32) {
                for (int64_t i = tid; i < n_top; i += blockDim.x) {
                    const int32_t p = top_k_row[i*sk0];
                    if (p < 0 || p >= n_pick) {
                        continue;
                    }
                    if (score_row != nullptr && !(score_row[p*ss0] > -INFINITY)) {
                        continue;
                    }
                    const int32_t * blk = blocks_ptr + p*sb1;
                    for (int64_t k = 0; k < n_blk; ++k) {
                        mark_cell(blk[k*sb0], q, mask_row);
                    }
                }
            } else {
                for (int64_t i = 0; i < n_top; ++i) {
                    const int32_t p = top_k_row[i*sk0];
                    if (p < 0 || p >= n_pick) {
                        continue;
                    }
                    if (score_row != nullptr && !(score_row[p*ss0] > -INFINITY)) {
                        continue;
                    }
                    const int32_t * blk = blocks_ptr + p*sb1;
                    for (int64_t k = tid; k < n_blk; k += blockDim.x) {
                        mark_cell(blk[k*sb0], q, mask_row);
                    }
                }
            }

            if (extra_row != nullptr) {
                for (int64_t i = tid; i < n_extra; i += blockDim.x) {
                    mark_cell(extra_row[i*se0], q, mask_row);
                }
            }
        }
        __syncthreads();

        for (int v0 = 0; v0 < n_vec4; v0 += blockDim.x) {
            const int      v        = v0 + tid;
            const uint4    w4       = v < n_vec4 ? smem_v4[v] : make_uint4(0u, 0u, 0u, 0u);
            const uint32_t words[4] = { w4.x, w4.y, w4.z, w4.w };

            uint32_t active[4] = { 0u, 0u, 0u, 0u };
            int cnt = 0;
            if ((w4.x | w4.y | w4.z | w4.w) != 0u) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t word = words[j];
                    if (word != 0u) {
                        if constexpr (ncols1 == 1) {
                            active[j] = word;
                        } else {
                            active[j] = ((word & 0x000000ffu) != 0 ? 1u : 0u) |
                                        ((word & 0x0000ff00u) != 0 ? 2u : 0u) |
                                        ((word & 0x00ff0000u) != 0 ? 4u : 0u) |
                                        ((word & 0xff000000u) != 0 ? 8u : 0u);
                        }
                        cnt += __popc(active[j]);
                    }
                }
            }

            const int lane_scan = warp_prefix_inclusive_sum<int, warp_size>(cnt);
            const int lane_off  = lane_scan - cnt;

            if (lane == warp_size - 1) {
                warp_offsets[warp] = lane_scan;
            }
            __syncthreads();

            if (tid == 0) {
                int offset = 0;
#pragma unroll
                for (int iw = 0; iw < n_warps; ++iw) {
                    const int wc = warp_offsets[iw];
                    warp_offsets[iw] = offset;
                    offset += wc;
                }
                iter_count = offset;
            }
            __syncthreads();

            if (cnt > 0) {
                int dst = row_count + warp_offsets[warp] + lane_off;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t mask = active[j];
                    if (mask != 0u) {
                        const int64_t word_col_base = chunk_start + int64_t(v*4 + j)*cols_per_word;
                        while (mask != 0u) {
                            const int bit = __ffs((int) mask) - 1;
                            if (dst < n_kv_max) {
                                indices[dst] = int32_t(word_col_base + bit);
                                if constexpr (ncols1 > 1) {
                                    sparse_qmask[dst] = uint8_t((words[j] >> (8*bit)) & 0xffu);
                                }
                            }
                            ++dst;
                            mask &= mask - 1;
                        }
                    }
                }
            }

            row_count += iter_count;
            __syncthreads();

            if (row_count >= n_kv_max) {
                break;
            }
        }
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
        if constexpr (ncols1 > 1) {
            sparse_qmask[i] = 0;
        }
    }
    if (tid == 0) {
        counts_ptr[list_idx] = count;
    }
    __syncthreads();

    // the dependent grid reads indices and sparse_qmask, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is not supported on MUSA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1       ? flash_attn_mask_to_sparse_indices<1, false> :
                        n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>  :
                                             flash_attn_mask_to_sparse_indices<8, false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_MUSA)
}

void ggml_cuda_flash_attn_ext_compact_top_k_mask(
        const ggml_tensor * top_k_mask, const ggml_tensor * fa_mask, int32_t * indices, int32_t * counts,
        uint8_t * sparse_qmask, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(top_k_mask, fa_mask, indices, counts, sparse_qmask, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is not supported on MUSA");
#else
    const ggml_tensor * base_mask = top_k_mask->src[0];
    const ggml_tensor * top_k     = top_k_mask->src[1];
    const ggml_tensor * score     = top_k_mask->src[2];
    const ggml_tensor * blocks    = top_k_mask->src[3];
    const ggml_tensor * extra     = top_k_mask->src[4];

    const int64_t n_kv   = base_mask->ne[0];
    const int64_t n_pick = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    const size_t  si     = sizeof(int32_t);
    const size_t  sf     = sizeof(float);

    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, fa_mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);

    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1 ? flash_attn_top_k_mask_to_sparse_indices<1> :
                                      flash_attn_top_k_mask_to_sparse_indices<8>;

    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) base_mask->data, (const int32_t *) top_k->data,
        score  ? (const float   *) score->data  : nullptr,
        blocks ? (const int32_t *) blocks->data : nullptr,
        extra  ? (const int32_t *) extra->data  : nullptr,
        indices, counts, sparse_qmask,
        n_kv, n_queries, n_kv_max,
        fa_mask->ne[1], top_k_mask->ne[1],
        base_mask->ne[1], base_mask->ne[3],
        top_k->ne[0], n_pick, blocks ? blocks->ne[0] : 1, extra ? extra->ne[0] : 0,
        base_mask->nb[1]/sizeof(half), base_mask->nb[3]/sizeof(half),
        top_k->nb[0]/si, top_k->nb[1]/si, top_k->nb[3]/si,
        score  ? score->nb[0]/sf  : 0, score  ? score->nb[1]/sf  : 0, score  ? score->nb[3]/sf  : 0,
        blocks ? blocks->nb[0]/si : 0, blocks ? blocks->nb[1]/si : 0,
        extra  ? extra->nb[0]/si  : 0, extra  ? extra->nb[1]/si  : 0, extra  ? extra->nb[3]/si  : 0);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // On CDNA only the DKQ=576 kernel variant with ncols2=16 is compiled with sparse support.
    if (amd_mfma_available(cc) && !ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(int(Q->ne[0]), int(V->ne[0]), 1, 16)) {
        return false;
    }

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return ((GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc)) || amd_mfma_available(cc)) && dst->src[5] == nullptr &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, GGML_CUDA_CC_IS_NVIDIA(cc) ? 2*n_gather : 2LL*n_kv_max);
#endif // !defined(GGML_USE_MUSA)
}

static bool ggml_cuda_flash_attn_ext_mma_f16_use_gqa_opt(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // with kv_rows the kernel iterates over the mask columns
    const int64_t n_kv = dst->src[5] ? dst->src[5]->ne[0] : K->ne[1];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    // a mask with a head dimension is not shared by the Q heads of a K/V head
    bool use_gqa_opt = mask && mask->ne[2] == 1 && max_bias == 0.0f && n_kv % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }
    return use_gqa_opt;
}

static int ggml_cuda_flash_attn_ext_mma_f16_get_ncols2_576(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (amd_mfma_available(cc) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 16)) {
        return 16;
    }
    if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
        if (cc >= GGML_CUDA_CC_GB10) {
            return Q->ne[1] <= 8 ? 16 : 4;
        }
        if (cc >= GGML_CUDA_CC_BLACKWELL) {
            return (Q->ne[1] <= 4 && K->ne[1] >= 65536) ? 16 : 4;
        }
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return Q->ne[1] <= 4 ? 16 : 4;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            if (Q->ne[1] <= 4) {
                return K->ne[1] <= 16384 ? 16 : 32;
            }
            return 4;
        }
        // Volta:
        return 4;
    }
    if (gqa_ratio % 16 == 0) {
        return 16;
    }
    return 4;
}

static int ggml_cuda_flash_attn_ext_mma_f16_get_ncols1(
        const int cc, const ggml_tensor * dst, const int DKQ, const int DV, const int ncols2) {
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_MUSA)
    if (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        const bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            return 1;
        }
    }
#else
    GGML_UNUSED(DV);
#endif // !defined(GGML_USE_MUSA)

    if (ncols2 <= 8 && turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
        return 8 / ncols2;
    }

    if (ncols2 <= 16 && Q->ne[1] <= 16/ncols2) {
        return 16 / ncols2;
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        return 32 / ncols2;
    }

    return 64 / ncols2;
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * top_k_mask = nullptr) {
    const int cc     = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int ncols1 = ggml_cuda_flash_attn_ext_mma_f16_get_ncols1(cc, dst, DKQ, DV, ncols2);

#if !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ncols1 == 1) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst, top_k_mask);
            return;
        }
    }
#endif // !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (ncols1 == 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst, top_k_mask);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (ncols1 == 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst, top_k_mask);
            return;
        }
    }

    if (ncols1 == 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst, top_k_mask);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst, top_k_mask);
}

static int ggml_cuda_flash_attn_ext_mma_f16_get_ncols2(const int cc, const ggml_tensor * dst, const int DKQ) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const bool use_gqa_opt = ggml_cuda_flash_attn_ext_mma_f16_use_gqa_opt(dst);

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            return 8;
        }
        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            return 4;
        }
        if (DKQ <= 256) {
            return (use_gqa_opt && gqa_ratio % 2 == 0) ? 2 : 1;
        }
        return 0;
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            return 8;
        }
        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            return 4;
        }
        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            return 2;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        return 8;
    }
    if (use_gqa_opt && gqa_ratio > 2) {
        return 4;
    }
    if (use_gqa_opt && gqa_ratio > 1) {
        return 2;
    }
    return DKQ <= 256 ? 1 : 0;
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * top_k_mask = nullptr) {
    const int cc     = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int ncols2 = ggml_cuda_flash_attn_ext_mma_f16_get_ncols2(cc, dst, DKQ);

    switch (ncols2) {
        case 8:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst, top_k_mask);
            return;
        case 4:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst, top_k_mask);
            return;
        case 2:
            if constexpr (DKQ <= 256) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst, top_k_mask);
                return;
            }
            break;
        case 1:
            if constexpr (DKQ <= 256) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst, top_k_mask);
                return;
            }
            break;
        default:
            break;
    }
    GGML_ABORT("fatal error");
}

static void ggml_cuda_flash_attn_ext_mma_f16(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * top_k_mask = nullptr) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 40:
            GGML_ASSERT(V->ne[0] == 40);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 40,  40>(ctx, dst, top_k_mask);
            break;
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst, top_k_mask);
            break;
        case 72:
            GGML_ASSERT(V->ne[0] == 72);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 72,  72>(ctx, dst, top_k_mask);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst, top_k_mask);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst, top_k_mask);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst, top_k_mask);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst, top_k_mask);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst, top_k_mask);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst, top_k_mask);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst, top_k_mask);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst, top_k_mask);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst, top_k_mask);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            switch (ggml_cuda_flash_attn_ext_mma_f16_get_ncols2_576(cc, dst)) {
                case 16:
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst, top_k_mask);
                    break;
                case 32:
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst, top_k_mask);
                    break;
                default:
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst, top_k_mask);
                    break;
            }
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        if (head_size == (D) && type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {                          \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = type_K_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_K_in;
    const ggml_type type_V = type_V_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_V_in;
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, BF16)

    FATTN_VEC_CASES_ALL_D(F16, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q4_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q4_1, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q5_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q5_1, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q8_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(BF16, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, F8_E4M3)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_F8_E4M3:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * kv_rows = dst->src[5];

    // with kv_rows the kernel iterates over the mask columns
    const int64_t n_kv = kv_rows ? kv_rows->ne[0] : K->ne[1];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, if the mask has a head dimension,
    //     or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && mask->ne[2] == 1 && max_bias == 0.0f && n_kv % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        // the kv_rows mask is compacted per sequence, it has no head dimension
        if (kv_rows) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // the logit softcap kernels are compiled for these head sizes only
        float logit_softcap;
        memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
        if (logit_softcap != 0.0f && Q->ne[0] != 128 && Q->ne[0] != 256) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }

    // only the MMA kernel reads K/V through kv_rows
    if (kv_rows) {
        return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && mask && Q->ne[0] != 40 && Q->ne[0] != 72 ?
            BEST_FATTN_KERNEL_MMA_F16 : BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc)) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && (!mask || mask->ne[2] == 1) && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // On CDNA, the sparse MMA kernel is used for DKQ=576 regardless of the head size heuristics.
    if (amd_mfma_available(cc) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

#if !defined(GGML_USE_MUSA)
static bool ggml_cuda_flash_attn_ext_will_use_sparse(const int device, const ggml_tensor * dst) {
    if (ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_MMA_F16) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (dst->src[5] != nullptr || mask == nullptr) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    const int cc  = ggml_cuda_info().devices[device].cc;
    const int DKQ = int(Q->ne[0]);
    const int DV  = int(V->ne[0]);

    int ncols2 = 0;
    if ((DKQ == 256 && DV == 256) || (DKQ == 512 && DV == 512)) {
        ncols2 = ggml_cuda_flash_attn_ext_mma_f16_get_ncols2(cc, dst, DKQ);
    } else if (DKQ == 576 && DV == 512) {
        ncols2 = ggml_cuda_flash_attn_ext_mma_f16_get_ncols2_576(cc, dst);
    } else {
        return false;
    }
    if (ncols2 <= 0) {
        return false;
    }

    const int ncols1 = ggml_cuda_flash_attn_ext_mma_f16_get_ncols1(cc, dst, DKQ, DV, ncols2);
    return ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2) &&
           ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, ncols1, ncols2);
}
#endif // !defined(GGML_USE_MUSA)

bool ggml_cuda_flash_attn_ext_can_fuse_top_k_mask(const int device, const ggml_tensor * top_k_mask, const ggml_tensor * dst) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(device, top_k_mask, dst);
    return false;
#else
    if (top_k_mask == nullptr || dst == nullptr ||
            top_k_mask->op != GGML_OP_TOP_K_MASK || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    const ggml_tensor * fa_mask   = dst->src[3];
    const ggml_tensor * base_mask = top_k_mask->src[0];
    const ggml_tensor * top_k     = top_k_mask->src[1];
    const ggml_tensor * score     = top_k_mask->src[2];
    const ggml_tensor * blocks    = top_k_mask->src[3];
    const ggml_tensor * extra     = top_k_mask->src[4];

    if (fa_mask == nullptr || base_mask == nullptr || top_k == nullptr) {
        return false;
    }

    const bool mask_reshaped = fa_mask != top_k_mask;
    if (mask_reshaped && (fa_mask->op != GGML_OP_RESHAPE || fa_mask->src[0] != top_k_mask)) {
        return false;
    }

    if (top_k_mask->type != GGML_TYPE_F16 || base_mask->type != GGML_TYPE_F16 || fa_mask->type != GGML_TYPE_F16) {
        return false;
    }

    if (top_k_mask->ne[0] != fa_mask->ne[0] || base_mask->ne[0] != top_k_mask->ne[0] ||
            top_k_mask->ne[0] > INT32_MAX || fa_mask->nb[1] > INT32_MAX ||
            top_k_mask->ne[2] != 1 || fa_mask->ne[2] != 1 || base_mask->ne[2] != 1 ||
            top_k_mask->ne[1] * top_k_mask->ne[3] != fa_mask->ne[1] * fa_mask->ne[3]) {
        return false;
    }

    if (base_mask->nb[0] != sizeof(half) ||
            base_mask->nb[1] % sizeof(half) != 0 || base_mask->nb[3] % sizeof(half) != 0) {
        return false;
    }

    if (mask_reshaped) {
        const size_t row_bytes = size_t(base_mask->ne[0]) * sizeof(half);
        const bool is_1d_bcast = base_mask->ne[1] == 1 && base_mask->ne[3] == 1;
        const bool is_full_contig = base_mask->ne[1] == top_k_mask->ne[1] &&
                                    base_mask->ne[3] == top_k_mask->ne[3] &&
                                    ggml_is_contiguous(base_mask) &&
                                    (base_mask->ne[1] == 1 || base_mask->nb[1] == row_bytes) &&
                                    (base_mask->ne[3] == 1 || base_mask->nb[3] == size_t(base_mask->ne[1]) * row_bytes);
        if (!is_1d_bcast && !is_full_contig) {
            return false;
        }
    } else {
        if ((base_mask->ne[1] != 1 && (base_mask->ne[1] != top_k_mask->ne[1] || base_mask->nb[1] > INT32_MAX)) ||
                (base_mask->ne[3] != 1 && base_mask->ne[3] != top_k_mask->ne[3])) {
            return false;
        }
    }

    auto strides_aligned = [](const ggml_tensor * t, size_t elem_size) {
        return t == nullptr || (t->nb[0] % elem_size == 0 && t->nb[1] % elem_size == 0 && t->nb[3] % elem_size == 0);
    };
    if (!strides_aligned(top_k, sizeof(int32_t)) || !strides_aligned(score, sizeof(float)) ||
            (blocks != nullptr && (blocks->nb[0] % sizeof(int32_t) != 0 || blocks->nb[1] % sizeof(int32_t) != 0)) ||
            !strides_aligned(extra, sizeof(int32_t))) {
        return false;
    }

    return ggml_cuda_flash_attn_ext_will_use_sparse(device, dst);
#endif // !defined(GGML_USE_MUSA)
}

void ggml_cuda_flash_attn_ext_top_k_mask_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * top_k_mask, ggml_tensor * dst) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(ctx, top_k_mask, dst);
    GGML_ABORT("sparse flash attention is not supported on MUSA");
#else
    ggml_cuda_set_device(ctx.device);
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst, top_k_mask);
#endif // !defined(GGML_USE_MUSA)
}
