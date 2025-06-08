#include "common.cuh"
#include "fattn-common.cuh"
#include "paged_attn_common.cuh" // For paged view structures

template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
#ifndef GGML_USE_HIP
__launch_bounds__(D, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_vec_ext_f16(
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
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f16_t vec_dot_KQ = get_vec_dot_KQ_f16<D>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f16_t dequantize_1_v = get_dequantize_1_f16(type_V);

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb02* blockIdx.z              + nb01*ic0;
    K += nb12*(blockIdx.z / gqa_ratio);
    V += nb22*(blockIdx.z / gqa_ratio);

    const half * maskh = (const half   *)  mask + ne11*ic0;

    const float slopef = get_alibi_slope(max_bias, blockIdx.z, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ half KQ[ncols*D];
    half2 * KQ2 = (half2 *) KQ;

    half kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -HALF_MAX_HALF;
    }
    half kqsum[ncols] = {0.0f};

    __shared__ half kqmax_shared[ncols][WARP_SIZE];
    __shared__ half kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ half maskh_shared[ncols*D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskh_shared[j*D + tid] = 0.0f;
    }

    __syncthreads();

    // Convert Q to half2 (f16 K) or q8_1 (quantized K) and store in registers:
    half2  Q_h2[ncols][D/(2*WARP_SIZE)];
    int   Q_i32[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D/(sizeof(int)*QK8_1)];
    half2  Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_half2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<half2>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 tmp = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
            }
        }
    }


#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -HALF_MAX_HALF;
    }
    __syncthreads();

    half2 VKQ[ncols] = {{0.0f, 0.0f}};

    for (int k_VKQ_0 = blockIdx.y*D; k_VKQ_0 < ne11; k_VKQ_0 += gridDim.y*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskh_shared[j*D + tid] = slopeh*maskh[j*ne11 + k_VKQ_0 + tid];
            }

            __syncthreads();

            // When using multiple parallel sequences in llama.cpp, some KV slices can be fully masked out.
            // In such cases, skip the KV slice.
            // On AMD __all_sync would not work correctly because it assumes a warp size of 64.
#ifndef GGML_USE_HIP
            bool skip = true;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    const float2 tmp = __half22float2(((const half2 *) maskh_shared)[j*(D/2) + i]);
                    skip = skip && isinf(tmp.x) && isinf(tmp.y);
                }
            }
            if (__all_sync(0xFFFFFFFF, skip)) {
                __syncthreads();
                continue;
            }
#endif // GGML_USE_HIP
        }

        // For unknown reasons using a half array of size 1 for kqmax_new causes a performance regression,
        // see https://github.com/ggerganov/llama.cpp/pull/7061 .
        // Therefore this variable is defined twice but only used once (so that the compiler can optimize out the unused variable).
        half kqmax_new = kqmax[0];
        half kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                half sum = vec_dot_KQ(K + (k_VKQ_0 + i_KQ)*nb11, Q_h2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum((float)sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += maskh_shared[j*D + i_KQ];

                if (ncols == 1) {
                    kqmax_new        = ggml_cuda_hmax(kqmax_new,        sum);
                } else {
                    kqmax_new_arr[j] = ggml_cuda_hmax(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = ncols == 1 ? kqmax_new : kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < D; k0 += 2) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            half2 V_k;
            reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k_VKQ_0 + k0 + 0)*nb21, tid);
            reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k_VKQ_0 + k0 + 1)*nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k*KQ2[j*(D/2) + k0/2];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum((float)kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum((float)kqsum[j_VKQ]);

        half dst_val = (__low2half(VKQ[j_VKQ]) + __high2half(VKQ[j_VKQ]));
        if (gridDim.y == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        const int j_dst = (ic0 + j_VKQ)*gridDim.y + blockIdx.y;
        dst[j_dst*D*gridDim.z + D*blockIdx.z + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((ic0 + tid)*gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
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


// Paged version of the F16 vector flash attention kernel
template<int D, int ncols, ggml_type type_K_dummy, ggml_type type_V_dummy, bool use_logit_softcap> // D == head size
#ifndef GGML_USE_HIP
__launch_bounds__(D, 1) // Max threads per block is D
#endif
static __global__ void flash_attn_vec_ext_f16_paged(
        const char * __restrict__ Q_data, // Q data (contiguous)
        const paged_kv_sequence_view_gpu k_view, // Paged K view
        const paged_kv_sequence_view_gpu v_view, // Paged V view
        const char * __restrict__ mask_data, // Mask data (contiguous)
        float      * __restrict__ dst_data,  // Output
        float2     * __restrict__ dst_meta, // For fixup/metadata if stream_k or parallel_blocks > 1
        const float scale,
        const float max_bias,
        const float m0, const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        // Q dimensions
        const int q_ne0_dkq,    // Q head dim (DKQ, matches template D)
        const int q_ne1_seqlen, // Q seq len (number of queries this kernel call processes for this head)
        const int q_ne2_nhead,  // Q num heads (global index for this head in blockIdx.z)
        // const int q_ne3_batch, (assumed 1)
        // K metadata (seq_len from k_view.num_tokens_in_logical_sequence, head_dim from k_view.k_head_size_elements)
        // const int k_ne0_dkq,
        // const int k_ne1_seqlen_kv,
        const int k_ne2_nhead_kv, // Total K/V heads for GQA mapping (k_view.num_k_heads_total)
        // V metadata (similar to K)
        // const int v_ne0_dv,
        // const int v_ne1_seqlen_kv,
        // const int v_ne2_nhead_kv,
        // Mask dimensions/strides
        const int mask_ne1_qlen,  // Mask dim for k_seq_len (or broadcastable) - passed as ne31 in original
        const int mask_nb1_bytes, // Mask stride for k_seq_len dim (bytes) - passed as nb31 in original
        // Q strides (elements) - these are for the Q_data pointer
        const int q_nb1_elements, // Stride for Q's seq_len dim
        const int q_nb2_elements, // Stride for Q's num_heads dim
        // Dst strides (elements)
        const int dst_nb1_elements, // Stride for Dst's seq_len dim
        const int dst_nb2_elements  // Stride for Dst's num_heads dim
        // K/V strides are not needed as access is via paged views
) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
    // Suppress unused warnings for dummy template args if not used
    (void)type_K_dummy; (void)type_V_dummy;

    // const int D_actual = q_ne0_dkq; // Should match template D
    // const int q_seq_len_processed_per_block = ncols; // From template ncols

    // Example: Thread indices
    // const int tid_in_head_dim = threadIdx.x; // 0 to D-1 (if blockDim.x = D)
    // const int q_token_idx_in_tile = threadIdx.y; // 0 to ncols-1 (if blockDim.y = ncols)
    // const int q_head_global_idx = blockIdx.z; // Global head index

    // --- Gather K/V data for the current Q token (or Q tile element) ---
    // Vector kernels often process one Q element against all K/V elements.
    // For a given Q_i, iterate k_idx from 0 to k_view.num_tokens_in_logical_sequence:
    //   k_vec_ptr = get_paged_kv_data_ptr_cuda<half>(&k_view, k_idx, current_k_head_for_q_head, false);
    //   v_vec_ptr = get_paged_kv_data_ptr_cuda<half>(&v_view, k_idx, current_k_head_for_q_head, true);
    //   Load elements from k_vec_ptr and v_vec_ptr into registers.
    //   Perform dot product for Q_i * K_k, apply scale, mask, softmax (potentially partial), multiply by V_k.

    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        printf("SKETCH: __global__ flash_attn_vec_ext_f16_paged kernel launched. Q_head_dim %d, Q_seq_len %d, K_seq_len %d\n",
            q_ne0_dkq, q_ne1_seqlen, k_view.num_tokens_in_logical_sequence);
        printf("K_view: dtype %d, k_heads %d, k_dim %d, v_heads %d, v_dim %d, v_offset_bytes %u, elem_size %u\n",
            (int)k_view.dtype, k_view.num_k_heads_total, k_view.k_head_size_elements,
            k_view.num_v_heads_total, k_view.v_head_size_elements,
            (unsigned)k_view.v_block_start_offset_bytes, (unsigned)k_view.element_size_bytes
        );
    }
    // Suppress unused warnings
    (void)Q_data; (void)mask_data; (void)dst_data; (void)dst_meta;
    (void)scale; (void)max_bias; (void)m0; (void)m1; (void)n_head_log2; (void)logit_softcap;
    (void)q_ne0_dkq; (void)q_ne1_seqlen; (void)q_ne2_nhead; (void)k_ne2_nhead;
    (void)mask_ne1_qlen; (void)mask_nb1_bytes;
    (void)q_nb1_elements; (void)q_nb2_elements; (void)dst_nb1_elements; (void)dst_nb2_elements;
#if defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)

    // Constants for kernel behavior
    constexpr vec_dot_KQ_f16_t vec_dot_KQ_fn = get_vec_dot_KQ_f16<D>(k_view.dtype); // Get appropriate dot product function
    constexpr bool Q_is_q8_1_for_dot = k_view.dtype != GGML_TYPE_F16; // Q needs to be q8_1 if K is quantized for vec_dot_KQ
    constexpr dequantize_1_f16_t dequantize_v_fn = get_dequantize_1_f16(v_view.dtype);

    // Thread indexing: each thread processes one Q element against all K/V elements.
    // blockIdx.x corresponds to the Q element index within a head and batch.
    // blockIdx.y could be used if ncols > 1 (multiple Q elements per thread block).
    // blockIdx.z corresponds to batch_idx * num_q_heads + q_head_idx.

    const int current_q_token_idx_in_block = blockIdx.x * ncols + threadIdx.y; // If ncols > 1
    // If ncols = 1 (typical for vector path if host iterates Q sequence):
    // const int current_q_token_idx_in_block = blockIdx.x; // This is the i-th Q vector this block processes.
    // This assumes gridDim.x is q_seq_len / ncols.
    // For simplicity, let's assume ncols = 1 (host iterates Q sequence for vector kernels).
    // So, blockIdx.x is the current Q token index (0 to q_ne1_seqlen - 1).
    // And this kernel instance computes one full output vector for one Q.

    if (current_q_token_idx_in_block >= q_ne1_seqlen) {
        return;
    }

    const int q_batch_head_idx = blockIdx.z; // Global head index (batch_idx * q_ne2_nhead + head_idx)
    const int gqa_ratio = q_ne2_nhead / k_view.num_k_heads_total;
    const int actual_kv_head_idx = q_batch_head_idx / gqa_ratio;

    // Load Q vector for the current Q token and head into registers
    // Q_data points to start of Q tensor. Strides are in elements.
    // Q is F32, but kernel is F16, so Q is converted to F16 by host or needs conversion here.
    // Assuming Q is pre-converted to F16 if this is an F16 kernel, or using float for Q and half for K/V.
    // The original kernels take `const char * Q` and cast. Let's assume Q is F32 as per original vec kernels.
    const float* q_vec_ptr = (const float*)(Q_data +
                                (size_t)q_batch_head_idx * q_nb2_elements * sizeof(float) +       // Offset to current head
                                (size_t)current_q_token_idx_in_block * q_nb1_elements * sizeof(float)); // Offset to current Q token in sequence

    // For Q_q8_1 path in vec_dot_KQ, Q needs to be quantized or prepared.
    // This sketch assumes Q is F32 and K is F16 for simplicity of dot product example.
    // If K is quantized, Q must be prepared for ggml_cuda_dp4a.
    // For now, let's use a simplified F16 dot product sketch.

    half q_reg[D/2]; // Assuming D is head_dim, store as half2
    if (!Q_is_q8_1_for_dot) { // If K is F16, Q should be F16 for f16 dot product
        for(int i = threadIdx.x; i < D/2; i += blockDim.x) {
            float2 q_f2 = ((const float2*)q_vec_ptr)[i];
            q_reg[i] = make_half2(q_f2.x, q_f2.y);
        }
    } else {
        // TODO: If K is quantized, Q needs to be quantized to Q8_1 for vec_dot_KQ.
        // This involves quantizing q_vec_ptr into registers Q_i32_reg and Q_ds_reg.
        // This part is complex and omitted in this sketch.
    }

    half max_qk_val = -HALF_MAX_HALF;
    half sum_qk_exp_val = 0.0h;

    // Pass 1: Calculate max_qk and sum of exp(qk - max_qk_old)
    // This loop is over the K/V sequence length
    for (int kv_idx = 0; kv_idx < k_view.num_tokens_in_logical_sequence; ++kv_idx) {
        const half* k_head_ptr = get_paged_kv_data_ptr_cuda<half>(&k_view, kv_idx, actual_kv_head_idx, false);
        if (k_head_ptr == nullptr) continue; // Skip if page not mapped or out of bounds

        half qk_dot = 0.0h;
        if (!Q_is_q8_1_for_dot) { // Simple F16 dot F16
            for (int i = threadIdx.x; i < D/2; i += blockDim.x) { // Each thread does part of dot product
                half2 k_val_h2 = k_head_ptr[i]; // Assumes k_head_ptr is half2 aligned
                qk_dot += q_reg[i].x * k_val_h2.x + q_reg[i].y * k_val_h2.y;
            }
        } else {
            // qk_dot = vec_dot_KQ_fn((const char*)k_head_ptr, q_reg_f32_equivalent_for_vec_dot, Q_i32_reg, Q_ds_reg);
            // This needs Q to be prepared as q8_1 if K is quantized.
        }
        qk_dot = warp_reduce_sum_half(qk_dot); // Sum over threads in warp (assuming blockDim.x is warpSize)

        if (threadIdx.x == 0) { // One thread per warp updates max_qk
            if (mask_data) {
                // Mask is [seq_q, seq_k] or broadcastable. Here current_q_token_idx_in_block is q_idx, kv_idx is k_idx.
                // Mask stride mask_nb1_bytes is for q_idx.
                const half mask_val = ((const half*)(mask_data + (size_t)current_q_token_idx_in_block * mask_nb1_bytes))[kv_idx];
                qk_dot += mask_val * slope; // ALiBi slope might be 0 if max_bias is 0
            }
            if (use_logit_softcap) qk_dot = logit_softcap * tanhf(qk_dot);

            max_qk_val = max(max_qk_val, qk_dot);
        }
    }
    // Broadcast max_qk_val to all threads in warp
    max_qk_val = __shfl_sync(0xFFFFFFFF, max_qk_val, 0);

    // Pass 2: Calculate sum_exp and weighted V sum
    half out_acc_reg[D/2]; // Accumulator for output, assuming DV=D
    for(int i=0; i<D/2; ++i) out_acc_reg[i] = make_half2(0.0f, 0.0f);

    for (int kv_idx = 0; kv_idx < k_view.num_tokens_in_logical_sequence; ++kv_idx) {
        const half* k_head_ptr = get_paged_kv_data_ptr_cuda<half>(&k_view, kv_idx, actual_kv_head_idx, false);
        const half* v_head_ptr = get_paged_kv_data_ptr_cuda<half>(&v_view, kv_idx, actual_kv_head_idx, true);

        if (k_head_ptr == nullptr || v_head_ptr == nullptr) continue;

        half qk_dot = 0.0h;
        if (!Q_is_q8_1_for_dot) {
            for (int i = threadIdx.x; i < D/2; i += blockDim.x) {
                half2 k_val_h2 = k_head_ptr[i];
                qk_dot += q_reg[i].x * k_val_h2.x + q_reg[i].y * k_val_h2.y;
            }
        } // else: handle quantized K case for dot product
        qk_dot = warp_reduce_sum_half(qk_dot);

        if (threadIdx.x == 0) { // One thread per warp calculates softmax score and updates V
            if (mask_data) {
                const half mask_val = ((const half*)(mask_data + (size_t)current_q_token_idx_in_block * mask_nb1_bytes))[kv_idx];
                qk_dot += mask_val * slope;
            }
            if (use_logit_softcap) qk_dot = logit_softcap * tanhf(qk_dot);

            half softmax_score = hexp(qk_dot - max_qk_val);
            sum_qk_exp_val += softmax_score;

            // Aggregate V
            for (int i_v = 0; i_v < v_view.v_head_size_elements / 2; ++i_v) { // Iterate over V head dim (half2 elements)
                 half2 v_val_h2 = ((const half2*)v_head_ptr)[i_v]; // Assume v_head_ptr is half2 aligned
                 out_acc_reg[i_v].x += softmax_score * v_val_h2.x;
                 out_acc_reg[i_v].y += softmax_score * v_val_h2.y;
            }
        }
    }
    // Broadcast sum_qk_exp_val and normalize output accumulator
    sum_qk_exp_val = __shfl_sync(0xFFFFFFFF, sum_qk_exp_val, 0);
    if (sum_qk_exp_val == 0.0h) sum_qk_exp_val = 1.0h; // Avoid division by zero

    float* dst_float_ptr = (float*)(dst_data +
                              (size_t)q_batch_head_idx * dst_nb2_elements * sizeof(float) +
                              (size_t)current_q_token_idx_in_block * dst_nb1_elements * sizeof(float));

    for (int i = threadIdx.x; i < D/2; i += blockDim.x) {
        half2 final_val_h2;
        final_val_h2.x = out_acc_reg[i].x / sum_qk_exp_val;
        final_val_h2.y = out_acc_reg[i].y / sum_qk_exp_val;
        // Output is F32
        float2 final_val_f2 = __half22float2(final_val_h2);
        ((float2*)dst_float_ptr)[i] = final_val_f2;
    }

    // Suppress unused warnings for a more complete parameter list that launch_fattn_paged expects
    (void)q_ne0_dkq; (void)dst_meta; (void)k_ne2_nhead; (void)ncols;
#else
    // Original NO_DEVICE_CODE and unused parameter list
#endif
}


template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_f16_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = D/WARP_SIZE;


template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_f16_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = D/WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f16<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    constexpr bool need_f16_K = D != 128;
    constexpr bool need_f16_V = D != 128 && D != 64;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (Q->ne[1] == 1 || GGML_CUDA_CC_IS_NVIDIA(cc)) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block = 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 8;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_F16_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f16_case                     \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);
