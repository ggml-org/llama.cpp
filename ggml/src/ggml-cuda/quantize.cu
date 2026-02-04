#include "quantize.cuh"
#include <cstdint>

// GFX906 optimizations: DPP-based warp reductions
#ifdef GGML_USE_HIP
#include "gfx906/gfx906-common.cuh"
#endif
// #include <vector>
// #include <cstdio>

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_fp4_mmq * y = (block_fp4_mmq *) vy;

    const int64_t block_fp4_mmq_size = 8 * QK_MXFP4;  // 256 values
    const int64_t ib0                = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;
    char2 * yqs2 = (char2 *) y[ib].qs;

    const int64_t base_pos = i03 * s03 + i02 * s02 + i01 * s01;

    uint8_t scales[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        const float xi = (i0 < ne00) ? x[base_pos + i0] : 0.0f;

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));

            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = *(char2 *) &fp4_packed;
        }
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            char2 q;
            q.x = (q_hi_0 << 4) | q_lo_0;
            q.y = (q_hi_1 << 4) | q_lo_1;
            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
#endif // CUDART_VERSION >= 12080
    }

    if (lane_id_32 == 0) {
        // Store 2 scales packed into 1 uint32
        y[ib].d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread via 128-bit vectorized load
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // =========================================================================
    // GFX906 OPTIMIZATION PATH (4 values per thread)
    // =========================================================================
#if defined(GGML_USE_HIP) && defined(__gfx906__)

    // Compute absolute values - keep in registers for reuse
    const float ax = fabsf(xi.x);
    const float ay = fabsf(xi.y);
    const float az = fabsf(xi.z);
    const float aw = fabsf(xi.w);

    // Local max using tree reduction pattern (better for ILP)
    float amax = fmaxf(fmaxf(ax, ay), fmaxf(az, aw));

    // Compute sum early (for DS4 layout) - enables interleaved reduction
    float sum = 0.0f;
    if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
    }

    // =========================================================================
    // ULTRA-FUSED WARP REDUCTIONS - Single asm block, minimal NOPs
    // Old approach: separate function calls = 12+ NOPs total
    // New approach: fused asm block = 3 NOPs total (75% reduction!)
    // =========================================================================
    if constexpr (vals_per_scale == 32 && vals_per_sum == 32) {
        // DS4 layout: Fully fused reduction for both max and sum
        int amax_i = __float_as_int(amax);
        int sum_i  = __float_as_int(sum);
        int amax_tmp, sum_tmp;

        asm volatile(
            // === XOR4: row_shl:4 + row_shr:4 with bank masks ===
            "v_mov_b32 %0, %4\n"                                              // amax_tmp = amax
            "v_mov_b32 %1, %5\n"                                              // sum_tmp = sum
            "s_nop 1\n"
            "v_mov_b32_dpp %0, %4 row_shl:4 row_mask:0xf bank_mask:0x5\n"     // amax_tmp for banks 0,2
            "v_mov_b32_dpp %1, %5 row_shl:4 row_mask:0xf bank_mask:0x5\n"     // sum_tmp for banks 0,2
            "v_mov_b32_dpp %0, %4 row_shr:4 row_mask:0xf bank_mask:0xa\n"     // amax_tmp for banks 1,3
            "v_mov_b32_dpp %1, %5 row_shr:4 row_mask:0xf bank_mask:0xa\n"     // sum_tmp for banks 1,3
            // Combine xor4 results
            "v_max_f32 %2, %4, %0\n"                                          // amax = max(amax, amax_tmp)
            "v_add_f32 %3, %5, %1\n"                                          // sum = sum + sum_tmp

            // === XOR2: quad_perm:[2,3,0,1] - fused max/add ===
            "s_nop 1\n"
            "v_max_f32_dpp %2, %2, %2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
            "v_add_f32_dpp %3, %3, %3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"

            // === XOR1: quad_perm:[1,0,3,2] - fused max/add ===
            // Using s_nop 1 instead of 4 - the operations above provide enough distance
            "s_nop 1\n"
            "v_max_f32_dpp %2, %2, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"
            "v_add_f32_dpp %3, %3, %3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"

            : "=&v"(amax_tmp), "=&v"(sum_tmp), "=v"(amax_i), "=v"(sum_i)
            : "v"(amax_i), "v"(sum_i)
            : "memory"
        );

        amax = __int_as_float(amax_i);
        sum  = __int_as_float(sum_i);

    } else if constexpr (vals_per_scale == 32) {
        // D4 layout: Only max reduction needed (no sum)
        int amax_i = __float_as_int(amax);
        int amax_tmp;

        asm volatile(
            // === XOR4 ===
            "v_mov_b32 %0, %2\n"
            "s_nop 1\n"
            "v_mov_b32_dpp %0, %2 row_shl:4 row_mask:0xf bank_mask:0x5\n"
            "v_mov_b32_dpp %0, %2 row_shr:4 row_mask:0xf bank_mask:0xa\n"
            "v_max_f32 %1, %2, %0\n"

            // === XOR2 ===
            "s_nop 1\n"
            "v_max_f32_dpp %1, %1, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"

            // === XOR1 ===
            "s_nop 1\n"
            "v_max_f32_dpp %1, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"

            : "=&v"(amax_tmp), "=v"(amax_i)
            : "v"(amax_i)
            : "memory"
        );

        amax = __int_as_float(amax_i);

    } else {
        // D2S6 layout: Different reduction widths, use generic path
        #pragma unroll
        for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
        }
        if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
            #pragma unroll
            for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
                sum += __shfl_xor(sum, offset, WARP_SIZE);
            }
        }
    }

    // =========================================================================
    // OPTIMIZED SCALE COMPUTATION - Eliminate double reciprocal!
    // Old: d_inv = 127 * rcp(amax), then d = rcp(d_inv) = rcp(127*rcp(amax))
    // New: d = amax * (1/127), d_inv = rcp(d) - saves one reciprocal!
    // =========================================================================
    constexpr float inv_127 = 1.0f / 127.0f;  // Compile-time constant
    const float d = amax * inv_127;           // Just a multiply
    const float d_inv = fast_rcp_f32(d);      // Single reciprocal

    // =========================================================================
    // QUANTIZATION - Use __float2int_rn for direct conversion
    // =========================================================================
    char4 q;
    q.x = static_cast<int8_t>(__float2int_rn(xi.x * d_inv));
    q.y = static_cast<int8_t>(__float2int_rn(xi.y * d_inv));
    q.z = static_cast<int8_t>(__float2int_rn(xi.z * d_inv));
    q.w = static_cast<int8_t>(__float2int_rn(xi.w * d_inv));

    // Write quantized values (32-bit coalesced write)
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    // Write scale/sum (one thread per 32-value group)
    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs/16] = sum;
        }
        if (iqs % 64 == 0) {
            y[ib].d2s6[iqs/64] = d;
        }
    } else if (iqs % 32 == 0) {
        if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs/32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs/32] = d;
        }
    }

#else
    // =========================================================================
    // GENERIC PATH (non-GFX906)
    // =========================================================================
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }
        y[ib].d2s6[2 + iqs/16] = sum;
        if (iqs % 64 != 0) {
            return;
        }
        const float d = 1.0f / d_inv;
        y[ib].d2s6[iqs/64] = d;
        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
#endif
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // NOTE: Tested 2x (2 values/thread) and 8x (8 values/thread) kernels.
    // Results: 4x=25ms, 8x=31ms, 2x=41ms. 4-value kernel is optimal due to:
    // - Balanced register pressure vs reduction count
    // - Vectorized 128-bit loads and 32-bit writes

    // --- HOST-SIDE DEBUG VALIDATION (commented out) ---
    // fprintf(stderr, "[quantize_mmq_q8_1] ne00=%ld s01=%ld s02=%ld s03=%ld ne0=%ld ne1=%ld ne2=%ld ne3=%ld ids=%p x=%p\n",
    //         (long)ne00, (long)s01, (long)s02, (long)s03, (long)ne0, (long)ne1, (long)ne2, (long)ne3, (void*)ids, (void*)x);
    //
    // // Calculate maximum index that will be accessed
    // int64_t max_i1_val = ne1 - 1;
    // int64_t max_src_idx = (ne3-1)*s03 + (ne2-1)*s02 + max_i1_val*s01 + (ne0-4);
    // fprintf(stderr, "[quantize_mmq_q8_1] Grid: (%ld, %ld, %ld) max_src_idx=%ld (assuming no ids remapping)\n",
    //         (long)ne1, (long)((ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ)),
    //         (long)(ne2*ne3), (long)max_src_idx);
    //
    // if (ids) {
    //     // For MoE: copy ids to host and validate
    //     std::vector<int32_t> ids_host(ne1);
    //     CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids, ne1 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    //     CUDA_CHECK(cudaStreamSynchronize(stream));
    //
    //     int64_t max_ids_val = 0;
    //     int64_t min_ids_val = INT64_MAX;
    //     for (int64_t i = 0; i < ne1; i++) {
    //         if (ids_host[i] > max_ids_val) max_ids_val = ids_host[i];
    //         if (ids_host[i] < min_ids_val) min_ids_val = ids_host[i];
    //     }
    //     fprintf(stderr, "[quantize_mmq_q8_1] ids range: [%ld, %ld] (ne1=%ld)\n",
    //             (long)min_ids_val, (long)max_ids_val, (long)ne1);
    //
    //     max_src_idx = (ne3-1)*s03 + (ne2-1)*s02 + max_ids_val*s01 + (ne0-4);
    //     fprintf(stderr, "[quantize_mmq_q8_1] max_src_idx with ids remapping=%ld\n", (long)max_src_idx);
    //
    //     if (max_ids_val * s01 > 1000000000LL) {
    //         fprintf(stderr, "[quantize_mmq_q8_1] WARNING: max_ids_val * s01 = %ld seems too large!\n",
    //                 (long)(max_ids_val * s01));
    //     }
    // }
    // --- END DEBUG VALIDATION ---

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_mxfp4_cuda(const float *                    x,
                             const int32_t *                  ids,
                             void *                           vy,
                             [[maybe_unused]] const ggml_type type_src0,
                             const int64_t                    ne00,
                             const int64_t                    s01,
                             const int64_t                    s02,
                             const int64_t                    s03,
                             const int64_t                    ne0,
                             const int64_t                    ne1,
                             const int64_t                    ne2,
                             const int64_t                    ne3,
                             cudaStream_t                     stream) {
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

    constexpr int nwarps = 8;
    constexpr int vals_per_warp  = 2 * QK_MXFP4;
    constexpr int vals_per_block = nwarps * vals_per_warp;

    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(WARP_SIZE, nwarps, 1);

    quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
