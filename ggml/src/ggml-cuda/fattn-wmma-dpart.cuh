#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "mma.cuh"

// FlashAttention for small GQA batches on RDNA3 WMMA with the head dimension split across the warps ("d-partitioned"),
// e.g. the speculative-decoding verify of a GQA-6 model: 5 tokens x 6 Q heads = 30 rows per KV head.
//
// For a GQA ratio that is not a power of 2 the other kernels can use only a small divisor of the ratio as ncols2, so
// several blocks stream the same K/V and the tile kernel keeps VKQ in f16. This kernel instead:
//   - packs all gqa_ratio*ne01 <= 32 Q rows of one KV head into two 16-row tiles (row r = token r/gqa_ratio,
//     head r%gqa_ratio), so one block reads each K/V byte of its KV range exactly once;
//   - KQ: warp w keeps the f16 Q fragments of its d slice [w*DKQ/nwarps, (w+1)*DKQ/nwarps) in registers, multiplies
//     them with only that slice of each K row and writes a partial KQ (rows x keys, f32) to its own SRAM slab; the
//     softmax adds the nwarps partials (f32), applies the mask and the online softmax in f32 and stores P as f16;
//   - VKQ: warp w owns the output slice [w*DV/nwarps, (w+1)*DV/nwarps). It loads its slice of the iteration's V rows
//     with contiguous 16-byte loads at the top of the iteration (they are in flight during the KQ phase), writes them
//     transposed into its own slab once the softmax has consumed the partial KQ there, and accumulates VKQ in f32
//     (v_wmma_f32_16x16x16_f16);
//   - the KV dimension is split over gridDim.y blocks (interleaved nbatch_fa chunks, as in the tile kernel), the partial
//     results are combined in f32 by flash_attn_combine_results (launch_fattn without stream-k).
// 23 KB of SRAM per block (no Q staging) and <= 256 VGPRs without spills: 5 blocks per WGP on gfx1100.

static constexpr int ggml_cuda_fattn_wmma_dpart_nwarps    =  4;
static constexpr int ggml_cuda_fattn_wmma_dpart_nrows     = 32; // packed Q rows per block: 2 tiles of 16
static constexpr int ggml_cuda_fattn_wmma_dpart_nbatch_fa = 32; // KV rows per iteration: 2 tiles of 16

template <int DKQ, int DV>
__launch_bounds__(ggml_cuda_fattn_wmma_dpart_nwarps*32, 1)
static __global__ void flash_attn_ext_wmma_dpart(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        const int  * __restrict__ KV_rows,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    using namespace ggml_cuda_mma;
    // A/B operand: 16 rows x 16 halves, lane l holds row l % 16 (the two halves of the wave hold the same data).
    typedef tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_AB;
    // Result: mma(C, A, B) accumulates C(i, j) += sum_k B(i, k)*A(j, k); lane l holds i = l % 16, j = 2*e + l/16 (e < 8).
    typedef tile<16, 16, float, DATA_LAYOUT_I_MAJOR>          tile_C;

    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_wmma_dpart_nwarps;
    constexpr int nrows     = ggml_cuda_fattn_wmma_dpart_nrows;
    constexpr int nbatch_fa = ggml_cuda_fattn_wmma_dpart_nbatch_fa;
    constexpr int DKQ_w     = DKQ/nwarps;    // KQ: d slice of a warp
    constexpr int DV_w      = DV/nwarps;     // VKQ: output slice of a warp
    constexpr int stride_S  = nbatch_fa + 8; // floats per row of a partial KQ slab, also halves per row of the V^T slice
    constexpr int stride_P  = nbatch_fa + 8; // halves per P row
    constexpr int kps       = nbatch_fa/4;   // softmax: 4 threads per row, kps keys per thread
    static_assert(DKQ_w % 16 == 0 && DV_w % 16 == 0, "bad head size");
    static_assert(nrows == 2*16 && nbatch_fa == 2*16 && nrows == nwarps*warp_size/4, "bad tile config");
    static_assert(DV_w == 8*8 && nbatch_fa == 4*8 && warp_size == 8*4, "the V loads assume 8 x 8 halves per lane");
    static_assert(DV_w*stride_S <= 2*nrows*stride_S, "the V^T slice does not fit into the slab");
    static_assert(kps == 8, "the softmax loads assume 8 keys per thread");

    // not supported by this kernel (host gate): sinks, ALiBi, logit softcap, KV_max (mask scan), KV_rows, more than one sequence
    GGML_UNUSED_VARS(sinks, KV_max, KV_rows, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, nb03, ne10, ne13, nb13, nb23,
        ne31, ne32, ne33, nb32, nb33);

    const int w    = threadIdx.y;
    const int lane = threadIdx.x;

    // One block per (KV head, KV split); ne03 == 1 and gqa_ratio*ne01 <= nrows (host gate).
    const int gqa_ratio = ne02 / ne12;
    const int nrows_used = int(ne01.z)*gqa_ratio;
    const int head0      = blockIdx.z*gqa_ratio;

    const half2 * K_h2  = (const half2 *) (K + int64_t(blockIdx.z)*nb12);
    const half2 * V_h2  = (const half2 *) (V + int64_t(blockIdx.z)*nb22);
    const half  * maskh = (const half  *) mask;
    const int stride_K2   = nb11 / sizeof(half2);
    const int stride_V2   = nb21 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    // S_sh, per warp: the partial KQ of its d slice [row][key] (f32); after the softmax the V^T of its output slice [d][key] (f16).
    __shared__ __align__(16) float S_sh[nwarps*nrows*stride_S];
    __shared__ __align__(16) half  P_sh[nrows*stride_P];
    __shared__ float               r_sh[nrows]; // per row: KQ_max_scale of the iteration; at the end the output scale
    float * S_w  = S_sh + w*(nrows*stride_S);
    half  * VT_w = (half *) S_w;

    // Q -> f16 (* scale) in registers: the fragments of this warp's d slice for both row tiles. Rows past nrows_used are 0.
    tile_AB Q_A[2][DKQ_w/16];
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
        const int  r     = 16*rt + tile_AB::get_i(0);
        const bool valid = r < nrows_used;
        const int  t     = valid ? r / gqa_ratio : 0;
        const int  h     = head0 + (valid ? r % gqa_ratio : 0);
        const float * Q_f = (const float *) (Q + int64_t(t)*nb01 + int64_t(h)*nb02) + w*DKQ_w;
#pragma unroll
        for (int c = 0; c < DKQ_w/16; ++c) {
            __align__(16) float tmp[16];
#pragma unroll
            for (int i = 0; i < 16; i += 4) {
                ggml_cuda_memcpy_1<16>(tmp + i, Q_f + 16*c + i);
            }
#pragma unroll
            for (int l = 0; l < tile_AB::ne; ++l) {
                Q_A[rt][c].x[l] = valid ? make_half2(tmp[2*l + 0]*scale, tmp[2*l + 1]*scale) : make_half2(0.0f, 0.0f);
            }
        }
    }

    // Softmax: thread (w, lane) owns row r_sm, keys k_sm .. k_sm + kps-1 of each iteration (4 threads per row, in one warp).
    const int  r_sm       = w*(nrows/nwarps) + lane/4;
    const int  k_sm       = (lane % 4)*kps;
    const bool r_sm_valid = r_sm < nrows_used;
    const half * mask_row = maskh + int64_t(min(r_sm / gqa_ratio, int(ne01.z) - 1))*stride_mask + k_sm;
    float KQ_max = -FLT_MAX/2.0f;
    float KQ_sum = 0.0f;

    tile_C VKQ_C[2][DV_w/16]; // [row tile][d chunk]: rows i, d j

    // V: lane holds the 8 halves [dv_v, dv_v + 8) of the rows kv_v .. kv_v + 7 of an iteration (8 x 16-byte loads).
    const int kv_v = 8*(lane / 8);
    const int dv_v = w*DV_w + 8*(lane % 8);
    half2 V_r[8][4];

    const int k_VKQ_step = gridDim.y*nbatch_fa;
    int       k_VKQ_0    = blockIdx.y*nbatch_fa; // < ne11: launch_fattn caps gridDim.y at the number of KV tiles
    for (; k_VKQ_0 < ne11; k_VKQ_0 += k_VKQ_step) {
        // The mask row first, as one 16-byte load (every stride is a multiple of 16 bytes: gqa_opt), kept ahead of the K / V loads.
        __align__(16) half mask_r[kps];
        int4 mask_i4 = *(const int4 *) (mask_row + k_VKQ_0);
        __builtin_amdgcn_sched_barrier(0);

        // V rows of this iteration: in flight during the KQ phase, consumed after the softmax.
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            ggml_cuda_memcpy_1<16>(V_r[t], V_h2 + int64_t(k_VKQ_0 + kv_v + t)*stride_V2 + dv_v/2);
        }

        // ---- partial KQ over this warp's d slice, one key tile at a time: [row tile]
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            tile_C KQ_C[2];
#pragma unroll
            for (int c = 0; c < DKQ_w/16; ++c) {
                tile_AB K_B;
                load_ldmatrix(K_B, K_h2 + int64_t(k_VKQ_0 + 16*kt)*stride_K2 + (w*DKQ_w + 16*c)/2, stride_K2);
#pragma unroll
                for (int rt = 0; rt < 2; ++rt) {
                    mma(KQ_C[rt], Q_A[rt][c], K_B); // (i = key, j = row) += K(key, d) * Q(row, d)
                }
            }
#pragma unroll
            for (int rt = 0; rt < 2; ++rt) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    S_w[(16*rt + tile_C::get_j(l))*stride_S + 16*kt + tile_C::get_i(l)] = KQ_C[rt].x[l];
                }
            }
        }

        // The mask is consumed only here, after the KQ phase, so its round trip runs under the K waits.
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("" : "+v"(mask_i4.x), "+v"(mask_i4.y), "+v"(mask_i4.z), "+v"(mask_i4.w));
        memcpy(mask_r, &mask_i4, sizeof(mask_r));
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();

        // ---- KQ = sum of the nwarps partials (in warp order), mask, online softmax in f32, P in f16
        __align__(16) float KQ[kps];
        ggml_cuda_memcpy_1<16>(KQ + 0, S_sh + r_sm*stride_S + k_sm + 0);
        ggml_cuda_memcpy_1<16>(KQ + 4, S_sh + r_sm*stride_S + k_sm + 4);
#pragma unroll
        for (int q = 1; q < nwarps; ++q) {
            __align__(16) float tmp[kps];
            ggml_cuda_memcpy_1<16>(tmp + 0, S_sh + q*(nrows*stride_S) + r_sm*stride_S + k_sm + 0);
            ggml_cuda_memcpy_1<16>(tmp + 4, S_sh + q*(nrows*stride_S) + r_sm*stride_S + k_sm + 4);
#pragma unroll
            for (int k = 0; k < kps; ++k) {
                KQ[k] += tmp[k];
            }
        }
        float KQ_max_new = KQ_max;
#pragma unroll
        for (int k = 0; k < kps; ++k) {
            KQ[k] = r_sm_valid ? KQ[k] + __half2float(mask_r[k]) : -INFINITY;
            KQ_max_new = fmaxf(KQ_max_new, KQ[k] + FATTN_KQ_MAX_OFFSET);
        }
        KQ_max_new = warp_reduce_max<4>(KQ_max_new);
        const float KQ_max_scale = expf(KQ_max - KQ_max_new);
        KQ_max = KQ_max_new;

        __align__(16) half P[kps];
        float KQ_sum_add = 0.0f;
#pragma unroll
        for (int k = 0; k < kps; ++k) {
            P[k] = __float2half(expf(KQ[k] - KQ_max));
            KQ_sum_add += __half2float(P[k]); // the sum of the f16 values that VKQ uses
        }
        KQ_sum_add = warp_reduce_sum<4>(KQ_sum_add);
        KQ_sum = KQ_sum*KQ_max_scale + KQ_sum_add;
        ggml_cuda_memcpy_1<16>(P_sh + r_sm*stride_P + k_sm, P);
        if (lane % 4 == 0) {
            r_sh[r_sm] = KQ_max_scale;
        }
        __syncthreads();

        // ---- V^T of this warp's output slice into its own slab (no other warp reads it any more in this iteration)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            __align__(16) half tmp[8];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                tmp[t] = i % 2 == 0 ? __low2half(V_r[t][i/2]) : __high2half(V_r[t][i/2]);
            }
            ggml_cuda_memcpy_1<16>(VT_w + (8*(lane % 8) + i)*stride_S + kv_v, tmp);
        }
        // The V^T stores and the loads below are by the same warp: order them (__syncwarp in HIP).
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        // ---- VKQ = VKQ*KQ_max_scale + P V, f32 accumulators
#pragma unroll
        for (int rt = 0; rt < 2; ++rt) {
            const float s = r_sh[16*rt + tile_C::get_i(0)];
#pragma unroll
            for (int c = 0; c < DV_w/16; ++c) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    VKQ_C[rt][c].x[l] *= s;
                }
            }
        }
#pragma unroll
        for (int ks = 0; ks < nbatch_fa/16; ++ks) {
            tile_AB P_B[2];
#pragma unroll
            for (int rt = 0; rt < 2; ++rt) {
                load_ldmatrix(P_B[rt], (const half2 *) P_sh + 16*rt*(stride_P/2) + ks*8, stride_P/2);
            }
#pragma unroll
            for (int c = 0; c < DV_w/16; ++c) {
                tile_AB VT_A;
                load_ldmatrix(VT_A, (const half2 *) VT_w + 16*c*(stride_S/2) + ks*8, stride_S/2);
#pragma unroll
                for (int rt = 0; rt < 2; ++rt) {
                    mma(VKQ_C[rt][c], VT_A, P_B[rt]); // (i = row, j = d) += P(row, key) * V(key, d)
                }
            }
        }
    }

    // ---- write back: the final result (gridDim.y == 1) or the partial result + (max, rowsum) for flash_attn_combine_results
    __syncthreads(); // every warp is done with r_sh
    if (lane % 4 == 0 && r_sm_valid) {
        r_sh[r_sm] = gridDim.y == 1 ? 1.0f/KQ_sum : 1.0f;
    }
    __syncthreads();
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
        const int r = 16*rt + tile_C::get_i(0);
        if (r < nrows_used) {
            const int    j_dst_unrolled = ((r / gqa_ratio)*ne02 + head0 + r % gqa_ratio)*int(gridDim.y) + int(blockIdx.y);
            const float  s      = r_sh[r];
            float      * dst_r  = dst + int64_t(j_dst_unrolled)*DV + w*DV_w;
#pragma unroll
            for (int c = 0; c < DV_w/16; ++c) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    dst_r[16*c + tile_C::get_j(l)] = VKQ_C[rt][c].x[l]*s;
                }
            }
        }
    }
    if (gridDim.y != 1 && lane % 4 == 0 && r_sm_valid) {
        const int j_dst_unrolled = ((r_sm / gqa_ratio)*ne02 + head0 + r_sm % gqa_ratio)*int(gridDim.y) + int(blockIdx.y);
        dst_meta[j_dst_unrolled] = make_float2(KQ_max, KQ_sum);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, KV_rows, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
}

bool ggml_cuda_fattn_wmma_dpart_supported(int cc, const ggml_tensor * dst, bool gqa_opt);
void ggml_cuda_flash_attn_ext_wmma_dpart(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
