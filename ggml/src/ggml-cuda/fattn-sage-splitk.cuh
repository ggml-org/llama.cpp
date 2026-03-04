#pragma once

// Sage Split-K / Tile Flash Attention for gfx906 (MI50)
// Decode: Split-K parallelism (1 Q row, split sequence into chunks)
// Prefill: Q8 tile kernel (TILE_Q Q rows, iterate K tiles with sdot4)
// Supports D=128 and D=256

#include "common.cuh"
#include "fattn-common.cuh"

#define SAGE_CHUNK 128
#define Q8_BLK_SIZE 34
#define Q4_BLK_SIZE 18

// === gfx906 intrinsics ===
static __device__ __forceinline__ int sage_sdot4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}
static __device__ __forceinline__ int sage_sdot8(int a, int b, int c) {
    return __builtin_amdgcn_sdot8(a, b, c, false);
}
static __device__ __forceinline__ float sage_fast_exp(float x) {
    float r;
    asm volatile("v_exp_f32 %0, %1" : "=v"(r) : "v"(x * 1.4426950408889634f));
    return r;
}
static __device__ __forceinline__ float sage_fast_rcp(float x) {
    float r;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

// ================== FP16 Split-K Decode Kernel ==================
// Block: D threads (one per dimension element)
template <int D>
static __global__ void sage_splitk_fp16_kernel(
    const float * __restrict__ Q,
    const char  * __restrict__ K_data,
    const char  * __restrict__ V_data,
    float       * __restrict__ partial_out,
    float       * __restrict__ chunk_meta,
    const half  * __restrict__ mask_data,
    const int S, const int H, const int Hkv, const int nchunks,
    const size_t nb11, const size_t nb12, const size_t nb13,
    const size_t nb21, const size_t nb22, const size_t nb23,
    const size_t mask_stride,
    const int seq_idx,
    const float scale,
    const bool v_is_q8
) {
    const int h = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave_id = tid / 64;
    const int lane_id = tid % 64;
    const int n_waves = D / 64;
    const int hkv = h * Hkv / H;

    const int chunk_start = chunk_id * SAGE_CHUNK;
    const int chunk_end   = min(chunk_start + SAGE_CHUNK, S);
    const int chunk_len   = chunk_end - chunk_start;

    const char * K_base = K_data + seq_idx * nb13 + hkv * nb12;
    const char * V_base = V_data + seq_idx * nb23 + hkv * nb22;

    // Each thread loads one pair of fp16 values from Q
    float q0 = Q[lane_id * 2 + 0 + h * D];
    float q1 = Q[lane_id * 2 + 1 + h * D];
    // For D=256: wave 0 handles dims 0..127, wave 1 handles 128..255, etc.
    // Each wave of 64 threads covers 128 dims (2 per thread)
    float q0_w, q1_w;
    if constexpr (D > 128) {
        q0_w = Q[wave_id * 128 + lane_id * 2 + 0 + h * D];
        q1_w = Q[wave_id * 128 + lane_id * 2 + 1 + h * D];
    }

    __shared__ float s_scores[SAGE_CHUNK];

    // Compute QK dot products - each wave covers part of D, then reduce across waves
    for (int ki = 0; ki < chunk_len; ki++) {
        const int s = chunk_start + ki;
        const half * k_row = (const half *)(K_base + (size_t)s * nb11);

        float dot;
        if constexpr (D == 128) {
            // Original: 2 waves take turns on K entries
            if (ki % n_waves != wave_id) continue;
            dot = q0 * __half2float(k_row[lane_id * 2 + 0])
                + q1 * __half2float(k_row[lane_id * 2 + 1]);
        } else {
            // D>128: each wave does its portion, reduce across waves
            dot = q0_w * __half2float(k_row[wave_id * 128 + lane_id * 2 + 0])
                + q1_w * __half2float(k_row[wave_id * 128 + lane_id * 2 + 1]);
        }

        // Intra-wave reduction
        for (int offset = 32; offset > 0; offset >>= 1)
            dot += __shfl_xor(dot, offset);

        if constexpr (D == 128) {
            if (lane_id == 0) {
                float score = dot * scale;
                if (mask_data) score += __half2float(mask_data[s]);
                s_scores[ki] = score;
            }
        } else {
            // Cross-wave reduction via shared memory
            __shared__ float s_wave_dots[4]; // max 4 waves
            if (lane_id == 0) s_wave_dots[wave_id] = dot;
            __syncthreads();
            if (tid == 0) {
                float total = 0;
                for (int w = 0; w < n_waves; w++) total += s_wave_dots[w];
                float score = total * scale;
                if (mask_data) score += __half2float(mask_data[s]);
                s_scores[ki] = score;
            }
            __syncthreads();
        }
    }
    __syncthreads();

    __shared__ float s_max, s_sum;
    if (tid == 0) {
        float m = -1e30f;
        for (int i = 0; i < chunk_len; i++) m = fmaxf(m, s_scores[i]);
        s_max = m;
        float sum = 0.0f;
        for (int i = 0; i < chunk_len; i++) {
            s_scores[i] = expf(s_scores[i] - m);
            sum += s_scores[i];
        }
        s_sum = sum;
    }
    __syncthreads();

    float acc = 0.0f;
    if (v_is_q8) {
        int vb_idx = tid / 32;
        int vb_off = tid % 32;
        for (int si = 0; si < chunk_len; si++) {
            const int s = chunk_start + si;
            const char * v_row = V_base + (size_t)s * nb21;
            const char * vblk = v_row + vb_idx * Q8_BLK_SIZE;
            half vd; memcpy(&vd, vblk, 2);
            float v_scale = __half2float(vd);
            int8_t v_q = *((const int8_t *)(vblk + 2) + vb_off);
            acc += s_scores[si] * ((float)v_q * v_scale);
        }
    } else {
        for (int si = 0; si < chunk_len; si++) {
            const int s = chunk_start + si;
            const half * v_row = (const half *)(V_base + (size_t)s * nb21);
            acc += s_scores[si] * __half2float(v_row[tid]);
        }
    }

    partial_out[(h * nchunks + chunk_id) * D + tid] = acc;
    if (tid == 0) {
        chunk_meta[(h * nchunks + chunk_id) * 2 + 0] = s_max;
        chunk_meta[(h * nchunks + chunk_id) * 2 + 1] = s_sum;
    }
}

// ================== Q8_0 Split-K Decode Kernel ==================
template <int D>
static __global__ __launch_bounds__(256, 2) void sage_splitk_q8_kernel(
    const float * __restrict__ Q,
    const char  * __restrict__ K_data,
    const char  * __restrict__ V_data,
    float       * __restrict__ partial_out,
    float       * __restrict__ chunk_meta,
    const half  * __restrict__ mask_data,
    const int S, const int H, const int Hkv, const int nchunks,
    const size_t nb11, const size_t nb12, const size_t nb13,
    const size_t nb21, const size_t nb22, const size_t nb23,
    const size_t mask_stride,
    const int seq_idx,
    const float scale,
    const bool v_is_q8
) {
    constexpr int blocks_per_row = D / 32;
    const int h = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int tid = threadIdx.x;
    const int hkv = h * Hkv / H;

    const int chunk_start = chunk_id * SAGE_CHUNK;
    const int chunk_end   = min(chunk_start + SAGE_CHUNK, S);
    const int chunk_len   = chunk_end - chunk_start;

    const char * K_base = K_data + seq_idx * nb13 + hkv * nb12;
    const char * V_base = V_data + seq_idx * nb23 + hkv * nb22;

    // Quantize Q to int8
    float q_val = Q[tid + h * D];
    float q_abs = fabsf(q_val);

    constexpr int n_waves = D / 64;
    __shared__ float s_amax[n_waves];
    float warp_max = q_abs;
    for (int offset = 32; offset > 0; offset >>= 1)
        warp_max = fmaxf(warp_max, __shfl_xor(warp_max, offset));
    if (tid % 64 == 0) s_amax[tid / 64] = warp_max;
    __syncthreads();
    float amax = s_amax[0];
    for (int w = 1; w < n_waves; w++) amax = fmaxf(amax, s_amax[w]);
    float q_scale = amax / 127.0f;
    float q_inv = (q_scale > 0.0f) ? 1.0f / q_scale : 0.0f;
    int8_t q_int = (int8_t)max(-127, min(127, (int)roundf(q_val * q_inv)));

    __shared__ int q_packed[D / 4];
    __shared__ int8_t q_bytes[D];
    q_bytes[tid] = q_int;
    __syncthreads();
    if (tid < D / 4) {
        int val;
        memcpy(&val, &q_bytes[tid * 4], 4);
        q_packed[tid] = val;
    }
    __syncthreads();

    __shared__ float s_scores[SAGE_CHUNK];

    for (int ki = tid; ki < chunk_len; ki += D) {
        const int s = chunk_start + ki;
        const char * k_row = K_base + (size_t)s * nb11;

        float dot = 0.0f;
        #pragma unroll 4
        for (int b = 0; b < blocks_per_row; b++) {
            const char * blk = k_row + b * Q8_BLK_SIZE;
            half k_d;
            memcpy(&k_d, blk, 2);
            float k_scale = __half2float(k_d);

            const int8_t * qs = (const int8_t *)(blk + 2);
            int acc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_packed;
                memcpy(&k_packed, &qs[p * 4], 4);
                acc = sage_sdot4(q_packed[b * 8 + p], k_packed, acc);
            }
            dot += (float)acc * k_scale;
        }
        float score = dot * q_scale * scale;
        if (mask_data) score += __half2float(mask_data[s]);
        s_scores[ki] = score;
    }
    __syncthreads();

    __shared__ float s_max, s_sum;
    if (tid == 0) {
        float m = -1e30f;
        for (int i = 0; i < chunk_len; i++) m = fmaxf(m, s_scores[i]);
        s_max = m;
        float sum = 0.0f;
        for (int i = 0; i < chunk_len; i++) {
            s_scores[i] = expf(s_scores[i] - m);
            sum += s_scores[i];
        }
        s_sum = sum;
    }
    __syncthreads();

    float acc = 0.0f;
    if (v_is_q8) {
        int vb_idx = tid / 32;
        int vb_off = tid % 32;
        for (int si = 0; si < chunk_len; si++) {
            const int s = chunk_start + si;
            const char * v_row = V_base + (size_t)s * nb21;
            const char * vblk = v_row + vb_idx * Q8_BLK_SIZE;
            half vd; memcpy(&vd, vblk, 2);
            float v_scale = __half2float(vd);
            int8_t v_q = *((const int8_t *)(vblk + 2) + vb_off);
            acc += s_scores[si] * ((float)v_q * v_scale);
        }
    } else {
        for (int si = 0; si < chunk_len; si++) {
            const int s = chunk_start + si;
            const half * v_row = (const half *)(V_base + (size_t)s * nb21);
            acc += s_scores[si] * __half2float(v_row[tid]);
        }
    }

    partial_out[(h * nchunks + chunk_id) * D + tid] = acc;
    if (tid == 0) {
        chunk_meta[(h * nchunks + chunk_id) * 2 + 0] = s_max;
        chunk_meta[(h * nchunks + chunk_id) * 2 + 1] = s_sum;
    }
}

// ================== Q4_0 Split-K Decode Kernel (v_dot8_i32_i4) ==================
// Reads block_q4_0 K directly, uses v_dot8 for QK dot product.
// Q is quantized to int4 at runtime, packed in q4_0-interleaved order.
// V can be fp16 or q8_0.
template <int D>
static __global__ __launch_bounds__(256, 2) void sage_splitk_q4_kernel(
    const float * __restrict__ Q,
    const char  * __restrict__ K_data,
    const char  * __restrict__ V_data,
    float       * __restrict__ partial_out,
    float       * __restrict__ chunk_meta,
    const half  * __restrict__ mask_data,
    const int S, const int H, const int Hkv, const int nchunks,
    const size_t nb11, const size_t nb12, const size_t nb13,
    const size_t nb21, const size_t nb22, const size_t nb23,
    const size_t mask_stride,
    const int seq_idx,
    const float scale,
    const bool v_is_q8
) {
    constexpr int blocks_per_row = D / 32;
    const int h = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int tid = threadIdx.x;
    const int hkv = h * Hkv / H;

    const int chunk_start = chunk_id * SAGE_CHUNK;
    const int chunk_end   = min(chunk_start + SAGE_CHUNK, S);
    const int chunk_len   = chunk_end - chunk_start;

    const char * K_base = K_data + seq_idx * nb13 + hkv * nb12;
    const char * V_base = V_data + seq_idx * nb23 + hkv * nb22;

    // Quantize Q to int4 and pack in q4_0-interleaved order
    float q_val = (tid < D) ? Q[tid + h * D] : 0.0f;
    float q_abs = fabsf(q_val);

    constexpr int n_waves = (D > 64) ? D / 64 : 1;
    __shared__ float s_amax[n_waves > 4 ? n_waves : 4];
    float warp_max = q_abs;
    for (int offset = 32; offset > 0; offset >>= 1)
        warp_max = fmaxf(warp_max, __shfl_xor(warp_max, offset));
    if (tid % 64 == 0) s_amax[tid / 64] = warp_max;
    __syncthreads();
    float amax = s_amax[0];
    for (int w = 1; w < n_waves; w++) amax = fmaxf(amax, s_amax[w]);
    float q_scale = amax / 7.0f;
    float q_inv = (q_scale > 0.0f) ? 1.0f / q_scale : 0.0f;
    int8_t q_int4_val = (int8_t)max(-8, min(7, __float2int_rn(q_val * q_inv)));

    // Store int4 Q values, then pack in q4_0 interleaved order
    __shared__ int8_t q_i4[D];
    __shared__ int q_packed[D / 8]; // D/32 blocks × 4 groups per block = D/8 total
    if (tid < D) q_i4[tid] = q_int4_val;
    __syncthreads();

    // Pack: for block b, group p: q4_0 interleaving
    // Byte i contains: low nibble = elem[4p+i], high nibble = elem[4p+16+i]
    if (tid < D / 8) {
        int b = tid / 4;  // block index
        int p = tid % 4;  // group within block
        int base = b * 32;
        uint32_t packed = 0;
        for (int i = 0; i < 4; i++) {
            uint8_t lo = (uint8_t)(q_i4[base + 4*p + i] & 0xF);
            uint8_t hi = (uint8_t)(q_i4[base + 4*p + 16 + i] & 0xF);
            ((uint8_t*)&packed)[i] = (hi << 4) | lo;
        }
        q_packed[tid] = (int)packed;
    }
    __syncthreads();

    __shared__ float s_scores[SAGE_CHUNK];

    for (int ki = tid; ki < chunk_len; ki += D) {
        const int s = chunk_start + ki;
        const char * k_row = K_base + (size_t)s * nb11;

        float dot = 0.0f;
        #pragma unroll 4
        for (int b = 0; b < blocks_per_row; b++) {
            const char * blk = k_row + b * Q4_BLK_SIZE;
            half k_d;
            memcpy(&k_d, blk, 2);
            float k_scale_val = __half2float(k_d);

            const uint8_t * qs = (const uint8_t *)(blk + 2);
            int acc = 0;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int k_packed;
                memcpy(&k_packed, &qs[p * 4], 4);
                k_packed ^= 0x88888888; // unsigned [0,15] → signed [-8,+7]
                acc = sage_sdot8(q_packed[b * 4 + p], k_packed, acc);
            }
            dot += (float)acc * k_scale_val;
        }
        float score = dot * q_scale * scale;
        if (mask_data) score += __half2float(mask_data[s]);
        s_scores[ki] = score;
    }
    __syncthreads();

    __shared__ float s_max, s_sum;
    if (tid == 0) {
        float m = -1e30f;
        for (int i = 0; i < chunk_len; i++) m = fmaxf(m, s_scores[i]);
        s_max = m;
        float sum = 0.0f;
        for (int i = 0; i < chunk_len; i++) {
            s_scores[i] = expf(s_scores[i] - m);
            sum += s_scores[i];
        }
        s_sum = sum;
    }
    __syncthreads();

    // V accumulation (same as q8 kernel)
    float acc = 0.0f;
    if (v_is_q8) {
        int vb_idx = tid / 32;
        int vb_off = tid % 32;
        for (int si = 0; si < chunk_len; si++) {
            const int s = chunk_start + si;
            const char * v_row = V_base + (size_t)s * nb21;
            const char * vblk = v_row + vb_idx * Q8_BLK_SIZE;
            half vd; memcpy(&vd, vblk, 2);
            float v_scale_val = __half2float(vd);
            int8_t v_q = *((const int8_t *)(vblk + 2) + vb_off);
            acc += s_scores[si] * ((float)v_q * v_scale_val);
        }
    } else {
        for (int si = 0; si < chunk_len; si++) {
            const int s = chunk_start + si;
            const half * v_row = (const half *)(V_base + (size_t)s * nb21);
            acc += s_scores[si] * __half2float(v_row[tid]);
        }
    }

    partial_out[(h * nchunks + chunk_id) * D + tid] = acc;
    if (tid == 0) {
        chunk_meta[(h * nchunks + chunk_id) * 2 + 0] = s_max;
        chunk_meta[(h * nchunks + chunk_id) * 2 + 1] = s_sum;
    }
}

// ================== Reduce Kernel ==================
template <int D>
static __global__ void sage_splitk_reduce(
    const float * __restrict__ partial_out,
    const float * __restrict__ chunk_meta,
    float       * __restrict__ output,
    const int H, const int nchunks
) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;

    float global_max = -1e30f;
    for (int c = 0; c < nchunks; c++)
        global_max = fmaxf(global_max, chunk_meta[(h * nchunks + c) * 2 + 0]);

    float global_sum = 0.0f;
    float out = 0.0f;
    for (int c = 0; c < nchunks; c++) {
        float correction = expf(chunk_meta[(h * nchunks + c) * 2 + 0] - global_max);
        global_sum += chunk_meta[(h * nchunks + c) * 2 + 1] * correction;
        out += partial_out[(h * nchunks + c) * D + d] * correction;
    }
    output[d + h * D] = (global_sum > 0.0f) ? (out / global_sum) : 0.0f;
}

// ================== Q4_0 Prefill Tile Kernel (v_dot8_i32_i4 QK, half2 SV) ===========
// Same structure as Q8 kernel but:
// - Q quantized to int4 (not int8), packed in q4_0-interleaved order
// - K read from block_q4_0 (18 bytes/block), using v_dot8 for QK
// - 4 v_dot8 per block (vs 8 v_dot4 for q8_0): same compute, 47% less K bandwidth

template <int D, int TQ, int TK>
static __global__ __launch_bounds__(256, 2) void sage_prefill_q4_kernel(
    const char  * __restrict__ Q_data,
    const char  * __restrict__ K_data,
    const char  * __restrict__ V_data,
    char        * __restrict__ dst_data,
    const half  * __restrict__ mask_data,
    const int n_tokens, const int S, const int H, const int Hkv,
    const size_t Q_nb1, const size_t Q_nb2, const size_t Q_nb3,
    const size_t K_nb1, const size_t K_nb2, const size_t K_nb3,
    const size_t V_nb1, const size_t V_nb2, const size_t V_nb3,
    const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3,
    const size_t mask_nb1,
    const int seq_idx,
    const float scale
) {
    constexpr int BPR = D / 32;
    constexpr int NTHREADS = 256;
    constexpr int TPG = D / 2;
    constexpr int Q_GROUPS = NTHREADS / TPG;
    constexpr int QPG = TQ / Q_GROUPS;

    const int q_tile = blockIdx.x;
    const int h      = blockIdx.y;
    const int tid    = threadIdx.x;
    const int dim_lo = 2 * (tid % TPG);
    const int q_group = tid / TPG;
    const int hkv    = h * Hkv / H;
    const int q_base = q_tile * TQ;
    const int wave_id = tid >> 6;
    const int lane_id = tid & 63;

    // Q packed as int4 in q4_0-interleaved order: D/8 packed ints per Q row
    __shared__ int    smem_Qp[TQ][D/8];
    __shared__ float  smem_Qs[TQ];
    __shared__ float  smem_QK[TQ][TK];
    __shared__ int    smem_Kp[TK][D/4]; // reuse D/4 slots (only D/8 used for q4, pad is fine)
    __shared__ float  smem_Ks[TK * BPR];
    __shared__ half2  smem_Vh2[TK][D/2];

    float O_lo[QPG], O_hi[QPG], m_acc[QPG], l_acc[QPG];
    #pragma unroll
    for (int i = 0; i < QPG; i++) {
        O_lo[i] = 0; O_hi[i] = 0; m_acc[i] = -1e30f; l_acc[i] = 0;
    }

    const char * Q_seq  = Q_data + seq_idx * Q_nb3;
    const char * K_head = K_data + seq_idx * K_nb3 + hkv * K_nb2;
    const char * V_head = V_data + seq_idx * V_nb3 + hkv * V_nb2;

    // === Q quantization to int4 ===
    {
        constexpr int Q_WAVES = TPG / 64;
        float   (* s_wmax)[Q_WAVES] = (float (*)[Q_WAVES])smem_Ks;
        int8_t  (* s_qb)[D]         = (int8_t (*)[D])smem_QK;
        const int group_wave = wave_id - q_group * Q_WAVES;

        #pragma unroll
        for (int qi = 0; qi < QPG; qi++) {
            int q_idx = q_group * QPG + qi;
            int q_pos = q_base + q_idx;
            const float * qr = (q_pos < n_tokens)
                ? (const float *)(Q_seq + h * Q_nb2 + q_pos * Q_nb1) : nullptr;
            float val0 = qr ? qr[dim_lo]     : 0.0f;
            float val1 = qr ? qr[dim_lo + 1] : 0.0f;
            float wmax = fmaxf(fabsf(val0), fabsf(val1));
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1)
                wmax = fmaxf(wmax, __shfl_xor(wmax, off));
            if (lane_id == 0) s_wmax[q_idx][group_wave] = wmax;
            __syncthreads();
            float amax = 0;
            for (int w = 0; w < Q_WAVES; w++) amax = fmaxf(amax, s_wmax[q_idx][w]);
            float qinv = (amax > 0.0f) ? (7.0f / amax) : 0.0f;
            s_qb[q_idx][dim_lo]     = (int8_t)max(-8, min(7, __float2int_rn(val0 * qinv)));
            s_qb[q_idx][dim_lo + 1] = (int8_t)max(-8, min(7, __float2int_rn(val1 * qinv)));
            if (dim_lo == 0) smem_Qs[q_idx] = amax / 7.0f;
            __syncthreads();
        }

        // Pack Q in q4_0-interleaved order
        for (int pi = tid; pi < TQ * (D/8); pi += NTHREADS) {
            int qi = pi / (D/8);
            int pack_idx = pi % (D/8);
            int b = pack_idx / 4;
            int p = pack_idx % 4;
            int base = b * 32;
            uint32_t packed = 0;
            for (int i = 0; i < 4; i++) {
                uint8_t lo = (uint8_t)(s_qb[qi][base + 4*p + i] & 0xF);
                uint8_t hi = (uint8_t)(s_qb[qi][base + 4*p + 16 + i] & 0xF);
                ((uint8_t*)&packed)[i] = (hi << 4) | lo;
            }
            smem_Qp[qi][pack_idx] = (int)packed;
        }
        __syncthreads();
    }

    // === K tile loop ===
    for (int k_start = 0; k_start < S; k_start += TK) {
        if (mask_data) {
            int last_q = min(q_base + TQ - 1, n_tokens - 1);
            half mv = *(const half *)((const char *)mask_data + last_q * mask_nb1 + k_start * sizeof(half));
            if (__hisinf(mv)) { break; }
        }
        int tile_len = min(TK, S - k_start);

        // Load K (q4_0) + V (fp16) into smem
        {
            for (int bi = tid; bi < TK * BPR; bi += NTHREADS) {
                int k_row = bi / BPR;
                int blk_id = bi % BPR;
                if (k_row < tile_len) {
                    const char * kblk = K_head + (size_t)(k_start + k_row) * K_nb1 + blk_id * Q4_BLK_SIZE;
                    half kd; memcpy(&kd, kblk, 2);
                    smem_Ks[k_row * BPR + blk_id] = __half2float(kd);
                    const uint8_t * qs = (const uint8_t *)(kblk + 2);
                    #pragma unroll
                    for (int p = 0; p < 4; p++) {
                        int kpacked; memcpy(&kpacked, &qs[p*4], 4);
                        kpacked ^= 0x88888888;
                        smem_Kp[k_row][blk_id * 4 + p] = kpacked;
                    }
                }
            }
            constexpr int V_H2 = TK * (D / 2);
            for (int vi = tid; vi < V_H2; vi += NTHREADS) {
                int row  = vi / (D / 2);
                int col2 = vi % (D / 2);
                if (row < tile_len) {
                    const half * vr = (const half *)(V_head + (size_t)(k_start + row) * V_nb1);
                    smem_Vh2[row][col2] = *(const half2 *)&vr[col2 * 2];
                } else {
                    smem_Vh2[row][col2] = __float2half2_rn(0.0f);
                }
            }
        }
        __syncthreads();

        // QK^T via sdot8
        {
            constexpr int QK_TOTAL = TQ * TK;
            for (int idx = tid; idx < QK_TOTAL; idx += NTHREADS) {
                int q_idx = idx / TK;
                int k_idx = idx % TK;
                float dot_f = -1e30f;
                if (k_idx < tile_len && (q_base + q_idx) < n_tokens) {
                    dot_f = 0.0f;
                    #pragma unroll 4
                    for (int b = 0; b < BPR; b++) {
                        int bacc = 0;
                        #pragma unroll
                        for (int p = 0; p < 4; p++)
                            bacc = sage_sdot8(smem_Qp[q_idx][b*4+p], smem_Kp[k_idx][b*4+p], bacc);
                        dot_f += (float)bacc * smem_Ks[k_idx * BPR + b];
                    }
                    dot_f *= smem_Qs[q_idx] * scale;
                    if (mask_data) {
                        int q_pos = q_base + q_idx;
                        dot_f += __half2float(*(const half *)((const char *)mask_data + q_pos * mask_nb1 + (k_start + k_idx) * sizeof(half)));
                    }
                }
                smem_QK[q_idx][k_idx] = dot_f;
            }
        }
        __syncthreads();

        // Online softmax + SV (identical to q8 kernel)
        {
            const int dim2 = tid % TPG;
            #pragma unroll
            for (int qi = 0; qi < QPG; qi++) {
                int q_idx = q_group * QPG + qi;
                if (q_idx >= TQ || q_base + q_idx >= n_tokens) continue;
                float tmax = -1e30f;
                #pragma unroll
                for (int k = 0; k < TK; k++)
                    tmax = fmaxf(tmax, smem_QK[q_idx][k]);
                float m_new = fmaxf(m_acc[qi], tmax);
                float corr = sage_fast_exp(m_acc[qi] - m_new);
                O_lo[qi] *= corr;
                O_hi[qi] *= corr;
                l_acc[qi] *= corr;
                #pragma unroll
                for (int k = 0; k < TK; k++) {
                    float w = sage_fast_exp(smem_QK[q_idx][k] - m_new);
                    half2 v2 = smem_Vh2[k][dim2];
                    float2 vf = __half22float2(v2);
                    O_lo[qi] += w * vf.x;
                    O_hi[qi] += w * vf.y;
                    l_acc[qi] += w;
                }
                m_acc[qi] = m_new;
            }
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int qi = 0; qi < QPG; qi++) {
        int q_idx = q_group * QPG + qi;
        if (q_idx >= TQ) break;
        int q_pos = q_base + q_idx;
        if (q_pos < n_tokens) {
            float * out = (float *)((char *)dst_data + h * dst_nb1 + q_pos * dst_nb2 + seq_idx * dst_nb3);
            float inv_l = (l_acc[qi] > 0.0f) ? sage_fast_rcp(l_acc[qi]) : 0.0f;
            out[dim_lo]     = O_lo[qi] * inv_l;
            out[dim_lo + 1] = O_hi[qi] * inv_l;
        }
    }
}

// ================== Hybrid Q8Q+Q4K Prefill Kernel ==================
// Q quantized to INT8 (high precision), K stored as q4_0 (dequantized to INT8 in kernel)
// Uses sage_sdot4 (v_dot4_i32_i8) for QK matmul
// This gives q4_0 bandwidth savings with INT8-quality Q quantization
template <int D, int TQ, int TK>
static __global__ __launch_bounds__(256, 2) void sage_prefill_q4k_q8q_kernel(
    const char  * __restrict__ Q_data,
    const char  * __restrict__ K_data,
    const char  * __restrict__ V_data,
    char        * __restrict__ dst_data,
    const half  * __restrict__ mask_data,
    const int n_tokens, const int S, const int H, const int Hkv,
    const size_t Q_nb1, const size_t Q_nb2, const size_t Q_nb3,
    const size_t K_nb1, const size_t K_nb2, const size_t K_nb3,
    const size_t V_nb1, const size_t V_nb2, const size_t V_nb3,
    const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3,
    const size_t mask_nb1,
    const int seq_idx,
    const float scale
) {
    constexpr int BPR = D / 32;  // blocks per row for q4_0 (same as q8_0: 32 elements per block)
    constexpr int NTHREADS = 256;
    constexpr int TPG = D / 2;
    constexpr int Q_GROUPS = NTHREADS / TPG;
    constexpr int QPG = TQ / Q_GROUPS;

    const int q_tile = blockIdx.x;
    const int h      = blockIdx.y;
    const int tid    = threadIdx.x;
    const int dim_lo = 2 * (tid % TPG);
    const int q_group = tid / TPG;
    const int hkv    = h * Hkv / H;
    const int q_base = q_tile * TQ;
    const int wave_id = tid >> 6;
    const int lane_id = tid & 63;

    // Q packed as INT8 (same as q8 kernel): D/4 packed ints per Q row
    __shared__ int    smem_Qp[TQ][D/4];
    __shared__ float  smem_Qs[TQ];
    // smem_QK must fit int8_t[TQ][D] for Q quantization aliasing (D/4 floats per row)
    static constexpr int QK_COLS_H = (D / (int)sizeof(float) > TK) ? D / (int)sizeof(float) : TK;
    __shared__ float  smem_QK[TQ][QK_COLS_H];
    // K dequantized to INT8, packed as D/4 ints per K row (same layout as q8 kernel)
    __shared__ int    smem_Kp[TK][D/4];
    __shared__ float  smem_Ks[TK * BPR];
    __shared__ half2  smem_Vh2[TK][D/2];

    float O_lo[QPG], O_hi[QPG], m_acc[QPG], l_acc[QPG];
    #pragma unroll
    for (int i = 0; i < QPG; i++) {
        O_lo[i] = 0; O_hi[i] = 0; m_acc[i] = -1e30f; l_acc[i] = 0;
    }

    const char * Q_seq  = Q_data + seq_idx * Q_nb3;
    const char * K_head = K_data + seq_idx * K_nb3 + hkv * K_nb2;
    const char * V_head = V_data + seq_idx * V_nb3 + hkv * V_nb2;

    // === Q quantization to INT8 (identical to q8 kernel) ===
    {
        constexpr int Q_WAVES = TPG / 64;
        float   (* s_wmax)[Q_WAVES] = (float (*)[Q_WAVES])smem_Ks;
        int8_t  (* s_qb)[D]         = (int8_t (*)[D])smem_QK;
        const int group_wave = wave_id - q_group * Q_WAVES;

        #pragma unroll
        for (int qi = 0; qi < QPG; qi++) {
            int q_idx = q_group * QPG + qi;
            int q_pos = q_base + q_idx;
            const float * qr = (q_pos < n_tokens)
                ? (const float *)(Q_seq + h * Q_nb2 + q_pos * Q_nb1) : nullptr;
            float val0 = qr ? qr[dim_lo]     : 0.0f;
            float val1 = qr ? qr[dim_lo + 1] : 0.0f;
            float wmax = fmaxf(fabsf(val0), fabsf(val1));
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1)
                wmax = fmaxf(wmax, __shfl_xor(wmax, off));
            if (lane_id == 0) s_wmax[q_idx][group_wave] = wmax;
            __syncthreads();
            float amax = 0;
            for (int w = 0; w < Q_WAVES; w++) amax = fmaxf(amax, s_wmax[q_idx][w]);
            float qinv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            s_qb[q_idx][dim_lo]     = (int8_t)max(-127, min(127, __float2int_rn(val0 * qinv)));
            s_qb[q_idx][dim_lo + 1] = (int8_t)max(-127, min(127, __float2int_rn(val1 * qinv)));
            if (dim_lo == 0) smem_Qs[q_idx] = amax / 127.0f;
            __syncthreads();
        }

        // Pack Q as 4×int8 → int32 (standard byte packing for sdot4)
        for (int pi = tid; pi < TQ * (D/4); pi += NTHREADS) {
            int qi = pi / (D/4);
            int d4 = pi % (D/4);
            int packed; memcpy(&packed, &s_qb[qi][d4 * 4], 4);
            smem_Qp[qi][d4] = packed;
        }
        __syncthreads();
    }

    // === K tile loop ===
    for (int k_start = 0; k_start < S; k_start += TK) {
        if (mask_data) {
            int last_q = min(q_base + TQ - 1, n_tokens - 1);
            half mv = *(const half *)((const char *)mask_data + last_q * mask_nb1 + k_start * sizeof(half));
            if (__hisinf(mv)) { break; }
        }
        int tile_len = min(TK, S - k_start);

        // Load K from q4_0, dequantize nibbles to INT8, pack as int32 for sdot4
        {
            for (int bi = tid; bi < TK * BPR; bi += NTHREADS) {
                int k_row = bi / BPR;
                int blk_id = bi % BPR;
                if (k_row < tile_len) {
                    const char * kblk = K_head + (size_t)(k_start + k_row) * K_nb1 + blk_id * Q4_BLK_SIZE;
                    half kd; memcpy(&kd, kblk, 2);
                    float k_scale_val = __half2float(kd);
                    // Store a combined scale: K block scale (maps int4 range [-8,7] to float)
                    // For sdot4 with int8 Q, we need K values as int8.
                    // q4_0: val = (nibble - 8) * scale. We want int8 K values where
                    // the dot product result * scale gives the correct answer.
                    // Strategy: dequant nibble to int8 directly: nibble - 8 maps [-8, 7]
                    // which fits in int8. Then combined_scale = Q_scale * K_block_scale.
                    smem_Ks[k_row * BPR + blk_id] = k_scale_val;

                    const uint8_t * qs = (const uint8_t *)(kblk + 2);
                    // q4_0 format: 16 bytes = 32 nibbles, stored as pairs (lo|hi) in each byte
                    // Dequant: for byte b, lo_nibble = (b & 0xF) - 8, hi_nibble = (b >> 4) - 8
                    // Pack as int8×4 for sdot4
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        // 4 consecutive bytes of q4_0 data = 8 nibbles → 8 int8 values
                        // We need to pack them as int8×4 for sdot4
                        uint32_t raw; memcpy(&raw, &qs[p * 2], 4);  // 4 bytes = 8 nibbles
                        // Actually q4_0 has 16 bytes for 32 values, indexed as:
                        // byte[i] has lo nibble = element[i] and hi nibble = element[i+16]
                        // So for block with 32 elements, bytes 0..15 store pairs
                        
                        // Let's do it properly: extract 4 int8 values from q4_0 block
                        // Elements in a q4_0 block are: for byte j (0..15):
                        //   element[j]    = (byte[j] & 0xF) - 8
                        //   element[j+16] = (byte[j] >> 4) - 8
                        // We want to pack 4 consecutive elements as int8×4
                        
                        int8_t vals[4];
                        if (p < 4) {
                            // Elements 0..15 (lo nibbles)
                            for (int i = 0; i < 4; i++) {
                                uint8_t byte_val = qs[p * 4 + i];
                                vals[i] = (int8_t)((byte_val & 0xF) - 8);
                            }
                        } else {
                            // Elements 16..31 (hi nibbles)
                            for (int i = 0; i < 4; i++) {
                                uint8_t byte_val = qs[(p - 4) * 4 + i];
                                vals[i] = (int8_t)((byte_val >> 4) - 8);
                            }
                        }
                        int packed;
                        memcpy(&packed, vals, 4);
                        smem_Kp[k_row][blk_id * 8 + p] = packed;
                    }
                }
            }
            // Load V (fp16)
            constexpr int V_H2 = TK * (D / 2);
            for (int vi = tid; vi < V_H2; vi += NTHREADS) {
                int row  = vi / (D / 2);
                int col2 = vi % (D / 2);
                if (row < tile_len) {
                    const half * vr = (const half *)(V_head + (size_t)(k_start + row) * V_nb1);
                    smem_Vh2[row][col2] = *(const half2 *)&vr[col2 * 2];
                } else {
                    smem_Vh2[row][col2] = __float2half2_rn(0.0f);
                }
            }
        }
        __syncthreads();

        // QK^T via sdot4 (INT8 Q × INT8 K from dequantized q4_0)
        {
            constexpr int QK_TOTAL = TQ * TK;
            for (int idx = tid; idx < QK_TOTAL; idx += NTHREADS) {
                int q_idx = idx / TK;
                int k_idx = idx % TK;
                float dot_f = -1e30f;
                if (k_idx < tile_len && (q_base + q_idx) < n_tokens) {
                    dot_f = 0.0f;
                    #pragma unroll 4
                    for (int b = 0; b < BPR; b++) {
                        int bacc = 0;
                        #pragma unroll
                        for (int p = 0; p < 8; p++)
                            bacc = sage_sdot4(smem_Qp[q_idx][b*8+p], smem_Kp[k_idx][b*8+p], bacc);
                        dot_f += (float)bacc * smem_Ks[k_idx * BPR + b];
                    }
                    dot_f *= smem_Qs[q_idx] * scale;
                    if (mask_data) {
                        int q_pos = q_base + q_idx;
                        dot_f += __half2float(*(const half *)((const char *)mask_data + q_pos * mask_nb1 + (k_start + k_idx) * sizeof(half)));
                    }
                }
                smem_QK[q_idx][k_idx] = dot_f;
            }
        }
        __syncthreads();

        // Online softmax + SV (identical to q8/q4 kernels)
        {
            const int dim2 = tid % TPG;
            #pragma unroll
            for (int qi = 0; qi < QPG; qi++) {
                int q_idx = q_group * QPG + qi;
                if (q_idx >= TQ || q_base + q_idx >= n_tokens) continue;
                float tmax = -1e30f;
                #pragma unroll
                for (int k = 0; k < TK; k++)
                    tmax = fmaxf(tmax, smem_QK[q_idx][k]);
                float m_new = fmaxf(m_acc[qi], tmax);
                float corr = sage_fast_exp(m_acc[qi] - m_new);
                O_lo[qi] *= corr;
                O_hi[qi] *= corr;
                l_acc[qi] *= corr;
                #pragma unroll
                for (int k = 0; k < TK; k++) {
                    float w = sage_fast_exp(smem_QK[q_idx][k] - m_new);
                    half2 v2 = smem_Vh2[k][dim2];
                    float2 vf = __half22float2(v2);
                    O_lo[qi] += w * vf.x;
                    O_hi[qi] += w * vf.y;
                    l_acc[qi] += w;
                }
                m_acc[qi] = m_new;
            }
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int qi = 0; qi < QPG; qi++) {
        int q_idx = q_group * QPG + qi;
        if (q_idx >= TQ) break;
        int q_pos = q_base + q_idx;
        if (q_pos < n_tokens) {
            float * out = (float *)((char *)dst_data + h * dst_nb1 + q_pos * dst_nb2 + seq_idx * dst_nb3);
            float inv_l = (l_acc[qi] > 0.0f) ? sage_fast_rcp(l_acc[qi]) : 0.0f;
            out[dim_lo]     = O_lo[qi] * inv_l;
            out[dim_lo + 1] = O_hi[qi] * inv_l;
        }
    }
}


// ================== Q8 Prefill Tile Kernel (half2 SV, K/V union) ===========
// 256 threads, half2 thread mapping (each thread handles 2 V dims).
// K and V share the same smem via union (K freed after QK, V loaded for SV).
// D=128: TPG=64, 4 groups, QPG=TQ/4
// D=256: TPG=128, 2 groups, QPG=TQ/2

template <int D, int TQ, int TK>
static __global__ __launch_bounds__(256, 2) void sage_prefill_q8_kernel(
    const char  * __restrict__ Q_data,
    const char  * __restrict__ K_data,
    const char  * __restrict__ V_data,
    char        * __restrict__ dst_data,
    const half  * __restrict__ mask_data,
    const int n_tokens, const int S, const int H, const int Hkv,
    const size_t Q_nb1, const size_t Q_nb2, const size_t Q_nb3,
    const size_t K_nb1, const size_t K_nb2, const size_t K_nb3,
    const size_t V_nb1, const size_t V_nb2, const size_t V_nb3,
    const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3,
    const size_t mask_nb1,
    const int seq_idx,
    const float scale
) {
    constexpr int BPR = D / 32;
    constexpr int NTHREADS = 256;
    constexpr int TPG = D / 2;
    constexpr int Q_GROUPS = NTHREADS / TPG;
    constexpr int QPG = TQ / Q_GROUPS;

    const int q_tile = blockIdx.x;
    const int h      = blockIdx.y;
    const int tid    = threadIdx.x;
    const int dim_lo = 2 * (tid % TPG);
    const int q_group = tid / TPG;
    const int hkv    = h * Hkv / H;
    const int q_base = q_tile * TQ;
    const int wave_id = tid >> 6;
    const int lane_id = tid & 63;

    __shared__ int    smem_Qp[TQ][D/4];
    __shared__ float  smem_Qs[TQ];
    // smem_QK must fit int8_t[TQ][D] for Q quantization aliasing (D/4 floats per row)
    static constexpr int QK_COLS = (D / (int)sizeof(float) > TK) ? D / (int)sizeof(float) : TK;
    __shared__ float  smem_QK[TQ][QK_COLS];
    __shared__ int    smem_Kp[TK][D/4];
    __shared__ float  smem_Ks[TK * BPR];
    __shared__ half2  smem_Vh2[TK][D/2];

    float O_lo[QPG], O_hi[QPG], m_acc[QPG], l_acc[QPG];
    #pragma unroll
    for (int i = 0; i < QPG; i++) {
        O_lo[i] = 0; O_hi[i] = 0; m_acc[i] = -1e30f; l_acc[i] = 0;
    }

    const char * Q_seq  = Q_data + seq_idx * Q_nb3;
    const char * K_head = K_data + seq_idx * K_nb3 + hkv * K_nb2;
    const char * V_head = V_data + seq_idx * V_nb3 + hkv * V_nb2;

    // === Q quantization ===
    {
        constexpr int Q_WAVES = TPG / 64;
        // Alias Q quantization temps over smem_QK/smem_Ks (unused during Q quant)
        float   (* s_wmax)[Q_WAVES] = (float (*)[Q_WAVES])smem_Ks; // needs TQ*Q_WAVES*4 ≤ TK*BPR*4
        int8_t  (* s_qb)[D]         = (int8_t (*)[D])smem_QK;      // needs TQ*D ≤ TQ*QK_COLS*4
        const int group_wave = wave_id - q_group * Q_WAVES;

        #pragma unroll
        for (int qi = 0; qi < QPG; qi++) {
            int q_idx = q_group * QPG + qi;
            int q_pos = q_base + q_idx;
            const float * qr = (q_pos < n_tokens)
                ? (const float *)(Q_seq + h * Q_nb2 + q_pos * Q_nb1) : nullptr;
            float val0 = qr ? qr[dim_lo]     : 0.0f;
            float val1 = qr ? qr[dim_lo + 1] : 0.0f;
            float wmax = fmaxf(fabsf(val0), fabsf(val1));
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1)
                wmax = fmaxf(wmax, __shfl_xor(wmax, off));
            if (lane_id == 0) s_wmax[q_idx][group_wave] = wmax;
            __syncthreads();
            float amax = 0;
            for (int w = 0; w < Q_WAVES; w++) amax = fmaxf(amax, s_wmax[q_idx][w]);
            float qinv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            s_qb[q_idx][dim_lo]     = (int8_t)max(-127, min(127, __float2int_rn(val0 * qinv)));
            s_qb[q_idx][dim_lo + 1] = (int8_t)max(-127, min(127, __float2int_rn(val1 * qinv)));
            if (dim_lo == 0) smem_Qs[q_idx] = amax / 127.0f;
            __syncthreads();
        }

        for (int pi = tid; pi < TQ * (D/4); pi += NTHREADS) {
            int qi = pi / (D/4);
            int d4 = pi % (D/4);
            int packed; memcpy(&packed, &s_qb[qi][d4 * 4], 4);
            smem_Qp[qi][d4] = packed;
        }
        __syncthreads();
    }

    // === K tile loop ===
    for (int k_start = 0; k_start < S; k_start += TK) {
        if (mask_data) {
            int last_q = min(q_base + TQ - 1, n_tokens - 1);
            half mv = *(const half *)((const char *)mask_data + last_q * mask_nb1 + k_start * sizeof(half));
            if (__hisinf(mv)) { break; }
        }
        int tile_len = min(TK, S - k_start);

        // Load K into smem_KV.k
        {
            constexpr int K_BLOCKS = TK * BPR;
            for (int bi = tid; bi < K_BLOCKS; bi += NTHREADS) {
                int k_row = bi / BPR;
                int blk_id = bi % BPR;
                if (k_row < tile_len) {
                    const char * kblk = K_head + (size_t)(k_start + k_row) * K_nb1 + blk_id * Q8_BLK_SIZE;
                    half kd; memcpy(&kd, kblk, 2);
                    smem_Ks[k_row * BPR + blk_id] = __half2float(kd);
                    const int8_t * qs = (const int8_t *)(kblk + 2);
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        int packed; memcpy(&packed, &qs[p*4], 4);
                        smem_Kp[k_row][blk_id * 8 + p] = packed;
                    }
                }
            }
            // V loading: simultaneous with K (separate smem, no conflict)
            constexpr int V_H2 = TK * (D / 2);
            for (int vi = tid; vi < V_H2; vi += NTHREADS) {
                int row  = vi / (D / 2);
                int col2 = vi % (D / 2);
                if (row < tile_len) {
                    const half * vr = (const half *)(V_head + (size_t)(k_start + row) * V_nb1);
                    smem_Vh2[row][col2] = *(const half2 *)&vr[col2 * 2];
                } else {
                    smem_Vh2[row][col2] = __float2half2_rn(0.0f);
                }
            }
        }
        __syncthreads();

        // QK^T via sdot4
        {
            constexpr int QK_TOTAL = TQ * TK;
            for (int idx = tid; idx < QK_TOTAL; idx += NTHREADS) {
                int q_idx = idx / TK;
                int k_idx = idx % TK;
                float dot_f = -1e30f;
                if (k_idx < tile_len && (q_base + q_idx) < n_tokens) {
                    dot_f = 0.0f;
                    #pragma unroll 4
                    for (int b = 0; b < BPR; b++) {
                        int bacc = 0;
                        #pragma unroll
                        for (int p = 0; p < 8; p++)
                            bacc = sage_sdot4(smem_Qp[q_idx][b*8+p], smem_Kp[k_idx][b*8+p], bacc);
                        dot_f += (float)bacc * smem_Ks[k_idx * BPR + b];
                    }
                    dot_f *= smem_Qs[q_idx] * scale;
                    if (mask_data) {
                        int q_pos = q_base + q_idx;
                        dot_f += __half2float(*(const half *)((const char *)mask_data + q_pos * mask_nb1 + (k_start + k_idx) * sizeof(half)));
                    }
                }
                smem_QK[q_idx][k_idx] = dot_f;
            }
        }
        __syncthreads();


        // Online softmax + SV
        {
            const int dim2 = tid % TPG;
            #pragma unroll
            for (int qi = 0; qi < QPG; qi++) {
                int q_idx = q_group * QPG + qi;
                if (q_idx >= TQ || q_base + q_idx >= n_tokens) continue;
                float tmax = -1e30f;
                #pragma unroll
                for (int k = 0; k < TK; k++)
                    tmax = fmaxf(tmax, smem_QK[q_idx][k]);
                float m_new = fmaxf(m_acc[qi], tmax);
                float corr = sage_fast_exp(m_acc[qi] - m_new);
                O_lo[qi] *= corr;
                O_hi[qi] *= corr;
                l_acc[qi] *= corr;
                #pragma unroll
                for (int k = 0; k < TK; k++) {
                    float w = sage_fast_exp(smem_QK[q_idx][k] - m_new);
                    half2 v2 = smem_Vh2[k][dim2];
                    float2 vf = __half22float2(v2);
                    O_lo[qi] += w * vf.x;
                    O_hi[qi] += w * vf.y;
                    l_acc[qi] += w;
                }
                m_acc[qi] = m_new;
            }
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int qi = 0; qi < QPG; qi++) {
        int q_idx = q_group * QPG + qi;
        if (q_idx >= TQ) break;
        int q_pos = q_base + q_idx;
        if (q_pos < n_tokens) {
            float * out = (float *)((char *)dst_data + h * dst_nb1 + q_pos * dst_nb2 + seq_idx * dst_nb3);
            float inv_l = (l_acc[qi] > 0.0f) ? sage_fast_rcp(l_acc[qi]) : 0.0f;
            out[dim_lo]     = O_lo[qi] * inv_l;
            out[dim_lo + 1] = O_hi[qi] * inv_l;
        }
    }
}

void ggml_cuda_flash_attn_ext_sage_splitk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q->type == GGML_TYPE_F32);

    const int D       = Q->ne[0];
    const int n_tokens = Q->ne[1];
    const int H       = Q->ne[2];
    const int Hkv     = K->ne[2];
    const int S       = K->ne[1];
    const int batch   = Q->ne[3];

    GGML_ASSERT(D == 128 || D == 256);

    float scale;
    memcpy(&scale, (const float *) dst->op_params, sizeof(float));

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    // === Prefill path (n_tokens > 1, quantized K, fp16 V) ===
    if (n_tokens > 1 && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) && V->type == GGML_TYPE_F16) {
        // D=128: TQ=32 (2 groups × 16 rows), D=256: TQ=16 (1 group × 16 rows)
        constexpr int TQ128 = 32, TQ256 = 16, TK = 32;
        const int TQ = (D == 128) ? TQ128 : TQ256;
        const int n_q_tiles = (n_tokens + TQ - 1) / TQ;
        dim3 grid(n_q_tiles, H);
        dim3 block(256);

        const half * mask_ptr = nullptr;
        size_t mask_nb1 = 0;

        for (int seq = 0; seq < batch; seq++) {
            if (mask) {
                mask_ptr = (const half *)((const char *)mask->data + (seq % mask->ne[3]) * mask->nb[3]);
                mask_nb1 = mask->nb[1];
            }
            if (K->type == GGML_TYPE_Q4_0) {
                if (D == 128) {
                    sage_prefill_q4k_q8q_kernel<128, TQ128, TK><<<grid, block, 0, stream>>>(
                        (const char *)Q->data, (const char *)K->data,
                        (const char *)V->data, (char *)dst->data,
                        mask_ptr,
                        n_tokens, S, H, Hkv,
                        Q->nb[1], Q->nb[2], Q->nb[3],
                        K->nb[1], K->nb[2], K->nb[3],
                        V->nb[1], V->nb[2], V->nb[3],
                        dst->nb[1], dst->nb[2], dst->nb[3],
                        mask_nb1, seq, scale
                    );
                } else {
                    sage_prefill_q4k_q8q_kernel<256, TQ256, TK><<<grid, block, 0, stream>>>(
                        (const char *)Q->data, (const char *)K->data,
                        (const char *)V->data, (char *)dst->data,
                        mask_ptr,
                        n_tokens, S, H, Hkv,
                        Q->nb[1], Q->nb[2], Q->nb[3],
                        K->nb[1], K->nb[2], K->nb[3],
                        V->nb[1], V->nb[2], V->nb[3],
                        dst->nb[1], dst->nb[2], dst->nb[3],
                        mask_nb1, seq, scale
                    );
                }
            } else { // Q8_0
                if (D == 128) {
                    sage_prefill_q8_kernel<128, TQ128, TK><<<grid, block, 0, stream>>>(
                        (const char *)Q->data, (const char *)K->data,
                        (const char *)V->data, (char *)dst->data,
                        mask_ptr,
                        n_tokens, S, H, Hkv,
                        Q->nb[1], Q->nb[2], Q->nb[3],
                        K->nb[1], K->nb[2], K->nb[3],
                        V->nb[1], V->nb[2], V->nb[3],
                        dst->nb[1], dst->nb[2], dst->nb[3],
                        mask_nb1, seq, scale
                    );
                } else {
                    sage_prefill_q8_kernel<256, TQ256, TK><<<grid, block, 0, stream>>>(
                        (const char *)Q->data, (const char *)K->data,
                        (const char *)V->data, (char *)dst->data,
                        mask_ptr,
                        n_tokens, S, H, Hkv,
                        Q->nb[1], Q->nb[2], Q->nb[3],
                        K->nb[1], K->nb[2], K->nb[3],
                        V->nb[1], V->nb[2], V->nb[3],
                        dst->nb[1], dst->nb[2], dst->nb[3],
                        mask_nb1, seq, scale
                    );
                }
            }
        }
        return;
    }

    // === Decode path (n_tokens == 1) ===
    GGML_ASSERT(n_tokens == 1);

    const int nchunks = (S + SAGE_CHUNK - 1) / SAGE_CHUNK;

    ggml_cuda_pool_alloc<float> partial_out(pool, H * nchunks * D);
    ggml_cuda_pool_alloc<float> chunk_meta(pool, H * nchunks * 2);

    dim3 grid(H, nchunks);
    dim3 block(D);

    for (int seq = 0; seq < batch; seq++) {
        const float * Q_ptr = (const float *)((const char *)Q->data + seq * Q->nb[3]);

        const half * mask_ptr = nullptr;
        size_t mask_stride_val = 0;
        if (mask) {
            mask_ptr = (const half *)((const char *)mask->data + (seq % mask->ne[3]) * mask->nb[3]);
            mask_stride_val = mask->nb[1] / sizeof(half);
        }

        if (K->type == GGML_TYPE_F16) {
            if (D == 128) {
                bool v_is_q8_fp16 = (V->type == GGML_TYPE_Q8_0);
                sage_splitk_fp16_kernel<128><<<grid, block, 0, stream>>>(
                    Q_ptr, (const char *)K->data, (const char *)V->data,
                    partial_out.ptr, chunk_meta.ptr, mask_ptr,
                    S, H, Hkv, nchunks,
                    K->nb[1], K->nb[2], K->nb[3],
                    V->nb[1], V->nb[2], V->nb[3],
                    mask_stride_val, seq, scale, v_is_q8_fp16
                );
            } else {
                bool v_is_q8_fp16 = (V->type == GGML_TYPE_Q8_0);
                sage_splitk_fp16_kernel<256><<<grid, block, 0, stream>>>(
                    Q_ptr, (const char *)K->data, (const char *)V->data,
                    partial_out.ptr, chunk_meta.ptr, mask_ptr,
                    S, H, Hkv, nchunks,
                    K->nb[1], K->nb[2], K->nb[3],
                    V->nb[1], V->nb[2], V->nb[3],
                    mask_stride_val, seq, scale, v_is_q8_fp16
                );
            }
        } else if (K->type == GGML_TYPE_Q8_0) {
            bool v_is_q8 = (V->type == GGML_TYPE_Q8_0);
            if (D == 128) {
                sage_splitk_q8_kernel<128><<<grid, block, 0, stream>>>(
                    Q_ptr, (const char *)K->data, (const char *)V->data,
                    partial_out.ptr, chunk_meta.ptr, mask_ptr,
                    S, H, Hkv, nchunks,
                    K->nb[1], K->nb[2], K->nb[3],
                    V->nb[1], V->nb[2], V->nb[3],
                    mask_stride_val, seq, scale, v_is_q8
                );
            } else {
                sage_splitk_q8_kernel<256><<<grid, block, 0, stream>>>(
                    Q_ptr, (const char *)K->data, (const char *)V->data,
                    partial_out.ptr, chunk_meta.ptr, mask_ptr,
                    S, H, Hkv, nchunks,
                    K->nb[1], K->nb[2], K->nb[3],
                    V->nb[1], V->nb[2], V->nb[3],
                    mask_stride_val, seq, scale, v_is_q8
                );
            }
        } else if (K->type == GGML_TYPE_Q4_0) {
            bool v_is_q8 = (V->type == GGML_TYPE_Q8_0);
            if (D == 128) {
                sage_splitk_q4_kernel<128><<<grid, block, 0, stream>>>(
                    Q_ptr, (const char *)K->data, (const char *)V->data,
                    partial_out.ptr, chunk_meta.ptr, mask_ptr,
                    S, H, Hkv, nchunks,
                    K->nb[1], K->nb[2], K->nb[3],
                    V->nb[1], V->nb[2], V->nb[3],
                    mask_stride_val, seq, scale, v_is_q8
                );
            } else {
                sage_splitk_q4_kernel<256><<<grid, block, 0, stream>>>(
                    Q_ptr, (const char *)K->data, (const char *)V->data,
                    partial_out.ptr, chunk_meta.ptr, mask_ptr,
                    S, H, Hkv, nchunks,
                    K->nb[1], K->nb[2], K->nb[3],
                    V->nb[1], V->nb[2], V->nb[3],
                    mask_stride_val, seq, scale, v_is_q8
                );
            }
        } else {
            GGML_ABORT("sage-splitk: unsupported K type");
        }

        float * dst_ptr = (float *)((char *)dst->data + seq * dst->nb[3]);
        if (D == 128) {
            sage_splitk_reduce<128><<<dim3(H), dim3(128), 0, stream>>>(
                partial_out.ptr, chunk_meta.ptr, dst_ptr, H, nchunks
            );
        } else {
            sage_splitk_reduce<256><<<dim3(H), dim3(256), 0, stream>>>(
                partial_out.ptr, chunk_meta.ptr, dst_ptr, H, nchunks
            );
        }
    }
}
