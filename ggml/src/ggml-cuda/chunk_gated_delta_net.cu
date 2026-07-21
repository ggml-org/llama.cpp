// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Chunked Gated Delta Net prefill: fwdsub intra -> Q@K^T precompute -> WMMA state+output.
// H-state GEMMs use bf16; k^T v and qk*v_new use fp16; accum/gating/state stay fp32.
//
#include "chunk_gated_delta_net.cuh"

#include <cmath>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Intra-chunk forward substitution. Builds the strictly-lower coupling matrix
//   L[t][s] = beta[t] * exp(g_cum[t]-g_cum[s]) * (K[t].K[s])   (s < t, else 0)
// and solves (I + L) x = b twice: b = beta*exp(g_cum)*K -> K_cumdecay, b = beta*V -> V_corr.
// Also emits g_cum. Grid (B*H, num_chunks); 128 threads. V may use a fused-QKV token stride.
template <int CS, int BK>
__launch_bounds__(128, 4) __global__ void cgdr_fwdsub_intra_kernel(
    const float * __restrict__ K,     // (B, T, H, K)
    const float * __restrict__ V,     // (B, T, H, V), token stride = v_tok_stride (may be fused QKV)
    const float * __restrict__ Beta,  // (B, T, H)
    const float * __restrict__ G,     // (B, T, H)
    float * __restrict__ V_corr,      // (B, H, C, CS, V) output
    float * __restrict__ K_cumdecay,  // (B, H, C, CS, K) output
    float * __restrict__ G_cum_out,   // (B, H, C, CS)    output
    int       B,
    int       seq_len,
    int       H,
    int       num_chunks,
    int       K_dim,
    int       V_dim,
    int       num_k_heads,   // q/k head count (H if MHA; H is the v-head count for GQA)
    long long v_tok_stride)  // elements between V tokens (H*V_dim if contiguous; QKV row width if fused)
{
    static_assert(BK == 128, "cgdr_fwdsub_intra_kernel: BK=128 only");

    // SMEM layout:
    //   s_K   [CS][BK+1] -- padded stride SK=BK+1=129 eliminates SMEM bank conflicts
    //   s_L   [CS][CS]   -- coupling matrix L
    //   s_gcum[CS], s_beta[CS]
    constexpr int           SK = BK + 1;
    extern __shared__ float smem[];
    float *                 s_K    = smem;
    float *                 s_L    = s_K + CS * SK;
    float *                 s_gcum = s_L + CS * CS;
    float *                 s_beta = s_gcum + CS;

    const int tid       = threadIdx.x;
    const int pid_bh    = blockIdx.x;
    const int pid_chunk = blockIdx.y;
    const int b         = pid_bh / H;
    const int h         = pid_bh % H;       // v-head
    const int h_k       = h % num_k_heads;  // GQA: v-head -> shared k-head (identity when num_k_heads==H)
    const int t_off     = pid_chunk * CS;

    // K uses the (un-repeated) k-head count for its token/head stride; V/beta/g are per v-head.
    const long long HK = (long long) num_k_heads * K_dim;

    // V may be a strided view of the fused QKV buffer: token stride is v_tok_stride (not H*V_dim),
    // but v-heads stay packed (head stride V_dim) and elements contiguous, so only the token stride
    // and seq stride (= seq_len * v_tok_stride) differ from the contiguous [B,T,H,V] case.
    const float * k_chunk    = K + (long long) b * seq_len * HK + t_off * HK + h_k * K_dim;
    const float * v_chunk    = V + (long long) b * seq_len * v_tok_stride + t_off * v_tok_stride + h * V_dim;
    const float * beta_chunk = Beta + (long long) b * seq_len * H + t_off * H + h;
    const float * g_chunk    = G + (long long) b * seq_len * H + t_off * H + h;

    const long long out_k_base = (long long) pid_bh * num_chunks * CS * K_dim + (long long) pid_chunk * CS * K_dim;
    const long long out_v_base = (long long) pid_bh * num_chunks * CS * V_dim + (long long) pid_chunk * CS * V_dim;
    const long long out_g_base = (long long) pid_bh * num_chunks * CS + (long long) pid_chunk * CS;

    const int valid_cs = min(CS, seq_len - t_off);

    // Step 0: load K, beta, g into SMEM
    for (int i = tid; i < CS * BK; i += 128) {
        int t = i / BK, k = i % BK;
        s_K[t * SK + k] = (t < valid_cs) ? k_chunk[(long long) t * HK + k] : 0.f;
    }
    for (int i = tid; i < CS; i += 128) {
        s_beta[i] = (i < valid_cs) ? beta_chunk[(long long) i * H] : 0.f;
        s_gcum[i] = (i < valid_cs) ? g_chunk[(long long) i * H] : 0.f;
    }
    __syncthreads();

    // Step 1: g_cum prefix sum (serial, thread 0)
    if (tid == 0) {
        float acc = 0.f;
        for (int t = 0; t < CS; t++) {
            acc += s_gcum[t];
            s_gcum[t] = acc;
        }
    }
    __syncthreads();

    // Step 2: build L coupling matrix (exact FP32 scalar dot products)
    for (int idx = tid; idx < CS * CS; idx += 128) {
        const int t = idx / CS, s = idx % CS;
        if (s < t) {
            float kkt = 0.f;
            for (int k = 0; k < BK; k++) {
                kkt += s_K[t * SK + k] * s_K[s * SK + k];
            }
            s_L[idx] = s_beta[t] * __expf(s_gcum[t] - s_gcum[s]) * kkt;
        } else {
            s_L[idx] = 0.f;
        }
    }
    __syncthreads();

    // Step 3: write G_cum_out (K/Q are not staged -- later kernels read them raw)
    for (int i = tid; i < CS; i += 128) {
        G_cum_out[out_g_base + i] = s_gcum[i];
    }

    // Step 4: K_cumdecay via forward substitution
    {
        const int k_col = tid;
        if (k_col < K_dim) {
            float xreg[CS];
            for (int t = 0; t < CS; t++) {
                float xt = s_beta[t] * __expf(s_gcum[t]) * s_K[t * SK + k_col];
                for (int s = 0; s < t; s++) {
                    xt -= s_L[t * CS + s] * xreg[s];
                }
                xreg[t] = xt;
            }
            for (int t = 0; t < CS; t++) {
                K_cumdecay[out_k_base + (long long) t * K_dim + k_col] = xreg[t];
            }
        }
    }
    __syncthreads();

    // Step 5: V_corr via forward substitution (reuses s_K for V tile staging)
    const int num_vt = (V_dim + BK - 1) / BK;
    for (int vt = 0; vt < num_vt; vt++) {
        const int v_off  = vt * BK;
        const int v_cols = min(BK, V_dim - v_off);

        for (int i = tid; i < CS * BK; i += 128) {
            const int t = i / BK, v = i % BK;
            float     val = 0.f;
            if (t < valid_cs && v < v_cols) {
                val = v_chunk[(long long) t * v_tok_stride + v_off + v] * s_beta[t];
            }
            s_K[t * SK + v] = val;
        }
        __syncthreads();

        if (tid < v_cols) {
            float     xreg[CS];
            const int v_col = tid;
            for (int t = 0; t < CS; t++) {
                float xt = s_K[t * SK + v_col];
                for (int s = 0; s < t; s++) {
                    xt -= s_L[t * CS + s] * xreg[s];
                }
                xreg[t] = xt;
            }
            for (int t = 0; t < CS; t++) {
                V_corr[out_v_base + (long long) t * V_dim + (v_off + v_col)] = xreg[t];
            }
        }
        __syncthreads();
    }
}

// Masked Q@K^T on tensor cores (bf16 WMMA, one warp per block):
//   qk_buf[i,j] = (Q_ch . K_ch[j]) * exp(g_cum[i] - g_cum[j])   for j <= i, else 0.
// Grid (B*H, num_chunks); 32 threads. Requires CS==16, BK%16==0.
template <int CS, int BK>
__launch_bounds__(32, 8) __global__ void cgdr_precompute_qk_wmma_kernel(const float * __restrict__ Q_raw,
                                                                        const float * __restrict__ K_raw,
                                                                        const float * __restrict__ g_cum,
                                                                        float * __restrict__ qk_buf,
                                                                        int   num_chunks,
                                                                        float scale,
                                                                        int   H,
                                                                        int   num_k_heads,
                                                                        int   seq_len) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    static_assert(CS == 16, "WMMA preqk requires CS=16");
    static_assert(BK % 16 == 0, "BK must be a multiple of 16");
    constexpr int KT = BK / 16;

    // SMEM layout (no padding needed: bf16 tiles at BK=128 have no bank conflicts)
    extern __shared__ char _smem[];
    auto *                 s_Q    = reinterpret_cast<__nv_bfloat16 *>(_smem);
    auto *                 s_K    = reinterpret_cast<__nv_bfloat16 *>(_smem + CS * BK * sizeof(__nv_bfloat16));
    auto *                 s_gcum = reinterpret_cast<float *>(_smem + 2 * CS * BK * sizeof(__nv_bfloat16));
    auto *                 s_acc  = s_gcum + CS;  // fp32 [CSxCS] accumulator store

    const int bh  = blockIdx.x;
    const int c   = blockIdx.y;
    const int tid = threadIdx.x;  // 0..31

    const int       b_idx = bh / H;
    const int       h_idx = bh % H;                  // v-head
    const int       h_k   = h_idx % num_k_heads;           // GQA: v-head -> shared k-head
    const long long HK    = (long long) num_k_heads * BK;  // q/k token stride (un-repeated k-head count)
    const int       t_off = c * CS;

    const float * Q_chunk = Q_raw + (long long) b_idx * seq_len * HK + t_off * HK + h_k * BK;
    const float * K_chunk = K_raw + (long long) b_idx * seq_len * HK + t_off * HK + h_k * BK;

    if (tid < CS) {  // only CS of the 32 threads have work
        s_gcum[tid] = g_cum[(bh * num_chunks + c) * CS + tid];
    }

    // Load Q and K (float->bf16); each token row has stride HK. The last chunk may be partial when
    // seq_len is not a multiple of CS -- zero-fill rows past valid_cs to avoid out-of-bounds reads.
    const int valid_cs = min(CS, seq_len - t_off);
    for (int i = tid; i < CS * BK; i += 32) {
        const int   row = i / BK, col = i % BK;
        const float qv = (row < valid_cs) ? Q_chunk[(long long) row * HK + col] : 0.f;
        const float kv = (row < valid_cs) ? K_chunk[(long long) row * HK + col] : 0.f;
        s_Q[i]         = __float2bfloat16(qv * scale);
        s_K[i]         = __float2bfloat16(kv);
    }
    __syncthreads();

    // acc = Q @ K^T. Loading K col_major with ld=BK gives the transposed tile for free.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    #pragma unroll
    for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fA;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fB;
        wmma::load_matrix_sync(fA, s_Q + kt * 16, BK);
        wmma::load_matrix_sync(fB, s_K + kt * 16, BK);
        wmma::mma_sync(acc, fA, fB, acc);
    }
    wmma::store_matrix_sync(s_acc, acc, CS, wmma::mem_row_major);
    __syncthreads();

    // Causal mask + cumulative-decay scaling, then write qk_buf.
    float *       O_base = qk_buf + (long long) (bh * num_chunks + c) * CS * CS;
    constexpr int EPT    = CS * CS / 32;
    #pragma unroll
    for (int e = 0; e < EPT; e++) {
        const int flat = tid + e * 32;
        const int row  = flat / CS;
        const int col  = flat % CS;
        O_base[flat]   = (col <= row) ? s_acc[flat] * __expf(s_gcum[row] - s_gcum[col]) : 0.f;
    }
#else
    // bf16 WMMA requires SM80+ (Ampere); never launched on older GPUs (dispatch guards cc>=Ampere).
    (void) Q_raw;
    (void) K_raw;
    (void) g_cum;
    (void) qk_buf;
    (void) num_chunks;
    (void) scale;
    (void) H;
    (void) num_k_heads;
    (void) seq_len;
    __trap();
#endif
}

// Recurrent state update + fused output. Grid tiles V by BV; H stays in fp32 h_regs across chunks
// with a per-chunk bf16 copy (s_hbf16) for the WMMA B-operand. Matmuls are m16n16k16 WMMA (fp32
// accum). H SMEM is V-major [BV][BK], matching GGML [bh][v][k]. Grid (B*H, V_dim/BV); NT threads.

// fp32 -> fp16, clamped to finite fp16 range.
__device__ __forceinline__ __half cgdr_to_fp16(float v) {
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    return __float2half(v);
}

template <int CS, int BK, int BV, int NT, int OCC>
__launch_bounds__(NT, OCC) __global__ void cgdr_state_wmma_kernel(
    const float * __restrict__ V_corr,
    const float * __restrict__ K_cumdecay,
    const float * __restrict__ K_raw,   // raw K input [B,T,H,K]
    const float * __restrict__ Q_raw,   // raw Q input [B,T,H,K]
    const float * __restrict__ G_cum_in,
    const float * __restrict__ QK_buf,  // [B*H, num_chunks, CS, CS] -- fused output input
    float * __restrict__ Output,        // [B, T, H, V_dim] GGML layout -- direct output write
    const float * __restrict__ InitState,
    float * __restrict__ FinalState,
    float scale,  // Q scale = 1/sqrt(K_dim)
    int   num_chunks,
    int   H,
    int   num_k_heads,
    int   V_dim,
    int   seq_len) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
    using namespace nvcuda;
    static_assert(BK == 128, "bf16 state kernel requires BK=128");
    static_assert(CS == 16, "bf16 state kernel requires CS=16");
    static_assert(BV % 16 == 0, "BV must be a multiple of 16");
    static_assert(NT % 32 == 0, "NT must be a multiple of warp size");
    static_assert((CS * BV) % NT == 0, "CS*BV must be divisible by NT");
    static_assert((BK * BV) % NT == 0, "BK*BV must be divisible by NT");
    static_assert(NT / 32 >= BV / 16, "need at least BV/16 warps for WMMA n-tiles");

    // SMEM layout:
    //   s_hbf16[BKxBV]  bf16  -- H staging for steps 2b/2.5 WMMA B-operand
    //   s_kbuf_bf16[CSxBK] bf16 -- K_cumdecay/Q (steps 2a/2.5); same buffer reused as fp16 s_kch/qkb
    //   s_result[CSxBV]  fp32  -- WMMA accumulator; reused as s_vnew fp16 (step 4)
    //   s_gcum[CS]       fp32
    //   s_hdelta[BKxBV]  fp32  -- step-5 delta / output WMMA scratch (V-major)
    constexpr int H_BYTES    = BK * BV * (int) sizeof(__nv_bfloat16);
    constexpr int KBUF_BYTES = CS * BK * (int) sizeof(__nv_bfloat16);
    constexpr int RES_BYTES  = CS * BV * (int) sizeof(float);
    constexpr int GCUM_BYTES = CS * (int) sizeof(float);

    extern __shared__ char _smem_bf16[];
    __nv_bfloat16 *        s_hbf16     = reinterpret_cast<__nv_bfloat16 *>(_smem_bf16);
    __nv_bfloat16 *        s_kbuf_bf16 = reinterpret_cast<__nv_bfloat16 *>(_smem_bf16 + H_BYTES);
    float *                s_result    = reinterpret_cast<float *>(_smem_bf16 + H_BYTES + KBUF_BYTES);
    float *                s_gcum      = reinterpret_cast<float *>(_smem_bf16 + H_BYTES + KBUF_BYTES + RES_BYTES);
    float *  s_hdelta = reinterpret_cast<float *>(_smem_bf16 + H_BYTES + KBUF_BYTES + RES_BYTES + GCUM_BYTES);
    // fp16 operands for the step-5 (k^T v) and output WMMAs, aliasing the bf16/fp32 SMEM above:
    __half * s_kch    = reinterpret_cast<__half *>(s_kbuf_bf16);  // step 3 stages K_raw (reuses the 2a/2.5 buffer)
    __half * s_vnew   = reinterpret_cast<__half *>(s_result);     // step 4 stages v_new (overlays fp32 WMMA result)

    const int     pid_bh  = blockIdx.x;
    const int     tile_v  = blockIdx.y;
    const int     v_off   = tile_v * BV;
    const int     tid     = threadIdx.x;
    const int     warp_id = tid / 32;
    constexpr int EPT     = (CS * BV) / NT;
    constexpr int EPT_H   = (BK * BV) / NT;
    constexpr int N_TILES = BV / 16;

    const long long bh_off = pid_bh;
    const long long off_k  = bh_off * (long long) num_chunks * CS * BK;
    const long long off_v  = bh_off * (long long) num_chunks * CS * V_dim;

    const float * Vcorr_base = V_corr + off_v;
    const float * Kcd_base   = K_cumdecay + off_k;
    const float * Gcum_base  = G_cum_in + bh_off * (long long) num_chunks * CS;

    const int       b_idx   = pid_bh / H;
    const int       h_idx   = pid_bh % H;            // v-head
    const int       h_k     = h_idx % num_k_heads;           // GQA: v-head -> shared k-head
    const long long HK      = (long long) num_k_heads * BK;  // q/k token stride (un-repeated k-head count)
    const long long T_total = seq_len;                       // actual token count (NOT num_chunks*CS, which rounds up)
    float *       out_bh = Output + (long long) b_idx * T_total * H * V_dim + (long long) h_idx * V_dim + v_off;
    const float * Kraw_bh = K_raw + (long long) b_idx * seq_len * HK + h_k * BK;
    const float * Qraw_bh = Q_raw + (long long) b_idx * seq_len * HK + h_k * BK;

    // FP32 H state in thread registers -- persistent across all chunks (no per-chunk bf16 rounding).
    float h_regs[EPT_H];

    // Initialize h_regs from InitState (GDN always has an input recurrent state s0), then prime
    // s_hbf16 for the first chunk's WMMA.
    {
        const long long src_base = bh_off * (long long) V_dim * BK;
        for (int j = 0; j < EPT_H; j++) {
            const int idx = tid + j * NT;
            h_regs[j]     = InitState[src_base + (idx / BK + v_off) * BK + (idx % BK)];
            s_hbf16[idx]  = __float2bfloat16(h_regs[j]);
        }
    }
    __syncthreads();

    for (int ci = 0; ci < num_chunks; ci++) {
        const float * vcorr_ptr  = Vcorr_base + (long long) ci * CS * V_dim;
        const float * kcd_ptr    = Kcd_base + (long long) ci * CS * BK;
        const float * gcum_ptr   = Gcum_base + (long long) ci * CS;
        const float * kraw_chunk = Kraw_bh + (long long) ci * CS * HK;
        const float * qraw_chunk = Qraw_bh + (long long) ci * CS * HK;
        const int     valid_cs   = min(CS, seq_len - ci * CS);  // < CS on the last chunk if seq_len % CS != 0
        float         vnew_regs[EPT];
        float         oi_regs[EPT];

        // Step 1a: G_cum -> s_gcum
        for (int i = tid; i < CS; i += NT) {
            s_gcum[i] = gcum_ptr[i];
        }

        // Step 2a: K_cumdecay -> s_kbuf_bf16 (fp32 -> bf16, no clamp -- bf16 has fp32 range)
        for (int i = tid; i < CS * BK; i += NT) {
            s_kbuf_bf16[i] = __float2bfloat16(kcd_ptr[i]);
        }

        __syncthreads();

        // Step 2b: s_result[CSxBV] = WMMA(s_kbuf_bf16 @ s_hbf16)   (v_new = u - w*h)
        if (warp_id < N_TILES) {
            const int                                            n_off = warp_id * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            #pragma unroll
            for (int k = 0; k < BK; k += 16) {
                wmma::load_matrix_sync(a_frag, s_kbuf_bf16 + k, BK);
                wmma::load_matrix_sync(b_frag, s_hbf16 + n_off * BK + k, BK);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(s_result + n_off, acc, BV, wmma::mem_row_major);
        }

        __syncthreads();

        // V_new = V_corr - s_result; kept in registers.
        for (int j = 0; j < EPT; j++) {
            const int idx   = tid + j * NT;
            const int t_idx = idx / BV;
            const int v_loc = idx % BV;
            const int g_idx = t_idx * V_dim + v_off + v_loc;
            float     vn    = vcorr_ptr[g_idx] - s_result[t_idx * BV + v_loc];
            if (!isfinite(vn)) {
                vn = 0.f;
            }
            vnew_regs[j] = vn;
        }

        // Step 2.5: Q_raw -> s_kbuf_bf16, then WMMA for O_inter (o_inter = (q*scale) @ h)
        __syncthreads();

        for (int i = tid; i < CS * BK; i += NT) {
            const int   t = i / BK, k = i % BK;
            const float qv = (t < valid_cs) ? qraw_chunk[(long long) t * HK + k] : 0.f;
            s_kbuf_bf16[i] = __float2bfloat16(qv * scale);
        }

        __syncthreads();

        if (warp_id < N_TILES) {
            const int                                            n_off = warp_id * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            #pragma unroll
            for (int k = 0; k < BK; k += 16) {
                wmma::load_matrix_sync(a_frag, s_kbuf_bf16 + k, BK);
                wmma::load_matrix_sync(b_frag, s_hbf16 + n_off * BK + k, BK);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(s_result + n_off, acc, BV, wmma::mem_row_major);
        }

        __syncthreads();

        // O_inter = s_result x exp(g_cum[t]); kept in registers.
        for (int j = 0; j < EPT; j++) {
            const int idx   = tid + j * NT;
            const int t_idx = idx / BV;
            const int v_loc = idx % BV;
            float     oi    = s_result[t_idx * BV + v_loc] * __expf(fminf(s_gcum[t_idx], 88.72f));
            if (!isfinite(oi)) {
                oi = 0.f;
            }
            oi_regs[j] = oi;
        }

        // Step 3: K_raw -> s_kch fp16 (row-major [CS][BK], decay-scaled by exp(g_last-g_cum[t]))
        __syncthreads();

        const float g_last = s_gcum[CS - 1];
        for (int i = tid; i < BK * CS; i += NT) {
            const int   k     = i % BK;
            const int   t     = i / BK;
            const float kv    = (t < valid_cs) ? kraw_chunk[(long long) t * HK + k] : 0.f;
            s_kch[t * BK + k] = cgdr_to_fp16(kv * __expf(g_last - s_gcum[t]));
        }

        // Step 4: vnew_regs -> s_vnew 16-bit (no global read)
        for (int j = 0; j < EPT; j++) {
            s_vnew[tid + j * NT] = cgdr_to_fp16(vnew_regs[j]);
        }

        __syncthreads();

        // Step 5: delta[v][k] = sum_t Vnew[t][v] * Kch[t][k].
        //   A = Vnew^T (col_major s_vnew, lda=BV), B = Kch (row_major s_kch, ldb=BK);
        //   stored row_major into V-major s_hdelta[v*BK+k] so it matches h_regs indexing.
        if (warp_id < N_TILES) {
            const int                                                           m_off = warp_id * 16;  // v-tile base
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, s_vnew + m_off, BV);
            #pragma unroll
            for (int nk = 0; nk < BK; nk += 16) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.f);
                wmma::load_matrix_sync(b_frag, s_kch + nk, BK);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
                wmma::store_matrix_sync(s_hdelta + m_off * BK + nk, acc, BK, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // H = exp(g_last)*H + delta  (fp32 accumulation preserved in h_regs)
        const float exp_g = __expf(g_last);
        for (int j = 0; j < EPT_H; j++) {
            h_regs[j] = exp_g * h_regs[j] + s_hdelta[tid + j * NT];
            if (!isfinite(h_regs[j])) {
                h_regs[j] = 0.f;
            }
        }
        __syncthreads();  // all reads of s_hdelta done before output WMMA overwrites it

        // Refresh s_hbf16 from fp32 h_regs for the next chunk's B-matrix (no clamp -- bf16 fp32-range).
        for (int j = 0; j < EPT_H; j++) {
            s_hbf16[tid + j * NT] = __float2bfloat16(h_regs[j]);
        }

        // Output (fp16 WMMA): O[t][v] = O_inter + sum_t' qk[t][t'] * Vnew[t'][v]
        //   qk loaded fp32 from global, downcast to fp16.
        {
            __half *      s_qkb   = reinterpret_cast<__half *>(s_kbuf_bf16);  // reuse kbuf: [CS][CS] fp16
            const float * qk_base = QK_buf + (bh_off * (long long) num_chunks + ci) * CS * CS;
            for (int i = tid; i < CS * CS; i += NT) {
                s_qkb[i] = cgdr_to_fp16(qk_base[i]);
            }
            __syncthreads();

            // A = qk (row_major, lda=CS), B = Vnew (row_major s_vnew, ldb=BV).
            if (warp_id < N_TILES) {
                const int                                            n_off = warp_id * 16;  // v-tile base
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.f);
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, s_qkb, CS);
                wmma::load_matrix_sync(b_frag, s_vnew + n_off, BV);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
                wmma::store_matrix_sync(s_hdelta + n_off, acc, BV, wmma::mem_row_major);
            }
            __syncthreads();

            // O = O_intra (s_hdelta[CS][BV] row-major) + O_inter (oi_regs). Skip padding tokens on
            // the last (partial) chunk -- their output row is past seq_len (out-of-bounds write).
            float * out_chunk = out_bh + (long long) ci * CS * H * V_dim;
            #pragma unroll
            for (int j = 0; j < EPT; j++) {
                const int idx = tid + j * NT;
                const int t_p = idx / BV;
                const int v_p = idx % BV;
                if (t_p < valid_cs) {
                    out_chunk[(long long) t_p * H * V_dim + v_p] = s_hdelta[t_p * BV + v_p] + oi_regs[j];
                }
            }
        }

        __syncthreads();  // ensure s_hbf16 refresh + output done before next chunk

    }  // end chunk loop

    // Write final H state to GGML V-major [bh][v][k] (fp32 from h_regs, no conversion loss).
    {
        float * dst = FinalState + bh_off * (long long) V_dim * BK;
        for (int j = 0; j < EPT_H; j++) {
            const int idx                             = tid + j * NT;
            dst[(idx / BK + v_off) * BK + (idx % BK)] = h_regs[j];
        }
    }
#else
    // bf16 WMMA needs SM80+; the dispatch predicate keeps this off pre-Ampere GPUs. Compiling the
    // body out lets sm_75 etc. build, and __trap() catches any launch that slips through.
    (void) V_corr;
    (void) K_cumdecay;
    (void) K_raw;
    (void) Q_raw;
    (void) G_cum_in;
    (void) QK_buf;
    (void) Output;
    (void) InitState;
    (void) FinalState;
    (void) scale;
    (void) num_chunks;
    (void) H;
    (void) num_k_heads;
    (void) V_dim;
    (void) seq_len;
    __trap();
#endif
}

// Dynamic SMEM bytes per kernel launch (the <<<>>> third argument).
static inline size_t cgdr_smem_fwdsub_intra(int CS, int BK) {
    return ((size_t) CS * (BK + 1) + (size_t) CS * CS + 2 * (size_t) CS) * sizeof(float);
}

static inline size_t cgdr_smem_preqk_wmma(int CS, int BK) {
    // s_Q[CSxBK] bf16 + s_K[CSxBK] bf16 + s_gcum[CS] fp32 + s_acc[CSxCS] fp32
    return (size_t) 2 * CS * BK * sizeof(__nv_bfloat16) + (size_t) (CS + CS * CS) * sizeof(float);
}

static inline size_t cgdr_smem_state_wmma(int CS, int BK, int BV) {
    size_t s_h      = (size_t) BK * BV * sizeof(__nv_bfloat16);  // bf16 H state (B-matrix)
    size_t s_kbuf   = (size_t) CS * BK * sizeof(__nv_bfloat16);  // bf16 kbuf
    size_t s_res    = (size_t) CS * BV * sizeof(float);          // fp32 WMMA result (2b/2.5); s_vnew overlay
    size_t s_gcum   = (size_t) CS * sizeof(float);               // fp32 gcum
    size_t s_hdelta = (size_t) BK * BV * sizeof(float);          // fp32 WMMA scratch (step 5 delta / output)
    return s_h + s_kbuf + s_res + s_gcum + s_hdelta;
}

// Launches the three-stage pipeline. CS/BK are hardwired to 16/128 (eligibility guarantees
// K_dim==128; partial final chunks are handled via valid_cs in the kernels).
static void ggml_cuda_op_gated_delta_net_chunked_impl(ggml_backend_cuda_context & ctx,
                                                      ggml_tensor *               dst,
                                                      int                         B,
                                                      int                         T,
                                                      int                         H,
                                                      int                         num_k_heads,
                                                      int                         K_dim,
                                                      int                         V_dim,
                                                      int                         num_chunks,
                                                      const float *               q_in,
                                                      const float *               k_in,
                                                      const float *               v_in,
                                                      const float *               g_in,
                                                      const float *               b_in,
                                                      const float *               s_d,
                                                      float                       scale,
                                                      long long                   v_tok_stride,
                                                      cudaStream_t                stream) {
    ggml_cuda_pool & pool = ctx.pool();

    constexpr int            CS               = 16;
    // Pad each scratch buffer by a small tail. WMMA store_matrix_sync writes a full 16-wide tile; on
    // the last chunk/tile a store can touch a few elements past the logical size, so over-allocate to
    // keep those writes inside the pool block (128 B is comfortably more than one tile's overrun).
    static constexpr int64_t POOL_GUARD_ELEMS = 32;
    const int64_t            cs_v             = (int64_t) B * H * num_chunks * CS * V_dim;
    const int64_t            cs_k             = (int64_t) B * H * num_chunks * CS * K_dim;
    const int64_t            cs_g             = (int64_t) B * H * num_chunks * CS;
    const int64_t            cs_qk            = (int64_t) B * H * num_chunks * CS * CS;

    ggml_cuda_pool_alloc<float> v_corr_buf(pool, cs_v + POOL_GUARD_ELEMS);
    ggml_cuda_pool_alloc<float> k_cumdecay_buf(pool, cs_k + POOL_GUARD_ELEMS);
    ggml_cuda_pool_alloc<float> g_cum_buf(pool, cs_g + POOL_GUARD_ELEMS);
    ggml_cuda_pool_alloc<float> qk_buf(pool, cs_qk + POOL_GUARD_ELEMS);

    // Stage 1 -- intra pass: exact FP32 forward substitution -> V_corr, K_cumdecay, G_cum.
    {
        const size_t fs_smem = cgdr_smem_fwdsub_intra(CS, K_dim);  // 9.2 KB < 48 KB -> no opt-in needed
        const dim3   intra_grid(B * H, num_chunks, 1);
        cgdr_fwdsub_intra_kernel<CS, 128><<<intra_grid, 128, fs_smem, stream>>>(
            k_in, v_in, b_in, g_in, v_corr_buf.get(), k_cumdecay_buf.get(), g_cum_buf.get(), B, T, H, num_chunks, K_dim,
            V_dim, num_k_heads, v_tok_stride);
    }
    CUDA_CHECK(cudaGetLastError());

    // Stage 2 -- preqk pass: masked Q@K^T with BF16 WMMA (32 threads/block, tensor cores).
    {
        const size_t qk_smem = cgdr_smem_preqk_wmma(CS, K_dim);  // 9.1 KB < 48 KB -> no opt-in needed
        const dim3   qk_grid(B * H, num_chunks, 1);
        cgdr_precompute_qk_wmma_kernel<CS, 128><<<qk_grid, 32, qk_smem, stream>>>(
            q_in, k_in, g_cum_buf.get(), qk_buf.get(), num_chunks, scale, H, num_k_heads, T);
    }
    CUDA_CHECK(cudaGetLastError());

    // Stage 3 -- state+output pass: WMMA tensor cores, fixed tile (BV=32/NT=256/OCC=4). ~30 KB
    // dynamic SMEM, under the 48 KB default, so no cudaFuncAttribute opt-in needed.
    const int64_t state_offset = (int64_t) V_dim * H * T * B;
    float *       state_dst    = (float *) dst->data + state_offset;
    {
        constexpr int BV = 32, NT = 256, OCC = 4;
        const size_t  st_smem = cgdr_smem_state_wmma(CS, 128, BV);
        const dim3    state_grid(B * H, V_dim / BV, 1);
        cgdr_state_wmma_kernel<CS, 128, BV, NT, OCC><<<state_grid, NT, st_smem, stream>>>(
            v_corr_buf.get(), k_cumdecay_buf.get(), k_in, q_in, g_cum_buf.get(), qk_buf.get(), (float *) dst->data,
            s_d, state_dst, scale, num_chunks, H, num_k_heads, V_dim, T);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Public entry: validates the op, extracts dims, and runs the pipeline. Selected by the
// eligibility check in gated_delta_net.cu; otherwise the recurrent path runs.
void ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    const int V_dim = (int) nev0;
    const int H     = (int) nev1;
    const int T     = (int) nev2;
    const int B     = (int) nev3;

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    const int K_dim       = (int) neq0;
    const int num_k_heads = (int) neq1;  // q/k head count; <= H (v-head count) for GQA

    GGML_ASSERT(ggml_is_contiguous(src_q));
    GGML_ASSERT(ggml_is_contiguous(src_k));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    // V may be a strided view of the fused QKV buffer (no cont in the model graph). The intra kernel
    // handles an arbitrary token stride, but still needs V contiguous within a token: elements packed
    // (nb0 == elt) and v-heads packed (nb1 == V_dim*elt). Only the token stride (nb2) may differ.
    const size_t vsz = ggml_type_size(src_v->type);
    GGML_ASSERT(src_v->nb[0] == vsz && src_v->nb[1] == (size_t) V_dim * vsz &&
                "chunked GDN requires V contiguous within a token");
    const long long v_tok_stride = (long long) (src_v->nb[2] / vsz);
    // The state kernel uses BK=128 as the state's key-row stride, but GGML stores the state square
    // ([S_v, S_v] per head), so the layouts only line up at K_dim == V_dim == 128. The eligibility
    // predicate already guarantees this; re-assert so a future dispatch bug fails loudly instead of
    // silently corrupting state cells.
    GGML_ASSERT(K_dim == 128 && V_dim == 128 && "chunked GDN requires K_dim == V_dim == 128");
    // GQA: Q and K share a head count (num_k_heads); each v-head maps to k-head (h_v % num_k_heads),
    // which only tiles cleanly when the v-head count H is a multiple of num_k_heads.
    GGML_ASSERT(src_k->ne[1] == num_k_heads && "chunked GDN: Q and K must have the same head count");
    GGML_ASSERT(H % num_k_heads == 0 && "chunked GDN: v-head count must be a multiple of k-head count");
    // T need not be a multiple of CS=16: the last chunk may be partial and the kernels guard it.

    // 1/sqrt(S_v), matching the recurrent kernel and the CPU/OpenVINO references (== 1/sqrt(K_dim)
    // here, since K_dim == V_dim).
    const float  scale  = 1.0f / sqrtf((float) V_dim);
    cudaStream_t stream = ctx.stream();

    const float * s_d  = (const float *) src_state->data;
    const float * q_in = (const float *) src_q->data;
    const float * k_in = (const float *) src_k->data;
    const float * v_in = (const float *) src_v->data;
    const float * g_in = (const float *) src_g->data;
    const float * b_in = (const float *) src_beta->data;

    // Recurrent (gated_delta_net.cu) handles everything this path can't
    // (kda, K>1, K!=128, non-contiguous, single-token decode).
    // num_chunks = ceil(T/CS): the last chunk may be partial; the kernels guard the padding tokens.
    const int num_chunks = (T + 15) / 16;
    ggml_cuda_op_gated_delta_net_chunked_impl(ctx, dst, B, T, H, num_k_heads, K_dim, V_dim, num_chunks, q_in, k_in,
                                              v_in, g_in, b_in, s_d, scale, v_tok_stride, stream);
}

#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("chunked GDN is not supported on HIP/MUSA");
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
