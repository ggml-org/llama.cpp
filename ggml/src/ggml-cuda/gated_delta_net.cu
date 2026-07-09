#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mma.cuh"

#include <cctype>

// The chunked prefill path expresses its three GEMMs through the shared ggml_cuda_mma tile
// interface (mma.cuh) so the *same source* runs on NVIDIA tensor cores (Turing+) and AMD MFMA
// (CDNA). The generic mma(tile<16,16,float>, tile<16,8,half2>, tile<16,8,half2>) primitive gives
// f16 inputs with f32 accumulation on both. Architectures whose input fragment layout is not the
// plain I_MAJOR one used here (Volta, RDNA3) fall back to the scalar path below, as do CPU builds.
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
#define FGDN_TILE_MMA 1
#endif

#ifdef FGDN_TILE_MMA
// Output (matrix C) fragment layout: J_MAJOR on CDNA/RDNA4, I_MAJOR on NVIDIA. get_i/get_j on the
// tile report where each thread's accumulator element lands, so loads/stores stay arch-agnostic.
#if defined(AMD_MFMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
static constexpr ggml_cuda_mma::data_layout FGDN_C_DL = ggml_cuda_mma::DATA_LAYOUT_J_MAJOR;
#else
static constexpr ggml_cuda_mma::data_layout FGDN_C_DL = ggml_cuda_mma::DATA_LAYOUT_I_MAJOR;
#endif

// Load a 16×16 f16 mma input fragment directly from an f32 LDS buffer, converting on the fly (no
// half staging buffer). The logical operand element (row, kcol) lives at base[row*row_stride +
// kcol*k_stride]; a half2 lane packs two adjacent contraction elements. Choosing (row_stride,
// k_stride) lets the same routine read a matrix or its transpose straight out of the phase LDS:
//   K[t][i]  : base=K_lds, row_stride=S_v, k_stride=1        (contract i)
//   Sᵀ[c][i] : base=S_lds, row_stride=1,   k_stride=N_COLS   (contract i, transposed read)
//   Kᵀ[i][s] : base=K_lds, row_stride=1,   k_stride=S_v      (contract s, transposed read)
static __device__ __forceinline__ void fgdn_load_frag(
        ggml_cuda_mma::tile<16, 8, half2, ggml_cuda_mma::DATA_LAYOUT_I_MAJOR> & t,
        const float * __restrict__ base, const int row_stride, const int k_stride, const int k0) {
#pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        const int i = t.get_i(l);
        const int k = k0 + 2 * t.get_j(l);
        t.x[l] = __floats2half2_rn(base[i * row_stride + k * k_stride],
                                   base[i * row_stride + (k + 1) * k_stride]);
    }
}

// C[16][16] += A(16 × 16·nk) @ B(16 × 16·nk)ᵀ, contracting the shared 16·nk dimension, reading both
// operands directly from f32 LDS (see fgdn_load_frag). f16 inputs, f32 accumulation, on tensor
// cores / MFMA via the shared ggml_cuda_mma primitive.
static __device__ __forceinline__ void fgdn_mma16(
        ggml_cuda_mma::tile<16, 16, float, FGDN_C_DL> & C,
        const float * __restrict__ a, const int a_row_stride, const int a_k_stride,
        const float * __restrict__ b, const int b_row_stride, const int b_k_stride, const int nk) {
    using namespace ggml_cuda_mma;
#pragma unroll 1
    for (int kb = 0; kb < nk; ++kb) {
        tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> A;
        tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> B;
        fgdn_load_frag(A, a, a_row_stride, a_k_stride, kb * 16);
        fgdn_load_frag(B, b, b_row_stride, b_k_stride, kb * 16);
        mma(C, A, B);
    }
}
#endif // FGDN_TILE_MMA

// =====================================================================================
// Chunked prefix-scan kernel for gated DeltaNet (non-KDA, S_v=128). The GEMMs run on tensor
// cores / MFMA via the shared ggml_cuda_mma interface (see Part 1 above).
//
// Algorithm (FLA-style chunked delta rule, derived for the scalar-gate variant):
//
// Per-token recurrence (matches the original kernel):
//     S_t  = exp(g_t) * (I - β_t · k_t k_t^T) · S_{t-1}  +  β_t · k_t · v_t^T
//     attn_t = (1/sqrt(S_v)) · q_t^T · S_t
//
// This is equivalent to a delta rule with effective gate g̃_t = exp(g_t):
//     S_t  = g̃_t · S_{t-1}  +  β_t · k_t · δ_t^T,  where δ_t = v_t - g̃_t · (k_t^T S_{t-1})
//
// Within a chunk [t0, t0+C), let cumulative gate G_t = ∏_{s=t0+1..t} g̃_s (G_{t0} = 1)
// and decay D[t,s] = G_t / G_s. Then:
//
//     u_t   = v_t - G_t · (k_t^T S_{t0})                                 -- "external" RHS
//     L[t,s] = D[t,s] · β_s · (k_t · k_s)   for s < t in chunk           -- intra-chunk coupling
//     (I + L) · Δ = U                                                    -- C×C lower-tri solve
//                                                                           (parallel across S_v cols)
//     S_C   = G_C · S_{t0}  +  Σ_{s=1..C} D[C,s] · β_s · k_s · δ_s^T     -- new state
//     attn_t = (1/sqrt(S_v)) · ( G_t · q_t^T S_{t0}
//                              + Σ_{s≤t} D[t,s] · β_s · (q_t · k_s) · δ_s^T )
//
// It replaces the per-token kernel's serial inner loop with one chunk iteration of batched work:
// K/Q are loaded once per chunk and reused across KKᵀ/QKᵀ and the K@S/Q@S projections, and the
// three matmuls run f16-in/f32-acc on tensor cores / MFMA. Short sequences (e.g. MTP-verify) keep
// the per-token kernel; dispatch is on n_tokens (see Part 3).
//
// Layout: the block owns N_COLS columns of S, resident in LDS as S_lds[i*N_COLS+c]; the grid z-dim
// (S_v/N_COLS) splits each head's columns across blocks. Shared Phases 1-5 are computed once per
// block; the triangular solve (Phase 7) runs column-parallel. Inputs are L2-normalised so f16 GEMMs
// stay within op tolerance.
//
// Scope: non-KDA and S_v=128 only (Qwen3.6); everything else falls back to the per-token kernel.
// keep_rs / MTP: supported. For K = n_rs_seq+1 > 1 the chunk loop also materialises the per-token
// state for the last K global tokens (the only ones the recurrent cache keeps) into the K snapshot
// slots; the bulk of the prefix still runs as full chunks, so the speedup is kept.
// =====================================================================================
template <int S_v, int CHUNK_C, int N_COLS, bool keep_rs_t>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * 4, 1)
gated_delta_net_chunked_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ curr_state,
        float       * __restrict__ dst,
        float       * __restrict__ state,
        int64_t H,
        int64_t n_tokens,
        int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic,
        const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K) {
    constexpr int num_warps = 4;
    static_assert(S_v % N_COLS == 0, "S_v must be a multiple of N_COLS");

    const int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp     = threadIdx.y;
    const int      col_base = blockIdx.z * N_COLS;   // first state column this block owns

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data_base            = dst + (sequence * n_tokens * H + h_idx) * S_v;
    const float * curr_state_base     = curr_state + (sequence * H + h_idx) * S_v * S_v;
    float * state_out_base            = state + (sequence * H + h_idx) * S_v * S_v;

    // LDS layout. All C×C arrays use CHUNK_C stride (so partial-chunk C<CHUNK_C indexes match
    // the fixed-size 16×16 tile store in Phase 4). S_lds holds this block's N_COLS columns of S.
    __shared__ float K_lds [CHUNK_C * S_v];
    __shared__ float Q_lds [CHUNK_C * S_v];
    __shared__ float beta_lds[CHUNK_C];
    // Glog_lds[t] = sum_{s<=t in chunk} g_s, kept in log-space (the model emits g in log-space;
    // the per-token kernel does expf(*g_t)). We never materialise exp(cumsum) as one value: for
    // Qwen3.6 gates the cumsum can reach large negative magnitudes where expf underflows to 0 and
    // a naive G[t]/G[s] becomes 0/0=NaN. Instead only the *bounded* difference Glog[t]-Glog[s] is
    // exponentiated (D[t,s]), which is bounded by (t-s)*max|log g| ≤ 16*max|log g|.
    __shared__ float Glog_lds[CHUNK_C];                   // cumsum_{s≤t} log g_s within chunk
    __shared__ float D_lds [CHUNK_C * CHUNK_C];           // D[t,s] = expf(Glog[t]-Glog[s]), s≤t
    __shared__ float KK_lds[CHUNK_C * CHUNK_C];           // k_t · k_s
    __shared__ float QK_lds[CHUNK_C * CHUNK_C];           // q_t · k_s
    __shared__ float L_lds [CHUNK_C * CHUNK_C];           // D[t,s] β_s KK[t,s] for s<t (else 0)
    __shared__ float S_lds [S_v * N_COLS];                // owned columns of state S (resident)
    __shared__ float delta_lds[CHUNK_C * N_COLS];         // u then Δ, C×N_COLS
    __shared__ float qS_lds[CHUNK_C * N_COLS];            // q_t·S projection, C×N_COLS (Phase 6 -> 8)
    __shared__ float W_lds [CHUNK_C * N_COLS];            // W[s,c]=D[C-1,s]·β_s·Δ[s,c] for state update

#ifdef FGDN_TILE_MMA
    // The tile path needs the canonical 16×16 shapes; other geometry uses the scalar path. Operands
    // are read straight from the f32 phase LDS (fgdn_load_frag), so no half staging buffers.
    constexpr bool tile_ok = (CHUNK_C == 16) && (S_v % 16 == 0) && (N_COLS == 16);
#endif

    const int tid_in_block = warp * warp_size + lane;
    const int threads_per_block = num_warps * warp_size;

    const float * k_base    = k + iq3 * sq3 + iq1 * sq1;
    const float * q_base    = q + iq3 * sq3 + iq1 * sq1;
    const float * v_base    = v + sequence * sv3 + h_idx * sv1;
    const int64_t gb_offset = sequence * sb3 + h_idx * sb1;
    const float * beta_base = beta + gb_offset;
    const float * g_base    = g    + gb_offset;           // non-KDA: scalar per (head,seq,t)

    // ---- Load owned columns of S into LDS: S_lds[i*N_COLS + c] = S[i][col_base+c] ----
    for (int idx = tid_in_block; idx < S_v * N_COLS; idx += threads_per_block) {
        const int i = idx / N_COLS;
        const int c = idx % N_COLS;
        S_lds[i * N_COLS + c] = curr_state_base[(col_base + c) * S_v + i];
    }
    __syncthreads();

    for (int chunk_start = 0; chunk_start < n_tokens; chunk_start += CHUNK_C) {
        const int C = (n_tokens - chunk_start < CHUNK_C) ? (int)(n_tokens - chunk_start) : CHUNK_C;

        // ---- Phase 1: load K, Q chunks into LDS ----
        for (int idx = tid_in_block; idx < C * S_v; idx += threads_per_block) {
            const int t = idx / S_v;
            const int i = idx % S_v;
            const int t_global = chunk_start + t;
            K_lds[t * S_v + i] = k_base[t_global * sq2 + i];
            Q_lds[t * S_v + i] = q_base[t_global * sq2 + i];
        }
        for (int t = tid_in_block; t < C; t += threads_per_block) {
            const int t_global = chunk_start + t;
            beta_lds[t] = beta_base[t_global * sb2];
        }
        // Zero-pad partial-chunk rows [C, CHUNK_C) so the fixed-size 16×16 tile in Phase 4 reads
        // defined LDS. Outputs for those rows are never consumed. No-op when C==CHUNK_C.
        for (int idx = tid_in_block; idx < (CHUNK_C - C) * S_v; idx += threads_per_block) {
            const int t = C + idx / S_v;
            const int i = idx % S_v;
            K_lds[t * S_v + i] = 0.0f;
            Q_lds[t * S_v + i] = 0.0f;
        }
        // ---- Phase 2: cumulative log-gate Glog[t] = sum_{s≤t} log g_s (intra-chunk) ----
        if (tid_in_block == 0) {
            float acc = 0.0f;
            for (int t = 0; t < C; t++) {
                acc += g_base[(chunk_start + t) * sb2];
                Glog_lds[t] = acc;
            }
        }
        __syncthreads();

        // ---- Phase 3: D[t,s] = expf(Glog[t] - Glog[s]) for s ≤ t (else 0) ----
        for (int idx = tid_in_block; idx < C * C; idx += threads_per_block) {
            const int t = idx / C;
            const int s = idx % C;
            D_lds[t * CHUNK_C + s] = (s <= t) ? expf(Glog_lds[t] - Glog_lds[s]) : 0.0f;
        }

        // ---- Phase 4: KK[t,s] = k_t · k_s,  QK[t,s] = q_t · k_s ----
        // Two independent 16×16 GEMMs (KK = K@Kᵀ, QK = Q@Kᵀ), contracting S_v. The tile path runs
        // them through ggml_cuda_mma (tensor core / MFMA); the scalar path is the portable fallback.
#ifdef FGDN_TILE_MMA
        if constexpr (tile_ok) {
            if (warp < 2) {   // warp 0: KK, warp 1: QK
                ggml_cuda_mma::tile<16, 16, float, FGDN_C_DL> Cacc;
                const float * Amat = (warp == 0) ? K_lds : Q_lds;
                fgdn_mma16(Cacc, Amat, S_v, 1, K_lds, S_v, 1, S_v / 16);
                float * out = (warp == 0) ? KK_lds : QK_lds;
#pragma unroll
                for (int l = 0; l < Cacc.ne; ++l) {
                    out[Cacc.get_i(l) * CHUNK_C + Cacc.get_j(l)] = Cacc.x[l];
                }
            }
        } else
#endif // FGDN_TILE_MMA
        {
            for (int idx = tid_in_block; idx < C * C; idx += threads_per_block) {
                const int t = idx / C;
                const int s = idx % C;
                float kk = 0.0f, qk = 0.0f;
#pragma unroll
                for (int i = 0; i < S_v; i++) {
                    const float ks = K_lds[s * S_v + i];
                    kk += K_lds[t * S_v + i] * ks;
                    qk += Q_lds[t * S_v + i] * ks;
                }
                KK_lds[t * CHUNK_C + s] = kk;
                QK_lds[t * CHUNK_C + s] = qk;
            }
        }
        __syncthreads();

        // ---- Phase 5: L[t,s] = D[t,s] β_s KK[t,s] for s<t, else 0 ----
        for (int idx = tid_in_block; idx < C * C; idx += threads_per_block) {
            const int t = idx / C;
            const int s = idx % C;
            L_lds[t * CHUNK_C + s] = (s < t) ? (D_lds[t * CHUNK_C + s] * beta_lds[s] * KK_lds[t * CHUNK_C + s]) : 0.0f;
        }
        __syncthreads();

        // ---- Phase 6: kS = K@S and qS = Q@S (the two big projections), then u = v - exp(Glog)·kS ----
        // Both are (C×S_v)@(S_v×N_COLS) GEMMs. Expressed as A@Bᵀ with B = Sᵀ (STh_lds), contracting
        // S_v. kS folds into u (delta_lds); qS is stashed for Phase 8.
#ifdef FGDN_TILE_MMA
        if constexpr (tile_ok) {
            // kS = K@S (warp 0), qS = Q@S (warp 1). Expressed as A@Bᵀ with B = Sᵀ read transposed
            // straight from S_lds (row_stride=1, k_stride=N_COLS); contract i over S_v.
            if (warp < 2) {
                ggml_cuda_mma::tile<16, 16, float, FGDN_C_DL> Cacc;
                const float * Amat = (warp == 0) ? K_lds : Q_lds;   // warp0: kS, warp1: qS
                fgdn_mma16(Cacc, Amat, S_v, 1, S_lds, 1, N_COLS, S_v / 16);
                const bool is_kS = (warp == 0);
#pragma unroll
                for (int l = 0; l < Cacc.ne; ++l) {
                    const int t = Cacc.get_i(l);
                    const int c = Cacc.get_j(l);
                    if (is_kS) {
                        if (t < C) {
                            const float v_tc = v_base[(chunk_start + t) * sv2 + col_base + c];
                            delta_lds[t * N_COLS + c] = v_tc - expf(Glog_lds[t]) * Cacc.x[l];
                        }
                    } else {
                        qS_lds[t * N_COLS + c] = Cacc.x[l];
                    }
                }
            }
        } else
#endif // FGDN_TILE_MMA
        {
            for (int idx = tid_in_block; idx < C * N_COLS; idx += threads_per_block) {
                const int t = idx / N_COLS;
                const int c = idx % N_COLS;
                float kS = 0.0f, qS = 0.0f;
#pragma unroll
                for (int i = 0; i < S_v; i++) {
                    const float s = S_lds[i * N_COLS + c];
                    kS += K_lds[t * S_v + i] * s;
                    qS += Q_lds[t * S_v + i] * s;
                }
                const float v_tc = v_base[(chunk_start + t) * sv2 + col_base + c];
                delta_lds[t * N_COLS + c] = v_tc - expf(Glog_lds[t]) * kS;
                qS_lds[t * N_COLS + c] = qS;
            }
        }
        __syncthreads();

        // ---- Phase 7: forward solve (I + L) Δ = U ----
        // Each column c is an independent forward substitution; one thread owns a whole column so
        // no intra-solve sync is needed.
        for (int c = tid_in_block; c < N_COLS; c += threads_per_block) {
            for (int t = 0; t < C; t++) {
                float dt = delta_lds[t * N_COLS + c];
                for (int s = 0; s < t; s++) {
                    dt -= L_lds[t * CHUNK_C + s] * delta_lds[s * N_COLS + c];
                }
                delta_lds[t * N_COLS + c] = dt;
            }
        }
        __syncthreads();

        // ---- Phase 8: attn[t,c] = scale·(exp(Glog[t])·qS[t,c] + Σ_{s≤t} D β_s QK δ[s,c]) ----
        for (int idx = tid_in_block; idx < C * N_COLS; idx += threads_per_block) {
            const int t = idx / N_COLS;
            const int c = idx % N_COLS;
            float internal = 0.0f;
            for (int s = 0; s <= t; s++) {
                internal += D_lds[t * CHUNK_C + s] * beta_lds[s] * QK_lds[t * CHUNK_C + s] * delta_lds[s * N_COLS + c];
            }
            attn_data_base[(chunk_start + t) * S_v * H + col_base + c] =
                (expf(Glog_lds[t]) * qS_lds[t * N_COLS + c] + internal) * scale;
        }
        __syncthreads();

        // ---- keep_rs / MTP: emit per-token state snapshots for the last K global tokens ----
        // The recurrent cache only keeps the last K = n_rs_seq+1 per-token states (for speculative
        // rollback). For those tokens we materialise S_t = G_t·S_start + Σ_{s≤t} D[t,s]·β_s·k_s·δ_sᵀ
        // from the chunk-start state (S_lds is still S_old here, before Phase 9 updates it).
        if constexpr (keep_rs_t) {
            // slot 0 = most recent state, slot s = s tokens back (matches per-token path).
            for (int t = 0; t < C; ++t) {
                const int slot = (int) n_tokens - 1 - (chunk_start + t);
                if (slot < 0 || slot >= K) continue;
                const float G_t = expf(Glog_lds[t]);
                float * snap = state_out_base + (int64_t) slot * state_slot_stride;
                for (int idx = tid_in_block; idx < S_v * N_COLS; idx += threads_per_block) {
                    const int i = idx / N_COLS;
                    const int c = idx % N_COLS;
                    float acc = G_t * S_lds[i * N_COLS + c];
                    for (int s = 0; s <= t; ++s) {
                        acc += D_lds[t * CHUNK_C + s] * beta_lds[s] * K_lds[s * S_v + i] * delta_lds[s * N_COLS + c];
                    }
                    snap[(col_base + c) * S_v + i] = acc;
                }
            }
            __syncthreads();   // finish reading S_lds before Phase 9 overwrites it
        }

        // ---- Phase 9: S_new[i,c] = exp(Glog[C-1])·S_old[i,c] + (Kᵀ @ W)[i,c] ----
        // W[s,c] = D[C-1,s]·β_s·Δ[s,c]. update = Kᵀ(S_v×C) @ W(C×N_COLS) -> S_v×N_COLS, contract C.
        const float G_C = expf(Glog_lds[C - 1]);
        // Build W in LDS (zero-pad rows [C,CHUNK_C): D[C-1,s] is stale there, so set W=0 explicitly).
        for (int idx = tid_in_block; idx < CHUNK_C * N_COLS; idx += threads_per_block) {
            const int s = idx / N_COLS;
            const int c = idx % N_COLS;
            W_lds[s * N_COLS + c] = (s < C)
                ? (D_lds[(C - 1) * CHUNK_C + s] * beta_lds[s] * delta_lds[s * N_COLS + c])
                : 0.0f;
        }
        __syncthreads();
#ifdef FGDN_TILE_MMA
        if constexpr (tile_ok) {
            // update[i][c] = (Kᵀ@W)[i][c] = A@Bᵀ with A = Kᵀ[i][s] (from K_lds, row_stride=1,
            // k_stride=S_v) and B = Wᵀ[c][s] (from W_lds, row_stride=1, k_stride=N_COLS), contracting
            // the chunk dim C=CHUNK_C. Output over (S_v/16) row-tiles × 1 col-tile, read transposed
            // straight from the phase LDS (no staging).
            constexpr int MT = S_v / 16;   // row-tiles over the state
            for (int mi = warp; mi < MT; mi += num_warps) {
                ggml_cuda_mma::tile<16, 16, float, FGDN_C_DL> Cacc;
                fgdn_mma16(Cacc, K_lds + mi * 16, 1, S_v, W_lds, 1, N_COLS, CHUNK_C / 16);
#pragma unroll
                for (int l = 0; l < Cacc.ne; ++l) {
                    const int i = mi * 16 + Cacc.get_i(l);
                    const int c = Cacc.get_j(l);
                    S_lds[i * N_COLS + c] = G_C * S_lds[i * N_COLS + c] + Cacc.x[l];
                }
            }
        } else
#endif // FGDN_TILE_MMA
        {
            for (int idx = tid_in_block; idx < S_v * N_COLS; idx += threads_per_block) {
                const int i = idx / N_COLS;
                const int c = idx % N_COLS;
                float upd = 0.0f;
                for (int s = 0; s < C; s++) {
                    upd += K_lds[s * S_v + i] * W_lds[s * N_COLS + c];
                }
                S_lds[i * N_COLS + c] = G_C * S_lds[i * N_COLS + c] + upd;
            }
        }
        __syncthreads();
    }

    // ---- Final write-back of the single chunk-final state (non keep_rs only) ----
    // For keep_rs the last K per-token states were already written to the snapshot slots above.
    if constexpr (!keep_rs_t) {
        for (int idx = tid_in_block; idx < S_v * N_COLS; idx += threads_per_block) {
            const int i = idx / N_COLS;
            const int c = idx % N_COLS;
            state_out_base[(col_base + c) * S_v + i] = S_lds[i * N_COLS + c];
        }
    }
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Minimum chunk length (tokens) for the chunked path to win: the LDS setup and C×C matmuls
// amortise only over enough tokens. Measured crossover on MI250X (gfx90a): the chunked kernel is
// a clear, reproducible win from ≥512 tokens (1.08× at 512, scaling to 1.6× at 4096) for the
// 48-head geometry, and neutral/slower below. Tuned for non-KDA, S_v=128.
#ifndef GGML_CUDA_DELTANET_CHUNKED_MIN_TOKENS
#define GGML_CUDA_DELTANET_CHUNKED_MIN_TOKENS 512
#endif
// Minimum chunked grid blocks per SM/CU for the chunked path to win. The chunked grid launches
// H * n_seqs * (S_v/N_COLS) blocks; the column-grouped layout recomputes the shared Phases 1-5
// per block, so it only pays off once there is enough parallelism to hide that redundancy. On
// MI250X (nsm≈104/GCD) the kernel wins at ≈384 blocks (full 48-head geometry) and is neutral at
// ≈256 (32 heads); 3×nsm sits between, scaling with device size.
#ifndef GGML_CUDA_DELTANET_MIN_BLOCKS_PER_SM
#define GGML_CUDA_DELTANET_MIN_BLOCKS_PER_SM 3
#endif
#ifndef GGML_CUDA_DELTANET_CHUNK_C
// CHUNK_C controls the chunked-scan tile size. LDS budget per block (CHUNK_C tokens, S_v=128):
//   K_lds + Q_lds:  2 * CHUNK_C * S_v * 4 bytes
//   D + KK + QK + L: 4 * CHUNK_C * CHUNK_C * 4 bytes
//   misc:            ~256 bytes
// CHUNK_C=16 → ~17 KB; CHUNK_C=32 → ~37 KB. MI250X has 64 KB LDS/CU so both fit, but
// CHUNK_C=16 measured fastest on gfx90a (more blocks resident per CU; the C×C work and
// triangular solve grow with CHUNK_C while the per-token amortisation flattens out).
#define GGML_CUDA_DELTANET_CHUNK_C 16
#endif
#ifndef GGML_CUDA_DELTANET_NCOLS
// State columns owned per block (column grouping). Grid z-dim = S_v / N_COLS. Larger N_COLS =>
// wider MFMA tiles + less Phase 1-5 redundancy, but more LDS (S_lds = S_v*N_COLS floats) and
// fewer blocks (lower occupancy). On MI250X, 16 measured fastest: 2 blocks/CU outweighs the
// extra shared-phase redundancy. 32 fits (~43 KB LDS, 1 block/CU); 64 exceeds the 64 KB budget.
#define GGML_CUDA_DELTANET_NCOLS 16
#endif

// Kill-switch for the chunked path. Parse the env var as a boolean instead of a bare
// getenv() != nullptr check: the latter would treat GGML_CUDA_DISABLE_DELTANET_CHUNKED=0
// (or "false") as "disable", which is the opposite of what a user expects. Only explicit
// truthy values ("1", "true", "yes", "on") disable; unset / "0" / "false" / "off" keep it on.
static bool ggml_cuda_deltanet_chunked_disabled() {
    const char * e = getenv("GGML_CUDA_DISABLE_DELTANET_CHUNKED");
    if (e == nullptr) {
        return false;
    }
    // Exact, case-insensitive match against truthy tokens. A first-char check would be
    // surprising (e.g. "tomato" would disable, "10" would not); match the whole string.
    auto ieq = [](const char * a, const char * b) {
        for (; *a != '\0' && *b != '\0'; ++a, ++b) {
            if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) {
                return false;
            }
        }
        return *a == *b;
    };
    return ieq(e, "1") || ieq(e, "true") || ieq(e, "yes") || ieq(e, "on");
}

// Host-side mirror of the device FGDN_TILE_MMA guard: the chunked kernel's GEMMs go through the
// shared ggml_cuda_mma tile interface, so it runs wherever that interface has a f16-in/f32-acc
// 16×16×16 mma with the plain I_MAJOR input layout: AMD CDNA (MFMA), NVIDIA Turing+ (tensor cores),
// and RDNA4 (WMMA). Volta / RDNA3 use a mirrored input layout and fall back to the per-token kernel.
static bool fgdn_chunked_arch_ok(const int cc) {
    return amd_mfma_available(cc)
        || turing_mma_available(cc)
        || (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc));
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    // Chunked-scan path (shared ggml_cuda_mma tile interface -> NVIDIA tensor cores + AMD MFMA).
    // Batches CHUNK_C tokens so the recurrence becomes GEMMs (KKᵀ/QKᵀ, the kS/qS projections and the
    // Kᵀ@W state update) instead of a serial per-token rank-1 update. On MI250X (gfx90a) this is
    // ~1.3-2.4× faster than the per-token kernel at prefill and scales with sequence length, while
    // matching it numerically (covered by the test_gated_delta_net prefill cases in test-backend-ops).
    // Numerical stability comes from the log-space cumulative gate (see the Glog_lds comment block).
    //
    // Gated to the cases where it is both correct and a win: non-KDA, S_v=128, an arch whose mma.cuh
    // f16 path uses the plain I_MAJOR input layout (CDNA / Turing+ / RDNA4), chunks long enough to
    // amortise the LDS/C×C setup (n_tokens), and enough grid parallelism to hide the per-block
    // shared-phase recompute (blocks per SM). keep_rs (MTP) is supported: the kernel emits the last
    // K per-token snapshots. Every other configuration (Volta/RDNA3, S_v≠128, short sequences, few
    // heads, KDA) falls through to the per-token kernel below, so this path can never regress them.
    // Set GGML_CUDA_DISABLE_DELTANET_CHUNKED=1 to force the per-token kernel even when eligible.
    const int     cc             = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int     nsm            = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int64_t chunked_blocks = H * n_seqs * (S_v / GGML_CUDA_DELTANET_NCOLS);
    const bool eligible = !KDA
                       && S_v == 128
                       && fgdn_chunked_arch_ok(cc)
                       && n_tokens >= GGML_CUDA_DELTANET_CHUNKED_MIN_TOKENS
                       && chunked_blocks >= (int64_t) GGML_CUDA_DELTANET_MIN_BLOCKS_PER_SM * nsm;
    const bool use_chunked = eligible && !ggml_cuda_deltanet_chunked_disabled();
    if (use_chunked) {
        // Column-grouped layout: each block owns N_COLS columns of S; grid z-dim = S_v / N_COLS.
        constexpr int n_cols = GGML_CUDA_DELTANET_NCOLS;
        dim3 cgrid(H, n_seqs, (S_v + n_cols - 1) / n_cols);
        dim3 cblock(warp_size, num_warps, 1);
        gated_delta_net_chunked_cuda<128, GGML_CUDA_DELTANET_CHUNK_C, n_cols, keep_rs_t><<<cgrid, cblock, 0, stream>>>(
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
        return;
    }

    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
