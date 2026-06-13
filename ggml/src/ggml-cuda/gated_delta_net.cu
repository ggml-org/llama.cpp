#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// =====================================================================================
// Chunked prefix-scan kernel for gated DeltaNet (non-KDA, S_v=128) on AMD MFMA hardware.
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
// Why chunked beats per-token sequential on a long prefill:
//   - Each chunk loads K/Q once into LDS and reuses it for KK^T, QK^T, and the per-col
//     dot products (k_t·S_{t0}), (q_t·S_{t0}). The per-token kernel re-reads K,Q from
//     global on every step; the chunked kernel pays that bandwidth once per C tokens.
//   - The C×C matmuls KK^T and QK^T expose dense, MFMA-friendly shapes.
//   - The forward solve has a sequential C-step inner loop, but it is parallel across
//     the S_v=128 output columns (one per warp), which exactly matches the existing
//     warp-per-column layout of the original kernel.
//
// For Qwen3.6 prefill (n_tokens >> CHUNK_C) this kernel replaces the C-iteration inner
// loop of the per-token kernel with one chunk iteration that does the equivalent work
// in batched form. For short n_tokens (e.g. MTP-verify) the per-token kernel still wins
// because the C×C matrix overhead dominates; we dispatch on n_tokens at runtime.
//
// Limitations of this first version:
//   - non-KDA only (scalar gate per token). Qwen3.6 uses non-KDA — KDA models keep the
//     original per-token kernel.
//   - no keep_intermediates support (assert at runtime).
//   - S_v specialised to 128 only (Qwen3.6). Other S_v fall back to per-token.
// =====================================================================================
// -------------------------------------------------------------------------------------
// Column-grouped, state-in-LDS layout.
//
// The block owns N_COLS contiguous columns of the S_v×S_v state S and keeps them resident
// in LDS (S_lds[i*N_COLS + c] = S[row=i][col=col_base+c]). The grid z-dim is S_v/N_COLS so
// the H heads' 128 columns are split across (S_v/N_COLS) blocks per head instead of one warp
// per column. This turns the per-column Phases 6/8/9 into block-wide C×N_COLS and S_v×N_COLS
// tiles computed cooperatively by all threads, which is the shape the MFMA GEMMs need:
//   - kS = K@S, qS = Q@S projections (Phase 6)         -> MFMA, contract S_v
//   - the Kᵀ@W state update S_new = G·S + Kᵀ@W (Phase 9) -> MFMA, contract C
// The GEMMs use the f16 MFMA (mfma_f32_16x16x16f16, K=16/issue, 4x the f32_16x16x4 path) with
// f32 accumulation; inputs are L2-normalised / well-scaled so f16 stays within the op tolerance.
// while the shared Phases 1-5 (KKᵀ/QKᵀ, decay, triangular factors) are computed once per block
// instead of once per column, and the triangular solve (Phase 7) runs column-parallel.
// -------------------------------------------------------------------------------------
template <int S_v, int CHUNK_C, int N_COLS>
__global__ void __launch_bounds__(64 * 4, 1)
gated_delta_net_chunked_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ curr_state,
        float       * __restrict__ dst,
        int64_t H,
        int64_t n_tokens,
        int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic,
        const uint3 rq3_magic,
        float scale) {
    constexpr int num_warps = 4;
    static_assert(S_v % N_COLS == 0, "S_v must be a multiple of N_COLS");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp     = threadIdx.y;
    const int      col_base = blockIdx.z * N_COLS;   // first state column this block owns

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float * attn_data_base         = dst + (sequence * n_tokens * H + h_idx) * S_v;
    const float * curr_state_base  = curr_state + (sequence * H + h_idx) * S_v * S_v;
    float * state_out_base         = dst + attn_score_elems + (sequence * H + h_idx) * S_v * S_v;

    // LDS layout. All C×C arrays use CHUNK_C stride (so partial-chunk C<CHUNK_C indexes match
    // the fixed-size MFMA store in Phase 4). S_lds holds this block's N_COLS columns of S.
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

    const int tid_in_block = warp * 64 + lane;
    const int threads_per_block = num_warps * 64;

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
        // Zero-pad partial-chunk rows [C, CHUNK_C) so the fixed-size 16×16 MFMA tile in Phase 4
        // reads defined LDS. Outputs for those rows are never consumed. No-op when C==CHUNK_C.
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

        // ---- Phase 4: KK[t,s] = k_t · k_s,  QK[t,s] = q_t · k_s (MFMA on CDNA) ----
#ifdef AMD_MFMA_AVAILABLE
        if constexpr (CHUNK_C == 16 && (S_v % 16) == 0) {
            // Each C×C tile is one 16×16 MFMA output, contracting S_v in K=16 f16 sub-steps.
            // Layout (ggml_cuda_mma tile<16,16,float>): inputs A,B are I_MAJOR (a=A[lane%16][lane/16],
            // b=B[lane/16][lane%16]); output C is J_MAJOR so acc[l] lands at (t=4*(lane/16)+l, s=lane%16).
            // For both KK and QK the A/B fragment loads resolve to (row=lane%16, col=ki*4+lane/16).
            using floatx4_mfma_t = __attribute__((ext_vector_type(4))) float;
            using halfx4_mfma_t  = __attribute__((ext_vector_type(4))) _Float16;
            if (warp < 2) {
                const float * Amat = (warp == 0) ? K_lds : Q_lds;
                const int row = lane & 15;
                const int kg  = lane >> 4;
                floatx4_mfma_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
                // f16 MFMA contracts K=16/call (4x the f32_16x16x4 path); lane group kg owns the
                // contiguous K-block [16*kb + 4*kg, +4). Inputs are f16, accumulate stays f32.
#pragma unroll
                for (int kb = 0; kb < S_v / 16; kb++) {
                    halfx4_mfma_t a, b;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int icol = kb * 16 + 4 * kg + j;
                        a[j] = (_Float16) Amat [row * S_v + icol];
                        b[j] = (_Float16) K_lds[row * S_v + icol];
                    }
                    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
                }
                float * out = (warp == 0) ? KK_lds : QK_lds;
                const int s_out  = lane & 15;
                const int t_base = 4 * (lane >> 4);
#pragma unroll
                for (int l = 0; l < 4; l++) {
                    out[(t_base + l) * CHUNK_C + s_out] = acc[l];
                }
            }
        } else
#endif // AMD_MFMA_AVAILABLE
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
        // Each projection is a (C×S_v)@(S_v×N_COLS) GEMM -> C×N_COLS output, tiled into 16×16 MFMA
        // tiles (1 × N_COLS/16 tiles), contracting S_v=128 in K=4 steps. kS tiles fold straight into
        // u (delta_lds); qS tiles are stashed in qS_lds for Phase 8. Layout matches Phase 4: A,B
        // I_MAJOR (a=A[lane%16][lane/16], b=B[lane/16][lane%16]); acc[l] J_MAJOR at (t=4*(lane/16)+l,
        // n=lane%16).
#ifdef AMD_MFMA_AVAILABLE
        if constexpr (CHUNK_C == 16 && (S_v % 16) == 0 && (N_COLS % 16) == 0) {
            constexpr int NT = N_COLS / 16;            // output N-tiles per matrix
            using floatx4_mfma_t = __attribute__((ext_vector_type(4))) float;
            using halfx4_mfma_t  = __attribute__((ext_vector_type(4))) _Float16;
            const int row = lane & 15;
            const int kg  = lane >> 4;
            // tiles [0,NT) = kS, [NT,2NT) = qS; strided across warps so any N_COLS is covered.
            for (int tile = warp; tile < 2 * NT; tile += num_warps) {
                const bool is_kS  = (tile < NT);
                const int  n_tile = is_kS ? tile : (tile - NT);
                const float * Amat = is_kS ? K_lds : Q_lds;
                floatx4_mfma_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
                // f16 MFMA: K=16/call contracting S_v; f32 accumulate.
#pragma unroll
                for (int kb = 0; kb < S_v / 16; kb++) {
                    halfx4_mfma_t a, b;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int i = kb * 16 + 4 * kg + j;
                        a[j] = (_Float16) Amat [row * S_v + i];
                        b[j] = (_Float16) S_lds[i * N_COLS + n_tile * 16 + row];
                    }
                    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
                }
                const int c = n_tile * 16 + row;       // n = lane%16
#pragma unroll
                for (int l = 0; l < 4; l++) {
                    const int t = 4 * kg + l;
                    if (is_kS) {
                        if (t < C) {
                            const float v_tc = v_base[(chunk_start + t) * sv2 + col_base + c];
                            delta_lds[t * N_COLS + c] = v_tc - expf(Glog_lds[t]) * acc[l];
                        }
                    } else {
                        qS_lds[t * N_COLS + c] = acc[l];
                    }
                }
            }
        } else
#endif // AMD_MFMA_AVAILABLE
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
        // no intra-solve sync is needed (replaces the old lane-0-only serial solve).
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
        // qS was computed by MFMA in Phase 6 (qS_lds). The remaining `internal` is a small C-step
        // reduction over the chunk; small enough to keep scalar.
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
#ifdef AMD_MFMA_AVAILABLE
        if constexpr (CHUNK_C == 16 && (S_v % 16) == 0 && (N_COLS % 16) == 0) {
            // 16×16 output tiles over (S_v/16)×(N_COLS/16); contract C=16 in 4 steps of K=4.
            // A = Kᵀ (a = A[i][s] = K[s][i], row i=lane%16 within tile); B = W (b=W[s][c]); acc[l]
            // J_MAJOR at (i = mi*16 + 4*(lane/16)+l, c = ni*16 + lane%16).
            constexpr int MT = S_v / 16;
            constexpr int NT = N_COLS / 16;
            using floatx4_mfma_t = __attribute__((ext_vector_type(4))) float;
            using halfx4_mfma_t  = __attribute__((ext_vector_type(4))) _Float16;
            const int row = lane & 15;
            const int kg  = lane >> 4;
            for (int tile = warp; tile < MT * NT; tile += num_warps) {
                const int mi = tile / NT;
                const int ni = tile % NT;
                floatx4_mfma_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
                // contract C=CHUNK_C=16 in a single f16 MFMA call (was 4x f32_16x16x4).
#pragma unroll
                for (int kb = 0; kb < CHUNK_C / 16; kb++) {
                    halfx4_mfma_t a, b;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int s = kb * 16 + 4 * kg + j;
                        a[j] = (_Float16) K_lds[s * S_v + mi * 16 + row];   // K[s][i], i=mi*16+row
                        b[j] = (_Float16) W_lds[s * N_COLS + ni * 16 + row]; // W[s][c], c=ni*16+row
                    }
                    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
                }
                const int c = ni * 16 + row;
#pragma unroll
                for (int l = 0; l < 4; l++) {
                    const int i = mi * 16 + 4 * kg + l;
                    S_lds[i * N_COLS + c] = G_C * S_lds[i * N_COLS + c] + acc[l];
                }
            }
        } else
#endif // AMD_MFMA_AVAILABLE
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

    // ---- Final write-back of S to global: S[i][col_base+c] ----
    for (int idx = tid_in_block; idx < S_v * N_COLS; idx += threads_per_block) {
        const int i = idx / N_COLS;
        const int c = idx % N_COLS;
        state_out_base[(col_base + c) * S_v + i] = S_lds[i * N_COLS + c];
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
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    // input state layout (D, K, n_seqs) — seq stride is K * D = K * H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * K * H * S_v * S_v + h_idx * S_v * S_v;
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
            // slot mapping: target_slot = t - shift. When n_tokens < K only the last n_tokens slots
            // are written; earlier slots are left untouched (caller-owned).
            const int shift = (int) n_tokens - K;

            const int64_t state_size_per_token = S_v * S_v * H * n_seqs; // per-slot stride in output
            const int target_slot = t - shift;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = (dst + attn_score_elems) + target_slot * state_size_per_token + state_out_offset;
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
    return (e[0] == '1' && e[1] == '\0')                                   // "1"
        ||  e[0] == 't' || e[0] == 'T'                                     // true
        ||  e[0] == 'y' || e[0] == 'Y'                                     // yes
        || ((e[0] == 'o' || e[0] == 'O') && (e[1] == 'n' || e[1] == 'N')); // on (not off)
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    // Chunked-scan path (CDNA / AMD MFMA only). Batches CHUNK_C tokens so the recurrence becomes
    // MFMA GEMMs (kS/qS projections and the Kᵀ@W state update) instead of a serial per-token rank-1
    // update. On MI250X (gfx90a) this is ~1.3× faster than the per-token kernel at prefill and scales
    // with sequence length, while matching it numerically (max |chunked − per_token| ≤ ~1.2e-4 at
    // fp32 round-off; covered by the test_gated_delta_net prefill cases in test-backend-ops).
    // Numerical stability comes from the log-space cumulative gate (see the Glog_lds comment block).
    //
    // Gated strictly to the cases where it is both correct and a win: non-KDA, S_v=128, CDNA MFMA
    // hardware, 64-wide warps, chunks long enough to amortise the LDS/C×C setup (n_tokens), and
    // enough grid parallelism to hide the per-block shared-phase recompute (blocks per SM). Every
    // other configuration (NVIDIA, S_v≠128, short sequences, few heads, KDA, keep_rs) falls through
    // to the per-token kernel below, so this path can never regress them. Set
    // GGML_CUDA_DISABLE_DELTANET_CHUNKED=1 to force the per-token kernel even when eligible.
    const int     nsm            = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int64_t chunked_blocks = H * n_seqs * (S_v / GGML_CUDA_DELTANET_NCOLS);
    const bool eligible = !KDA && !keep_rs_t
                       && S_v == 128
                       && amd_mfma_available(cc)
                       && warp_size == 64
                       && n_tokens >= GGML_CUDA_DELTANET_CHUNKED_MIN_TOKENS
                       && chunked_blocks >= (int64_t) GGML_CUDA_DELTANET_MIN_BLOCKS_PER_SM * nsm;
    const bool use_chunked = eligible && !ggml_cuda_deltanet_chunked_disabled();
    if (use_chunked) {
        // Column-grouped layout: each block owns N_COLS columns of S; grid z-dim = S_v / N_COLS.
        constexpr int n_cols = GGML_CUDA_DELTANET_NCOLS;
        dim3 cgrid(H, n_seqs, (S_v + n_cols - 1) / n_cols);
        dim3 cblock(64, num_warps, 1);
        gated_delta_net_chunked_cuda<128, GGML_CUDA_DELTANET_CHUNK_C, n_cols><<<cgrid, cblock, 0, stream>>>(
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
        return;
    }

    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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

    // state is 3D (S_v*S_v*H, K, n_seqs); K is the snapshot slot count.
    const int K = (int) src_state->ne[1];
    const bool keep_rs = K > 1;

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        }
    }
}
