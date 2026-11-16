#include "fattn-sparse-cdna.cuh"
#include "ggml-cuda/fattn-common.cuh"
#include "ggml-cuda/fattn-tile.cuh"

#include <algorithm>

// One block per (K/V segment, K/V head, sequence) handles all 12*ncols Q rows of the K/V head.
// The block compacts the visible mask columns of its segment into LDS, in rounds of ncr columns.
// A wave takes 16 compacted rows at a time and loads K and V directly in MFMA operand layout (16x16x16 f16):
// - S^T = K Q^T: lane l loads K row l%16, dims [64*(l/16), 64*(l/16) + 64). Q in LDS uses the same dim order.
// - O^T += V^T P^T: the S^T result of lane l (kv rows 4*(l/16) + r, Q row l%16) is the P^T operand.
//   Lane l loads V rows 4*(l/16) + i, dims [16*(l%16), 16*(l%16) + 16). Output tile t holds the dims 16*m + t.
// The mask values of a group are gathered from global memory together with K and V.
// The softmax uses exp2: log2(e) is folded into the Q scale and the mask, KQ max values are in log2 units.
// The waves of a block are merged in LDS. The result of a segment is stored normalized (O/l) as f16 together with (max, sum),
// the segments are merged by flash_attn_sparse_cdna_combine.

#define FATTN_SPARSE_CDNA_GQA      12
#define FATTN_SPARSE_CDNA_D        256
#define FATTN_SPARSE_CDNA_NWAVES   4
#define FATTN_SPARSE_CDNA_NTHREADS (FATTN_SPARSE_CDNA_NWAVES*64)
#define FATTN_SPARSE_CDNA_NSEG_MAX 64 // the combine handles 8 segments per thread group
#define FATTN_SPARSE_CDNA_MAX_SLACK 11.0f // log2 units, P <= 2^11 fits into f16
#define FATTN_SPARSE_CDNA_MAX_OFFSET 3.0f // FATTN_KQ_MAX_OFFSET in log2 units

template <int ncols>
struct fattn_sparse_cdna_config {
    static constexpr int nq       = FATTN_SPARSE_CDNA_GQA*ncols;     // Q rows
    static constexpr int nqt      = (nq + 15)/16;                    // Q tiles of 16 rows
    static constexpr int qsplit   = nqt > 3 ? 2 : 1;                 // waves that share the K/V rows, nqt/qsplit tiles each
    static constexpr int nqtw     = nqt/qsplit;
    static constexpr int nstreams = FATTN_SPARSE_CDNA_NWAVES/qsplit; // waves with different K/V rows
    static constexpr int cpt      = 8;                               // mask columns per thread and round
    static constexpr int ncr      = cpt*FATTN_SPARSE_CDNA_NTHREADS;  // mask columns per round
    static_assert(nqt % qsplit == 0, "bad qsplit");
};

typedef _Float16 fsc_half4  __attribute__((ext_vector_type(4)));
typedef _Float16 fsc_half8  __attribute__((ext_vector_type(8)));
typedef float    fsc_float4 __attribute__((ext_vector_type(4)));
typedef int      fsc_int4   __attribute__((ext_vector_type(4)));

// scale MFMA accumulators in place: with plain code the compiler keeps a VGPR copy of all accumulators for the rare rescale
// the hazard recognizer does not pad inline asm: the s_nop cover MFMA -> accvgpr_read and accvgpr_write -> accvgpr_read/MFMA
template <int n>
static __device__ __forceinline__ void fsc_scale_acc(fsc_float4 (&acc)[n], const float scale) {
    asm volatile("s_nop 7\n\ts_nop 7\n\ts_nop 4" ::: "memory");
#pragma unroll
    for (int i = 0; i < n; ++i) {
        float x0 = acc[i][0];
        float x1 = acc[i][1];
        float x2 = acc[i][2];
        float x3 = acc[i][3];
        float tmp;
        asm volatile(
            "s_nop 1\n\t"
            "v_accvgpr_read_b32 %4, %0\n\ts_nop 1\n\tv_mul_f32 %4, %5, %4\n\ts_nop 1\n\tv_accvgpr_write_b32 %0, %4\n\ts_nop 1\n\t"
            "v_accvgpr_read_b32 %4, %1\n\ts_nop 1\n\tv_mul_f32 %4, %5, %4\n\ts_nop 1\n\tv_accvgpr_write_b32 %1, %4\n\ts_nop 1\n\t"
            "v_accvgpr_read_b32 %4, %2\n\ts_nop 1\n\tv_mul_f32 %4, %5, %4\n\ts_nop 1\n\tv_accvgpr_write_b32 %2, %4\n\ts_nop 1\n\t"
            "v_accvgpr_read_b32 %4, %3\n\ts_nop 1\n\tv_mul_f32 %4, %5, %4\n\ts_nop 1\n\tv_accvgpr_write_b32 %3, %4\n\ts_nop 1\n\t"
            "s_nop 1"
            : "+a"(x0), "+a"(x1), "+a"(x2), "+a"(x3), "=&v"(tmp) : "v"(scale));
        acc[i] = fsc_float4{x0, x1, x2, x3};
    }
    asm volatile("s_nop 4" ::: "memory");
}

template <int ncols>
__launch_bounds__(FATTN_SPARSE_CDNA_NTHREADS, 1)
static __global__ void flash_attn_sparse_cdna(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const char * __restrict__ mask,
        half * __restrict__ dst_parts, float2 * __restrict__ dst_meta,
        const float scale, const int ne01, const int ne11, const int seg_len,
        const int nb01, const int nb02, const int64_t nb03,
        const int nb11, const int nb12, const int64_t nb13,
        const int nb21, const int nb22, const int64_t nb23,
        const int ne33, const int nb31, const int64_t nb33) {
#if defined(CDNA1)
    using cfg = fattn_sparse_cdna_config<ncols>;
    constexpr int gqa      = FATTN_SPARSE_CDNA_GQA;
    constexpr int D        = FATTN_SPARSE_CDNA_D;
    constexpr int nwaves   = FATTN_SPARSE_CDNA_NWAVES;
    constexpr int nthreads = FATTN_SPARSE_CDNA_NTHREADS;
    constexpr int nqt      = cfg::nqt;
    constexpr int qsplit   = cfg::qsplit;
    constexpr int nqtw     = cfg::nqtw;
    constexpr int nstreams = cfg::nstreams;
    constexpr int cpt      = cfg::cpt;
    constexpr int ncr      = cfg::ncr;

    const int seg  = blockIdx.x;
    const int kvh  = blockIdx.y;
    const int seq  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid % 64;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64); // uniform loops keep the accumulators in AGPRs
    const int qpart  = wave % qsplit;
    const int stream = wave / qsplit;

    // Q in B operand layout, then the compacted columns; at the end the same memory is used to merge the waves
    constexpr size_t nb_Q    = size_t(nqt)*16*64*sizeof(fsc_half4);
    constexpr size_t nb_idx  = size_t(ncr)*sizeof(int);
    constexpr int    nbuf    = nstreams/2*qsplit; // merge buffers of 64 floats per lane
    constexpr size_t nb_obuf = size_t(nbuf)*16*64*sizeof(float4);
    constexpr size_t nb_smem = nb_Q + nb_idx > nb_obuf ? nb_Q + nb_idx : nb_obuf;
    __shared__ __align__(16) char smem[nb_smem];
    __shared__ int    nsel_wave[nwaves];
    __shared__ float2 ml_lds[nstreams][nqt*16];

    fsc_half4 * Q_lds    = (fsc_half4 *) smem;
    int       * idx_sel  = (int  *) (smem + nb_Q);
    float4    * obuf     = (float4 *) smem;

    const char * K_base = K + nb13*seq + int64_t(nb12)*kvh;
    const char * V_base = V + nb23*seq + int64_t(nb22)*kvh;
    const half * maskh  = (const half *) (mask + nb33*(seq % ne33));

    const int seg0    = seg*seg_len;
    const int seg1    = min(seg0 + seg_len, ne11);
    const int nrounds = (seg1 - seg0 + ncr - 1)/ncr;

    // the loads are unconditional (clamped), invalid columns are set to -inf in the compaction
    __align__(16) half mreg[ncols][cpt];
    auto load_mask = [&](const int r) {
        const int c0 = min(seg0 + r*ncr + tid*cpt, seg1 - cpt);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            ggml_cuda_memcpy_1<cpt*sizeof(half)>(mreg[j], maskh + min(j, ne01 - 1)*(nb31/sizeof(half)) + c0);
        }
    };
    load_mask(0);

    // Q, scaled and converted to f16. Q row q = j*gqa + c, dim d is in step (d%64)/4 of lane group d/64.
    {
        const float scale_log2 = scale*float(M_LOG2E);
        const float * Qf = (const float *) (Q + nb03*seq + int64_t(nb02)*(kvh*gqa));
        constexpr int nload = nqt*16*64/nthreads;
        float4 Q_r[nload];
        // all loads first: a conditional load would be waited for before the next one is issued
#pragma unroll
        for (int k = 0; k < nload; ++k) {
            const int q = (k*nthreads + tid) / 64;
            const int j = min(q / gqa, ne01 - 1);
            Q_r[k] = *(const float4 *) (Qf + j*(nb01/sizeof(float)) + (q % gqa)*(nb02/sizeof(float)) + 4*(tid % 64));
        }
#pragma unroll
        for (int k = 0; k < nload; ++k) {
            const int i  = k*nthreads + tid;
            const int q  = i / 64;
            const int d4 = i % 64;
            const float4 v = q / gqa < ne01 ? Q_r[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const fsc_half4 vh = {(_Float16) (v.x*scale_log2), (_Float16) (v.y*scale_log2), (_Float16) (v.z*scale_log2), (_Float16) (v.w*scale_log2)};
            Q_lds[((q/16)*16 + d4 % 16)*64 + (d4/16)*16 + q % 16] = vh;
        }
    }

    float      KQ_max[nqtw];
    float      KQ_sum[nqtw];
    fsc_float4 VKQ[nqtw][16];
#pragma unroll
    for (int qt = 0; qt < nqtw; ++qt) {
        KQ_max[qt] = -FLT_MAX/2.0f;
        KQ_sum[qt] = 0.0f;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            VKQ[qt][t] = fsc_float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    int nsel = 0;

    // rows past nsel repeat the last row, their mask value is replaced by -inf in compute
    auto load_KV = [&](const int g, fsc_int4 * K_r, fsc_int4 * V_r, half * M_r) {
        const int k0 = 16*g;
        const fsc_int4 * K_p = (const fsc_int4 *) (K_base + int64_t(idx_sel[min(k0 + lane % 16, nsel - 1)])*nb11 + 128*(lane/16));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            K_r[i] = K_p[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = idx_sel[min(k0 + 4*(lane/16) + i, nsel - 1)];
            const fsc_int4 * V_p = (const fsc_int4 *) (V_base + int64_t(col)*nb21 + 32*(lane % 16));
            V_r[2*i + 0] = V_p[0];
            V_r[2*i + 1] = V_p[1];
#pragma unroll
            for (int qt = 0; qt < nqtw; ++qt) {
                const int j = ncols == 1 ? 0 : min(((qpart*nqtw + qt)*16 + lane % 16) / gqa, ne01 - 1);
                M_r[qt*4 + i] = maskh[j*(nb31/sizeof(half)) + col];
            }
        }
    };

    auto compute = [&](const int g, const fsc_int4 * K_r, const fsc_int4 * V_r, const half * M_r) {
        const int k0 = 16*g;

        // opaque offset: otherwise Q is hoisted out of the loop into 32 VGPRs per tile
        int Q_off = qpart*nqtw*16*64 + lane;
        asm volatile("" : "+v"(Q_off));

        fsc_float4 KQ[nqtw];
#pragma unroll
        for (int qt = 0; qt < nqtw; ++qt) {
            KQ[qt] = fsc_float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const fsc_half8 K8 = __builtin_bit_cast(fsc_half8, K_r[s/2]);
            const fsc_half4 A  = s % 2 == 0 ? K8.lo : K8.hi;
#pragma unroll
            for (int qt = 0; qt < nqtw; ++qt) {
                const fsc_half4 B = Q_lds[Q_off + (qt*16 + s)*64];
                KQ[qt] = __builtin_amdgcn_mfma_f32_16x16x16f16(A, B, KQ[qt], 0, 0, 0);
            }
        }

        fsc_half4 P[nqtw];
#pragma unroll
        for (int qt = 0; qt < nqtw; ++qt) {
            float KQ_max_lane = -INFINITY;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float m = k0 + 4*(lane/16) + r < nsel ? __half2float(M_r[qt*4 + r]) : -INFINITY;
                KQ[qt][r] = fmaf(m, float(M_LOG2E), KQ[qt][r]);
                KQ_max_lane = fmaxf(KQ_max_lane, KQ[qt][r]);
            }

            // lazy rescale: the max of a Q row only moves if a value exceeds it by more than FATTN_SPARSE_CDNA_MAX_SLACK
            if (__any(KQ_max_lane > KQ_max[qt] + FATTN_SPARSE_CDNA_MAX_SLACK)) {
                float KQ_max_row = fmaxf(KQ_max_lane, __shfl_xor(KQ_max_lane, 16, 64));
                KQ_max_row = fmaxf(KQ_max_row, __shfl_xor(KQ_max_row, 32, 64));
                const bool  raise        = KQ_max_row > KQ_max[qt] + FATTN_SPARSE_CDNA_MAX_SLACK;
                const float KQ_max_new   = raise ? KQ_max_row + FATTN_SPARSE_CDNA_MAX_OFFSET : KQ_max[qt];
                const float KQ_max_scale = exp2f(KQ_max[qt] - KQ_max_new);
                KQ_sum[qt] *= KQ_max_scale;
                // VKQ is still 0 if the old max is the initial value
                if (__any(raise && KQ_max[qt] != -FLT_MAX/2.0f)) {
                    fsc_scale_acc(VKQ[qt], KQ_max_scale);
                }
                KQ_max[qt] = KQ_max_new;
            }

#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float p = exp2f(KQ[qt][r] - KQ_max[qt]);
                KQ_sum[qt] += p;
                P[qt][r] = (_Float16) p;
            }
        }

        fsc_half8 V8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            V8[i] = __builtin_bit_cast(fsc_half8, V_r[i]);
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const fsc_half4 A = {V8[0 + t/8][t % 8], V8[2 + t/8][t % 8], V8[4 + t/8][t % 8], V8[6 + t/8][t % 8]};
#pragma unroll
            for (int qt = 0; qt < nqtw; ++qt) {
                VKQ[qt][t] = __builtin_amdgcn_mfma_f32_16x16x16f16(A, P[qt], VKQ[qt][t], 0, 0, 0);
            }
        }
    };

    for (int r = 0; r < nrounds; ++r) {
        // compact the columns that any query can see
        const int c0 = seg0 + r*ncr + tid*cpt;
        uint32_t sel = 0;
#pragma unroll
        for (int l = 0; l < cpt; ++l) {
            bool visible = false;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                visible = visible || ((ncols == 1 || j < ne01) && __half2float(mreg[j][l]) > -INFINITY);
            }
            sel |= uint32_t(visible) << l;
        }
        sel = c0 < seg1 ? sel : 0;
        // the order of the compacted columns does not matter: column l of all lanes, then column l + 1
        int pos_l[cpt];
        int nsel_w = 0;
#pragma unroll
        for (int l = 0; l < cpt; ++l) {
            const uint64_t b = __ballot((sel >> l) & 1);
            pos_l[l] = nsel_w + __builtin_amdgcn_mbcnt_hi(uint32_t(b >> 32), __builtin_amdgcn_mbcnt_lo(uint32_t(b), 0));
            nsel_w  += __popcll(b);
        }
        if (lane == 0) {
            nsel_wave[wave] = nsel_w;
        }
        __syncthreads();

        int pos0 = 0;
        nsel = 0;
#pragma unroll
        for (int w = 0; w < nwaves; ++w) {
            pos0 += w < wave ? nsel_wave[w] : 0;
            nsel += nsel_wave[w];
        }
        nsel = __builtin_amdgcn_readfirstlane(nsel);
#pragma unroll
        for (int l = 0; l < cpt; ++l) {
            if (sel & (1u << l)) {
                idx_sel[pos0 + pos_l[l]] = c0 + l;
            }
        }
        __syncthreads();

        if (r + 1 < nrounds) {
            load_mask(r + 1);
        }

        // the K/V rows of the next group are loaded while the current group is computed
        // the loads are unconditional, a branch around them makes the compiler wait for all loads in flight
        fsc_int4 K_a[8], V_a[8], K_b[8], V_b[8];
        half     M_a[4*nqtw], M_b[4*nqtw];
        int g = stream;
        if (16*g < nsel) {
            load_KV(g, K_a, V_a, M_a);
        }
        while (16*g < nsel) {
            load_KV(g + nstreams, K_b, V_b, M_b);
            compute(g, K_a, V_a, M_a);
            g += nstreams;
            if (16*g >= nsel) {
                break;
            }
            load_KV(g + nstreams, K_a, V_a, M_a);
            compute(g, K_b, V_b, M_b);
            g += nstreams;
        }
        // wait for the unused prefetch: otherwise the in-flight loads of the next round's loop are waited for in every iteration
        __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
        __syncthreads();
    }

    // merge the waves with different K/V rows, write the result of the segment
#pragma unroll
    for (int qt = 0; qt < nqtw; ++qt) {
        KQ_sum[qt] += __shfl_xor(KQ_sum[qt], 16, 64);
        KQ_sum[qt] += __shfl_xor(KQ_sum[qt], 32, 64);
        if (lane < 16) {
            ml_lds[stream][(qpart*nqtw + qt)*16 + lane] = make_float2(KQ_max[qt], KQ_sum[qt]);
        }
    }
    __syncthreads();

    const int     nrows   = gqa*ne01;
    const int64_t row_dst = (int64_t(seq*gridDim.y + kvh)*gridDim.x + seg)*nrows;

    for (int q = tid; q < nqt*16; q += nthreads) {
        float KQ_max_blk = -FLT_MAX/2.0f;
#pragma unroll
        for (int st = 0; st < nstreams; ++st) {
            KQ_max_blk = fmaxf(KQ_max_blk, ml_lds[st][q].x);
        }
        float KQ_sum_blk = 0.0f;
#pragma unroll
        for (int st = 0; st < nstreams; ++st) {
            KQ_sum_blk += ml_lds[st][q].y*exp2f(ml_lds[st][q].x - KQ_max_blk);
        }
        if (q < nrows) {
            dst_meta[row_dst + q] = make_float2(KQ_max_blk, KQ_sum_blk);
        }
    }

    // tree over the streams: the upper half writes its results to LDS, the lower half adds them
    // LDS float atomics and read-modify-write in LDS were slow. Lane l holds Q row l%16, dims 16*(4*(l/16) + r) + t.
#pragma unroll
    for (int qt = 0; qt < nqtw; ++qt) {
        const int q = (qpart*nqtw + qt)*16 + lane % 16;
        float KQ_max_blk = -FLT_MAX/2.0f;
#pragma unroll
        for (int st = 0; st < nstreams; ++st) {
            KQ_max_blk = fmaxf(KQ_max_blk, ml_lds[st][q].x);
        }
        const float KQ_max_scale = exp2f(KQ_max[qt] - KQ_max_blk);

        float4 acc[16]; // chunk c = 4*r + t/4
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            acc[c].x = VKQ[qt][4*(c % 4) + 0][c / 4]*KQ_max_scale;
            acc[c].y = VKQ[qt][4*(c % 4) + 1][c / 4]*KQ_max_scale;
            acc[c].z = VKQ[qt][4*(c % 4) + 2][c / 4]*KQ_max_scale;
            acc[c].w = VKQ[qt][4*(c % 4) + 3][c / 4]*KQ_max_scale;
        }

#pragma unroll
        for (int nh = nstreams/2; nh >= 1; nh /= 2) {
            if (stream >= nh && stream < 2*nh) {
                float4 * buf = obuf + ((stream - nh)*qsplit + qpart)*16*64 + lane;
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    buf[c*64] = acc[c];
                }
            }
            __syncthreads();
            if (stream < nh) {
                const float4 * buf = obuf + (stream*qsplit + qpart)*16*64 + lane;
                float4 tmp[16];
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    tmp[c] = buf[c*64];
                }
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    acc[c].x += tmp[c].x;
                    acc[c].y += tmp[c].y;
                    acc[c].z += tmp[c].z;
                    acc[c].w += tmp[c].w;
                }
            }
            __syncthreads();
        }

        // normalize, convert to f16 and store through LDS so that the global stores are contiguous
        constexpr int stage_s = D + 8; // row stride in halfs
        half * stage = (half *) smem;
        if (stream == 0) {
            float KQ_sum_blk = 0.0f;
#pragma unroll
            for (int st = 0; st < nstreams; ++st) {
                KQ_sum_blk += ml_lds[st][q].y*exp2f(ml_lds[st][q].x - KQ_max_blk);
            }
            const float inv = KQ_sum_blk > 0.0f ? 1.0f/KQ_sum_blk : 0.0f;
            half * stage_q = stage + (qpart*16 + lane % 16)*stage_s + 64*(lane/16);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const half2 lo = make_half2(acc[c].x*inv, acc[c].y*inv);
                const half2 hi = make_half2(acc[c].z*inv, acc[c].w*inv);
                half2 * p = (half2 *) (stage_q + 16*(c / 4) + 4*(c % 4));
                p[0] = lo;
                p[1] = hi;
            }
        }
        __syncthreads();
#pragma unroll
        for (int i0 = 0; i0 < qsplit*16*(D/8); i0 += nthreads) {
            const int i   = i0 + tid;
            const int qp  = i / (16*(D/8));
            const int q16 = (i / (D/8)) % 16;
            const int d8  = i % (D/8);
            const int qd  = (qp*nqtw + qt)*16 + q16;
            if (qd < nrows) {
                *(int4 *) (dst_parts + (row_dst + qd)*D + 8*d8) = *(const int4 *) (stage + (qp*16 + q16)*stage_s + 8*d8);
            }
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, dst_parts, dst_meta, scale, ne01, ne11, seg_len,
        nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne33, nb31, nb33);
    NO_DEVICE_CODE;
#endif // defined(CDNA1)
}

// one block per output row: 32 lanes read 16 bytes of a segment row, the 8 groups of 32 lanes take every 8th segment
// the partial results and (max, sum) are loaded at once, the max is only needed for the weights
__launch_bounds__(256, 1)
static __global__ void flash_attn_sparse_cdna_combine(
        const half * __restrict__ parts, const float2 * __restrict__ meta, const float * __restrict__ sinks,
        float * __restrict__ dst, const int nseg, const int ne01, const int ne02) {
    constexpr int gqa = FATTN_SPARSE_CDNA_GQA;
    constexpr int D   = FATTN_SPARSE_CDNA_D;
    constexpr int nit = FATTN_SPARSE_CDNA_NSEG_MAX/8;
    static_assert(D == 32*8, "bad D");

    const int j   = blockIdx.x / ne02;
    const int h   = blockIdx.x % ne02;
    const int seq = blockIdx.y;
    const int tid = threadIdx.x;
    const int c   = tid % 32;
    const int sg  = tid / 32;

    const int     nrows = gqa*ne01;
    const int64_t row0  = int64_t(seq*(ne02/gqa) + h/gqa)*nseg*nrows + j*gqa + h % gqa;

    int4   v[nit];
    float2 ml[nit];
#pragma unroll
    for (int it = 0; it < nit; ++it) {
        const int     s   = min(sg + 8*it, nseg - 1);
        const int64_t row = row0 + int64_t(s)*nrows;
        v[it]  = *(const int4 *) (parts + row*D + 8*c);
        ml[it] = meta[row];
    }

    // the KQ max values are in log2 units
    const float sink = sinks ? sinks[h]*float(M_LOG2E) : -FLT_MAX/2.0f;
    float KQ_max = sink;
#pragma unroll
    for (int it = 0; it < nit; ++it) {
        KQ_max = sg + 8*it < nseg ? fmaxf(KQ_max, ml[it].x) : KQ_max;
    }
#pragma unroll
    for (int offset = 32; offset < 64; offset <<= 1) {
        KQ_max = fmaxf(KQ_max, __shfl_xor(KQ_max, offset, 64));
    }
    __shared__ float max_s[4];
    __shared__ float den_s[8];
    __shared__ float num_s[8][D + 4];
    if (tid % 64 == 0) {
        max_s[tid/64] = KQ_max;
    }
    __syncthreads();
    KQ_max = fmaxf(fmaxf(max_s[0], max_s[1]), fmaxf(max_s[2], max_s[3]));

    float num[8] = {0.0f};
    float den    = 0.0f;
#pragma unroll
    for (int it = 0; it < nit; ++it) {
        const float w = sg + 8*it < nseg ? ml[it].y*exp2f(ml[it].x - KQ_max) : 0.0f;
        den += w;
        const half2 * vh = (const half2 *) &v[it];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(vh[k]);
            num[2*k + 0] += w*f.x;
            num[2*k + 1] += w*f.y;
        }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        num_s[sg][8*c + k] = num[k];
    }
    if (c == 0) {
        den_s[sg] = den;
    }
    __syncthreads();

    float sum = 0.0f;
    float den_tot = sinks ? exp2f(sink - KQ_max) : 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        sum     += num_s[k][tid];
        den_tot += den_s[k];
    }
    dst[(int64_t(seq*ne01 + j)*ne02 + h)*D + tid] = sum/den_tot;
}

bool ggml_hip_flash_attn_ext_sparse_cdna_supported(const int cc, const ggml_tensor * dst) {
    if (!GGML_CUDA_CC_IS_CDNA1(cc) || !ggml_cuda_flash_attn_ext_tile_shall_use_sparse(cc, dst)) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];

    // Q rows are read with 16 byte loads, K/V rows with 16 byte loads at offsets of 32 bytes
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || (sinks && sinks->type != GGML_TYPE_F32)) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K, V}) {
        if ((uintptr_t) t->data % 16 != 0 || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0) {
            return false;
        }
    }
    return Q->ne[2] == FATTN_SPARSE_CDNA_GQA*K->ne[2] && K->ne[3] == Q->ne[3] && V->ne[3] == Q->ne[3] &&
        ggml_nbytes(K) <= INT_MAX && ggml_nbytes(V) <= INT_MAX;
}

template <int ncols>
static void ggml_hip_flash_attn_ext_sparse_cdna_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    cudaStream_t stream = ctx.stream();
    const int nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int ne01 = Q->ne[1];
    const int ne02 = Q->ne[2];
    const int nkvh = K->ne[2];
    const int nseq = Q->ne[3];
    const int ne11 = K->ne[1];

    // about one block per CU
    int nseg = std::clamp<int>(nsm/(nkvh*nseq), 1, FATTN_SPARSE_CDNA_NSEG_MAX);
    const int seg_len = GGML_PAD((ne11 + nseg - 1)/nseg, 8);
    nseg = (ne11 + seg_len - 1)/seg_len;

    const int nrows = FATTN_SPARSE_CDNA_GQA*ne01;
    ggml_cuda_pool_alloc<half>   parts(ctx.pool(), size_t(nseq)*nkvh*nseg*nrows*FATTN_SPARSE_CDNA_D);
    ggml_cuda_pool_alloc<float2> meta (ctx.pool(), size_t(nseq)*nkvh*nseg*nrows);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    flash_attn_sparse_cdna<ncols><<<dim3(nseg, nkvh, nseq), FATTN_SPARSE_CDNA_NTHREADS, 0, stream>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
        parts.ptr, meta.ptr, scale, ne01, ne11, seg_len,
        Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
        mask->ne[3], mask->nb[1], mask->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    flash_attn_sparse_cdna_combine<<<dim3(ne01*ne02, nseq), FATTN_SPARSE_CDNA_D, 0, stream>>>(
        parts.ptr, meta.ptr, sinks ? (const float *) sinks->data : nullptr, (float *) dst->data, nseg, ne01, ne02);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_hip_flash_attn_ext_sparse_cdna(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int ne01 = dst->src[0]->ne[1];
    if (ne01 == 1) {
        ggml_hip_flash_attn_ext_sparse_cdna_case<1>(ctx, dst);
    } else if (ne01 == 2) {
        ggml_hip_flash_attn_ext_sparse_cdna_case<2>(ctx, dst);
    } else if (ne01 <= 4) {
        ggml_hip_flash_attn_ext_sparse_cdna_case<4>(ctx, dst);
    } else {
        GGML_ASSERT(ne01 <= 8);
        ggml_hip_flash_attn_ext_sparse_cdna_case<8>(ctx, dst);
    }
}
