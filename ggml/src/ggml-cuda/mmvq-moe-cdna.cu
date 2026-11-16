#include "mmvq-moe-cdna.cuh"
#include "unary.cuh"

// A CUDA block computes a tile of rows of one expert for one (token, slot) pair.
// With several tokens (multi) a block also computes the other pairs with the same expert, their blocks exit. So the weights are read once:
// - Q4_K, Q5_K: up to npmax pairs in one pass over the weights, with a sum per pair.
// - Q5_1, Q8_0 (loop_pairs): the weights stay in registers and the pairs are computed one after another, several sums take too many registers.
// Rows are split into units of 32 values. A row has nlanes lanes, lane c of a row takes units c + nlanes*i, i < nu, with 16 byte loads.
// The q8_1 values of the pair are copied to shared memory in unit order: values [0, 16) and [16, 32) of the unit and its scales.
// For the K-quants the scales (and Q5_K qh) of the super-blocks of the tile are also copied to shared memory, the lanes only load qs.

static constexpr int MMVQ_MOE_CDNA_NWARPS = 4;
static constexpr int MMVQ_MOE_CDNA_MAX_NU = 8;

// limits for the search of pairs with the same expert, 2 ballots of 4 tokens x 16 slots per wave
static constexpr int MMVQ_MOE_CDNA_DEDUP_MAX_TOKENS = 8;
static constexpr int MMVQ_MOE_CDNA_DEDUP_MAX_SLOTS  = 16;

// min. waves per SIMD for the register allocation, with 1 the compiler hoists all shared memory loads and only 1-2 waves fit
static constexpr int MMVQ_MOE_CDNA_MIN_WAVES = 4;

static constexpr __host__ __device__ bool mmvq_moe_cdna_type_supported(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0;
}

static constexpr __host__ __device__ int mmvq_moe_cdna_nlanes(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 16 : 4;
}

// alignment of the src0 loads
static constexpr __host__ __device__ int mmvq_moe_cdna_alignment(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 16 : (type == GGML_TYPE_Q5_1 ? 8 : 4);
}

// src0 registers per unit and lane
static constexpr __host__ __device__ int mmvq_moe_cdna_nregs(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return 4;  // 16 bytes qs
        case GGML_TYPE_Q5_K: return 4;  // 16 bytes qs
        case GGML_TYPE_Q5_1: return 6;  // dm, qh, 16 bytes qs
        case GGML_TYPE_Q8_0: return 9;  // 36 bytes from the preceding 4 byte boundary
        default:             return 1;
    }
}

// 16 byte pieces in shared memory per super-block: dm + scales, Q5_K also qh
static constexpr __host__ __device__ int mmvq_moe_cdna_nshared(const ggml_type type) {
    return type == GGML_TYPE_Q4_K ? 1 : (type == GGML_TYPE_Q5_K ? 3 : 0);
}

#if defined(CDNA)
static __device__ __forceinline__ int4 mmvq_moe_cdna_load_a4(const char * p) {
    int4 r;
    __builtin_memcpy(&r, __builtin_assume_aligned(p, 4), sizeof(r));
    return r;
}

static __device__ __forceinline__ void mmvq_moe_cdna_put(int * v, const int4 a) {
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
}

// Unit i of lane c of a row, the K-quants have 8 units per super-block.
template <ggml_type type>
static __device__ __forceinline__ void mmvq_moe_cdna_load_x(int * v, const char * row, const int c, const int i) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        mmvq_moe_cdna_put(v, *(const int4 *) (row + (2*i + c/8)*sizeof(block_q4_K) + 16 + 16*(c % 8)));
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        mmvq_moe_cdna_put(v, *(const int4 *) (row + (2*i + c/8)*sizeof(block_q5_K) + 48 + 16*(c % 8)));
    } else if constexpr (type == GGML_TYPE_Q5_1) {
        const char * p = row + (c + 4*i)*sizeof(block_q5_1);
        const int2 a = *(const int2 *) p;
        v[0] = a.x;
        v[1] = a.y;
        mmvq_moe_cdna_put(v + 2, mmvq_moe_cdna_load_a4(p + 8));
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        const char * p = row + (((c + 4*i)*sizeof(block_q8_0)) & ~3);
        mmvq_moe_cdna_put(v + 0, mmvq_moe_cdna_load_a4(p));
        mmvq_moe_cdna_put(v + 4, mmvq_moe_cdna_load_a4(p + 16));
        v[8] = *(const int *) (p + 32);
    }
}

// Sub-block scales and mins of a K-quant super-block for the sub-blocks 2*g and 2*g + 1, same as vec_dot_q4_K_q8_1.
static __device__ __forceinline__ void mmvq_moe_cdna_scales_K(const int4 h, const int g, int & sc0, int & sc1, int & m0, int & m1) {
    const int      jm = 16*(g & 1);
    const uint32_t s0 = (uint32_t(h.y) >> jm) & 0xFFFF;
    const uint32_t s2 = (uint32_t(h.z) >> jm) & 0xFFFF;
    const uint32_t s4 = (uint32_t(h.w) >> jm) & 0xFFFF;
    const uint32_t hi = -(uint32_t) (g >= 2);
    const uint32_t sc = ((s0 & 0x3F3F) & ~hi) | ((((s4 >> 0) & 0x0F0F) | ((s0 & 0xC0C0) >> 2)) & hi);
    const uint32_t m  = ((s2 & 0x3F3F) & ~hi) | ((((s4 >> 4) & 0x0F0F) | ((s2 & 0xC0C0) >> 2)) & hi);
    sc0 = sc & 0xFF;
    sc1 = sc >> 8;
    m0  = m & 0xFF;
    m1  = m >> 8;
}

// xs: the shared memory pieces of the super-block, y: values [0, 16) and [16, 32) of the unit, aux: the q8_1 scales, see mmvq_moe_cdna_store_y.
template <ggml_type type>
static __device__ __forceinline__ float mmvq_moe_cdna_dot(
        const int * v, const int4 * xs, const int4 ylo, const int4 yhi, const float4 aux, const int c) {
    const int ul[4] = {ylo.x, ylo.y, ylo.z, ylo.w};
    const int uh[4] = {yhi.x, yhi.y, yhi.z, yhi.w};
    int sl = 0;
    int sh = 0;

    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const int  g  = (c % 8)/2;
        const int4 h  = xs[0];
        int        qh[4];
        if constexpr (type == GGML_TYPE_Q5_K) {
            mmvq_moe_cdna_put(qh, xs[1 + c % 2]);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            int vl = (v[k] >> 0) & 0x0F0F0F0F;
            int vh = (v[k] >> 4) & 0x0F0F0F0F;
            if constexpr (type == GGML_TYPE_Q5_K) {
                vl |= ((qh[k] >> (2*g + 0)) << 4) & 0x10101010;
                vh |= ((qh[k] >> (2*g + 1)) << 4) & 0x10101010;
            }
            sl = ggml_cuda_dp4a(vl, ul[k], sl);
            sh = ggml_cuda_dp4a(vh, uh[k], sh);
        }
        int sc0, sc1, m0, m1;
        mmvq_moe_cdna_scales_K(h, g, sc0, sc1, m0, m1);
        const float2 dm = __half22float2(*(const half2 *) &h.x);
        return dm.x*(aux.x*float(sc0*sl) + aux.y*float(sc1*sh)) - dm.y*(aux.z*float(m0) + aux.w*float(m1));
    } else if constexpr (type == GGML_TYPE_Q5_1) {
        GGML_UNUSED(xs);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int qh = v[1] >> (4*k);
            int vl = (v[2 + k] >> 0) & 0x0F0F0F0F;
            vl |= (qh <<  4) & 0x00000010;
            vl |= (qh << 11) & 0x00001000;
            vl |= (qh << 18) & 0x00100000;
            vl |= (qh << 25) & 0x10000000;
            int vh = (v[2 + k] >> 4) & 0x0F0F0F0F;
            vh |= (qh >> 12) & 0x00000010;
            vh |= (qh >>  5) & 0x00001000;
            vh |= (qh <<  2) & 0x00100000;
            vh |= (qh <<  9) & 0x10000000;
            sl = ggml_cuda_dp4a(vl, ul[k], sl);
            sh = ggml_cuda_dp4a(vh, uh[k], sh);
        }
        const float2 dm = __half22float2(*(const half2 *) &v[0]);
        return dm.x*aux.x*float(sl + sh) + dm.y*aux.y;
    } else {
        GGML_UNUSED(xs);
        // odd lanes have blocks at 2 bytes past a 4 byte boundary
        const bool odd = c & 1;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int ql = odd ? v[k + 1] : __builtin_amdgcn_alignbyte(v[k + 1], v[k + 0], 2);
            const int qh = odd ? v[k + 5] : __builtin_amdgcn_alignbyte(v[k + 5], v[k + 4], 2);
            sl = ggml_cuda_dp4a(ql, ul[k], sl);
            sh = ggml_cuda_dp4a(qh, uh[k], sh);
        }
        const float2 d2 = __half22float2(*(const half2 *) &v[0]);
        return (odd ? d2.y : d2.x)*aux.x*float(sl + sh);
    }
}

// For the K-quants unit u holds 16 values of sub-block 2*g and 16 values of sub-block 2*g + 1, else a whole q8_1 block.
struct mmvq_moe_cdna_y {
    int4 lo;
    int4 hi;
    int  ds_lo;
    int  ds_hi;
};

template <ggml_type type>
static __device__ __forceinline__ mmvq_moe_cdna_y mmvq_moe_cdna_load_y(const char * y, const int u) {
    mmvq_moe_cdna_y r;
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const char * blo = y + (8*(u/8) + 2*((u % 8)/2))*sizeof(block_q8_1);
        const char * bhi = blo + sizeof(block_q8_1);
        r.lo    = mmvq_moe_cdna_load_a4(blo + 4 + 16*(u % 2));
        r.hi    = mmvq_moe_cdna_load_a4(bhi + 4 + 16*(u % 2));
        r.ds_lo = *(const int *) blo;
        r.ds_hi = *(const int *) bhi;
    } else {
        const char * b = y + u*sizeof(block_q8_1);
        r.lo    = mmvq_moe_cdna_load_a4(b + 4);
        r.hi    = mmvq_moe_cdna_load_a4(b + 20);
        r.ds_lo = *(const int *) b;
        r.ds_hi = 0;
    }
    return r;
}

// aux: K-quants (d8 lo, d8 hi, d8*sum lo, d8*sum hi), else (d8, d8*sum).
template <ggml_type type>
static __device__ __forceinline__ void mmvq_moe_cdna_store_y(const mmvq_moe_cdna_y & r, int4 & ylo, int4 & yhi, float4 & aux) {
    ylo = r.lo;
    yhi = r.hi;
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const float dlo = __low2float(*(const half2 *) &r.ds_lo);
        const float dhi = __low2float(*(const half2 *) &r.ds_hi);
        int slo = 0;
        int shi = 0;
        slo = ggml_cuda_dp4a(0x01010101, r.lo.x, slo);
        slo = ggml_cuda_dp4a(0x01010101, r.lo.y, slo);
        slo = ggml_cuda_dp4a(0x01010101, r.lo.z, slo);
        slo = ggml_cuda_dp4a(0x01010101, r.lo.w, slo);
        shi = ggml_cuda_dp4a(0x01010101, r.hi.x, shi);
        shi = ggml_cuda_dp4a(0x01010101, r.hi.y, shi);
        shi = ggml_cuda_dp4a(0x01010101, r.hi.z, shi);
        shi = ggml_cuda_dp4a(0x01010101, r.hi.w, shi);
        aux = make_float4(dlo, dhi, dlo*float(slo), dhi*float(shi));
    } else {
        const float2 ds = __half22float2(*(const half2 *) &r.ds_lo);
        aux = make_float4(ds.x, ds.y, 0.0f, 0.0f);
    }
}
#endif // defined(CDNA)

// max. pairs per block in one pass: the y of the pairs in shared memory, at most 16 KB per block for 4 blocks per CU, one unit per thread, at most 4 sums
template <ggml_type type, int nu, bool has_gate>
static constexpr __host__ __device__ int mmvq_moe_cdna_npairs_max() {
    constexpr int nunits = mmvq_moe_cdna_nlanes(type)*nu;
    constexpr int nxs    = (has_gate ? 2 : 1)*MMVQ_MOE_CDNA_NWARPS*64/mmvq_moe_cdna_nlanes(type)*(nunits/8)*mmvq_moe_cdna_nshared(type);
    constexpr int n_lds  = (16*1024 - 16*nxs)/(3*16*nunits);
    constexpr int n_thr  = MMVQ_MOE_CDNA_NWARPS*64/nunits;
    constexpr int n      = n_lds < n_thr ? n_lds : n_thr;
    return n < 1 ? 1 : (n > 4 ? 4 : n);
}

// multi with gate: 3 waves per SIMD, with 4 the registers are too few to issue all weight loads at once
template <ggml_type type, int nu, bool has_gate, bool multi>
__launch_bounds__(MMVQ_MOE_CDNA_NWARPS*64, multi && has_gate ? 3 : MMVQ_MOE_CDNA_MIN_WAVES)
static __global__ void mul_mat_vec_q_moe_cdna(
        const char * __restrict__ vx, const char * __restrict__ vy, const int32_t * __restrict__ ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const int nrows_x, const int64_t stride_row_x, const int64_t stride_channel_x,
        const int stride_col_y, const int stride_channel_y, const int nchannels_y,
        const int stride_col_dst, const int stride_channel_dst, const int ids_stride) {
#if defined(CDNA)
    constexpr int nlanes         = mmvq_moe_cdna_nlanes(type);
    constexpr int nregs          = mmvq_moe_cdna_nregs(type);
    constexpr int nunits         = nlanes*nu;
    constexpr int rows_per_warp  = 64/nlanes;
    constexpr int rows_per_block = MMVQ_MOE_CDNA_NWARPS*rows_per_warp;
    constexpr int nshared        = mmvq_moe_cdna_nshared(type);
    constexpr int nsb            = nshared > 0 ? nunits/8 : 1; // super-blocks per row
    constexpr int nmat           = has_gate ? 2 : 1;
    constexpr int nxs            = nmat*rows_per_block*nsb*nshared;
    // Q5_1, Q8_0: x has too many registers for several sums, the block keeps x and computes the pairs one after another
    constexpr bool loop_pairs    = multi && nlanes == 4;
    static_assert(!loop_pairs || !has_gate, "no gate with loop_pairs");
    constexpr int npmax          = multi && !loop_pairs ? mmvq_moe_cdna_npairs_max<type, nu, has_gate>() : 1;
    constexpr int nthreads       = MMVQ_MOE_CDNA_NWARPS*64;
    static_assert(npmax*nunits <= nthreads, "one y unit per thread");

    __shared__ int4   y_lo[npmax*nunits];
    __shared__ int4   y_hi[npmax*nunits];
    __shared__ float4 y_aux[npmax*nunits];
    __shared__ int4   x_sh[nxs > 0 ? nxs : 1];

    const int lane   = threadIdx.x;
    const int warp   = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int token  = blockIdx.z;
    const int slot   = blockIdx.y;
    const int expert = ids[token*ids_stride + slot];

    // multi: the pairs with the same expert as bits token*16 + slot.
    // The pair with rank r among them computes the pairs r, ..., r + npmax - 1 if r % npmax == 0, else nothing.
    // loop_pairs: the pair with rank 0 computes all of them.
    uint64_t same0 = 0; // tokens 0-3
    uint64_t same1 = 0; // tokens 4-7
    if constexpr (multi) {
        int e[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            e[h] = ids[min(4*h + lane/16, int(gridDim.z) - 1)*ids_stride + min(lane % 16, int(gridDim.y) - 1)];
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const bool valid = (4*h + lane/16 < int(gridDim.z)) & (lane % 16 < int(gridDim.y));
            const uint64_t b = __ballot(valid & (e[h] == expert));
            same0 = h == 0 ? b : same0;
            same1 = h == 1 ? b : same1;
        }
        const int      q     = token*16 + slot;
        const uint64_t lt0   = q < 64 ? (uint64_t(1) << q) - 1 : ~uint64_t(0);
        const uint64_t lt1   = q < 64 ? 0 : (uint64_t(1) << (q - 64)) - 1;
        const int      rank  = __builtin_popcountll(same0 & lt0) + __builtin_popcountll(same1 & lt1);
        if (loop_pairs ? rank != 0 : rank % npmax != 0) {
            return;
        }
        same0 &= ~lt0;
        same1 &= ~lt1;
    }

    // No reference to the kernel argument fusion in a lambda, that puts it into scratch memory.
    const char  * gate      = (const char *) fusion.gate;
    const float * x_bias    = (const float *) fusion.x_bias;
    const float * gate_bias = (const float *) fusion.gate_bias;
    const ggml_glu_op glu_op    = fusion.glu_op;
    const float       glu_limit = fusion.glu_limit;

    if constexpr (loop_pairs) {
        const int c   = lane % nlanes;
        const int row = int(blockIdx.x)*rows_per_block + warp*rows_per_warp + lane/nlanes;
        const int tid = warp*64 + lane;
        const int uy  = min(tid, nunits - 1);
        const int64_t offset_x = expert*stride_channel_x + min(row, nrows_x - 1)*stride_row_x;

        auto y_ptr = [&](const int t, const int s) {
            return vy + (t*stride_col_y + (s % nchannels_y)*stride_channel_y)*sizeof(block_q8_1);
        };

        const mmvq_moe_cdna_y ys = mmvq_moe_cdna_load_y<type>(y_ptr(token, slot), uy);
        int x[nu][nregs];
#pragma unroll
        for (int i = 0; i < nu; ++i) {
            mmvq_moe_cdna_load_x<type>(x[i], vx + offset_x, c, i);
        }
        mmvq_moe_cdna_store_y<type>(ys, y_lo[uy], y_hi[uy], y_aux[uy]);
        __syncthreads();

        const int      q0  = token*16 + slot;
        const uint64_t own = uint64_t(1) << (q0 % 64);
        same0 &= q0 < 64 ? ~own : ~uint64_t(0);
        same1 &= q0 < 64 ? ~uint64_t(0) : ~own;
        int t = token;
        int s = slot;
        while (true) {
            // keep the unpacking of x in the loop, out of it the unpacked x takes twice the registers
#pragma unroll
            for (int i = 0; i < nu; ++i) {
#pragma unroll
                for (int k = 0; k < nregs; ++k) {
                    asm volatile("" : "+v"(x[i][k]));
                }
            }

            // the y of the next pair is loaded during the computation of this one. No ctz of 0, it is undefined.
            const bool     lo = same0 != 0;
            const uint64_t m  = lo ? same0 : same1;
            const int      qn = m != 0 ? (lo ? 0 : 64) + __builtin_ctzll(m) : -1;
            same0 = lo ? m & (m - 1) : same0;
            same1 = lo ? same1 : m & (m - 1);
            mmvq_moe_cdna_y yn;
            if (qn >= 0) {
                yn = mmvq_moe_cdna_load_y<type>(y_ptr(qn/16, qn % 16), uy);
            }

            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < nu; ++i) {
                const int u = c + nlanes*i;
                sum += mmvq_moe_cdna_dot<type>(x[i], x_sh, y_lo[u], y_hi[u], y_aux[u], c);
            }
            sum = warp_reduce_sum<nlanes>(sum);
            if (c == 0 && row < nrows_x) {
                if (x_bias) {
                    sum += x_bias[expert*stride_channel_dst + row];
                }
                dst[t*stride_col_dst + s*stride_channel_dst + row] = sum;
            }

            if (qn < 0) {
                break;
            }
            __syncthreads();
            mmvq_moe_cdna_store_y<type>(yn, y_lo[uy], y_hi[uy], y_aux[uy]);
            __syncthreads();
            t = qn/16;
            s = qn % 16;
        }
        return;
    }

    // The block computes np <= npmax pairs (token pt[p], slot ps[p]) in one pass over x, x[i] is not needed after unit i.
    auto compute_pairs = [&](const int (&pt)[npmax], const int (&ps)[npmax], const int np) {
        const int c   = lane % nlanes;
        const int row = int(blockIdx.x)*rows_per_block + warp*rows_per_warp + lane/nlanes;
        const int tid = warp*64 + lane;
        const int64_t offset_x = expert*stride_channel_x + min(row, nrows_x - 1)*stride_row_x;

        // Loads in the order y, x, x_sh: y has the shortest latency and s_waitcnt vmcnt counts in order.
        // Threads past the end load and store a copy of the last element. Thread k stages unit k % nunits of pair k / nunits.
        const int ky = min(tid, npmax*nunits - 1);
        int ty = pt[0];
        int sy = ps[0];
#pragma unroll
        for (int p = 1; p < npmax; ++p) {
            ty = ky/nunits == p && p < np ? pt[p] : ty;
            sy = ky/nunits == p && p < np ? ps[p] : sy;
        }
        const mmvq_moe_cdna_y ys = mmvq_moe_cdna_load_y<type>(
            vy + (ty*stride_col_y + (sy % nchannels_y)*stride_channel_y)*sizeof(block_q8_1), ky % nunits);

        int x[nu][nregs];
#pragma unroll
        for (int i = 0; i < nu; ++i) {
            mmvq_moe_cdna_load_x<type>(x[i], vx + offset_x, c, i);
        }
        [[maybe_unused]] int xg[has_gate ? nu : 1][nregs];
        if constexpr (has_gate) {
#pragma unroll
            for (int i = 0; i < nu; ++i) {
                mmvq_moe_cdna_load_x<type>(xg[i], gate + offset_x, c, i);
            }
        }

        constexpr int nxs_thread = (nxs + nthreads - 1)/nthreads;
        [[maybe_unused]] int xs[nxs_thread > 0 ? nxs_thread : 1][4]; // no int4: its copies are memcpy and keep the array in scratch memory
        if constexpr (nshared > 0) {
#pragma unroll
            for (int j = 0; j < nxs_thread; ++j) {
                const int k  = min(j*nthreads + tid, nxs - 1);
                const int m  = k / (rows_per_block*nsb*nshared);
                const int r  = (k / (nsb*nshared)) % rows_per_block;
                const int sb = (k / nshared) % nsb;
                const int p  = k % nshared;
                const int rx = min(int(blockIdx.x)*rows_per_block + r, nrows_x - 1);
                const char * px = (m == 0 ? vx : gate) + expert*stride_channel_x + rx*stride_row_x;
                mmvq_moe_cdna_put(xs[j], *(const int4 *) (px + sb*ggml_cuda_type_traits<type>::bs + 16*p));
            }
        }

        mmvq_moe_cdna_store_y<type>(ys, y_lo[ky], y_hi[ky], y_aux[ky]);
        if constexpr (nshared > 0) {
#pragma unroll
            for (int j = 0; j < nxs_thread; ++j) {
                x_sh[min(j*nthreads + tid, nxs - 1)] = make_int4(xs[j][0], xs[j][1], xs[j][2], xs[j][3]);
            }
        }
        __syncthreads();

        float sum[npmax];
        [[maybe_unused]] float sum_gate[npmax];
#pragma unroll
        for (int p = 0; p < npmax; ++p) {
            sum[p]      = 0.0f;
            sum_gate[p] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < nu; ++i) {
            const int u  = c + nlanes*i;
            // super-block of the unit in x_sh
            const int js = ((warp*rows_per_warp + lane/nlanes)*nsb + (nshared > 0 ? u/8 : 0))*nshared;
#pragma unroll
            for (int p = 0; p < npmax; ++p) {
                if (p < np) {
                    const int4   ylo = y_lo[p*nunits + u];
                    const int4   yhi = y_hi[p*nunits + u];
                    const float4 aux = y_aux[p*nunits + u];
                    sum[p] += mmvq_moe_cdna_dot<type>(x[i], x_sh + js, ylo, yhi, aux, c);
                    if constexpr (has_gate) {
                        sum_gate[p] += mmvq_moe_cdna_dot<type>(xg[i], x_sh + js + rows_per_block*nsb*nshared, ylo, yhi, aux, c);
                    }
                }
            }
        }

#pragma unroll
        for (int p = 0; p < npmax; ++p) {
            if (p < np) {
                float r = warp_reduce_sum<nlanes>(sum[p]);
                [[maybe_unused]] float rg = 0.0f;
                if constexpr (has_gate) {
                    rg = warp_reduce_sum<nlanes>(sum_gate[p]);
                }
                if (c == 0 && row < nrows_x) {
                    const int bias_idx = expert*stride_channel_dst + row;
                    if (x_bias) {
                        r += x_bias[bias_idx];
                    }
                    if constexpr (has_gate) {
                        if (gate_bias) {
                            rg += gate_bias[bias_idx];
                        }
                        r = ggml_cuda_op_glu_single(glu_op, rg, r, glu_limit);
                    }
                    dst[pt[p]*stride_col_dst + ps[p]*stride_channel_dst + row] = r;
                }
            }
        }
    };

    // the own pair and the next pairs with the same expert
    int pt[npmax];
    int ps[npmax];
    auto next_pairs = [&]() {
        const int n  = __builtin_popcountll(same0) + __builtin_popcountll(same1);
        const int np = n < npmax ? n : npmax;
#pragma unroll
        for (int p = 0; p < npmax; ++p) {
            // no ctz of 0, it is undefined. Selects, not branches: with branches same0 and same1 end up in scratch memory.
            const bool     lo = same0 != 0;
            const uint64_t m  = lo ? same0 : same1;
            const int      q  = m != 0 ? (lo ? 0 : 64) + __builtin_ctzll(m) : 0;
            same0 = lo ? m & (m - 1) : same0;
            same1 = lo ? same1 : m & (m - 1);
            pt[p] = p < np ? q/16 : pt[0];
            ps[p] = p < np ? q % 16 : ps[0];
        }
        return np;
    };

    if constexpr (!multi) {
        pt[0] = token;
        ps[0] = slot;
        compute_pairs(pt, ps, 1);
        return;
    }
    compute_pairs(pt, ps, next_pairs());
#else
    GGML_UNUSED_VARS(vx, vy, ids, fusion, dst, nrows_x, stride_row_x, stride_channel_x, stride_col_y, stride_channel_y,
        nchannels_y, stride_col_dst, stride_channel_dst, ids_stride);
    NO_DEVICE_CODE;
#endif // defined(CDNA)
}

template <ggml_type type, int nu>
static void mul_mat_vec_q_moe_cdna_launch(
        const ggml_tensor * src0, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int64_t ncols_dst, const int64_t nchannels_dst, const int64_t nchannels_y, const int64_t stride_col_y, const int64_t stride_channel_y,
        const int64_t stride_col_dst, const int64_t stride_channel_dst, const int64_t ids_stride, cudaStream_t stream) {
    constexpr int rows_per_block = MMVQ_MOE_CDNA_NWARPS*64/mmvq_moe_cdna_nlanes(type);

    const int nrows_x = src0->ne[1];
    const dim3 block_nums((nrows_x + rows_per_block - 1)/rows_per_block, nchannels_dst, ncols_dst);
    const dim3 block_dims(64, MMVQ_MOE_CDNA_NWARPS, 1);

    // with 1 token only duplicate ids within the token could share a block
    const bool multi = ncols_dst > 1 && ncols_dst <= MMVQ_MOE_CDNA_DEDUP_MAX_TOKENS && nchannels_dst <= MMVQ_MOE_CDNA_DEDUP_MAX_SLOTS;

    const char * vx = (const char *) src0->data;
    const char * y  = (const char *) vy;
    // the fused Q5_1 and Q8_0 kernels with several pairs per block spill registers
    constexpr bool multi_gate = mmvq_moe_cdna_nlanes(type) == 16;
    if (fusion.gate) {
        if (multi_gate && multi) {
            mul_mat_vec_q_moe_cdna<type, nu, true, multi_gate><<<block_nums, block_dims, 0, stream>>>(
                vx, y, ids, fusion, dst, nrows_x, src0->nb[1], src0->nb[2], stride_col_y, stride_channel_y, nchannels_y,
                stride_col_dst, stride_channel_dst, ids_stride);
        } else {
            mul_mat_vec_q_moe_cdna<type, nu, true, false><<<block_nums, block_dims, 0, stream>>>(
                vx, y, ids, fusion, dst, nrows_x, src0->nb[1], src0->nb[2], stride_col_y, stride_channel_y, nchannels_y,
                stride_col_dst, stride_channel_dst, ids_stride);
        }
    } else {
        if (multi) {
            mul_mat_vec_q_moe_cdna<type, nu, false, true><<<block_nums, block_dims, 0, stream>>>(
                vx, y, ids, fusion, dst, nrows_x, src0->nb[1], src0->nb[2], stride_col_y, stride_channel_y, nchannels_y,
                stride_col_dst, stride_channel_dst, ids_stride);
        } else {
            mul_mat_vec_q_moe_cdna<type, nu, false, false><<<block_nums, block_dims, 0, stream>>>(
                vx, y, ids, fusion, dst, nrows_x, src0->nb[1], src0->nb[2], stride_col_y, stride_channel_y, nchannels_y,
                stride_col_dst, stride_channel_dst, ids_stride);
        }
    }
}

template <ggml_type type>
static void mul_mat_vec_q_moe_cdna_switch_nu(
        const ggml_tensor * src0, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int64_t ncols_dst, const int64_t nchannels_dst, const int64_t nchannels_y, const int64_t stride_col_y, const int64_t stride_channel_y,
        const int64_t stride_col_dst, const int64_t stride_channel_dst, const int64_t ids_stride, cudaStream_t stream) {
    const int64_t nu = src0->ne[0] / (QK8_1*mmvq_moe_cdna_nlanes(type));
    switch (nu) {
        case 1: mul_mat_vec_q_moe_cdna_launch<type, 1>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream); break;
        case 2: mul_mat_vec_q_moe_cdna_launch<type, 2>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream); break;
        case 3: mul_mat_vec_q_moe_cdna_launch<type, 3>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream); break;
        case 4: mul_mat_vec_q_moe_cdna_launch<type, 4>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream); break;
        case 5: mul_mat_vec_q_moe_cdna_launch<type, 5>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream); break;
        case 6: mul_mat_vec_q_moe_cdna_launch<type, 6>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream); break;
        case 7: mul_mat_vec_q_moe_cdna_launch<type, 7>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream); break;
        case 8: mul_mat_vec_q_moe_cdna_launch<type, 8>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_mmvq_moe_cdna_supported(const ggml_tensor * src0, const ggml_tensor * dst, const int cc, const ggml_cuda_mm_fusion_args_device & fusion) {
    if (!GGML_CUDA_CC_IS_CDNA(cc) || !mmvq_moe_cdna_type_supported(src0->type)) {
        return false;
    }
    if (fusion.x_scale || fusion.gate_scale || fusion.shared_up) {
        return false;
    }
    // fused Q5_K is not faster than mul_mat_vec_q for 1-2 tokens
    if (src0->type == GGML_TYPE_Q5_K && fusion.gate) {
        return false;
    }
    if (src0->ne[3] != 1 || dst->ne[3] != 1) {
        return false;
    }
    const int64_t unit_k = QK8_1*mmvq_moe_cdna_nlanes(src0->type);
    if (src0->ne[0] % unit_k != 0 || src0->ne[0]/unit_k > MMVQ_MOE_CDNA_MAX_NU) {
        return false;
    }
    const size_t alignment = mmvq_moe_cdna_alignment(src0->type);
    if (!ggml_cuda_is_aligned(src0, alignment)) {
        return false;
    }
    if (fusion.gate && (uintptr_t) fusion.gate % alignment != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_mmvq_moe_cdna(
        const ggml_tensor * src0, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int64_t ncols_dst, const int64_t nchannels_dst, const int64_t nchannels_y, const int64_t stride_col_y, const int64_t stride_channel_y,
        const int64_t stride_col_dst, const int64_t stride_channel_dst, const int64_t ids_stride, cudaStream_t stream) {
    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_moe_cdna_switch_nu<GGML_TYPE_Q4_K>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_moe_cdna_switch_nu<GGML_TYPE_Q5_K>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_moe_cdna_switch_nu<GGML_TYPE_Q5_1>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_moe_cdna_switch_nu<GGML_TYPE_Q8_0>(src0, vy, ids, fusion, dst, ncols_dst, nchannels_dst, nchannels_y, stride_col_y, stride_channel_y, stride_col_dst, stride_channel_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
