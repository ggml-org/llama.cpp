#include "mmq-few.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <type_traits>

// NVIDIA only: the kernel uses the NVIDIA int8 MMA fragment layout and cp.async.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

using namespace ggml_cuda_mma;

// Few-row MMQ: int8 MMA mat-mul for 2-24 src1 columns. A CUDA block has 16 src0 rows (32 for 2 column groups on large matrices).
// Its warps split K in units of 256 values and copy raw src0 rows into their own cp.async ring, src1 is quantized to the MMA fragment layout.
// MUL_MAT_ID: one column group per expert and CUDA block, for short rows the warps split the rows instead of K (64 rows per block).

static constexpr int MMQ_FEW_MAX_NCG = 3;  // max. number of column groups
static constexpr int MMQ_FEW_NROWS   = 16; // src0 rows of one MMA tile
static constexpr int MMQ_FEW_NWARPS  = 4;

template <ggml_type type>
static constexpr __host__ __device__ int mmq_few_unit_size_of() {
    return QK_K/ggml_cuda_type_traits<type>::qk*ggml_cuda_type_traits<type>::bs;
}

// src0 bytes per 256 values, 0 for the types without a few-row decoder.
static constexpr __host__ __device__ int mmq_few_unit_size(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return mmq_few_unit_size_of<GGML_TYPE_Q8_0>();
        case GGML_TYPE_Q3_K:   return mmq_few_unit_size_of<GGML_TYPE_Q3_K>();
        case GGML_TYPE_Q4_K:   return mmq_few_unit_size_of<GGML_TYPE_Q4_K>();
        case GGML_TYPE_Q5_K:   return mmq_few_unit_size_of<GGML_TYPE_Q5_K>();
        case GGML_TYPE_Q6_K:   return mmq_few_unit_size_of<GGML_TYPE_Q6_K>();
        case GGML_TYPE_IQ3_S:  return mmq_few_unit_size_of<GGML_TYPE_IQ3_S>();
        case GGML_TYPE_IQ4_NL: return mmq_few_unit_size_of<GGML_TYPE_IQ4_NL>();
        case GGML_TYPE_IQ4_XS: return mmq_few_unit_size_of<GGML_TYPE_IQ4_XS>();
        default:               return 0;
    }
}

// Q4_K and Q5_K also need the src1 sums for their mins.
static constexpr __host__ __device__ mmq_q8_1_ds_layout mmq_few_ds_layout(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
}

// Units of 256 values per ring stage: 1 for Q6_K, 3 column groups (fewer free registers) and 32-row tiles (2 blocks per SM), else 2.
static constexpr __host__ __device__ int mmq_few_nunits_stage(const ggml_type type, const int ncg, const int nmt) {
    return type == GGML_TYPE_Q6_K || ncg > 2 || nmt > 1 ? 1 : 2;
}

// A third ring stage only makes Q6_K faster, the other types get slower on large matrices.
static constexpr __host__ __device__ int mmq_few_nstages(const ggml_type type) {
    return type == GGML_TYPE_Q6_K ? 3 : 2;
}

// The src0 data of a ring stage does not fill whole 16 byte chunks, it is copied from the preceding 16 byte boundary.
static constexpr __host__ __device__ bool mmq_few_unaligned(const ggml_type type, const int nunits_stage) {
    return (nunits_stage*mmq_few_unit_size(type)) % 16 != 0;
}

// for unaligned data the max. over the 8 possible start offsets
static constexpr __host__ __device__ int mmq_few_nchunks(const ggml_type type, const int nunits_stage) {
    const int nbytes = nunits_stage*mmq_few_unit_size(type);
    return mmq_few_unaligned(type, nunits_stage) ? (14 + nbytes + 15)/16 : nbytes/16;
}

// Unaligned rows are padded to an odd number of 16 byte chunks against bank conflicts, for aligned rows the padding costs more than it saves.
static constexpr __host__ __device__ int mmq_few_sram_stride(const ggml_type type, const int nunits_stage) {
    const int nchunks = mmq_few_nchunks(type, nunits_stage);
    return mmq_few_unaligned(type, nunits_stage) && nchunks % 2 == 0 ? 16*(nchunks + 1) : 16*nchunks;
}

static constexpr __host__ __device__ int mmq_few_nbytes_shared(const ggml_type type, const int ncg, const int nmt) {
    return MMQ_FEW_NWARPS*mmq_few_nstages(type)*nmt*MMQ_FEW_NROWS*mmq_few_sram_stride(type, mmq_few_nunits_stage(type, ncg, nmt));
}

// With Q6_K and Q8_0 two CUDA blocks with 32-row tiles would not fit into an SM.
static constexpr __host__ __device__ bool mmq_few_tall_supported(const ggml_type type) {
    return type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS;
}

// MUL_MAT_ID with fewer than 2 units of 256 values per warp: the warps split the rows, each warp has its own 16 rows.
static constexpr __host__ __device__ bool mmq_few_ksplit(const bool has_ids, const int nunits) {
    return !has_ids || nunits >= 2*MMQ_FEW_NWARPS;
}

#if defined(CP_ASYNC_AVAILABLE)

// The src1 fragments of a lane for one unit of 256 values, see quantize.cuh.
struct mmq_few_y_frag {
    int4 qs[4];
    int4 ds[4];
};

static __device__ __forceinline__ void mmq_few_load_y(mmq_few_y_frag & yf, const int4 * y) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.qs[j] = y[j*WARP_SIZE + threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.ds[j] = y[4*WARP_SIZE + 4*(threadIdx.x % 4) + j];
    }
}

static __device__ __forceinline__ int2 mmq_few_get_b(const mmq_few_y_frag & yf, const int s) {
    const int4 q = yf.qs[s/2];
    return s % 2 == 0 ? make_int2(q.x, q.y) : make_int2(q.z, q.w);
}

template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ float2 mmq_few_get_d(const mmq_few_y_frag & yf, const int s) {
    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        const float * d = (const float *) &yf.ds[s/2];
        return make_float2(d[2*(s % 2) + 0], d[2*(s % 2) + 1]);
    } else {
        const half2 * ds = (const half2 *) &yf.ds[s/2];
        return make_float2(__low2float(ds[2*(s % 2) + 0]), __low2float(ds[2*(s % 2) + 1]));
    }
}

static __device__ __forceinline__ float2 mmq_few_get_sum(const mmq_few_y_frag & yf, const int s) {
    const half2 * ds = (const half2 *) &yf.ds[s/2];
    return make_float2(__high2float(ds[2*(s % 2) + 0]), __high2float(ds[2*(s % 2) + 1]));
}

// C.x[0], C.x[1]: row g, columns 2t and 2t+1. C.x[2], C.x[3]: row g+8.
static __device__ __forceinline__ tile<16, 8, int> mmq_few_mma(const int (&lo)[2], const int (&hi)[2], const int2 b) {
    tile<16, 8, int> A;
    A.x[0] = lo[0];
    A.x[1] = lo[1];
    A.x[2] = hi[0];
    A.x[3] = hi[1];
    tile<8, 8, int> B;
    B.x[0] = b.x;
    B.x[1] = b.y;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

static __device__ __forceinline__ tile<16, 8, int> mmq_few_mma_k16(const int (&a)[2], const int b) {
    tile<16, 4, int> A;
    A.x[0] = a[0];
    A.x[1] = a[1];
    tile<8, 4, int> B;
    B.x[0] = b;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

// sum_d += d * (C * ls) of sub-block s for all column groups, ls are the src0 row scales
template <ggml_type type, int ncg, typename scale_t>
static __device__ __forceinline__ void mmq_few_mma_add(
        const int (&lo)[2], const int (&hi)[2], const scale_t (&ls)[2], const mmq_few_y_frag (&yf)[ncg], const int s, float (&sum_d)[ncg][4]) {
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
        const tile<16, 8, int> C = mmq_few_mma(lo, hi, mmq_few_get_b(yf[c], s));

        const float2 d = mmq_few_get_d<mmq_few_ds_layout(type)>(yf[c], s);
        sum_d[c][0] += d.x * float(C.x[0]*ls[0]);
        sum_d[c][1] += d.y * float(C.x[1]*ls[0]);
        sum_d[c][2] += d.x * float(C.x[2]*ls[1]);
        sum_d[c][3] += d.y * float(C.x[3]*ls[1]);
    }
}

// Same with one k=16 MMA per half of sub-block s, the int8 scales of the halves are bytes 0 and 1 of sc[r].
template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_mma_k16_add(
        const int (&lo)[2], const int (&hi)[2], const int (&sc)[2], const mmq_few_y_frag (&yf)[ncg], const int s, float (&sum_d)[ncg][4]) {
    const int sc_lo[2] = {(int8_t) sc[0], (int8_t) sc[1]};
    const int sc_hi[2] = {(int8_t) (sc[0] >> 8), (int8_t) (sc[1] >> 8)};
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
        const int2 b = mmq_few_get_b(yf[c], s);
        const tile<16, 8, int> C_lo = mmq_few_mma_k16(lo, b.x);
        const tile<16, 8, int> C_hi = mmq_few_mma_k16(hi, b.y);

        const float2 d = mmq_few_get_d<mmq_few_ds_layout(type)>(yf[c], s);
        sum_d[c][0] += d.x * float(C_lo.x[0]*sc_lo[0] + C_hi.x[0]*sc_hi[0]);
        sum_d[c][1] += d.y * float(C_lo.x[1]*sc_lo[0] + C_hi.x[1]*sc_hi[0]);
        sum_d[c][2] += d.x * float(C_lo.x[2]*sc_lo[1] + C_hi.x[2]*sc_hi[1]);
        sum_d[c][3] += d.y * float(C_lo.x[3]*sc_lo[1] + C_hi.x[3]*sc_hi[1]);
    }
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_add_scaled(float (&sum)[ncg][4], const float (&d)[2], const float (&sum_d)[ncg][4]) {
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[c][l] += d[l/2]*sum_d[c][l];
        }
    }
}

// vec_dot: one unit of 256 values of the 16 rows of a warp, x[0] and x[1] are the rows g and g+8 of lane (g, t).
// The lane decodes the values [4t, 4t + 4) and [16 + 4t, 16 + 4t + 4) of each sub-block, its A fragments.

template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q45_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
    const int t = threadIdx.x % 4;

    int   qs[2][4][2];
    [[maybe_unused]] int qh[2][2]; // Q5_K: bit s holds the high bit of sub-block s
    int   sc[2][2];
    int   mn[2][2];
    float d[2];
    float dmin[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_t * bx = (const block_t *) x[r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qs[r][j][0] = get_int_b4(bx->qs + 32*j,      t);
            qs[r][j][1] = get_int_b4(bx->qs + 32*j + 16, t);
        }
        if constexpr (type == GGML_TYPE_Q5_K) {
            qh[r][0] = get_int_b4(bx->qh,      t);
            qh[r][1] = get_int_b4(bx->qh + 16, t);
        }
        const int4 hdr = *(const int4 *) bx; // dm and the 12 scale bytes
        const int scales[3] = {hdr.y, hdr.z, hdr.w};
        sc[r][0] = unpack_scales_q45_K(scales, 0);
        sc[r][1] = unpack_scales_q45_K(scales, 1);
        mn[r][0] = unpack_scales_q45_K(scales, 2);
        mn[r][1] = unpack_scales_q45_K(scales, 3);
        const float2 dm = __half22float2(bx->dm);
        d[r]    = dm.x;
        dmin[r] = dm.y;
    }

    float sum_d[ncg][4] = {{0.0f}};
    float sum_m[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int j     = s/2;
        const int shift = 4*(s % 2);
        int lo[2];
        int hi[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            lo[r] = (qs[r][j][0] >> shift) & 0x0F0F0F0F;
            hi[r] = (qs[r][j][1] >> shift) & 0x0F0F0F0F;
            if constexpr (type == GGML_TYPE_Q5_K) {
                lo[r] |= ((qh[r][0] >> s) & 0x01010101) << 4;
                hi[r] |= ((qh[r][1] >> s) & 0x01010101) << 4;
            }
        }
        const int   ls[2] = {(sc[0][s/4] >> (8*(s % 4))) & 0xFF, (sc[1][s/4] >> (8*(s % 4))) & 0xFF};
        const float m0    = (mn[0][s/4] >> (8*(s % 4))) & 0xFF;
        const float m1    = (mn[1][s/4] >> (8*(s % 4))) & 0xFF;
        mmq_few_mma_add<type>(lo, hi, ls, yf, s, sum_d);
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
            const float2 sm = mmq_few_get_sum(yf[c], s);
            sum_m[c][0] += m0 * sm.x;
            sum_m[c][1] += m0 * sm.y;
            sum_m[c][2] += m1 * sm.x;
            sum_m[c][3] += m1 * sm.y;
        }
    }
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[c][l] += d[l/2]*sum_d[c][l] - dmin[l/2]*sum_m[c][l];
        }
    }
}

// Q6_K and Q3_K have a scale per 16 values: one k=16 MMA per half of a sub-block.
template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q6_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        d[r] = __half2float(((const block_q6_K *) x[r])->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int ql[2][2][2]; // [row][ql offset 0 or 32][lo or hi]
        int qh[2][2];
        int sc[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q6_K * bx = (const block_q6_K *) x[r];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                ql[r][k][0] = get_int_b2(bx->ql + 64*n + 32*k,      t);
                ql[r][k][1] = get_int_b2(bx->ql + 64*n + 32*k + 16, t);
            }
            qh[r][0] = get_int_b2(bx->qh + 32*n,      t);
            qh[r][1] = get_int_b2(bx->qh + 32*n + 16, t);
            sc[r][0] = get_int_b2(bx->scales + 8*n, 0);
            sc[r][1] = get_int_b2(bx->scales + 8*n, 1);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4*n + j;
            int lo[2];
            int hi[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                lo[r] = __vsub4(((ql[r][j % 2][0] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][0] >> (2*j)) & 0x03030303) << 4), 0x20202020);
                hi[r] = __vsub4(((ql[r][j % 2][1] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][1] >> (2*j)) & 0x03030303) << 4), 0x20202020);
            }
            const int sc_s[2] = {sc[0][j/2] >> (16*(j % 2)), sc[1][j/2] >> (16*(j % 2))}; // scales 2s and 2s+1
            mmq_few_mma_k16_add<GGML_TYPE_Q6_K>(lo, hi, sc_s, yf, s, sum_d);
        }
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q3_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    int   hm[2][2];
    int   sc[2][4]; // the 16 scales of a row as int8, minus 32
    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_q3_K * bx = (const block_q3_K *) x[r];
        hm[r][0] = ~get_int_b2(bx->hmask,      t); // a set bit means that 4 is subtracted
        hm[r][1] = ~get_int_b2(bx->hmask + 16, t);
#pragma unroll
        for (int ksc = 0; ksc < 4; ++ksc) {
            sc[r][ksc] = unpack_scales_q3_K(bx->scales, ksc);
        }
        d[r] = __half2float(bx->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int qs[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q3_K * bx = (const block_q3_K *) x[r];
            qs[r][0] = get_int_b2(bx->qs + 32*n,      t);
            qs[r][1] = get_int_b2(bx->qs + 32*n + 16, t);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4*n + j;
            int lo[2];
            int hi[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                lo[r] = __vsub4((qs[r][0] >> (2*j)) & 0x03030303, ((hm[r][0] >> s) & 0x01010101) << 2);
                hi[r] = __vsub4((qs[r][1] >> (2*j)) & 0x03030303, ((hm[r][1] >> s) & 0x01010101) << 2);
            }
            const int sc_s[2] = {sc[0][s/2] >> (16*(s % 2)), sc[1][s/2] >> (16*(s % 2))}; // scales 2s and 2s+1
            mmq_few_mma_k16_add<GGML_TYPE_Q3_K>(lo, hi, sc_s, yf, s, sum_d);
        }
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_iq4_xs(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
    int   scales_h[2];
    int   scales_l[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_iq4_xs * bx = (const block_iq4_xs *) x[r];
        d[r]        = __half2float(bx->d);
        scales_h[r] = bx->scales_h;
        scales_l[r] = get_int_b4(bx->scales_l, 0);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int lo[2];
        int hi[2];
        int ls[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_iq4_xs * bx = (const block_iq4_xs *) x[r];
            const int2 v = get_int_from_table_16(get_int_b4(bx->qs + 16*s, t), kvalues_iq4nl);
            lo[r] = v.x;
            hi[r] = v.y;
            ls[r] = (((scales_l[r] >> (4*s)) & 0x0F) | (((scales_h[r] >> (2*s)) & 0x03) << 4)) - 32;
        }
        mmq_few_mma_add<GGML_TYPE_IQ4_XS>(lo, hi, ls, yf, s, sum_d);
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

// Q8_0 and IQ4_NL have a block with its own scale per 32 values.
template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_q8_0_iq4_nl(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int   lo[2];
        int   hi[2];
        float d[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if constexpr (type == GGML_TYPE_Q8_0) {
                const block_q8_0 * bx = (const block_q8_0 *) x[r] + s;
                lo[r] = get_int_b2(bx->qs,      t);
                hi[r] = get_int_b2(bx->qs + 16, t);
                d[r]  = __half2float(bx->d);
            } else {
                const block_iq4_nl * bx = (const block_iq4_nl *) x[r] + s;
                const int2 v = get_int_from_table_16(get_int_b2(bx->qs, t), kvalues_iq4nl);
                lo[r] = v.x;
                hi[r] = v.y;
                d[r]  = __half2float(bx->d);
            }
        }
        mmq_few_mma_add<type>(lo, hi, d, yf, s, sum);
    }
}

template <int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot_iq3_s(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        d[r] = __half2float(((const block_iq3_s *) x[r])->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int lo[2];
        int hi[2];
        int ls[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_iq3_s * bx = (const block_iq3_s *) x[r];
            const int qh = bx->qh[s];
            // the values [4k, 4k + 4) of a sub-block come from grid entry k, with the signs in bits 4*(k%2) of byte k/2
            const int k_lo = t;
            const int k_hi = 4 + t;
            const int grid_lo = iq3s_grid[bx->qs[8*s + k_lo] | (((qh >> k_lo) & 1) << 8)];
            const int grid_hi = iq3s_grid[bx->qs[8*s + k_hi] | (((qh >> k_hi) & 1) << 8)];
            const int sign_lo = (bx->signs[4*s + k_lo/2] >> (4*(k_lo % 2))) & 0x0F;
            const int sign_hi = (bx->signs[4*s + k_hi/2] >> (4*(k_hi % 2))) & 0x0F;
            const int mask_lo = __vcmpne4(((sign_lo & 0x03) << 7) | ((sign_lo & 0x0C) << 21), 0x00000000);
            const int mask_hi = __vcmpne4(((sign_hi & 0x03) << 7) | ((sign_hi & 0x0C) << 21), 0x00000000);
            lo[r] = __vsub4(grid_lo ^ mask_lo, mask_lo);
            hi[r] = __vsub4(grid_hi ^ mask_hi, mask_hi);
            ls[r] = 1 + 2*((bx->scales[s/2] >> (4*(s % 2))) & 0x0F);
        }
        mmq_few_mma_add<GGML_TYPE_IQ3_S>(lo, hi, ls, yf, s, sum_d);
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_vec_dot(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL) {
        mmq_few_vec_dot_q8_0_iq4_nl<type, ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        mmq_few_vec_dot_q3_K<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        mmq_few_vec_dot_q45_K<type, ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        mmq_few_vec_dot_q6_K<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        mmq_few_vec_dot_iq3_s<ncg>(x, yf, sum);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        mmq_few_vec_dot_iq4_xs<ncg>(x, yf, sum);
    } else {
        static_assert(type == GGML_TYPE_COUNT, "unsupported type");
    }
}

#endif // defined(CP_ASYNC_AVAILABLE)

// has_ids: MUL_MAT_ID, blockIdx.y is a pair of quantize_mmq_few_id_q8_1, its expert is the src0 channel and dst dims 1 and 2 are slot and token.
template <ggml_type type, int ncg, int nmt, bool has_ids>
__launch_bounds__(MMQ_FEW_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_q_few(
        const char * x, const int4 * y_ptr, float * dst,
        const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x, const int stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int stride_sample_dst, const bool x_is_constant, const int * pairs) {
#if defined(CP_ASYNC_AVAILABLE)
    constexpr int nwarps    = MMQ_FEW_NWARPS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == WARP_SIZE, "unsupported warp size");
    static_assert(!has_ids || (ncg == 1 && nmt == 1), "unsupported MUL_MAT_ID config");

    constexpr int  unit_size    = mmq_few_unit_size(type);
    constexpr int  nrows        = nmt*MMQ_FEW_NROWS;
    constexpr int  nunits_stage = mmq_few_nunits_stage(type, ncg, nmt);
    constexpr int  nstages      = mmq_few_nstages(type);
    constexpr bool unaligned    = mmq_few_unaligned(type, nunits_stage);
    constexpr int  nchunks      = mmq_few_nchunks(type, nunits_stage);
    constexpr int  sram_stride  = mmq_few_sram_stride(type, nunits_stage);
    constexpr int  nbytes_stage = nrows*sram_stride;
    constexpr int  ncopies      = nrows*nchunks;
    static_assert(unit_size > 0, "unsupported type");

    extern __shared__ char data_mmq_few[];
    char * ring = data_mmq_few + threadIdx.y*nstages*nbytes_stage;

    const bool ksplit   = mmq_few_ksplit(has_ids, nunits);
    const int  row0_blk = blockIdx.x*(ksplit ? nrows : nwarps*nrows);
    const int  row0     = ksplit ? row0_blk : row0_blk + threadIdx.y*nrows;
    const int  iu_begin = ksplit ? nunits_stage*threadIdx.y : 0;
    const int  iu_step  = ksplit ? nunits_stage*nwarps : nunits_stage;

    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int nchannels   = gridDim.y;

    int ncols = ncols_dst;
    [[maybe_unused]] const int * pair = nullptr;
    if constexpr (has_ids) {
        ggml_cuda_pdl_sync(); // the pairs come from the preceding kernel
        pair  = pairs + channel_dst*MMQ_FEW_PAIR_INTS;
        ncols = pair[1];
        if (ncols == 0) {
            return;
        }
        x += int64_t(pair[0])*stride_channel_x;
    } else {
        x   += int64_t(sample_dst/sample_ratio)*stride_sample_x + int64_t(channel_dst/channel_ratio)*stride_channel_x;
        dst += int64_t(sample_dst)*stride_sample_dst + int64_t(channel_dst)*stride_channel_dst;
    }
    const int4 * GGML_CUDA_RESTRICT y = y_ptr + (int64_t(sample_dst)*nchannels + channel_dst)*nunits*ncg*MMQ_FEW_Q8_1_INT4;

    // No bounds check for the units, the gate makes nunits a multiple of nunits_stage.
    // The extra bytes of unaligned data are inside the allocation, which ggml aligns and pads to 128 bytes.
    auto load_stage = [&](const int stage, const int iu0) {
        const unsigned int dst_stage = ggml_cuda_cvta_generic_to_shared(ring + stage*nbytes_stage);
#pragma unroll
        for (int c0 = 0; c0 < ncopies; c0 += warp_size) {
            const int c  = c0 + threadIdx.x;
            const int i  = c / nchunks;
            const int ic = c % nchunks;
            if (ncopies % warp_size == 0 || c < ncopies) {
                const int row = min(row0 + i, nrows_x - 1);
                const char * src = x + int64_t(row)*stride_row_x + int64_t(iu0)*unit_size;
                if constexpr (unaligned) {
                    const int off = (uintptr_t) src % 16;
                    if (16*ic < off + nunits_stage*unit_size) {
                        cp_async_cg_16<0>(dst_stage + i*sram_stride + 16*ic, src - off + 16*ic);
                    }
                } else {
                    cp_async_cg_16<0>(dst_stage + i*sram_stride + 16*ic, src + 16*ic);
                }
            }
        }
    };

    // offsets of the rows of this lane from their 16 byte boundary
    int off_row[2*nmt] = {0};
    if constexpr (unaligned) {
#pragma unroll
        for (int r = 0; r < 2*nmt; ++r) {
            const int row = min(row0 + int(threadIdx.x)/4 + 8*r, nrows_x - 1);
            off_row[r] = (uintptr_t) (x + int64_t(row)*stride_row_x) % 16;
        }
    }

    // Only weights, which no preceding kernel writes, are read before the PDL wait.
    if (!x_is_constant) {
        ggml_cuda_pdl_sync();
    }

#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10
    // DGX Spark DRAM serves the short row pieces of the stage copies slowly, so the rows of the block are first prefetched into L2 in address order.
    {
        const int     tid       = threadIdx.y*warp_size + threadIdx.x;
        const int     nrows_blk = min(ksplit ? nrows : nwarps*nrows, nrows_x - row0_blk);
        const int64_t row_bytes = int64_t(nunits)*unit_size;
        const bool    contig    = stride_row_x == row_bytes;
        for (int ir = 0; ir < (contig ? 1 : nrows_blk); ++ir) {
            const uintptr_t p0 = (uintptr_t) (x + int64_t(row0_blk + ir)*stride_row_x);
            const uintptr_t p1 = p0 + (contig ? nrows_blk*row_bytes : row_bytes);
            for (uintptr_t p = (p0 & ~uintptr_t(127)) + 128*tid; p < p1; p += 128*nwarps*warp_size) {
                ggml_cuda_prefetch_l2((const void *) p);
            }
        }
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_GB10

#pragma unroll
    for (int stage = 0; stage < nstages - 1; ++stage) {
        const int iu0 = iu_begin + stage*iu_step;
        if (iu0 < nunits) {
            load_stage(stage, iu0);
        }
        cp_async_commit_group();
    }
    if (x_is_constant) {
        ggml_cuda_pdl_sync();
    }

    auto load_y = [&](mmq_few_y_frag (&yf)[ncg], const int iu) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
            mmq_few_load_y(yf[c], y + (int64_t(iu)*ncg + c)*MMQ_FEW_Q8_1_INT4);
        }
    };

    // The next src1 fragments are loaded right after the MMAs that use the current ones, an earlier load is slower.
    mmq_few_y_frag yf[nunits_stage][ncg];
    if (iu_begin < nunits) {
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
            load_y(yf[k], iu_begin + k);
        }
    }

    float sum[nmt][ncg][4] = {{{0.0f}}};

    for (int iter = 0, iu0 = iu_begin; iu0 < nunits; ++iter, iu0 += iu_step) {
        const int iu_load = iu0 + (nstages - 1)*iu_step;
        if (iu_load < nunits) {
            load_stage((iter + nstages - 1) % nstages, iu_load);
        }
        cp_async_commit_group();

        const bool has_next = iu0 + iu_step < nunits;

        cp_async_wait_group<nstages - 1>();
        ggml_cuda_syncwarp();

        const char * xs = ring + (iter % nstages)*nbytes_stage;
        const int off_stage = unaligned ? (iu0*unit_size) % 16 : 0;
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
#pragma unroll
            for (int m = 0; m < nmt; ++m) {
                const char * const xr[2] = {
                    xs + (MMQ_FEW_NROWS*m + threadIdx.x/4 + 0)*sram_stride + (off_row[2*m + 0] + off_stage) % 16 + k*unit_size,
                    xs + (MMQ_FEW_NROWS*m + threadIdx.x/4 + 8)*sram_stride + (off_row[2*m + 1] + off_stage) % 16 + k*unit_size,
                };
                mmq_few_vec_dot<type, ncg>(xr, yf[k], sum[m]);
            }
            if (has_next) {
                load_y(yf[k], iu0 + iu_step + k);
            }
        }

        ggml_cuda_syncwarp(); // the stage is overwritten in the next iteration
    }
    cp_async_wait_all();

    ggml_cuda_pdl_lc();

    // reduce the sums over the warps as one flat array
    if (ksplit) {
        constexpr int nsum = nmt*ncg*4;
        float * sum_flat = &sum[0][0][0];

        float * buf_sum = (float *) data_mmq_few;
        __syncthreads();
        if (threadIdx.y > 0) {
#pragma unroll
            for (int i = 0; i < nsum; ++i) {
                buf_sum[((threadIdx.y - 1)*nsum + i)*warp_size + threadIdx.x] = sum_flat[i];
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
#pragma unroll
        for (int w = 0; w < nwarps - 1; ++w) {
#pragma unroll
            for (int i = 0; i < nsum; ++i) {
                sum_flat[i] += buf_sum[(w*nsum + i)*warp_size + threadIdx.x];
            }
        }
    }

#pragma unroll
    for (int m = 0; m < nmt; ++m) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + MMQ_FEW_NROWS*m + tile<16, 8, int>::get_i(l);
                const int j = c*MMQ_FEW_NCOLS + tile<16, 8, int>::get_j(l);
                if (i < nrows_x && j < ncols) {
                    if constexpr (has_ids) {
                        const int e = pair[2 + j]; // token*n_expert_used + slot
                        dst[int64_t(e / ncols_dst)*stride_channel_dst + int64_t(e % ncols_dst)*stride_col_dst + i] = sum[m][c][l];
                    } else {
                        dst[int64_t(j)*stride_col_dst + i] = sum[m][c][l];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y_ptr, dst, nunits, nrows_x, ncols_dst, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_dst, x_is_constant, pairs);
    NO_DEVICE_CODE;
#endif // defined(CP_ASYNC_AVAILABLE)
}

// ggml_is_constant (static in ggml.c) on the tensor that a view points to
static bool mmq_few_is_constant(const ggml_tensor * src0) {
    const ggml_tensor * x_root = src0->view_src ? src0->view_src : src0;
    return x_root->buffer && ggml_backend_buffer_get_usage(x_root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        !(x_root->flags & GGML_TENSOR_FLAG_PARAM);
}

// MUL_MAT_ID: grid y is the pairs, the kernel takes the expert from the pair and does not use the channel and sample ratios.
template <ggml_type type, int ncg, int nmt, bool has_ids = false>
static void launch_mul_mat_q_few(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream, const int * pairs = nullptr, const int npairs_max = 0) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const int nunits = ne00 / QK_K;
    GGML_ASSERT(nunits % mmq_few_nunits_stage(type, ncg, nmt) == 0);

    const int warp_size     = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nbytes_shared = mmq_few_nbytes_shared(type, ncg, nmt);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_few<type, ncg, nmt, has_ids>), nbytes_shared);

    const bool x_is_constant = mmq_few_is_constant(src0);

    const int  nrows_blk = (mmq_few_ksplit(has_ids, nunits) ? 1 : MMQ_FEW_NWARPS)*nmt*MMQ_FEW_NROWS;
    const dim3 block_nums((ne01 + nrows_blk - 1) / nrows_blk, has_ids ? npairs_max : ne12, ne13);
    const dim3 block_dims(warp_size, MMQ_FEW_NWARPS, 1);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_q_few<type, ncg, nmt, has_ids>, params,
        (const char *) src0->data, src1_q, (float *) dst->data,
        nunits, (int) ne01, (int) ne1, (int64_t) nb01, (int) (nb1/sizeof(float)),
        (int) (ne12/ne02), (int64_t) nb02, (int) (nb2/sizeof(float)),
        (int) (ne13/ne03), (int64_t) nb03, (int) (nb3/sizeof(float)), x_is_constant, pairs);
}

static int mmq_few_ncg(const int64_t ncols) {
    return (ncols + MMQ_FEW_NCOLS - 1) / MMQ_FEW_NCOLS;
}

// With 2 column groups, 32-row tiles halve the src1 loads per src0 byte. With 3 column groups they spill registers.
static int mmq_few_nmt(const ggml_type type, const int ncg, const int64_t nrows_x) {
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    return mmq_few_tall_supported(type) && ncg == 2 && nrows_x >= int64_t(nsm)*2*MMQ_FEW_NROWS ? 2 : 1;
}

template <ggml_type type>
static void launch_mul_mat_q_few_tall(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream) {
    if constexpr (mmq_few_tall_supported(type)) {
        if (mmq_few_nmt(type, 2, src0->ne[1]) == 2) {
            launch_mul_mat_q_few<type, 2, 2>(src0, src1, dst, src1_q, stream);
            return;
        }
    }
    launch_mul_mat_q_few<type, 2, 1>(src0, src1, dst, src1_q, stream);
}

template <ggml_type type>
static void launch_mul_mat_q_few_switch_ncg(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, const int * pairs, const int npairs_max, const int ncg, cudaStream_t stream) {
    if (pairs) {
        launch_mul_mat_q_few<type, 1, 1, true>(src0, src1, dst, src1_q, stream, pairs, npairs_max);
        return;
    }
    switch (ncg) {
        case 1:
            launch_mul_mat_q_few<type, 1, 1>(src0, src1, dst, src1_q, stream);
            break;
        case 2:
            launch_mul_mat_q_few_tall<type>(src0, src1, dst, src1_q, stream);
            break;
        case 3:
            launch_mul_mat_q_few<type, 3, 1>(src0, src1, dst, src1_q, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(ne00 % QK_K == 0);

    cudaStream_t stream = ctx.stream();

    const int nunits = ne00 / QK_K;
    const int ncg    = ids ? 1 : mmq_few_ncg(ne11);

    ggml_cuda_pool_alloc<int>  pairs(ctx.pool());
    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool());
    int npairs_max = 0;
    if (ids) {
        GGML_ASSERT(ids->type == GGML_TYPE_I32);
        GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));
        GGML_ASSERT(ne1 == ids->ne[0] && ne13 == 1 && ne3 == 1);

        const int64_t n_expert_used = ids->ne[0];
        const int64_t n_tokens      = ids->ne[1];
        const int64_t nentries      = n_tokens*n_expert_used;
        const int64_t nused         = std::min(ne02, nentries);
        // an expert with c > 0 entries (at most one per token) has ceil(c/MMQ_FEW_NCOLS) pairs
        npairs_max = (int) std::min(nused + (nentries - nused)/MMQ_FEW_NCOLS, ne02*((n_tokens + MMQ_FEW_NCOLS - 1) / MMQ_FEW_NCOLS));

        pairs.alloc(int64_t(npairs_max)*MMQ_FEW_PAIR_INTS);
        src1_q.alloc(int64_t(npairs_max)*nunits*MMQ_FEW_Q8_1_INT4);
        quantize_mmq_few_id_q8_1_cuda((const float *) src1->data, (const int32_t *) ids->data, src1_q.get(), pairs.get(), mmq_few_ds_layout(src0->type),
            ne00, nb11/sizeof(float), nb12/sizeof(float), ne11, ne02, n_expert_used, n_tokens, ids->nb[1]/sizeof(int32_t), npairs_max, stream);
    } else {
        GGML_ASSERT(ne11 <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS);
        GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);

        src1_q.alloc(int64_t(ne12)*ne13*nunits*ncg*MMQ_FEW_Q8_1_INT4);
        quantize_mmq_few_q8_1_cuda((const float *) src1->data, src1_q.get(), mmq_few_ds_layout(src0->type), ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
            ne11, ne12, ne13, stream);
    }

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q8_0>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q3_K>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q4_K>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q5_K>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q6_K>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ3_S>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ4_NL>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ4_XS>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        default:
            GGML_ABORT("unsupported type for few-row MMQ");
    }
}

// Min. number of src1 columns (MUL_MAT_ID: tokens) for the few-row MMQ, 0 if it is not used for this type, GPU and row length.
static int ggml_cuda_mmq_few_min_ncols(const ggml_type type, const int cc, const int64_t ne00, const bool has_ids) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return 0;
#endif // GGML_CUDA_FORCE_CUBLAS

    // NVIDIA Ampere to Blackwell
    if (!ampere_mma_available(cc) || cc >= GGML_CUDA_CC_RUBIN) {
        return 0;
    }
    // on DGX Spark the kernel needs its L2 prefetch, which is only in the DGX Spark device code
    if (cc == GGML_CUDA_CC_GB10 && ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_GB10) {
        return 0;
    }
    // with fewer than 8 units of 256 values per row the warps idle, MMVQ and MMQ are better there (MUL_MAT_ID splits the rows instead)
    if (mmq_few_unit_size(type) == 0 || (!has_ids && ne00 < 8*QK_K)) {
        return 0;
    }
    if (cc == GGML_CUDA_CC_GB10) { // tuned on DGX Spark GB10, also for MUL_MAT_ID
        return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL ? 8 : 2;
    }
    if (has_ids) { // after MMVQ, which is faster up to its max. batch size
        return get_mmvq_mmid_max_batch(type, cc) + 1;
    }
    switch (type) { // all GPUs except DGX Spark, tuned on RTX 5090
        case GGML_TYPE_Q3_K:   return 3;
        case GGML_TYPE_Q5_K:   return 3;
        case GGML_TYPE_Q4_K:   return 4;
        case GGML_TYPE_IQ4_NL: return 4;
        case GGML_TYPE_Q6_K:   return 4;
        case GGML_TYPE_Q8_0:   return 5;
        case GGML_TYPE_IQ3_S:  return 5;
        case GGML_TYPE_IQ4_XS: return 6;
        default:               return 0;
    }
}

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const int cc) {
    const int64_t ncols     = ids ? ids->ne[1] : src1->ne[1];
    const int     min_ncols = ggml_cuda_mmq_few_min_ncols(src0->type, cc, src0->ne[0], ids != nullptr);
    if (min_ncols == 0 || ncols < min_ncols || ncols > MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS) {
        return false;
    }
    // MUL_MAT_ID: the pairs are built in shared memory
    if (ids && (2*src0->ne[2] + 1)*sizeof(int) > ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        return false;
    }
    // the row must hold a whole number of ring stages of the launched kernel
    const int ncg          = ids ? 1 : mmq_few_ncg(ncols);
    const int nmt          = ids ? 1 : mmq_few_nmt(src0->type, ncg, src0->ne[1]);
    const int nunits_stage = mmq_few_nunits_stage(src0->type, ncg, nmt);
    if (src0->ne[0] % (nunits_stage*QK_K) != 0) {
        return false;
    }
    // unaligned data needs the alignment of the decoder loads
    const int x_alignment = mmq_few_unaligned(src0->type, nunits_stage) ? (ggml_type_size(src0->type) % 4 == 0 ? 4 : 2) : 16;
    return ggml_cuda_is_aligned(src0, x_alignment) && ggml_cuda_is_aligned(src1, 16);
}

#elif defined(GGML_USE_HIP)

// Few-row MMQ for HIP: one CUDA block covers src0 rows, the lanes of its single warp split K.
// Each lane decodes the weights of a K block once and reuses them for the 4 src1 columns of
// the tile. The vec_dot functions below are the MMVQ decoders of vecdotq.cuh with the weight
// decode hoisted out of the column loop.

constexpr int mmq_few_vdr(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:   return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:   return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:   return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:   return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL: return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS: return VDR_IQ4_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:  return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:   return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:   return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:   return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:   return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:  return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:  return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:   return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:   return VDR_Q8_0_Q8_1_MMVQ;
        default:               return 0;
    }
}

// values per weight block, QK_K for the K-quants and the superblock formats
constexpr int mmq_few_block_span(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return QK8_0;
        case GGML_TYPE_IQ4_NL: return QK4_NL;
        case GGML_TYPE_Q4_0:   return QK4_0;
        case GGML_TYPE_Q4_1:   return QK4_1;
        case GGML_TYPE_Q5_0:   return QK5_0;
        case GGML_TYPE_Q5_1:   return QK5_1;
        case GGML_TYPE_MXFP4:  return QK_MXFP4;
        case GGML_TYPE_NVFP4:  return QK_NVFP4;
        default:               return QK_K;
    }
}

static __device__ __forceinline__ float mmq_few_vec_dot_q4_K(
        const int (&v0)[QR4_K], const int (&v1)[QR4_K], const uint8_t * sc, const uint8_t * m,
        const float2 dm, const block_q8_1 * y, const int bq8_offset, const int iqs8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * yb = y + bq8_offset + i;
        const float d8 = __low2float(yb->ds);
        const int * q8 = (const int *) yb->qs + iqs8;
        const int u0 = q8[0];
        const int u1 = q8[4];
        const int dot1 = ggml_cuda_dp4a(v1[i], u1, ggml_cuda_dp4a(v0[i], u0, 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, 0));
        sumf_d += d8 * (dot1 * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);
    }
    return dm.x*sumf_d - dm.y*sumf_m;
}

static __device__ __forceinline__ float mmq_few_vec_dot_q6_K(
        const int (&vl)[VDR_Q6_K_Q8_1_MMVQ], const int (&vh)[VDR_Q6_K_Q8_1_MMVQ],
        const int8_t * scales, const float d, const block_q8_1 * y, const int bq8_offset, const int iqs8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];
        int sumi = 0;
        const float d8 = __low2float(y[bq8_offset + 2*i].ds);
#pragma unroll
        for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
            const int u = get_int_b4(y[bq8_offset + 2*i].qs, iqs8 + j);
            const int vil = (vl[j] >> (4*i)) & 0x0F0F0F0F;
            const int vih = ((vh[j] >> (4*i)) << 4) & 0x30303030;
            sumi = ggml_cuda_dp4a(vil | vih, u, sumi);
            sumi = ggml_cuda_dp4a(0xE0E0E0E0, u, sumi);
        }
        sumf += d8 * (sumi * sc);
    }
    return d*sumf;
}

static __device__ __forceinline__ float mmq_few_vec_dot_q8_0(
        const int (&v)[VDR_Q8_0_Q8_1_MMVQ], const float d, const block_q8_1 * y, const int iqs8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        sumi = ggml_cuda_dp4a(v[i], get_int_b4(y->qs, iqs8 + i), sumi);
    }
    return d * __low2float(y->ds) * sumi;
}

static __device__ __forceinline__ float mmq_few_vec_dot_q3_K(
        const int (&vl)[VDR_Q3_K_Q8_1_MMVQ], const int (&vh)[VDR_Q3_K_Q8_1_MMVQ], const int (&scales)[3],
        const float d, const block_q8_1 * y, const int bq8_offset, const int iqs8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int sc_low  = (scales[i/2] >> (16*(i%2))) & 0x0F;
        const int sc_high = (scales[2] >> (16*(i%2) + 2*(i/2))) & 0x03;
        const int sc = (sc_low | (sc_high << 4)) - 32;
        int sumi = 0;
        const float d8 = __low2float(y[bq8_offset + i].ds);
#pragma unroll
        for (int j = 0; j < VDR_Q3_K_Q8_1_MMVQ; ++j) {
            const int u = get_int_b4(y[bq8_offset + i].qs, iqs8 + j);
            const int vil = (vl[j] >> (2*i)) & 0x03030303;
            const int vih = ((vh[j] >> i) << 2) & 0x04040404;
            sumi = ggml_cuda_dp4a(vil | vih, u, sumi);
            sumi = ggml_cuda_dp4a(0xFCFCFCFC, u, sumi);
        }
        sumf += d8 * (sumi * sc);
    }
    return d*sumf;
}

static __device__ __forceinline__ float mmq_few_vec_dot_q5_K(
        const int (&vl)[2], const int (&vh)[2], const uint8_t * sc, const uint8_t * m, const float2 dm,
        const block_q8_1 * y, const int bq8_offset, const int iqs8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const float d8 = __low2float(y[bq8_offset + i].ds);
        const int * q8 = (const int *) y[bq8_offset + i].qs + iqs8;
        const int u0 = q8[0];
        const int u1 = q8[4];
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int dot1 = ggml_cuda_dp4a(vl0i | vh0i, u0, ggml_cuda_dp4a(vl1i | vh1i, u1, 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u0, ggml_cuda_dp4a(0x01010101, u1, 0));
        sumf_d += d8 * (dot1 * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);
    }
    return dm.x*sumf_d - dm.y*sumf_m;
}

// shared by the 4-bit lookup formats, v0/v1 hold the table-decoded weight words
template <int n>
static __device__ __forceinline__ float mmq_few_vec_dot_table(
        const int (&v0)[n], const int (&v1)[n], const int ls, const float d,
        const block_q8_1 * y, const int iqs8) {
    const int * q8 = (const int *) y->qs + iqs8;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < n; ++l) {
        sumi = ggml_cuda_dp4a(v0[l], q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v1[l], q8[l + 4], sumi);
    }
    return d * __low2float(y->ds) * (sumi * ls);
}

static __device__ __forceinline__ float mmq_few_vec_dot_iq3_s(
        const int (&grid_l)[4], const int (&grid_h)[4], const int ls, const float d, const block_q8_1 * y) {
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int u0 = get_int_b4(y->qs, l0 + 0);
        const int u1 = get_int_b4(y->qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l[l0/2], u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h[l0/2], u1, sumi);
    }
    return d * __low2float(y->ds) * (sumi * ls);
}

static __device__ __forceinline__ float mmq_few_vec_dot_q4_0(
        const int (&v)[VDR_Q4_0_Q8_1_MMVQ], const float d, const block_q8_1 * y, const int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        const int u0 = get_int_b4(y->qs, iqs + i);
        const int u1 = get_int_b4(y->qs, iqs + i + QI4_0);
        sumi = ggml_cuda_dp4a((v[i] >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v[i] >> 4) & 0x0F0F0F0F, u1, sumi);
    }
    const float2 ds8f = __half22float2(y->ds);
    return d * (sumi * ds8f.x - (8*VDR_Q4_0_Q8_1_MMVQ/QI4_0) * ds8f.y);
}

static __device__ __forceinline__ float mmq_few_vec_dot_q4_1(
        const int (&v)[VDR_Q4_1_Q8_1_MMVQ], const float2 dm, const block_q8_1 * y, const int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        const int u0 = get_int_b4(y->qs, iqs + i);
        const int u1 = get_int_b4(y->qs, iqs + i + QI4_1);
        sumi = ggml_cuda_dp4a((v[i] >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v[i] >> 4) & 0x0F0F0F0F, u1, sumi);
    }
    const float2 dm4f = dm;
    const float2 ds8f = __half22float2(y->ds);
    return sumi * (dm4f.x * ds8f.x) + (dm4f.y * ds8f.y) / (QI8_1 / (VDR_Q4_1_Q8_1_MMVQ * QR4_1));
}

static __device__ __forceinline__ float mmq_few_vec_dot_q5_0(
        const int (&vl)[VDR_Q5_0_Q8_1_MMVQ], const int (&vh)[VDR_Q5_0_Q8_1_MMVQ],
        const float d, const block_q8_1 * y, const int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        const int u0 = get_int_b4(y->qs, iqs + i);
        const int u1 = get_int_b4(y->qs, iqs + i + QI5_0);
        int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
        vi0 |= (vh[i] <<  4) & 0x00000010;
        vi0 |= (vh[i] << 11) & 0x00001000;
        vi0 |= (vh[i] << 18) & 0x00100000;
        vi0 |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
        vi1 |= (vh[i] >> 12) & 0x00000010;
        vi1 |= (vh[i] >>  5) & 0x00001000;
        vi1 |= (vh[i] <<  2) & 0x00100000;
        vi1 |= (vh[i] <<  9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float2 ds8f = __half22float2(y->ds);
    return d * (sumi * ds8f.x - (16*VDR_Q5_0_Q8_1_MMVQ/QI5_0) * ds8f.y);
}

static __device__ __forceinline__ float mmq_few_vec_dot_q5_1(
        const int (&vl)[VDR_Q5_1_Q8_1_MMVQ], const int (&vh)[VDR_Q5_1_Q8_1_MMVQ],
        const float2 dm, const block_q8_1 * y, const int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        const int u0 = get_int_b4(y->qs, iqs + i);
        const int u1 = get_int_b4(y->qs, iqs + i + QI5_1);
        int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
        vi0 |= (vh[i] <<  4) & 0x00000010;
        vi0 |= (vh[i] << 11) & 0x00001000;
        vi0 |= (vh[i] << 18) & 0x00100000;
        vi0 |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
        vi1 |= (vh[i] >> 12) & 0x00000010;
        vi1 |= (vh[i] >>  5) & 0x00001000;
        vi1 |= (vh[i] <<  2) & 0x00100000;
        vi1 |= (vh[i] <<  9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float2 dm5f = dm;
    const float2 ds8f = __half22float2(y->ds);
    return sumi * (dm5f.x * ds8f.x) + (dm5f.y * ds8f.y) / (QI8_1 / (VDR_Q5_1_Q8_1_MMVQ * QR5_1));
}

static __device__ __forceinline__ float mmq_few_vec_dot_q2_K(
        const int (&v)[VDR_Q2_K_Q8_1_MMVQ], const int (&scales)[2], const float2 dm,
        const block_q8_1 * y, const int bq8_offset, const int iqs8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[i/2] >> (16*(i%2));
        int sumi_d = 0;
        int sumi_m = 0;
        const float d8 = __low2float(y[bq8_offset + i].ds);
#pragma unroll
        for (int j = 0; j < VDR_Q2_K_Q8_1_MMVQ; ++j) {
            const int u = get_int_b4(y[bq8_offset + i].qs, iqs8 + j);
            const int vi = (v[j] >> (2*i)) & 0x03030303;
            sumi_d = ggml_cuda_dp4a(vi, u, sumi_d);
            sumi_m = ggml_cuda_dp4a(0x01010101, u, sumi_m);
        }
        sumf_d += d8 * (sumi_d * (sc & 0xF));
        sumf_m += d8 * (sumi_m * ((sc >> 4) & 0xF));
    }
    return dm.x*sumf_d - dm.y*sumf_m;
}

static __device__ __forceinline__ float mmq_few_vec_dot_nvfp4(
        const int2 (&v0)[VDR_NVFP4_Q8_1_MMVQ/2], const int2 (&v1)[VDR_NVFP4_Q8_1_MMVQ/2],
        const int (&sc)[VDR_NVFP4_Q8_1_MMVQ/2], const block_q8_1 * y, const int iqs) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; ++i) {
        const int is = (iqs + 2*i) >> 1;
        const block_q8_1 * yb = y + (is >> 1);
        const int i8 = (is & 1) << 2;
        int sumi = ggml_cuda_dp4a(v0[i].x, get_int_b4(yb->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(v0[i].y, get_int_b4(yb->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(v1[i].x, get_int_b4(yb->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(v1[i].y, get_int_b4(yb->qs, i8 + 3), sumi);
        const float d = ggml_cuda_ue4m3_to_fp32(sc[i]) * __low2float(yb->ds);
        sum += d * float(sumi);
    }
    return sum;
}

template <ggml_type type>
static __global__ void mul_mat_q_few(
        const char * x, const block_q8_1 * y, float * dst,
        const int64_t nblocks, const int64_t stride_y, const int64_t stride_dst, const int ncols) {
    constexpr int ncols_tile = 4;
    constexpr int vdr = mmq_few_vdr(type);
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int64_t row = blockIdx.x;
    const int col = ncols_tile*blockIdx.y;
    const int tid = threadIdx.x;
    const int iqs = vdr*(tid % (qi/vdr));
    float sum[ncols_tile] = {0.0f};

    if constexpr (type == GGML_TYPE_Q4_K) {
        const int bq8_offset = QR4_K*((iqs/2)/(QI8_1/2));
        const int iqs8 = (iqs/2)%4;
        const int scale_idx = (bq8_offset/2) & 1;
        const uint32_t hi = (uint32_t) -(int32_t) (bq8_offset/2 >= 2);
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q4_K * bx = (const block_q4_K *) (x + row*nblocks*sizeof(block_q4_K)) + kbx;
            const int * q4 = (const int *) (bx->qs + 16*bq8_offset + 4*iqs8);
            const int q0 = q4[0];
            const int q1 = q4[4];
            int v0[QR4_K];
            int v1[QR4_K];
#pragma unroll
            for (int i = 0; i < QR4_K; ++i) {
                v0[i] = (q0 >> (4*i)) & 0x0F0F0F0F;
                v1[i] = (q1 >> (4*i)) & 0x0F0F0F0F;
            }
            const uint16_t * scales = (const uint16_t *) bx->scales;
            const uint32_t s0 = scales[scale_idx + 0];
            const uint32_t s2 = scales[scale_idx + 2];
            const uint32_t s4 = scales[scale_idx + 4];
            uint16_t aux[2];
            aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | (((s4 & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
            aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
            const uint8_t * sc = (const uint8_t *) aux;
            const uint8_t * m = sc + 2;
            const float2 dm = __half22float2(bx->dm);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q4_K(v0, v1, sc, m, dm,
                        y + (col + j)*stride_y + kbx*(QK_K/QK8_1), bq8_offset, iqs8);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const int bq8_offset = 2*QR6_K*(iqs/(QI6_K/2)) + (iqs % (QI6_K/2))/(QI6_K/4);
        const int scale_offset = (QI6_K/4)*(iqs/(QI6_K/2)) + (iqs % (QI6_K/2))/(QI6_K/8);
        const int vh_shift = 2*((iqs % (QI6_K/2))/(QI6_K/4));
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q6_K * bx = (const block_q6_K *) (x + row*nblocks*sizeof(block_q6_K)) + kbx;
            int vl[VDR_Q6_K_Q8_1_MMVQ];
            int vh[VDR_Q6_K_Q8_1_MMVQ];
#pragma unroll
            for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
                vl[j] = get_int_b2(bx->ql, iqs + j);
                vh[j] = get_int_b2(bx->qh, (QI6_K/4)*(iqs/(QI6_K/2)) + iqs % (QI6_K/4) + j) >> vh_shift;
            }
            const int8_t * scales = bx->scales + scale_offset;
            const float d = __half2float(bx->d);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q6_K(vl, vh, scales, d,
                        y + (col + j)*stride_y + kbx*(QK_K/QK8_1), bq8_offset, iqs % QI8_1);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        // Q8_0 blocks cover QK8_0 values, not QK_K like the K-quants
        const int iqs8 = iqs % QI8_1;
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q8_0 * bx = (const block_q8_0 *) (x + row*nblocks*sizeof(block_q8_0)) + kbx;
            int v[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
            for (int j = 0; j < VDR_Q8_0_Q8_1_MMVQ; ++j) {
                v[j] = get_int_b2(bx->qs, iqs8 + j);
            }
            const float d = __half2float(bx->d);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q8_0(v, d,
                        y + (col + j)*stride_y + kbx*(QK8_0/QK8_1), iqs8);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        const int bq8_offset = QR3_K*(iqs/(QI3_K/2));
        const int iqs8 = iqs % QI8_1;
        const int sc_shift = 8*(iqs8/(QI8_1/2)) + 4*(iqs/(QI3_K/2));
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q3_K * bx = (const block_q3_K *) (x + row*nblocks*sizeof(block_q3_K)) + kbx;
            int vl[VDR_Q3_K_Q8_1_MMVQ];
            int vh[VDR_Q3_K_Q8_1_MMVQ];
#pragma unroll
            for (int j = 0; j < VDR_Q3_K_Q8_1_MMVQ; ++j) {
                vl[j] = get_int_b2(bx->qs, iqs + j);
                vh[j] = get_int_b2(bx->hmask, iqs8 + j) >> bq8_offset;
            }
            int scales[3];
#pragma unroll
            for (int l = 0; l < 3; ++l) {
                scales[l] = get_int_b2(bx->scales, l) >> sc_shift;
            }
            const float d = bx->d;
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q3_K(vl, vh, scales, d,
                        y + (col + j)*stride_y + kbx*(QK_K/QK8_1), bq8_offset, iqs8);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        const int bq8_offset = QR5_K*((iqs/2)/(QI8_1/2));
        const int iqs8 = (iqs/2)%4;
        const int scale_idx = (bq8_offset/2) & 1;
        const uint32_t hi = (uint32_t) -(int32_t) (bq8_offset/2 >= 2);
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q5_K * bx = (const block_q5_K *) (x + row*nblocks*sizeof(block_q5_K)) + kbx;
            const int * ql = (const int *) (bx->qs + 16*bq8_offset + 4*iqs8);
            const int * qh = (const int *) (bx->qh + 4*iqs8);
            int vl[2];
            int vh[2];
            vl[0] = ql[0];
            vl[1] = ql[4];
            vh[0] = qh[0] >> bq8_offset;
            vh[1] = qh[4] >> bq8_offset;
            const uint16_t * scales = (const uint16_t *) bx->scales;
            const uint32_t s0 = scales[scale_idx + 0];
            const uint32_t s2 = scales[scale_idx + 2];
            const uint32_t s4 = scales[scale_idx + 4];
            uint16_t aux[2];
            aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | (((s4 & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
            aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
            const uint8_t * sc = (const uint8_t *) aux;
            const uint8_t * m = sc + 2;
            const float2 dm = __half22float2(bx->dm);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q5_K(vl, vh, sc, m, dm,
                        y + (col + j)*stride_y + kbx*(QK_K/QK8_1), bq8_offset, iqs8);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        // IQ4_NL blocks cover QK4_NL values, like Q8_0 they are not QK_K superblocks
        const int iqs8 = iqs % QI8_1;
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_iq4_nl * bx = (const block_iq4_nl *) (x + row*nblocks*sizeof(block_iq4_nl)) + kbx;
            int v0[VDR_IQ4_NL_Q8_1_MMVQ];
            int v1[VDR_IQ4_NL_Q8_1_MMVQ];
#pragma unroll
            for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMVQ; ++l) {
                const int2 v = get_int_from_table_16(get_int_b2(bx->qs, iqs8 + l), kvalues_iq4nl);
                v0[l] = v.x;
                v1[l] = v.y;
            }
            const float d = __half2float(bx->d);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_table(v0, v1, 1, d,
                        y + (col + j)*stride_y + kbx*(QK4_NL/QK8_1), iqs8);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        // IQ4_XS blocks are QK_K superblocks like the K-quants
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_iq4_xs * bx = (const block_iq4_xs *) (x + row*nblocks*sizeof(block_iq4_xs)) + kbx;
            int v0[4];
            int v1[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int2 v = get_int_from_table_16(get_int_b4(bx->qs, iqs + j), kvalues_iq4nl);
                v0[j] = v.x;
                v1[j] = v.y;
            }
            const int ls = ((bx->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bx->scales_h >> (iqs/2)) & 0x03) << 4);
            const float d = __half2float(bx->d);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_table(v0, v1, ls - 32, d,
                        y + (col + j)*stride_y + kbx*(QK_K/QK8_1) + iqs/4, 0);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        // IQ3_S blocks are QK_K superblocks, the grid and signs are decoded once per kbx
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_iq3_s * bx = (const block_iq3_s *) (x + row*nblocks*sizeof(block_iq3_s)) + kbx;
            const int2 qs_packed = make_int2(get_int_b2(bx->qs, iqs + 0), get_int_b2(bx->qs, iqs + 1));
            const uint8_t * qs = (const uint8_t *) &qs_packed;
            const int qh = bx->qh[iqs/2];
            const int signs_packed_32 = get_int_b2(bx->signs, iqs/2);
            const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
            int grid_l[4];
            int grid_h[4];
#pragma unroll
            for (int l0 = 0; l0 < 8; l0 += 2) {
                const int2 grid_pos = make_int2(
                    iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                    iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
                const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
                const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
                grid_l[l0/2] = __vsub4(grid_pos.x ^ signs0, signs0);
                grid_h[l0/2] = __vsub4(grid_pos.y ^ signs1, signs1);
            }
            const int ls = 1 + 2*((bx->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
            const float d = __half2float(bx->d);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_iq3_s(grid_l, grid_h, ls, d,
                        y + (col + j)*stride_y + kbx*(QK_K/QK8_1) + iqs/2);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        // Q4_0 blocks cover QK4_0 values, not QK_K superblocks
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q4_0 * bx = (const block_q4_0 *) (x + row*nblocks*sizeof(block_q4_0)) + kbx;
            int v[VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
            for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
                v[i] = get_int_b2(bx->qs, iqs + i);
            }
            const float d = __half2float(bx->d);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q4_0(v, d,
                        y + (col + j)*stride_y + kbx*(QK4_0/QK8_1), iqs);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q4_1) {
        // Q4_1 blocks cover QK4_1 values, not QK_K superblocks
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q4_1 * bx = (const block_q4_1 *) (x + row*nblocks*sizeof(block_q4_1)) + kbx;
            int v[VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
            for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
                v[i] = get_int_b4(bx->qs, iqs + i);
            }
            const float2 dm = __half22float2(bx->dm);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q4_1(v, dm,
                        y + (col + j)*stride_y + kbx*(QK4_1/QK8_1), iqs);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        // Q5_0 blocks cover QK5_0 values, not QK_K superblocks
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q5_0 * bx = (const block_q5_0 *) (x + row*nblocks*sizeof(block_q5_0)) + kbx;
            int vl[VDR_Q5_0_Q8_1_MMVQ];
            int vh[VDR_Q5_0_Q8_1_MMVQ];
#pragma unroll
            for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
                vl[i] = get_int_b2(bx->qs, iqs + i);
                vh[i] = get_int_b2(bx->qh, 0) >> (4 * (iqs + i));
            }
            const float d = __half2float(bx->d);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q5_0(vl, vh, d,
                        y + (col + j)*stride_y + kbx*(QK5_0/QK8_1), iqs);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q5_1) {
        // Q5_1 blocks cover QK5_1 values, not QK_K superblocks
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q5_1 * bx = (const block_q5_1 *) (x + row*nblocks*sizeof(block_q5_1)) + kbx;
            int vl[VDR_Q5_1_Q8_1_MMVQ];
            int vh[VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
            for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
                vl[i] = get_int_b4(bx->qs, iqs + i);
                vh[i] = get_int_b4(bx->qh, 0) >> (4 * (iqs + i));
            }
            const float2 dm = __half22float2(bx->dm);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q5_1(vl, vh, dm,
                        y + (col + j)*stride_y + kbx*(QK5_1/QK8_1), iqs);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        // Q2_K blocks are QK_K superblocks like the other K-quants
        const int bq8_offset = QR2_K*(iqs/QI8_1);
        const int iqs8 = iqs % QI8_1;
        const int sc_shift = 8*(iqs8/(QI8_1/2));
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_q2_K * bx = (const block_q2_K *) (x + row*nblocks*sizeof(block_q2_K)) + kbx;
            int v[VDR_Q2_K_Q8_1_MMVQ];
#pragma unroll
            for (int j = 0; j < VDR_Q2_K_Q8_1_MMVQ; ++j) {
                v[j] = get_int_b4(bx->qs, iqs + j);
            }
            int scales[2];
            scales[0] = get_int_b4(bx->scales, 2*(iqs/QI8_1) + 0) >> sc_shift;
            scales[1] = get_int_b4(bx->scales, 2*(iqs/QI8_1) + 1) >> sc_shift;
            const float2 dm = __half22float2(bx->dm);
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_q2_K(v, scales, dm,
                        y + (col + j)*stride_y + kbx*(QK_K/QK8_1), bq8_offset, iqs8);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_MXFP4) {
        // MXFP4 blocks cover QK_MXFP4 values, not QK_K superblocks
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_mxfp4 * bx = (const block_mxfp4 *) (x + row*nblocks*sizeof(block_mxfp4)) + kbx;
            int v0[VDR_MXFP4_Q8_1_MMVQ];
            int v1[VDR_MXFP4_Q8_1_MMVQ];
#pragma unroll
            for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
                const int2 v = get_int_from_table_16(get_int_b1(bx->qs, iqs + l), kvalues_mxfp4);
                v0[l] = v.x;
                v1[l] = v.y;
            }
            const float d = ggml_cuda_e8m0_to_fp32(bx->e) * 0.5f;
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_table(v0, v1, 1, d,
                        y + (col + j)*stride_y + kbx*(QK_MXFP4/QK8_1), iqs);
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_NVFP4) {
        // NVFP4 blocks cover QK_NVFP4 values with a UE4M3 scale per 16-value sub-block
        for (int64_t kbx = tid/(qi/vdr); kbx < nblocks; kbx += vdr*warp_size/qi) {
            const block_nvfp4 * bx = (const block_nvfp4 *) (x + row*nblocks*sizeof(block_nvfp4)) + kbx;
            int2 v0[VDR_NVFP4_Q8_1_MMVQ/2];
            int2 v1[VDR_NVFP4_Q8_1_MMVQ/2];
            int sc[VDR_NVFP4_Q8_1_MMVQ/2];
#pragma unroll
            for (int i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; ++i) {
                const int iqs0 = iqs + 2*i;
                v0[i] = get_int_from_table_16(get_int_b4(bx->qs, iqs0 + 0), kvalues_mxfp4);
                v1[i] = get_int_from_table_16(get_int_b4(bx->qs, iqs0 + 1), kvalues_mxfp4);
                sc[i] = bx->d[iqs0 >> 1];
            }
#pragma unroll
            for (int j = 0; j < ncols_tile; ++j) {
                if (col + j < ncols) {
                    sum[j] += mmq_few_vec_dot_nvfp4(v0, v1, sc,
                        y + (col + j)*stride_y + kbx*(QK_NVFP4/QK8_1), iqs);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_tile; ++j) {
        sum[j] = warp_reduce_sum<warp_size>(sum[j]);
        if (tid == 0 && col + j < ncols) {
            dst[(col + j)*stride_dst + row] = sum[j];
        }
    }
}


template <ggml_type type>
static void launch_mul_mat_q_few(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const block_q8_1 * y, const int64_t k_padded, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 blocks(src0->ne[1], (src1->ne[1] + 3)/4);
    const dim3 threads(warp_size);
    const ggml_cuda_kernel_launch_params params(blocks, threads, 0, stream);
    ggml_cuda_kernel_launch(mul_mat_q_few<type>, params, (const char *) src0->data, y, (float *) dst->data,
        src0->ne[0]/mmq_few_block_span(type), k_padded/QK8_1, dst->nb[1]/sizeof(float), (int) src1->ne[1]);
}

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int64_t k_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool(), src1->ne[1]*k_padded/QK8_1);
    quantize_row_q8_1_cuda((const float *) src1->data, y.get(), src1->ne[0], src1->nb[1]/sizeof(float),
        src1->nb[2]/sizeof(float), src1->nb[3]/sizeof(float), k_padded, src1->ne[1], 1, 1, stream);

    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few<GGML_TYPE_Q4_K>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few<GGML_TYPE_Q6_K>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q8_0:
            launch_mul_mat_q_few<GGML_TYPE_Q8_0>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q_few<GGML_TYPE_Q3_K>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few<GGML_TYPE_Q5_K>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_q_few<GGML_TYPE_IQ4_NL>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_q_few<GGML_TYPE_IQ4_XS>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_q_few<GGML_TYPE_IQ3_S>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q4_0:
            launch_mul_mat_q_few<GGML_TYPE_Q4_0>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q4_1:
            launch_mul_mat_q_few<GGML_TYPE_Q4_1>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q5_0:
            launch_mul_mat_q_few<GGML_TYPE_Q5_0>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q5_1:
            launch_mul_mat_q_few<GGML_TYPE_Q5_1>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_Q2_K:
            launch_mul_mat_q_few<GGML_TYPE_Q2_K>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_MXFP4:
            launch_mul_mat_q_few<GGML_TYPE_MXFP4>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        case GGML_TYPE_NVFP4:
            launch_mul_mat_q_few<GGML_TYPE_NVFP4>(src0, src1, dst, y.get(), k_padded, stream);
            break;
        default:
            GGML_ABORT("unsupported type for few-row MMQ");
    }
}

static bool mmq_few_supported(const ggml_type type, const int64_t nrows, const int64_t ncols, const int64_t k) {
    switch (type) {
        case GGML_TYPE_Q2_K:
            return nrows < 1024 || ncols < 23 || k > 3072;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const int cc) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED_VARS(src0, src1, cc);
    return false;
#else
    // TODO: validated on gfx906 (VEGA20) only, the other GCN targets are untested
    return GGML_CUDA_CC_IS_GCN(cc) && mmq_few_supported(src0->type, src0->ne[1], src1->ne[1], src0->ne[0]) && src1->type == GGML_TYPE_F32 &&
        src1->ne[1] >= 9 && src1->ne[1] <= 24 && ggml_is_matrix(src0) && ggml_is_matrix(src1) &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_cuda_is_aligned(src0, 4) &&
        ggml_cuda_is_aligned(src1, 4) && src0->ne[0] % mmq_few_block_span(src0->type) == 0 &&
        src0->ne[0] >= 2048 && src0->ne[0] <= INT_MAX && src0->ne[1] <= 1024;
#endif // GGML_CUDA_FORCE_CUBLAS
}

#else // defined(GGML_USE_MUSA)

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst);
    GGML_ABORT("few-row MMQ is not supported on HIP or MUSA");
}

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const int cc) {
    GGML_UNUSED_VARS(src0, src1, ids, cc);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
