#include "mmq-few.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <type_traits>

// mul_mat_q_few for NVIDIA uses the int8 MMA fragment layout and cp.async, mul_mat_q_few_hip for RDNA the WMMA layout. Not on MUSA.
#if !defined(GGML_USE_MUSA)

using namespace ggml_cuda_mma;

// Few-row MMQ: int8 MMA mat-mul for 2-24 src1 columns. A CUDA block has 16 src0 rows (32 for 2 column groups on large matrices).
// Its warps split K in units of 256 values and copy raw src0 rows into their own cp.async ring, src1 is quantized to the MMA fragment layout.

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

template <ggml_type type, int ncg, int nmt>
__launch_bounds__(MMQ_FEW_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_q_few(
        const char * x, const int4 * y_ptr, float * dst,
        const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x, const int stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int stride_sample_dst, const bool x_is_constant) {
#if defined(CP_ASYNC_AVAILABLE)
    constexpr int nwarps    = MMQ_FEW_NWARPS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == WARP_SIZE, "unsupported warp size");

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

    const int row0        = blockIdx.x*nrows;
    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int nchannels   = gridDim.y;

    x   += int64_t(sample_dst/sample_ratio)*stride_sample_x + int64_t(channel_dst/channel_ratio)*stride_channel_x;
    dst += int64_t(sample_dst)*stride_sample_dst + int64_t(channel_dst)*stride_channel_dst;
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

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    // DGX Spark DRAM serves the short row pieces of the stage copies slowly, so the rows of the block are first prefetched into L2 in address order.
    {
        const int     tid       = threadIdx.y*warp_size + threadIdx.x;
        const int     nrows_blk = min(nrows, nrows_x - row0);
        const int64_t row_bytes = int64_t(nunits)*unit_size;
        const bool    contig    = stride_row_x == row_bytes;
        for (int ir = 0; ir < (contig ? 1 : nrows_blk); ++ir) {
            const uintptr_t p0 = (uintptr_t) (x + int64_t(row0 + ir)*stride_row_x);
            const uintptr_t p1 = p0 + (contig ? nrows_blk*row_bytes : row_bytes);
            for (uintptr_t p = (p0 & ~uintptr_t(127)) + 128*tid; p < p1; p += 128*nwarps*warp_size) {
                ggml_cuda_prefetch_l2((const void *) p);
            }
        }
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK

    const int iu_step = nunits_stage*nwarps;

#pragma unroll
    for (int stage = 0; stage < nstages - 1; ++stage) {
        const int iu0 = nunits_stage*threadIdx.y + stage*iu_step;
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
    if (nunits_stage*threadIdx.y < nunits) {
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
            load_y(yf[k], nunits_stage*threadIdx.y + k);
        }
    }

    float sum[nmt][ncg][4] = {{{0.0f}}};

    for (int iter = 0, iu0 = nunits_stage*threadIdx.y; iu0 < nunits; ++iter, iu0 += iu_step) {
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

#pragma unroll
    for (int m = 0; m < nmt; ++m) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + MMQ_FEW_NROWS*m + tile<16, 8, int>::get_i(l);
                const int j = c*MMQ_FEW_NCOLS + tile<16, 8, int>::get_j(l);
                if (i < nrows_x && j < ncols_dst) {
                    dst[int64_t(j)*stride_col_dst + i] = sum[m][c][l];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y_ptr, dst, nunits, nrows_x, ncols_dst, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_dst, x_is_constant);
    NO_DEVICE_CODE;
#endif // defined(CP_ASYNC_AVAILABLE)
}

// AMD version with WMMA on RDNA and MFMA on CDNA, for up to 16 src1 columns. Both multiply 16x16 int8 matrices, see mma.cuh:
//   WMMA: A = src1, B = src0, C = A @ B.
//     A and B: lane l holds the column resp. row l%16, the values [16k, 16k + 16) on RDNA3 and [8*(l/16), 8*(l/16) + 8) on RDNA4.
//     C: lane l holds the src1 columns 2v + l/16 (RDNA3) resp. 8*(l/16) + v (RDNA4) of the src0 row l%16.
//   MFMA: A = src0, B = src1, C[i][j] = sum_k A[i][k] * B[j][k].
//     A and B: lane l holds the src0 row resp. src1 column l%16, the values [4*(l/16), 4*(l/16) + 4) of the MFMA.
//     C: lane l holds the src1 columns 4*(l/16) + v of the src0 row l%16.
// So the src0 scales of a row stay in the lane that decodes it, the src1 scales of the C columns come with the layout in quantize.cuh.
// The lanes decode their rows straight from global memory. A ring as in mul_mat_q_few, filled through registers without cp.async, was slower.
// A second group of 16 columns spills registers with ROCm 7.2.

static constexpr __host__ __device__ bool mmq_few_hip_supported(const ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

static constexpr int MMQ_FEW_CDNA_MAX_NWARPS = 8;

static constexpr int MMQ_FEW_CDNA_NC         = 4; // C registers per lane and tile, 1 per src1 column

// the reduction over the warps, up to 8 C registers per lane, up to 64 lanes per warp, up to 4 tiles per warp
static constexpr int mmq_few_hip_nbytes_shared(const int nrt) {
    return (MMQ_FEW_NWARPS - 1)*nrt*8*64*sizeof(float);
}

// n ints at the even byte offset o of the 4 byte aligned p, with whole 4 byte loads. In the unrolled decoders o is a constant.
template <int n>
static __device__ __forceinline__ void mmq_few_hip_get_ints_b2(const char * p, const int o, int * dst) {
    const int * w = (const int *) (p + (o & ~3));
    if (o % 4 == 0) {
#pragma unroll
        for (int i = 0; i < n; ++i) {
            dst[i] = w[i];
        }
    } else {
        int lo = w[0];
#pragma unroll
        for (int i = 0; i < n; ++i) {
            const int hi = w[i + 1];
            dst[i] = __builtin_amdgcn_alignbyte(hi, lo, 2);
            lo = hi;
        }
    }
}

#if defined(AMD_MFMA_AVAILABLE)

// CDNA uses MFMAs with blocks: A = src1, B = src0, each block of C has its own sub-block of src0 and 4 values of K per MFMA.
// A lane decodes the sub-blocks of chain 0 and 1 of its slot in its src0 row from 16 byte loads, see mmq_few_cdna_row/slot/sub.
// A chain of up to 8 MFMAs covers a sub-block, then the C registers get the scales of the sub-block.
//   ncg == 0: 4-block MFMA 16x16x4 for 16 src1 columns. C block b is in the registers 4b + r, it holds the sub-block of slot b,
//             lane l holds the src1 column 4*(l/16) + r of the src0 row l%16.
//   ncg  > 0: 16-block MFMA 4x4x4 for 4*ncg src1 columns, one chain per group of 4 columns.
//             Lane l holds block l/4, the src1 columns 4*cg + r of its own row and sub-block. The slots are summed at the end.
// With few columns the 16-block MFMA saves most of the work on C, with 16 columns it needs 4 times the MFMAs and registers.
template <int ncg>
static constexpr int mmq_few_cdna_nc() {
    return ncg == 0 ? MMQ_FEW_CDNA_NC : MMQ_FEW_CDNA_NC*ncg;
}

// sub-block of chain c of slot g
template <ggml_type type>
static __device__ __forceinline__ int mmq_few_cdna_sub(const int g, const int c) {
    return type == GGML_TYPE_Q6_K ? 4*(g/2) + 2*c + g % 2 : 2*g + c;
}

// The src0 row and the slot of a lane, the slot selects the sub-blocks of the lane, see mmq_few_cdna_sub.
// ncg == 0: the row is l%16 and the slot l/16, fixed by the 4-block MFMA.
// ncg  > 0: lane l has block l/4, 16 lanes hold 4 rows and 4 slots. A row then takes fewer L1 cache lines per 16 lanes.
template <int ncg>
static __device__ __forceinline__ int mmq_few_cdna_row() {
    return ncg == 0 ? threadIdx.x % 16 : 4*(threadIdx.x / 16) + threadIdx.x % 4;
}
template <int ncg>
static __device__ __forceinline__ int mmq_few_cdna_slot() {
    return ncg == 0 ? threadIdx.x / 16 : (threadIdx.x / 4) % 4;
}

using mmq_few_cdna_acc16 = __attribute__((__vector_size__(16*sizeof(int)))) int;
using mmq_few_cdna_acc4  = __attribute__((__vector_size__( 4*sizeof(int)))) int;

// A zero the compiler cannot share between chains, it would copy a shared zero C through VGPRs for each chain.
static __device__ __forceinline__ int mmq_few_cdna_zero() {
    int z;
    asm volatile("v_mov_b32 %0, 0" : "=v"(z));
    return z;
}

// C of a chain of n MFMAs, C[4*bb + r] for block bb of the 4-block MFMA, C[r] for the 16-block MFMA
template <int ncg, int n>
static __device__ __forceinline__ void mmq_few_cdna_mma(int (&C)[ncg == 0 ? 16 : 4], const int * a, const int * b) {
    const int z = mmq_few_cdna_zero();
    if constexpr (ncg == 0) {
        mmq_few_cdna_acc16 acc = {z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z};
#pragma unroll
        for (int i = 0; i < n; ++i) {
            acc = __builtin_amdgcn_mfma_i32_16x16x4i8(a[i], b[i], acc, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            C[i] = acc[i];
        }
    } else {
        mmq_few_cdna_acc4 acc = {z, z, z, z};
#pragma unroll
        for (int i = 0; i < n; ++i) {
            acc = __builtin_amdgcn_mfma_i32_4x4x4i8(a[i], b[i], acc, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            C[i] = acc[i];
        }
    }
}

// 16 bytes from the 4 byte aligned p
static __device__ __forceinline__ int4 mmq_few_cdna_load_b4(const int * p) {
    int4 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 4), sizeof(int4));
    return v;
}

// The raw src0 data of a lane for one unit and tile, loaded one unit ahead of its use.
// Q4_K, Q5_K, Q8_0 and IQ4_NL: chain c holds sub-block 2g + c.
// Q6_K: chain c holds sub-block 4*(g/2) + 2c + g%2, the low and high nibbles of the same ql bytes.
struct mmq_few_cdna_x {
    int4 v[5];
    int  w[5];
};

template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_cdna_load_x(mmq_few_cdna_x & xr, const char * x) {
    const int g = mmq_few_cdna_slot<ncg>();
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
        const block_t * bx = (const block_t *) __builtin_assume_aligned(x, 16);
        xr.v[0] = *(const int4 *) bx; // dm and the 12 scale bytes
        xr.v[1] = ((const int4 *) (bx->qs + 32*g))[0];
        xr.v[2] = ((const int4 *) (bx->qs + 32*g))[1];
        if constexpr (type == GGML_TYPE_Q5_K) {
            xr.v[3] = ((const int4 *) bx->qh)[0];
            xr.v[4] = ((const int4 *) bx->qh)[1];
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        // Q6_K units are only 2 byte aligned: read 4 byte aligned words, the decoder shifts them by the offset
        const int   off = (uintptr_t) x % 4;
        const int * p   = (const int *) (x - off);
        xr.v[0] = mmq_few_cdna_load_b4(p + 8*g);                // ql bytes [32g, 32g + 32)
        xr.v[1] = mmq_few_cdna_load_b4(p + 8*g + 4);
        xr.v[2] = mmq_few_cdna_load_b4(p + 32 + 8*(g/2));       // qh bytes [32*(g/2), 32*(g/2) + 32)
        xr.v[3] = mmq_few_cdna_load_b4(p + 32 + 8*(g/2) + 4);
        xr.v[4] = mmq_few_cdna_load_b4(p + 48);                 // scales
        xr.w[0] = p[8*g + 8];
        xr.w[1] = p[32 + 8*(g/2) + 8];
        xr.w[2] = p[off == 0 ? 51 : 52]; // only used with off == 2, stays inside the unit
        xr.w[3] = *(const uint16_t *) &((const block_q6_K *) x)->d;
        xr.w[4] = off;
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        // the 2 blocks of 18 bytes at 36g
        const int * p = (const int *) (x + 36*g);
        xr.v[0] = mmq_few_cdna_load_b4(p);
        xr.v[1] = mmq_few_cdna_load_b4(p + 4);
        xr.w[0] = p[8];
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        // the 2 blocks of 34 bytes at 68g
        const int * p = (const int *) (x + 68*g);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            xr.v[k] = mmq_few_cdna_load_b4(p + 4*k);
        }
        xr.w[0] = p[16];
    } else {
        static_assert(type == GGML_TYPE_COUNT, "unsupported type");
    }
}

// The src1 data of a lane for one unit, see quantize.cuh. The columns beyond ne1 are zero.
// The d and sum values of a unit are read with one load per lane, then go through the LDS of the warp to the lanes that need them.
// 16 loads per lane straight from global memory, mostly the same address in a lane group, take most of the L1 bandwidth.
template <int ncg>
struct mmq_few_cdna_y {
    int    qs[ncg == 0 ? 1 : ncg][16]; // A of MFMA i of chain c of column group cg in qs[cg][8c + i]
    float4 ds;
};

template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmq_few_cdna_load_y(mmq_few_cdna_y<ncg> & yr, const int4 * y) {
    if constexpr (ncg == 0) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int4 q = y[k*64 + threadIdx.x];
            yr.qs[0][4*k + 0] = q.x;
            yr.qs[0][4*k + 1] = q.y;
            yr.qs[0][4*k + 2] = q.z;
            yr.qs[0][4*k + 3] = q.w;
        }
    } else {
        // 16 consecutive lanes read 256 consecutive bytes, 4 slots of 4 columns
#pragma unroll
        for (int cg = 0; cg < ncg; ++cg) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int4 q0 = y[((cg*2 + c)*2 + 0)*16 + threadIdx.x % 16];
                const int4 q1 = y[((cg*2 + c)*2 + 1)*16 + threadIdx.x % 16];
                yr.qs[cg][8*c + 0] = q0.x;
                yr.qs[cg][8*c + 1] = q0.y;
                yr.qs[cg][8*c + 2] = q0.z;
                yr.qs[cg][8*c + 3] = q0.w;
                yr.qs[cg][8*c + 4] = q1.x;
                yr.qs[cg][8*c + 5] = q1.y;
                yr.qs[cg][8*c + 6] = q1.z;
                yr.qs[cg][8*c + 7] = q1.w;
            }
        }
    }
    yr.ds = ((const float4 *) (y + 256))[threadIdx.x];
}

// ds[4*s + k]: d of sub-block s of the columns 4k + r, ds[32 + 4*s + k]: sum (DS4)
template <int ncg>
static __device__ __forceinline__ const float4 * mmq_few_cdna_store_ds(const mmq_few_cdna_y<ncg> & yr, float4 * ds_warp) {
    __builtin_amdgcn_wave_barrier(); // the previous unit is read
    ds_warp[threadIdx.x] = yr.ds;
    __builtin_amdgcn_wave_barrier();
    return ds_warp;
}

// vec_dot: one unit of 256 values of the nrt tiles of 16 rows.
// For C register i of a chain: the block bb and the column group k of ds, the sub-block is mmq_few_cdna_sub(sb, c).
template <int ncg>
static __device__ __forceinline__ int mmq_few_cdna_sb(const int bb) {
    return ncg == 0 ? bb : mmq_few_cdna_slot<ncg>();
}
template <int ncg>
static __device__ __forceinline__ int mmq_few_cdna_k(const int cg) {
    return ncg == 0 ? threadIdx.x / 16 : cg;
}

template <ggml_type type, int ncg, int nrt>
static __device__ __forceinline__ void mmq_few_cdna_vec_dot_q45_K(
        const mmq_few_cdna_x (&xr)[nrt], const mmq_few_cdna_y<ncg> & yr, const float4 * ds, float (&sum)[nrt][mmq_few_cdna_nc<ncg>()]) {
    constexpr int ncgr = ncg == 0 ? 1 : ncg;
    constexpr int nblk = ncg == 0 ? 4 : 1;
    const int g = mmq_few_cdna_slot<ncg>();
#pragma unroll
    for (int t = 0; t < nrt; ++t) {
        const int    scales[3] = {xr[t].v[0].y, xr[t].v[0].z, xr[t].v[0].w};
        const int    sc[2]     = {unpack_scales_q45_K(scales, 0), unpack_scales_q45_K(scales, 1)};
        const int    mn[2]     = {unpack_scales_q45_K(scales, 2), unpack_scales_q45_K(scales, 3)};
        const float2 dm        = __half22float2(*(const half2 *) &xr[t].v[0].x);
        const int    qs[8]     = {xr[t].v[1].x, xr[t].v[1].y, xr[t].v[1].z, xr[t].v[1].w, xr[t].v[2].x, xr[t].v[2].y, xr[t].v[2].z, xr[t].v[2].w};
        const int    qh[8]     = {xr[t].v[3].x, xr[t].v[3].y, xr[t].v[3].z, xr[t].v[3].w, xr[t].v[4].x, xr[t].v[4].y, xr[t].v[4].z, xr[t].v[4].w};

        float sum_d[ncgr][MMQ_FEW_CDNA_NC] = {};
        float sum_m[ncgr][MMQ_FEW_CDNA_NC] = {};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            int b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                b[i] = (qs[i] >> (4*c)) & 0x0F0F0F0F;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    // bit 2j + c of qh holds the high bit of sub-block 2j + c
                    b[i] |= ((qh[i] >> (2*g + c)) & 0x01010101) << 4;
                }
            }
#pragma unroll
            for (int cg = 0; cg < ncgr; ++cg) {
                int C[nblk*4];
                mmq_few_cdna_mma<ncg, 8>(C, yr.qs[cg] + 8*c, b);
#pragma unroll
                for (int bb = 0; bb < nblk; ++bb) {
                    const int    s  = mmq_few_cdna_sub<type>(mmq_few_cdna_sb<ncg>(bb), c);
                    const int    ls = ((s < 4 ? sc[0] : sc[1]) >> (8*(s % 4))) & 0xFF;
                    const float4 dy = ds[4*s + mmq_few_cdna_k<ncg>(cg)];
                    sum_d[cg][0] += dy.x * float(__mul24(C[4*bb + 0], ls));
                    sum_d[cg][1] += dy.y * float(__mul24(C[4*bb + 1], ls));
                    sum_d[cg][2] += dy.z * float(__mul24(C[4*bb + 2], ls));
                    sum_d[cg][3] += dy.w * float(__mul24(C[4*bb + 3], ls));
                }
            }
        }
        // the mins of the sub-blocks of the C registers: all for ncg == 0, the 2 of the lane group else
#pragma unroll
        for (int i = 0; i < (ncg == 0 ? 8 : 2); ++i) {
            const int   s = ncg == 0 ? i : mmq_few_cdna_sub<type>(g, i);
            const float m = ((s < 4 ? mn[0] : mn[1]) >> (8*(s % 4))) & 0xFF;
#pragma unroll
            for (int cg = 0; cg < ncgr; ++cg) {
                const float4 sy = ds[32 + 4*s + mmq_few_cdna_k<ncg>(cg)];
                sum_m[cg][0] += m * sy.x;
                sum_m[cg][1] += m * sy.y;
                sum_m[cg][2] += m * sy.z;
                sum_m[cg][3] += m * sy.w;
            }
        }
#pragma unroll
        for (int cg = 0; cg < ncgr; ++cg) {
#pragma unroll
            for (int r = 0; r < MMQ_FEW_CDNA_NC; ++r) {
                sum[t][MMQ_FEW_CDNA_NC*cg + r] += dm.x*sum_d[cg][r] - dm.y*sum_m[cg][r];
            }
        }
    }
}

template <int ncg, int nrt>
static __device__ __forceinline__ void mmq_few_cdna_vec_dot_q6_K(
        const mmq_few_cdna_x (&xr)[nrt], const mmq_few_cdna_y<ncg> & yr, const float4 * ds, float (&sum)[nrt][mmq_few_cdna_nc<ncg>()]) {
    constexpr int ncgr = ncg == 0 ? 1 : ncg;
    constexpr int nblk = ncg == 0 ? 4 : 1;
    const int g = mmq_few_cdna_slot<ncg>();
#pragma unroll
    for (int t = 0; t < nrt; ++t) {
        const int off = xr[t].w[4];
        const int wl[9] = {xr[t].v[0].x, xr[t].v[0].y, xr[t].v[0].z, xr[t].v[0].w, xr[t].v[1].x, xr[t].v[1].y, xr[t].v[1].z, xr[t].v[1].w, xr[t].w[0]};
        const int wh[9] = {xr[t].v[2].x, xr[t].v[2].y, xr[t].v[2].z, xr[t].v[2].w, xr[t].v[3].x, xr[t].v[3].y, xr[t].v[3].z, xr[t].v[3].w, xr[t].w[1]};
        const int ws[5] = {xr[t].v[4].x, xr[t].v[4].y, xr[t].v[4].z, xr[t].v[4].w, xr[t].w[2]};
        int ql[8];
        int qh[8];
        int sc[4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ql[i] = __builtin_amdgcn_alignbyte(wl[i + 1], wl[i], off);
            qh[i] = __builtin_amdgcn_alignbyte(wh[i + 1], wh[i], off);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc[i] = __builtin_amdgcn_alignbyte(ws[i + 1], ws[i], off);
        }
        const float d = __half2float(__ushort_as_half((unsigned short) xr[t].w[3]));

        float sum_d[ncgr][MMQ_FEW_CDNA_NC] = {};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int k = 2*c + g % 2; // the 2 bit field of qh
            int b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                b[i] = __vsub4(((ql[i] >> (4*c)) & 0x0F0F0F0F) | (((qh[i] >> (2*k)) & 0x03030303) << 4), 0x20202020);
            }
#pragma unroll
            for (int cg = 0; cg < ncgr; ++cg) {
                // a scale per 16 values: one chain per half of the sub-block
                int C_lo[nblk*4];
                int C_hi[nblk*4];
                mmq_few_cdna_mma<ncg, 4>(C_lo, yr.qs[cg] + 8*c + 0, b + 0);
                mmq_few_cdna_mma<ncg, 4>(C_hi, yr.qs[cg] + 8*c + 4, b + 4);
#pragma unroll
                for (int bb = 0; bb < nblk; ++bb) {
                    const int    s     = mmq_few_cdna_sub<GGML_TYPE_Q6_K>(mmq_few_cdna_sb<ncg>(bb), c);
                    const int    sc2   = s < 2 ? sc[0] : s < 4 ? sc[1] : s < 6 ? sc[2] : sc[3];
                    const int    scs   = sc2 >> (16*(s % 2)); // the scales 2s and 2s + 1
                    const int    sc_lo = (int8_t) (scs >> 0);
                    const int    sc_hi = (int8_t) (scs >> 8);
                    const float4 dy    = ds[4*s + mmq_few_cdna_k<ncg>(cg)];
                    sum_d[cg][0] += dy.x * float(__mul24(C_lo[4*bb + 0], sc_lo) + __mul24(C_hi[4*bb + 0], sc_hi));
                    sum_d[cg][1] += dy.y * float(__mul24(C_lo[4*bb + 1], sc_lo) + __mul24(C_hi[4*bb + 1], sc_hi));
                    sum_d[cg][2] += dy.z * float(__mul24(C_lo[4*bb + 2], sc_lo) + __mul24(C_hi[4*bb + 2], sc_hi));
                    sum_d[cg][3] += dy.w * float(__mul24(C_lo[4*bb + 3], sc_lo) + __mul24(C_hi[4*bb + 3], sc_hi));
                }
            }
        }
#pragma unroll
        for (int cg = 0; cg < ncgr; ++cg) {
#pragma unroll
            for (int r = 0; r < MMQ_FEW_CDNA_NC; ++r) {
                sum[t][MMQ_FEW_CDNA_NC*cg + r] += d*sum_d[cg][r];
            }
        }
    }
}

// Q8_0 and IQ4_NL have a block with its own d per 32 values, for ncg == 0 the d of the other slots come from the lanes of these slots.
template <ggml_type type, int ncg, int nrt>
static __device__ __forceinline__ void mmq_few_cdna_vec_dot_q8_0_iq4_nl(
        const mmq_few_cdna_x (&xr)[nrt], const mmq_few_cdna_y<ncg> & yr, const float4 * ds, float (&sum)[nrt][mmq_few_cdna_nc<ncg>()]) {
    constexpr int ncgr = ncg == 0 ? 1 : ncg;
    constexpr int nblk = ncg == 0 ? 4 : 1;
    // the 4 byte words of the 2 blocks, the d of the first block in the low half of word 0, the qs follow it
    constexpr int nw = type == GGML_TYPE_Q8_0 ? 17 : 9;
#pragma unroll
    for (int t = 0; t < nrt; ++t) {
        int w[nw];
#pragma unroll
        for (int k = 0; k < nw/4; ++k) {
            w[4*k + 0] = xr[t].v[k].x;
            w[4*k + 1] = xr[t].v[k].y;
            w[4*k + 2] = xr[t].v[k].z;
            w[4*k + 3] = xr[t].v[k].w;
        }
        w[nw - 1] = xr[t].w[0];
        constexpr int nq = nw/2; // qs words per block, the second block starts at word nq
        int qs[2][nq];
#pragma unroll
        for (int i = 0; i < nq; ++i) {
            qs[0][i] = __builtin_amdgcn_alignbyte(w[i + 1], w[i], 2);
            qs[1][i] = w[nq + 1 + i];
        }
        const int dd = (w[0] & 0x0000FFFF) | (w[nq] & 0xFFFF0000); // d of the blocks 2g and 2g + 1

        float2 dx[nblk];
#pragma unroll
        for (int bb = 0; bb < nblk; ++bb) {
            const int v = ncg == 0 ? __shfl(dd, 16*bb + threadIdx.x % 16, 64) : dd;
            dx[bb] = __half22float2(*(const half2 *) &v);
        }
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            int b[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (type == GGML_TYPE_Q8_0) {
                    b[i + 0] = qs[c][i + 0];
                    b[i + 4] = qs[c][i + 4];
                } else {
                    const int2 v = get_int_from_table_16(qs[c][i], kvalues_iq4nl);
                    b[i + 0] = v.x;
                    b[i + 4] = v.y;
                }
            }
#pragma unroll
            for (int cg = 0; cg < ncgr; ++cg) {
                int C[nblk*4];
                mmq_few_cdna_mma<ncg, 8>(C, yr.qs[cg] + 8*c, b);
#pragma unroll
                for (int bb = 0; bb < nblk; ++bb) {
                    const int    s   = mmq_few_cdna_sub<type>(mmq_few_cdna_sb<ncg>(bb), c);
                    const float  dxs = c == 0 ? dx[bb].x : dx[bb].y;
                    const float4 dy  = ds[4*s + mmq_few_cdna_k<ncg>(cg)];
                    sum[t][MMQ_FEW_CDNA_NC*cg + 0] += dxs*dy.x * float(C[4*bb + 0]);
                    sum[t][MMQ_FEW_CDNA_NC*cg + 1] += dxs*dy.y * float(C[4*bb + 1]);
                    sum[t][MMQ_FEW_CDNA_NC*cg + 2] += dxs*dy.z * float(C[4*bb + 2]);
                    sum[t][MMQ_FEW_CDNA_NC*cg + 3] += dxs*dy.w * float(C[4*bb + 3]);
                }
            }
        }
    }
}

template <ggml_type type, int ncg, int nrt>
static __device__ __forceinline__ void mmq_few_cdna_vec_dot(
        const mmq_few_cdna_x (&xr)[nrt], const mmq_few_cdna_y<ncg> & yr, const float4 * ds, float (&sum)[nrt][mmq_few_cdna_nc<ncg>()]) {
    if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL) {
        mmq_few_cdna_vec_dot_q8_0_iq4_nl<type, ncg, nrt>(xr, yr, ds, sum);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        mmq_few_cdna_vec_dot_q45_K<type, ncg, nrt>(xr, yr, ds, sum);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        mmq_few_cdna_vec_dot_q6_K<ncg, nrt>(xr, yr, ds, sum);
    } else {
        static_assert(type == GGML_TYPE_COUNT, "unsupported type");
    }
}

#elif defined(AMD_WMMA_AVAILABLE)

// The A and B tiles of a lane for one WMMA, MIRRORED on RDNA3 where both half-warps hold the same data.
#if defined(AMD_WMMA_256B_AVAILABLE)
using mmq_few_hip_tile_ab = tile<16, 4, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
#else
using mmq_few_hip_tile_ab = tile<16, 4, int, DATA_LAYOUT_I_MAJOR>;
#endif
using mmq_few_hip_tile_c = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

// int32 per lane and WMMA, 4 on RDNA3 and 2 on RDNA4
static constexpr int MMQ_FEW_HIP_NE = mmq_few_hip_tile_ab::ne;
// int32 per lane and C tile, 1 per 2 src1 columns
static constexpr int MMQ_FEW_HIP_NC = 8;

// The src1 fragments of a lane for one sub-block, see quantize.cuh.
struct mmq_few_hip_y {
    int qs[2][MMQ_FEW_HIP_NE]; // [hh] holds the values [16*hh, 16*hh + 16) of the WMMA
    int ds[MMQ_FEW_HIP_NC];   // d (D4) resp. (d, sum) (DS4) of the C columns, ds[v] of C.x[v]
};

// The C registers of a lane, on RDNA3 the ones with v >= 4 hold only the columns >= 8.
template <bool half>
static constexpr int mmq_few_hip_nv() {
    return half && MMQ_FEW_HIP_NE == 4 ? 4 : MMQ_FEW_HIP_NC;
}

// the src0 row and the src1 column of C.x[v]
static __device__ __forceinline__ int mmq_few_hip_c_row(const int v) {
    return mmq_few_hip_tile_c::get_j(v);
}
static __device__ __forceinline__ int mmq_few_hip_c_col(const int v) {
    return mmq_few_hip_tile_c::get_i(v);
}

// half: only the first 8 columns are used, the others are zero.
template <bool half>
static __device__ __forceinline__ mmq_few_hip_y mmq_few_hip_load_y(const int4 * y, const int s) {
    const int m = threadIdx.x % 16;
    const int h = threadIdx.x / 16;

    mmq_few_hip_y yf;
    if constexpr (MMQ_FEW_HIP_NE == 2) {
        // RDNA4: the lane holds the values [16*hh + 8*h, 16*hh + 8*h + 8) of column m.
        // The src1 quantize kernel zeroes the columns beyond ne1, a branch to skip the unused ones is more expensive than the loads.
        const int2 * y2 = (const int2 *) y;
        const int2 q0 = y2[(s*MMQ_FEW_HIP_NCOLS + m)*4 + 0 + h];
        const int2 q1 = y2[(s*MMQ_FEW_HIP_NCOLS + m)*4 + 2 + h];
        yf.qs[0][0] = q0.x; yf.qs[0][1] = q0.y;
        yf.qs[1][0] = q1.x; yf.qs[1][1] = q1.y;
    } else if (half && m >= MMQ_FEW_HIP_NCOLS/2) {
        yf.qs[0][0] = 0; yf.qs[0][1] = 0; yf.qs[0][2] = 0; yf.qs[0][3] = 0;
        yf.qs[1][0] = 0; yf.qs[1][1] = 0; yf.qs[1][2] = 0; yf.qs[1][3] = 0;
    } else {
        const int4 q0 = y[(s*MMQ_FEW_HIP_NCOLS + m)*2 + 0];
        const int4 q1 = y[(s*MMQ_FEW_HIP_NCOLS + m)*2 + 1];
        yf.qs[0][0] = q0.x; yf.qs[0][1] = q0.y; yf.qs[0][2] = q0.z; yf.qs[0][3] = q0.w;
        yf.qs[1][0] = q1.x; yf.qs[1][1] = q1.y; yf.qs[1][2] = q1.z; yf.qs[1][3] = q1.w;
    }

    if constexpr (MMQ_FEW_HIP_NE == 2) {
        // RDNA4: C.x[v] is the column 8*h + v
        const int4 d0 = y[8*MMQ_FEW_HIP_NCOLS*2 + 4*s + 0 + h];
        const int4 d1 = y[8*MMQ_FEW_HIP_NCOLS*2 + 4*s + 2 + h];
        yf.ds[0] = d0.x; yf.ds[1] = d1.x; yf.ds[2] = d0.y; yf.ds[3] = d1.y;
        yf.ds[4] = d0.z; yf.ds[5] = d1.z; yf.ds[6] = d0.w; yf.ds[7] = d1.w;
    } else {
        const int4 ds0 = y[8*MMQ_FEW_HIP_NCOLS*2 + (s*2 + h)*2 + 0];
        const int4 ds1 = half ? make_int4(0, 0, 0, 0) : y[8*MMQ_FEW_HIP_NCOLS*2 + (s*2 + h)*2 + 1];
        yf.ds[0] = ds0.x; yf.ds[1] = ds0.y; yf.ds[2] = ds0.z; yf.ds[3] = ds0.w;
        yf.ds[4] = ds1.x; yf.ds[5] = ds1.y; yf.ds[6] = ds1.z; yf.ds[7] = ds1.w;
    }
    return yf;
}

template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ float mmq_few_hip_get_d(const mmq_few_hip_y & yf, const int v) {
    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        return __int_as_float(yf.ds[v]);
    } else {
        return __low2float(*(const half2 *) &yf.ds[v]);
    }
}

static __device__ __forceinline__ float mmq_few_hip_get_sum(const mmq_few_hip_y & yf, const int v) {
    return __high2float(*(const half2 *) &yf.ds[v]);
}

static __device__ __forceinline__ void mmq_few_hip_mma(mmq_few_hip_tile_c & C, const int * a, const int * b) {
    mmq_few_hip_tile_ab A;
    mmq_few_hip_tile_ab B;
#pragma unroll
    for (int l = 0; l < MMQ_FEW_HIP_NE; ++l) {
        A.x[l] = a[l];
        B.x[l] = b[l];
    }
    mma(C, A, B);
}

// vec_dot: one unit of 256 values of row l%16, x points to its data, y to the src1 fragments of the unit.
// The decoders write sub-block s of the row to b[8], the int32 at index i holds the values [4i, 4i + 4).
// One WMMA uses b + 4*hh + i0, with i0 = (4 - MMQ_FEW_HIP_NE)*(l/16) the offset of the values of the lane.

template <ggml_type type, bool half>
static __device__ __forceinline__ void mmq_few_hip_vec_dot_q45_K(const char * x, const int4 * y, float (&sum)[MMQ_FEW_HIP_NC]) {
    using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
    const block_t * bx = (const block_t *) __builtin_assume_aligned(x, 16);

    const int4  hdr       = *(const int4 *) bx; // dm and the 12 scale bytes
    const int   scales[3] = {hdr.y, hdr.z, hdr.w};
    const int   sc[2]     = {unpack_scales_q45_K(scales, 0), unpack_scales_q45_K(scales, 1)};
    const int   mn[2]     = {unpack_scales_q45_K(scales, 2), unpack_scales_q45_K(scales, 3)};
    const float2 dm       = __half22float2(bx->dm);

    [[maybe_unused]] int qh[8]; // Q5_K: bit s holds the high bit of sub-block s
    if constexpr (type == GGML_TYPE_Q5_K) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qh[i] = get_int_b4(bx->qh, i);
        }
    }

    float sum_d[8] = {0.0f};
    float sum_m[8] = {0.0f};
    const int i0 = (4 - MMQ_FEW_HIP_NE)*(threadIdx.x / 16);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int qs[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qs[i] = get_int_b4(bx->qs + 32*j, i);
        }
#pragma unroll
        for (int s = 2*j; s < 2*j + 2; ++s) {
            int b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                b[i] = (qs[i] >> (4*(s % 2))) & 0x0F0F0F0F;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    b[i] |= ((qh[i] >> s) & 0x01010101) << 4;
                }
            }
            const int   ls = (sc[s/4] >> (8*(s % 4))) & 0xFF;
            const float m  = (mn[s/4] >> (8*(s % 4))) & 0xFF;

            const mmq_few_hip_y yf = mmq_few_hip_load_y<half>(y, s);
            mmq_few_hip_tile_c C;
            mmq_few_hip_mma(C, yf.qs[0], b + i0 + 0);
            mmq_few_hip_mma(C, yf.qs[1], b + i0 + 4);
#pragma unroll
            for (int v = 0; v < mmq_few_hip_nv<half>(); ++v) {
                sum_d[v] += mmq_few_hip_get_d<MMQ_Q8_1_DS_LAYOUT_DS4>(yf, v) * float(C.x[v]*ls);
                sum_m[v] += m * mmq_few_hip_get_sum(yf, v);
            }
        }
    }
#pragma unroll
    for (int v = 0; v < MMQ_FEW_HIP_NC; ++v) {
        sum[v] += dm.x*sum_d[v] - dm.y*sum_m[v];
    }
}

// Q6_K has a scale per 16 values, one WMMA per half of a sub-block.
template <bool half>
static __device__ __forceinline__ void mmq_few_hip_vec_dot_q6_K(const char * x, const int4 * y, float (&sum)[MMQ_FEW_HIP_NC]) {
    const block_q6_K * bx = (const block_q6_K *) x;
    const float d = __half2float(bx->d);

    float sum_d[8] = {0.0f};
    const int i0 = (4 - MMQ_FEW_HIP_NE)*(threadIdx.x / 16);
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int ql[2][8]; // [ql offset 0 or 32]
        int qh[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ql[0][i] = get_int_b2(bx->ql + 64*n,      i);
            ql[1][i] = get_int_b2(bx->ql + 64*n + 32, i);
            qh[i]    = get_int_b2(bx->qh + 32*n,      i);
        }
        const int sc[2] = {get_int_b2(bx->scales + 8*n, 0), get_int_b2(bx->scales + 8*n, 1)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4*n + j;
            int b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                b[i] = __vsub4(((ql[j % 2][i] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[i] >> (2*j)) & 0x03030303) << 4), 0x20202020);
            }
            const int sc_lo = (int8_t) (sc[j/2] >> (16*(j % 2) + 0)); // scale of the values [0, 16)
            const int sc_hi = (int8_t) (sc[j/2] >> (16*(j % 2) + 8)); // scale of the values [16, 32)

            const mmq_few_hip_y yf = mmq_few_hip_load_y<half>(y, s);
            mmq_few_hip_tile_c C_lo;
            mmq_few_hip_tile_c C_hi;
            mmq_few_hip_mma(C_lo, yf.qs[0], b + i0 + 0);
            mmq_few_hip_mma(C_hi, yf.qs[1], b + i0 + 4);
#pragma unroll
            for (int v = 0; v < mmq_few_hip_nv<half>(); ++v) {
                sum_d[v] += mmq_few_hip_get_d<MMQ_Q8_1_DS_LAYOUT_D4>(yf, v) * float(C_lo.x[v]*sc_lo + C_hi.x[v]*sc_hi);
            }
        }
    }
#pragma unroll
    for (int v = 0; v < MMQ_FEW_HIP_NC; ++v) {
        sum[v] += d*sum_d[v];
    }
}

// IQ4_NL has a block with its own scale per 32 values.
template <bool half>
static __device__ __forceinline__ void mmq_few_hip_vec_dot_iq4_nl(const char * x, const int4 * y, float (&sum)[MMQ_FEW_HIP_NC]) {
    const int i0 = (4 - MMQ_FEW_HIP_NE)*(threadIdx.x / 16);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        // a unit starts 16 byte aligned, the qs of every second block 2 bytes off a 4 byte boundary
        const char * xa = (const char *) __builtin_assume_aligned(x, 16);
        int q4[4];
        mmq_few_hip_get_ints_b2<4>(xa, int(sizeof(block_iq4_nl))*s + 2, q4);
        int b[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = get_int_from_table_16(q4[i], kvalues_iq4nl);
            b[i + 0] = v.x;
            b[i + 4] = v.y;
        }
        const float d = __half2float(((const block_iq4_nl *) xa + s)->d);

        const mmq_few_hip_y yf = mmq_few_hip_load_y<half>(y, s);
        mmq_few_hip_tile_c C;
        mmq_few_hip_mma(C, yf.qs[0], b + i0 + 0);
        mmq_few_hip_mma(C, yf.qs[1], b + i0 + 4);
#pragma unroll
        for (int v = 0; v < mmq_few_hip_nv<half>(); ++v) {
            sum[v] += d*mmq_few_hip_get_d<MMQ_Q8_1_DS_LAYOUT_D4>(yf, v) * float(C.x[v]);
        }
    }
}

// Q8_0 has a block with its own scale per 32 values.
template <bool half>
static __device__ __forceinline__ void mmq_few_hip_vec_dot_q8_0(const char * x, const int4 * y, float (&sum)[MMQ_FEW_HIP_NC]) {
    const int i0 = (4 - MMQ_FEW_HIP_NE)*(threadIdx.x / 16);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        // a unit starts 16 byte aligned, the qs of every second block 2 bytes off a 4 byte boundary
        const char * xa = (const char *) __builtin_assume_aligned(x, 16);
        int b[8];
        mmq_few_hip_get_ints_b2<8>(xa, int(sizeof(block_q8_0))*s + 2, b);
        const float d = __half2float(((const block_q8_0 *) xa + s)->d);

        const mmq_few_hip_y yf = mmq_few_hip_load_y<half>(y, s);
        mmq_few_hip_tile_c C;
        mmq_few_hip_mma(C, yf.qs[0], b + i0 + 0);
        mmq_few_hip_mma(C, yf.qs[1], b + i0 + 4);
#pragma unroll
        for (int v = 0; v < mmq_few_hip_nv<half>(); ++v) {
            sum[v] += d*mmq_few_hip_get_d<MMQ_Q8_1_DS_LAYOUT_D4>(yf, v) * float(C.x[v]);
        }
    }
}

template <ggml_type type, bool half, int nrt>
static __device__ __forceinline__ void mmq_few_hip_vec_dot(const char * const * xr, const int4 * y, float (*sum)[MMQ_FEW_HIP_NC]) {
    // on RDNA a lane has no registers for a second tile, mmq_few_hip_nrt keeps nrt at 1 there
#pragma unroll
    for (int t = 0; t < nrt; ++t) {
        if constexpr (type == GGML_TYPE_IQ4_NL) {
            mmq_few_hip_vec_dot_iq4_nl<half>(xr[t], y, sum[t]);
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            mmq_few_hip_vec_dot_q8_0<half>(xr[t], y, sum[t]);
        } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            mmq_few_hip_vec_dot_q45_K<type, half>(xr[t], y, sum[t]);
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            mmq_few_hip_vec_dot_q6_K<half>(xr[t], y, sum[t]);
        } else {
            static_assert(type == GGML_TYPE_COUNT, "unsupported type");
        }
    }
}

#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

// The warps split K in units of 256 values, a block has nrt*16 src0 rows. half: at most 8 src1 columns.
template <ggml_type type, bool half, int nrt>
__launch_bounds__(MMQ_FEW_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_q_few_hip(
        const char * x, const int4 * y_ptr, float * dst,
        const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x, const int stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int stride_sample_dst) {
#if defined(AMD_WMMA_AVAILABLE)
    constexpr int nwarps    = MMQ_FEW_NWARPS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int unit_size = mmq_few_unit_size(type);
    static_assert(mmq_few_hip_supported(type), "unsupported type");

    extern __shared__ float buf_sum_mmq_few_hip[];

    const int row0        = blockIdx.x*(nrt*MMQ_FEW_NROWS);
    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int nchannels   = gridDim.y;

    x   += int64_t(sample_dst/sample_ratio)*stride_sample_x + int64_t(channel_dst/channel_ratio)*stride_channel_x;
    dst += int64_t(sample_dst)*stride_sample_dst + int64_t(channel_dst)*stride_channel_dst;
    const int4 * GGML_CUDA_RESTRICT y = y_ptr + (int64_t(sample_dst)*nchannels + channel_dst)*nunits*MMQ_FEW_HIP_Q8_1_INT4;

    const char * xr[nrt];
#pragma unroll
    for (int t = 0; t < nrt; ++t) {
        xr[t] = x + int64_t(min(row0 + t*MMQ_FEW_NROWS + int(threadIdx.x) % 16, nrows_x - 1))*stride_row_x;
    }

    float sum[nrt][MMQ_FEW_HIP_NC] = {};
    for (int iu = threadIdx.y; iu < nunits; iu += nwarps) {
        const char * xu[nrt];
#pragma unroll
        for (int t = 0; t < nrt; ++t) {
            xu[t] = xr[t] + int64_t(iu)*unit_size;
        }
        mmq_few_hip_vec_dot<type, half, nrt>(xu, y + int64_t(iu)*MMQ_FEW_HIP_Q8_1_INT4, sum);
    }

    // reduce the sums over the warps
    float * buf_sum = buf_sum_mmq_few_hip;
    if (threadIdx.y > 0) {
#pragma unroll
        for (int t = 0; t < nrt; ++t) {
#pragma unroll
            for (int v = 0; v < MMQ_FEW_HIP_NC; ++v) {
                buf_sum[(((threadIdx.y - 1)*nrt + t)*MMQ_FEW_HIP_NC + v)*warp_size + threadIdx.x] = sum[t][v];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int w = 0; w < nwarps - 1; ++w) {
#pragma unroll
        for (int t = 0; t < nrt; ++t) {
#pragma unroll
            for (int v = 0; v < MMQ_FEW_HIP_NC; ++v) {
                sum[t][v] += buf_sum[((w*nrt + t)*MMQ_FEW_HIP_NC + v)*warp_size + threadIdx.x];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < nrt; ++t) {
#pragma unroll
        for (int v = 0; v < mmq_few_hip_nv<half>(); ++v) {
            const int i = row0 + t*MMQ_FEW_NROWS + mmq_few_hip_c_row(v);
            const int j = mmq_few_hip_c_col(v);
            if (i < nrows_x && j < ncols_dst) {
                dst[int64_t(j)*stride_col_dst + i] = sum[t][v];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y_ptr, dst, nunits, nrows_x, ncols_dst, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_dst);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE)
}

// The CDNA version: the warps of a block split K, their number is chosen at launch so that all blocks fit on the GPU at once.
// The raw data of the next unit is loaded before the current one is decoded. ncg: see mmq_few_cdna_nc.
template <ggml_type type, int ncg, int nrt>
__launch_bounds__(MMQ_FEW_CDNA_MAX_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_q_few_cdna(
        const char * x, const int4 * y_ptr, float * dst,
        const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x, const int stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int stride_channel_dst,
        const int sample_ratio, const int64_t stride_sample_x, const int stride_sample_dst) {
#if defined(AMD_MFMA_AVAILABLE)
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int unit_size = mmq_few_unit_size(type);
    static_assert(mmq_few_hip_supported(type), "unsupported type");
    static_assert(warp_size == 64, "unsupported warp size");
    constexpr int nc = mmq_few_cdna_nc<ncg>();
    const int nwarps = blockDim.y;

    // the d and sum values of each warp, then the reduction over the warps
    extern __shared__ float4 data_mmq_few_cdna[];
    float4 * ds_warp = data_mmq_few_cdna + threadIdx.y*warp_size;

    const int row0        = blockIdx.x*(nrt*MMQ_FEW_NROWS);
    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int nchannels   = gridDim.y;

    x   += int64_t(sample_dst/sample_ratio)*stride_sample_x + int64_t(channel_dst/channel_ratio)*stride_channel_x;
    dst += int64_t(sample_dst)*stride_sample_dst + int64_t(channel_dst)*stride_channel_dst;
    const int4 * GGML_CUDA_RESTRICT y = y_ptr + (int64_t(sample_dst)*nchannels + channel_dst)*nunits*MMQ_FEW_HIP_Q8_1_INT4;

    const char * xr[nrt];
#pragma unroll
    for (int t = 0; t < nrt; ++t) {
        xr[t] = x + int64_t(min(row0 + t*MMQ_FEW_NROWS + mmq_few_cdna_row<ncg>(), nrows_x - 1))*stride_row_x;
    }

    // a warp without units loads the last unit and does not use it
    auto load = [&](mmq_few_cdna_x (&xu)[nrt], mmq_few_cdna_y<ncg> & yu, const int iu) {
        const int iuc = min(iu, nunits - 1);
#pragma unroll
        for (int t = 0; t < nrt; ++t) {
            mmq_few_cdna_load_x<type, ncg>(xu[t], xr[t] + int64_t(iuc)*unit_size);
        }
        mmq_few_cdna_load_y<type, ncg>(yu, y + int64_t(iuc)*MMQ_FEW_HIP_Q8_1_INT4);
    };

    // nstages buffers used in turn, the loads of a unit are issued nstages - 1 units ahead of its use.
    // The memory latency is much longer than the decoding of one unit. Copying between the buffers would cost many moves.
    // The main loop has no branches per unit, at a branch the compiler waits for all loads in flight.
    constexpr int nstages = 2;
    const int warp  = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int nunits_warp = (nunits - warp + nwarps - 1) / nwarps; // the units warp + i*nwarps

    mmq_few_cdna_x xb[nstages][nrt];
    mmq_few_cdna_y<ncg> yb[nstages];
#pragma unroll
    for (int k = 0; k < nstages - 1; ++k) {
        load(xb[k], yb[k], warp + k*nwarps);
    }

    float sum[nrt][nc] = {};
    int i = 0;
    for (; i + nstages <= nunits_warp; i += nstages) {
#pragma unroll
        for (int k = 0; k < nstages; ++k) {
            load(xb[(k + nstages - 1) % nstages], yb[(k + nstages - 1) % nstages], warp + (i + k + nstages - 1)*nwarps);
            mmq_few_cdna_vec_dot<type, ncg, nrt>(xb[k], yb[k], mmq_few_cdna_store_ds(yb[k], ds_warp), sum);
        }
    }
    // the last units are already loaded
#pragma unroll
    for (int k = 0; k < nstages - 1; ++k) {
        if (i + k < nunits_warp) {
            mmq_few_cdna_vec_dot<type, ncg, nrt>(xb[k], yb[k], mmq_few_cdna_store_ds(yb[k], ds_warp), sum);
        }
    }

    // with the 16-block MFMA the slots hold different sub-blocks of the same outputs
    if constexpr (ncg > 0) {
#pragma unroll
        for (int t = 0; t < nrt; ++t) {
#pragma unroll
            for (int v = 0; v < nc; ++v) {
                sum[t][v] += __shfl_xor(sum[t][v], 4, warp_size);
                sum[t][v] += __shfl_xor(sum[t][v], 8, warp_size);
            }
        }
    }

    // reduce the sums over the warps
    float * buf_sum = (float *) (data_mmq_few_cdna + nwarps*warp_size);
    if (threadIdx.y > 0) {
#pragma unroll
        for (int t = 0; t < nrt; ++t) {
#pragma unroll
            for (int v = 0; v < nc; ++v) {
                buf_sum[(((threadIdx.y - 1)*nrt + t)*nc + v)*warp_size + threadIdx.x] = sum[t][v];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
    for (int w = 0; w < nwarps - 1; ++w) {
#pragma unroll
        for (int t = 0; t < nrt; ++t) {
#pragma unroll
            for (int v = 0; v < nc; ++v) {
                sum[t][v] += buf_sum[((w*nrt + t)*nc + v)*warp_size + threadIdx.x];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < nrt; ++t) {
#pragma unroll
        for (int v = 0; v < nc; ++v) {
            const int i = row0 + t*MMQ_FEW_NROWS + mmq_few_cdna_row<ncg>();
            const int j = ncg == 0 ? 4*(threadIdx.x / 16) + v : v;
            if (i < nrows_x && j < ncols_dst && (ncg == 0 || mmq_few_cdna_slot<ncg>() == 0)) {
                dst[int64_t(j)*stride_col_dst + i] = sum[t][v];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y_ptr, dst, nunits, nrows_x, ncols_dst, stride_row_x, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_dst);
    NO_DEVICE_CODE;
#endif // defined(AMD_MFMA_AVAILABLE)
}

// ggml_is_constant (static in ggml.c) on the tensor that a view points to
static bool mmq_few_is_constant(const ggml_tensor * src0) {
    const ggml_tensor * x_root = src0->view_src ? src0->view_src : src0;
    return x_root->buffer && ggml_backend_buffer_get_usage(x_root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        !(x_root->flags & GGML_TENSOR_FLAG_PARAM);
}

template <ggml_type type, int ncg, int nmt>
static void launch_mul_mat_q_few(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const int nunits = ne00 / QK_K;
    GGML_ASSERT(nunits % mmq_few_nunits_stage(type, ncg, nmt) == 0);

    const int warp_size     = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nbytes_shared = mmq_few_nbytes_shared(type, ncg, nmt);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_few<type, ncg, nmt>), nbytes_shared);

    const bool x_is_constant = mmq_few_is_constant(src0);

    const dim3 block_nums((ne01 + nmt*MMQ_FEW_NROWS - 1) / (nmt*MMQ_FEW_NROWS), ne12, ne13);
    const dim3 block_dims(warp_size, MMQ_FEW_NWARPS, 1);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_q_few<type, ncg, nmt>, params,
        (const char *) src0->data, src1_q, (float *) dst->data,
        nunits, (int) ne01, (int) ne1, (int64_t) nb01, (int) (nb1/sizeof(float)),
        (int) (ne12/ne02), (int64_t) nb02, (int) (nb2/sizeof(float)),
        (int) (ne13/ne03), (int64_t) nb03, (int) (nb3/sizeof(float)), x_is_constant);
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
        const int4 * src1_q, const int ncg, cudaStream_t stream) {
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

template <ggml_type type, bool half, int nrt>
static void launch_mul_mat_q_few_hip(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const int nunits = ne00 / QK_K;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    const dim3 block_nums((ne01 + nrt*MMQ_FEW_NROWS - 1) / (nrt*MMQ_FEW_NROWS), ne12, ne13);
    const dim3 block_dims(warp_size, MMQ_FEW_NWARPS, 1);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, mmq_few_hip_nbytes_shared(nrt), stream);
    const auto launch = [&]() {
        ggml_cuda_kernel_launch(mul_mat_q_few_hip<type, half, nrt>, params,
            (const char *) src0->data, src1_q, (float *) dst->data,
            nunits, (int) ne01, (int) ne1, (int64_t) nb01, (int) (nb1/sizeof(float)),
            (int) (ne12/ne02), (int64_t) nb02, (int) (nb2/sizeof(float)),
            (int) (ne13/ne03), (int64_t) nb03, (int) (nb3/sizeof(float)));
    };
    launch();

    static const int bench_env = getenv("MMQ_FEW_BENCH") ? atoi(getenv("MMQ_FEW_BENCH")) : 0;
    hipStreamCaptureStatus cap;
    const bool bench = bench_env > 0 && hipStreamGetCaptureInfo(stream, &cap, nullptr) == hipSuccess &&
                       cap == hipStreamCaptureStatusNone;
    if (bench) {
        hipEvent_t e0;
        hipEvent_t e1;
        hipEventCreate(&e0);
        hipEventCreate(&e1);
        for (int i = 0; i < 3; ++i) {
            launch();
        }
        hipStreamSynchronize(stream);
        hipEventRecord(e0, stream);
        for (int i = 0; i < bench_env; ++i) {
            launch();
        }
        hipEventRecord(e1, stream);
        hipEventSynchronize(e1);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, e0, e1);
        printf("MMQ_FEW_BENCH type=%-8s nrt=%d half=%d %.2f us\n", ggml_type_name(type), nrt, (int) half, 1000.0f*ms/bench_env);
        fflush(stdout);
        hipEventDestroy(e0);
        hipEventDestroy(e1);
    }
}

// Groups of 4 src1 columns for the 16-block MFMA on CDNA, 0 for all 16 columns with the 4-block MFMA.
// Beyond 8 columns (Q6_K: 4) the src1 data of the 16-block MFMA takes too many registers.
static int mmq_few_cdna_ncg(const ggml_type type, const int64_t ncols) {
    return ncols <= (type == GGML_TYPE_Q6_K ? 4 : 8) ? (ncols + 3) / 4 : 0;
}

template <int ncg, int nrt>
static constexpr size_t mmq_few_cdna_nbytes_shared(const int nwarps) {
    return nwarps*64*sizeof(float4) + (nwarps - 1)*nrt*MMQ_FEW_CDNA_NC*(ncg == 0 ? 1 : ncg)*64*sizeof(float);
}

template <ggml_type type, int ncg, int nrt>
static void launch_mul_mat_q_few_cdna(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const int nunits = ne00 / QK_K;
    const int nsm    = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    const dim3    block_nums((ne01 + nrt*MMQ_FEW_NROWS - 1) / (nrt*MMQ_FEW_NROWS), ne12, ne13);
    const int64_t nblocks = int64_t(block_nums.x)*block_nums.y*block_nums.z;

    // about 2 waves per SIMD in total, the warps of a block split K
    int nwarps = std::min<int64_t>((8*nsm + nblocks - 1) / nblocks, MMQ_FEW_CDNA_MAX_NWARPS);
    nwarps = std::max(std::min(nwarps, nunits), std::min(2, nunits));

    const dim3 block_dims(64, nwarps, 1);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, mmq_few_cdna_nbytes_shared<ncg, nrt>(nwarps), stream);
    ggml_cuda_kernel_launch(mul_mat_q_few_cdna<type, ncg, nrt>, params,
        (const char *) src0->data, src1_q, (float *) dst->data,
        nunits, (int) ne01, (int) ne1, (int64_t) nb01, (int) (nb1/sizeof(float)),
        (int) (ne12/ne02), (int64_t) nb02, (int) (nb2/sizeof(float)),
        (int) (ne13/ne03), (int64_t) nb03, (int) (nb3/sizeof(float)));
}

template <ggml_type type>
static void launch_mul_mat_q_few_hip_switch_ncols(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int4 * src1_q, const int cc, cudaStream_t stream) {
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        switch (mmq_few_cdna_ncg(type, src1->ne[1])) {
            case 1:  launch_mul_mat_q_few_cdna<type, 1, 1>(src0, src1, dst, src1_q, stream); break;
            case 2:  launch_mul_mat_q_few_cdna<type, 2, 1>(src0, src1, dst, src1_q, stream); break;
            default: launch_mul_mat_q_few_cdna<type, 0, 1>(src0, src1, dst, src1_q, stream); break;
        }
        return;
    }
    if (src1->ne[1] <= MMQ_FEW_HIP_NCOLS/2) {
        launch_mul_mat_q_few_hip<type, true, 1>(src0, src1, dst, src1_q, stream);
        return;
    }
    launch_mul_mat_q_few_hip<type, false, 1>(src0, src1, dst, src1_q, stream);
}

// Min. and max. number of src1 columns for AMD GPUs, 0 if the type is not supported.
// Below the min. MMVQ is faster, above the max. MMQ.
// RDNA4 Q8_0 only beats MMVQ at 7 and 8 columns. On RDNA3 Q8_0 is not tested, on RDNA3.5 it would need more than 16 columns to beat MMQ.
static int2 mmq_few_hip_ncols_range(const ggml_type type, const int cc) {
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        switch (type) {
            case GGML_TYPE_Q8_0:   return make_int2(4, 16);
            case GGML_TYPE_Q4_K:   return make_int2(2, 16);
            case GGML_TYPE_Q5_K:   return make_int2(3, 16);
            case GGML_TYPE_Q6_K:   return make_int2(3, 16);
            case GGML_TYPE_IQ4_NL: return make_int2(2, 16);
            default:               return make_int2(0,  0);
        }
    }
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        switch (type) {
            case GGML_TYPE_Q8_0:   return make_int2(7,  8);
            case GGML_TYPE_Q4_K:   return make_int2(3, 16);
            case GGML_TYPE_Q5_K:   return make_int2(3, 16);
            case GGML_TYPE_Q6_K:   return make_int2(5, 16);
            case GGML_TYPE_IQ4_NL: return make_int2(6, 16);
            default:               return make_int2(0,  0);
        }
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        switch (type) {
            case GGML_TYPE_Q4_K:   return make_int2(2, 16);
            case GGML_TYPE_Q5_K:   return make_int2(2, 16);
            case GGML_TYPE_Q6_K:   return make_int2(5, 16);
            case GGML_TYPE_IQ4_NL: return make_int2(5, 16);
            default:               return make_int2(0,  0);
        }
    }
    switch (type) {
        case GGML_TYPE_Q4_K:   return make_int2(3, 16);
        case GGML_TYPE_Q5_K:   return make_int2(3,  8);
        case GGML_TYPE_Q6_K:   return make_int2(6, 16);
        case GGML_TYPE_IQ4_NL: return make_int2(7, 16);
        default:               return make_int2(0,  0);
    }
}

static bool mmq_few_use_hip(const int cc) {
    return GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc);
}

static void ggml_cuda_mul_mat_q_few_hip(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(ne11 <= MMQ_FEW_HIP_NCOLS);

    cudaStream_t stream = ctx.stream();

    const int cc     = ggml_cuda_info().devices[ctx.device].cc;
    const int nunits = ne00 / QK_K;

    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool(), int64_t(ne12)*ne13*nunits*MMQ_FEW_HIP_Q8_1_INT4);
    const bool narrow = GGML_CUDA_CC_IS_CDNA(cc) && mmq_few_cdna_ncg(src0->type, ne11) > 0;
    quantize_mmq_few_hip_q8_1_cuda((const float *) src1->data, src1_q.get(), src0->type, narrow, mmq_few_ds_layout(src0->type), ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
        ne11, ne12, ne13, stream);

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_q_few_hip_switch_ncols<GGML_TYPE_Q8_0>(src0, src1, dst, src1_q.get(), cc, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few_hip_switch_ncols<GGML_TYPE_Q4_K>(src0, src1, dst, src1_q.get(), cc, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few_hip_switch_ncols<GGML_TYPE_Q5_K>(src0, src1, dst, src1_q.get(), cc, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few_hip_switch_ncols<GGML_TYPE_Q6_K>(src0, src1, dst, src1_q.get(), cc, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_q_few_hip_switch_ncols<GGML_TYPE_IQ4_NL>(src0, src1, dst, src1_q.get(), cc, stream);
            break;
        default:
            GGML_ABORT("unsupported type for few-row MMQ with WMMA or MFMA");
    }
}

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(ne11 <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS);
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);

    if (mmq_few_use_hip(ggml_cuda_info().devices[ctx.device].cc)) {
        ggml_cuda_mul_mat_q_few_hip(ctx, src0, src1, dst);
        return;
    }

    cudaStream_t stream = ctx.stream();

    const int nunits = ne00 / QK_K;
    const int ncg    = mmq_few_ncg(ne11);

    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool(), int64_t(ne12)*ne13*nunits*ncg*MMQ_FEW_Q8_1_INT4);
    quantize_mmq_few_q8_1_cuda((const float *) src1->data, src1_q.get(), mmq_few_ds_layout(src0->type), ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
        ne11, ne12, ne13, stream);

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q8_0>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q3_K>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q4_K>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q5_K>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q6_K>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ3_S>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ4_NL>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_IQ4_XS>(src0, src1, dst, src1_q.get(), ncg, stream);
            break;
        default:
            GGML_ABORT("unsupported type for few-row MMQ");
    }
}

// Min. number of src1 columns for the few-row MMQ, 0 if it is not used for this type, GPU and row length.
static int ggml_cuda_mmq_few_min_ncols(const ggml_type type, const int cc, const int64_t ne00) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return 0;
#endif // GGML_CUDA_FORCE_CUBLAS

    if (mmq_few_use_hip(cc)) {
        return !mmq_few_hip_supported(type) || ne00 < 8*QK_K ? 0 : mmq_few_hip_ncols_range(type, cc).x;
    }

    // NVIDIA Ampere to Blackwell
    if (!ampere_mma_available(cc) || cc >= GGML_CUDA_CC_RUBIN) {
        return 0;
    }
    // on DGX Spark the kernel needs its L2 prefetch, which is only in the DGX Spark device code
    if (cc == GGML_CUDA_CC_DGX_SPARK && ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_DGX_SPARK) {
        return 0;
    }
    // with fewer than 8 units of 256 values per row the warps idle, MMVQ and MMQ are better there
    if (mmq_few_unit_size(type) == 0 || ne00 < 8*QK_K) {
        return 0;
    }
    if (cc == GGML_CUDA_CC_DGX_SPARK) { // tuned on DGX Spark GB10
        return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL ? 8 : 2;
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

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const int cc) {
    const int min_ncols = ggml_cuda_mmq_few_min_ncols(src0->type, cc, src0->ne[0]);
    if (min_ncols == 0 || src1->ne[1] < min_ncols || src1->ne[1] > MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS) {
        return false;
    }
    if (mmq_few_use_hip(cc)) {
        if (src1->ne[1] > mmq_few_hip_ncols_range(src0->type, cc).y) {
            return false;
        }
        // the decoders read whole 4 byte values from 16 byte aligned units, Q6_K values 2 bytes at a time
        const int x_alignment = mmq_few_unaligned(src0->type, 1) ? 2 : 16;
        return ggml_cuda_is_aligned(src0, x_alignment) && ggml_cuda_is_aligned(src1, 16);
    }
    // the row must hold a whole number of ring stages of the launched kernel
    const int ncg          = mmq_few_ncg(src1->ne[1]);
    const int nmt          = mmq_few_nmt(src0->type, ncg, src0->ne[1]);
    const int nunits_stage = mmq_few_nunits_stage(src0->type, ncg, nmt);
    if (src0->ne[0] % (nunits_stage*QK_K) != 0) {
        return false;
    }
    // unaligned data needs the alignment of the decoder loads
    const int x_alignment = mmq_few_unaligned(src0->type, nunits_stage) ? (ggml_type_size(src0->type) % 4 == 0 ? 4 : 2) : 16;
    return ggml_cuda_is_aligned(src0, x_alignment) && ggml_cuda_is_aligned(src1, 16);
}

#else // !defined(GGML_USE_MUSA)

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    GGML_ABORT("few-row MMQ is not supported on MUSA");
}

bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const int cc) {
    GGML_UNUSED_VARS(src0, src1, cc);
    return false;
}

#endif // !defined(GGML_USE_MUSA)
