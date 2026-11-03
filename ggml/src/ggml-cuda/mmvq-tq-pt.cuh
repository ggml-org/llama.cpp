#pragma once

#include "common.cuh"
#include "vecdotq.cuh"   // get_int_b2: block_tq1_0 is 54 and block_tq2_0 66 bytes, so a 4-byte
                         // load of an odd numbered block is not safe

#include <algorithm>
#include <cstdlib>

// mmvq.cuh carries no include guard at all (it is meant to be included exactly once per TU), so
// this header must not pull it in - quantize.cu includes us as well. The MoE kernel runs one warp
// per expert slot and there are never more slots than MMVQ_MAX_BATCH_SIZE, which is 8.
#define TQ_PT_MAX_SLOTS 8

// =============================================================================
// Planar-transposed (PT) activations for the ternary GEMV path (TQ1_0 and
// TQ2_0: the layout and the kernel geometry are type independent, only the
// weight-side digit extraction is not).
//
// Why. mul_mat_vec_q reads block_q8_1 activations in AoS order: 32 int8 quants
// followed by a half2, 36 bytes per block. A warp whose lanes cover different
// 256-element blocks of the same row therefore touches addresses 36 bytes apart,
// so each 4-byte load instruction is spread over ~10 L1 sectors and retires in
// ~10 wavefronts instead of 4. The ternary GEMV is LSU-bound long before it is
// bandwidth-bound - PrismML measured exactly this cap for PTQ1_0 and got 2.2-3.3x
// from the layout alone - so the fix is a layout, not a faster dot product.
//
// Layout of ONE activation column (ne0 elements, ne0 % 256 == 0, nblk = ne0/256):
//   [0, 256*nblk)         16 planes of nblk*16 bytes. Plane p, block kb holds the
//                         quants of elements kb*256 + 16*p .. +15, one byte each, so
//                         the lane that owns (row, kb) issues 16 aligned 16-byte
//                         loads and neighbouring lanes read neighbouring units.
//   [256*nblk, 288*nblk)  32 bytes per block = 8 half2(d, isum), one per
//                         32-element activation sub-block. d is the sub-block
//                         scale, isum the EXACT integer sum of its 32 quants kept
//                         as 16 raw bits in the high half (|isum| <= 4064 does not
//                         fit a half exactly, so it never becomes a float); see
//                         tq_pt_pack_ds / tq_pt_ds_isum.
//   total = 288*nblk = ne0*9/8 bytes = exactly the AoS block_q8_1 column stride,
//   so every stride the launcher already computes in block_q8_1 units
//   (stride_col_y, stride_channel_y, stride_sample_y) stays valid multiplied by
//   sizeof(block_q8_1), and the pool allocation does not change size.
//
// Arithmetic. Digits stay unsigned (0..2 for TQ1_0, 0..3 for TQ2_0) and are
// folded with dp4a against the raw quants; once per 32 elements
//     acc += (d_w * d_a) * (float)(sum(digit*q) - isum)
// which is algebraically identical to the AoS path's centered symbols
//     acc += (d_w * d_a) * (float)sum((digit-1)*q)
// and yields the same per-sub-block product bit for bit (both are exact integers
// before the single float multiply), so the two paths agree to within the order
// in which the 8 sub-block products are summed. Dropping the -1 bias also removes
// one __vsub4 per four weights. The multiply by the scales stays at the end of the
// sub-block, as required; nothing is dequantized and nothing is unpacked to float.
//
// Geometry. One lane owns one (row, 256-element block) pair: kb = lane % kb_lanes,
// row = row0 + warp*rows_per_warp + lane / kb_lanes, and the kb loop strides by
// kb_lanes for the (rare) case of more than 32 blocks per row. Rows are reduced
// with a butterfly shuffle over kb_lanes lanes (a power of two, enforced by the
// host side check), so no shared memory and no __syncthreads are needed.
// =============================================================================

#define TQ_PT_QK     256   // elements per ternary super-block (QK_K)
#define TQ_PT_SUB      8   // 32-element activation sub-blocks per super-block
#define TQ_PT_PLANES  16   // one byte plane per 16 elements (TQ_PT_QK/16)
#define TQ_PT_NWARPS   4   // plain MUL_MAT kernel width (128 threads)
#define TQ_PT_WARP    32   // PT kernels are written for 32-lane warps

// One PT column of ne0 elements, in bytes. 288 per 256-element block.
static constexpr __host__ __device__ size_t tq_pt_col_bytes(const int64_t ne0) {
    return (size_t) (ne0 / TQ_PT_QK) * (TQ_PT_PLANES*16 + TQ_PT_SUB*4);
}
static_assert(tq_pt_col_bytes(256) == 288, "PT column must match the AoS q8_1 stride");
static_assert(tq_pt_col_bytes(256) == 256*sizeof(block_q8_1)/QK8_1, "PT/AoS stride mismatch");

// The kernels read the planes with 16-byte loads, so every column base has to be 16-byte aligned.
// The launcher hands the stride over in block_q8_1 units and ne0 is padded to MATRIX_ROW_PADDING,
// which makes the byte stride (MATRIX_ROW_PADDING/QK8_1)*sizeof(block_q8_1) per padded row - lock
// that in here instead of discovering it as a misaligned address fault on device.
static_assert((MATRIX_ROW_PADDING/QK8_1)*sizeof(block_q8_1) % 16 == 0, "PT int4 loads would be misaligned");
static_assert(tq_pt_col_bytes(TQ_PT_QK) % 16 == 0, "PT planes would be misaligned");

// Runtime A/B switch: GGML_CUDA_NO_TQ_PT=1 forces the old AoS path without a rebuild.
static inline bool tq_pt_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_NO_TQ_PT");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// (d, isum) travels as one half2: d is a real half, isum is 16 raw bits in the high half (it can
// reach +-4064, which a half cannot represent exactly, so it must not go through float). Only
// make_half2/__low2float are used - the __short_as_half/__half_as_short bit casts would work too,
// but nothing else in ggml-cuda uses them, and a union is plain C++ that nvcc type-puns as
// documented. Two helpers so the packing exists in exactly one place.
static __device__ __forceinline__ half2 tq_pt_pack_ds(const float d, const int isum) {
    union { half2 h; unsigned int u; } v;
    v.h = make_half2(d, 0.0f);
    v.u = (v.u & 0x0000FFFFu) | (((unsigned int) (unsigned short) (short) isum) << 16);
    return v.h;
}

static __device__ __forceinline__ int tq_pt_ds_isum(const half2 ds) {
    union { half2 h; unsigned int u; } v;
    v.h = ds;
    return (int) (short) ((v.u >> 16) & 0xFFFFu);
}

static __device__ __forceinline__ int tq_pt_word(const int4 v, const int t) {
    return t == 0 ? v.x : (t == 1 ? v.y : (t == 2 ? v.z : v.w));
}

// Butterfly reduction confined to groups of `width` lanes (width a power of two).
static __device__ __forceinline__ float tq_pt_warp_reduce(float v, const int width) {
#pragma unroll
    for (int off = TQ_PT_WARP/2; off > 0; off >>= 1) {
        if (off < width) {
            v += __shfl_xor_sync(0xffffffff, v, off, TQ_PT_WARP); // explicit width: hip.h maps this to a 4-arg macro
        }
    }
    return v;
}

// Digits 0..2 of the four bytes of one weight word at ternary position n, selected
// by the word index w inside the 256-element block:
//   w  0..39 : qs[4*(w%8)],  n = w/8          (5 digits per byte, 8 words per digit)
//   w 40..59 : qs[32+4*(w%4)], n = (w-40)/4   (4 words per digit, 16 bytes left)
//   w 60..63 : qh[0..3],      n = w-60        (4 digits per byte, one word each)
// w is a compile time constant in every call below, so all three branches and the
// array indices fold away.
static __device__ __forceinline__ int tq1_0_pt_digits4(const int * w1, const int * w2, const int w3, const int w) {
    if (w < 40) {
        return ggml_cuda_tq1_0_digits4(w1[w & 7], ggml_cuda_tq1_0_pow3(w >> 3));
    } else if (w < 60) {
        return ggml_cuda_tq1_0_digits4(w2[w & 3], ggml_cuda_tq1_0_pow3((w - 40) >> 2));
    }
    return ggml_cuda_tq1_0_digits4(w3, ggml_cuda_tq1_0_pow3(w - 60));
}

// One 256-element TQ1_0 block against ncols_dst PT activation columns, block kb.
// ycol0 points at the first byte of activation column 0, col_stride_bytes at the
// byte distance between two columns (stride_col_y * sizeof(block_q8_1)).
template <int ncols_dst>
static __device__ __forceinline__ void tq1_0_pt_dot_block(
        const char * __restrict__ vx_block, const char * __restrict__ ycol0,
        const int nblk, const int kb, const size_t col_stride_bytes,
        float * __restrict__ acc) {

    const block_tq1_0 * bq = (const block_tq1_0 *) vx_block;

    // block_tq1_0 is 54 bytes, so odd blocks are only 2-byte aligned: assemble every
    // weight word from two 16-bit loads (get_int_b2), exactly like the AoS dot does.
    int w1[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        w1[t] = get_int_b2(bq->qs, t);
    }
    int w2[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        w2[t] = get_int_b2(bq->qs, 8 + t);
    }
    const int   w3 = get_int_b2(bq->qh, 0);
    const float dw = bq->d;

    // 18 aligned 16-byte loads per column cover the whole block: 16 planes + 32 bytes
    // of (d, isum) pairs.
    int4 alo[ncols_dst];
    int4 ahi[ncols_dst];

    // 8 half2(d, isum) per block, contiguous: one cache line per column
    const half2 * dsp[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        dsp[j] = (const half2 *) (ycol0 + j*col_stride_bytes
                                + (size_t) nblk*TQ_PT_PLANES*16 + (size_t) kb*TQ_PT_SUB*4);
    }

    int sumi[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumi[j] = 0;
    }

#pragma unroll
    for (int s = 0; s < TQ_PT_SUB; ++s) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const char * ycol = ycol0 + j*col_stride_bytes;
            alo[j] = *(const int4 *) (ycol + (size_t) ((2*s    )*nblk + kb)*16);
            ahi[j] = *(const int4 *) (ycol + (size_t) ((2*s + 1)*nblk + kb)*16);
        }

        // plane 2*s: words 8s+0 .. 8s+3
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int d4 = tq1_0_pt_digits4(w1, w2, w3, 8*s + t);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumi[j] = ggml_cuda_dp4a(d4, tq_pt_word(alo[j], t), sumi[j]);
            }
        }
        // plane 2*s+1: words 8s+4 .. 8s+7
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int d4 = tq1_0_pt_digits4(w1, w2, w3, 8*s + 4 + t);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumi[j] = ggml_cuda_dp4a(d4, tq_pt_word(ahi[j], t), sumi[j]);
            }
        }

        // one fold per 32 elements: the only place a scale is multiplied in
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const half2 dsv  = dsp[j][s];
            const float d_a  = __low2float(dsv);
            const int   isum = tq_pt_ds_isum(dsv);
            acc[j] += (dw * d_a) * (float) (sumi[j] - isum);
            sumi[j] = 0;
        }
    }
}

// One 256-element TQ2_0 block against ncols_dst PT activation columns, block kb. Same PT
// activation layout and the same fold as tq1_0_pt_dot_block above; only the weight side differs,
// and it is simpler. With e = 128*c + 32*l + m (common.cuh) the sub-block s = elements
// 32s..32s+31 has c = s/4 and l = s%4 for ALL of its elements, so its 32 weights are the 32
// contiguous bytes qs[32*(s/4) .. +31] read at one single shift: 8 words, one shift, one mask.
// Plane 2*s of the activation column holds elements 32s..32s+15 (weight words 0..3 of those 8)
// and plane 2*s+1 holds elements 32s+16..32s+31 (weight words 4..7), which is why the two int4
// loads below pair up with w[t] and w[4 + t].
template <int ncols_dst>
static __device__ __forceinline__ void tq2_0_pt_dot_block(
        const char * __restrict__ vx_block, const char * __restrict__ ycol0,
        const int nblk, const int kb, const size_t col_stride_bytes,
        float * __restrict__ acc) {

    const block_tq2_0 * bq = (const block_tq2_0 *) vx_block;

    // block_tq2_0 is 66 bytes, so odd blocks are only 2-byte aligned: assemble every weight word
    // from two 16-bit loads (get_int_b2), exactly like the AoS dot does. All 64 bytes of qs are
    // read up front (16 words); the sub-block loop below then only shifts and masks, and every
    // index into w[] is a compile time constant once the loop is unrolled.
    int w[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        w[t] = get_int_b2(bq->qs, t);
    }
    const float dw = bq->d;

    // 18 aligned 16-byte loads per column cover the whole block: 16 planes + 32 bytes
    // of (d, isum) pairs.
    int4 alo[ncols_dst];
    int4 ahi[ncols_dst];

    // 8 half2(d, isum) per block, contiguous: one cache line per column
    const half2 * dsp[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        dsp[j] = (const half2 *) (ycol0 + j*col_stride_bytes
                                + (size_t) nblk*TQ_PT_PLANES*16 + (size_t) kb*TQ_PT_SUB*4);
    }

    int sumi[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumi[j] = 0;
    }

#pragma unroll
    for (int s = 0; s < TQ_PT_SUB; ++s) {
        const int woff  = 8*(s >> 2);   // first of the 8 weight words of this sub-block
        const int shift = 2*(s & 3);    // their common bit pair position

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const char * ycol = ycol0 + j*col_stride_bytes;
            alo[j] = *(const int4 *) (ycol + (size_t) ((2*s    )*nblk + kb)*16);
            ahi[j] = *(const int4 *) (ycol + (size_t) ((2*s + 1)*nblk + kb)*16);
        }

        // plane 2*s: elements 32s .. 32s+15, weight words woff+0 .. woff+3
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int d4 = ggml_cuda_tq2_0_digits4(w[woff + t], shift);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumi[j] = ggml_cuda_dp4a(d4, tq_pt_word(alo[j], t), sumi[j]);
            }
        }
        // plane 2*s+1: elements 32s+16 .. 32s+31, weight words woff+4 .. woff+7
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int d4 = ggml_cuda_tq2_0_digits4(w[woff + 4 + t], shift);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumi[j] = ggml_cuda_dp4a(d4, tq_pt_word(ahi[j], t), sumi[j]);
            }
        }

        // one fold per 32 elements: the only place a scale is multiplied in
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const half2 dsv  = dsp[j][s];
            const float d_a  = __low2float(dsv);
            const int   isum = tq_pt_ds_isum(dsv);
            acc[j] += (dw * d_a) * (float) (sumi[j] - isum);
            sumi[j] = 0;
        }
    }
}

// Plain MUL_MAT (no ids, no fusion). Grid: (rows/rows_per_cta, nchannels_dst, nsamples_dst).
template <ggml_type type, int ncols_dst>
static __global__ void __launch_bounds__(TQ_PT_NWARPS*TQ_PT_WARP, 1) mul_mat_vec_tq_pt(
        const void * vx_ptr, const void * vy_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t nblk_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t channel_ratio, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst, const uint32_t sample_ratio) {

    const char * GGML_CUDA_RESTRICT vx = (const char *) vx_ptr;
    const char * GGML_CUDA_RESTRICT vy = (const char *) vy_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int bs = ggml_cuda_type_traits<type>::bs;
    static_assert(type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0, "mmvq-tq-pt: ternary types only");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int kb_lanes      = min((int) (ncols_x / TQ_PT_QK), TQ_PT_WARP);
    const int rows_per_warp = TQ_PT_WARP / kb_lanes;
    const int rows_per_cta  = TQ_PT_NWARPS * rows_per_warp;

    const int row0      = rows_per_cta*blockIdx.x;
    const int row_local = warp*rows_per_warp + lane/kb_lanes;
    const int row       = row0 + row_local;
    const int kb0       = lane % kb_lanes;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    // Once per thread, not per element: a plain 32-bit divide costs nothing here and keeps this
    // header independent of the fastdiv tables in common.cuh.
    const uint32_t channel_x   = channel_dst / channel_ratio;
    const uint32_t channel_y   = channel_dst;
    const uint32_t sample_x    = sample_dst / sample_ratio;
    const uint32_t sample_y    = sample_dst;

    float acc[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        acc[j] = 0.0f;
    }

    ggml_cuda_pdl_sync();

    if (row < (int) nrows_x) {
        const char * ycol  = vy + ((size_t) sample_y*stride_sample_y + (size_t) channel_y*stride_channel_y)*sizeof(block_q8_1);
        const size_t xoff  = ((size_t) sample_x*stride_sample_x + (size_t) channel_x*stride_channel_x
                            + (size_t) row*stride_row_x) * bs;
        const size_t cstep = (size_t) stride_col_y * sizeof(block_q8_1);

        for (int kb = kb0; kb < (int) (ncols_x / TQ_PT_QK); kb += kb_lanes) {
            if constexpr (type == GGML_TYPE_TQ1_0) {
                tq1_0_pt_dot_block<ncols_dst>(vx + xoff + (size_t) kb*bs, ycol, (int) nblk_y, kb, cstep, acc);
            } else {
                tq2_0_pt_dot_block<ncols_dst>(vx + xoff + (size_t) kb*bs, ycol, (int) nblk_y, kb, cstep, acc);
            }
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        acc[j] = tq_pt_warp_reduce(acc[j], kb_lanes);
    }

    if (kb0 == 0 && row < (int) nrows_x) {
        dst += (size_t) sample_dst*stride_sample_dst + (size_t) channel_dst*stride_channel_dst + row;
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            dst[j*stride_col_dst] = acc[j];
        }
    }
}

// MUL_MAT_ID. One warp per expert slot (threadIdx.y), mirroring mul_mat_vec_q_moe:
// channel_dst is the token, threadIdx.y the slot, ids[token + slot*ids_stride] the expert.
// Grid: (rows/rows_per_block, nchannels_dst).
template <ggml_type type>
static __global__ void __launch_bounds__(TQ_PT_MAX_SLOTS*TQ_PT_WARP, 1) mul_mat_vec_tq_pt_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t nchannels_y,
        const uint32_t nblk_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t ncols_dst, const uint32_t ids_stride) {

    const char    * GGML_CUDA_RESTRICT vx  = (const char *) vx_ptr;
    const char    * GGML_CUDA_RESTRICT vy  = (const char *) vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int bs = ggml_cuda_type_traits<type>::bs;
    static_assert(type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0, "mmvq-tq-pt: ternary types only");

    const int lane = threadIdx.x;
    const int slot = threadIdx.y;

    const int kb_lanes       = min((int) (ncols_x / TQ_PT_QK), TQ_PT_WARP);
    const int rows_per_block = TQ_PT_WARP / kb_lanes;

    const int row0 = rows_per_block*blockIdx.x;
    const int row  = row0 + lane/kb_lanes;
    const int kb0  = lane % kb_lanes;

    const uint32_t channel_dst = blockIdx.y;

    if (slot >= (int) ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();

    const uint32_t channel_x = ids[channel_dst + (size_t) slot*ids_stride];
    const uint32_t channel_y = channel_dst % nchannels_y;

    float acc = 0.0f;

    if (row < (int) nrows_x) {
        const char * ycol  = vy + ((size_t) channel_y*stride_channel_y + (size_t) slot*stride_col_y)*sizeof(block_q8_1);
        const size_t xoff  = ((size_t) channel_x*stride_channel_x + (size_t) row*stride_row_x) * bs;
        const size_t cstep = (size_t) stride_col_y * sizeof(block_q8_1);

        for (int kb = kb0; kb < (int) (ncols_x / TQ_PT_QK); kb += kb_lanes) {
            if constexpr (type == GGML_TYPE_TQ1_0) {
                tq1_0_pt_dot_block<1>(vx + xoff + (size_t) kb*bs, ycol, (int) nblk_y, kb, cstep, &acc);
            } else {
                tq2_0_pt_dot_block<1>(vx + xoff + (size_t) kb*bs, ycol, (int) nblk_y, kb, cstep, &acc);
            }
        }
    }

    ggml_cuda_pdl_lc();

    acc = tq_pt_warp_reduce(acc, kb_lanes);

    if (kb0 == 0 && row < (int) nrows_x) {
        dst[(size_t) channel_dst*stride_channel_dst + (size_t) slot*stride_col_dst + row] = acc;
    }
}

// --------------------------------------------------------------------------- //
// host side
// --------------------------------------------------------------------------- //

// PT is only taken when it cannot change semantics: TQ1_0/TQ2_0 weights, no fusion
// (the PT kernel has no gate/bias epilogue), a super-block aligned K, a power of
// two number of super-blocks per row (the shuffle reduction needs it) and 32-lane
// warps. Everything else keeps using mul_mat_vec_q with AoS activations.
static inline bool mul_mat_vec_tq_pt_supported(
        const ggml_type type, const int64_t ncols_x, const bool has_fusion, const int warp_size) {
    if (!tq_pt_enabled() || has_fusion || warp_size != TQ_PT_WARP) {
        return false;
    }
    if (type != GGML_TYPE_TQ1_0 && type != GGML_TYPE_TQ2_0) {
        return false;
    }
    if (ncols_x % TQ_PT_QK != 0) {
        return false;
    }
    const int64_t bpr = ncols_x / TQ_PT_QK;
    return bpr >= 1 && (bpr & (bpr - 1)) == 0; // power of two
}

template <ggml_type type, int c_ncols_dst>
static void mul_mat_vec_tq_pt_launch(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int nblk_y,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_dst, const int stride_channel_x, const int stride_channel_y,
        const int stride_channel_dst, const uint32_t channel_ratio, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint32_t sample_ratio, cudaStream_t stream) {

    GGML_ASSERT((stride_col_y*sizeof(block_q8_1)) % 16 == 0 && "PT column base must be 16-byte aligned");

    const int kb_lanes      = std::min(ncols_x / TQ_PT_QK, TQ_PT_WARP);
    const int rows_per_warp = TQ_PT_WARP / kb_lanes;
    const int rows_per_cta  = TQ_PT_NWARPS * rows_per_warp;

    const dim3 block_nums((nrows_x + rows_per_cta - 1) / rows_per_cta, nchannels_dst, nsamples_dst);
    const dim3 block_dims(TQ_PT_WARP, TQ_PT_NWARPS, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_tq_pt<type, c_ncols_dst>, launch_params,
        vx, vy, dst, (uint32_t) ncols_x, (uint32_t) nrows_x, (uint32_t) nblk_y,
        (uint32_t) stride_row_x, (uint32_t) stride_col_y, (uint32_t) stride_col_dst,
        (uint32_t) stride_channel_x, (uint32_t) stride_channel_y, (uint32_t) stride_channel_dst,
        channel_ratio, (uint32_t) stride_sample_x, (uint32_t) stride_sample_y,
        (uint32_t) stride_sample_dst, sample_ratio);
}

template <ggml_type type>
static void mul_mat_vec_tq_pt_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int nblk_y,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_dst, const int stride_channel_x, const int stride_channel_y,
        const int stride_channel_dst, const uint32_t channel_ratio, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint32_t sample_ratio, cudaStream_t stream) {

    switch (ncols_dst) {
        case 1:
            mul_mat_vec_tq_pt_launch<type, 1>(vx, vy, dst, ncols_x, nrows_x, nblk_y, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        case 2:
            mul_mat_vec_tq_pt_launch<type, 2>(vx, vy, dst, ncols_x, nrows_x, nblk_y, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        case 3:
            mul_mat_vec_tq_pt_launch<type, 3>(vx, vy, dst, ncols_x, nrows_x, nblk_y, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        case 4:
            mul_mat_vec_tq_pt_launch<type, 4>(vx, vy, dst, ncols_x, nrows_x, nblk_y, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        case 5:
            mul_mat_vec_tq_pt_launch<type, 5>(vx, vy, dst, ncols_x, nrows_x, nblk_y, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        case 6:
            mul_mat_vec_tq_pt_launch<type, 6>(vx, vy, dst, ncols_x, nrows_x, nblk_y, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        case 7:
            mul_mat_vec_tq_pt_launch<type, 7>(vx, vy, dst, ncols_x, nrows_x, nblk_y, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        case 8:
            mul_mat_vec_tq_pt_launch<type, 8>(vx, vy, dst, ncols_x, nrows_x, nblk_y, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

template <ggml_type type>
static void mul_mat_vec_tq_pt_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int nchannels_y, const int ncols_dst, const int nblk_y,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_dst, const int stride_channel_x, const int stride_channel_y,
        const int stride_channel_dst, const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT((stride_col_y*sizeof(block_q8_1)) % 16 == 0 && "PT column base must be 16-byte aligned");

    const int kb_lanes       = std::min(ncols_x / TQ_PT_QK, TQ_PT_WARP);
    const int rows_per_block = TQ_PT_WARP / kb_lanes;

    const dim3 block_nums((nrows_x + rows_per_block - 1) / rows_per_block, nchannels_dst);
    const dim3 block_dims(TQ_PT_WARP, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_tq_pt_moe<type>, launch_params,
        vx, vy, ids, dst, (uint32_t) ncols_x, (uint32_t) nrows_x, (uint32_t) nchannels_y,
        (uint32_t) nblk_y, (uint32_t) stride_row_x, (uint32_t) stride_col_y, (uint32_t) stride_col_dst,
        (uint32_t) stride_channel_x, (uint32_t) stride_channel_y, (uint32_t) stride_channel_dst,
        (uint32_t) ncols_dst, (uint32_t) ids_stride);
}

// Entry point used by ggml_cuda_mul_mat_vec_q. Argument order matches
// mul_mat_vec_q_switch_type so the two calls stay readable side by side.
static void mul_mat_vec_tq_pt(
        const void * vx, const ggml_type type, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x,
        const int stride_sample_y, const int stride_sample_dst, const int ids_stride,
        const int nblk_y, cudaStream_t stream) {

    GGML_ASSERT(mul_mat_vec_tq_pt_supported(type, ncols_x, false, TQ_PT_WARP));
    GGML_ASSERT(ncols_dst >= 1 && ncols_dst <= TQ_PT_MAX_SLOTS);

    // Plain divisors (see the kernels): used once per thread, so no fastdiv tables. The ratios
    // mirror what mul_mat_vec_q_switch_ncols_dst feeds to the AoS path.
    //
    // MUL_MAT_ID indexes the channel dimension through ids, so nchannels_dst/nchannels_x is
    // meaningless there (experts=256 vs expert slots=8 would never divide). Upstream guards the
    // same way: mmvf.cu has GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0) and mmvq.cu
    // computes channel_ratio_fd = ids ? make_uint3(0,0,0) : init_fastdiv_values(...).
    GGML_ASSERT(ids || (nchannels_x >= 1 && nchannels_dst % nchannels_x == 0 && nchannels_dst >= nchannels_x));
    GGML_ASSERT(nsamples_x  >= 1 && nsamples_dst  % nsamples_x  == 0 && nsamples_dst  >= nsamples_x);
    const uint32_t channel_ratio = ids ? 1u : (uint32_t) (nchannels_dst / nchannels_x);
    const uint32_t sample_ratio  = (uint32_t) (nsamples_dst / nsamples_x);

    if (ids) {
        switch (type) {
            case GGML_TYPE_TQ1_0:
                mul_mat_vec_tq_pt_moe_launch<GGML_TYPE_TQ1_0>(
                    vx, vy, ids, dst, ncols_x, nrows_x, nchannels_y, ncols_dst, nblk_y,
                    stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, stream);
                break;
            case GGML_TYPE_TQ2_0:
                mul_mat_vec_tq_pt_moe_launch<GGML_TYPE_TQ2_0>(
                    vx, vy, ids, dst, ncols_x, nrows_x, nchannels_y, ncols_dst, nblk_y,
                    stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, stream);
                break;
            default:
                GGML_ABORT("fatal error");
        }
        return;
    }

    switch (type) {
        case GGML_TYPE_TQ1_0:
            mul_mat_vec_tq_pt_switch_ncols<GGML_TYPE_TQ1_0>(
                vx, vy, dst, ncols_x, nrows_x, ncols_dst, nblk_y,
                stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        case GGML_TYPE_TQ2_0:
            mul_mat_vec_tq_pt_switch_ncols<GGML_TYPE_TQ2_0>(
                vx, vy, dst, ncols_x, nrows_x, ncols_dst, nblk_y,
                stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, sample_ratio, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
