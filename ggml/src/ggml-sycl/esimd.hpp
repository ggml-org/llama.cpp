#ifndef GGML_SYCL_ESIMD_HPP
#define GGML_SYCL_ESIMD_HPP

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "common.hpp"

namespace ggml_sycl_esimd {

constexpr int GGML_SYCL_DMMV_ESIMD_WG_SIZE = 4;

//
// Shared ESIMD building blocks for the reordered K-quant dequantize-matvec
// kernels.
//
// The reordered K-quant ESIMD matvec kernels share one skeleton: per super-block,
// load a 256-float activation slice, load one weight block, dequantize it into 8
// chunks of 32 and MAC each chunk against the matching activation slice, then
// reduce and run a lane-0 epilogue.
//
// Each K-quant kernel emits exactly 8 chunks of 32 mapping to activation slices
// 0..7, so the per-block work is captured by esimd_reorder_q_traits<T>::mac_pair,
// which dequantizes two weight blocks and MACs both against a shared activation
// vector with the two FMA chains interleaved (co-scheduled to hide FMA latency).
// The "pair" is the (row0,row1) row pair owned by one work-group, so the
// layout+dequant is written once per quant type here.
//

template <ggml_type T> struct esimd_reorder_q_traits;

// build a 32-lane vector whose low 16 lanes are `lo` and high 16 are `hi`
// (a super-chunk splits into two 16-wide halves with distinct scale/min codes).
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 32> splat_lo_hi(float lo, float hi) {
    using namespace sycl::ext::intel::esimd;
    simd<float, 32> v;
    v.select<16, 1>(0)  = lo;
    v.select<16, 1>(16) = hi;
    return v;
}

// unpack one block of Q4_K/Q5_K scale/min codes (get_scale_min_k4 layout) into 8
// float scales (dall * sc) and 8 float mins (-dmin * m); the min carries the
// negation so the dequant epilogue adds.
static ESIMD_INLINE void unpack_scale_min_k4(
        sycl::ext::intel::esimd::simd<uint8_t, 12> scales, float dall, float dmin,
        sycl::ext::intel::esimd::simd<float, 8> & scale_f,
        sycl::ext::intel::esimd::simd<float, 8> & min_f) {
    using namespace sycl::ext::intel::esimd;
    simd<uint8_t, 8> sc = 0;
    simd<uint8_t, 8> m  = 0;
    simd<uint8_t, 4> scale_lo = scales.select<4, 1>(0);
    simd<uint8_t, 4> min_lo   = scales.select<4, 1>(4);
    simd<uint8_t, 4> hi_bits  = scales.select<4, 1>(8);
    sc.select<4, 1>(0) = scale_lo & simd<uint8_t, 4>(0x3F);
    sc.select<4, 1>(4) = (hi_bits & simd<uint8_t, 4>(0x0F)) |
                         ((scale_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    m.select<4, 1>(0)  = min_lo & simd<uint8_t, 4>(0x3F);
    m.select<4, 1>(4)  = (hi_bits >> simd<uint8_t, 4>(4)) |
                         ((min_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    scale_f = convert<float>(sc) * dall;
    min_f   = convert<float>(m) * (-dmin);
}

// ---------------------------------------------------------------------------
// Q2_K, SOA reorder layout produced by reorder_qw_q2_k:
//   [qs: nb*(QK_K/4)] [scales: nb*(QK_K/16)] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
//
// 2 bits per weight. The 8 output chunks of 32 (matching dequantize_row_q2_K)
// map to super-chunk s (0..7): byte base 32*(s/4) into the 64-byte qs array,
// bit shift 2*(s%4); the low 16 lanes use scales[2s], the high 16 use
// scales[2s+1], with dl = d*(sc & 0xF), ml = dmin*(sc >> 4), deq = dl*q - ml.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q2_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * scales = qs + nb * (QK_K / 4);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * (QK_K / 16));
        return { qs, scales, dm };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 64> qs_a     = block_load<uint8_t, 64>(pa.qs + bia * (QK_K / 4));
        simd<uint8_t, 64> qs_b     = 0;
        simd<uint8_t, 16> scales_a = block_load<uint8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<uint8_t, 16> scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 64>(pb.qs + bib * (QK_K / 4));
            scales_b = block_load<uint8_t, 16>(pb.scales + bib * (QK_K / 16));
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        // per-chunk scale (d * (sc & 0xF)) and min (-dmin * (sc >> 4)), all 16 codes;
        // min carries the negation so the dequant epilogue adds (matches Q4_K/Q5_K)
        simd<float, 16> scale_f_a = convert<float>(scales_a & simd<uint8_t, 16>(0x0F)) * dall_a;
        simd<float, 16> min_f_a   = convert<float>(scales_a >> simd<uint8_t, 16>(4))  * (-dmin_a);
        simd<float, 16> scale_f_b = convert<float>(scales_b & simd<uint8_t, 16>(0x0F)) * dall_b;
        simd<float, 16> min_f_b   = convert<float>(scales_b >> simd<uint8_t, 16>(4))  * (-dmin_b);

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int     byte_base = 32 * (s / 4);
            const uint8_t shift     = (uint8_t) (2 * (s % 4));
            simd<float, 32> y_s = y_vec.select<32, 1>(s * 32);

            simd<uint8_t, 32> qa = (qs_a.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3);
            simd<uint8_t, 32> qb = (qs_b.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3);

            const float scale_a_lo = scale_f_a[2 * s + 0];
            const float scale_a_hi = scale_f_a[2 * s + 1];
            const float min_a_lo   = min_f_a[2 * s + 0];
            const float min_a_hi   = min_f_a[2 * s + 1];
            const float scale_b_lo = scale_f_b[2 * s + 0];
            const float scale_b_hi = scale_f_b[2 * s + 1];
            const float min_b_lo   = min_f_b[2 * s + 0];
            const float min_b_hi   = min_f_b[2 * s + 1];

            simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
            simd<float, 32> min_vec_a   = splat_lo_hi(min_a_lo, min_a_hi);
            simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);
            simd<float, 32> min_vec_b   = splat_lo_hi(min_b_lo, min_b_hi);

            simd<float, 32> deq_a = convert<float>(qa) * scale_vec_a + min_vec_a;
            simd<float, 32> deq_b = convert<float>(qb) * scale_vec_b + min_vec_b;

            acc_a += y_s * deq_a;
            acc_b += y_s * deq_b;
        }
    }
};

// ---------------------------------------------------------------------------
// Q3_K, SOA reorder layout produced by reorder_qw_q3_k:
//   [qs: nb*(QK_K/4)] [hmask: nb*(QK_K/8)] [scales: nb*12] [d: nb*sizeof(half)]
// with nb = nrows*num_blocks_per_row. Single super-block scale d, no dmin.
//
// 3 bits per weight: 2 low bits in qs, 1 high bit in hmask. The 8 output chunks
// of 32 (matching dequantize_row_q3_K) map to super-chunk s (0..7): byte base
// 32*(s/4) into the 64-byte qs array, bit shift 2*(s%4); the low 16 lanes use
// scale code 2s, the high 16 use 2s+1. hmask is a 32-byte array (like Q5_K's
// qh) where chunk s uses bit s of the same 32 bytes, but INVERTED: the value is
// (q & 3) - (hmask_bit_set ? 0 : 4), i.e. (q & 3) + 4*bit - 4.
//
// The 16 6-bit scale codes are packed into 12 bytes (get_scale_min layout for
// Q3_K): low nibbles from bytes 0..7, high 2 bits from bytes 8..11 shifted by
// 0/2/4/6; the dequant scale is d * (code - 32).
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q3_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    hmask;
        const uint8_t *    scales;
        const sycl::half * d;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * hmask  = qs + nb * (QK_K / 4);
        const uint8_t * scales = hmask + nb * (QK_K / 8);
        const sycl::half * d   = (const sycl::half *) (scales + nb * 12);
        return { qs, hmask, scales, d };
    }

    // unpack the 12 packed bytes into 16 6-bit scale codes (dequantize_row_q3_K
    // aux layout), returned as float scale = d * (code - 32).
    // done with wide (8/16-lane) ops rather than four 4-lane groups.
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 16> unpack_scales(
            sycl::ext::intel::esimd::simd<uint8_t, 12> in, float d) {
        using namespace sycl::ext::intel::esimd;

        // low 6-bit part: codes 0..7 = low nibble of bytes 0..7,
        //                 codes 8..15 = high nibble of bytes 0..7
        simd<uint8_t, 8>  lo8 = in.select<8, 1>(0);
        simd<uint8_t, 16> code;
        code.select<8, 1>(0) = lo8 & simd<uint8_t, 8>(0x0F);
        code.select<8, 1>(8) = lo8 >> simd<uint8_t, 8>(4);

        // high 2-bit part: bytes 8..11 replicated 4x, group g (0..3) shifted 2*g
        simd<uint8_t, 16> hib;
        hib.select<4, 1>(0)  = in.select<4, 1>(8);
        hib.select<4, 1>(4)  = in.select<4, 1>(8);
        hib.select<4, 1>(8)  = in.select<4, 1>(8);
        hib.select<4, 1>(12) = in.select<4, 1>(8);
        simd<uint8_t, 16> hshift;
        hshift.select<4, 1>(0)  = 0;
        hshift.select<4, 1>(4)  = 2;
        hshift.select<4, 1>(8)  = 4;
        hshift.select<4, 1>(12) = 6;
        hib = (hib >> hshift) & simd<uint8_t, 16>(0x03);

        code = code | (hib << simd<uint8_t, 16>(4));
        return (convert<float>(code) - 32.0f) * d;
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 64> qs_a     = block_load<uint8_t, 64>(pa.qs + bia * (QK_K / 4));
        simd<uint8_t, 64> qs_b     = 0;
        simd<uint8_t, 32> hmask_a  = block_load<uint8_t, 32>(pa.hmask + bia * (QK_K / 8));
        simd<uint8_t, 32> hmask_b  = 0;
        simd<uint8_t, 12> scales_a = block_load<uint8_t, 12>(pa.scales + bia * 12);
        simd<uint8_t, 12> scales_b = 0;

        const float d_a = (float) pa.d[bia];
        float d_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 64>(pb.qs + bib * (QK_K / 4));
            hmask_b  = block_load<uint8_t, 32>(pb.hmask + bib * (QK_K / 8));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * 12);
            d_b = (float) pb.d[bib];
        }

        simd<float, 16> scale_f_a = unpack_scales(scales_a, d_a);
        simd<float, 16> scale_f_b = unpack_scales(scales_b, d_b);

#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int     byte_base = 32 * (s / 4);
            const uint8_t shift     = (uint8_t) (2 * (s % 4));
            simd<float, 32> y_s = y_vec.select<32, 1>(s * 32);

            // 2 low bits from qs, high bit from hmask (bit s of the same 32 bytes);
            // value = (q & 3) + 4*bit - 4  (inverted hmask: subtract 4 when bit clear).
            // merge in the integer domain: q3 = (q & 3) | (bit << 2) in {0..7},
            // then a single convert + subtract yields q3 - 4 (one convert, not two)
            simd<uint16_t, 32> q3_a = convert<uint16_t>(
                    (qs_a.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3));
            q3_a |= convert<uint16_t>(
                    ((hmask_a >> simd<uint8_t, 32>((uint8_t) s)) & simd<uint8_t, 32>(1)) << simd<uint8_t, 32>(2));
            simd<uint16_t, 32> q3_b = convert<uint16_t>(
                    (qs_b.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3));
            q3_b |= convert<uint16_t>(
                    ((hmask_b >> simd<uint8_t, 32>((uint8_t) s)) & simd<uint8_t, 32>(1)) << simd<uint8_t, 32>(2));

            simd<float, 32> qf_a = convert<float>(q3_a) - 4.0f;
            simd<float, 32> qf_b = convert<float>(q3_b) - 4.0f;

            const float scale_a_lo = scale_f_a[2 * s + 0];
            const float scale_a_hi = scale_f_a[2 * s + 1];
            const float scale_b_lo = scale_f_b[2 * s + 0];
            const float scale_b_hi = scale_f_b[2 * s + 1];

            simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
            simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

            simd<float, 32> deq_a = qf_a * scale_vec_a;
            simd<float, 32> deq_b = qf_b * scale_vec_b;

            acc_a += y_s * deq_a;
            acc_b += y_s * deq_b;
        }
    }
};

// ---------------------------------------------------------------------------
// Q4_K, SOA reorder layout produced by reorder_qw_q4_k:
//   [qs: nb*(QK_K/2)] [scales: nb*K_SCALE_SIZE] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q4_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * scales = qs + nb * (QK_K / 2);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * K_SCALE_SIZE);
        return { qs, scales, dm };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * K_SCALE_SIZE);
        simd<uint8_t, 12>  scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * K_SCALE_SIZE);
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        simd<float, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int sb = 0; sb < 8; sb += 2) {
            const int q_offset = sb * 16;
            simd<float, 32> y_lo = y_vec.select<32, 1>(sb * 32);
            simd<float, 32> y_hi = y_vec.select<32, 1>((sb + 1) * 32);

            const float scale_a_lo = scale_f_a[sb];
            const float scale_a_hi = scale_f_a[sb + 1];
            const float min_a_lo   = min_f_a[sb];
            const float min_a_hi   = min_f_a[sb + 1];
            const float scale_b_lo = scale_f_b[sb];
            const float scale_b_hi = scale_f_b[sb + 1];
            const float min_b_lo   = min_f_b[sb];
            const float min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi = qs_hi_b.select<32, 1>(q_offset);

            simd<float, 32> deq_a_lo = convert<float>(qa_lo) * scale_a_lo + min_a_lo;
            simd<float, 32> deq_a_hi = convert<float>(qa_hi) * scale_a_hi + min_a_hi;
            simd<float, 32> deq_b_lo = convert<float>(qb_lo) * scale_b_lo + min_b_lo;
            simd<float, 32> deq_b_hi = convert<float>(qb_hi) * scale_b_hi + min_b_hi;

            acc_a += y_lo * deq_a_lo;
            acc_b += y_lo * deq_b_lo;
            acc_a += y_hi * deq_a_hi;
            acc_b += y_hi * deq_b_hi;
        }
    }
};

// ---------------------------------------------------------------------------
// Q5_K, SOA reorder layout produced by reorder_qw_q5_k:
//   [qs: nb*(QK_K/2)] [qh: nb*(QK_K/8)] [scales: nb*K_SCALE_SIZE] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
//
// Identical to Q4_K except each 4-bit quant gains a 5th (high) bit from qh:
// output chunk c (0..7) adds 16 when bit c of qh[l] is set, where qh[l] indexes
// the same 32 bytes for every chunk (matches dequantize_row_q5_K).
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q5_K> {
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    qh;
        const uint8_t *    scales;
        const sycl::half * dm;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t * qs     = (const uint8_t *) vx;
        const uint8_t * qh     = qs + nb * (QK_K / 2);
        const uint8_t * scales = qh + nb * (QK_K / 8);
        const sycl::half * dm  = (const sycl::half *) (scales + nb * K_SCALE_SIZE);
        return { qs, qh, scales, dm };
    }

    // extract bit `bit` (0..7) of each lane and move it to bit position 4,
    // e.g. for the 4-bit base quant's 5th (high) bit. `bit` is always a
    // compile-time-known unrolled loop constant at call sites, so this folds
    // to a single mask (bit==4), mask+left-shift (bit<4), or mask+right-shift
    // (bit>4) instead of the shift+mask+shift a naive `(qh>>bit & 1) << 4` emits.
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<uint16_t, 32> extract_bit_to_pos4(
            sycl::ext::intel::esimd::simd<uint8_t, 32> qh, int bit) {
        using namespace sycl::ext::intel::esimd;
        simd<uint16_t, 32> masked = convert<uint16_t>(qh & simd<uint8_t, 32>((uint8_t) (1u << bit)));
        if (bit < 4) {
            return masked << simd<uint16_t, 32>((uint16_t) (4 - bit));
        } else if (bit > 4) {
            return masked >> simd<uint16_t, 32>((uint16_t) (bit - 4));
        }
        return masked;
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 32>  qh_a     = block_load<uint8_t, 32>(pa.qh + bia * (QK_K / 8));
        simd<uint8_t, 32>  qh_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * K_SCALE_SIZE);
        simd<uint8_t, 12>  scales_b = 0;

        const float dall_a = (float) pa.dm[bia * 2 + 0];
        const float dmin_a = (float) pa.dm[bia * 2 + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            qh_b     = block_load<uint8_t, 32>(pb.qh + bib * (QK_K / 8));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * K_SCALE_SIZE);
            dall_b = (float) pb.dm[bib * 2 + 0];
            dmin_b = (float) pb.dm[bib * 2 + 1];
        }

        simd<float, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int sb = 0; sb < 8; sb += 2) {
            const int q_offset = sb * 16;
            simd<float, 32> y_lo = y_vec.select<32, 1>(sb * 32);
            simd<float, 32> y_hi = y_vec.select<32, 1>((sb + 1) * 32);

            const float scale_a_lo = scale_f_a[sb];
            const float scale_a_hi = scale_f_a[sb + 1];
            const float min_a_lo   = min_f_a[sb];
            const float min_a_hi   = min_f_a[sb + 1];
            const float scale_b_lo = scale_f_b[sb];
            const float scale_b_hi = scale_f_b[sb + 1];
            const float min_b_lo   = min_f_b[sb];
            const float min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo_u8 = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi_u8 = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo_u8 = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi_u8 = qs_hi_b.select<32, 1>(q_offset);
            simd<uint16_t, 32> qa_lo = convert<uint16_t>(qa_lo_u8);
            simd<uint16_t, 32> qa_hi = convert<uint16_t>(qa_hi_u8);
            simd<uint16_t, 32> qb_lo = convert<uint16_t>(qb_lo_u8);
            simd<uint16_t, 32> qb_hi = convert<uint16_t>(qb_hi_u8);

            // add the 5th bit: chunk sb uses qh bit sb, chunk sb+1 uses qh bit sb+1;
            // qh always indexes the same 32 bytes regardless of chunk
            qa_lo += extract_bit_to_pos4(qh_a, sb);
            qa_hi += extract_bit_to_pos4(qh_a, sb + 1);
            qb_lo += extract_bit_to_pos4(qh_b, sb);
            qb_hi += extract_bit_to_pos4(qh_b, sb + 1);

            simd<float, 32> deq_a_lo = convert<float>(qa_lo) * scale_a_lo + min_a_lo;
            simd<float, 32> deq_a_hi = convert<float>(qa_hi) * scale_a_hi + min_a_hi;
            simd<float, 32> deq_b_lo = convert<float>(qb_lo) * scale_b_lo + min_b_lo;
            simd<float, 32> deq_b_hi = convert<float>(qb_hi) * scale_b_hi + min_b_hi;

            acc_a += y_lo * deq_a_lo;
            acc_b += y_lo * deq_b_lo;
            acc_a += y_hi * deq_a_hi;
            acc_b += y_hi * deq_b_hi;
        }
    }
};

// ---------------------------------------------------------------------------
// Q6_K, SOA reorder layout:
//   [ql: nb*(QK_K/2)] [qh: nb*(QK_K/4)] [scales(int8): nb*(QK_K/16)] [d: nb*half]
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q6_K> {
    struct ptrs {
        const uint8_t *    ql;
        const uint8_t *    qh;
        const int8_t *     scales;
        const sycl::half * d;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb) {
        const uint8_t *    ql     = (const uint8_t *) vx;
        const uint8_t *    qh     = ql + nb * (QK_K / 2);
        const int8_t *     scales = (const int8_t *) (qh + nb * (QK_K / 4));
        const sycl::half * d      = (const sycl::half *) (scales + nb * (QK_K / 16));
        return { ql, qh, scales, d };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> ql_a     = block_load<uint8_t, 128>(pa.ql + bia * (QK_K / 2));
        simd<uint8_t, 128> ql_b     = 0;
        simd<uint8_t, 64>  qh_a     = block_load<uint8_t, 64>(pa.qh + bia * (QK_K / 4));
        simd<uint8_t, 64>  qh_b     = 0;
        simd<int8_t, 16>   scales_a = block_load<int8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<int8_t, 16>   scales_b = 0;

        const float d_a = (float) pa.d[bia];
        float d_b = 0.0f;
        if (has_b) {
            ql_b     = block_load<uint8_t, 128>(pb.ql + bib * (QK_K / 2));
            qh_b     = block_load<uint8_t, 64>(pb.qh + bib * (QK_K / 4));
            scales_b = block_load<int8_t, 16>(pb.scales + bib * (QK_K / 16));
            d_b = (float) pb.d[bib];
        }

        simd<float, 16> sc_a = convert<float>(scales_a);
        simd<float, 16> sc_b = convert<float>(scales_b);

#pragma unroll
        for (int im = 0; im < 2; ++im) {
            simd<uint8_t, 32> ql_lo_a   = ql_a.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_a   = ql_a.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_a = qh_a.select<32, 1>(32 * im);
            simd<uint8_t, 32> ql_lo_b   = ql_b.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_b   = ql_b.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_b = qh_b.select<32, 1>(32 * im);

            // reconstruct each 32-wide 6-bit group (matches dequantize_row_q6_K)
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                simd<float, 32> y_g = y_vec.select<32, 1>(32 * (4 * im + g));

                const float scale_a_lo = sc_a[8 * im + 2 * g + 0] * d_a;
                const float scale_a_hi = sc_a[8 * im + 2 * g + 1] * d_a;
                const float scale_b_lo = sc_b[8 * im + 2 * g + 0] * d_b;
                const float scale_b_hi = sc_b[8 * im + 2 * g + 1] * d_b;

                simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
                simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

                simd<uint8_t, 32> qa;
                simd<uint8_t, 32> qb;
                switch (g) {
                    case 0:
                        qa = (ql_lo_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        qb = (ql_lo_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        break;
                    case 1:
                        qa = (ql_hi_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        qb = (ql_hi_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        break;
                    case 2:
                        qa = (ql_lo_a >> simd<uint8_t, 32>(4)) | (qh_bits_a & simd<uint8_t, 32>(0x30));
                        qb = (ql_lo_b >> simd<uint8_t, 32>(4)) | (qh_bits_b & simd<uint8_t, 32>(0x30));
                        break;
                    default:
                        qa = (ql_hi_a >> simd<uint8_t, 32>(4)) | ((qh_bits_a & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        qb = (ql_hi_b >> simd<uint8_t, 32>(4)) | ((qh_bits_b & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        break;
                }

                simd<float, 32> deq_a = (convert<float>(qa) - 32.0f) * scale_vec_a;
                simd<float, 32> deq_b = (convert<float>(qb) - 32.0f) * scale_vec_b;

                acc_a += y_g * deq_a;
                acc_b += y_g * deq_b;
            }
        }
    }
};

//
// Multi-column mul_mat_vec for reordered K-quant weights on the XMX engines (int8 DPAS).
// A thread owns 16 weight rows (B operand, one dword is 4 weights of a row), the q8_1 activation
// columns are the A operand. A super-block is 8 groups of 32, run as 2 parts of 4 groups. A quant
// type supplies its loads and unpack as a traits struct.
//
using sycl::ext::intel::esimd::simd;
constexpr int XMX_ROWS        = 16;  // weight rows per thread (DPAS N)
constexpr int XMX_CHUNK       = 64;  // widest single launch, more columns run as repeated launches
constexpr int XMX_FUSED_CHUNK = GGML_SYCL_XMX_GLU_MAX_COLS;  // widest launch of the fused gate and up kernel

// threads that split K for one tile, sized so that the shared memory stays small
template <int NC> constexpr int xmx_n_split() {
    return NC <= 8 ? 32 : (NC == 16 ? 16 : 8);
}

inline bool xmx_supported(const void * vx, const int ncols) {
    // 2D block loads need a 64 byte aligned base
    return ncols % QK_K == 0 && (uintptr_t) vx % 64 == 0;
}

// transposed 2D block load of 16 rows, dword (kd, n) lands at [kd * 16 + n]
// 2D block loads need rows of at least 64 bytes, shorter rows (SR) are read row by row
template <int W, bool SR>
ESIMD_INLINE simd<uint32_t, W * XMX_ROWS> xmx_load_t(const void * base, unsigned pitch, unsigned h, int x, int y) {
    using namespace sycl::ext::intel::esimd;
    if constexpr (SR) {
        simd<uint32_t, W * XMX_ROWS> r;
        for (int n = 0; n < XMX_ROWS; ++n) {
            const unsigned row                = y + n < (int) h ? y + n : h;
            const char *   p                  = (const char *) base + (size_t) row * (pitch + 1);
            r.template select<W, XMX_ROWS>(n) = block_load<uint32_t, W>((const uint32_t *) p + x);
        }
        return r;
    } else {
        return load_2d<uint32_t, W, XMX_ROWS, 1, true, false>((const uint32_t *) base, pitch, h, pitch, x, y);
    }
}

// per byte (q - zero) as int8 for q in 0..127, no borrow between bytes
ESIMD_INLINE simd<uint32_t, 128> xmx_bytes_sub(simd<uint32_t, 128> x, uint32_t zero_x4) {
    return ((x | 0x80808080u) - zero_x4) ^ 0x80808080u;
}

// Q6_K reorder layout: [ql: nb*128] [qh: nb*64] [scales int8: nb*16] [d: nb*2]
template <bool SR> struct xmx_traits_q6_k {
    static constexpr bool short_rows = SR;
    static constexpr int  nc_wide = 7;
    static constexpr bool has_min = false;
    static constexpr bool plain   = false;
    static constexpr bool min16   = false;

    struct ctx {
        const uint8_t *    ql;
        const uint8_t *    qh;
        const int8_t *     scales;
        const sycl::half * dh;
        unsigned           h, pl, ph, ps;
        int                row0;
        int                bpr;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        c.ql     = (const uint8_t *) vx;
        c.qh     = c.ql + nb * (QK_K / 2);
        c.scales = (const int8_t *) (c.qh + nb * (QK_K / 4));
        c.dh     = (const sycl::half *) (c.scales + nb * (QK_K / 16));
        c.h      = nrows - 1;
        c.pl     = bpr * (QK_K / 2) - 1;
        c.ph     = bpr * (QK_K / 4) - 1;
        c.ps     = bpr * (QK_K / 16) - 1;
        c.row0   = row0;
        c.bpr    = bpr;
        return c;
    }

    // row scales of the 16 scale groups of a super-block
    struct scales_t {
        simd<int8_t, 256> sc8;
        simd<float, 16>   dv;

        ESIMD_INLINE simd<float, 16> get(int gr) {
            simd<int8_t, 16> s = sc8.template select<16, 4>(64 * (gr / 4) + (gr % 4));
            return sycl::ext::intel::esimd::convert<float>(s) * dv;
        }
    };

    // rows: clamped row index of each of the 16 lanes
    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t s;
        s.dv = convert<float>(gather<sycl::half, 16>(c.dh, (rows * c.bpr + sb) * (uint32_t) sizeof(sycl::half)));
        simd<uint32_t, 64> raw = xmx_load_t<4, SR>(c.scales, c.ps, c.h, sb * 4, c.row0);
        s.sc8                  = raw.template bit_cast_view<int8_t>();
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> a, b, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.a = xmx_load_t<8, SR>(c.ql, c.pl, c.h, sb * 32 + part * 16, c.row0);
        p.b = xmx_load_t<8, SR>(c.ql, c.pl, c.h, sb * 32 + part * 16 + 8, c.row0);
        p.h = xmx_load_t<8, SR>(c.qh, c.ph, c.h, sb * 16 + part * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) {
        simd<uint32_t, 128> qs = (g & 1) ? p.b : p.a;
        simd<uint32_t, 128> x;
        if (g == 0) {
            x = (qs & 0x0F0F0F0Fu) | ((p.h << 4) & 0x30303030u);
        } else if (g == 1) {
            x = (qs & 0x0F0F0F0Fu) | ((p.h << 2) & 0x30303030u);
        } else if (g == 2) {
            x = ((qs >> 4) & 0x0F0F0F0Fu) | (p.h & 0x30303030u);
        } else {
            x = ((qs >> 4) & 0x0F0F0F0Fu) | ((p.h >> 2) & 0x30303030u);
        }
        return xmx_bytes_sub(x, 0x20202020u);
    }
};

// Q3_K reorder layout: [qs: nb*64] [hmask: nb*32] [scales: nb*12] [d: nb*2]
// chunk s of 32: 2 bits from qs byte 32*(s/4)+l at shift 2*(s%4), high bit is hmask[l] bit s, value is q - 4
template <bool SR> struct xmx_traits_q3_k {
    static constexpr bool short_rows = SR;
    static constexpr int  nc_wide = 16;
    static constexpr bool has_min = false;
    static constexpr bool plain   = false;
    static constexpr bool min16   = false;

    struct ctx {
        const uint8_t *    qs;
        const uint8_t *    hm;
        const uint8_t *    sc;
        const sycl::half * dh;
        unsigned           h, pq, ph;
        int                row0;
        int                bpr;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        c.qs   = (const uint8_t *) vx;
        c.hm   = c.qs + nb * (QK_K / 4);
        c.sc   = c.hm + nb * (QK_K / 8);
        c.dh   = (const sycl::half *) (c.sc + nb * 12);
        c.h    = nrows - 1;
        c.pq   = bpr * (QK_K / 4) - 1;
        c.ph   = bpr * (QK_K / 8) - 1;
        c.row0 = row0;
        c.bpr  = bpr;
        return c;
    }

    struct scales_t {
        simd<float, 16> rs[16];

        ESIMD_INLINE simd<float, 16> get(int gr) { return rs[gr]; }
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi = rows * c.bpr + sb;
        simd<uint32_t, 16> w0 = gather<uint32_t, 16>((const uint32_t *) c.sc, bi * 12);
        simd<uint32_t, 16> w1 = gather<uint32_t, 16>((const uint32_t *) c.sc, bi * 12 + 4);
        simd<uint32_t, 16> w2 = gather<uint32_t, 16>((const uint32_t *) c.sc, bi * 12 + 8);
        simd<float, 16>    dv = convert<float>(gather<sycl::half, 16>(c.dh, bi * (uint32_t) sizeof(sycl::half)));
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const simd<uint32_t, 16> lo   = (i % 8) < 4 ? w0 : w1;
            const int                bl   = i % 4;
            simd<uint32_t, 16>       nib  = (lo >> (8 * bl + (i < 8 ? 0 : 4))) & 0xFu;
            simd<uint32_t, 16>       hi   = ((w2 >> (8 * bl + 2 * (i / 4))) & 3u) << 4;
            simd<int, 16>            code = nib | hi;
            s.rs[i]                       = convert<float>(code - 32) * dv;
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.q = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 16 + part * 8, c.row0);
        p.h = xmx_load_t<8, SR>(c.hm, c.ph, c.h, sb * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int part) {
        const int           s = 4 * part + g;
        simd<uint32_t, 128> x = ((p.q >> (2 * g)) & 0x03030303u) | (((p.h >> s) & 0x01010101u) << 2);
        return xmx_bytes_sub(x, 0x04040404u);
    }
};

// Q4_K and Q5_K (QH) reorder layout: [qs: nb*128] ([qh: nb*32]) [scales: nb*12] [dm: nb*4]
// group j of 32: nibble of qs byte 32*(j/2)+l (low for even j), Q5_K adds 16 * bit j of qh[l]
// value is dall * sc_j * q - dmin * m_j, one scale and one min per group
template <bool QH, bool SR> struct xmx_traits_q45_k {
    static constexpr bool short_rows = SR;
    static constexpr int  nc_wide = QH ? 16 : 7;
    static constexpr bool has_min = true;
    static constexpr bool plain   = false;
    static constexpr bool min16   = false;

    struct ctx {
        const uint8_t *    qs;
        const uint8_t *    qh;
        const uint8_t *    sc;
        const sycl::half * dm;
        unsigned           h, pq, ph;
        int                row0;
        int                bpr;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        c.qs   = (const uint8_t *) vx;
        c.qh   = c.qs + nb * (QK_K / 2);
        c.sc   = QH ? c.qh + nb * (QK_K / 8) : c.qh;
        c.dm   = (const sycl::half *) (c.sc + nb * 12);
        c.h    = nrows - 1;
        c.pq   = bpr * (QK_K / 2) - 1;
        c.ph   = bpr * (QK_K / 8) - 1;
        c.row0 = row0;
        c.bpr  = bpr;
        return c;
    }

    struct scales_t {
        simd<float, 16> rd[8];
        simd<float, 16> rm[8];

        ESIMD_INLINE simd<float, 16> get_d(int j) { return rd[j]; }

        ESIMD_INLINE simd<float, 16> get_m(int j) { return rm[j]; }
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi = rows * c.bpr + sb;
        simd<uint32_t, 16> w0 = gather<uint32_t, 16>((const uint32_t *) c.sc, bi * 12);
        simd<uint32_t, 16> w1 = gather<uint32_t, 16>((const uint32_t *) c.sc, bi * 12 + 4);
        simd<uint32_t, 16> w2 = gather<uint32_t, 16>((const uint32_t *) c.sc, bi * 12 + 8);
        simd<float, 16>    d  = convert<float>(gather<sycl::half, 16>(c.dm, bi * (2 * (uint32_t) sizeof(sycl::half))));
        simd<float, 16>    dmin = convert<float>(
            gather<sycl::half, 16>(c.dm, bi * (2 * (uint32_t) sizeof(sycl::half)) + (uint32_t) sizeof(sycl::half)));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int          jj = j % 4;
            simd<uint32_t, 16> sc, m;
            if (j < 4) {
                sc = (w0 >> (8 * jj)) & 63u;
                m  = (w1 >> (8 * jj)) & 63u;
            } else {
                sc = ((w2 >> (8 * jj)) & 0xFu) | (((w0 >> (8 * jj + 6)) & 3u) << 4);
                m  = ((w2 >> (8 * jj + 4)) & 0xFu) | (((w1 >> (8 * jj + 6)) & 3u) << 4);
            }
            s.rd[j] = convert<float>(simd<int, 16>(sc)) * d;
            s.rm[j] = convert<float>(simd<int, 16>(m)) * dmin;
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> a, b, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.a = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 32 + part * 16, c.row0);
        p.b = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 32 + part * 16 + 8, c.row0);
        if constexpr (QH) {
            p.h = xmx_load_t<8, SR>(c.qh, c.ph, c.h, sb * 8, c.row0);
        }
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int part) {
        simd<uint32_t, 128> qs = (g < 2) ? p.a : p.b;
        simd<uint32_t, 128> x  = (qs >> (4 * (g & 1))) & 0x0F0F0F0Fu;
        if constexpr (QH) {
            x |= ((p.h >> (4 * part + g)) & 0x01010101u) << 4;
        }
        return x;
    }
};

// Q2_K reorder layout: [qs: nb*64] [scales: nb*16] [dm: nb*4]
// chunk s of 32: 2 bits from qs byte 32*(s/4)+l at shift 2*(s%4), each 16 wide half has a 4 bit scale and min
// value is d * (code & 15) * q - dmin * (code >> 4)
template <bool SR> struct xmx_traits_q2_k {
    static constexpr bool short_rows = SR;
    static constexpr int  nc_wide = 16;
    static constexpr bool has_min = false;
    static constexpr bool plain   = false;
    static constexpr bool min16   = true;

    struct ctx {
        const uint8_t *    qs;
        const uint8_t *    sc;
        const sycl::half * dm;
        unsigned           h, pq, ps;
        int                row0;
        int                bpr;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        c.qs   = (const uint8_t *) vx;
        c.sc   = c.qs + nb * (QK_K / 4);
        c.dm   = (const sycl::half *) (c.sc + nb * (QK_K / 16));
        c.h    = nrows - 1;
        c.pq   = bpr * (QK_K / 4) - 1;
        c.ps   = bpr * (QK_K / 16) - 1;
        c.row0 = row0;
        c.bpr  = bpr;
        return c;
    }

    struct scales_t {
        simd<uint8_t, 256> code;
        simd<float, 16>    d, dmin;

        ESIMD_INLINE simd<uint8_t, 16> byte(int gr) { return code.template select<16, 4>(64 * (gr / 4) + (gr % 4)); }

        ESIMD_INLINE simd<float, 16> get_d(int gr) {
            simd<uint8_t, 16> c = byte(gr) & (uint8_t) 15;
            return sycl::ext::intel::esimd::convert<float>(c) * d;
        }

        ESIMD_INLINE simd<float, 16> get_m(int gr) {
            simd<uint8_t, 16> c = byte(gr) >> (uint8_t) 4;
            return sycl::ext::intel::esimd::convert<float>(c) * dmin;
        }
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi  = rows * c.bpr + sb;
        simd<uint32_t, 64> raw = xmx_load_t<4, SR>(c.sc, c.ps, c.h, sb * 4, c.row0);
        s.code                 = raw.template bit_cast_view<uint8_t>();
        s.d                    = convert<float>(gather<sycl::half, 16>(c.dm, bi * (2 * (uint32_t) sizeof(sycl::half))));
        s.dmin                 = convert<float>(
            gather<sycl::half, 16>(c.dm, bi * (2 * (uint32_t) sizeof(sycl::half)) + (uint32_t) sizeof(sycl::half)));
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.q = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 16 + part * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) {
        return (p.q >> (2 * g)) & 0x03030303u;
    }
};

// Q8_0 reorder layout: [qs: nrows*K bytes] [d: one half per 32 weights]
// the int8 weights feed DPAS as they are, a group is one block with one scale
template <bool SR> struct xmx_traits_q8_0 {
    static constexpr bool short_rows = SR;
    static constexpr int  nc_wide = 16;
    static constexpr bool has_min = false;
    static constexpr bool plain   = true;
    static constexpr bool min16   = false;

    struct ctx {
        const uint8_t *    qs;
        const sycl::half * dh;
        unsigned           h, pq;
        int                row0;
        int                bpr;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        ctx c;
        c.qs   = (const uint8_t *) vx;
        c.dh   = (const sycl::half *) (c.qs + (size_t) nrows * bpr * QK_K);
        c.h    = nrows - 1;
        c.pq   = bpr * QK_K - 1;
        c.row0 = row0;
        c.bpr  = bpr;
        return c;
    }

    struct scales_t {
        simd<float, 16> rd[8];

        ESIMD_INLINE simd<float, 16> get_d(int j) { return rd[j]; }
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> blk = (rows * c.bpr + sb) * (QK_K / QK8_0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s.rd[j] = convert<float>(gather<sycl::half, 16>(c.dh, (blk + j) * (uint32_t) sizeof(sycl::half)));
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q[4];
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            p.q[g] = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 64 + 8 * (4 * part + g), c.row0);
        }
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) { return p.q[g]; }
};

template <bool SR> using xmx_traits_q4_k = xmx_traits_q45_k<false, SR>;
template <bool SR> using xmx_traits_q5_k = xmx_traits_q45_k<true, SR>;

// registers per thread, the large file halves the resident threads but avoids spills in wide tiles, the fused
// kernel keeps two sets of weights and scales and always takes it
template <typename T, int NC, bool FUSED> constexpr int xmx_grf() {
    return FUSED || NC >= T::nc_wide ? 256 : 128;
}

// one group of 32 of one weight set, accumulated into Cf for all the columns of the tile
template <typename T, int NC, typename A2T, typename D8T>
ESIMD_INLINE void xmx_group(simd<float, NC * XMX_ROWS> & Cf,
                            const typename T::part_t &   pd,
                            typename T::scales_t &       sc,
                            A2T &                        A2,
                            D8T &                        d8f,
                            simd<int8_t, 512> &          Bl,
                            simd<int8_t, 512> &          Bh,
                            const simd<int8_t, 512> &    Ol,
                            const int                    g,
                            const int                    part) {
    using namespace sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;

    constexpr int  NT    = XMX_ROWS;
    constexpr int  RC    = NC < 8 ? NC : 8;
    constexpr int  NB    = NC / RC;
    constexpr bool split = !T::has_min && !T::plain && !T::min16;  // two scales per group of 32
    constexpr bool pair  = (RC * 32) % 64 == 0;                    // a block of RC rows fills whole registers

    const int gi = 4 * part + g;

    simd<uint32_t, 128> x = T::unpack(pd, g, part);
    simd<int8_t, 512>   B = x.template bit_cast_view<int8_t>();
    if constexpr (split) {
        Bl.template select<256, 1>(0)   = B.template select<256, 1>(0);
        Bh.template select<256, 1>(256) = B.template select<256, 1>(256);
    }

    simd<float, NT> rd;
    simd<float, NT> rm;
    simd<float, NT> rs0;
    simd<float, NT> rs1;

    // Q2_K: scale and min of each 16 wide half are 4 bit codes, the weight is d * scl * q - dmin * mc. scl * q
    // is at most 45, so a dword multiply keeps the bytes apart and the scaled weights go to DPAS as int8. The
    // min term sums a * mc over K, the codes mc take the place of the weights in a second DPAS.
    simd<int8_t, 512> Bs;
    simd<int8_t, 512> Bm;
    if constexpr (T::min16) {
        const simd<uint8_t, NT> c0  = sc.byte(2 * gi);
        const simd<uint8_t, NT> c1  = sc.byte(2 * gi + 1);
        simd<uint32_t, NT>      sl0 = convert<uint32_t>(simd<uint8_t, NT>(c0 & (uint8_t) 15));
        simd<uint32_t, NT>      sl1 = convert<uint32_t>(simd<uint8_t, NT>(c1 & (uint8_t) 15));
        simd<uint32_t, NT>      mc0 = convert<uint32_t>(simd<uint8_t, NT>(c0 >> (uint8_t) 4)) * 0x01010101u;
        simd<uint32_t, NT>      mc1 = convert<uint32_t>(simd<uint8_t, NT>(c1 >> (uint8_t) 4)) * 0x01010101u;
        simd<uint32_t, 128>     ws;
        simd<uint32_t, 128>     wm;
        ws.template select<64, 1>(0)  = x.template select<64, 1>(0) * sl0.template replicate<4>();
        ws.template select<64, 1>(64) = x.template select<64, 1>(64) * sl1.template replicate<4>();
        wm.template select<64, 1>(0)  = mc0.template replicate<4>();
        wm.template select<64, 1>(64) = mc1.template replicate<4>();
        Bs                            = ws.template bit_cast_view<int8_t>();
        Bm                            = wm.template bit_cast_view<int8_t>();
    }
    if constexpr (T::plain) {
        rd = sc.get_d(gi);
    } else if constexpr (T::has_min) {
        rd = sc.get_d(gi);
        rm = sc.get_m(gi);
    } else if constexpr (T::min16) {
        // scales are applied through Bs and Bm
    } else {
        rs0 = sc.get(2 * gi);
        rs1 = sc.get(2 * gi + 1);
    }

#pragma unroll
    for (int b = 0; b < NB; ++b) {
        simd<int8_t, RC * 32> A =
            A2[b].template select<RC * 8, 1>(pair ? (g % 2) * RC * 8 : 0).template bit_cast_view<int8_t>();
        simd<float, RC * NT> d8r =
            d8f[b].template select<RC, RC == 1 ? 1 : 8>(gi).template replicate_vs_w_hs<RC, 1, NT, 0>(0);
        auto Cb = Cf.template select<RC * NT, 1>(b * RC * NT);

        if constexpr (T::plain) {
            simd<int, RC * NT>   Cd = xmx::dpas<8, RC, int>(B, A);
            simd<float, RC * NT> z  = convert<float>(Cd) * rd.template replicate<RC>();
            Cb += z * d8r;
        } else if constexpr (T::has_min) {
            simd<int, RC * NT>   Cd = xmx::dpas<8, RC, int>(B, A);
            simd<int, RC * NT>   Cs = xmx::dpas<8, RC, int>(Ol, A);  // sums of the activations
            simd<float, RC * NT> z  = convert<float>(Cd) * rd.template replicate<RC>();
            z -= convert<float>(Cs) * rm.template replicate<RC>();
            Cb += z * d8r;
        } else if constexpr (T::min16) {
            simd<int, RC * NT>   Cs = xmx::dpas<8, RC, int>(Bs, A);
            simd<int, RC * NT>   Cm = xmx::dpas<8, RC, int>(Bm, A);
            simd<float, RC * NT> z  = convert<float>(Cs) * sc.d.template replicate<RC>();
            z -= convert<float>(Cm) * sc.dmin.template replicate<RC>();
            Cb += z * d8r;
        } else {
            simd<int, RC * NT>   C0 = xmx::dpas<8, RC, int>(Bl, A);
            simd<int, RC * NT>   C1 = xmx::dpas<8, RC, int>(Bh, A);
            simd<float, RC * NT> z  = convert<float>(C0) * rs0.template replicate<RC>();
            z += convert<float>(C1) * rs1.template replicate<RC>();
            Cb += z * d8r;
        }
    }
}

// activation of the GLU applied to the gate, uniform across the launch
template <int N> ESIMD_INLINE simd<float, N> xmx_glu_act(const simd<float, N> & gate, const int glu_op) {
    using namespace sycl::ext::intel::esimd;
    if (glu_op == GGML_GLU_OP_SWIGLU) {
        return gate / (1.0f + exp(-gate));
    }
    // GEGLU, the tanh is written with exp
    simd<float, N> a = 0.79788456f * gate * (1.0f + 0.044715f * gate * gate);
    simd<float, N> t = 1.0f - 2.0f / (exp(2.0f * a) + 1.0f);
    return 0.5f * gate * (1.0f + t);
}

// activations come in through 2D block loads, one for the (d, s) pairs of a super-block (s is unused) and one per two
// groups for the quants, the DPAS A operand needs no repacking. Groups with two scales (16 wide) mask the
// weights instead, zeros in B drop the other half of K.
// KS is the most threads that split K for one tile, the launch may use fewer.
// FUSED computes up * act(gate) from the weights vx (up) and vg (gate), reading the activations once.
template <typename T, int NC, int KS, bool FUSED>
ESIMD_INLINE void xmx_mul_mat(const void * vx,
                              const void * vg,
                              const char * vy,
                              float *      dst,
                              const int    ncols,
                              const int    nrows,
                              const int    ncv,
                              const int    stride_y_bytes,
                              const int    ldd,
                              const int    glu_op,
                              const int    tile,
                              const int    lid,
                              const int    ks) {
    using namespace sycl::ext::intel::esimd;

    constexpr int  NT    = XMX_ROWS;
    constexpr int  RC    = NC < 8 ? NC : 8;
    constexpr int  NB    = NC / RC;
    constexpr int  N_ACC = FUSED ? 2 : 1;
    constexpr bool pair  = (RC * 32) % 64 == 0;  // a block of RC rows fills whole registers

    if constexpr (KS > 1) {
        slm_init<N_ACC * KS * NC * NT * sizeof(float)>();
    }

    const int  bpr  = ncols / QK_K;
    const int  row0 = tile * NT;
    const auto c    = T::make(vx, nrows, bpr, row0);
    const auto cg   = T::make(FUSED ? vg : vx, nrows, bpr, row0);

    simd<uint32_t, NT> rows;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        rows[n] = row0 + n < nrows ? row0 + n : nrows - 1;
    }

    const uint32_t * yq  = (const uint32_t *) vy;
    const uint32_t * yd  = (const uint32_t *) (vy + ncols);
    const unsigned   yqw = ncols - 1;
    const unsigned   ydw = ncols / QK8_1 * sizeof(sycl::half2) - 1;
    const unsigned   yh  = ncv - 1;
    const unsigned   yp  = stride_y_bytes - 1;

    simd<float, NC * NT> Cf  = 0.0f;
    simd<float, NC * NT> Cfg = 0.0f;  // gate, only used when fused
    simd<int8_t, 512>    Bl  = int8_t(0);
    simd<int8_t, 512>    Bh  = int8_t(0);
    simd<int8_t, 512>    Ol  = int8_t(0);
    if constexpr (T::has_min) {
        Ol = int8_t(1);
    }

    for (int sb = lid; sb < bpr; sb += ks) {
        auto                 sc = T::load_scales(c, sb, rows);
        typename T::scales_t scg;
        if constexpr (FUSED) {
            scg = T::load_scales(cg, sb, rows);
        }

        // d of the 8 groups of the super-block, [column * 8 + group]
        simd<float, 8 * RC> d8f[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            simd<uint32_t, 8 * RC>    ds = load_2d<uint32_t, 8, RC, 1, false, false>(yd, ydw, yh, yp, sb * 8, b * RC);
            simd<sycl::half, 16 * RC> dh = ds.template bit_cast_view<sycl::half>();
            d8f[b]                       = convert<float>(simd<sycl::half, 8 * RC>(dh.template select<8 * RC, 2>(0)));
        }

#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const auto         pd = T::load_part(c, sb, part);
            typename T::part_t pdg;
            if constexpr (FUSED) {
                pdg = T::load_part(cg, sb, part);
            }

            simd<uint32_t, 16 * RC> A2[NB];  // quants of two groups

#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int gi = 4 * part + g;
                if constexpr (pair) {
                    if (g % 2 == 0) {
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            A2[b] =
                                load_2d<uint32_t, 8, RC, 2, false, false>(yq, yqw, yh, yp, sb * 64 + gi * 8, b * RC);
                        }
                    }
                } else {
#pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        A2[b].template select<RC * 8, 1>(0) =
                            load_2d<uint32_t, 8, RC, 1, false, false>(yq, yqw, yh, yp, sb * 64 + gi * 8, b * RC);
                    }
                }

                xmx_group<T, NC>(Cf, pd, sc, A2, d8f, Bl, Bh, Ol, g, part);
                if constexpr (FUSED) {
                    xmx_group<T, NC>(Cfg, pdg, scg, A2, d8f, Bl, Bh, Ol, g, part);
                }
            }
        }
    }

    if constexpr (KS > 1) {
        slm_block_store<float, NC * NT>(lid * NC * NT * sizeof(float), Cf);
        if constexpr (FUSED) {
            slm_block_store<float, NC * NT>((KS + lid) * NC * NT * sizeof(float), Cfg);
        }
        barrier();
        if (lid != 0) {
            return;
        }
        Cf  = 0.0f;
        Cfg = 0.0f;
#pragma unroll 4
        for (int j = 0; j < ks; ++j) {
            Cf += slm_block_load<float, NC * NT>(j * NC * NT * sizeof(float));
            if constexpr (FUSED) {
                Cfg += slm_block_load<float, NC * NT>((KS + j) * NC * NT * sizeof(float));
            }
        }
    }

    if constexpr (FUSED) {
        Cf *= xmx_glu_act<NC * NT>(Cfg, glu_op);
    }

    if (row0 + NT <= nrows && ldd % 4 == 0 && (uintptr_t) dst % 16 == 0) {
#pragma unroll
        for (int m = 0; m < NC; ++m) {
            if (NC <= 8 || m < ncv) {
                block_store<float, NT>(dst + (size_t) m * ldd + row0, Cf.template select<NT, 1>(m * NT));
            }
        }
    } else {
        // tail tile or unaligned output
        simd<uint32_t, NT> lane(0, 1);
        simd_mask<NT>      in_rows = (lane + row0) < (uint32_t) nrows;
#pragma unroll
        for (int m = 0; m < NC; ++m) {
            if (NC <= 8 || m < ncv) {
                scatter<float, NT>(dst + (size_t) m * ldd + row0, lane * (uint32_t) sizeof(float),
                                   Cf.template select<NT, 1>(m * NT), in_rows);
            }
        }
    }
}

// KS_MAX bounds the threads that split K for one tile, the launch uses as few as keep every thread to the same
// number of super-blocks as the busiest one
template <typename T, int NC, int KS_MAX, bool FUSED>
static void xmx_mul_mat_launch_ks(const void *    vx,
                                  const void *    vg,
                                  const char *    vy,
                                  float *         dst,
                                  const int       ncols,
                                  const int       nrows,
                                  const int       ncv,
                                  const int       stride_y_bytes,
                                  const int       ldd,
                                  const int       glu_op,
                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const size_t n_tiles = (nrows + XMX_ROWS - 1) / XMX_ROWS;
    const int    bpr     = ncols / QK_K;
    const int    rounds  = (bpr + KS_MAX - 1) / KS_MAX;
    const int    ks      = (bpr + rounds - 1) / rounds;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_tiles * ks), sycl::range<1>(ks)),
                         sycl::ext::oneapi::experimental::properties{
                             sycl::ext::intel::experimental::grf_size<xmx_grf<T, NC, FUSED>()> },
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                             xmx_mul_mat<T, NC, KS_MAX, FUSED>(vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd,
                                                               glu_op, (int) it.get_group(0), (int) it.get_local_id(0),
                                                               ks);
                         });
    });
}

template <typename T, int NC, bool FUSED>
static void xmx_mul_mat_launch(const void *    vx,
                               const void *    vg,
                               const char *    vy,
                               float *         dst,
                               const int       ncols,
                               const int       nrows,
                               const int       ncv,
                               const int       stride_y_bytes,
                               const int       ldd,
                               const int       glu_op,
                               dpct::queue_ptr stream) {
    // the fused kernel keeps two sets of accumulators in shared memory, half the threads for the same size
    constexpr int ks_max = xmx_n_split<NC>() / (FUSED ? 2 : 1);
    xmx_mul_mat_launch_ks<T, NC, ks_max, FUSED>(vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd, glu_op,
                                                stream);
}

// FUSED: vx and vg are the up and gate weights, dst gets up * act(gate) with glu_op a ggml_glu_op
template <typename T, bool FUSED = false>
static void xmx_mul_mat_ncols(const void *    vx,
                              const void *    vg,
                              const char *    vy,
                              float *         dst,
                              const int       ncols,
                              const int       nrows,
                              const int       ncols_dst,
                              const int       stride_y_bytes,
                              const int       ldd,
                              const int       glu_op,
                              dpct::queue_ptr stream) {
    constexpr int chunk_max = FUSED ? XMX_FUSED_CHUNK : XMX_CHUNK;
    for (int c0 = 0; c0 < ncols_dst; c0 += chunk_max) {
        const int    n  = std::min(chunk_max, ncols_dst - c0);
        const char * y  = vy + (size_t) c0 * stride_y_bytes;
        float *      d  = dst + (size_t) c0 * ldd;
        const int    nc = n > 32 ? 64 : (n > 16 ? 32 : (n > 8 ? 16 : n));
        switch (nc) {
#define XMX_CASE(N)                                                                                          \
    case N:                                                                                                  \
        xmx_mul_mat_launch<T, N, FUSED>(vx, vg, y, d, ncols, nrows, n, stride_y_bytes, ldd, glu_op, stream); \
        break
            XMX_CASE(1);
            XMX_CASE(2);
            XMX_CASE(3);
            XMX_CASE(4);
            XMX_CASE(5);
            XMX_CASE(6);
            XMX_CASE(7);
            XMX_CASE(8);
#undef XMX_CASE
            // short row variants are for tests, not built for the wide tiles
#define XMX_CASE_WIDE(N)                                                                                         \
    case N:                                                                                                      \
        if constexpr (!T::short_rows && (!FUSED || N <= XMX_FUSED_CHUNK)) {                                      \
            xmx_mul_mat_launch<T, N, FUSED>(vx, vg, y, d, ncols, nrows, n, stride_y_bytes, ldd, glu_op, stream); \
        }                                                                                                        \
        break
            XMX_CASE_WIDE(16);
            XMX_CASE_WIDE(32);
            XMX_CASE_WIDE(64);
#undef XMX_CASE_WIDE
            default:
                GGML_ABORT("unsupported ncols_dst=%d for XMX MMV", n);
        }
    }
}

} // namespace ggml_sycl_esimd

#endif // GGML_SYCL_ESIMD_HPP
