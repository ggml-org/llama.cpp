//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_ESIMD_HPP
#define GGML_SYCL_ESIMD_HPP

#include <sycl/ext/intel/esimd.hpp>

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

    // dequantize block bia of pa and block bib of pb, MAC both against y_vec into acc_a / acc_b
    // when has_b is false, block b is treated as all-zero and contributes nothing
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

        // unpack Q4_K scale/min codes (get_scale_min_k4 layout), vectorized, per block
        simd<uint8_t, 8> sc = 0;
        simd<uint8_t, 8> m  = 0;
        simd<float, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        {
            simd<uint8_t, 4> scale_lo = scales_a.select<4, 1>(0);
            simd<uint8_t, 4> min_lo   = scales_a.select<4, 1>(4);
            simd<uint8_t, 4> hi_bits  = scales_a.select<4, 1>(8);
            sc.select<4, 1>(0) = scale_lo & simd<uint8_t, 4>(0x3F);
            sc.select<4, 1>(4) = (hi_bits & simd<uint8_t, 4>(0x0F)) |
                                 ((scale_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
            m.select<4, 1>(0)  = min_lo & simd<uint8_t, 4>(0x3F);
            m.select<4, 1>(4)  = (hi_bits >> simd<uint8_t, 4>(4)) |
                                 ((min_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
            scale_f_a = convert<float>(sc) * dall_a;
            min_f_a   = convert<float>(m) * (-dmin_a);
        }
        {
            simd<uint8_t, 4> scale_lo = scales_b.select<4, 1>(0);
            simd<uint8_t, 4> min_lo   = scales_b.select<4, 1>(4);
            simd<uint8_t, 4> hi_bits  = scales_b.select<4, 1>(8);
            sc.select<4, 1>(0) = scale_lo & simd<uint8_t, 4>(0x3F);
            sc.select<4, 1>(4) = (hi_bits & simd<uint8_t, 4>(0x0F)) |
                                 ((scale_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
            m.select<4, 1>(0)  = min_lo & simd<uint8_t, 4>(0x3F);
            m.select<4, 1>(4)  = (hi_bits >> simd<uint8_t, 4>(4)) |
                                 ((min_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
            scale_f_b = convert<float>(sc) * dall_b;
            min_f_b   = convert<float>(m) * (-dmin_b);
        }

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0f);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0f);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

        // single fused dequant+MAC loop updating both accumulators each iteration;
        // the two acc chains are co-scheduled so FMA latency is overlapped
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
        const uint8_t * ql    = (const uint8_t *) vx;
        const uint8_t * qh    = ql + nb * (QK_K / 2);
        const int8_t *  scales = (const int8_t *) (qh + nb * (QK_K / 4));
        const sycl::half * d  = (const sycl::half *) (scales + nb * (QK_K / 16));
        return { ql, qh, scales, d };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> ql_a    = block_load<uint8_t, 128>(pa.ql + bia * (QK_K / 2));
        simd<uint8_t, 128> ql_b    = 0;
        simd<uint8_t, 64>  qh_a    = block_load<uint8_t, 64>(pa.qh + bia * (QK_K / 4));
        simd<uint8_t, 64>  qh_b    = 0;
        simd<int8_t, 16>   scale_a = block_load<int8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<int8_t, 16>   scale_b = 0;
        if (has_b) {
            ql_b    = block_load<uint8_t, 128>(pb.ql + bib * (QK_K / 2));
            qh_b    = block_load<uint8_t, 64>(pb.qh + bib * (QK_K / 4));
            scale_b = block_load<int8_t, 16>(pb.scales + bib * (QK_K / 16));
        }

        simd<float, 16> sc_a = convert<float>(scale_a);
        simd<float, 16> sc_b = convert<float>(scale_b);
        const float d_a = (float) pa.d[bia];
        const float d_b = has_b ? (float) pb.d[bib] : 0.0f;

        for (int im = 0; im < 2; ++im) {
            simd<uint8_t, 32> ql_lo_a   = ql_a.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_a   = ql_a.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_a = qh_a.select<32, 1>(32 * im);
            simd<uint8_t, 32> ql_lo_b   = ql_b.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_b   = ql_b.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_b = qh_b.select<32, 1>(32 * im);

            // four quant groups, both blocks interleaved per group
            // reconstruct each 32-wide 6-bit group (matches dequantize_row_q6_K)
            for (int g = 0; g < 4; ++g) {
                simd<float, 32> y_g = y_vec.select<32, 1>(32 * (4 * im + g));

                const float scale_a_lo = sc_a[8 * im + 2 * g + 0] * d_a;
                const float scale_a_hi = sc_a[8 * im + 2 * g + 1] * d_a;
                const float scale_b_lo = sc_b[8 * im + 2 * g + 0] * d_b;
                const float scale_b_hi = sc_b[8 * im + 2 * g + 1] * d_b;

                simd<float, 32> scale_vec_a;
                scale_vec_a.select<16, 1>(0)  = scale_a_lo;
                scale_vec_a.select<16, 1>(16) = scale_a_hi;
                simd<float, 32> scale_vec_b;
                scale_vec_b.select<16, 1>(0)  = scale_b_lo;
                scale_vec_b.select<16, 1>(16) = scale_b_hi;

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

} // namespace ggml_sycl_esimd

#endif // GGML_SYCL_ESIMD_HPP
