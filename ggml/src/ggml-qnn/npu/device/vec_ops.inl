#pragma once

#include "hexagon_npu.h"

#include <hexagon_types.h>

#include <cstdint>

namespace hexagon::vec {

template <typename _TElem,
          typename _TRet,
          HVX_Vector (*_MpyFunc)(HVX_Vector, HVX_Vector),
          HVX_Vector (*_AddFunc)(HVX_Vector, HVX_Vector),
          _TRet (*_ReduceFunc)(HVX_Vector)>
inline _TRet vec_dot_product_impl(const _TElem * src0, const _TElem * src1, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TElem);

    HVX_Vector *       src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector * const src0_vec_ptr_end = ((HVX_Vector *) src0) + count / kElementsPerVector;
    HVX_Vector *       src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector         prev0            = *src0_vec_ptr++;
    HVX_Vector         prev1            = *src1_vec_ptr++;
    HVX_Vector         sum              = Q6_V_vzero();

    if (src0_vec_ptr_end - src0_vec_ptr > 1) {
        HVX_Vector sum0 = Q6_V_vzero();
        HVX_Vector sum1 = Q6_V_vzero();

        do {
            HVX_VectorPair curr0 = reinterpret_cast<HVX_VectorPair *>(src0_vec_ptr)[0];
            HVX_VectorPair curr1 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];

            HVX_Vector l0 = Q6_V_valign_VVR(Q6_V_lo_W(curr0), prev0, (size_t) src0);
            HVX_Vector l1 = Q6_V_valign_VVR(Q6_V_lo_W(curr1), prev1, (size_t) src1);
            sum0          = _AddFunc(_MpyFunc(l0, l1), sum0);

            HVX_Vector h0 = Q6_V_valign_VVR(Q6_V_hi_W(curr0), Q6_V_lo_W(curr0), (size_t) src0);
            HVX_Vector h1 = Q6_V_valign_VVR(Q6_V_hi_W(curr1), Q6_V_lo_W(curr1), (size_t) src1);
            sum1          = _AddFunc(_MpyFunc(h0, h1), sum1);

            prev0 = Q6_V_hi_W(curr0);
            prev1 = Q6_V_hi_W(curr1);
            src0_vec_ptr += 2;
            src1_vec_ptr += 2;
        } while (src0_vec_ptr_end - src0_vec_ptr > 1);

        sum = _AddFunc(sum0, sum1);
    }

    if (src0_vec_ptr_end - src0_vec_ptr > 0) {
        HVX_Vector curr0 = *src0_vec_ptr++;
        HVX_Vector curr1 = *src1_vec_ptr++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        prev0            = curr0;
        prev1            = curr1;

        sum = _AddFunc(_MpyFunc(s0, s1), sum);
    }

    const size_t leftover = count % kElementsPerVector;
    if ((src0_vec_ptr_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        bool       should_fetch_src0 = leftover != 0 || !hexagon::is_addr_aligned(src0_vec_ptr);
        bool       should_fetch_src1 = leftover != 0 || !hexagon::is_addr_aligned(src1_vec_ptr);
        HVX_Vector curr0             = should_fetch_src0 ? *src0_vec_ptr : prev0;
        HVX_Vector curr1             = should_fetch_src1 ? *src1_vec_ptr : prev1;
        src0_vec_ptr += should_fetch_src0 ? 1 : 0;
        src1_vec_ptr += should_fetch_src1 ? 1 : 0;
        HVX_Vector s0 = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1 = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        prev0         = curr0;
        prev1         = curr1;

        sum = _AddFunc(_MpyFunc(s0, s1), sum);
    }

    if (leftover > 0) {
        // handle the leftover elements
        const size_t leftover_bytes = leftover * sizeof(_TElem);
        HVX_Vector   curr0 = (leftover_bytes + hexagon::unaligned_bytes(src0_vec_ptr) > hexagon::kBytesPerVector) ?
                                 *src0_vec_ptr :
                                 prev0;
        curr0              = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes + hexagon::unaligned_bytes(src1_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src1_vec_ptr :
                               prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        sum = _AddFunc(Q6_V_valign_VVR(_MpyFunc(curr0, curr1), Q6_V_vzero(), leftover_bytes), sum);
    }

    return _ReduceFunc(sum);
}

template <typename _TElem,
          typename _TRet,
          HVX_Vector (*_MpyFunc)(HVX_Vector, HVX_Vector),
          HVX_Vector (*_AddFunc)(HVX_Vector, HVX_Vector),
          _TRet (*_ReduceFunc)(HVX_Vector)>
inline _TRet vec_dot_product_aligned_impl(const _TElem * src0, const _TElem * src1, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TElem);

    HVX_Vector *       src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector * const src0_vec_ptr_end = ((HVX_Vector *) src0) + count / kElementsPerVector;
    HVX_Vector *       src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector         sum              = Q6_V_vzero();

    {
        HVX_Vector sum0 = Q6_V_vzero();
        HVX_Vector sum1 = Q6_V_vzero();
        if (src0_vec_ptr_end - src0_vec_ptr > 3) {
            HVX_Vector sum2 = Q6_V_vzero();
            HVX_Vector sum3 = Q6_V_vzero();

            do {
                HVX_VectorPair curr00 = reinterpret_cast<HVX_VectorPair *>(src0_vec_ptr)[0];
                HVX_VectorPair curr10 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];
                sum0                  = _AddFunc(_MpyFunc(Q6_V_lo_W(curr00), Q6_V_lo_W(curr10)), sum0);
                sum1                  = _AddFunc(_MpyFunc(Q6_V_hi_W(curr00), Q6_V_hi_W(curr10)), sum1);

                HVX_VectorPair curr01 = reinterpret_cast<HVX_VectorPair *>(src0_vec_ptr)[1];
                HVX_VectorPair curr11 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[1];
                sum2                  = _AddFunc(_MpyFunc(Q6_V_lo_W(curr01), Q6_V_lo_W(curr11)), sum2);
                sum3                  = _AddFunc(_MpyFunc(Q6_V_hi_W(curr01), Q6_V_hi_W(curr11)), sum3);

                src0_vec_ptr += 4;
                src1_vec_ptr += 4;
            } while (src0_vec_ptr_end - src0_vec_ptr > 3);

            sum0 = _AddFunc(sum2, sum0);
            sum1 = _AddFunc(sum3, sum1);
        }

        if (src0_vec_ptr_end - src0_vec_ptr > 1) {
            HVX_VectorPair curr0 = reinterpret_cast<HVX_VectorPair *>(src0_vec_ptr)[0];
            HVX_VectorPair curr1 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];
            src0_vec_ptr += 2;
            src1_vec_ptr += 2;

            sum0 = _AddFunc(_MpyFunc(Q6_V_lo_W(curr0), Q6_V_lo_W(curr1)), sum0);
            sum1 = _AddFunc(_MpyFunc(Q6_V_hi_W(curr0), Q6_V_hi_W(curr1)), sum1);
        }

        sum = _AddFunc(sum0, sum1);
    }

    if (src0_vec_ptr_end - src0_vec_ptr > 0) {
        HVX_Vector curr0 = src0_vec_ptr[0];
        HVX_Vector curr1 = src1_vec_ptr[0];

        sum = _AddFunc(_MpyFunc(curr0, curr1), sum);
    }

    return _ReduceFunc(sum);
}

inline HVX_Vector vec_mpy_qf32(HVX_Vector src0, HVX_Vector src1) {
    return Q6_Vqf32_vmpy_VsfVsf(src0, src1);
}

inline HVX_Vector vec_add_qf32(HVX_Vector sum, HVX_Vector result) {
    return Q6_Vqf32_vadd_Vqf32Vqf32(sum, result);
}

inline HVX_Vector vec_mpy_qf16(HVX_Vector src0, HVX_Vector src1) {
    return Q6_Vqf16_vmpy_VhfVhf(src0, src1);
}

inline HVX_Vector vec_add_qf16(HVX_Vector sum, HVX_Vector result) {
    return Q6_Vqf16_vadd_Vqf16Vqf16(sum, result);
}

template <typename _TElem0,
          typename _TElem1,
          typename _TRet,
          HVX_Vector_x2 (*_ExpandFunc)(HVX_Vector, HVX_Vector),
          HVX_Vector (*_MpyFunc)(HVX_Vector, HVX_Vector),
          HVX_Vector (*_AddFunc)(HVX_Vector, HVX_Vector),
          _TRet (*_ReduceFunc)(HVX_Vector)>
inline _TRet vec_dot_product_mixed_impl(const _TElem0 * src0, const _TElem1 * src1, size_t count) {
    static_assert(sizeof(_TElem0) < sizeof(_TElem1), "Element size mismatch: _TElem0 must be smaller than _TElem1");
    static_assert((sizeof(_TElem1) / sizeof(_TElem0)) == 2,
                  "Element size mismatch: _TElem1 must be twice the size of _TElem0");
    static_assert((sizeof(_TElem1) % sizeof(_TElem0)) == 0,
                  "Element size mismatch: _TElem1 must be a multiple of _TElem0");

    constexpr const size_t kElementsPerVector0 = hexagon::kBytesPerVector / sizeof(_TElem0);
    constexpr const size_t kElementsPerVector1 = hexagon::kBytesPerVector / sizeof(_TElem1);

    constexpr const __fp16 kOne = 1.0f;
    const HVX_Vector kOneV      = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(kOne));

    const _TElem0 * const src0_ptr_end     = src0 + count;
    HVX_Vector *          src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector *          src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector * const    src1_vec_ptr_end = ((HVX_Vector *) src1) + count / kElementsPerVector1;
    HVX_Vector            prev0            = *src0_vec_ptr++;
    HVX_Vector            prev1            = *src1_vec_ptr++;
    HVX_Vector            sum              = Q6_V_vzero();

    if (src1_vec_ptr_end - src1_vec_ptr > 1) {
        HVX_Vector sum0 = Q6_V_vzero();
        HVX_Vector sum1 = Q6_V_vzero();

        do {
            HVX_Vector     curr0 = src0_vec_ptr[0];
            HVX_VectorPair curr1 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];

            HVX_Vector    s0      = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
            HVX_Vector_x2 s0_pair = _ExpandFunc(s0, kOneV);

            HVX_Vector l1 = Q6_V_valign_VVR(Q6_V_lo_W(curr1), prev1, (size_t) src1);
            sum0          = _AddFunc(_MpyFunc(s0_pair.first, l1), sum0);

            HVX_Vector h1 = Q6_V_valign_VVR(Q6_V_hi_W(curr1), Q6_V_lo_W(curr1), (size_t) src1);
            sum1          = _AddFunc(_MpyFunc(s0_pair.second, h1), sum1);

            prev0 = curr0;
            prev1 = Q6_V_hi_W(curr1);
            src0_vec_ptr++;
            src1_vec_ptr += 2;
        } while (src1_vec_ptr_end - src1_vec_ptr > 1);

        sum = _AddFunc(sum0, sum1);
    }

    const size_t leftover1 = count % kElementsPerVector1;
    if ((src1_vec_ptr_end - ((HVX_Vector *) src1)) > 0) {
        // handle the last vector
        const bool should_fetch_src0 =
            reinterpret_cast<const _TElem0 *>(hexagon::align_down(src0_vec_ptr)) < src0_ptr_end;
        HVX_Vector curr0 = should_fetch_src0 ? *src0_vec_ptr : prev0;
        src0_vec_ptr += should_fetch_src0 ? 1 : 0;

        HVX_Vector    s0      = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector_x2 s0_pair = _ExpandFunc(s0, kOneV);

        const bool has_remaining_src1_vector = src1_vec_ptr_end - src1_vec_ptr > 0;
        if (has_remaining_src1_vector) {
            HVX_Vector curr1 = *src1_vec_ptr++;
            HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
            sum              = _AddFunc(_MpyFunc(s0_pair.first, s1), sum);
            prev1            = curr1;
        }

        bool       should_fetch_src1 = leftover1 != 0 || !hexagon::is_addr_aligned(src1_vec_ptr);
        HVX_Vector curr1             = should_fetch_src1 ? *src1_vec_ptr : prev1;
        src1_vec_ptr += should_fetch_src1 ? 1 : 0;
        HVX_Vector s1 = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        prev0         = curr0;
        prev1         = curr1;

        sum = _AddFunc(_MpyFunc(has_remaining_src1_vector ? s0_pair.second : s0_pair.first, s1), sum);
    }

    if (leftover1 > 0) {
        // handle the leftover elements
        const size_t leftover0       = count % kElementsPerVector0;
        const size_t leftover_bytes1 = leftover1 * sizeof(_TElem1);
        HVX_Vector   curr0 =
            reinterpret_cast<const _TElem0 *>(hexagon::align_down(src0_vec_ptr)) < src0_ptr_end ? *src0_vec_ptr : prev0;
        curr0 = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes1 + hexagon::unaligned_bytes(src1_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src1_vec_ptr :
                               prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        HVX_Vector_x2 curr0_pair = _ExpandFunc(curr0, kOneV);

        curr0 = leftover1 == leftover0 ? curr0_pair.first : curr0_pair.second;
        sum   = _AddFunc(Q6_V_valign_VVR(_MpyFunc(curr0, curr1), Q6_V_vzero(), leftover_bytes1), sum);
    }

    return _ReduceFunc(sum);
}

template <typename _TElem0,
          typename _TElem1,
          typename _TRet,
          HVX_Vector_x2 (*_ExpandFunc)(HVX_Vector, HVX_Vector),
          HVX_Vector (*_MpyFunc)(HVX_Vector, HVX_Vector),
          HVX_Vector (*_AddFunc)(HVX_Vector, HVX_Vector),
          _TRet (*_ReduceFunc)(HVX_Vector)>
inline _TRet vec_dot_product_mix_aligned_impl(const _TElem0 * src0, const _TElem1 * src1, size_t count) {
    static_assert(sizeof(_TElem0) < sizeof(_TElem1), "Element size mismatch: _TElem0 must be smaller than _TElem1");
    static_assert((sizeof(_TElem1) / sizeof(_TElem0)) == 2,
                  "Element size mismatch: _TElem1 must be twice the size of _TElem0");
    static_assert((sizeof(_TElem1) % sizeof(_TElem0)) == 0,
                  "Element size mismatch: _TElem1 must be a multiple of _TElem0");

    constexpr const size_t kElementsPerVector1 = hexagon::kBytesPerVector / sizeof(_TElem1);

    constexpr const __fp16 kOne = 1.0f;
    const HVX_Vector kOneV      = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(kOne));

    HVX_Vector *       src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector *       src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector * const src1_vec_ptr_end = ((HVX_Vector *) src1) + count / kElementsPerVector1;
    HVX_Vector         sum0             = Q6_V_vzero();
    HVX_Vector         sum1             = Q6_V_vzero();

    if (src1_vec_ptr_end - src1_vec_ptr > 3) {
        HVX_Vector sum2 = Q6_V_vzero();
        HVX_Vector sum3 = Q6_V_vzero();

        do {
            HVX_VectorPair curr0  = reinterpret_cast<HVX_VectorPair *>(src0_vec_ptr)[0];
            HVX_Vector_x2  curr00 = _ExpandFunc(Q6_V_lo_W(curr0), kOneV);
            HVX_VectorPair curr10 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];
            sum0                  = _AddFunc(_MpyFunc(curr00.first, Q6_V_lo_W(curr10)), sum0);
            sum1                  = _AddFunc(_MpyFunc(curr00.second, Q6_V_hi_W(curr10)), sum1);

            HVX_Vector_x2  curr01 = _ExpandFunc(Q6_V_hi_W(curr0), kOneV);
            HVX_VectorPair curr11 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[1];
            sum2                  = _AddFunc(_MpyFunc(curr01.first, Q6_V_lo_W(curr11)), sum2);
            sum3                  = _AddFunc(_MpyFunc(curr01.second, Q6_V_hi_W(curr11)), sum3);

            src0_vec_ptr += 2;
            src1_vec_ptr += 4;
        } while (src1_vec_ptr_end - src1_vec_ptr > 3);

        sum0 = _AddFunc(sum0, sum2);
        sum1 = _AddFunc(sum1, sum3);
    }

    if (src1_vec_ptr_end - src1_vec_ptr > 1) {
        HVX_Vector    curr0   = src0_vec_ptr[0];
        HVX_Vector_x2 s0_pair = _ExpandFunc(curr0, kOneV);

        HVX_VectorPair curr1 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];
        sum0                 = _AddFunc(_MpyFunc(s0_pair.first, Q6_V_lo_W(curr1)), sum0);
        sum1                 = _AddFunc(_MpyFunc(s0_pair.second, Q6_V_hi_W(curr1)), sum1);
    }

    return _ReduceFunc(_AddFunc(sum0, sum1));
}

template <HVX_Vector (*_Func)(HVX_Vector, HVX_UVector *, HVX_Vector),
          HVX_Vector (*_FuncScaleConvert)(float),
          typename _TParam>
inline void vec_scale_impl(const _TParam * src, float scale, _TParam * dst, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TParam);

    HVX_Vector *       src_vec_ptr = ((HVX_Vector *) src);
    HVX_Vector * const src_vec_end = ((HVX_Vector *) src) + (count / kElementsPerVector);
    HVX_UVector *      dst_vec_ptr = ((HVX_UVector *) dst);  // TODO: opt the unaligned case?
    HVX_Vector         prev        = *src_vec_ptr++;
    const size_t       leftover    = count % kElementsPerVector;

    HVX_Vector scale_vec = _FuncScaleConvert(scale);

    while (src_vec_end - src_vec_ptr > 1) {
        HVX_VectorPair curr = reinterpret_cast<HVX_VectorPair *>(src_vec_ptr)[0];
        src_vec_ptr += 2;

        HVX_Vector lo  = Q6_V_valign_VVR(Q6_V_lo_W(curr), prev, (size_t) src);
        dst_vec_ptr[0] = _Func(lo, dst_vec_ptr, scale_vec);

        HVX_Vector hi  = Q6_V_valign_VVR(Q6_V_hi_W(curr), Q6_V_lo_W(curr), (size_t) src);
        dst_vec_ptr[1] = _Func(hi, dst_vec_ptr + 1, scale_vec);

        dst_vec_ptr += 2;
        prev = Q6_V_hi_W(curr);
    }

    if (src_vec_end - src_vec_ptr > 0) {
        HVX_Vector curr = *src_vec_ptr++;
        HVX_Vector s0   = Q6_V_valign_VVR(curr, prev, (size_t) src);
        dst_vec_ptr[0]  = _Func(s0, dst_vec_ptr, scale_vec);
        dst_vec_ptr++;
        prev = curr;
    }

    if ((src_vec_end - ((HVX_Vector *) src)) > 0) {
        // handle the last vector
        bool       should_fetch_next = leftover == 0 && hexagon::is_addr_aligned(src_vec_ptr);
        HVX_Vector curr              = should_fetch_next ? prev : *src_vec_ptr;
        src_vec_ptr                  = should_fetch_next ? src_vec_ptr : src_vec_ptr + 1;
        HVX_Vector s0                = Q6_V_valign_VVR(curr, prev, (size_t) src);
        dst_vec_ptr[0]               = _Func(s0, dst_vec_ptr, scale_vec);
        dst_vec_ptr++;
        prev = curr;
    }

    if (leftover > 0) {
        // handle the leftover elements
        const size_t leftover_bytes = leftover * sizeof(_TParam);
        HVX_Vector   curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, _Func(curr, dst_vec_ptr, scale_vec));
    }
}

template <typename _TData> inline void vec_zero_impl(_TData * src, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TData);

    HVX_UVector *       src_vec_ptr = ((HVX_UVector *) src);
    HVX_UVector * const src_vec_end = ((HVX_UVector *) src) + (count / kElementsPerVector);

    while (src_vec_end - src_vec_ptr > 1) {
        src_vec_ptr[0] = Q6_V_vzero();
        src_vec_ptr[1] = Q6_V_vzero();
        src_vec_ptr += 2;
    }

    if (src_vec_end - src_vec_ptr > 0) {
        src_vec_ptr[0] = Q6_V_vzero();
        src_vec_ptr++;
    }

    const size_t leftover = count % kElementsPerVector;
    if (leftover > 0) {
        // handle the leftover elements
        const size_t leftover_bytes = leftover * sizeof(_TData);
        q6op_vstu_variable_ARV(src_vec_ptr, leftover_bytes, Q6_V_vzero());
    }
}

template <HVX_Vector (*_OpBinaryTransform)(HVX_Vector, HVX_Vector), typename _TyData>
inline void vec_trans_impl(const _TyData * src0, const _TyData * src1, _TyData * dst, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TyData);

    HVX_Vector *       src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector * const src0_vec_ptr_end = ((HVX_Vector *) src0) + count / kElementsPerVector;
    HVX_Vector *       src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector *       dst_vec_ptr      = ((HVX_Vector *) dst);  // framework will ensure the dst is aligned
    HVX_Vector         prev0            = *src0_vec_ptr++;
    HVX_Vector         prev1            = *src1_vec_ptr++;

    {
        while (src0_vec_ptr_end - src0_vec_ptr > 1) {
            HVX_VectorPair curr0 = reinterpret_cast<HVX_VectorPair *>(src0_vec_ptr)[0];
            HVX_VectorPair curr1 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];

            HVX_Vector l0  = Q6_V_valign_VVR(Q6_V_lo_W(curr0), prev0, (size_t) src0);
            HVX_Vector l1  = Q6_V_valign_VVR(Q6_V_lo_W(curr1), prev1, (size_t) src1);
            dst_vec_ptr[0] = _OpBinaryTransform(l0, l1);

            HVX_Vector h0  = Q6_V_valign_VVR(Q6_V_hi_W(curr0), Q6_V_lo_W(curr0), (size_t) src0);
            HVX_Vector h1  = Q6_V_valign_VVR(Q6_V_hi_W(curr1), Q6_V_lo_W(curr1), (size_t) src1);
            dst_vec_ptr[1] = _OpBinaryTransform(h0, h1);

            prev0 = Q6_V_hi_W(curr0);
            prev1 = Q6_V_hi_W(curr1);
            src0_vec_ptr += 2;
            src1_vec_ptr += 2;
            dst_vec_ptr += 2;
        }
    }

    if (src0_vec_ptr_end - src0_vec_ptr > 0) {
        HVX_Vector curr0 = *src0_vec_ptr++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = *src1_vec_ptr++;
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        dst_vec_ptr[0] = _OpBinaryTransform(s0, s1);

        prev0 = curr0;
        prev1 = curr1;
        dst_vec_ptr++;
    }

    const size_t leftover = count % kElementsPerVector;
    if ((src0_vec_ptr_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        bool should_fetch_src0 = leftover != 0 || !hexagon::is_addr_aligned(src0_vec_ptr);
        bool should_fetch_src1 = leftover != 0 || !hexagon::is_addr_aligned(src1_vec_ptr);

        HVX_Vector curr0 = should_fetch_src0 ? *src0_vec_ptr : prev0;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = should_fetch_src1 ? *src1_vec_ptr : prev1;
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        dst_vec_ptr[0] = _OpBinaryTransform(s0, s1);

        src0_vec_ptr += should_fetch_src0 ? 1 : 0;
        src1_vec_ptr += should_fetch_src1 ? 1 : 0;
        prev0 = curr0;
        prev1 = curr1;
        dst_vec_ptr++;
    }

    if (leftover > 0) {
        // handle the leftover elements
        const size_t leftover_bytes = leftover * sizeof(_TyData);
        HVX_Vector   curr0 = (leftover_bytes + hexagon::unaligned_bytes(src0_vec_ptr) > hexagon::kBytesPerVector) ?
                                 *src0_vec_ptr :
                                 prev0;
        curr0              = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes + hexagon::unaligned_bytes(src1_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src1_vec_ptr :
                               prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, _OpBinaryTransform(curr0, curr1));
    }
}

template <typename _TyData, typename _TyParam, HVX_Vector (*_OpBinaryTransform)(HVX_Vector, HVX_Vector, _TyParam)>
inline void vec_trans_with_param_impl(const _TyData * src0,
                                      const _TyData * src1,
                                      _TyData *       dst,
                                      size_t          count,
                                      _TyParam        param) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TyData);

    HVX_Vector *       src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector * const src0_vec_ptr_end = ((HVX_Vector *) src0) + count / kElementsPerVector;
    HVX_Vector *       src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector *       dst_vec_ptr      = ((HVX_Vector *) dst);  // framework will ensure the dst is aligned
    HVX_Vector         prev0            = *src0_vec_ptr++;
    HVX_Vector         prev1            = *src1_vec_ptr++;

    {
        while (src0_vec_ptr_end - src0_vec_ptr > 1) {
            HVX_VectorPair curr0 = reinterpret_cast<HVX_VectorPair *>(src0_vec_ptr)[0];
            HVX_VectorPair curr1 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];

            HVX_Vector l0  = Q6_V_valign_VVR(Q6_V_lo_W(curr0), prev0, (size_t) src0);
            HVX_Vector l1  = Q6_V_valign_VVR(Q6_V_lo_W(curr1), prev1, (size_t) src1);
            dst_vec_ptr[0] = _OpBinaryTransform(l0, l1, param);

            HVX_Vector h0  = Q6_V_valign_VVR(Q6_V_hi_W(curr0), Q6_V_lo_W(curr0), (size_t) src0);
            HVX_Vector h1  = Q6_V_valign_VVR(Q6_V_hi_W(curr1), Q6_V_lo_W(curr1), (size_t) src1);
            dst_vec_ptr[1] = _OpBinaryTransform(h0, h1, param);

            prev0 = Q6_V_hi_W(curr0);
            prev1 = Q6_V_hi_W(curr1);
            src0_vec_ptr += 2;
            src1_vec_ptr += 2;
            dst_vec_ptr += 2;
        }
    }

    if (src0_vec_ptr_end - src0_vec_ptr > 0) {
        HVX_Vector curr0 = *src0_vec_ptr++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = *src1_vec_ptr++;
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        dst_vec_ptr[0] = _OpBinaryTransform(s0, s1, param);

        prev0 = curr0;
        prev1 = curr1;
        dst_vec_ptr++;
    }

    const size_t leftover = count % kElementsPerVector;
    if ((src0_vec_ptr_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        bool should_fetch_src0 = leftover != 0 || !hexagon::is_addr_aligned(src0_vec_ptr);
        bool should_fetch_src1 = leftover != 0 || !hexagon::is_addr_aligned(src1_vec_ptr);

        HVX_Vector curr0 = should_fetch_src0 ? *src0_vec_ptr : prev0;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = should_fetch_src1 ? *src1_vec_ptr : prev1;
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        dst_vec_ptr[0] = _OpBinaryTransform(s0, s1, param);

        src0_vec_ptr += should_fetch_src0 ? 1 : 0;
        src1_vec_ptr += should_fetch_src1 ? 1 : 0;
        prev0 = curr0;
        prev1 = curr1;
        dst_vec_ptr++;
    }

    if (leftover > 0) {
        // handle the leftover elements
        const size_t leftover_bytes = leftover * sizeof(_TyData);
        HVX_Vector   curr0 = (leftover_bytes + hexagon::unaligned_bytes(src0_vec_ptr) > hexagon::kBytesPerVector) ?
                                 *src0_vec_ptr :
                                 prev0;
        curr0              = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes + hexagon::unaligned_bytes(src1_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src1_vec_ptr :
                               prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, _OpBinaryTransform(curr0, curr1, param));
    }
}

}  // namespace hexagon::vec
