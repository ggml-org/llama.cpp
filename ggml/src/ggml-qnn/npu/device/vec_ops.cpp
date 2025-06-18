#include "vec_ops.hpp"

#include <HTP/core/intrinsics.h>

#include "util.hpp"

namespace {

template <typename _TElem, HVX_Vector (*_MpyFunc)(HVX_Vector, HVX_Vector),
          HVX_Vector (*_AddFunc)(HVX_Vector, HVX_Vector), float (*_ReduceFunc)(HVX_Vector)>
inline float vec_dot_product_impl(const _TElem * src0, const _TElem * src1, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TElem);

    HVX_Vector *       src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector * const src0_vec_ptr_end = ((HVX_Vector *) src0) + count / kElementsPerVector;
    HVX_Vector *       src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector         prev0            = *src0_vec_ptr++;
    HVX_Vector         prev1            = *src1_vec_ptr++;
    HVX_Vector         sum              = Q6_V_vzero();
    HVX_Vector         sum0             = Q6_V_vzero();
    HVX_Vector         sum1             = Q6_V_vzero();

    while (src0_vec_ptr_end - src0_vec_ptr > 1) {
        HVX_VectorPair curr0 = reinterpret_cast<HVX_VectorPair *>(src0_vec_ptr)[0];
        HVX_VectorPair curr1 = reinterpret_cast<HVX_VectorPair *>(src1_vec_ptr)[0];

        HVX_Vector l0 = Q6_V_valign_VVR(Q6_V_lo_W(curr0), prev0, (size_t) src0);
        HVX_Vector l1 = Q6_V_valign_VVR(Q6_V_lo_W(curr1), prev1, (size_t) src1);
        HVX_Vector h0 = Q6_V_valign_VVR(Q6_V_hi_W(curr0), Q6_V_lo_W(curr0), (size_t) src0);
        HVX_Vector h1 = Q6_V_valign_VVR(Q6_V_hi_W(curr1), Q6_V_lo_W(curr1), (size_t) src1);
        prev0         = Q6_V_hi_W(curr0);
        prev1         = Q6_V_hi_W(curr1);
        src0_vec_ptr += 2;
        src1_vec_ptr += 2;

        sum0 = _AddFunc(_MpyFunc(l0, l1), sum0);
        sum1 = _AddFunc(_MpyFunc(h0, h1), sum1);
    }

    sum = _AddFunc(sum0, sum1);
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

    const size_t leftover_bytes = leftover * sizeof(_TElem);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 = (leftover_bytes + hexagon::unaligned_bytes(src0_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src0_vec_ptr :
                               prev0;
        curr0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes + hexagon::unaligned_bytes(src1_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src1_vec_ptr :
                               prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        sum = _AddFunc(Q6_V_valign_VVR(_MpyFunc(curr0, curr1), Q6_V_vzero(), leftover_bytes), sum);
    }

    return _ReduceFunc(sum);
}

template <typename _TElem, HVX_Vector (*_MpyFunc)(HVX_Vector, HVX_Vector),
          HVX_Vector (*_AddFunc)(HVX_Vector, HVX_Vector), float (*_ReduceFunc)(HVX_Vector)>
inline float vec_dot_product_aligned_impl(const _TElem * src0, const _TElem * src1, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TElem);

    HVX_Vector *       src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector * const src0_vec_ptr_end = ((HVX_Vector *) src0) + count / kElementsPerVector;
    HVX_Vector *       src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector         sum0             = Q6_V_vzero();
    HVX_Vector         sum1             = Q6_V_vzero();

    while (src0_vec_ptr_end - src0_vec_ptr > 1) {
        HVX_Vector curr0_lo = src0_vec_ptr[0];
        HVX_Vector curr0_hi = src0_vec_ptr[1];
        HVX_Vector curr1_lo = src1_vec_ptr[0];
        HVX_Vector curr1_hi = src1_vec_ptr[1];
        src0_vec_ptr += 2;
        src1_vec_ptr += 2;

        sum0 = _AddFunc(_MpyFunc(curr0_lo, curr1_lo), sum0);
        sum1 = _AddFunc(_MpyFunc(curr0_hi, curr1_hi), sum1);
    }

    return _ReduceFunc(_AddFunc(sum0, sum1));
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

}  // namespace

namespace hexagon {

float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count) {
    return vec_dot_product_impl<float, vec_mpy_qf32, vec_add_qf32, hexagon::vec_reduction_qf32_f32>(src0, src1, count);
}

float vec_dot_product_aligned_f32_f32(const float * src0, const float * src1, size_t count) {
    return vec_dot_product_aligned_impl<float, vec_mpy_qf32, vec_add_qf32, hexagon::vec_reduction_qf32_f32>(src0, src1,
                                                                                                            count);
}

float vec_dot_product_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count) {
    return vec_dot_product_impl<npu_device_fp16_t, vec_mpy_qf16, vec_add_qf16, hexagon::vec_reduction_qf16_f32>(
        src0, src1, count);
}

float vec_dot_product_aligned_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count) {
    return vec_dot_product_aligned_impl<npu_device_fp16_t, vec_mpy_qf16, vec_add_qf16, hexagon::vec_reduction_qf16_f32>(
        src0, src1, count);
}

}  // namespace hexagon
