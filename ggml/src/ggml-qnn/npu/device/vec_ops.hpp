#pragma once

#include "hexagon_npu.h"

#include <hexagon_types.h>

#include <cstdint>

namespace hexagon {

constexpr const size_t kBytesPerVector = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kAlignMask      = kBytesPerVector - 1;

template <typename T, int N> struct HEXAGON_pack {
    T val[N];
};

using HVX_Vector_x2     = HEXAGON_pack<HVX_Vector, 2>;
using HVX_Vector_x3     = HEXAGON_pack<HVX_Vector, 3>;
using HVX_Vector_x4     = HEXAGON_pack<HVX_Vector, 4>;
using HVX_Vector_x5     = HEXAGON_pack<HVX_Vector, 5>;
using HVX_VectorPair_x2 = HEXAGON_pack<HVX_VectorPair, 2>;
using HVX_VectorPair_x3 = HEXAGON_pack<HVX_VectorPair, 3>;
using HVX_VectorPair_x4 = HEXAGON_pack<HVX_VectorPair, 4>;
using HVX_VectorPred_x3 = HEXAGON_pack<HVX_VectorPred, 3>;

typedef union {
    HVX_VectorPair VV;

    struct {
        HVX_Vector lo;
        HVX_Vector hi;
    } V;
} HVX_DV;

typedef union {
    HVX_Vector v;
    float      f32[kBytesPerVector / sizeof(float)];
    uint32_t   u32[kBytesPerVector / sizeof(uint32_t)];
    __fp16 f16[kBytesPerVector / sizeof(__fp16)];
    uint16_t u16[kBytesPerVector / sizeof(uint16_t)];
    uint8_t  u8[kBytesPerVector];
} HVX_VectorAlias;

inline size_t get_aligned_size(size_t size) {
    return (size + kAlignMask) & ~kAlignMask;
}

inline size_t unaligned_bytes(const void * addr) {
    return ((size_t) addr) & kAlignMask;
}

template <typename _TyData> inline const _TyData * align_down(const _TyData * addr) {
    return reinterpret_cast<const _TyData *>(reinterpret_cast<const uint8_t *>(addr) - unaligned_bytes(addr));
}

inline size_t bytes_to_vector_boundary(const void * addr) {
    return kBytesPerVector - unaligned_bytes(addr);
}

inline bool is_addr_aligned(const void * addr) {
    return unaligned_bytes(addr) == 0;
}

inline bool is_size_aligned(size_t size) {
    return (size & kAlignMask) == 0;
}

inline float get_flt0_from_fltv(HVX_Vector vect) {
    static_assert(sizeof(vect[0]) == sizeof(float), "vect[0] should be a float");
    int32_t i = vect[0];
    return reinterpret_cast<float &>(i);
}

inline HVX_UVector Q6_V_vmemu_R(const void * unaligned_ptr) {
    return *reinterpret_cast<const HVX_UVector *>(unaligned_ptr);
}

inline HVX_Vector Q6_V_vmem_R(const void * aligned_ptr) {
    return *reinterpret_cast<const HVX_Vector *>(aligned_ptr);
}

constexpr const size_t kL2CacheSize         = 8 * 1024;  // // 8KB L2 cache
constexpr const size_t kL2FetchAheadVectors = kL2CacheSize / kBytesPerVector;

inline void l2fetch(const void * p, uint32_t stride, uint32_t width, uint32_t height, uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__(" l2fetch(%0,%1) " : : "r"(p), "r"(control));
}

inline void l2fetch_row(const uint8_t * row_ptr, size_t bytes) {
    // TODO: should we use small kL2FetchAheadVectors?
    int32_t l2fetch_vectors = Q6_R_min_RR(bytes / kBytesPerVector, kL2FetchAheadVectors);
    hexagon::l2fetch(row_ptr, kBytesPerVector, kBytesPerVector, l2fetch_vectors, 0);
}

template <uint32_t _TyBytes> inline void q6op_vstu_variable_ARV(void * addr, HVX_Vector vin) {
    vin                      = Q6_V_vlalign_VVR(vin, vin, (size_t) addr);  //rotate as needed.
    uint32_t       left_off  = unaligned_bytes(addr);
    uint32_t       right_off = left_off + _TyBytes;
    HVX_VectorPred qL_not    = Q6_Q_vsetq_R((size_t) addr);
    HVX_VectorPred qR        = Q6_Q_vsetq2_R(right_off);
    if (right_off > 128) {
        Q6_vmaskedstoreq_QAV(qR, (HVX_Vector *) addr + 1, vin);
        qR = Q6_Q_vcmp_eq_VbVb(vin, vin);  // all 1's
    }
    qL_not = Q6_Q_or_QQn(qL_not, qR);
    Q6_vmaskedstorenq_QAV(qL_not, (HVX_Vector *) addr, vin);
}

template <uint32_t _TyBytes> inline void q6op_vstu_variable_aligned(void * addr, HVX_Vector vin) {
    HVX_VectorPred qR = Q6_Q_vsetq2_R(_TyBytes);
    Q6_vmaskedstorenq_QAV(qR, (HVX_Vector *) addr, vin);
}

inline void q6op_vstu_variable_ARV(void * addr, int n, HVX_Vector vin) {
    vin                      = Q6_V_vlalign_VVR(vin, vin, (size_t) addr);  //rotate as needed.
    unsigned       left_off  = unaligned_bytes(addr);
    unsigned       right_off = left_off + n;
    HVX_VectorPred qL_not    = Q6_Q_vsetq_R((size_t) addr);
    HVX_VectorPred qR        = Q6_Q_vsetq2_R(right_off);
    if (right_off > 128) {
        Q6_vmaskedstoreq_QAV(qR, (HVX_Vector *) addr + 1, vin);
        qR = Q6_Q_vcmp_eq_VbVb(vin, vin);  // all 1's
    }
    qL_not = Q6_Q_or_QQn(qL_not, qR);
    Q6_vmaskedstorenq_QAV(qL_not, (HVX_Vector *) addr, vin);
}

inline HVX_Vector vec_reduction_qf32(HVX_Vector sums) {
    constexpr const size_t kFloatsPerVector = hexagon::kBytesPerVector / sizeof(float);
    static_assert(kFloatsPerVector == 32, "kFloatsPerVector should be 32");

    sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 16 * sizeof(float)));
    sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 8 * sizeof(float)));
    sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 4 * sizeof(float)));
    sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 2 * sizeof(float)));
    sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, sizeof(float)));
    return sums;
}

inline float vec_reduction_f32_qf32(HVX_Vector sums) {
    return get_flt0_from_fltv(Q6_Vsf_equals_Vqf32(vec_reduction_qf32(sums)));
}

inline HVX_Vector vec_reduction_qf16(HVX_Vector sums) {
    constexpr const size_t kFloatsPerVector = hexagon::kBytesPerVector / sizeof(npu_device_fp16_t);
    static_assert(kFloatsPerVector == 64, "kFloatsPerVector should be 64");

    sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 32 * sizeof(npu_device_fp16_t)));
    sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 16 * sizeof(npu_device_fp16_t)));
    sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 8 * sizeof(npu_device_fp16_t)));
    sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 4 * sizeof(npu_device_fp16_t)));
    sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 2 * sizeof(npu_device_fp16_t)));
    sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, sizeof(npu_device_fp16_t)));
    return sums;
}

inline float vec_reduction_qf16_f32(HVX_Vector sums) {
    HVX_Vector vect = Q6_Vhf_equals_Vqf16(vec_reduction_qf16(sums));
    uint16_t   i    = (vect[0] & 0xffff);
    return reinterpret_cast<__fp16 &>(i);
}

inline HVX_Vector hvx_scale_f32(float scale) {
    return Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(scale));
}

inline HVX_Vector hvx_vec_scale_f32_f32(HVX_Vector src, HVX_UVector *, HVX_Vector scale_vec) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(src, scale_vec));
}

inline HVX_Vector hvx_vec_mad_f32_f32(HVX_Vector src, HVX_UVector * dst_ptr, HVX_Vector scale_vec) {
    HVX_Vector dst = *dst_ptr;  // TODO: opt the unaligned case?
    src            = Q6_Vqf32_vmpy_VsfVsf(src, scale_vec);
    src            = Q6_Vqf32_vadd_Vqf32Vsf(src, dst);
    return Q6_Vsf_equals_Vqf32(src);
}

inline HVX_Vector hvx_scale_f16(float scale) {
    __fp16 f16_scale = scale;
    return Q6_Vh_vsplat_R(reinterpret_cast<const npu_device_fp16_t &>(f16_scale));
}

inline HVX_Vector hvx_vec_scale_f16_f16(HVX_Vector src, HVX_UVector *, HVX_Vector scale_vec) {
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(src, scale_vec));
}

inline HVX_Vector hvx_vec_mad_f16_f16(HVX_Vector src, HVX_UVector * dst_ptr, HVX_Vector scale_vec) {
    HVX_Vector dst    = *dst_ptr;  // TODO: opt the unaligned case?
    HVX_Vector scaled = Q6_Vqf16_vmpy_VhfVhf(src, scale_vec);
    HVX_Vector result = Q6_Vqf16_vadd_Vqf16Vhf(scaled, dst);
    return Q6_Vhf_equals_Vqf16(result);
}

inline HVX_Vector hvx_nop(float scale) {
    return HVX_Vector();
}

inline HVX_Vector hvx_passthru(HVX_Vector src, HVX_UVector *, HVX_Vector) {
    return src;
}

}  // namespace hexagon

#include "vec_math.inl"
#include "vec_ops.inl"
#include "vec_quant.inl"

namespace hexagon {

inline void vec_scale_f32(const float * src, float scale, float * dst, size_t count) {
    using namespace hexagon::vec;
    vec_scale_impl<hvx_vec_scale_f32_f32, hvx_scale_f32, float>(src, scale, dst, count);
}

inline void vec_mad_f32(const float * src, float scale, float * dst, size_t count) {
    using namespace hexagon::vec;
    vec_scale_impl<hvx_vec_mad_f32_f32, hvx_scale_f32, float>(src, scale, dst, count);
}

inline void vec_cpy_f32(const float * src, float * dst, size_t count) {
    using namespace hexagon::vec;
    vec_scale_impl<hvx_passthru, hvx_nop, float>(src, 0, dst, count);
}

inline void vec_zero_f32(float * src, size_t count) {
    using namespace hexagon::vec;
    vec_zero_impl<float>(src, count);
}

inline void vec_scale_f16(const npu_device_fp16_t * src, float scale, npu_device_fp16_t * dst, size_t count) {
    using namespace hexagon::vec;
    vec_scale_impl<hvx_vec_scale_f16_f16, hvx_scale_f16, npu_device_fp16_t>(src, scale, dst, count);
}

inline void vec_mad_f16(const npu_device_fp16_t * src, float scale, npu_device_fp16_t * dst, size_t count) {
    using namespace hexagon::vec;
    vec_scale_impl<hvx_vec_mad_f16_f16, hvx_scale_f16, npu_device_fp16_t>(src, scale, dst, count);
}

inline void vec_cpy_f16(const npu_device_fp16_t * src, npu_device_fp16_t * dst, size_t count) {
    using namespace hexagon::vec;
    vec_scale_impl<hvx_passthru, hvx_nop, npu_device_fp16_t>(src, 0, dst, count);
}

inline void vec_zero_f16(npu_device_fp16_t * src, size_t count) {
    using namespace hexagon::vec;
    vec_zero_impl<npu_device_fp16_t>(src, count);
}

template <typename _TElem0, typename _TElem1>
inline bool is_dot_product_aligned(const _TElem0 * src0, const _TElem1 * src1, size_t count) {
    static_assert(sizeof(_TElem0) <= sizeof(_TElem1), "src0 should be smaller than src1");

    if ((src0 && !hexagon::is_addr_aligned(src0)) || (src1 && !hexagon::is_addr_aligned(src1))) {
        return false;
    }

    if (count % (hexagon::kBytesPerVector / sizeof(_TElem0)) != 0) {
        return false;
    }

    return true;
}

inline HVX_Vector vec_dot_product_vqf32_f32_f32(const float * src0, const float * src1, size_t count) {
    using namespace hexagon::vec;
    return vec_dot_product_impl<float, HVX_Vector, vec_mpy_qf32, vec_add_qf32, vec_reduction_qf32>(src0, src1, count);
}

inline HVX_Vector vec_dot_product_aligned_vqf32_f32_f32(const float * src0, const float * src1, size_t count) {
    using namespace hexagon::vec;
    return vec_dot_product_aligned_impl<float, HVX_Vector, vec_mpy_qf32, vec_add_qf32, vec_reduction_qf32>(src0, src1,
                                                                                                           count);
}

inline float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count) {
    using namespace hexagon::vec;
    return vec_dot_product_impl<float, float, vec_mpy_qf32, vec_add_qf32, vec_reduction_f32_qf32>(src0, src1, count);
}

inline float vec_dot_product_aligned_f32_f32(const float * src0, const float * src1, size_t count) {
    using namespace hexagon::vec;
    return vec_dot_product_aligned_impl<float, float, vec_mpy_qf32, vec_add_qf32, vec_reduction_f32_qf32>(src0, src1,
                                                                                                          count);
}

inline bool is_f32_f32_dot_product_aligned(const float * src0, const float * src1, size_t count) {
    return is_dot_product_aligned<float, float>(src0, src1, count);
}

inline HVX_Vector vec_dot_product_vqf16_f16_f16(const npu_device_fp16_t * src0,
                                                const npu_device_fp16_t * src1,
                                                size_t                    count) {
    using namespace hexagon::vec;
    return vec_dot_product_impl<npu_device_fp16_t, HVX_Vector, vec_mpy_qf16, vec_add_qf16, vec_reduction_qf16>(
        src0, src1, count);
}

inline HVX_Vector vec_dot_product_aligned_vqf16_f16_f16(const npu_device_fp16_t * src0,
                                                        const npu_device_fp16_t * src1,
                                                        size_t                    count) {
    using namespace hexagon::vec;
    return vec_dot_product_aligned_impl<npu_device_fp16_t, HVX_Vector, vec_mpy_qf16, vec_add_qf16, vec_reduction_qf16>(
        src0, src1, count);
}

inline float vec_dot_product_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count) {
    using namespace hexagon::vec;
    return vec_dot_product_impl<npu_device_fp16_t, float, vec_mpy_qf16, vec_add_qf16, vec_reduction_qf16_f32>(
        src0, src1, count);
}

inline float vec_dot_product_aligned_f16_f16(const npu_device_fp16_t * src0,
                                             const npu_device_fp16_t * src1,
                                             size_t                    count) {
    using namespace hexagon::vec;
    return vec_dot_product_aligned_impl<npu_device_fp16_t, float, vec_mpy_qf16, vec_add_qf16, vec_reduction_qf16_f32>(
        src0, src1, count);
}

inline bool is_f16_f16_dot_product_aligned(const npu_device_fp16_t * src0,
                                           const npu_device_fp16_t * src1,
                                           size_t                    count) {
    return is_dot_product_aligned<npu_device_fp16_t, npu_device_fp16_t>(src0, src1, count);
}

inline HVX_Vector vec_dot_product_vqf32_f16_f32(const npu_device_fp16_t * src0, const float * src1, size_t count) {
    using namespace hexagon::vec;
    using namespace hexagon::vec::math;
    return vec_dot_product_mixed_impl<npu_device_fp16_t, float, HVX_Vector, hvx_vsf_convert_vhf, vec_mpy_qf32,
                                      vec_add_qf32, vec_reduction_qf32>(src0, src1, count);
}

inline HVX_Vector vec_dot_product_aligned_vqf32_f16_f32(const npu_device_fp16_t * src0,
                                                        const float *             src1,
                                                        size_t                    count) {
    using namespace hexagon::vec;
    using namespace hexagon::vec::math;
    return vec_dot_product_mix_aligned_impl<npu_device_fp16_t, float, HVX_Vector, hvx_vsf_convert_vhf, vec_mpy_qf32,
                                            vec_add_qf32, vec_reduction_qf32>(src0, src1, count);
}

inline float vec_dot_product_f16_f32(const npu_device_fp16_t * src0, const float * src1, size_t count) {
    using namespace hexagon::vec;
    using namespace hexagon::vec::math;
    return vec_dot_product_mixed_impl<npu_device_fp16_t, float, float, hvx_vsf_convert_vhf, vec_mpy_qf32, vec_add_qf32,
                                      vec_reduction_f32_qf32>(src0, src1, count);
}

inline float vec_dot_product_aligned_f16_f32(const npu_device_fp16_t * src0, const float * src1, size_t count) {
    using namespace hexagon::vec;
    using namespace hexagon::vec::math;
    return vec_dot_product_mix_aligned_impl<npu_device_fp16_t, float, float, hvx_vsf_convert_vhf, vec_mpy_qf32,
                                            vec_add_qf32, vec_reduction_f32_qf32>(src0, src1, count);
}

inline HVX_Vector vec_dot_product_vqf32_q40_f32(const npu_device_block_q4_0 * src0,
                                                const float *                 src1,
                                                size_t                        count,
                                                const HVX_Vector              table) {
    using namespace hexagon::vec;
    using namespace hexagon::vec::math;
    using namespace hexagon::vec::quant;

    alignas(hexagon::kBytesPerVector) static const HVX_Vector qs_indices =
        make_qs_load_mask<npu_device_block_q4_0, q4_qs_shuff_idx>();
    alignas(hexagon::kBytesPerVector) static const HVX_Vector scale_indices =
        Q6_Vh_vshuff_Vh(make_scale_load_mask<npu_device_block_q4_0>());

    return vec_dot_product_quant_impl<npu_device_block_q4_0, float, HVX_Vector, load_dequant_vec_q40_qf32_4blocks,
                                      load_dequant_vec_q40_qf32_2blocks, load_dequant_vec_q40_qf32_1block,
                                      vec_reduction_qf32>(src0, src1, count, qs_indices, scale_indices, table);
}

inline bool is_f16_f32_dot_product_aligned(const npu_device_fp16_t * src0, const float * src1, size_t count) {
    return is_dot_product_aligned<npu_device_fp16_t, float>(src0, src1, count);
}

template <typename _TFunc> struct dot_func_traits {};

template <typename _TData, typename _TReturn> struct dot_func_traits<_TReturn (*)(_TData, _TData, size_t)> {
    using param_type  = std::remove_const_t<std::remove_pointer_t<_TData>>;
    using return_type = _TReturn;
};

template <auto _DotFunc, typename _TReturn = typename dot_func_traits<decltype(_DotFunc)>::return_type>
_TReturn type_erase_dot_func(const void * src0, const void * src1, size_t count) {
    using param_type = typename dot_func_traits<decltype(_DotFunc)>::param_type;

    auto * src0_typed = reinterpret_cast<const param_type *>(src0);
    auto * src1_typed = reinterpret_cast<const param_type *>(src1);
    return _DotFunc(src0_typed, src1_typed, count);
}

inline HVX_Vector vec_silu_f32_f32(HVX_Vector x, HVX_VectorPair_x4 coeff) {
    using namespace hexagon::vec::math;

    constexpr float kMaxExp = 88.02f;  // log(INF)

    const HVX_Vector max_exp = Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(kMaxExp));
    HVX_Vector       one     = Q6_V_vsplat_R(0x3F800000);

    // x/(1.0f + expf(-x));
    HVX_Vector     exp_neg_x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(Q6_V_vzero(), x));
    HVX_VectorPred pred0     = Q6_Q_vcmp_gt_VsfVsf(exp_neg_x, max_exp);
    HVX_Vector     denom     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(qhmath_hvx_exp_vf(exp_neg_x), one));
    HVX_Vector     out       = qhmath_hvx_div_vf(x, denom, coeff);
    out                      = Q6_V_vmux_QVV(pred0, Q6_V_vzero(), out);
    return out;
}

inline HVX_Vector vec_silu_f16_f16(HVX_Vector x, HVX_VectorPair_x4 coeff) {
    using namespace hexagon::vec::math;

    constexpr __fp16 kMaxExp = 11.0898664f;  // log(INF)

    const HVX_Vector max_exp = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(kMaxExp));
    HVX_Vector       one     = Q6_Vh_vsplat_R(0x3c00);

    // x/(1.0f + expf(-x));
    HVX_Vector     exp_neg_x = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(Q6_V_vzero(), x));
    HVX_VectorPred pred0     = Q6_Q_vcmp_gt_VhfVhf(exp_neg_x, max_exp);
    HVX_Vector     denom     = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(qhmath_hvx_exp_vhf(exp_neg_x), one));
    HVX_Vector     out       = qhmath_hvx_div_vhf(x, denom, coeff);
    out                      = Q6_V_vmux_QVV(pred0, Q6_V_vzero(), out);
    return out;
}

inline HVX_Vector vec_swiglu_f32_f32(HVX_Vector x, HVX_Vector g, HVX_VectorPair_x4 coeff) {
    HVX_Vector silu = vec_silu_f32_f32(x, coeff);
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(silu, g));
}

inline HVX_Vector vec_swiglu_f16_f16(HVX_Vector x, HVX_Vector g, HVX_VectorPair_x4 coeff) {
    HVX_Vector silu = vec_silu_f16_f16(x, coeff);
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(silu, g));
}

}  // namespace hexagon
