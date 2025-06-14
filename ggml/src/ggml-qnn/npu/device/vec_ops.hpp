#pragma once

#include <hexagon_types.h>
#include <HTP/core/intrinsics.h>

#include <cstdint>

#include "hexagon_npu.h"

namespace hexagon {

constexpr const size_t kBytesPerVector = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kAlignMask      = kBytesPerVector - 1;

inline size_t unaligned_bytes(const void * addr) {
    return ((size_t) addr) & kAlignMask;
}

inline bool is_addr_aligned(const void * addr) {
    return unaligned_bytes(addr) == 0;
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

/*
 * This function converts a vector of IEEE float elements to a vector of qf32 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vqf32_convert_vsf(HVX_Vector vin) {
    return Q6_Vqf32_vadd_VsfVsf(vin, Q6_V_vzero());
}

/*
 * This function converts a vector of IEEE half float elements to a vector of qf16 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vqf16_convert_vhf(HVX_Vector vin) {
    return Q6_Vqf16_vadd_VhfVhf(vin, Q6_V_vzero());
}

/*
 * This function converts a pair of vectors of qf32 elements to a vector of IEEE half float elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vhf_convert_vqf32(HVX_VectorPair vin_vp) {
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(vin_vp));
}

/*
 * This function converts a vector of qf16 elements to a pair of vectors of qf32 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_VectorPair qhmath_hvx_vqf32_convert_vqf16(HVX_Vector vxl) {
    HVX_VectorPair vxw_vp, exponent_vp;
    HVX_Vector     mantissa_mask = Q6_Vh_vsplat_R(0xffe0);
    HVX_Vector     exp_mask      = Q6_Vh_vsplat_R(0x1f);
    HVX_Vector     exp_offset    = Q6_Vh_vsplat_R(0x70);
    HVX_Vector     mant32_shift  = Q6_Vh_vsplat_R(0x10);
    HVX_Vector     reql, reqh, vxl_w, vxh_w, mantissa;
    HVX_Vector     el_exponent, eh_exponent;

    el_exponent = Q6_V_vand_VV(exp_mask, vxl);
    // Obtain the mantissa part: bits (5-15)
    mantissa    = Q6_V_vand_VV(mantissa_mask, vxl);
    // Convert qf16 biassed exponent to qf32 biased exponent
    // new exp = exp + ( 127 (qf32 bias) -15(qf16 biass) ) = 112
    el_exponent = Q6_Vh_vadd_VhVh(exp_offset, el_exponent);

    vxw_vp = Q6_Ww_vunpack_Vh(mantissa);
    vxl_w  = Q6_V_lo_W(vxw_vp);
    vxh_w  = Q6_V_hi_W(vxw_vp);

    exponent_vp = Q6_Ww_vunpack_Vh(el_exponent);
    el_exponent = Q6_V_lo_W(exponent_vp);
    eh_exponent = Q6_V_hi_W(exponent_vp);
    // Convert q16 mantiss to q32 mantissa
    reql        = Q6_Vw_vasl_VwVw(vxl_w, mant32_shift);
    reqh        = Q6_Vw_vasl_VwVw(vxh_w, mant32_shift);
    // Add the exponent
    vxl_w       = Q6_Vw_vadd_VwVw(reql, el_exponent);
    vxh_w       = Q6_Vw_vadd_VwVw(reqh, eh_exponent);

    return Q6_W_vcombine_VV(vxh_w, vxl_w);
}

inline HVX_VectorPair hvx_vqf32_convert_vhf(HVX_Vector vxl) {
    return qhmath_hvx_vqf32_convert_vqf16(qhmath_hvx_vqf16_convert_vhf(vxl));
}

inline HVX_Vector vec_reduction_qf32(HVX_Vector sums) {
    constexpr const size_t kFloatsPerVector = hexagon::kBytesPerVector / sizeof(float);
    static_assert(kFloatsPerVector == 32 || kFloatsPerVector == 16, "kFloatsPerVector should be 16 or 32");

    // TODO: do we have a better way to do the reduction?
    switch (kFloatsPerVector) {
        default:
        case 32:
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 16 * sizeof(float)));
            // fallthrough
        case 16:
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 8 * sizeof(float)));
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 4 * sizeof(float)));
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 2 * sizeof(float)));
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, sizeof(float)));
            break;
    }

    return sums;
}

inline float vec_reduction_qf32_f32(HVX_Vector sums) {
    return get_flt0_from_fltv(Q6_Vsf_equals_Vqf32(vec_reduction_qf32(sums)));
}

inline HVX_Vector vec_reduction_qf16(HVX_Vector sums) {
    constexpr const size_t kFloatsPerVector = hexagon::kBytesPerVector / sizeof(npu_device_fp16_t);
    static_assert(kFloatsPerVector == 64 || kFloatsPerVector == 32, "kFloatsPerVector should be 32 or 64");

    // TODO: do we have a better way to do the reduction?
    switch (kFloatsPerVector) {
        default:
        case 64:
            sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 32 * sizeof(npu_device_fp16_t)));
            // fallthrough
        case 32:
            sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 16 * sizeof(npu_device_fp16_t)));
            sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 8 * sizeof(npu_device_fp16_t)));
            sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 4 * sizeof(npu_device_fp16_t)));
            sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, 2 * sizeof(npu_device_fp16_t)));
            sums = Q6_Vqf16_vadd_Vqf16Vqf16(sums, Q6_V_vror_VR(sums, sizeof(npu_device_fp16_t)));
            break;
    }

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

template <HVX_Vector (*_Func)(HVX_Vector, HVX_UVector *, HVX_Vector), HVX_Vector (*_FuncScaleConvert)(float),
          typename _TParam>
inline void vec_scale_impl(const _TParam * src, float scale, _TParam * dst, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TParam);

    HVX_Vector *       src_vec_ptr    = ((HVX_Vector *) src);
    HVX_Vector * const src_vec_end    = ((HVX_Vector *) src) + (count / kElementsPerVector);
    HVX_UVector *      dst_vec_ptr    = ((HVX_UVector *) dst);  // TODO: opt the unaligned case?
    HVX_Vector         prev           = *src_vec_ptr++;
    const size_t       leftover       = count % kElementsPerVector;
    const size_t       leftover_bytes = leftover * sizeof(_TParam);

    HVX_Vector scale_vec = _FuncScaleConvert(scale);

    while (src_vec_end - src_vec_ptr > 1) {
        HVX_VectorPair curr = reinterpret_cast<HVX_VectorPair *>(src_vec_ptr)[0];
        src_vec_ptr += 2;

        HVX_Vector lo = Q6_V_valign_VVR(Q6_V_lo_W(curr), prev, (size_t) src);
        HVX_Vector hi = Q6_V_valign_VVR(Q6_V_hi_W(curr), Q6_V_lo_W(curr), (size_t) src);

        dst_vec_ptr[0] = _Func(lo, dst_vec_ptr, scale_vec);
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
        HVX_Vector curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, _Func(curr, dst_vec_ptr, scale_vec));
    }
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

inline void vec_scale_f32(const float * src, float scale, float * dst, size_t count) {
    vec_scale_impl<hvx_vec_scale_f32_f32, hvx_scale_f32, float>(src, scale, dst, count);
}

inline void vec_mad_f32(const float * src, float scale, float * dst, size_t count) {
    vec_scale_impl<hvx_vec_mad_f32_f32, hvx_scale_f32, float>(src, scale, dst, count);
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

inline void vec_scale_f16(const npu_device_fp16_t * src, float scale, npu_device_fp16_t * dst, size_t count) {
    vec_scale_impl<hvx_vec_scale_f16_f16, hvx_scale_f16, npu_device_fp16_t>(src, scale, dst, count);
}

inline void vec_mad_f16(const npu_device_fp16_t * src, float scale, npu_device_fp16_t * dst, size_t count) {
    vec_scale_impl<hvx_vec_mad_f16_f16, hvx_scale_f16, npu_device_fp16_t>(src, scale, dst, count);
}

float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count);
float vec_dot_product_aligned_f32_f32(const float * src0, const float * src1, size_t count);

float vec_dot_product_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count);
float vec_dot_product_aligned_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count);

float vec_dot_product_f16_f32(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count);

}  // namespace hexagon
