#pragma once

#include "op_types.hpp"
#include "type_traits.hpp"
#include "vec_ops.hpp"

namespace hexagon {

template <HVX_Vector (*_OpBinaryTransform)(HVX_Vector, HVX_Vector)>
inline void vec_op_f32_f32(const float * src0, const float * src1, float * dst, size_t count) {
    using namespace hexagon::vec;
    vec_trans_impl<_OpBinaryTransform, float>(src0, src1, dst, count);
}

inline HVX_Vector vadd_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

inline HVX_Vector vsub_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

inline HVX_Vector vmul_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

template <HVX_Vector (*_OpBinaryTransform)(HVX_Vector, HVX_Vector)>
inline void vec_op_f16_f16(const npu_device_fp16_t * src0,
                           const npu_device_fp16_t * src1,
                           npu_device_fp16_t *       dst,
                           size_t                    count) {
    using namespace hexagon::vec;
    vec_trans_impl<_OpBinaryTransform, npu_device_fp16_t>(src0, src1, dst, count);
}

template <HVX_Vector (*_OpUnaryTransform)(HVX_VectorPair)>
inline void unary_vec_op_f16_f32(const float * src, npu_device_fp16_t * dst, size_t count, size_t) {
    // TODO: remove the unused param

    using namespace hexagon::vec;
    vec_trans_with_half_ret_impl<_OpUnaryTransform, float, npu_device_fp16_t>(src, dst, count);
}

inline HVX_Vector vadd_f16_f16(HVX_Vector a, HVX_Vector b) {
    // TODO: fix this since qf16 has less precision than fp16
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a, b));
}

inline HVX_Vector vsub_f16_f16(HVX_Vector a, HVX_Vector b) {
    // TODO: fix this since qf16 has less precision than fp16
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(a, b));
}

inline HVX_Vector vmul_f16_f16(HVX_Vector a, HVX_Vector b) {
    return Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b));
}

inline HVX_Vector vequals_f16_f32(HVX_VectorPair a) {
    const HVX_Vector kZeroV = Q6_V_vzero();
    HVX_Vector       lo     = Q6_Vqf32_vadd_Vqf32Vsf(kZeroV, Q6_V_lo_W(a));
    HVX_Vector       hi     = Q6_Vqf32_vadd_Vqf32Vsf(kZeroV, Q6_V_hi_W(a));
    a                       = Q6_W_vcombine_VV(hi, lo);
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(a));
}

template <typename T> struct get_data_type {};

template <typename _TyData> struct get_data_type<void (*)(const _TyData *, const _TyData *, _TyData *, size_t)> {
    using type = _TyData;
};

template <typename _TyInput, typename _TyOutput, typename _TyParam>
struct get_data_type<void (*)(const _TyInput *, _TyOutput *, size_t, _TyParam)> {
    using type        = _TyInput;
    using output_type = _TyOutput;
    using param_type  = typename std::remove_cv<typename std::remove_reference<_TyParam>::type>::type;
};

template <auto _RowFunc> bool element_wise_op(hexagon::tensor * out, hexagon::compute_params * params) {
    using data_type = typename get_data_type<decltype(_RowFunc)>::type;

    if (!out) {
        return false;
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "element_wise_op requires max dims 4");
    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    if (src0->get_ne(0) != src1->get_ne(0)) {
        // TODO: handle this case
        DEVICE_LOG_ERROR("src0[0] and src1[0] not match: %ld vs %ld\n", (long) src0->get_ne(0), (long) src1->get_ne(0));
        return false;
    }

    const auto total_rows = out->get_ne(3) * out->get_ne(2) * out->get_ne(1);
    const auto start_end  = params->get_work_slice(total_rows);
    if (start_end.first >= start_end.second) {
        return true;
    }

    const auto src_row_bytes         = src0->get_ne(0) * sizeof(data_type);
    const auto src_row_bytes_aligned = hexagon::get_aligned_size(src_row_bytes);
    uint8_t *  src_cache_ptr         = params->get_vtcm_cache(src_row_bytes_aligned * 4);
    if (!src_cache_ptr) {
        DEVICE_LOG_ERROR("element_wise_op: failed to get VTCM cache, size: %zu\n", size_t(src_row_bytes_aligned * 4));
        return false;
    }

    uint8_t * dst_ptr = out->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("element_wise_op: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) out,
                         hexagon::get_type_name(out->get_type()));
        return false;
    }

    const uint8_t * src0_ptr      = src0->get_read_buffer(true);  // TODO: avoid invalidation
    const uint8_t * src1_ptr      = src1->get_read_buffer(true);  // TODO: avoid invalidation
    const auto      rows_per_cube = out->get_ne(2) * out->get_ne(1);

    uint8_t * src0_read_cache_ptr  = src_cache_ptr;
    uint8_t * src0_write_cache_ptr = src_cache_ptr + src_row_bytes_aligned;
    uint8_t * src1_read_cache_ptr  = src_cache_ptr + src_row_bytes_aligned * 2;
    uint8_t * src1_write_cache_ptr = src_cache_ptr + src_row_bytes_aligned * 3;

    {
        const auto i03 = start_end.first / rows_per_cube;
        const auto i02 = start_end.first / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i01 = start_end.first % out->get_ne(1);  // TODO: should we use divide instead of mod?
        const auto i13 = i03 % src1->get_ne(3);
        const auto i12 = i02 % src1->get_ne(2);
        const auto i11 = i01 % src1->get_ne(1);

        auto * src0_row = src0_ptr + i03 * src0->get_nb(3) + i02 * src0->get_nb(2) + i01 * src0->get_nb(1);
        auto * src1_row = src1_ptr + i13 * src1->get_nb(3) + i12 * src1->get_nb(2) + i11 * src1->get_nb(1);
        if (!params->initiate_dma_row_transfer(src0_row, src0_write_cache_ptr, src1_row, src1_write_cache_ptr,
                                               src_row_bytes)) {
            DEVICE_LOG_ERROR("element_wise_op: failed to initiate dma transfer\n");
            return false;
        }
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(out, params->get_thread_index());

    for (int64_t ir = start_end.first; ir < start_end.second; ++ir) {
        const auto i03     = ir / rows_per_cube;
        const auto i02     = ir / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i01     = ir % out->get_ne(1);  // TODO: should we use divide instead of mod?
        const auto ir_next = ir + 1;

        auto * dst_row = dst_ptr + i03 * out->get_nb(3) + i02 * out->get_nb(2) + i01 * out->get_nb(1);
        {
            std::swap(src0_read_cache_ptr, src0_write_cache_ptr);
            std::swap(src1_read_cache_ptr, src1_write_cache_ptr);
            params->wait_for_dma();
        }

        if (ir_next < start_end.second) {
            const auto i03_next = ir_next / rows_per_cube;
            const auto i02_next = ir_next / out->get_ne(1) - i03_next * out->get_ne(2);
            const auto i01_next = ir_next % out->get_ne(1);
            const auto i13_next = i03_next % src1->get_ne(3);
            const auto i12_next = i02_next % src1->get_ne(2);
            const auto i11_next = i01_next % src1->get_ne(1);

            auto * src0_next_row =
                src0_ptr + i03_next * src0->get_nb(3) + i02_next * src0->get_nb(2) + i01_next * src0->get_nb(1);
            auto * src1_next_row =
                src1_ptr + i13_next * src1->get_nb(3) + i12_next * src1->get_nb(2) + i11_next * src1->get_nb(1);
            if (!params->initiate_dma_row_transfer(src0_next_row, src0_write_cache_ptr, src1_next_row,
                                                   src1_write_cache_ptr, src_row_bytes)) {
                DEVICE_LOG_ERROR("element_wise_op: failed to continue DMA transfer\n");
                return false;
            }
        }

        _RowFunc(reinterpret_cast<const data_type *>(src0_read_cache_ptr),
                 reinterpret_cast<const data_type *>(src1_read_cache_ptr), reinterpret_cast<data_type *>(dst_row),
                 static_cast<size_t>(out->get_ne(0)));
    }

    out->release_write_buffer();  // mark the output tensor as modified
    return true;
}

bool is_element_wise_op_supported(const npu_device_tensor_op_spec * op_spec,
                                  const npu_device_tensor_spec *    dst,
                                  const npu_device_tensor_spec *    srcs,
                                  size_t                            src_len) {
    const auto op = op_spec->op;
    if (op != NPU_OP_ADD && op != NPU_OP_SUB && op != NPU_OP_MUL) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (!dst || !srcs || src_len < 2) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", hexagon::op_get_name(op));
        return false;
    }

    const auto & src0 = srcs[0];
    const auto & src1 = srcs[1];
    if (dst->type != src0.type || dst->type != src1.type) {
        DEVICE_LOG_DEBUG("[%s]src0.type and dst.type mismatch: %s vs %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(src0.type), hexagon::get_type_name(dst->type));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32 && dst->type != NPU_DATA_TYPE_F16) {
        DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(dst->type));
        return false;
    }

    // TODO: fix FP16 add/sub
    if (dst->type == NPU_DATA_TYPE_F16 && op != NPU_OP_MUL) {
        DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(dst->type));
        return false;
    }

    if (src0.ne[0] != src1.ne[0]) {
        DEVICE_LOG_DEBUG("[%s]src0.ne[0] and src1.ne[0] not match: %ld vs %ld\n", hexagon::op_get_name(op),
                         (long) src0.ne[0], (long) src1.ne[0]);
        return false;
    }

    if (!hexagon::is_same_shape(src0, *dst)) {
        DEVICE_LOG_DEBUG("[%s]src0 and dst have different shape\n", hexagon::op_get_name(op));
        return false;
    }

    return true;
}

bool is_element_wise_op_required_sync(npu_device_tensor_op       prev_op,
                                      const npu_device_ne_type & prev_ne,
                                      npu_device_tensor_op       op,
                                      const npu_device_ne_type & ne) {
    NPU_UNUSED(prev_ne);
    NPU_UNUSED(op);
    NPU_UNUSED(ne);
    return prev_op != NPU_OP_ADD && prev_op != NPU_OP_SUB && prev_op != NPU_OP_MUL && prev_op != NPU_OP_RMS_NORM &&
           prev_op != NPU_OP_COUNT;
}

void rms_norm_vec_f32(const float * src, float * dst, size_t count, float eps) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(float);

    HVX_Vector *       src_vec_ptr = ((HVX_Vector *) src);
    HVX_Vector * const src_vec_end = ((HVX_Vector *) src) + (count / kElementsPerVector);
    HVX_Vector         prev        = *src_vec_ptr++;
    HVX_Vector         sum         = Q6_V_vzero();
    while (src_vec_ptr < src_vec_end) {
        HVX_Vector curr = *src_vec_ptr++;
        HVX_Vector s0   = Q6_V_valign_VVR(curr, prev, (size_t) src);
        sum             = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(s0, s0));
        prev            = curr;
    }

    const size_t leftover = count % kElementsPerVector;
    if ((src_vec_end - ((HVX_Vector *) src)) > 0) {
        // handle the last vector
        bool       should_fetch_src = leftover != 0 || !hexagon::is_addr_aligned(src_vec_ptr);
        HVX_Vector curr             = should_fetch_src ? *src_vec_ptr : prev;
        src_vec_ptr += should_fetch_src ? 1 : 0;
        HVX_Vector s0 = Q6_V_valign_VVR(curr, prev, (size_t) src);
        sum           = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(s0, s0));
        prev          = curr;
    }

    if (leftover > 0) {
        // handle the leftover elements
        const size_t leftover_bytes = leftover * sizeof(float);
        HVX_Vector   curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        sum  = Q6_Vqf32_vadd_Vqf32Vqf32(sum,
                                        Q6_V_valign_VVR(Q6_Vqf32_vmpy_VsfVsf(curr, curr), Q6_V_vzero(), leftover_bytes));
    }

    const float mean  = hexagon::vec_reduction_f32_qf32(sum) / count;  // TODO: figure out how to do division in vector
    const float scale = 1.0f / sqrtf(mean + eps);                      // TODO: use buildin blas sqrtf?
    hexagon::vec_scale_f32(src, scale, dst, count);
}

// TODO: merge with element_wise_op?
template <auto _RowFunc> bool unary_op(hexagon::tensor * out, hexagon::compute_params * params) {
    using input_type  = typename get_data_type<decltype(_RowFunc)>::type;
    using output_type = typename get_data_type<decltype(_RowFunc)>::output_type;
    using param_type  = typename get_data_type<decltype(_RowFunc)>::param_type;

    if (!out) {
        return false;
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "element_wise_op requires max dims 4");
    auto * src0 = out->get_src(0);
    if (!src0) {
        return true;  // skip if no src
    }

    auto * dst_ptr = out->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("unary_op: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) out,
                         hexagon::get_type_name(out->get_type()));
        return false;
    }

    const auto * src0_ptr      = src0->get_read_buffer();
    auto         total_rows    = out->get_ne(3) * out->get_ne(2) * out->get_ne(1);
    const auto   rows_per_cube = out->get_ne(2) * out->get_ne(1);
    const auto   start_end     = params->get_work_slice(total_rows);
    if (start_end.first >= start_end.second) {
        return true;
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(out, params->get_thread_index());

    const auto   param           = out->get_op_param<param_type>(0);
    const size_t valid_row_bytes = src0->get_ne(0) * sizeof(input_type);
    for (int64_t ir = start_end.first; ir < start_end.second; ++ir) {
        const auto i03 = ir / rows_per_cube;
        const auto i02 = ir / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i01 = ir % out->get_ne(1);  // TODO: should we use divide instead of mod?

        auto * src0_row = src0_ptr + i03 * src0->get_nb(3) + i02 * src0->get_nb(2) + i01 * src0->get_nb(1);
        auto * dst_row  = dst_ptr + i03 * out->get_nb(3) + i02 * out->get_nb(2) + i01 * out->get_nb(1);
        if (ir + 1 < start_end.second) {
            hexagon::l2fetch_row(src0_row + src0->get_nb(1), valid_row_bytes);
        }

        _RowFunc(reinterpret_cast<const input_type *>(src0_row), reinterpret_cast<output_type *>(dst_row),
                 static_cast<size_t>(out->get_ne(0)), param);
    }

    out->release_write_buffer();  // mark the output tensor as modified
    return true;
}

bool is_unary_op_supported(const npu_device_tensor_op_spec * op_spec,
                           const npu_device_tensor_spec *    dst,
                           const npu_device_tensor_spec *    srcs,
                           size_t                            src_len) {
    const auto op = op_spec->op;
    if (op != NPU_OP_RMS_NORM && op != NPU_OP_CPY) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (!dst || !srcs || src_len < 1) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", hexagon::op_get_name(op));
        return false;
    }

    const auto & src0 = srcs[0];
    if (!hexagon::is_same_shape(src0, *dst)) {
        DEVICE_LOG_DEBUG("[%s]src0 and dst have different shape\n", hexagon::op_get_name(op));
        return false;
    }

    if (op == NPU_OP_RMS_NORM) {
        if (dst->type != src0.type) {
            DEVICE_LOG_DEBUG("[%s]src0.type and dst.type mismatch: %s vs %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(src0.type), hexagon::get_type_name(dst->type));
            return false;
        }

        if (dst->type != NPU_DATA_TYPE_F32) {
            DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(dst->type));
            return false;
        }
    } else {
        if (dst->nb[1] < dst->nb[0] || src0.nb[1] < src0.nb[0]) {
            // TODO: support non-continuous row
            DEVICE_LOG_DEBUG("[%s]unsupported non-continuous row\n", hexagon::op_get_name(op));
            return false;
        }

        if (dst->type != NPU_DATA_TYPE_F16 || src0.type != NPU_DATA_TYPE_F32) {
            // TODO: support more types
            DEVICE_LOG_DEBUG("[%s]unsupported data type src:%s dst:%s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(src0.type), hexagon::get_type_name(dst->type));
            return false;
        }
    }

    return true;
}

bool is_unary_op_required_sync(npu_device_tensor_op       prev_op,
                               const npu_device_ne_type & prev_ne,
                               npu_device_tensor_op       op,
                               const npu_device_ne_type & ne) {
    NPU_UNUSED(prev_ne);
    NPU_UNUSED(op);
    NPU_UNUSED(ne);
    return prev_op != NPU_OP_ADD && prev_op != NPU_OP_SUB && prev_op != NPU_OP_MUL && prev_op != NPU_OP_RMS_NORM &&
           prev_op != NPU_OP_COUNT;
}

}  // namespace hexagon
