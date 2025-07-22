

#include "op_impl.hpp"

#include <type_traits>

#include "op_flash_attn.hpp"
#include "op_mul_mat.hpp"
#include "op_rope.hpp"
#include "type_traits.hpp"
#include "vec_ops.hpp"

namespace {

template <HVX_Vector (*_OpBinaryTransform)(HVX_Vector, HVX_Vector)>
inline void vec_op_f32_f32(const float * src0, const float * src1, size_t count, float * dst) {
    using namespace hexagon::vec;
    vec_trans_op_impl<_OpBinaryTransform, float>(src0, src1, count, dst);
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
inline void vec_op_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count,
                           npu_device_fp16_t * dst) {
    using namespace hexagon::vec;
    vec_trans_op_impl<_OpBinaryTransform, npu_device_fp16_t>(src0, src1, count, dst);
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

template <typename T> struct get_data_type {};

template <typename _TyData> struct get_data_type<void (*)(const _TyData *, const _TyData *, size_t, _TyData *)> {
    using type = _TyData;
};

template <typename _TyData, typename _TyParam>
struct get_data_type<void (*)(const _TyData *, size_t, _TyParam, _TyData *)> {
    using type       = _TyData;
    using param_type = typename std::remove_cv<typename std::remove_reference<_TyData>::type>::type;
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

    uint8_t * dst_ptr = out->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("element_wise_op: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) out,
                         hexagon::get_type_name(out->get_type()));
        return false;
    }

    const uint8_t * src0_ptr      = src0->get_read_buffer();
    const uint8_t * src1_ptr      = src1->get_read_buffer();
    auto            total_rows    = out->get_ne(3) * out->get_ne(2) * out->get_ne(1);
    const auto      rows_per_cube = out->get_ne(2) * out->get_ne(1);
    const auto      start_end     = params->get_work_slice(total_rows);
    if (start_end.first >= start_end.second) {
        return true;
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(out, params->get_thread_index());

    const size_t valid_row_bytes = src0->get_ne(0) * sizeof(data_type);
    for (int64_t ir = start_end.first; ir < start_end.second; ++ir) {
        const auto i03 = ir / rows_per_cube;
        const auto i02 = ir / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i01 = ir % out->get_ne(1);  // TODO: should we use divide instead of mod?
        const auto i13 = i03 % src1->get_ne(3);
        const auto i12 = i02 % src1->get_ne(2);
        const auto i11 = i01 % src1->get_ne(1);

        auto * src1_plane = src1_ptr + i13 * src1->get_nb(3) + i12 * src1->get_nb(2);
        auto * src0_row   = src0_ptr + i03 * src0->get_nb(3) + i02 * src0->get_nb(2) + i01 * src0->get_nb(1);
        auto * src1_row   = src1_plane + i11 * src1->get_nb(1);
        auto * dst_row    = dst_ptr + i03 * out->get_nb(3) + i02 * out->get_nb(2) + i01 * out->get_nb(1);
        if (ir + 1 < start_end.second) {
            hexagon::l2fetch_row(src0_row + src0->get_nb(1), valid_row_bytes);
            hexagon::l2fetch_row(src1_row + src1->get_nb(1), valid_row_bytes);
        }

        _RowFunc(reinterpret_cast<const data_type *>(src0_row), reinterpret_cast<const data_type *>(src1_row),
                 static_cast<size_t>(out->get_ne(0)), reinterpret_cast<data_type *>(dst_row));
    }

    out->release_write_buffer();  // mark the output tensor as modified
    return true;
}

bool is_element_wise_op_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                                  const npu_device_tensor_spec * srcs, size_t src_len) {
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

void rms_norm_vec_f32(const float * src, size_t count, float eps, float * dst) {
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
    using data_type  = typename get_data_type<decltype(_RowFunc)>::type;
    using param_type = typename get_data_type<decltype(_RowFunc)>::param_type;

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
    const size_t valid_row_bytes = src0->get_ne(0) * sizeof(data_type);
    for (int64_t ir = start_end.first; ir < start_end.second; ++ir) {
        const auto i03 = ir / rows_per_cube;
        const auto i02 = ir / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i01 = ir % out->get_ne(1);  // TODO: should we use divide instead of mod?

        auto * src0_row = src0_ptr + i03 * src0->get_nb(3) + i02 * src0->get_nb(2) + i01 * src0->get_nb(1);
        auto * dst_row  = dst_ptr + i03 * out->get_nb(3) + i02 * out->get_nb(2) + i01 * out->get_nb(1);
        if (ir + 1 < start_end.second) {
            hexagon::l2fetch_row(src0_row + src0->get_nb(1), valid_row_bytes);
        }

        _RowFunc(reinterpret_cast<const data_type *>(src0_row), static_cast<size_t>(out->get_ne(0)), param,
                 reinterpret_cast<data_type *>(dst_row));
    }

    out->release_write_buffer();  // mark the output tensor as modified
    return true;
}

bool is_unary_op_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                           const npu_device_tensor_spec * srcs, size_t src_len) {
    if (op != NPU_OP_RMS_NORM) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (!dst || !srcs || src_len < 1) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", hexagon::op_get_name(op));
        return false;
    }

    const auto & src0 = srcs[0];
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

    if (!hexagon::is_same_shape(src0, *dst)) {
        DEVICE_LOG_DEBUG("[%s]src0 and dst have different shape\n", hexagon::op_get_name(op));
        return false;
    }

    return true;
}

struct op_capabilities {
    npu_device_tensor_op               op;
    hexagon::op_is_supported_func_type is_supported;
    hexagon::compute_func_type         compute_funcs[NPU_DATA_TYPE_COUNT];
    bool                               requires_thread_barrier = false;
};

constexpr const op_capabilities kOpCapabilities[] = {
    {
     NPU_OP_MUL_MAT,                                                           hexagon::is_mul_mat_supported,
     {
            hexagon::mul_mat_f32,  // NPU_DATA_TYPE_F32
            nullptr,               // NPU_DATA_TYPE_F16
        },                                                                                                             true, // requires_thread_barrier
    },
    {
     NPU_OP_ADD,                                                                         is_element_wise_op_supported,
     {
            element_wise_op<vec_op_f32_f32<vadd_f32_f32>>,  // NPU_DATA_TYPE_F32
            element_wise_op<vec_op_f16_f16<vadd_f16_f16>>,  // NPU_DATA_TYPE_F16
        },                                                                                                                   false,                                                                               // requires_thread_barrier
    },
    {
     NPU_OP_SUB, is_element_wise_op_supported,
     {
            element_wise_op<vec_op_f32_f32<vsub_f32_f32>>,  // NPU_DATA_TYPE_F32
            element_wise_op<vec_op_f16_f16<vsub_f16_f16>>,  // NPU_DATA_TYPE_F16
        },                                                                                                             false,                                                                                                                       // requires_thread_barrier
    },
    {
     NPU_OP_MUL,                                                                   is_element_wise_op_supported,
     {
            element_wise_op<vec_op_f32_f32<vmul_f32_f32>>,  // NPU_DATA_TYPE_F32
            element_wise_op<vec_op_f16_f16<vmul_f16_f16>>,  // NPU_DATA_TYPE_F16
        },                                                      false,                                                                                                             // requires_thread_barrier
    },
    {
     NPU_OP_RMS_NORM,                                                                     is_unary_op_supported,
     {
            unary_op<rms_norm_vec_f32>,  // NPU_DATA_TYPE_F32
            nullptr,                     // NPU_DATA_TYPE_F16
        }, false,                           // requires_thread_barrier
    },
    {
     NPU_OP_FLASH_ATTN,hexagon::is_flash_attn_supported,
     {
            hexagon::flash_attn_f32,  // NPU_DATA_TYPE_F32
            nullptr,                  // NPU_DATA_TYPE_F16
        }, true,                         // requires_thread_barrier
    },
    {
     NPU_OP_ROPE,                                                        hexagon::is_rope_supported,
     {
            hexagon::rope_f32,  // NPU_DATA_TYPE_F32
            nullptr,            // NPU_DATA_TYPE_F16
        }, false,                  // requires_thread_barrier
    },
};

static_assert(kOpCapabilities[NPU_OP_MUL_MAT].compute_funcs[NPU_DATA_TYPE_F32] == hexagon::mul_mat_f32,
              "kOpArray[NPU_OP_MUL_MAT] != mul_mat_f32");

static_assert(std::size(kOpCapabilities) == NPU_OP_COUNT);
static_assert(kOpCapabilities[NPU_OP_MUL_MAT].op == NPU_OP_MUL_MAT, "kOpArray[NPU_OP_MUL_MAT].op != NPU_OP_MUL_MAT");
static_assert(kOpCapabilities[NPU_OP_MUL].op == NPU_OP_MUL, "kOpArray[NPU_OP_MUL].op != NPU_OP_MUL");
static_assert(kOpCapabilities[NPU_OP_RMS_NORM].op == NPU_OP_RMS_NORM,
              "kOpArray[NPU_OP_RMS_NORM].op != NPU_OP_RMS_NORM");
static_assert(kOpCapabilities[NPU_OP_FLASH_ATTN].op == NPU_OP_FLASH_ATTN,
              "kOpArray[NPU_OP_FLASH_ATTN].op != NPU_OP_FLASH_ATTN");
static_assert(kOpCapabilities[NPU_OP_ROPE].op == NPU_OP_ROPE, "kOpArray[NPU_OP_ROPE].op != NPU_OP_ROPE");

hexagon::compute_func_type get_compute_func_impl(npu_device_tensor_op op, npu_device_tensor_data_type type) {
    if (op >= NPU_OP_COUNT) {
        return nullptr;
    }

    return kOpCapabilities[op].compute_funcs[type];
}

}  // namespace

namespace hexagon {

compute_func_type get_compute_func(tensor * dst) {
    return get_compute_func_impl(dst->get_op(), dst->get_type());
}

bool requires_thread_barrier(npu_device_tensor_op op) {
    if (op >= NPU_OP_COUNT) {
        return false;
    }

    return kOpCapabilities[op].requires_thread_barrier;
}

bool support_op(npu_device_tensor_op op, const npu_device_tensor_spec * dst, const npu_device_tensor_spec * srcs,
                size_t src_len) {
    auto is_supported_func = kOpCapabilities[op].is_supported;
    if (!is_supported_func || !is_supported_func(op, dst, srcs, src_len)) {
        DEVICE_LOG_DEBUG("[%s]unsupported, is_supported_func return false\n", op_get_name(op));
        return false;
    }

    if (get_compute_func_impl(op, dst->type) == nullptr) {
        DEVICE_LOG_DEBUG("[%s]unsupported, get_compute_func failed, type: %s\n", op_get_name(op),
                         get_type_name(dst->type));
        return false;
    }

    return true;
}

}  // namespace hexagon
