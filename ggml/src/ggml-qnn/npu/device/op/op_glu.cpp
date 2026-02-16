
#include "op_glu.hpp"

#include "type_traits.hpp"
#include "util.hpp"

namespace {

template <typename T> struct get_data_type {};

template <typename _TyData, typename _TyParam>
struct get_data_type<void (*)(const _TyData *, const _TyData *, _TyData *, size_t, _TyParam)> {
    using type       = _TyData;
    using param_type = typename std::remove_cv<typename std::remove_reference<_TyParam>::type>::type;
};

inline float dummy_load_coeff() {
    // This is a dummy function to satisfy the template requirements.
    // In practice, this should be replaced with a proper coefficient loading function.
    return 0;
}

inline float expf_f16_guard_inf(float x) {
    // Avoid overflow for large values, f16: log(65504)
    constexpr float kMaxExp = 11.0898664f;

    if (x >= kMaxExp) {
        // Avoid overflow for large values
        return std::numeric_limits<float>::infinity();
    }

    return std::expf(x);
}

inline void glu_vec_op_f16_f16(const __fp16 * src0, const __fp16 * src1, __fp16 * dst, size_t count, float coeff) {
    // TODO: use simd version, for some input hexagon intrinsics will generate nan instead of inf.
    for (uint32_t i = 0; i < count; ++i) {
        float x = src0[i];
        float g = src1[i];

        dst[i] = (x / (1.0f + expf_f16_guard_inf(-x))) * g;
    }
}

inline void glu_vec_op_f32_f32(const float *              src0,
                               const float *              src1,
                               float *                    dst,
                               size_t                     count,
                               hexagon::HVX_VectorPair_x4 coeff) {
    using namespace hexagon::vec;
    vec_trans_impl<hexagon::vec_swiglu_f32_f32, float, hexagon::HVX_VectorPair_x4>(src0, src1, dst, count, coeff);
}

template <auto _GluRowFunc, auto _CoeffLoadFunc>
bool glu_impl(hexagon::tensor * out, hexagon::compute_params * params) {
    using data_type  = typename get_data_type<decltype(_GluRowFunc)>::type;
    using param_type = typename get_data_type<decltype(_GluRowFunc)>::param_type;
    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "element_wise_op requires max dims 4");
    static_assert(std::is_same_v<param_type, decltype(_CoeffLoadFunc())>,
                  "GluRowFunc must have the same param type as CoeffLoadFunc");

    if (!out) {
        return false;
    }

    const bool has_src1 = out->get_src(1) != nullptr;
    auto *     src0     = out->get_src(0);
    auto *     src1     = has_src1 ? out->get_src(1) : src0;
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    const auto total_cols = has_src1 ? src0->get_ne(0) : src0->get_ne(0) / 2;
    if (out->get_ne(0) != total_cols) {
        DEVICE_LOG_ERROR("[hexagon-npu][GLU]out.ne[0] (%ld) != total_cols (%d)\n", (long) out->get_ne(0),
                         (int) total_cols);
        return false;
    }

    auto       total_rows    = out->get_ne(3) * out->get_ne(2) * out->get_ne(1);
    const auto rows_per_cube = out->get_ne(2) * out->get_ne(1);
    const auto start_end     = params->get_work_slice(total_rows);
    if (start_end.first >= start_end.second) {
        return true;
    }

    uint8_t * dst_ptr = out->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("[hexagon-npu][GLU]glu_impl: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) out,
                         hexagon::get_type_name(out->get_type()));
        return false;
    }

    const int32_t   swapped  = out->get_op_param<int32_t>(1);
    const uint8_t * src0_ptr = src0->get_read_buffer();
    const uint8_t * src1_ptr = has_src1 ? src1->get_read_buffer() : (src0_ptr + total_cols * sizeof(data_type));
    if (swapped) {
        std::swap(src0_ptr, src1_ptr);
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(out, params->get_thread_index());

    auto         coeff           = _CoeffLoadFunc();
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

        _GluRowFunc(reinterpret_cast<const data_type *>(src0_row), reinterpret_cast<const data_type *>(src1_row),
                    reinterpret_cast<data_type *>(dst_row), static_cast<size_t>(total_cols), coeff);
    }

    out->release_write_buffer();  // mark the output tensor as modified
    return true;
}

template <npu_device_tensor_data_type _DataType>
bool glu_compute(hexagon::tensor * out, hexagon::compute_params * params) {
    using namespace hexagon::vec::math;

    if (out->get_op_param<int32_t>(0) != NPU_GLU_OP_SWIGLU) {
        DEVICE_LOG_ERROR("Invalid GLU op type: %d\n", (int) out->get_op_param<int32_t>(0));
        return false;
    }

    if (out->get_type() != _DataType) {
        DEVICE_LOG_ERROR("GLU op type mismatch: %s vs %s\n", hexagon::get_type_name(out->get_type()),
                         hexagon::get_type_name(_DataType));
        return false;
    }

    if constexpr (_DataType == NPU_DATA_TYPE_F32) {
        return glu_impl<glu_vec_op_f32_f32, qhmath_load_div_sf_ltu>(out, params);
    } else if constexpr (_DataType == NPU_DATA_TYPE_F16) {
        return glu_impl<glu_vec_op_f16_f16, dummy_load_coeff>(out, params);
    }

    DEVICE_LOG_ERROR("Unsupported GLU data type: %s\n", hexagon::get_type_name(out->get_type()));
    return true;
}

}  // namespace

namespace hexagon {

bool glu_f32(hexagon::tensor * out, hexagon::compute_params * params) {
    return glu_compute<npu_device_tensor_data_type::NPU_DATA_TYPE_F32>(out, params);
}

bool glu_f16(hexagon::tensor * out, hexagon::compute_params * params) {
    return glu_compute<npu_device_tensor_data_type::NPU_DATA_TYPE_F16>(out, params);
}

bool is_glu_op_supported(const npu_device_tensor_op_spec * op_spec,
                         const npu_device_tensor_spec *    dst,
                         const npu_device_tensor_spec *    srcs,
                         size_t                            src_len) {
    const auto op = op_spec->op;
    if (op != NPU_OP_GLU) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (op_spec->params[0] != NPU_GLU_OP_SWIGLU) {
        DEVICE_LOG_DEBUG("[%s]unsupported GLU op type: %d\n", hexagon::op_get_name(op), (int) op_spec->params[0]);
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

    if (dst->type != NPU_DATA_TYPE_F32 && dst->type != NPU_DATA_TYPE_F16) {
        DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(dst->type));
        return false;
    }

    if (src_len > 1) {
        if (!hexagon::is_same_shape(src0, *dst) || !hexagon::is_same_shape(srcs[1], *dst)) {
            DEVICE_LOG_DEBUG("[%s]src0, src1 and dst have different shape\n", hexagon::op_get_name(op));
            return false;  // src0 and src1 have the same shape as dst
        }
    } else {
        static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "GLU requires max dims 4");
        if (src0.ne[0] / 2 != dst->ne[0] || src0.ne[1] != dst->ne[1] || src0.ne[2] != dst->ne[2] ||
            src0.ne[3] != dst->ne[3]) {
            DEVICE_LOG_DEBUG("[%s]src0 and dst have different shape: src0.ne[0]: %ld, dst.ne[0]: %ld\n",
                             hexagon::op_get_name(op), (long) src0.ne[0], (long) dst->ne[0]);
            return false;
        }
    }

    return true;
}

bool is_glu_required_sync(npu_device_tensor_op       prev_op,
                          const npu_device_ne_type & prev_ne,
                          npu_device_tensor_op       op,
                          const npu_device_ne_type & ne) {
    NPU_UNUSED(prev_ne);
    NPU_UNUSED(op);
    NPU_UNUSED(ne);
    return prev_op == NPU_OP_MUL_MAT;
}

}  // namespace hexagon
