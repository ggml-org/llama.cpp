#include "op_mul_mat.hpp"

#include "thread_pool.hpp"  // TODO: remove this dependency
#include "type_traits.hpp"
#include "vec_ops.hpp"
#include "vtcm_mem.hpp"

namespace {

template <typename T> struct get_data_type {};

template <typename _TyData0, typename _TyData1>
struct get_data_type<float (*)(const _TyData0 *, const _TyData1 *, size_t)> {
    using data_type0 = _TyData0;
    using data_type1 = _TyData1;
};

template <auto _DotFunc>
void mul_mat_impl(hexagon::tensor * src0, hexagon::tensor * src1, hexagon::tensor * dst,
                  hexagon::compute_params * params) {
    using data_type0 = typename get_data_type<decltype(_DotFunc)>::data_type0;
    using data_type1 = typename get_data_type<decltype(_DotFunc)>::data_type1;

    const bool is_quantized         = hexagon::is_quantized_type(src0->get_type());
    const auto src0_actual_row_size = hexagon::get_dequantized_row_size(src0);
    auto *     dequantize_row_func  = hexagon::get_type_traits(src0->get_type()).to_float;
    if (is_quantized && dequantize_row_func == nullptr) {
        DEVICE_LOG_ERROR("Unsupported quantized src0 type: %d, dequantize_row_func is null\n", src0->get_type());
        return;
    }

    const auto r02          = src1->get_ne(2) / src0->get_ne(2);
    const auto r03          = src1->get_ne(3) / src0->get_ne(3);
    const auto total_planes = dst->get_ne(3) * dst->get_ne(2);

    auto start_end_plane   = std::pair<int64_t, int64_t>{ 0, total_planes };
    auto start_end_row     = std::pair<int64_t, int64_t>{ 0, dst->get_ne(1) };
    auto start_end_element = std::pair<int64_t, int64_t>{ 0, dst->get_ne(0) };

    if (total_planes >= params->get_thread_count()) {
        start_end_plane = params->get_work_slice(total_planes);
    } else if (dst->get_ne(1) >= params->get_thread_count()) {
        start_end_row = params->get_work_slice(dst->get_ne(1));
    } else {
        start_end_element = params->get_work_slice(dst->get_ne(0));
    }

    if (start_end_plane.second <= start_end_plane.first || start_end_row.second <= start_end_row.first ||
        start_end_element.second <= start_end_element.first) {
        DEVICE_LOG_DEBUG(
            "mul_mat_impl: no work to do, start_end_plane: (%ld, %ld), start_end_row: (%ld, %ld), "
            "start_end_element: (%ld, %ld)\n",
            start_end_plane.first, start_end_plane.second, start_end_row.first, start_end_row.second,
            start_end_element.first, start_end_element.second);
        return;
    }

    // cache the src0 plane in VTCM
    size_t          src0_plane_slice_row_count = start_end_element.second - start_end_element.first;
    size_t          src0_plane_cache_size      = 0;
    uint8_t *       src0_plane_cache_ptr       = nullptr;
    const uint8_t * last_cached_plane_ptr      = nullptr;
    bool            is_mem_cache               = false;
    if (is_quantized) {
        src0_plane_slice_row_count =
            std::min(params->get_vtcm_quota_size() / src0_actual_row_size, src0_plane_slice_row_count);
        src0_plane_cache_size = src0_actual_row_size * src0_plane_slice_row_count;
        src0_plane_cache_ptr  = params->get_vtcm_cache(src0_plane_cache_size);
        if (src0_plane_cache_ptr == nullptr) {
            DEVICE_LOG_DEBUG(
                "mul_mat_impl: failed to get VTCM cache for src0, size: %zu, src0_plane_slice_row_count: %zu, "
                "src0_actual_row_size: %zu, will fallback to mem cache\n",
                src0_plane_cache_size, src0_plane_slice_row_count, src0_actual_row_size);
            src0_plane_cache_ptr = params->get_mem_cache(src0_plane_cache_size);
            is_mem_cache         = true;
        }
    }

    DEVICE_LOG_DEBUG(
        "mul_mat_impl src0_actual_row_size: %zu, src0_plane_slice_row_count: %zu, is_quantized: %d, vtcm_mem: "
        "%p(%zu)\n",
        src0_actual_row_size, src0_plane_slice_row_count, is_quantized, (void *) src0_plane_cache_ptr,
        src0_plane_cache_size);

    const size_t valid_row0_bytes = src0->get_ne(0) * sizeof(data_type0);
    const size_t valid_row1_bytes = src1->get_ne(0) * sizeof(data_type1);
    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(dst, params->get_thread_index(), mul_mat);

    uint8_t * dst_ptr = dst->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("mul_mat_impl: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) dst,
                         hexagon::get_type_name(dst->get_type()));
        return;
    }

    const uint8_t * src0_ptr = src0->get_read_buffer();
    const uint8_t * src1_ptr = src1->get_read_buffer();
    for (int64_t ip = start_end_plane.first; ip < start_end_plane.second; ip++) {
        const auto   i3         = ip / dst->get_ne(2);
        const auto   i2         = ip - i3 * dst->get_ne(2);
        const auto * src1_plane = src1_ptr + i3 * src1->get_nb(3) + i2 * src1->get_nb(2);
        auto *       dst_plane  = dst_ptr + i3 * dst->get_nb(3) + i2 * dst->get_nb(2);
        for (int64_t col_idx = start_end_element.first; col_idx < start_end_element.second;
             col_idx += src0_plane_slice_row_count) {
            const auto actual_row_count =
                std::min<int64_t>(src0_plane_slice_row_count,
                                  start_end_element.second - col_idx);  // number of rows in this slice
            const uint8_t * src0_plane =
                src0_ptr + i3 / r03 * src0->get_nb(3) + i2 / r02 * src0->get_nb(2) + col_idx * src0->get_nb(1);
            if (src0_plane_cache_ptr) {
                if (last_cached_plane_ptr != src0_plane) {
                    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 0, dequant);

                    for (int64_t ir = 0; ir < (int64_t) actual_row_count; ir++) {
                        auto * src0_row = src0_plane + ir * src0->get_nb(1);
                        if (ir + 1 < actual_row_count) {
                            hexagon::l2fetch_row(src0_row + src0->get_nb(1), src0->get_nb(1));
                        }

                        auto * dst_row = reinterpret_cast<hexagon::dequantized_element_type *>(
                            src0_plane_cache_ptr + ir * src0_actual_row_size);
                        dequantize_row_func(src0_row, reinterpret_cast<hexagon::dequantized_element_type *>(dst_row),
                                            src0->get_ne(0), params->f16_to_f32_table);
                    }

                    last_cached_plane_ptr = src0_plane;
                }

                src0_plane = src0_plane_cache_ptr;
            }

            for (int64_t i1 = start_end_row.first; i1 < start_end_row.second; i1++) {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 1, vec_dot);
                auto *  src1_row = src1_plane + i1 * src1->get_nb(1);
                auto *  dst_row  = reinterpret_cast<float *>(dst_plane + i1 * dst->get_nb(1)) + col_idx;
                int64_t i0       = 0;
                for (; i0 + 1 < (int64_t) actual_row_count; i0 += 2) {
                    auto * src0_row = src0_plane + i0 * src0_actual_row_size;
                    if (!src0_plane_cache_ptr || is_mem_cache) {
                        hexagon::l2fetch_row(src0_row + src0_actual_row_size, valid_row0_bytes);
                    }

                    // TODO: figure dst how to handle a entire row
                    dst_row[i0] = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row),
                                           reinterpret_cast<const data_type1 *>(src1_row), (size_t) src0->get_ne(0));
                    dst_row[i0 + 1] =
                        _DotFunc(reinterpret_cast<const data_type0 *>(src0_row + src0_actual_row_size),
                                 reinterpret_cast<const data_type1 *>(src1_row), (size_t) src0->get_ne(0));
                }

                if (ip + 1 < start_end_plane.second) {
                    hexagon::l2fetch_row(src1_row + src1->get_nb(1), valid_row1_bytes);
                }

                if (i0 < (int64_t) actual_row_count) {
                    auto * src0_row = src0_plane + i0 * src0_actual_row_size;
                    dst_row[i0]     = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row),
                                               reinterpret_cast<const data_type1 *>(src1_row), (size_t) src0->get_ne(0));
                }
            }
        }
    }

    dst->release_write_buffer();  // mark the output tensor as modified
}

bool is_quantized_mul_mat_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1) {
    if (src1.type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0.type(%s) and src1.type(%s) mismatch and src1 is not F32\n",
                         hexagon::get_type_name(src0.type), hexagon::get_type_name(src1.type));
        return false;
    }

    const auto type_traits = hexagon::get_type_traits(src0.type);
    if (!type_traits.is_quantized || type_traits.to_float == nullptr) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0.type(%s) and src1.type(%s) mismatch and src0 is not quantized\n",
                         hexagon::get_type_name(src0.type), hexagon::get_type_name(src1.type));
        return false;
    }

    if (src0.ne[0] % type_traits.blck_size) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0.type(%s) ne[0] is not aligned: %ld\n", hexagon::get_type_name(src0.type),
                         (long) src0.ne[0]);
        return false;
    }

    const auto vtcm_thread_quota_size = hexagon::default_thread_pool::get_per_thread_vtcm_quota();
    if (src0.ne[0] * sizeof(hexagon::dequantized_element_type) > vtcm_thread_quota_size) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0.type(%s) ne[0] is too large: %ld, vtcm_thread_quota_size: %zu\n",
                         hexagon::get_type_name(src0.type), (long) src0.ne[0], vtcm_thread_quota_size);
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT]supported quantized src0.type(%s) and src1.type(%s)\n",
                     hexagon::get_type_name(src0.type), hexagon::get_type_name(src1.type));
    return true;
}

}  // namespace

namespace hexagon {

bool mul_mat_f32(hexagon::tensor * out, compute_params * params) {
    if (!out) {
        return false;
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "mul_mat_f32 requires max dims 4");
    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    switch (src1->get_type()) {
        case NPU_DATA_TYPE_F32:
            if (src0->get_type() == NPU_DATA_TYPE_F16) {
                mul_mat_impl<hexagon::vec_dot_product_f16_f32>(src0, src1, out, params);
            } else {
                mul_mat_impl<hexagon::vec_dot_product_f32_f32>(src0, src1, out, params);
            }
            return true;

        case NPU_DATA_TYPE_F16:
            mul_mat_impl<hexagon::vec_dot_product_f16_f16>(src0, src1, out, params);
            return true;
        default:
            break;
    }

    DEVICE_LOG_ERROR("Unsupported src1 tensor type: %s\n", get_type_name(src1->get_type()));
    return false;
}

bool is_mul_mat_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                          const npu_device_tensor_spec * srcs, size_t src_len) {
    if (op != NPU_OP_MUL_MAT) {
        DEVICE_LOG_DEBUG("op is not MUL_MAT: %d\n", op);
        return false;
    }

    if (!dst || !srcs || src_len < 2) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", hexagon::op_get_name(op));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]dst type is not F32: %s\n", op_get_name(op), get_type_name(dst->type));
        return false;
    }

    const auto & src0 = srcs[0];
    const auto & src1 = srcs[1];
    if (src0.type != src1.type) {
#ifdef GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
        if (!is_quantized_mul_mat_supported(src0, src1)) {
            return false;
        }
#else
        DEVICE_LOG_DEBUG("[%s]src0.type(%s) and src1.type(%s) mismatch and quantized tensors are not supported\n",
                         op_get_name(op), get_type_name(src0.type), get_type_name(src1.type));
        return false;
#endif
    }

    if (src0.ne[0] != src1.ne[0] || src0.ne[1] != dst->ne[0]) {
        DEVICE_LOG_DEBUG("[%s]src0 and src1 cannot multiply: %ldx%ld vs %ldx%ld\n", op_get_name(op), (long) src0.ne[0],
                         (long) src0.ne[1], (long) src1.ne[0], (long) src1.ne[1]);
        return false;
    }

    if (src1.ne[1] != dst->ne[1] || src1.ne[2] != dst->ne[2] || src1.ne[3] != dst->ne[3]) {
        DEVICE_LOG_DEBUG("[%s]src1 and dst dimensions not match: %ldx%ld vs %ldx%ld\n", op_get_name(op),
                         (long) src1.ne[2], (long) src1.ne[3], (long) dst->ne[2], (long) dst->ne[3]);
        return false;
    }

    if (src1.ne[2] % src0.ne[2] || src1.ne[3] % src0.ne[3]) {
        DEVICE_LOG_DEBUG("[%s]src0 cannot broadcast to src1: %ldx%ld vs %ldx%ld\n", op_get_name(op), (long) src0.ne[2],
                         (long) src0.ne[3], (long) src1.ne[2], (long) src1.ne[3]);
        return false;
    }

    return true;
}

}  // namespace hexagon
