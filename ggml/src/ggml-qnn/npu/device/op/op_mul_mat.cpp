#include "op_mul_mat.hpp"

#include "thread_pool.hpp"  // TODO: remove this dependency
#include "type_traits.hpp"
#include "vec_ops.hpp"

namespace {

inline std::pair<size_t, size_t> unflatten_i3_i2(size_t idx, const hexagon::tensor * t) {
    const auto i3 = idx / t->get_ne(2);
    const auto i2 = idx - i3 * t->get_ne(2);
    return { i3, i2 };
}

template <typename _T> struct get_data_type {};

template <typename _TData0, typename _TData1>
struct get_data_type<HVX_Vector (*)(const _TData0 *, const _TData1 *, size_t)> {
    using data_type0 = _TData0;
    using data_type1 = _TData1;
};

template <typename _TData0, typename _TData1>
struct get_data_type<HVX_Vector (*)(const _TData0 *, const _TData1 *, size_t, const HVX_Vector)> {
    using data_type0 = _TData0;
    using data_type1 = _TData1;
};

template <typename _TRet> struct convert_vector {};

template <> struct convert_vector<float> {
    static float convert(HVX_Vector vec) { return hexagon::get_flt0_from_fltv(Q6_Vsf_equals_Vqf32(vec)); }
};

inline std::pair<int64_t, int64_t> unflatten_i3_i2(int64_t idx, const hexagon::tensor * t) {
    const auto i3 = idx / t->get_ne(2);
    const auto i2 = idx - i3 * t->get_ne(2);
    return { i3, i2 };
}

template <> struct convert_vector<npu_device_fp16_t> {
    static float convert(HVX_Vector vec) {
        HVX_Vector vect = Q6_Vhf_equals_Vqf16(vec);
        uint16_t   i    = (vect[0] & 0xffff);
        return reinterpret_cast<__fp16 &>(i);
    }
};

template <bool _IsQuantized>
inline bool init_dma_transfer(hexagon::compute_params * params,
                              const uint8_t *           src,
                              uint8_t *                 dst,
                              size_t                    width,
                              size_t                    height,
                              size_t                    src_stride,
                              size_t                    dst_stride) {
    if constexpr (_IsQuantized) {
        if (!params->initiate_dma_row_transfer(src, dst, src_stride * height)) {
            return false;
        }
    } else {
        if (!params->initiate_dma_plane_transfer(src, dst, width, height, src_stride, dst_stride)) {
            return false;
        }
    }

    return true;
}

template <auto _DotFunc, typename... _TExtraArgs>
inline void batched_row_dot(const uint8_t * src0_plane,
                            const size_t    src0_ne0,
                            const size_t    src0_nb1,
                            const uint8_t * src1_row,
                            const size_t    src1_nb1,
                            float *         dst_row,
                            const size_t    slice_rows,
                            const size_t    src1_fetch_row_bytes,
                            _TExtraArgs... args) {
    using data_type0 = typename get_data_type<decltype(_DotFunc)>::data_type0;
    using data_type1 = typename get_data_type<decltype(_DotFunc)>::data_type1;

    size_t i0 = 0;
    for (; i0 + 1 < slice_rows; i0 += 2) {
        auto * src0_row = src0_plane + i0 * src0_nb1;

        // TODO: figure dst how to handle a entire row
        auto res0 = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row),
                             reinterpret_cast<const data_type1 *>(src1_row), src0_ne0, args...);

        // TODO: figure dst how to handle a entire row
        auto res1 = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row + src0_nb1),
                             reinterpret_cast<const data_type1 *>(src1_row), src0_ne0, args...);

        {
            dst_row[i0]     = convert_vector<data_type1>::convert(res0);
            dst_row[i0 + 1] = convert_vector<data_type1>::convert(res1);
        }
    }

    if (src1_fetch_row_bytes > 0) {
        hexagon::l2fetch_row(src1_row + src1_nb1, src1_fetch_row_bytes);
    }

    if (i0 < slice_rows) {
        auto * src0_row = src0_plane + i0 * src0_nb1;
        auto   res      = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row),
                                   reinterpret_cast<const data_type1 *>(src1_row), src0_ne0, args...);
        dst_row[i0]     = convert_vector<data_type1>::convert(res);
    }
}

template <auto _DotFunc, bool _IsSrcQuantized>
inline void mul_mat_impl(hexagon::tensor *         src0,
                         hexagon::tensor *         src1,
                         hexagon::tensor *         dst,
                         hexagon::compute_params * params) {
    using data_type0 = typename get_data_type<decltype(_DotFunc)>::data_type0;
    using data_type1 = typename get_data_type<decltype(_DotFunc)>::data_type1;

    const auto src0_row_stride         = hexagon::get_dequantized_row_size(src0);
    auto *     dequantize_row_func     = hexagon::get_type_traits(src0->get_type()).to_float;
    auto *     load_dequant_table_func = hexagon::get_type_traits(src0->get_type()).load_dequant_table;
    if (_IsSrcQuantized && dequantize_row_func == nullptr) {
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
    } else if (dst->get_ne(0) >= params->get_thread_count()) {
        start_end_element = params->get_work_slice(dst->get_ne(0));
    } else {
        start_end_row = params->get_work_slice(dst->get_ne(1));
    }

    if (start_end_plane.second <= start_end_plane.first || start_end_row.second <= start_end_row.first ||
        start_end_element.second <= start_end_element.first || start_end_plane.first < 0 || start_end_row.first < 0 ||
        start_end_element.first < 0) {
        DEVICE_LOG_DEBUG(
            "mul_mat_impl: no work to do, start_end_plane: (%lld, %lld), start_end_row: (%lld, %lld), "
            "start_end_element: (%lld, %lld)\n",
            start_end_plane.first, start_end_plane.second, start_end_row.first, start_end_row.second,
            start_end_element.first, start_end_element.second);
        return;
    }

    const uint8_t * src0_ptr = src0->get_read_buffer(true);  // TODO: avoid invalidation

                                                             // cache the src0 plane in VTCM
    const size_t valid_src0_row_bytes = _IsSrcQuantized ? src0->get_nb(1) : (src0->get_ne(0) * sizeof(data_type0));
    const size_t src1_row_stride      = hexagon::get_aligned_size(src1->get_nb(1));

    // TODO: figure out why we have to add padding after src0 plane cache
    const size_t src0_plane_slice_row_count =
        std::min<size_t>((params->get_vtcm_quota_size() - src1_row_stride) / (src0_row_stride * 2),
                         start_end_element.second - start_end_element.first);
    uint8_t *       src0_plane_read_cache_ptr     = nullptr;
    uint8_t *       src0_plane_write_cache_ptr    = nullptr;
    size_t          src0_plane_write_cache_offset = 0;
    const uint8_t * last_write_cached_plane_ptr   = nullptr;
    const uint8_t * last_read_cached_plane_ptr    = nullptr;

    {
        const size_t src0_plane_cache_size = src0_row_stride * src0_plane_slice_row_count;
        src0_plane_read_cache_ptr          = params->get_vtcm_cache(src0_plane_cache_size * 2);
        if (!src0_plane_read_cache_ptr) {
            DEVICE_LOG_ERROR(
                "mul_mat_impl: failed to get VTCM cache for src0, size: %zu, src0_plane_slice_row_count: %zu, "
                "src0_row_stride: %zu, will fallback to mem cache\n",
                src0_plane_cache_size, src0_plane_slice_row_count, src0_row_stride);
            return;
        }

        src0_plane_write_cache_ptr = src0_plane_read_cache_ptr + src0_plane_cache_size;
        if constexpr (_IsSrcQuantized) {
            src0_plane_write_cache_offset =
                src0_plane_cache_size - size_t(src0->get_nb(1) * src0_plane_slice_row_count);
        }

        DEVICE_LOG_DEBUG(
            "[%d]mul_mat_impl, src0_row_stride:%zu, valid_src0_row_bytes:%zu, src_nb0:%zu, "
            "slice_row_count:%zu, write_cache_offset: %zu, "
            "total_planes:%lld, planes:[%d,%d), rows:[%d,%d), elems:[%d,%d), is_quant:%d, "
            "vtcm_mem:%p(%zu)\n",
            (int) params->get_thread_index(), src0_row_stride, valid_src0_row_bytes, (size_t) src0->get_nb(1),
            src0_plane_slice_row_count, src0_plane_write_cache_offset, total_planes, (int) start_end_plane.first,
            (int) start_end_plane.second, (int) start_end_row.first, (int) start_end_row.second,
            (int) start_end_element.first, (int) start_end_element.second, _IsSrcQuantized,
            (void *) src0_plane_read_cache_ptr, params->get_vtcm_quota_size());
    }

    {
        const auto [i3, i2]        = unflatten_i3_i2(start_end_plane.first, dst);
        const uint8_t * src0_plane = src0_ptr + i3 / r03 * src0->get_nb(3) + i2 / r02 * src0->get_nb(2) +
                                     start_end_element.first * src0->get_nb(1);
        const size_t next_row_count =
            std::min<size_t>(src0_plane_slice_row_count,
                             start_end_element.second - start_end_element.first);  // number of rows in this slice
        if (!init_dma_transfer<_IsSrcQuantized>(
                params, src0_plane, src0_plane_write_cache_ptr + src0_plane_write_cache_offset, valid_src0_row_bytes,
                next_row_count, src0->get_nb(1), src0->get_nb(1))) {
            DEVICE_LOG_ERROR("mul_mat_impl: failed to continue dma transfer for src0 plane, is_quant: %d\n",
                             (int) _IsSrcQuantized);
            return;
        }

        DEVICE_LOG_DEBUG("mul_mat_impl: [i2,i3]:[%d,%d], src0_plane:%p, row_count:%zu\n", (int) i2, (int) i3,
                         (void *) src0_plane, next_row_count);

        last_write_cached_plane_ptr = src0_plane;
    }

    const size_t valid_src1_row_bytes =
        src0->get_ne(0) * sizeof(data_type1);  // src0 and src1 should have the same element count in the 1st dimension
    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(dst, params->get_thread_index(), mul_mat);

    uint8_t * dst_ptr = dst->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("[%d]mul_mat_impl: dst_ptr is not writable, tensor: %p, type: %s\n",
                         (int) params->get_thread_index(), (void *) dst, hexagon::get_type_name(dst->get_type()));
        return;
    }

    const uint8_t * src1_ptr      = src1->get_read_buffer();
    const auto      dequant_table = load_dequant_table_func ? load_dequant_table_func() : HVX_Vector();
    for (size_t ip = start_end_plane.first; ip < size_t(start_end_plane.second); ip++) {
        const auto [i3, i2]             = unflatten_i3_i2(ip, dst);
        const auto *    src1_plane      = src1_ptr + i3 * src1->get_nb(3) + i2 * src1->get_nb(2);
        auto *          dst_plane       = dst_ptr + i3 * dst->get_nb(3) + i2 * dst->get_nb(2);
        const uint8_t * src0_plane_base = src0_ptr + i3 / r03 * src0->get_nb(3) + i2 / r02 * src0->get_nb(2);
        for (size_t col_idx = start_end_element.first; col_idx < size_t(start_end_element.second);
             col_idx += src0_plane_slice_row_count) {
            const uint8_t * src0_plane = src0_plane_base + col_idx * src0->get_nb(1);
            const size_t    slice_rows =
                std::min<size_t>(src0_plane_slice_row_count,
                                 start_end_element.second - col_idx);  // number of rows in this slice

            {
                const uint8_t * src0_next_plane = last_write_cached_plane_ptr;
                size_t          next_row_count  = 0;
                if (col_idx + src0_plane_slice_row_count < start_end_element.second) {
                    const auto next_col_idx = col_idx + src0_plane_slice_row_count;
                    src0_next_plane         = src0_plane_base + next_col_idx * src0->get_nb(1);
                    next_row_count =
                        std::min<size_t>(src0_plane_slice_row_count,
                                         start_end_element.second - next_col_idx);  // number of rows in this slice
                } else if (ip + 1 < start_end_plane.second) {
                    // prefetch the next plane's first slice
                    const auto [i3_next, i2_next] = unflatten_i3_i2(ip + 1, dst);
                    const uint8_t * src0_next_plane_base =
                        src0_ptr + i3_next / r03 * src0->get_nb(3) + i2_next / r02 * src0->get_nb(2);
                    src0_next_plane = src0_next_plane_base + start_end_element.first * src0->get_nb(1);
                    next_row_count  = std::min<size_t>(
                        src0_plane_slice_row_count,
                        start_end_element.second - start_end_element.first);  // number of rows in this slice
                }

                if (last_read_cached_plane_ptr != src0_plane) {
                    std::swap(src0_plane_read_cache_ptr, src0_plane_write_cache_ptr);
                    params->wait_for_dma();
                }

                if (last_write_cached_plane_ptr != src0_next_plane) {
                    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 2, dma);
                    if (!init_dma_transfer<_IsSrcQuantized>(
                            params, src0_next_plane, src0_plane_write_cache_ptr + src0_plane_write_cache_offset,
                            valid_src0_row_bytes, next_row_count, src0->get_nb(1), src0->get_nb(1))) {
                        DEVICE_LOG_ERROR("mul_mat_impl: failed to continue dma transfer for src0 plane\n");
                        return;
                    }

                    last_write_cached_plane_ptr = src0_next_plane;
                }
            }

            if constexpr (_IsSrcQuantized) {
                if (last_read_cached_plane_ptr != src0_plane) {
                    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 0, dequant);
                    const uint8_t * src0_quant_plane = src0_plane_read_cache_ptr + src0_plane_write_cache_offset;
                    for (size_t ir = 0; ir < slice_rows; ir++) {
                        auto * src0_row       = src0_quant_plane + ir * src0->get_nb(1);
                        auto * cached_row_ptr = src0_plane_read_cache_ptr + ir * src0_row_stride;
                        dequantize_row_func(src0_row, reinterpret_cast<hexagon::dequant_output_type *>(cached_row_ptr),
                                            src0->get_ne(0), dequant_table);
                    }
                }
            }

            last_read_cached_plane_ptr = src0_plane;

            if (start_end_row.second > start_end_row.first) {
                hexagon::l2fetch_row(src1_plane + start_end_row.first * src1->get_nb(1), valid_src1_row_bytes);
            }

            for (size_t i1 = start_end_row.first; i1 < size_t(start_end_row.second); i1++) {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 1, dot);
                auto * src1_row = src1_plane + i1 * src1->get_nb(1);
                auto * dst_row  = reinterpret_cast<float *>(dst_plane + i1 * dst->get_nb(1)) + col_idx;
                batched_row_dot<_DotFunc>(src0_plane_read_cache_ptr, src0->get_ne(0), src0_row_stride, src1_row,
                                          src1->get_nb(1), dst_row, slice_rows,
                                          (ip + 1 < start_end_plane.second) ? valid_src1_row_bytes : 0);
            }
        }
    }

    dst->release_write_buffer();  // mark the output tensor as modified
}

template <auto _DotFunc, bool _IsSrcQuantized>
inline void mul_mat_gemv_impl(hexagon::tensor *         src0,
                              hexagon::tensor *         src1,
                              hexagon::tensor *         dst,
                              hexagon::compute_params * params) {
    using data_type0 = typename get_data_type<decltype(_DotFunc)>::data_type0;
    using data_type1 = typename get_data_type<decltype(_DotFunc)>::data_type1;

    auto * dequantize_row_func     = hexagon::get_type_traits(src0->get_type()).to_float;
    auto * load_dequant_table_func = hexagon::get_type_traits(src0->get_type()).load_dequant_table;
    if (_IsSrcQuantized && dequantize_row_func == nullptr) {
        DEVICE_LOG_ERROR("Unsupported quantized src0 type: %d, dequantize_row_func is null\n", src0->get_type());
        return;
    }

    if (dst->get_ne(0) < params->get_thread_count()) {
        DEVICE_LOG_ERROR("Unsupported src1 tensor shape for gemv: %s, ne: %lldx%lldx%lldx%lld\n",
                         hexagon::get_type_name(src1->get_type()), src1->get_ne(0), src1->get_ne(1), src1->get_ne(2),
                         src1->get_ne(3));
        return;
    }

    const auto start_end_element = params->get_work_slice(dst->get_ne(0));
    if (start_end_element.second <= start_end_element.first || start_end_element.first < 0) {
        DEVICE_LOG_DEBUG(
            "mul_mat_gemv_impl: no work to do, start_end_plane: [0, 1), start_end_row: [0, 1), "
            "start_end_element: [%lld, %lld)\n",
            start_end_element.first, start_end_element.second);
        return;
    }

    const auto      src0_row_stride      = hexagon::get_dequantized_row_size(src0);
    const uint8_t * src0_ptr             = src0->get_read_buffer(true);  // TODO: avoid invalidation
    const size_t    valid_src0_row_bytes = _IsSrcQuantized ? src0->get_nb(1) : (src0->get_ne(0) * sizeof(data_type0));

    // cache the src0 plane in VTCM
    const size_t src1_row_stride = hexagon::get_aligned_size(src1->get_nb(1));
    const size_t src0_plane_slice_row_count =
        std::min<size_t>((params->get_vtcm_quota_size() - src1_row_stride) / (src0_row_stride * 2),
                         start_end_element.second - start_end_element.first);

    uint8_t * src0_plane_read_cache_ptr     = nullptr;
    uint8_t * src0_plane_write_cache_ptr    = nullptr;
    size_t    src0_plane_write_cache_offset = 0;
    uint8_t * src1_row_cache_ptr            = nullptr;

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(dst, params->get_thread_index(), mul_mat);
    {
        const size_t src0_plane_cache_size = src0_row_stride * src0_plane_slice_row_count;
        src0_plane_read_cache_ptr          = params->get_vtcm_cache(src0_plane_cache_size * 2 + src1_row_stride);
        if (!src0_plane_read_cache_ptr) {
            DEVICE_LOG_ERROR(
                "mul_mat_gemv_impl: failed to get VTCM cache for src0, size: %zu, src0_plane_slice_row_count: %zu, "
                "src0_row_stride: %zu, will fallback to mem cache\n",
                src0_plane_cache_size, src0_plane_slice_row_count, src0_row_stride);
            return;
        }

        src0_plane_write_cache_ptr = src0_plane_read_cache_ptr + src0_plane_cache_size;
        src1_row_cache_ptr         = src0_plane_write_cache_ptr + src0_plane_cache_size;

        if constexpr (_IsSrcQuantized) {
            src0_plane_write_cache_offset = src0_plane_cache_size - (src0->get_nb(1) * src0_plane_slice_row_count);
        }

        DEVICE_LOG_DEBUG(
            "mul_mat_gemv_impl: src0_row_stride: %zu, src0_plane_slice_row_count: %zu, "
            "src0_plane_write_cache_offset: %zu, src0.nb[1]: %d, is_quantized: %d, vtcm_mem: %p(%zu)\n",
            src0_row_stride, src0_plane_slice_row_count, src0_plane_write_cache_offset, int(src0->get_nb(1)),
            _IsSrcQuantized, (void *) src0_plane_read_cache_ptr, src0_plane_cache_size);
    }

    uint8_t * dst_ptr = dst->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("mul_mat_gemv_impl: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) dst,
                         hexagon::get_type_name(dst->get_type()));
        return;
    }

    const uint8_t * src1_ptr = src1->get_read_buffer();

    {
        if (!params->initiate_dma_row_transfer(src1_ptr, src1_row_cache_ptr, src1->get_ne(0) * sizeof(data_type1))) {
            DEVICE_LOG_ERROR("mul_mat_gemv_impl: failed to initiate dma transfer for src1\n");
            return;
        }

        const uint8_t * src0_plane = src0_ptr + start_end_element.first * src0->get_nb(1);
        const size_t    next_row_count =
            std::min<size_t>(src0_plane_slice_row_count,
                             start_end_element.second - start_end_element.first);  // number of rows in this slice
        params->wait_for_dma();

        if (!init_dma_transfer<_IsSrcQuantized>(
                params, src0_plane, src0_plane_write_cache_ptr + src0_plane_write_cache_offset, valid_src0_row_bytes,
                next_row_count, src0->get_nb(1), src0->get_nb(1))) {
            DEVICE_LOG_ERROR("mul_mat_gemv_impl: failed to initiate dma plane transfer for src0 plane, is_quant: %d\n",
                             (int) _IsSrcQuantized);
            return;
        }
    }

    const auto dequant_table = load_dequant_table_func ? load_dequant_table_func() : HVX_Vector();
    {
        for (size_t col_idx = start_end_element.first; col_idx < size_t(start_end_element.second);
             col_idx += src0_plane_slice_row_count) {
            const size_t slice_rows =
                std::min<size_t>(src0_plane_slice_row_count,
                                 start_end_element.second - col_idx);  // number of rows in this slice
            const size_t next_col_idx = col_idx + src0_plane_slice_row_count;
            std::swap(src0_plane_read_cache_ptr, src0_plane_write_cache_ptr);
            params->wait_for_dma();

            if (next_col_idx < start_end_element.second) {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 2, dma);
                const uint8_t * src0_next_plane = src0_ptr + next_col_idx * src0->get_nb(1);
                const size_t    next_row_count =
                    std::min<size_t>(src0_plane_slice_row_count,
                                     start_end_element.second - next_col_idx);  // number of rows in this slice
                if (!init_dma_transfer<_IsSrcQuantized>(
                        params, src0_next_plane, src0_plane_write_cache_ptr + src0_plane_write_cache_offset,
                        valid_src0_row_bytes, next_row_count, src0->get_nb(1), src0->get_nb(1))) {
                    DEVICE_LOG_ERROR(
                        "mul_mat_gemv_impl: failed to continue dma plane transfer for src0 plane, is_quant: %d\n",
                        (int) _IsSrcQuantized);
                    return;
                }
            }

            if constexpr (_IsSrcQuantized) {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 0, dequant);
                const uint8_t * src0_quant_plane = src0_plane_read_cache_ptr + src0_plane_write_cache_offset;
                for (size_t ir = 0; ir < slice_rows; ir++) {
                    auto * src0_row       = src0_quant_plane + ir * src0->get_nb(1);
                    auto * cached_row_ptr = src0_plane_read_cache_ptr + ir * src0_row_stride;
                    dequantize_row_func(src0_row, reinterpret_cast<hexagon::dequant_output_type *>(cached_row_ptr),
                                        src0->get_ne(0), dequant_table);
                }
            }

            {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 1, dot);
                auto * dst_row = reinterpret_cast<float *>(dst_ptr) + col_idx;
                batched_row_dot<_DotFunc>(src0_plane_read_cache_ptr, src0->get_ne(0), src0_row_stride,
                                          src1_row_cache_ptr, src1->get_nb(1), dst_row, slice_rows, 0);
            }
        }
    }

    dst->release_write_buffer();  // mark the output tensor as modified
}

template <auto _DotFunc>
inline void mul_mat_gemv_quant_impl(hexagon::tensor *         src0,
                                    hexagon::tensor *         src1,
                                    hexagon::tensor *         dst,
                                    hexagon::compute_params * params) {
    // TODO: merge with mul_mat_gemv_impl?

    using data_type1 = typename get_data_type<decltype(_DotFunc)>::data_type1;

    if (dst->get_ne(0) < params->get_thread_count()) {
        DEVICE_LOG_ERROR("Unsupported src1 tensor shape for gemv: %s, ne: %lldx%lldx%lldx%lld\n",
                         hexagon::get_type_name(src1->get_type()), src1->get_ne(0), src1->get_ne(1), src1->get_ne(2),
                         src1->get_ne(3));
        return;
    }

    const auto src0_row_stride   = src0->get_nb(1);
    const auto start_end_element = params->get_work_slice(dst->get_ne(0));
    if (start_end_element.second <= start_end_element.first || start_end_element.first < 0) {
        DEVICE_LOG_DEBUG(
            "mul_mat_gemv_quant_impl: no work to do, start_end_plane: [0, 1), start_end_row: [0, 1), "
            "start_end_element: [%lld, %lld)\n",
            start_end_element.first, start_end_element.second);
        return;
    }

    const uint8_t * src0_ptr = src0->get_read_buffer(true);  // TODO: avoid invalidation

    // cache the src0 plane in VTCM
    const size_t src1_row_stride = hexagon::get_aligned_size(src1->get_nb(1));
    const size_t src0_plane_slice_row_count =
        std::min<size_t>((params->get_vtcm_quota_size() - src1_row_stride) / (src0_row_stride * 2),
                         start_end_element.second - start_end_element.first);

    uint8_t * src0_plane_read_cache_ptr  = nullptr;
    uint8_t * src0_plane_write_cache_ptr = nullptr;
    uint8_t * src1_row_cache_ptr         = nullptr;

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(dst, params->get_thread_index(), mul_mat);
    {
        const size_t src0_plane_cache_size = src0_row_stride * src0_plane_slice_row_count;
        src0_plane_read_cache_ptr          = params->get_vtcm_cache(src0_plane_cache_size * 2 + src1_row_stride);
        if (!src0_plane_read_cache_ptr) {
            DEVICE_LOG_ERROR(
                "mul_mat_gemv_quant_impl: failed to get VTCM cache for src0, size: %zu, src0_plane_slice_row_count: "
                "%zu, "
                "src0_row_stride: %zu, will fallback to mem cache\n",
                src0_plane_cache_size, src0_plane_slice_row_count, src0_row_stride);
            return;
        }

        src0_plane_write_cache_ptr = src0_plane_read_cache_ptr + src0_plane_cache_size;
        src1_row_cache_ptr         = src0_plane_write_cache_ptr + src0_plane_cache_size;

        DEVICE_LOG_DEBUG(
            "mul_mat_gemv_quant_impl: src0_row_stride: %zu, src0_plane_slice_row_count: %zu, src0.nb[1]: %d, vtcm_mem: "
            "%p(%zu)\n",
            src0_row_stride, src0_plane_slice_row_count, int(src0->get_nb(1)), (void *) src0_plane_read_cache_ptr,
            src0_plane_cache_size);
    }

    uint8_t * dst_ptr = dst->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("mul_mat_gemv_quant_impl: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) dst,
                         hexagon::get_type_name(dst->get_type()));
        return;
    }

    const uint8_t * src1_ptr = src1->get_read_buffer();

    {
        if (!params->initiate_dma_row_transfer(src1_ptr, src1_row_cache_ptr, src1->get_ne(0) * sizeof(data_type1))) {
            DEVICE_LOG_ERROR("mul_mat_gemv_quant_impl: failed to initiate dma transfer for src1\n");
            return;
        }

        const uint8_t * src0_plane = src0_ptr + start_end_element.first * src0_row_stride;
        const size_t    next_row_count =
            std::min<size_t>(src0_plane_slice_row_count,
                             start_end_element.second - start_end_element.first);  // number of rows in this slice
        params->wait_for_dma();

        if (!init_dma_transfer<true>(params, src0_plane, src0_plane_write_cache_ptr, src0_row_stride, next_row_count,
                                     src0_row_stride, src0_row_stride)) {
            DEVICE_LOG_ERROR("mul_mat_gemv_quant_impl: failed to initiate dma plane transfer for src0 plane\n");
            return;
        }
    }

    auto *     load_dequant_table_func = hexagon::get_type_traits(src0->get_type()).load_dequant_table;
    const auto dequant_table           = load_dequant_table_func ? load_dequant_table_func() : HVX_Vector();
    {
        for (size_t col_idx = start_end_element.first; col_idx < size_t(start_end_element.second);
             col_idx += src0_plane_slice_row_count) {
            const size_t slice_rows =
                std::min<size_t>(src0_plane_slice_row_count,
                                 start_end_element.second - col_idx);  // number of rows in this slice
            const size_t next_col_idx = col_idx + src0_plane_slice_row_count;
            std::swap(src0_plane_read_cache_ptr, src0_plane_write_cache_ptr);
            params->wait_for_dma();

            if (next_col_idx < start_end_element.second) {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 1, dma);
                const uint8_t * src0_next_plane = src0_ptr + next_col_idx * src0_row_stride;
                const size_t    next_row_count =
                    std::min<size_t>(src0_plane_slice_row_count,
                                     start_end_element.second - next_col_idx);  // number of rows in this slice
                if (!init_dma_transfer<true>(params, src0_next_plane, src0_plane_write_cache_ptr, src0_row_stride,
                                             next_row_count, src0_row_stride, src0_row_stride)) {
                    DEVICE_LOG_ERROR("mul_mat_gemv_quant_impl: failed to continue dma plane transfer for src0 plane\n");
                    return;
                }
            }

            {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 0, dot);
                auto * dst_row = reinterpret_cast<float *>(dst_ptr) + col_idx;
                batched_row_dot<_DotFunc, const HVX_Vector>(src0_plane_read_cache_ptr, src0->get_ne(0), src0_row_stride,
                                                            src1_row_cache_ptr, src1->get_nb(1), dst_row, slice_rows, 0,
                                                            dequant_table);
            }
        }
    }

    dst->release_write_buffer();  // mark the output tensor as modified
}

bool is_src_cacheable(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1) {
    const auto & src0_type_traits = hexagon::get_type_traits(src0.type);
    if (src0_type_traits.to_float == nullptr) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0.type(%s) cannot be cached, to_float is null\n",
                         hexagon::get_type_name(src0.type));
        return false;
    }

    const auto   vtcm_thread_quota_size = hexagon::default_thread_pool::get_per_thread_vtcm_quota();
    const size_t src0_type_size =
        src0_type_traits.is_quantized ? sizeof(hexagon::dequant_output_type) : src0_type_traits.type_size;
    const auto & src1_type_traits = hexagon::get_type_traits(src1.type);
    const bool   is_gemv          = src1.ne[1] == 1 && src1.ne[2] == 1 && src1.ne[3] == 1;
    size_t       min_cache_size   = is_gemv ? (src1.ne[0] * src1_type_traits.type_size) : 0;
    min_cache_size += src0.ne[0] * src0_type_size;
    if (min_cache_size > vtcm_thread_quota_size) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0.type(%s) min_cache_size is too large: %ld, vtcm_thread_quota_size: %zu\n",
                         hexagon::get_type_name(src0.type), (long) min_cache_size, vtcm_thread_quota_size);
        return false;
    }

    return true;
}

bool is_quantized_mul_mat_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1) {
    if (src1.type != NPU_DATA_TYPE_F32 && src1.type != NPU_DATA_TYPE_F16) {
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

    if (!is_src_cacheable(src0, src1)) {
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT]supported quantized src0.type(%s) and src1.type(%s)\n",
                     hexagon::get_type_name(src0.type), hexagon::get_type_name(src1.type));
    return true;
}

bool is_mul_mat_f16_f32_src_tensors_aligned(hexagon::tensor * src0,
                                            hexagon::tensor * src1,
                                            bool              is_src0_cached,
                                            bool              is_src1_cached) {
    const auto * src1_ptr = is_src1_cached ? nullptr : src1->get_read_buffer_as<float>();
    const auto * src0_ptr = is_src0_cached ? nullptr : src0->get_read_buffer_as<npu_device_fp16_t>();

    if (!hexagon::is_f16_f32_dot_product_aligned(src0_ptr, src1_ptr, src0->get_ne(0))) {
        DEVICE_LOG_DEBUG("[MUL_MAT][f16_f32]src_tensors_unaligned: ne[0]: %ld\n", (long) src0->get_ne(0));
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT][f16_f32]src_tensors_aligned: ne[0]: %ld\n", (long) src0->get_ne(0));
    return true;
}

bool is_mul_mat_f16_f16_src_tensors_aligned(hexagon::tensor * src0, hexagon::tensor * src1, bool is_src0_quantized) {
    const auto * src1_ptr = src1->get_read_buffer_as<npu_device_fp16_t>();
    const auto * src0_ptr = is_src0_quantized ? nullptr : src0->get_read_buffer_as<npu_device_fp16_t>();

    if (!hexagon::is_f16_f16_dot_product_aligned(src0_ptr, src1_ptr, src0->get_ne(0))) {
        DEVICE_LOG_DEBUG("[MUL_MAT][f16_f16]src_tensors_unaligned: ne[0]: %ld\n", (long) src0->get_ne(0));
        return false;
    }

    if (!is_src0_quantized && !hexagon::is_size_aligned(src0->get_nb(1))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0 tensor nb[1] is not aligned: %zu\n", src0->get_nb(1));
        return false;
    }

    if (!hexagon::is_size_aligned(src1->get_nb(1))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src1 tensor nb[1] is not aligned: %zu\n", src1->get_nb(1));
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT]src_tensors_aligned: ne[0]: %ld\n", (long) src0->get_ne(0));
    return true;
}

bool is_mul_mat_f32_f32_src_tensors_aligned(hexagon::tensor * src0, hexagon::tensor * src1) {
    const auto * src1_ptr = src1->get_read_buffer_as<float>();
    const auto * src0_ptr = src0->get_read_buffer_as<float>();

    if (!hexagon::is_f32_f32_dot_product_aligned(src0_ptr, src1_ptr, src0->get_ne(0))) {
        DEVICE_LOG_DEBUG("[MUL_MAT][f32_f32]src_tensors_unaligned: ne[0]: %ld\n", (long) src0->get_ne(0));
        return false;
    }

    if (!hexagon::is_size_aligned(src0->get_nb(1))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0 tensor nb[1] is not aligned: %zu\n", src0->get_nb(1));
        return false;
    }

    if (!hexagon::is_size_aligned(src1->get_nb(1))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src1 tensor nb[1] is not aligned: %zu\n", src1->get_nb(1));
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT]src_tensors_aligned: ne[0]: %ld\n", (long) src0->get_ne(0));
    return true;
}

typedef void (*mul_mat_func_type)(hexagon::tensor *         src0,
                                  hexagon::tensor *         src1,
                                  hexagon::tensor *         dst,
                                  hexagon::compute_params * params);

constexpr const size_t kMulMatGemvBaseIndex = 2;

constexpr const mul_mat_func_type kMulMatF32F32Funcs[4] = {
    // quantized and non-quantized
    mul_mat_impl<hexagon::vec_dot_product_vqf32_f32_f32, false>,               // F32 * F32 unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf32_f32_f32, false>,       // F32 * F32 aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_vqf32_f32_f32, false>,          // F32 * F32 gemv
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf32_f32_f32, false>,  // F32 * F32 gemv
};

constexpr const mul_mat_func_type kMulMatF16F32QuantizedFuncs[4] = {
    // quantized and non-quantized
    mul_mat_impl<hexagon::vec_dot_product_vqf32_f16_f32, true>,               // F16 * F32 quantized unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf32_f16_f32, true>,       // F16 * F32 quantized aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_vqf32_f16_f32, true>,          // F16 * F32 quantized unaligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf32_f16_f32, true>,  // F16 * F32 quantized aligned
};

constexpr const mul_mat_func_type kMulMatF16F32Funcs[4] = {
    // quantized and non-quantized
    mul_mat_impl<hexagon::vec_dot_product_vqf32_f16_f32, false>,               // F16 * F32 unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf32_f16_f32, false>,       // F16 * F32 aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_vqf32_f16_f32, false>,          // F16 * F32 unaligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf32_f16_f32, false>,  // F16 * F32 aligned
};

constexpr const mul_mat_func_type kMulMatF16QuantizedFuncs[4] = {
    mul_mat_impl<hexagon::vec_dot_product_vqf16_f16_f16, true>,               // F16 * F16 quantized unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, true>,       // F16 * F16 quantized aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, true>,  // F16 * F16 quantized gemv
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, true>,  // F16 * F16 quantized gemv
};

constexpr const mul_mat_func_type kMulMatF16Funcs[4] = {
    mul_mat_impl<hexagon::vec_dot_product_vqf16_f16_f16, false>,               // F16 * F16 unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, false>,       // F16 * F16 aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_vqf16_f16_f16, false>,          // F16 * F16 gemv
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, false>,  // F16 * F16 gemv
};

}  // namespace

namespace hexagon {

bool mul_mat_f32(hexagon::tensor * out, compute_params * params) {
    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "mul_mat_f32 requires max dims 4");
    static_assert(std::is_same<hexagon::dequant_output_type, float>::value ||
                      std::is_same<hexagon::dequant_output_type, npu_device_fp16_t>::value,
                  "dequant_output_type must be float or npu_device_fp16_t");

    if (!out) {
        return false;
    }

    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    const bool is_src0_quantized = is_quantized_type(src0->get_type());
    const bool is_gemv           = src1->get_ne(1) == 1 && src1->get_ne(2) == 1 && src1->get_ne(3) == 1;
    const auto base_index        = is_gemv ? kMulMatGemvBaseIndex : 0;
    switch (src1->get_type()) {
        case NPU_DATA_TYPE_F32:
            if (is_src0_quantized) {
                if (is_gemv && src0->get_type() == NPU_DATA_TYPE_Q4_0) {
                    // TODO: move to array
                    mul_mat_gemv_quant_impl<hexagon::vec_dot_product_vqf32_q40_f32>(src0, src1, out, params);
                } else {
                    kMulMatF16F32QuantizedFuncs[is_mul_mat_f16_f32_src_tensors_aligned(src0, src1, true, is_gemv) +
                                                base_index](src0, src1, out, params);
                }
            } else if (src0->get_type() == NPU_DATA_TYPE_F16) {
                kMulMatF16F32Funcs[is_mul_mat_f16_f32_src_tensors_aligned(src0, src1, true, is_gemv) + base_index](
                    src0, src1, out, params);
            } else {
                kMulMatF32F32Funcs[is_mul_mat_f32_f32_src_tensors_aligned(src0, src1) + base_index](src0, src1, out,
                                                                                                    params);
            }
            return true;
        case NPU_DATA_TYPE_F16:
            if (is_src0_quantized) {
                kMulMatF16QuantizedFuncs[is_mul_mat_f16_f16_src_tensors_aligned(src0, src1, is_src0_quantized) +
                                         base_index](src0, src1, out, params);
            } else {
                kMulMatF16Funcs[is_mul_mat_f16_f16_src_tensors_aligned(src0, src1, is_src0_quantized) + base_index](
                    src0, src1, out, params);
            }
            return true;
        default:
            break;
    }

    DEVICE_LOG_ERROR("[MUL_MAT]Unsupported src1 tensor type: %s\n", get_type_name(src1->get_type()));
    return false;
}

bool is_mul_mat_supported(const npu_device_tensor_op_spec * op_spec,
                          const npu_device_tensor_spec *    dst,
                          const npu_device_tensor_spec *    srcs,
                          size_t                            src_len) {
    const auto op = op_spec->op;
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
        if (src1.type == NPU_DATA_TYPE_F32 && src0.type == NPU_DATA_TYPE_F16) {
            // F16 * F32 is supported
            DEVICE_LOG_DEBUG("[%s]src0.type(%s) and src1.type(%s) mismatch, but src0 is F16 and src1 is F32\n",
                             op_get_name(op), get_type_name(src0.type), get_type_name(src1.type));
        } else {
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
    }

    if (is_transposed_or_permuted(src0.nb)) {
        // TODO: fix permuted src0
        DEVICE_LOG_DEBUG("[%s]src0 is transposed or permuted, disabled\n", op_get_name(op));
        return false;
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

    if (src1.ne[1] == 1 && src1.ne[2] == 1 && src1.ne[3] == 1 && dst->ne[0] < hexagon::kMaxThreadCount) {
        DEVICE_LOG_DEBUG("[%s]src1 is scalar and dst cannot be parallelized: %ld\n", op_get_name(op),
                         (long) dst->ne[0]);
        return false;
    }

    return true;
}

bool is_mul_mat_required_sync(npu_device_tensor_op       prev_op,
                              const npu_device_ne_type & prev_ne,
                              npu_device_tensor_op       op,
                              const npu_device_ne_type & ne) {
    NPU_UNUSED(prev_op);
    NPU_UNUSED(prev_ne);
    NPU_UNUSED(op);
    NPU_UNUSED(ne);
    return prev_op != NPU_OP_MUL_MAT || !is_same_shape(prev_ne, ne);
}

}  // namespace hexagon
