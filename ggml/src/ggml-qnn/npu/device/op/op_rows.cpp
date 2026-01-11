#include "op_rows.hpp"

#include "type_traits.hpp"

namespace {

template <typename idx_t> void set_rows_impl(hexagon::tensor * out, hexagon::compute_params * params) {
    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);

    const auto total_rows = src0->get_ne(3) * src0->get_ne(2) * src0->get_ne(1);
    const auto start_end  = params->get_work_slice(total_rows);
    if (start_end.first >= start_end.second) {
        return;
    }

    uint8_t * dst_ptr = out->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("set_rows_impl: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) out,
                         hexagon::get_type_name(out->get_type()));
        return;
    }

    const uint8_t * src0_ptr      = src0->get_read_buffer(true);  // TODO: avoid invalidation
    const uint8_t * src1_ptr      = src1->get_read_buffer(true);  // TODO: avoid invalidation
    const size_t    rows_per_cube = src0->get_ne(2) * src0->get_ne(1);

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(out, params->get_thread_index());
    auto from_float = hexagon::get_type_traits(out->get_type()).from_float;
    for (size_t ir = start_end.first; ir < size_t(start_end.second); ++ir) {
        const size_t i03 = ir / rows_per_cube;
        const size_t i02 = ir / src0->get_ne(1) - i03 * src0->get_ne(2);
        const size_t i01 = ir % src0->get_ne(1);
        const size_t i12 = i03 % src1->get_ne(2);
        const size_t i11 = i02 % src1->get_ne(1);
        const size_t i10 = i01;

        const size_t i1 = *reinterpret_cast<const idx_t *>(src1_ptr + i10 * src1->get_nb(0) + i11 * src1->get_nb(1) +
                                                           i12 * src1->get_nb(2));
        from_float(reinterpret_cast<const float *>(src0_ptr + i01 * src0->get_nb(1) + i02 * src0->get_nb(2) +
                                                   i03 * src0->get_nb(3)),
                   dst_ptr + i1 * out->get_nb(1) + i02 * out->get_nb(2) + i03 * out->get_nb(3),
                   size_t(src0->get_ne(0)));
    }

    out->release_write_buffer();  // mark the output tensor as modified
}

}  // namespace

namespace hexagon {

bool get_rows_f32(tensor * out, compute_params * params) {
    // TODO: implement get_rows
    return false;
}

bool set_rows_generic(tensor * out, compute_params * params) {
    if (!out) {
        return false;
    }

    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        DEVICE_LOG_ERROR("set_rows_generic: missing src0 or src1\n");
        return false;
    }

    switch (src1->get_type()) {
        case NPU_DATA_TYPE_I32:
            set_rows_impl<int32_t>(out, params);
            break;
        case NPU_DATA_TYPE_I64:
            set_rows_impl<int64_t>(out, params);
            break;
        default:
            DEVICE_LOG_ERROR("set_rows_generic: unsupported src1 type: %s\n", hexagon::get_type_name(src1->get_type()));
            return false;
    }
    return true;
}

bool is_rows_supported(const npu_device_tensor_op_spec * op_spec,
                       const npu_device_tensor_spec *    dst,
                       const npu_device_tensor_spec *    srcs,
                       size_t                            src_len) {
    const auto op = op_spec->op;
    if (op != NPU_OP_GET_ROWS && op != NPU_OP_SET_ROWS) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (src_len < 2) {
        DEVICE_LOG_DEBUG("[%s]invalid src_len: %zu\n", hexagon::op_get_name(op), src_len);
        return false;
    }

    const auto & src0 = srcs[0];
    const auto & src1 = srcs[1];
    if (op == NPU_OP_GET_ROWS) {
        if (dst->ne[0] != src0.ne[0]) {
            DEVICE_LOG_DEBUG("[%s]dst.ne[0] and src0.ne[0] not match: %ld vs %ld\n", hexagon::op_get_name(op),
                             (long) dst->ne[0], (long) src0.ne[0]);
            return false;
        }

        if (dst->type != src0.type) {
            DEVICE_LOG_DEBUG("[%s]dst.type and src0.type mismatch: %s vs %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(dst->type), hexagon::get_type_name(src0.type));
            return false;
        }

        // TODO: remove this limitation
        return false;
    } else {
        // NPU_OP_SET_ROWS
        if (dst->ne[0] != src0.ne[0] || dst->ne[2] != src0.ne[2] || dst->ne[3] != src0.ne[3]) {
            DEVICE_LOG_DEBUG("[%s]dst.ne[0], src0.ne[0] and src0.ne[2], src0.ne[3] not match: %ld vs %ld, %ld, %ld\n",
                             hexagon::op_get_name(op), (long) dst->ne[0], (long) src0.ne[0], (long) src0.ne[2],
                             (long) src0.ne[3]);
            return false;
        }

        if (src0.type != NPU_DATA_TYPE_F32) {
            DEVICE_LOG_DEBUG("[%s]src0.type is not F32: %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(src0.type));
            return false;
        }

        if (src1.type != NPU_DATA_TYPE_I32 && src1.type != NPU_DATA_TYPE_I64) {
            DEVICE_LOG_DEBUG("[%s]src1.type is not I32 or I64: %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(src1.type));
            return false;
        }

        if (dst->type != src0.type && !get_type_traits(dst->type).from_float) {
            DEVICE_LOG_DEBUG("[%s]dst.from_float is null: %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(dst->type));
            return false;
        }
    }

    return true;
}

bool is_rows_required_sync(npu_device_tensor_op       prev_op,
                           const npu_device_ne_type & prev_ne,
                           npu_device_tensor_op       op,
                           const npu_device_ne_type & ne) {
    // TODO: implement is_rows_required_sync
    return false;
}

}  // namespace hexagon
