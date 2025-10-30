
#include "op_flash_attn.hpp"

#include "type_traits.hpp"
#include "util.hpp"
#include "vec_ops.hpp"

namespace {

// TODO: use a more efficient conversion
inline float f16_to_f32(const npu_device_fp16_t src) {
    return reinterpret_cast<const __fp16 &>(src);
}

// From: ggml/src/ggml-cpu/ops.cpp
template <bool _IsKvF16, bool _HasMask>
void flash_attn_impl(hexagon::tensor *         out,
                     const hexagon::tensor *   q,
                     const hexagon::tensor *   k,
                     const hexagon::tensor *   v,
                     const hexagon::tensor *   mask,
                     const hexagon::tensor *   sinks,
                     hexagon::compute_params * params) {
    static_assert(3 <= hexagon::kMaxParamsCount, "flash_attn op params count exceeds max params count");

    constexpr const npu_device_tensor_data_type kKvDataType = _IsKvF16 ? NPU_DATA_TYPE_F16 : NPU_DATA_TYPE_F32;
    constexpr const bool                        kHasMask    = _HasMask;

    if (k->get_type() != kKvDataType || v->get_type() != k->get_type()) {
        DEVICE_LOG_ERROR("flash_attn_impl: k and v must have same type, got k: %s, v: %s\n",
                         hexagon::get_type_name(k->get_type()), hexagon::get_type_name(v->get_type()));
        return;
    }

    if (kHasMask != (mask != nullptr)) {
        DEVICE_LOG_ERROR("flash_attn_impl: mask is required when kHasMask is true\n");
        return;
    }

    float       scale         = out->get_op_param<float>(0);
    const float max_bias      = out->get_op_param<float>(1);
    const float logit_softcap = out->get_op_param<float>(2);

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // broadcast factors
    const int64_t rk2 = q->get_ne(2) / k->get_ne(2);
    const int64_t rk3 = q->get_ne(3) / k->get_ne(3);
    const int64_t rv2 = q->get_ne(2) / v->get_ne(2);
    const int64_t rv3 = q->get_ne(3) / v->get_ne(3);

    const uint32_t n_head      = q->get_ne(2);
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const auto &         k_type_traits = hexagon::get_type_traits(kKvDataType);
    const auto           q_to_kv_type  = k_type_traits.from_float;
    constexpr const auto kq_vec_dot    = _IsKvF16 ? hexagon::type_erase_dot_func<hexagon::vec_dot_product_f16_f16> :
                                                    hexagon::type_erase_dot_func<hexagon::vec_dot_product_f32_f32>;
    if (!q_to_kv_type) {
        DEVICE_LOG_ERROR("flash_attn_impl: unsupported data type for q, k, or v\n");
        return;
    }

    const int64_t total_rows    = q->get_ne(1) * q->get_ne(2) * q->get_ne(3);  // total number of rows in Q
    const auto    start_end_row = params->get_work_slice(total_rows);          // work slice for this thread

    const auto DK          = k->get_ne(0);
    const auto DV          = v->get_ne(0);
    const auto row_bytes_q = q->get_ne(0) * hexagon::get_type_traits(q->get_type()).type_size;
    const auto row_bytes_k = DK * k_type_traits.type_size;
    const auto row_bytes_v = DV * hexagon::get_type_traits(v->get_type()).type_size;

    constexpr const size_t kFloatsPerVectorPair = hexagon::kBytesPerVector * 2 / sizeof(float);
    const auto             aligned_dk = (DK + kFloatsPerVectorPair - 1) / kFloatsPerVectorPair * kFloatsPerVectorPair;
    const auto             aligned_dv = (DV + kFloatsPerVectorPair - 1) / kFloatsPerVectorPair * kFloatsPerVectorPair;
    size_t                 total_cache_size = sizeof(float) * (aligned_dk + 2 * aligned_dv);
    auto *                 cache_ptr        = params->get_vtcm_cache(total_cache_size);
    if (!cache_ptr) {
        DEVICE_LOG_ERROR("Failed to allocate VTCM cache for flash_attn: %zu bytes\n", total_cache_size);
        return;
    }

    // loop over n_batch and n_head
    constexpr bool is_v_f16           = _IsKvF16;  // check if V is in FP16 format, otherwise it is in FP32 format
    const auto     rows_per_batch     = q->get_ne(2) * q->get_ne(1);
    const auto     out_rows_per_batch = out->get_ne(2) * out->get_ne(1);
    uint8_t *      dst_ptr            = out->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("flash_attn_impl: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) out,
                         hexagon::get_type_name(out->get_type()));
        return;
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(out, params->get_thread_index(), flash_attn);
    const uint8_t * q_ptr     = q->get_read_buffer();
    const uint8_t * k_ptr     = k->get_read_buffer();
    const uint8_t * v_ptr     = v->get_read_buffer();
    const uint8_t * mask_ptr  = kHasMask ? mask->get_read_buffer() : nullptr;
    const uint8_t * sinks_ptr = sinks ? sinks->get_read_buffer() : nullptr;
    float *         VKQ32     = reinterpret_cast<float *>(cache_ptr);          // FP32 VKQ accumulator
    auto * VKQ16 = reinterpret_cast<npu_device_fp16_t *>(VKQ32 + aligned_dv);  // (temporary) FP16 VKQ accumulator
    auto * Q_q   = reinterpret_cast<npu_device_fp16_t *>(
        VKQ32 + 2 * aligned_dv);  // (temporary) buffer for Q converted to quantized/FP16
    for (auto ir = start_end_row.first; ir < start_end_row.second; ++ir) {
        // q indices
        const auto iq3 = ir / rows_per_batch;
        const auto iq2 = (ir - iq3 * rows_per_batch) / q->get_ne(1);
        const auto iq1 = (ir - iq3 * rows_per_batch - iq2 * q->get_ne(1));

        const auto * q_data = q_ptr + (iq1 * q->get_nb(1) + iq2 * q->get_nb(2) + iq3 * q->get_nb(3));
        hexagon::l2fetch_row(q_data, row_bytes_q);

        const uint32_t h = iq2;  // head index
        const float    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;       // sum
        float M = -INFINITY;  // maximum KQ value

        if constexpr (is_v_f16) {
            memset(VKQ16, 0, DV * sizeof(npu_device_fp16_t));
        } else {
            memset(VKQ32, 0, DV * sizeof(float));
        }

        const npu_device_fp16_t * mp =
            kHasMask ? reinterpret_cast<const npu_device_fp16_t *>(mask_ptr + iq1 * mask->get_nb(1) +
                                                                   (iq2 % mask->get_ne(2)) * mask->get_nb(2) +
                                                                   (iq3 % mask->get_ne(3)) * mask->get_nb(3)) :
                       nullptr;

        q_to_kv_type(reinterpret_cast<const float *>(q_data), Q_q, DK);

        if (kHasMask) {
            hexagon::l2fetch_row(reinterpret_cast<const uint8_t *>(mp), mask->get_nb(1));
        }

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf
        const auto * k_plane_ptr = k_ptr + ik2 * k->get_nb(2) + ik3 * k->get_nb(3);
        const auto * v_plane_ptr = v_ptr + iv2 * v->get_nb(2) + iv3 * v->get_nb(3);
        for (int64_t ic = 0; ic < k->get_ne(1); ++ic) {
            DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(flash_attn, 0, loop);
            float mv = kHasMask ? (slope * f16_to_f32(mp[ic])) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s = 0.f;
            {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(flash_attn, 1, kq_dot);
                const auto * k_data = k_plane_ptr + ic * k->get_nb(1);
                if (ic < k->get_ne(1) - 1) {
                    hexagon::l2fetch_row(k_data + k->get_nb(1), row_bytes_k);
                }

                s = kq_vec_dot(k_data, Q_q, DK);   // KQ value
                s = s * scale;                     // scale KQ value
                if (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);  // TODO: vectorize this?
                }

                s += mv;  // apply mask
            }

            const float Mold = M;

            float ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f;  // post-softmax KQ value, expf(s - M)

            const auto * v_data = v_plane_ptr + ic * v->get_nb(1);
            if (ic < v->get_ne(1)) {
                hexagon::l2fetch_row(v_data, row_bytes_v);
            }

            if constexpr (is_v_f16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M  = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    hexagon::vec_scale_f16(VKQ16, ms, VKQ16, DV);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(flash_attn, 2, mad);
                hexagon::vec_mad_f16(reinterpret_cast<const npu_device_fp16_t *>(v_data), vs, VKQ16, DV);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M  = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    hexagon::vec_scale_f32(VKQ32, ms, VKQ32, DV);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(flash_attn, 2, mad);
                {
                    // V is F32
                    hexagon::vec_mad_f32(reinterpret_cast<const float *>(v_data), vs, VKQ32, DV);
                }
            }

            S = S * ms + vs;  // scale and increment sum with partial sum
        }

        if constexpr (is_v_f16) {
            // TODO: use a more efficient conversion
            for (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = f16_to_f32(VKQ16[d]);
            }
        }

        if (sinks_ptr) {
            const float s = reinterpret_cast<const float *>(sinks_ptr)[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                hexagon::vec_scale_f32(VKQ32, ms, VKQ32, DV);
            } else {
                vs = expf(s - M);
            }

            S = S * ms + vs;
        }

        // V /= S
        const float S_inv = 1.0f / S;
        hexagon::vec_scale_f32(VKQ32, S_inv, VKQ32, DV);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // permute(0, 2, 1, 3)
        hexagon::vec_cpy_f32(
            reinterpret_cast<const float *>(VKQ32),
            reinterpret_cast<float *>(dst_ptr + (i3 * out_rows_per_batch + i2 + i1 * out->get_ne(1)) * out->get_nb(1)),
            out->get_ne(0));
    }

    out->release_write_buffer();  // mark the output tensor as modified
}

}  // namespace

namespace hexagon {

bool flash_attn_f32(tensor * out, compute_params * params) {
    if (!out || !params) {
        DEVICE_LOG_DEBUG("invalid out or params\n");
        return false;
    }

    const auto * q = out->get_src(0);
    const auto * k = out->get_src(1);
    const auto * v = out->get_src(2);
    if (!q || !k || !v) {
        DEVICE_LOG_DEBUG("invalid src tensors: q: %p, k: %p, v: %p\n", (void *) q, (void *) k, (void *) v);
        return false;
    }

    const auto * mask  = out->get_src(3);
    const auto * sinks = out->get_src(4);
    if (k->get_type() == NPU_DATA_TYPE_F16) {
        if (mask) {
            flash_attn_impl<true, true>(out, q, k, v, mask, sinks, params);
        } else {
            flash_attn_impl<true, false>(out, q, k, v, mask, sinks, params);
        }
    } else {
        if (mask) {
            flash_attn_impl<false, true>(out, q, k, v, mask, sinks, params);
        } else {
            flash_attn_impl<false, false>(out, q, k, v, mask, sinks, params);
        }
    }
    return true;
}

bool is_flash_attn_supported(const npu_device_tensor_op_spec * op_spec,
                             const npu_device_tensor_spec *    dst,
                             const npu_device_tensor_spec *    srcs,
                             size_t                            src_len) {
    const auto op = op_spec->op;
    if (op != NPU_OP_FLASH_ATTN) {
        DEVICE_LOG_DEBUG("op is not NPU_OP_FLASH_ATTN: %d\n", op);
        return false;
    }

    if (!dst || !srcs || src_len < 4) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", op_get_name(op));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]dst type is not F32: %s\n", op_get_name(op), get_type_name(dst->type));
        return false;
    }

    const auto * q = &srcs[0];
    if (q->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]q type is not F32: %s\n", op_get_name(op), get_type_name(q->type));
        return false;
    }

    const auto * k = &srcs[1];
    if (k->type != NPU_DATA_TYPE_F16) {  // TODO: support more k types
        DEVICE_LOG_DEBUG("[%s]k type is not F16: %s\n", op_get_name(op), get_type_name(k->type));
        return false;
    }

    const auto * v = &srcs[2];
    if (v->type != k->type) {  // TODO: support more v types
        DEVICE_LOG_DEBUG("[%s]v type is not the same as k: %s vs %s\n", op_get_name(op), get_type_name(v->type),
                         get_type_name(k->type));
        return false;
    }

    const auto * mask = &srcs[3];
    if (mask->type != NPU_DATA_TYPE_F16) {
        DEVICE_LOG_DEBUG("[%s]mask type is not F16: %s\n", op_get_name(op), get_type_name(mask->type));
        return false;
    }

    if (dst->ne[0] != v->ne[0] || dst->ne[2] != q->ne[1]) {
        DEVICE_LOG_DEBUG(
            "[%s]dst shape does not match q and v: dst ne: %lld, %lld, %lld, %lld, q ne: %lld, %lld, %lld, %lld, "
            "v ne: %lld, %lld, %lld, %lld\n",
            op_get_name(op), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], q->ne[0], q->ne[1], q->ne[2], q->ne[3],
            v->ne[0], v->ne[1], v->ne[2], v->ne[3]);
        return false;
    }

    if (is_transposed_or_permuted(dst->nb)) {
        DEVICE_LOG_DEBUG("[%s]dst cannot be transposed or permuted, nb: %zu, %zu, %zu, %zu\n", op_get_name(op),
                         (size_t) dst->nb[0], (size_t) dst->nb[1], (size_t) dst->nb[2], (size_t) dst->nb[3]);
        return false;
    }

    if (q->ne[0] != k->ne[0]) {
        DEVICE_LOG_DEBUG(
            "[%s]q and k shapes do not match: q ne: %lld, %lld, %lld, %lld, k ne: %lld, %lld, %lld, %lld\n",
            op_get_name(op), q->ne[0], q->ne[1], q->ne[2], q->ne[3], k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
        return false;
    }

    return true;
}

bool is_flash_attn_required_sync(npu_device_tensor_op       prev_op,
                                 const npu_device_ne_type & prev_ne,
                                 npu_device_tensor_op       op,
                                 const npu_device_ne_type & ne) {
    NPU_UNUSED(prev_ne);
    NPU_UNUSED(op);
    NPU_UNUSED(ne);
    return prev_op != NPU_OP_COUNT;
}

}  // namespace hexagon
