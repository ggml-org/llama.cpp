#include "op_rope.hpp"

#include "type_traits.hpp"

#ifndef M_PI
#    define M_PI (3.14159265358979323846)
#endif

namespace {

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
}

void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    // start and end correction dims
    float start = floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   = ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0]     = std::max<float>(0, start);
    dims[1]     = std::min<float>(n_dims - 1, end);
}

float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / std::max<float>(0.001f, high - low);
    return 1 - std::min<float>(1, std::max<float>(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
void rope_yarn(float   theta_extrap,
               float   freq_scale,
               float   corr_dims[2],
               int64_t i0,
               float   ext_factor,
               float   mscale,
               float * cos_theta,
               float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta        = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta          = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

void rope_cache_init(float         theta_base,
                     float         freq_scale,
                     const float * freq_factors,
                     float         corr_dims[2],
                     int64_t       ne0,
                     float         ext_factor,
                     float         mscale,
                     float *       cache,
                     float         sin_sign,
                     float         theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

void mrope_cache_init(float         theta_base_t,
                      float         theta_base_h,
                      float         theta_base_w,
                      float         theta_base_e,
                      const int     sections[4],
                      bool          indep_sects,
                      float         freq_scale,
                      const float * freq_factors,
                      float         corr_dims[2],
                      int64_t       ne0,
                      float         ext_factor,
                      float         mscale,
                      float *       cache,
                      float         sin_sign,
                      float         theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta_t   = theta_base_t;
    float theta_h   = theta_base_h;
    float theta_w   = theta_base_w;
    float theta_e   = theta_base_e;  // extra position id for vision encoder
    int   sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int   sec_w     = sections[1] + sections[0];
    int   sec_e     = sections[2] + sec_w;

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;

        int sector = (i0 / 2) % sect_dims;
        if (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            if (sector == 0) {
                theta_t = theta_base_t;
            } else if (sector == sections[0]) {
                theta_h = theta_base_h;
            } else if (sector == sec_w) {
                theta_w = theta_base_w;
            } else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta = theta_t;
        if (sector >= sections[0] && sector < sec_w) {
            theta = theta_h;
        } else if (sector >= sec_w && sector < sec_w + sections[2]) {
            theta = theta_w;
        } else if (sector >= sec_w + sections[2]) {
            theta = theta_e;
        }

        rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

template <bool _IsNeoX, bool _IsMrope, bool _IsVision>
bool rope_impl(hexagon::tensor * out, hexagon::compute_params * params) {
    const auto * src0 = out->get_src(0);
    const auto * src1 = out->get_src(1);
    const auto * src2 = out->get_src(2);

    const int n_dims      = out->get_op_param<int32_t>(1);
    const int n_ctx_orig  = out->get_op_param<int32_t>(4);
    const int sections[4] = {
        out->get_op_param<int32_t>(11),
        out->get_op_param<int32_t>(12),
        out->get_op_param<int32_t>(13),
        out->get_op_param<int32_t>(14),
    };

    const float freq_base   = out->get_op_param<float>(5);
    const float freq_scale  = out->get_op_param<float>(6);
    const float ext_factor  = out->get_op_param<float>(7);
    const float attn_factor = out->get_op_param<float>(8);
    const float beta_fast   = out->get_op_param<float>(9);
    const float beta_slow   = out->get_op_param<float>(10);
    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    if (_IsMrope && sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
        DEVICE_LOG_ERROR("[ROPE]invalid sections for MROPE: %d, %d, %d\n", sections[0], sections[1], sections[2]);
        return false;  // invalid sections for MROPE
    }

    if (n_dims % 2 || (_IsVision && n_dims != out->get_ne(0) / 2)) {
        DEVICE_LOG_ERROR("[ROPE]invalid n_dims for vision ROPE: %d, expected: %lld\n", n_dims, out->get_ne(0) / 2);
        return false;  // invalid n_dims for vision ROPE
    }

    // cache size is (ne0 + CACHE_LINE_SIZE_F32)
    const size_t total_cache_size = hexagon::get_aligned_size(out->get_ne(0) * sizeof(float));
    auto *       cache_ptr        = params->get_vtcm_cache(total_cache_size);
    if (!cache_ptr) {
        DEVICE_LOG_ERROR("[ROPE]Failed to allocate VTCM cache for flash_attn: %zu bytes\n", total_cache_size);
        return false;  // failed to allocate cache
    }

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        if (src2->get_type() != NPU_DATA_TYPE_F32 || src2->get_ne(0) < n_dims / 2) {
            DEVICE_LOG_ERROR("[ROPE]src2 type is not F32 or F16: %s\n", hexagon::get_type_name(src2->get_type()));
            return false;  // unsupported src2 type
        }

        freq_factors = src2->get_read_buffer_as<float>();
    }

    const int64_t total_planes = out->get_ne(3) * out->get_ne(2);
    const auto    start_end_plane =
        params->get_work_slice(total_planes);  // TODO: figure out how to use row slice for inplace rope
    if (start_end_plane.first >= start_end_plane.second) {
        return true;
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(out, params->get_thread_index(), rope);

    const float     sin_sign      = 1.0f;
    const int32_t * pos           = src1->get_read_buffer_as<int32_t>();
    const uint8_t * src0_data_ptr = src0->get_read_buffer();
    uint8_t *       dst_data_ptr  = out->get_write_buffer();
    for (int64_t ip = start_end_plane.first; ip < start_end_plane.second; ip++) {
        int64_t i3    = ip / out->get_ne(2);  // batch
        int64_t i2    = ip % out->get_ne(2);  // seq-len
        float * cache = reinterpret_cast<float *>(cache_ptr);
        if constexpr (!_IsMrope) {
            DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(rope, 0, cache);
            const int64_t p = pos[i2];
            rope_cache_init(p, freq_scale, freq_factors, corr_dims, out->get_ne(0), ext_factor, attn_factor, cache,
                            sin_sign, theta_scale);
        } else {
            DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(rope, 0, cache);
            const int64_t p_t = pos[i2];
            const int64_t p_h = pos[i2 + out->get_ne(2)];
            const int64_t p_w = pos[i2 + out->get_ne(2) * 2];
            const int64_t p_e = pos[i2 + out->get_ne(2) * 3];
            mrope_cache_init(p_t, p_h, p_w, p_e, sections, _IsVision, freq_scale, freq_factors, corr_dims,
                             out->get_ne(0), ext_factor, attn_factor, cache, sin_sign, theta_scale);
        }

        DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(rope, 1, loop);
        const uint8_t * src0_plane = src0_data_ptr + i3 * src0->get_nb(3) + i2 * src0->get_nb(2);
        uint8_t *       dst_plane  = dst_data_ptr + i3 * out->get_nb(3) + i2 * out->get_nb(2);
        for (int64_t i1 = 0; i1 < out->get_ne(1); i1++) {  // attn-heads
            const uint8_t * src0_row = src0_plane + i1 * src0->get_nb(1);
            uint8_t *       dst_row  = dst_plane + i1 * out->get_nb(1);
            if constexpr (_IsNeoX || _IsMrope) {
                if constexpr (_IsVision) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const int64_t ic = i0 / 2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src      = (float *) (src0_row + ic * src0->get_nb(0));
                        float *             dst_data = (float *) (dst_row + ic * out->get_nb(0));

                        const float x0 = src[0];
                        const float x1 = src[n_dims];

                        dst_data[0]      = x0 * cos_theta - x1 * sin_theta;
                        dst_data[n_dims] = x0 * sin_theta + x1 * cos_theta;
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const int64_t ic = i0 / 2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src      = (float *) (src0_row + ic * src0->get_nb(0));
                        float *             dst_data = (float *) (dst_row + ic * out->get_nb(0));

                        const float x0 = src[0];
                        const float x1 = src[n_dims / 2];

                        dst_data[0]          = x0 * cos_theta - x1 * sin_theta;
                        dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
                    }
                }
            } else {
                for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                    const float cos_theta = cache[i0 + 0];
                    const float sin_theta = cache[i0 + 1];

                    const float * const src      = (float *) (src0_row + i0 * src0->get_nb(0));
                    float *             dst_data = (float *) (dst_row + i0 * out->get_nb(0));

                    const float x0 = src[0];
                    const float x1 = src[1];

                    dst_data[0] = x0 * cos_theta - x1 * sin_theta;
                    dst_data[1] = x0 * sin_theta + x1 * cos_theta;
                }
            }

            if constexpr (_IsVision) {
                for (int64_t i0 = n_dims; i0 < out->get_ne(0); i0 += 2) {
                    const int64_t ic = i0 / 2;

                    const float cos_theta = cache[i0 + 0];
                    const float sin_theta = cache[i0 + 1];

                    const float * const src      = (float *) (src0_row + ic * src0->get_nb(0));
                    float *             dst_data = (float *) (dst_row + ic * out->get_nb(0));

                    const float x0 = src[0];
                    const float x1 = src[n_dims];

                    dst_data[0]      = x0 * cos_theta - x1 * sin_theta;
                    dst_data[n_dims] = x0 * sin_theta + x1 * cos_theta;
                }
            } else {
                // fill the remain channels with data from src tensor
                hexagon::vec_cpy_f32(reinterpret_cast<const float *>(src0_row + n_dims * src0->get_nb(0)),
                                     reinterpret_cast<float *>(dst_row + n_dims * out->get_nb(0)),
                                     out->get_ne(0) - n_dims);
            }
        }
    }

    out->release_write_buffer();
    return true;
}

typedef bool (*rope_impl_func)(hexagon::tensor * out, hexagon::compute_params * params);

constexpr const rope_impl_func kRopeImplFuncs[8] = {
    rope_impl<false, false, false>,  // IsNotNeoX, IsNotMrope, IsNotVision
    rope_impl<false, false, true>,   // IsNotNeoX, IsNotMrope, IsVision
    rope_impl<false, true, false>,   // IsNotNeoX, IsMrope, IsNotVision
    rope_impl<false, true, true>,    // IsNotNeoX, IsMrope, IsVision
    rope_impl<true, false, false>,   // IsNeoX, IsNotMrope, IsNotVision
    rope_impl<true, false, true>,    // IsNeoX, IsNotMrope, IsVision
    rope_impl<true, true, false>,    // IsNeoX, IsMrope, IsNotVision
    rope_impl<true, true, true>,     // IsNeoX, IsMrope, IsVision
};

}  // namespace

namespace hexagon {

bool rope_f32(tensor * out, compute_params * params) {
    const int  mode      = out->get_op_param<int32_t>(2);
    const bool is_neox   = mode & NPU_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & NPU_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
    const bool is_vision = mode == NPU_ROPE_TYPE_VISION;

    size_t impl_index = is_neox ? 4 : 0;
    impl_index += is_mrope ? 2 : 0;
    impl_index += is_vision ? 1 : 0;

    if (impl_index >= sizeof(kRopeImplFuncs) / sizeof(kRopeImplFuncs[0])) {
        DEVICE_LOG_ERROR("[ROPE]invalid impl_index: %zu\n", impl_index);
        return false;  // invalid impl index
    }

    return kRopeImplFuncs[impl_index](out, params);
}

bool is_rope_supported(const npu_device_tensor_op_spec * op_spec,
                       const npu_device_tensor_spec *    dst,
                       const npu_device_tensor_spec *    srcs,
                       size_t                            src_len) {
    const auto op = op_spec->op;
    if (op != NPU_OP_ROPE) {
        DEVICE_LOG_DEBUG("[%s]op is not ROPE\n", op_get_name(op));
        return false;
    }

    if (src_len < 2 || !dst || !srcs) {
        // freq can be optional, but we require at least 2 srcs: src0 and src1
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", op_get_name(op));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]dst type is not F32: %s\n", op_get_name(op), get_type_name(dst->type));
        return false;  // add more dst type if needed
    }

    const auto & src0 = srcs[0];
    if (src0.type != dst->type) {
        DEVICE_LOG_DEBUG("[%s]src0 type is not the same as dst type: %s vs %s\n", op_get_name(op),
                         get_type_name(src0.type), get_type_name(dst->type));
        return false;  // unsupported src0 type
    }

    const auto & src1 = srcs[1];
    if (src1.type != NPU_DATA_TYPE_I32) {
        DEVICE_LOG_DEBUG("[%s]src1 type is not I32: %s\n", op_get_name(op), get_type_name(src1.type));
        return false;  // unsupported src1 type
    }

    if (src_len > 2) {
        const auto & src2 = srcs[2];
        if (src2.type != NPU_DATA_TYPE_F32) {
            DEVICE_LOG_DEBUG("[%s]src2 type is not F32: %s\n", op_get_name(op), get_type_name(src2.type));
            return false;  // unsupported src2 type
        }

        DEVICE_LOG_DEBUG("[%s]freq is present\n", op_get_name(op));
    }

    if (!is_same_shape(src0, *dst)) {
        DEVICE_LOG_DEBUG("[%s]src0 and dst have different shape\n", op_get_name(op));
        return false;
    }

    // TODO: check the params for ROPE operation
    return true;  // ROPE operation is not supported yet
}

bool is_rope_required_sync(npu_device_tensor_op       prev_op,
                           const npu_device_ne_type & prev_ne,
                           npu_device_tensor_op       op,
                           const npu_device_ne_type & ne) {
    NPU_UNUSED(prev_op);
    NPU_UNUSED(prev_ne);
    NPU_UNUSED(op);
    NPU_UNUSED(ne);
    return false;
}

}  // namespace hexagon
