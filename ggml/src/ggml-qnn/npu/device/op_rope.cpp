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
void rope_yarn(float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
               float * cos_theta, float * sin_theta) {
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

void rope_cache_init(float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0,
                     float ext_factor, float mscale, float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

void mrope_cache_init(float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e,
                      const int sections[4], bool indep_sects, float freq_scale, const float * freq_factors,
                      float corr_dims[2], int64_t ne0, float ext_factor, float mscale, float * cache, float sin_sign,
                      float theta_scale) {
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
                ;
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

}  // namespace

namespace hexagon {

bool rope_f32(tensor * out, compute_params * params) {
    const tensor * src0 = out->get_src(0);
    const tensor * src1 = out->get_src(1);
    const tensor * src2 = out->get_src(2);

    const int n_dims      = out->get_op_param<int32_t>(1);
    const int mode        = out->get_op_param<int32_t>(2);
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

    const bool is_neox   = mode & NPU_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & NPU_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
    const bool is_vision = mode == NPU_ROPE_TYPE_VISION;

    if (is_mrope && sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
        DEVICE_LOG_ERROR("ROPE: invalid sections for MROPE: %d, %d, %d\n", sections[0], sections[1], sections[2]);
        return false;  // invalid sections for MROPE
    }

    if (is_vision && n_dims != out->get_ne(0) / 2) {
        DEVICE_LOG_ERROR("ROPE: invalid n_dims for vision ROPE: %d, expected: %d\n", n_dims, out->get_ne(0) / 2);
        return false;  // invalid n_dims for vision ROPE
    }

    // cache size is (ne0 + CACHE_LINE_SIZE_F32)
    const size_t total_cache_size = hexagon::get_aligned_size(out->get_ne(0) * sizeof(float));
    auto *       cache_ptr        = params->get_vtcm_cache(total_cache_size);
    if (!cache_ptr) {
        DEVICE_LOG_ERROR("Failed to allocate VTCM cache for flash_attn: %zu bytes\n", total_cache_size);
        return false;  // failed to allocate cache
    }

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        if (src2->get_type() != NPU_DATA_TYPE_F32 || src2->get_ne(0) < n_dims / 2) {
            DEVICE_LOG_ERROR("ROPE: src2 type is not F32 or F16: %s\n", get_type_name(src2->get_type()));
            return false;  // unsupported src2 type
        }

        freq_factors = src2->get_read_buffer_as<float>();
    }

    const auto total_rows    = out->get_ne(3) * out->get_ne(2) * out->get_ne(1);
    const auto start_end_row = params->get_work_slice(total_rows);

    // row index used to determine which thread to use
    int ir = 0;

    const float     sin_sign      = 1.0f;
    const int32_t * pos           = src1->get_read_buffer_as<int32_t>();
    const uint8_t * src0_data_ptr = src0->get_read_buffer();
    uint8_t *       dst_data_ptr  = out->get_write_buffer();
    for (int64_t i3 = 0; i3 < out->get_ne(3); i3++) {      // batch
        for (int64_t i2 = 0; i2 < out->get_ne(2); i2++) {  // seq-len
            float * cache = reinterpret_cast<float *>(cache_ptr);
            if (!is_mrope) {
                const int64_t p = pos[i2];
                rope_cache_init(p, freq_scale, freq_factors, corr_dims, out->get_ne(0), ext_factor, attn_factor, cache,
                                sin_sign, theta_scale);
            } else {
                const int64_t p_t = pos[i2];
                const int64_t p_h = pos[i2 + out->get_ne(2)];
                const int64_t p_w = pos[i2 + out->get_ne(2) * 2];
                const int64_t p_e = pos[i2 + out->get_ne(2) * 3];
                mrope_cache_init(p_t, p_h, p_w, p_e, sections, is_vision, freq_scale, freq_factors, corr_dims,
                                 out->get_ne(0), ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }

            for (int64_t i1 = 0; i1 < out->get_ne(1); i1++) {  // attn-heads
                if (ir++ < start_end_row.first) {              // TODO: optimize this
                    continue;
                }
                if (ir > start_end_row.second) {  // TODO: optimize this
                    break;
                }

                if (is_neox || is_mrope) {
                    if (is_vision) {
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const int64_t ic = i0 / 2;

                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];

                            const float * const src =
                                (float *) (src0_data_ptr + i3 * src0->get_nb(3) + i2 * src0->get_nb(2) +
                                           i1 * src0->get_nb(1) + ic * src0->get_nb(0));
                            float * dst_data = (float *) (dst_data_ptr + i3 * out->get_nb(3) + i2 * out->get_nb(2) +
                                                          i1 * out->get_nb(1) + ic * out->get_nb(0));

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

                            const float * const src =
                                (float *) (src0_data_ptr + i3 * src0->get_nb(3) + i2 * src0->get_nb(2) +
                                           i1 * src0->get_nb(1) + ic * src0->get_nb(0));
                            float * dst_data = (float *) (dst_data_ptr + i3 * out->get_nb(3) + i2 * out->get_nb(2) +
                                                          i1 * out->get_nb(1) + ic * out->get_nb(0));

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

                        const float * const src =
                            (float *) (src0_data_ptr + i3 * src0->get_nb(3) + i2 * src0->get_nb(2) +
                                       i1 * src0->get_nb(1) + i0 * src0->get_nb(0));
                        float * dst_data = (float *) (dst_data_ptr + i3 * out->get_nb(3) + i2 * out->get_nb(2) +
                                                      i1 * out->get_nb(1) + i0 * out->get_nb(0));

                        const float x0 = src[0];
                        const float x1 = src[1];

                        dst_data[0] = x0 * cos_theta - x1 * sin_theta;
                        dst_data[1] = x0 * sin_theta + x1 * cos_theta;
                    }
                }

                if (is_vision) {
                    for (int64_t i0 = n_dims; i0 < out->get_ne(0); i0 += 2) {
                        const int64_t ic = i0 / 2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src =
                            (float *) (src0_data_ptr + i3 * src0->get_nb(3) + i2 * src0->get_nb(2) +
                                       i1 * src0->get_nb(1) + ic * src0->get_nb(0));
                        float * dst_data = (float *) (dst_data_ptr + i3 * out->get_nb(3) + i2 * out->get_nb(2) +
                                                      i1 * out->get_nb(1) + ic * out->get_nb(0));

                        const float x0 = src[0];
                        const float x1 = src[n_dims];

                        dst_data[0]      = x0 * cos_theta - x1 * sin_theta;
                        dst_data[n_dims] = x0 * sin_theta + x1 * cos_theta;
                    }
                } else {
                    // fill the remain channels with data from src tensor
                    for (int64_t i0 = n_dims; i0 < out->get_ne(0); i0 += 2) {
                        const float * const src =
                            (float *) (src0_data_ptr + i3 * src0->get_nb(3) + i2 * src0->get_nb(2) +
                                       i1 * src0->get_nb(1) + i0 * src0->get_nb(0));
                        float * dst_data = (float *) (dst_data_ptr + i3 * out->get_nb(3) + i2 * out->get_nb(2) +
                                                      i1 * out->get_nb(1) + i0 * out->get_nb(0));

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            }
        }
    }

    out->release_write_buffer();
    return true;
}

bool is_rope_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst, const npu_device_tensor_spec * srcs,
                       size_t src_len) {
    if (src_len < 3 || !dst || !srcs) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", op_get_name(op));
        return false;
    }

    return false;  // ROPE operation is not supported yet
}

}  // namespace hexagon
