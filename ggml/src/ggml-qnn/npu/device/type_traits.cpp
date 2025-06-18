#include "type_traits.hpp"

#include <hexagon_types.h>

#include <array>

#include "op_types.hpp"  // TODO: remove this include
#include "vec_ops.hpp"

static_assert(sizeof(npu_device_block_q4_k) ==
                  2 * sizeof(npu_device_fp16_t) + QUANT_K_SCALE_SIZE + QUANT_K_BLOCK_SIZE / 2,
              "wrong q4_K block size/padding");

static_assert(sizeof(npu_device_block_q4_0) == sizeof(npu_device_fp16_t) + QUANT_BLOCK_SIZE / 2,
              "wrong q4_0 block size/padding");

static_assert(sizeof(npu_device_block_q8_0) == sizeof(npu_device_fp16_t) + QUANT_BLOCK_SIZE,
              "wrong q8_0 block size/padding");

namespace {

inline float to_float(const npu_device_fp16_t src) {
    return reinterpret_cast<const __fp16 &>(src);
}

inline npu_device_fp16_t to_fp16(const float src) {
    __fp16 f16_value = static_cast<__fp16>(src);
    return reinterpret_cast<const npu_device_fp16_t &>(f16_value);
}

template <typename _TBlock> inline HVX_Vector load_block_generic(const _TBlock & src) {
    uint8_t buffer[hexagon::kBytesPerVector];

    static_assert(sizeof(buffer) == sizeof(HVX_Vector), "wrong cvt size/padding");
    static_assert(sizeof(buffer) >= sizeof(src.qs), "wrong q4_0 block size/padding");

    memcpy(&buffer[0], src.qs, sizeof(src.qs));
    return *reinterpret_cast<HVX_UVector *>(buffer);
}

template <typename _TBlock> inline HVX_Vector load_dual_block_generic(const _TBlock & src1, const _TBlock & src2) {
    uint8_t buffer[hexagon::kBytesPerVector];

    static_assert(sizeof(buffer) == sizeof(HVX_Vector), "wrong cvt size/padding");
    static_assert(sizeof(buffer) >= sizeof(src1.qs) * 2, "wrong q4_0 block size/padding");

    memcpy(&buffer[0], src1.qs, sizeof(src1.qs));
    memcpy(&buffer[sizeof(src1.qs)], src2.qs, sizeof(src2.qs));
    return *reinterpret_cast<HVX_UVector *>(buffer);
}

inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int   i   = reinterpret_cast<const int &>(val);
    return (i & 0x007fffff) - 0x00400000;
}

float make_qkx2_quants(int n, int nmax, const float * x, const float * weights, uint8_t * L, float * the_min,
                       uint8_t * Laux, float rmin, float rdelta, int nstep, bool use_mad) {
    float min   = x[0];
    float max   = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) {
            min = x[i];
        }
        if (x[i] > max) {
            max = x[i];
        }
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        *the_min = -min;
        return 0.f;
    }
    float iscale   = nmax / (max - min);
    float scale    = 1 / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l      = nearest_int(iscale * (x[i] - min));
        L[i]       = std::max<int>(0, std::min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff       = use_mad ? fabsf(diff) : diff * diff;
        float w    = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale      = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l   = nearest_int(iscale * (x[i] - min));
            l       = std::max<int>(0, std::min(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min   = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff       = use_mad ? fabsf(diff) : diff * diff;
                float w    = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale    = this_scale;
                min      = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

void quantize_row_fp16(const float * src, void * dst, size_t count, const float * f16_to_f32_table) {
    auto * out = reinterpret_cast<npu_device_fp16_t *>(dst);
    // TODO: use hvx intrinsics for better performance
    for (size_t i = 0; i < count; i++) {
        out[i] = to_fp16(src[i]);
    }
}

void quantize_row_q8_0(const float * src, void * dst, size_t count, const float * f16_to_f32_table) {
    const int nb  = count / QUANT_BLOCK_SIZE;
    auto *    out = reinterpret_cast<npu_device_block_q8_0 *>(dst);

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;  // absolute max

        for (int j = 0; j < QUANT_BLOCK_SIZE; j++) {
            const float v = src[i * QUANT_BLOCK_SIZE + j];
            amax          = std::max(amax, fabsf(v));
        }

        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        out[i].d = to_fp16(d);

        for (int j = 0; j < QUANT_BLOCK_SIZE; ++j) {
            const float x0 = src[i * QUANT_BLOCK_SIZE + j] * id;

            out[i].qs[j] = roundf(x0);
        }
    }
}

void quantize_row_q4_0(const float * src, void * dst, size_t count, const float * f16_to_f32_table) {
    constexpr const int qk = QUANT_BLOCK_SIZE;

    const int nb  = count / qk;
    auto *    out = reinterpret_cast<npu_device_block_q4_0 *>(dst);

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;  // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = src[i * qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f / d : 0.0f;

        out[i].d = to_fp16(d);

        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = src[i * qk + 0 + j] * id;
            const float x1 = src[i * qk + qk / 2 + j] * id;

            const uint8_t xi0 = std::min<int8_t>(15, (x0 + 8.5f));
            const uint8_t xi1 = std::min<int8_t>(15, (x1 + 8.5f));

            out[i].qs[j] = xi0;
            out[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q4_K(const float * src, void * dst, size_t count, const float * f16_to_f32_table) {
    const int nb  = count / QUANT_K_BLOCK_SIZE;
    auto *    out = reinterpret_cast<npu_device_block_q4_k *>(dst);

    uint8_t L[QUANT_K_BLOCK_SIZE];
    uint8_t Laux[32];
    float   weights[32];
    float   mins[QUANT_K_BLOCK_SIZE / 32];
    float   scales[QUANT_K_BLOCK_SIZE / 32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0;  // as we are deducting the min, scales are always positive
        float max_min   = 0;
        for (int j = 0; j < QUANT_K_BLOCK_SIZE / 32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) {
                sum_x2 += src[32 * j + l] * src[32 * j + l];
            }
            float av_x = sqrtf(sum_x2 / 32);
            for (int l = 0; l < 32; ++l) {
                weights[l] = av_x + fabsf(src[32 * j + l]);
            }
            scales[j] =
                make_qkx2_quants(32, 15, src + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min   = max_min > 0 ? 63.f / max_min : 0.f;
        for (int j = 0; j < QUANT_K_BLOCK_SIZE / 32; ++j) {
            uint8_t ls = nearest_int(inv_scale * scales[j]);
            uint8_t lm = nearest_int(inv_min * mins[j]);
            ls         = std::min<uint8_t>(63, ls);
            lm         = std::min<uint8_t>(63, lm);
            if (j < 4) {
                out[i].scales[j]     = ls;
                out[i].scales[j + 4] = lm;
            } else {
                out[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                out[i].scales[j - 4] |= ((ls >> 4) << 6);
                out[i].scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
        out[i].d    = to_fp16(max_scale / 63.f);
        out[i].dmin = to_fp16(max_min / 63.f);

        uint8_t sc, m;
        for (int j = 0; j < QUANT_K_BLOCK_SIZE / 32; ++j) {
            get_scale_min_k4(j, out[i].scales, &sc, &m);
            const float d = f16_to_f32_table[out[i].d] * sc;
            if (!d) {
                continue;
            }
            const float dm = f16_to_f32_table[out[i].dmin] * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l          = nearest_int((src[32 * j + ii] + dm) / d);
                l              = std::max<int>(0, std::min<int>(15, l));
                L[32 * j + ii] = l;
            }
        }

        uint8_t * q = out[i].qs;
        for (int j = 0; j < QUANT_K_BLOCK_SIZE; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = L[j + l] | (L[j + l + 32] << 4);
            }
            q += 32;
        }

        src += QUANT_K_BLOCK_SIZE;
    }
}

void dequantize_row_q8_0(const void * src, float * dst, size_t count, const float * f16_to_f32_table) {
    constexpr const int qk = QUANT_BLOCK_SIZE;
    static_assert(QUANT_BLOCK_SIZE == hexagon::kBytesPerVector / sizeof(float));

    const int     nb      = count / qk;
    const auto *  src_ptr = reinterpret_cast<const npu_device_block_q8_0 *>(src);
    HVX_UVector * out     = ((HVX_UVector *) dst);  // TODO: opt for aligned access

    for (int i = 0; i < nb; i++) {
        const auto & src = src_ptr[i];
        HVX_Vector   d   = Q6_Vh_vsplat_R(src.d);

        HVX_Vector     q_lo = load_block_generic(src);
        HVX_VectorPair q    = Q6_Wh_vunpack_Vb(q_lo);
        q                   = Q6_Wh_vunpack_Vb(Q6_V_lo_W(q));
        q_lo                = Q6_Vhf_equals_Vh(Q6_V_lo_W(q));
        q                   = Q6_Wqf32_vmpy_VhfVhf(q_lo, d);
        out[i]              = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(q));
    }
}

void dequantize_row_q4_0(const void * src, float * dst, size_t count, const float * f16_to_f32_table) {
    constexpr const int qk = QUANT_BLOCK_SIZE;
    static_assert(qk % 2 == 0, "qk must be even");
    static_assert(QUANT_BLOCK_SIZE == hexagon::kBytesPerVector / sizeof(float));
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    const int     nb      = count / qk;
    const auto *  src_ptr = reinterpret_cast<const npu_device_block_q4_0 *>(src);
    HVX_Vector    mask    = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector    minus   = Q6_Vb_vsplat_R(8);
    HVX_UVector * out     = ((HVX_UVector *) dst);  // TODO: opt for aligned access

    const int loop_count = nb - (nb % 2);
    for (int i = 0; i < loop_count; i += 2) {
        const auto & src1 = src_ptr[i];
        const auto & src2 = src_ptr[i + 1];

        HVX_Vector d1 = Q6_Vh_vsplat_R(src1.d);
        HVX_Vector d2 = Q6_Vh_vsplat_R(src2.d);
        HVX_Vector d  = Q6_Vh_vshuff_Vh(Q6_V_valign_VVR(d2, d1, hexagon::kBytesPerVector / 2));

        HVX_Vector     q_lo = load_dual_block_generic(src1, src2);
        HVX_Vector     q_hi = Q6_Vub_vlsr_VubR(q_lo, 4);
        HVX_VectorPair q    = Q6_W_vshuff_VVR(q_hi, Q6_V_vand_VV(q_lo, mask), kSizeOfQs);
        q_lo                = Q6_V_valign_VVR(Q6_V_lo_W(q), Q6_V_vzero(), hexagon::kBytesPerVector / 2);
        q_lo                = Q6_V_valign_VVR(Q6_V_hi_W(q), q_lo, hexagon::kBytesPerVector / 2);
        q_lo                = Q6_Vb_vshuff_Vb(q_lo);
        q_lo                = Q6_Vb_vsub_VbVb(q_lo, minus);
        q                   = Q6_Wh_vunpack_Vb(q_lo);
        q_lo                = Q6_Vhf_equals_Vh(Q6_V_lo_W(q));
        q                   = Q6_Wqf32_vmpy_VhfVhf(q_lo, d);
        out[i]              = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(q));
        out[i + 1]          = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(q));
    }

    if (loop_count < nb) {
        const auto & curr_blk = src_ptr[nb - 1];
        HVX_Vector   d        = Q6_Vh_vsplat_R(curr_blk.d);

        HVX_Vector q_lo = load_block_generic(curr_blk);
        HVX_Vector q_hi = Q6_Vub_vlsr_VubR(q_lo, 4);
        q_lo            = Q6_V_valign_VVR(Q6_V_vand_VV(q_lo, mask), Q6_V_vzero(), sizeof(curr_blk.qs));
        q_lo            = Q6_V_valign_VVR(q_hi, q_lo, hexagon::kBytesPerVector - sizeof(curr_blk.qs));
        q_lo            = Q6_Vb_vsub_VbVb(q_lo, minus);

        HVX_VectorPair q = Q6_Wh_vunpack_Vb(q_lo);
        q                = Q6_Wh_vunpack_Vb(Q6_V_lo_W(q));
        q_lo             = Q6_Vhf_equals_Vh(Q6_V_lo_W(q));
        q                = Q6_Wqf32_vmpy_VhfVhf(q_lo, d);
        out[nb - 1]      = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(q));
    }
}

void dequantize_row_q4_K(const void * src, float * dst, size_t count, const float * f16_to_f32_table) {
    const int    nb      = count / QUANT_K_BLOCK_SIZE;
    const auto * src_ptr = reinterpret_cast<const npu_device_block_q4_k *>(src);

    // TODO: use intrinsics
    for (int i = 0; i < nb; i++) {
        const uint8_t * q = src_ptr[i].qs;

        const float d   = f16_to_f32_table[src_ptr[i].d];
        const float min = f16_to_f32_table[src_ptr[i].dmin];

        int          is     = 0;
        uint8_t      sc     = 0;
        uint8_t      m      = 0;
        const auto * scales = src_ptr[i].scales;
        for (int j = 0; j < QUANT_K_BLOCK_SIZE; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                dst[0]  = d1 * (q[l] & 0xF) - m1;
                dst[32] = d2 * ((q[l] >> 4) & 0xF) - m2;
                dst++;
            }
            dst += 32;
            q += 32;
            is += 2;
        }
    }
}

template <typename _TFunc> struct dot_func_traits {};

template <typename _TData> struct dot_func_traits<float (*)(_TData, _TData, size_t)> {
    using param_type = std::remove_const_t<std::remove_pointer_t<_TData>>;
};

template <auto _Func> float wrap_dot_func(const void * src0, const void * src1, size_t count) {
    using param_type = typename dot_func_traits<decltype(_Func)>::param_type;
    return _Func(reinterpret_cast<const param_type *>(src0), reinterpret_cast<const param_type *>(src1), count);
}

constexpr const hexagon::device_type_traits kDeviceTypeTraits[] = {
    { NPU_DATA_TYPE_F32, "F32", 1, sizeof(float), false, nullptr, nullptr,
     wrap_dot_func<hexagon::vec_dot_product_f32_f32> },
    { NPU_DATA_TYPE_F16, "F16", 1, sizeof(npu_device_fp16_t), false, nullptr, quantize_row_fp16,
     wrap_dot_func<hexagon::vec_dot_product_f16_f16> },
    { NPU_DATA_TYPE_Q8_0, "Q8_0", QUANT_BLOCK_SIZE, sizeof(npu_device_block_q8_0), true, dequantize_row_q8_0,
     quantize_row_q8_0 },
    { NPU_DATA_TYPE_Q4_0, "Q4_0", QUANT_BLOCK_SIZE, sizeof(npu_device_block_q4_0), true, dequantize_row_q4_0,
     quantize_row_q4_0 },
    { NPU_DATA_TYPE_Q4_K, "Q4_K", QUANT_K_BLOCK_SIZE, sizeof(npu_device_block_q4_k), true, dequantize_row_q4_K,
     quantize_row_q4_K },
};

static_assert(std::size(kDeviceTypeTraits) == NPU_DATA_TYPE_COUNT,
              "kDeviceTypeTraits size mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_F32].type == NPU_DATA_TYPE_F32,
              "kDeviceTypeTraits F32 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_F16].type == NPU_DATA_TYPE_F16,
              "kDeviceTypeTraits F16 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_Q8_0].type == NPU_DATA_TYPE_Q8_0,
              "kDeviceTypeTraits Q8_0 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_Q4_0].type == NPU_DATA_TYPE_Q4_0,
              "kDeviceTypeTraits Q4_0 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_Q4_K].type == NPU_DATA_TYPE_Q4_K,
              "kDeviceTypeTraits Q4_K type mismatch with npu_device_tensor_data_type enum");

}  // namespace

namespace hexagon {

bool init_f16_f32_table(float * table, size_t count) {
    constexpr const size_t kTableSize = (1U << 16);
    if (count < kTableSize) {
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        table[i] = to_float(i);
    }

    return true;
}

const device_type_traits & get_type_traits(npu_device_tensor_data_type type) {
    return kDeviceTypeTraits[type];
}

}  // namespace hexagon
