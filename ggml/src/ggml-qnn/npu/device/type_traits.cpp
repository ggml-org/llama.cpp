#include "type_traits.hpp"

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

inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    // TODO: use intrinsics
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

float make_qkx2_quants(int           n,
                       int           nmax,
                       const float * x,
                       const float * weights,
                       uint8_t *     L,
                       float *       the_min,
                       uint8_t *     Laux,
                       float         rmin,
                       float         rdelta,
                       int           nstep,
                       bool          use_mad) {
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

void quantize_row_fp16(const float * src, void * dst, size_t count) {
    auto * out = reinterpret_cast<npu_device_fp16_t *>(dst);
    // TODO: use hvx intrinsics for better performance
    for (size_t i = 0; i < count; i++) {
        out[i] = to_fp16(src[i]);
    }
}

void quantize_row_q8_0(const float * src, void * dst, size_t count) {
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

void quantize_row_q4_0(const float * src, void * dst, size_t count) {
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

void quantize_row_q4_K(const float * src, void * dst, size_t count) {
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
            const float d = to_float(out[i].d) * sc;
            if (!d) {
                continue;
            }
            const float dm = to_float(out[i].dmin) * m;
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

void dequantize_row_q8_0(const void * src, hexagon::dequant_output_type * dst, size_t count, HVX_Vector) {
    using namespace hexagon::vec::quant;

    constexpr const int qk = QUANT_BLOCK_SIZE;
    static_assert(QUANT_BLOCK_SIZE == hexagon::kBytesPerVector / sizeof(float));

    alignas(hexagon::kBytesPerVector) static const HVX_Vector qs_indices = make_qs_load_mask<npu_device_block_q8_0>();
    alignas(hexagon::kBytesPerVector) static const HVX_Vector scale_indices =
        make_scale_load_mask<npu_device_block_q8_0>();

    const int    nb      = count / qk;
    const auto * src_ptr = reinterpret_cast<const npu_device_block_q8_0 *>(src);
    auto *       dst_ptr = ((hexagon::dequant_output_type *) dst);  // TODO: opt for aligned access

    int i = 0;
    for (; i + 1 < nb; i += 2) {
        auto qs = load_dual_block_generic(src_ptr + i, qs_indices, scale_indices);

        HVX_Vector q_lo   = Q6_Vhf_equals_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(qs.val[0])));
        HVX_Vector result = Q6_Vqf16_vmpy_VhfVhf(q_lo, qs.val[1]);

        *reinterpret_cast<HVX_UVector *>(dst_ptr) = Q6_Vhf_equals_Vqf16(result);
        dst_ptr += qk * 2;
    }

    if (i < nb) {
        const auto & src = src_ptr[i];

        HVX_Vector scales = Q6_Vh_vsplat_R(src.d);

        HVX_Vector q_lo   = load_block_generic(src);
        q_lo              = Q6_Vhf_equals_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(q_lo)));
        HVX_Vector result = Q6_Vqf16_vmpy_VhfVhf(q_lo, scales);
        hexagon::q6op_vstu_variable_ARV<hexagon::kBytesPerVector / 2>(
            dst_ptr,
            Q6_Vhf_equals_Vqf16(result));  // TODO: opt the store
    }
}

template <bool _IsDstAligned>
void dequantize_row_q4_0_impl(const void * src, hexagon::dequant_output_type * dst, size_t count, HVX_Vector table) {
    using namespace hexagon::vec::quant;

    constexpr const size_t   kElemsPerVec = hexagon::kBytesPerVector / sizeof(hexagon::dequant_output_type);
    constexpr const uint32_t kSizeOfQs    = sizeof(npu_device_block_q4_0::qs);
    constexpr const int      qk           = QUANT_BLOCK_SIZE;
    static_assert(qk % 2 == 0, "qk must be even");
    static_assert(QUANT_BLOCK_SIZE == hexagon::kBytesPerVector / sizeof(float));

    alignas(hexagon::kBytesPerVector) static const HVX_Vector qs_indices = make_qs_load_mask<npu_device_block_q4_0>();
    alignas(hexagon::kBytesPerVector) static const HVX_Vector scale_indices =
        make_scale_load_mask<npu_device_block_q4_0>();

    const int    nb      = count / qk;
    const auto * src_ptr = reinterpret_cast<const npu_device_block_q4_0 *>(src);

    hexagon::dequant_output_type * dst_ptr = dst;
    int                            i       = 0;
    for (; i + 5 < nb; i += 6) {
        auto       qs      = load_hexa_block_generic(src_ptr + i, qs_indices, scale_indices);
        auto       res01   = dequantize_vec_q40_qf16_4blocks(qs.val[0], qs.val[1], qs.val[2], table);
        HVX_Vector block45 = Q6_V_vror_VR(qs.val[0], kSizeOfQs * 4);
        auto       res2    = dequantize_vec_q40_qf16_2blocks(block45, qs.val[3], table);
        if constexpr (_IsDstAligned) {
            reinterpret_cast<HVX_Vector *>(dst_ptr)[0] = Q6_Vhf_equals_Vqf16(res01.val[0]);
            reinterpret_cast<HVX_Vector *>(dst_ptr)[1] = Q6_Vhf_equals_Vqf16(res01.val[1]);
            reinterpret_cast<HVX_Vector *>(dst_ptr)[2] = Q6_Vhf_equals_Vqf16(res2);
        } else {
            reinterpret_cast<HVX_UVector *>(dst_ptr)[0] = Q6_Vhf_equals_Vqf16(res01.val[0]);
            reinterpret_cast<HVX_UVector *>(dst_ptr)[1] = Q6_Vhf_equals_Vqf16(res01.val[1]);
            reinterpret_cast<HVX_UVector *>(dst_ptr)[2] = Q6_Vhf_equals_Vqf16(res2);
        }
        dst_ptr += kElemsPerVec * 3;
    }

    for (; i + 3 < nb; i += 4) {
        auto qs    = load_qual_block_generic(src_ptr + i, qs_indices, scale_indices);
        auto res01 = dequantize_vec_q40_qf16_4blocks(qs.val[0], qs.val[1], qs.val[2], table);
        if constexpr (_IsDstAligned) {
            reinterpret_cast<HVX_Vector *>(dst_ptr)[0] = Q6_Vhf_equals_Vqf16(res01.val[0]);
            reinterpret_cast<HVX_Vector *>(dst_ptr)[1] = Q6_Vhf_equals_Vqf16(res01.val[1]);
        } else {
            reinterpret_cast<HVX_UVector *>(dst_ptr)[0] = Q6_Vhf_equals_Vqf16(res01.val[0]);
            reinterpret_cast<HVX_UVector *>(dst_ptr)[1] = Q6_Vhf_equals_Vqf16(res01.val[1]);
        }
        dst_ptr += kElemsPerVec * 2;
    }

    for (; i + 1 < nb; i += 2) {
        auto qs  = load_dual_block_generic(src_ptr + i, qs_indices, scale_indices);
        auto res = dequantize_vec_q40_qf16_2blocks(qs.val[0], qs.val[1], table);
        if constexpr (_IsDstAligned) {
            *reinterpret_cast<HVX_Vector *>(dst_ptr) = Q6_Vhf_equals_Vqf16(res);
        } else {
            *reinterpret_cast<HVX_UVector *>(dst_ptr) = Q6_Vhf_equals_Vqf16(res);
        }
        dst_ptr += kElemsPerVec;
    }

    if (i < nb) {
        const auto & curr_blk = src_ptr[nb - 1];
        HVX_Vector   scales   = Q6_Vh_vsplat_R(curr_blk.d);

        HVX_Vector     qs   = load_block_generic(curr_blk);
        HVX_Vector     q_lo = qs;
        HVX_Vector     q_hi = Q6_Vub_vlsr_VubR(qs, 4);
        HVX_VectorPair qp0  = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs);

        q_lo = Q6_Vb_vshuff_Vb(Q6_V_lo_W(qp0));
        qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

        q_lo = Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(qp0), scales);
        q_lo = Q6_Vhf_equals_Vqf16(q_lo);

        if constexpr (_IsDstAligned) {
            hexagon::q6op_vstu_variable_aligned<hexagon::kBytesPerVector / 2>(dst_ptr, q_lo);
        } else {
            hexagon::q6op_vstu_variable_ARV<hexagon::kBytesPerVector / 2>(dst_ptr, q_lo);
        }
    }
}

HVX_Vector load_dequant_table_q4_0() {
    constexpr const int kTableSize   = 1 << 4;  // 4 bits per value, 16 values
    constexpr const int kQ4ZeroPoint = 8;       // zero point for q4_0 quantization
    static_assert(kTableSize <= hexagon::kBytesPerVector / sizeof(__fp16), "table too large");

    alignas(hexagon::kBytesPerVector) static const HVX_Vector result = []() -> HVX_Vector {
        alignas(hexagon::kBytesPerVector) hexagon::HVX_VectorAlias table;
        table.v = Q6_V_vzero();
        for (int i = 0; i < kTableSize; ++i) {
            table.f16[i * 2] = i - kQ4ZeroPoint;  // TODO: vectorize this?
        }
        return table.v;
    }();

    return result;
}

void dequantize_row_q4_0(const void * src, hexagon::dequant_output_type * dst, size_t count, HVX_Vector table) {
    const bool dst_aligned = hexagon::is_addr_aligned(dst);
    if (dst_aligned) {
        dequantize_row_q4_0_impl<true>(src, dst, count, table);
    } else {
        dequantize_row_q4_0_impl<false>(src, dst, count, table);
    }
}

HVX_Vector load_dequant_table_q4_k() {
    constexpr const int kTableSize = 1 << 4;  // 4 bits per value, 16 values
    static_assert(kTableSize <= hexagon::kBytesPerVector / sizeof(__fp16), "table too large");

    alignas(hexagon::kBytesPerVector) static const HVX_Vector result = []() -> HVX_Vector {
        alignas(hexagon::kBytesPerVector) hexagon::HVX_VectorAlias table;
        table.v = Q6_V_vzero();
        for (int i = 0; i < kTableSize; ++i) {
            table.f16[i * 2] = i;  // TODO: vectorize this?
        }
        return table.v;
    }();

    return result;
}

void dequantize_row_q4_K(const void * src, hexagon::dequant_output_type * dst, size_t count, HVX_Vector table) {
    constexpr const int kQuantSubBlockSize = 32;

    const int    nb      = count / QUANT_K_BLOCK_SIZE;
    const auto * src_ptr = reinterpret_cast<const npu_device_block_q4_k *>(src);
    auto *       dst_ptr = reinterpret_cast<npu_device_fp16_t *>(dst);

    const HVX_VectorPred scale_mask = Q6_Q_vsetq_R(hexagon::kBytesPerVector / 2);

    alignas(hexagon::kBytesPerVector * 4) union {
        HVX_VectorPair p[2];
        HVX_Vector     v[4];
    } dual_pair;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = src_ptr[i].qs;

        HVX_Vector qv = *reinterpret_cast<const HVX_UVector *>(q);

        HVX_Vector     q_lo = qv;
        HVX_Vector     q_hi = Q6_Vub_vlsr_VubR(qv, 4);
        HVX_VectorPair qp   = Q6_W_vshuff_VVR(q_hi, q_lo, kQuantSubBlockSize * 3);

        q_lo = Q6_V_lo_W(qp);
        q_hi = Q6_V_hi_W(qp);

        q_lo = Q6_Vb_vshuff_Vb(q_lo);
        q_hi = Q6_Vb_vshuff_Vb(q_hi);

        dual_pair.p[0] = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);
        dual_pair.p[1] = Q6_Wh_vlut16_VbVhR_nomatch(q_hi, table, 0);

        const __fp16 d   = reinterpret_cast<const __fp16 &>(src_ptr[i].d);
        const __fp16 min = reinterpret_cast<const __fp16 &>(src_ptr[i].dmin);

        int          is     = 0;
        uint8_t      sc     = 0;
        uint8_t      m      = 0;
        const auto * scales = src_ptr[i].scales;
        for (int j = 0; j < QUANT_K_BLOCK_SIZE; j += 128) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const __fp16 d0 = d * sc;
            const __fp16 m0 = min * m;

            HVX_Vector dv0 = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(d0));
            HVX_Vector dm0 = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(m0));

            get_scale_min_k4(is + 1, scales, &sc, &m);
            const __fp16 d1 = d * sc;
            const __fp16 m1 = min * m;

            HVX_Vector dv1 = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(d1));
            HVX_Vector dm1 = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(m1));

            get_scale_min_k4(is + 2, scales, &sc, &m);
            const __fp16 d2 = d * sc;
            const __fp16 m2 = min * m;

            HVX_Vector dv2 = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(d2));
            HVX_Vector dm2 = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(m2));

            get_scale_min_k4(is + 3, scales, &sc, &m);
            const __fp16 d3 = d * sc;
            const __fp16 m3 = min * m;

            HVX_Vector dv3 = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(d3));
            HVX_Vector dm3 = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t &>(m3));

            HVX_Vector dv01 = Q6_V_vmux_QVV(scale_mask, dv0, dv1);
            HVX_Vector dm01 = Q6_V_vmux_QVV(scale_mask, dm0, dm1);

            HVX_Vector dv23 = Q6_V_vmux_QVV(scale_mask, dv2, dv3);
            HVX_Vector dm23 = Q6_V_vmux_QVV(scale_mask, dm2, dm3);

            q_lo = Q6_Vqf16_vmpy_VhfVhf(dual_pair.v[j / 64], dv01);
            q_lo = Q6_Vqf16_vsub_Vqf16Vhf(q_lo, dm01);

            q_hi = Q6_Vqf16_vmpy_VhfVhf(dual_pair.v[j / 64 + 1], dv23);
            q_hi = Q6_Vqf16_vsub_Vqf16Vhf(q_hi, dm23);

            reinterpret_cast<HVX_UVector *>(dst_ptr)[0] = Q6_Vhf_equals_Vqf16(q_lo);
            reinterpret_cast<HVX_UVector *>(dst_ptr)[1] = Q6_Vhf_equals_Vqf16(q_hi);

            dst_ptr += 128;
            is += 4;
        }
    }
}

void copy_row_f16(const void * src, hexagon::dequant_output_type * dst, size_t count, HVX_Vector) {
    hexagon::vec_cpy_f16(reinterpret_cast<const npu_device_fp16_t *>(src), dst, count);
}

template <typename _TSrc, typename _TDst, typename... _TExtArgs>
void copy_row_f32(const _TSrc * src, _TDst * dst, size_t count, _TExtArgs...) {
    hexagon::vec_cpy_f32(reinterpret_cast<const float *>(src), reinterpret_cast<float *>(dst), count);
}

constexpr const hexagon::device_type_traits kDeviceTypeTraits[] = {
    { NPU_DATA_TYPE_F32, "F32", 1, sizeof(float), false, copy_row_f32<void, hexagon::dequant_output_type, HVX_Vector>,
     copy_row_f32<float, void>, hexagon::type_erase_dot_func<hexagon::vec_dot_product_f32_f32>,
     hexagon::type_erase_dot_func<hexagon::vec_dot_product_aligned_f32_f32>,
     hexagon::type_erase_dot_func<hexagon::is_f32_f32_dot_product_aligned> },
    { NPU_DATA_TYPE_F16, "F16", 1, sizeof(npu_device_fp16_t), false, copy_row_f16, quantize_row_fp16,
     hexagon::type_erase_dot_func<hexagon::vec_dot_product_f16_f16>,
     hexagon::type_erase_dot_func<hexagon::vec_dot_product_aligned_f16_f16>,
     hexagon::type_erase_dot_func<hexagon::is_f16_f16_dot_product_aligned> },
    { NPU_DATA_TYPE_I32, "I32", 1, sizeof(int32_t), false },
    { NPU_DATA_TYPE_I64, "I64", 1, sizeof(int64_t), false },
    { NPU_DATA_TYPE_Q8_0, "Q8_0", QUANT_BLOCK_SIZE, sizeof(npu_device_block_q8_0), true, dequantize_row_q8_0,
     quantize_row_q8_0 },
    { NPU_DATA_TYPE_Q4_0, "Q4_0", QUANT_BLOCK_SIZE, sizeof(npu_device_block_q4_0), true, dequantize_row_q4_0,
     quantize_row_q4_0, nullptr, nullptr, nullptr, load_dequant_table_q4_0 },
    { NPU_DATA_TYPE_Q4_K, "Q4_K", QUANT_K_BLOCK_SIZE, sizeof(npu_device_block_q4_k), true, dequantize_row_q4_K,
     quantize_row_q4_K, nullptr, nullptr, nullptr, load_dequant_table_q4_k },
};

static_assert(std::size(kDeviceTypeTraits) == NPU_DATA_TYPE_COUNT,
              "kDeviceTypeTraits size mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_F32].type == NPU_DATA_TYPE_F32,
              "kDeviceTypeTraits F32 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_F16].type == NPU_DATA_TYPE_F16,
              "kDeviceTypeTraits F16 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_I32].type == NPU_DATA_TYPE_I32,
              "kDeviceTypeTraits I32 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_I64].type == NPU_DATA_TYPE_I64,
              "kDeviceTypeTraits I64 type mismatch with npu_device_tensor_data_type enum");
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

size_t get_dequantized_row_size(const tensor * tensor) {
    if (!is_quantized_type(tensor->get_type())) {
        return tensor->get_nb(1);  // for f32 and f16
    }

    auto row_elems_count = tensor->get_ne(0);
    return hexagon::get_aligned_size(
        row_elems_count * sizeof(dequant_output_type));  // dequant_output_type is currently restricted to f32
}

}  // namespace hexagon
