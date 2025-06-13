#include "quants.hpp"

#include <hexagon_types.h>

#include <array>

#include "op_types.hpp"  // TODO: remove this include

static_assert(sizeof(npu_device_block_q4_K) ==
                  2 * sizeof(npu_device_fp16_t) + QUANT_K_SCALE_SIZE + QUANT_K_BLOCK_SIZE / 2,
              "wrong q4_K block size/padding");

static_assert(sizeof(npu_device_block_q4_0) == sizeof(npu_device_fp16_t) + QUANT_BLOCK_SIZE / 2,
              "wrong q4_0 block size/padding");

static_assert(sizeof(npu_device_block_q8_0) == sizeof(npu_device_fp16_t) + QUANT_BLOCK_SIZE,
              "wrong q8_0 block size/padding");

namespace {

inline HVX_Vector vmemu(const void * unaligned_ptr) {
    HVX_Vector ret = *reinterpret_cast<const HVX_UVector *>(unaligned_ptr);
    return ret;
}

inline float to_float(const npu_device_fp16_t src) {
    return reinterpret_cast<const __fp16 &>(src);
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
        d1            = Q6_V_valign_VVR(d1, Q6_V_vzero(), hexagon::kBytesPerVector / 2);
        d1            = Q6_V_valign_VVR(d2, d1, hexagon::kBytesPerVector / 2);
        HVX_Vector d  = Q6_Vh_vshuff_Vh(d1);

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
    const auto * src_ptr = reinterpret_cast<const npu_device_block_q4_K *>(src);

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

constexpr const hexagon::device_type_traits kDeviceTypeTraits[] = {
    { NPU_DATA_TYPE_F32,  "F32",  1,                  false, nullptr             },
    { NPU_DATA_TYPE_F16,  "F16",  1,                  false, nullptr             },
    { NPU_DATA_TYPE_Q8_0, "Q8_0", QUANT_BLOCK_SIZE,   true,  dequantize_row_q8_0 },
    { NPU_DATA_TYPE_Q4_0, "Q4_0", QUANT_BLOCK_SIZE,   true,  dequantize_row_q4_0 },
    { NPU_DATA_TYPE_Q4_K, "Q4_K", QUANT_K_BLOCK_SIZE, true,  dequantize_row_q4_K },
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
