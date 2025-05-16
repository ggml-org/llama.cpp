#include "quants.hpp"

#include <hexagon_types.h>

#include <array>

static_assert(sizeof(npu_device_block_q4_K) ==
                  2 * sizeof(npu_device_fp16_t) + QUANT_K_SCALE_SIZE + QUANT_K_BLOCK_SIZE / 2,
              "wrong q4_K block size/padding");

static_assert(sizeof(npu_device_block_q4_0) == sizeof(npu_device_fp16_t) + QUANT_BLOCK_SIZE / 2,
              "wrong q4_0 block size/padding");

static_assert(sizeof(npu_device_block_q8_0) == sizeof(npu_device_fp16_t) + QUANT_BLOCK_SIZE,
              "wrong q8_0 block size/padding");

namespace {

inline float to_float(const npu_device_fp16_t src) {
    union {
        __fp16 f16;
        npu_device_fp16_t u16;
    } f16;

    f16.u16 = src;
    return f16.f16;
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
    constexpr const int qk      = QUANT_BLOCK_SIZE;
    const int           nb      = count / qk;
    const auto *        src_ptr = reinterpret_cast<const npu_device_block_q8_0 *>(src);

    // TODO: use intrinsics
    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32_table[src_ptr[i].d];

        for (int j = 0; j < qk; ++j) {
            dst[i * qk + j] = src_ptr[i].qs[j] * d;
        }
    }
}

void dequantize_row_q4_0(const void * src, float * dst, size_t count, const float * f16_to_f32_table) {
    constexpr const int qk = QUANT_BLOCK_SIZE;
    static_assert(qk % 2 == 0, "qk must be even");

    const int    nb      = count / qk;
    const auto * src_ptr = reinterpret_cast<const npu_device_block_q4_0 *>(src);

    // TODO: use intrinsics
    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32_table[src_ptr[i].d];

        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (src_ptr[i].qs[j] & 0x0F) - 8;
            const int x1 = ((src_ptr[i].qs[j] >> 4) & 0xF) - 8;

            dst[i * qk + j + 0]      = x0 * d;
            dst[i * qk + j + qk / 2] = x1 * d;
        }
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
