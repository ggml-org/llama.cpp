//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "onednn-woq.hpp"

#include "convert.hpp"

namespace ggml_sycl::onednn_woq {

bool pack_q4_0_aos_to_s4(const void * src,
                         int64_t      m,
                         int64_t      k,
                         packed_weights & out,
                         std::string * error) {
    if (!src || m <= 0 || k <= 0) {
        if (error) {
            *error = "invalid arguments for pack_q4_0_aos_to_s4";
        }
        return false;
    }
    if (k % QK4_0 != 0) {
        if (error) {
            *error = "Q4_0 WoQ requires K divisible by QK4_0";
        }
        return false;
    }

    const int64_t groups = k / QK4_0;
    const size_t total_elems = static_cast<size_t>(k) * static_cast<size_t>(m);
    out.s4.assign((total_elems + 1) / 2, 0);
    out.scales.assign(static_cast<size_t>(groups) * static_cast<size_t>(m), 0.0f);
    out.zero_points.assign(static_cast<size_t>(groups) * static_cast<size_t>(m), 0);
    out.group_size = QK4_0;
    out.scales_mask = (1 << 0) | (1 << 1);
    out.zero_points_mask = (1 << 0) | (1 << 1);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, k);
    for (int64_t row = 0; row < m; ++row) {
        const uint8_t * row_ptr = static_cast<const uint8_t *>(src) + static_cast<size_t>(row) * row_bytes;
        const block_q4_0 * blocks = reinterpret_cast<const block_q4_0 *>(row_ptr);
        for (int64_t g = 0; g < groups; ++g) {
            const block_q4_0 & blk = blocks[g];
            const ggml_fp16_t scale_bits = *reinterpret_cast<const ggml_fp16_t *>(&blk.d);
            const float scale = ggml_fp16_to_fp32(scale_bits);
            out.scales[static_cast<size_t>(g) * static_cast<size_t>(m) + static_cast<size_t>(row)] = scale;

            const uint8_t * qs = blk.qs;
            for (int i = 0; i < QK4_0 / 2; ++i) {
                const uint8_t byte = qs[i];
                const int8_t v0 = static_cast<int8_t>((byte & 0x0F) - 8);
                const int8_t v1 = static_cast<int8_t>((byte >> 4) - 8);

                const int64_t k0 = g * QK4_0 + i * 2;
                const int64_t k1 = k0 + 1;

                const size_t idx0 = static_cast<size_t>(k0) * static_cast<size_t>(m) + static_cast<size_t>(row);
                const size_t idx1 = static_cast<size_t>(k1) * static_cast<size_t>(m) + static_cast<size_t>(row);

                const size_t byte0 = idx0 / 2;
                const size_t byte1 = idx1 / 2;
                const uint8_t nib0 = static_cast<uint8_t>(v0) & 0x0F;
                const uint8_t nib1 = static_cast<uint8_t>(v1) & 0x0F;

                if ((idx0 & 1u) == 0) {
                    out.s4[byte0] = static_cast<uint8_t>((out.s4[byte0] & 0xF0) | nib0);
                } else {
                    out.s4[byte0] = static_cast<uint8_t>((out.s4[byte0] & 0x0F) | (nib0 << 4));
                }
                if ((idx1 & 1u) == 0) {
                    out.s4[byte1] = static_cast<uint8_t>((out.s4[byte1] & 0xF0) | nib1);
                } else {
                    out.s4[byte1] = static_cast<uint8_t>((out.s4[byte1] & 0x0F) | (nib1 << 4));
                }
            }
        }
    }

    return true;
}

bool supports_dequant_fp16(ggml_type type) {
    ggml_tensor dummy_src{};
    ggml_tensor dummy_dst{};
    dummy_dst.src[0] = &dummy_src;
    return ggml_get_to_fp16_sycl(type, &dummy_dst, true) != nullptr;
}

} // namespace ggml_sycl::onednn_woq
