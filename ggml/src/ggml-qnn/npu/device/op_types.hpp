#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "hexagon_npu.h"
#include "tensor.hpp"
#include "util.hpp"
#include "vtcm_mem.hpp"

namespace hexagon {

struct compute_params {
    const size_t                       tidx;
    const size_t                       tcnt;
    const float *                      f16_to_f32_table;
    std::unique_ptr<hexagon::vtcm_mem> vtcm_cache;
    std::unique_ptr<uint8_t[]>         mem_cache;
    size_t                             mem_cache_size = 0;

    uint8_t * get_cache(size_t size, bool fallback_to_mem) {
        if (!vtcm_cache || vtcm_cache->get_size() < size) {
            vtcm_cache = std::make_unique<hexagon::vtcm_mem>(size, false);
        }

        if (vtcm_cache->is_valid()) {
            return vtcm_cache->get_mem();
        }

        if (!fallback_to_mem) {
            DEVICE_LOG_DEBUG("vtcm_mem not valid, return nullptr\n");
            return nullptr;
        }

        DEVICE_LOG_DEBUG("vtcm_mem not valid, allocate from mem_cache\n");
        if (!mem_cache || mem_cache_size < size) {
            mem_cache      = std::make_unique<uint8_t[]>(size + 256);
            mem_cache_size = mem_cache ? size : 0;
        }

        return mem_cache.get();
    }
};

typedef bool (*compute_func_type)(tensor * dst, compute_params * params);
typedef bool (*op_is_supported_func_type)(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                                          const npu_device_tensor_spec & dst, npu_device_tensor_op op);

inline constexpr std::pair<int64_t, int64_t> get_thread_work_slice(int64_t total, size_t tidx, size_t tcnt) {
    const auto elements_per_thread = (total + tcnt - 1) / tcnt;
    const auto start               = tidx * elements_per_thread;
    const auto end                 = std::min<int64_t>(start + elements_per_thread, total);
    return { start, end };
}

}  // namespace hexagon
