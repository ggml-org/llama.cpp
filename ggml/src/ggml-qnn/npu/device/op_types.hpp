#pragma once

#include <hexagon_types.h>

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
    const size_t                       vtcm_quota_size;
    const float *                      f16_to_f32_table;
    std::unique_ptr<hexagon::vtcm_mem> vtcm_cache;
    std::unique_ptr<uint8_t[]>         mem_cache;
    size_t                             mem_cache_size = 0;

    uint8_t * get_vtcm_cache(size_t size) {
        if (!vtcm_cache || vtcm_cache->get_size() < size) {
            vtcm_cache = std::make_unique<hexagon::vtcm_mem>(size, false);
        }

        if (!vtcm_cache->is_valid()) {
            return nullptr;
        }

        return vtcm_cache->get_mem();
    }

    uint8_t * get_mem_cache(size_t size) {
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
    if (total <= 0 || tidx >= tcnt) {
        return { 0, 0 };  // No work for this thread
    }

    const auto elements_per_thread = total / tcnt;
    const auto remainder           = total % tcnt;

    int64_t start = 0;
    int64_t end   = 0;
    if (tidx < remainder) {
        // First 'remainder' threads get one extra item
        start = tidx * (elements_per_thread + 1);
        end   = start + elements_per_thread + 1;
    } else {
        // Remaining threads get the base number of elements
        start = remainder * (elements_per_thread + 1) + (tidx - remainder) * elements_per_thread;
        end   = start + elements_per_thread;
    }

    return { start, std::min(end, total) };
}

constexpr const size_t kBytesPerVector      = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kAlignMask           = kBytesPerVector - 1;
constexpr const size_t kL2CacheSize         = 8 * 1024;            // // 8KB L2 cache
constexpr const size_t kL2FetchAheadVectors = kL2CacheSize / kBytesPerVector;

}  // namespace hexagon
