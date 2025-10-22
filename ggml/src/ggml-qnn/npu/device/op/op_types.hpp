#pragma once

#include "hexagon_npu.h"
#include "tensor.hpp"
#include "thread_pool.hpp"
#include "util.hpp"
#include "vec_ops.hpp"

#include <hexagon_types.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

namespace hexagon {

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

struct compute_params {
    default_thread_pool::thread_params * const thread_params;
    const float *                              f16_to_f32_table;

    uint8_t * get_vtcm_cache(size_t size) { return thread_params->get_vtcm_cache(size); }

    std::pair<int64_t, int64_t> get_work_slice(int64_t total) const {
        return get_thread_work_slice(total, thread_params->tidx, thread_params->tcnt);
    }

    size_t get_vtcm_quota_size() const { return thread_params->vtcm_quota_size; }

    size_t get_thread_count() const { return thread_params->tcnt; }

    size_t get_thread_index() const { return thread_params->tidx; }

    bool initiate_dma_row_transfer(const uint8_t * src, uint8_t * dst, size_t size) {
        return thread_params->initiate_dma_row_transfer(src, dst, size);
    }

    bool initiate_dma_row_transfer(const uint8_t * src0,
                                   uint8_t *       dst0,
                                   const uint8_t * src1,
                                   uint8_t *       dst1,
                                   size_t          size) {
        return thread_params->initiate_dma_row_transfer(src0, dst0, src1, dst1, size);
    }

    bool initiate_dma_plane_transfer(const uint8_t * src,
                                     uint8_t *       dst,
                                     size_t          width,
                                     size_t          height,
                                     size_t          src_stride,
                                     size_t          dst_stride) {
        return thread_params->initiate_dma_plane_transfer(src, dst, width, height, src_stride, dst_stride);
    }

    void wait_for_dma() { thread_params->wait_for_dma(); }
};

typedef bool (*compute_func_type)(tensor * dst, compute_params * params);
typedef bool (*op_is_supported_func_type)(const npu_device_tensor_op_spec * op_spec,
                                          const npu_device_tensor_spec *    dst,
                                          const npu_device_tensor_spec *    srcs,
                                          size_t                            src_len);
typedef bool (*op_required_sync_func_type)(npu_device_tensor_op       prev_op,
                                           const npu_device_ne_type & prev_ne,
                                           npu_device_tensor_op       op,
                                           const npu_device_ne_type & ne);

}  // namespace hexagon
