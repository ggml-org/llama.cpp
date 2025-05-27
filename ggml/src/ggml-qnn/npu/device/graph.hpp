#pragma once

#include <memory>

#include "hexagon_npu.h"
#include "tensor.hpp"
#include "thread_pool.hpp"

namespace hexagon {

class graph {
  public:
    // TODO: add execute direction here
    explicit graph() noexcept;

    ~graph() noexcept;

    void set_tensor(const npu_device_tensor_handle_t * tensors, int tensor_count);

    bool compute(default_thread_pool * thread_pool, const float * f16_to_f32_table);

  private:
    static void thread_pool_task(default_thread_pool * pool, size_t thread_idx, size_t thread_count, graph * graph);
    void        compute_impl(default_thread_pool * pool, size_t thread_idx, size_t thread_count);

    std::unique_ptr<tensor *[]> _tensors;
    size_t                      _tensor_count     = 0;
    size_t                      _vtcm_quota_size  = 0;
    const float *               _f16_to_f32_table = nullptr;

    DISABLE_COPY_AND_MOVE(graph);
};

}  // namespace hexagon
