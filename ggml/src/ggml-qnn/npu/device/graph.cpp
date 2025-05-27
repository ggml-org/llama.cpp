
#include "graph.hpp"

#include <new>

#include "op_impl.hpp"
#include "util.hpp"
#include "vtcm_mem.hpp"

namespace hexagon {

graph::graph() noexcept {
    _vtcm_quota_size = hexagon::vtcm_mem::get_avail_block_size();  // TODO: move to device init?
    DEVICE_LOG_DEBUG("graph(%p) created: vtcm quota size: %zu\n", (void *) this, _vtcm_quota_size);
}

graph::~graph() noexcept {
    _tensors.reset();
    DEVICE_LOG_DEBUG("graph(%p) destroyed\n", (void *) this);
}

void graph::set_tensor(const npu_device_tensor_handle_t * tensors, int tensor_count) {
    if (tensor_count <= 0) {
        _tensors.reset();
        _tensor_count = 0;
        return;
    }

    _tensors = std::make_unique<tensor *[]>(size_t(tensor_count));
    for (int i = 0; i < tensor_count; ++i) {
        auto * tensor_obj = reinterpret_cast<tensor *>(tensors[i]);
        _tensors[i]       = tensor_obj;
        DEVICE_LOG_DEBUG("graph(%p) set_tensor[%d]: %p(%p,%p), op: %s\n", (void *) this, i, (void *) tensor_obj,
                         (void *) tensor_obj->get_src(0), (void *) tensor_obj->get_src(1),
                         op_get_name(tensor_obj->get_op()));
    }

    _tensor_count = tensor_count;
    DEVICE_LOG_DEBUG("graph(%p) tensor count: %zu\n", (void *) this, _tensor_count);
}

bool graph::compute(default_thread_pool * thread_pool, const float * f16_to_f32_table) {
    if (_tensors == nullptr || !_tensor_count) {
        DEVICE_LOG_DEBUG("graph(%p) no tensors to compute\n", (void *) this);
        return true;  // return success if no tensors to compute
    }

    DEVICE_LOG_DEBUG("graph(%p) compute\n", (void *) this);

    DEVICE_SCOPED_PERFORMANCE_TRACKER("[%p]compute", (void *) this);
    _f16_to_f32_table = f16_to_f32_table;
    if (thread_pool) {
        thread_pool->sync_execute(reinterpret_cast<default_thread_pool::task_type>(&graph::thread_pool_task), this);
    } else {
        compute_impl(nullptr, 0, 1);
    }

    _f16_to_f32_table = nullptr;
    return true;
}

void graph::thread_pool_task(default_thread_pool * pool, size_t thread_idx, size_t thread_count, graph * graph) {
    graph->compute_impl(pool, thread_idx, thread_count);
}

void graph::compute_impl(default_thread_pool * pool, size_t thread_idx, size_t thread_count) {
    hexagon::compute_params params = { thread_idx, thread_count, _vtcm_quota_size / thread_count, _f16_to_f32_table };

    for (size_t i = 0; i < _tensor_count; ++i) {
        auto * dst  = _tensors[i];
        auto   op   = dst->get_op();
        auto * func = get_compute_func(dst);
        if (func == nullptr) {
            DEVICE_LOG_ERROR("graph(%p) tensor[%zu] op %d not supported\n", (void *) this, i, op);
            return;
        }
        if (!func(dst, &params)) {
            DEVICE_LOG_ERROR("graph(%p) tensor[%zu] op %d compute failed\n", (void *) this, i, op);
        }

        DEVICE_SCOPED_PERFORMANCE_TRACKER("[%p]sync_thread, tidx: %zu", (void *) this, thread_idx);

        const bool should_sync = requires_thread_barrier(op);
        if (pool && should_sync && i < _tensor_count - 1) {
            pool->sync_thread();
        }
        dst->invalidate();
    }
}

}  // namespace hexagon
