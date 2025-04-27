
#include "graph.hpp"

#include <new>

#include "op_impl.hpp"
#include "util.hpp"

namespace hexagon {

graph::graph() noexcept {
    DEVICE_LOG_DEBUG("graph(%p) created\n", (void *) this);
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
        DEVICE_LOG_DEBUG("graph(%p) set_tensor[%d]: %p(%p,%p), op: %d\n", (void *) this, i, (void *) tensor_obj,
                         (void *) tensor_obj->get_src(0), (void *) tensor_obj->get_src(1), tensor_obj->get_op());
    }

    _tensor_count = tensor_count;
    DEVICE_LOG_DEBUG("graph(%p) tensor count: %zu\n", (void *) this, _tensor_count);
}

bool graph::compute(default_thread_pool * thread_pool) {
    if (!_tensors || !_tensor_count) {
        DEVICE_LOG_DEBUG("graph(%p) no tensors to compute\n", (void *) this);
        return true;  // return success if no tensors to compute
    }

    DEVICE_LOG_DEBUG("graph(%p) compute\n", (void *) this);
    thread_pool->sync_execute(reinterpret_cast<default_thread_pool::task_type>(&graph::thread_pool_task), this);

    for (size_t i = 0; i < _tensor_count; ++i) {
        auto * dst = _tensors[i];
        dst->flush();  // TODO: optimize this
    }

    return true;
}

void graph::thread_pool_task(default_thread_pool * pool, size_t thread_idx, size_t thread_count, graph * graph) {
    NPU_UNUSED(pool);
    graph->compute_impl(thread_idx, thread_count);
}

void graph::compute_impl(size_t thread_idx, size_t thread_count) {
    for (size_t i = 0; i < _tensor_count; ++i) {
        auto * dst  = _tensors[i];
        auto   op   = dst->get_op();
        auto * func = get_compute_func(op);
        if (!func) {
            DEVICE_LOG_ERROR("graph(%p) tensor[%zu] op %d not supported\n", (void *) this, i, op);
            return;
        }

        if (!func(dst, thread_idx, thread_count)) {
            DEVICE_LOG_ERROR("graph(%p) tensor[%zu] op %d compute failed\n", (void *) this, i, op);
            return;
        }
    }
}

}  // namespace hexagon
