
#include "graph.hpp"

#include "op_registry.hpp"
#include "util.hpp"
#include "vtcm_mem.hpp"

#include <new>

namespace hexagon {

graph::graph() noexcept {
    DEVICE_LOG_DEBUG("graph(%p) created\n", (void *) this);
}

graph::~graph() noexcept {
    _tensors.reset();
    DEVICE_LOG_DEBUG("graph(%p) destroyed\n", (void *) this);
}

void graph::set_tensor(const npu_device_tensor_handle_t * tensors, int tensor_count) {
    if (tensor_count <= 0 || !tensors) {
        _tensors.reset();
        _tensor_count = 0;
        DEVICE_LOG_DEBUG("graph(%p) set_tensor: no tensors to set\n", (void *) this);
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
        thread_pool->sync_execute(&graph::thread_pool_task, this);
    } else {
        default_thread_pool::thread_params param = {
            0, 1, nullptr, hexagon::vtcm_mem::get_avail_block_size()
        };  // TODO: should have a better way to initialize thread_params

        compute_impl(nullptr, &param);
    }

    _tensors[_tensor_count - 1]->invalidate();
    _f16_to_f32_table = nullptr;
    return true;
}

void graph::thread_pool_task(default_thread_pool *                pool,
                             default_thread_pool::thread_params * thread_params,
                             void *                               graph) {
    reinterpret_cast<hexagon::graph *>(graph)->compute_impl(pool, thread_params);
}

void graph::compute_impl(default_thread_pool * pool, default_thread_pool::thread_params * thread_params) {
    hexagon::compute_params params = { thread_params, _f16_to_f32_table };

    npu_device_tensor_op prev_op = NPU_OP_COUNT;
    npu_device_ne_type   prev_ne = {};

    for (size_t i = 0; i < _tensor_count; ++i) {
        auto *       dst     = _tensors[i];
        auto         op      = dst->get_op();
        const auto & ne      = dst->get_info().ne;
        const auto * op_name = op_get_name(op);
        auto *       func    = get_compute_func(dst);
        if (!func) {
            DEVICE_LOG_ERROR("[%p][%s]graph tensor[%zu] op not supported\n", (void *) this, op_name, i);
            return;
        }

        const bool should_sync = requires_thread_barrier(prev_op, prev_ne, op, ne);
        if (pool && should_sync) {
            // For the last tensor, the thread pool will handle synchronization
            DEVICE_SCOPED_PERFORMANCE_TRACKER("[%p][%s]sync_thread, tidx: %zu, tensor[%zu/%zu]", (void *) this, op_name,
                                              params.get_thread_index(), i + 1, _tensor_count);
            pool->sync_thread();
        }

        prev_op = op;
        memcpy(&prev_ne, &ne, sizeof(prev_ne));

        if (!func(dst, &params)) {
            DEVICE_LOG_ERROR("[%p][%s]graph tensor[%zu] op %d compute failed\n", (void *) this, op_name, i, op);
        }
    }
}

}  // namespace hexagon
