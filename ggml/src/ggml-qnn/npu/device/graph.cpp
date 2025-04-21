
#include "graph.hpp"

#include <new>

#include "op_impl.hpp"
#include "util.hpp"

namespace hexagon {

graph::~graph() noexcept {
    if (_tensors) {
        delete[] _tensors;
    }
}

void graph::set_tensor(const npu_device_tensor_handle_t * tensors, int tensor_count) {
    if (_tensor_count > 0) {
        delete[] _tensors;
    }

    if (tensor_count <= 0) {
        _tensors      = nullptr;
        _tensor_count = 0;
        return;
    }

    _tensors = new (std::nothrow) tensor *[tensor_count];
    for (int i = 0; i < tensor_count; ++i) {
        auto * tensor_obj = reinterpret_cast<tensor *>(tensors[i]);
        _tensors[i]       = tensor_obj;
        DEVICE_LOG_DEBUG("graph(%p) set_tensor[%d]: %p(%p,%p), op: %d\n", (void *) this, i, (void *) tensor_obj,
                         (void *) tensor_obj->get_src(0), (void *) tensor_obj->get_src(1), tensor_obj->get_op());
    }

    _tensor_count = tensor_count;
    DEVICE_LOG_DEBUG("graph(%p) tensor count: %zu\n", (void *) this, _tensor_count);
}

bool graph::compute() {
    if (!_tensors || !_tensor_count) {
        DEVICE_LOG_DEBUG("graph(%p) no tensors to compute\n", (void *) this);
        return true;  // return success if no tensors to compute
    }

    DEVICE_LOG_DEBUG("graph(%p) compute\n", (void *) this);
    for (size_t i = 0; i < _tensor_count; ++i) {
        auto * dst  = _tensors[i];
        auto   op   = dst->get_op();
        auto * func = get_compute_func(op);
        if (!func) {
            DEVICE_LOG_ERROR("graph(%p) tensor[%zu] op %d not supported\n", (void *) this, i, op);
            return false;
        }

        if (!func(dst)) {
            DEVICE_LOG_ERROR("graph(%p) tensor[%zu] op %d compute failed\n", (void *) this, i, op);
            return false;
        }

        dst->flush();  // TODO: optimize this
    }

    return true;
}

}  // namespace hexagon
