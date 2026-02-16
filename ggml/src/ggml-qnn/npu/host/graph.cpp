#include "graph.hpp"

#include "profiler.hpp"
#include "tensor.hpp"

namespace hexagon {

host_graph::host_graph(ggml_cgraph * cgraph, remote_handle64 device_handle) : _device_handle(device_handle) {
    auto status = npu_device_graph_init(_device_handle, &_graph_handle);
    if (status != AEE_SUCCESS) {
        LOG_ERROR("Failed to init graph: %d", (int) status);
        _graph_handle = 0;
        return;
    }

    update(cgraph);
}

host_graph::~host_graph() {
    if (_graph_handle) {
        npu_device_graph_free(_device_handle, _graph_handle);
        _graph_handle = 0;
    }
}

bool host_graph::update(ggml_cgraph * cgraph) {
    if (!_graph_handle) {
        LOG_ERROR("host_graph not initialized\n");
        return false;
    }

    PROFILER_LOG_DEBUG("[%p]host_graph::update started\n", (void *) this);
    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]update, handle(%p)", (void *) this, (void *) _graph_handle);

    _tensor_handles.clear();
    _tensor_update_configs.clear();
    _tensor_handles.reserve(cgraph->n_nodes);
    _tensor_update_configs.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * node = cgraph->nodes[i];
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_RESHAPE) {
            // skip view liked ops
            LOG_DEBUG("node[%d]%s(%s), addr: %p, type: %s, dims: %ldx%ldx%ldx%ld, skipped\n", i, ggml_get_name(node),
                      ggml_op_desc(node), (void *) node, ggml_type_name(node->type), (long) node->ne[0],
                      (long) node->ne[1], (long) node->ne[2], (long) node->ne[3]);
            continue;
        }

        // TODO: move to tensor?
        auto * tensor_obj = host_tensor::from_ggml_tensor(node);
        if (!tensor_obj) {
            LOG_DEBUG("Unable to get host tensor from ggml tensor: %p\n", (void *) node);
            continue;
        }

        _tensor_handles.push_back(tensor_obj->get_device_tensor_handle());
        _tensor_update_configs.push_back(tensor_obj->update_hosts_params_only(node));

        PROFILER_LOG_DEBUG("node[%d]%s(%s), addr(%p), %ldx%ldx%ldx%ld%s, handle(%p)\n", i, ggml_get_name(node),
                           ggml_op_desc(node), (void *) tensor_obj, (long) tensor_obj->get_ne(0),
                           (long) tensor_obj->get_ne(1), (long) tensor_obj->get_ne(2), (long) tensor_obj->get_ne(3),
                           ggml_type_name(node->type), (void *) tensor_obj->get_device_tensor_handle());
    }

    GGML_ASSERT(_tensor_handles.size() == _tensor_update_configs.size());

    constexpr const npu_device_tensor_handle_t      kEmptyTensorHandle = 0;
    constexpr const npu_device_tensor_update_config kEmptyUpdateConfig = {};

    auto ret = npu_device_graph_set_tensor_with_param(
        _device_handle, _graph_handle, _tensor_handles.size() ? _tensor_handles.data() : &kEmptyTensorHandle,
        (int) _tensor_handles.size(),
        _tensor_update_configs.size() ? _tensor_update_configs.data() : &kEmptyUpdateConfig,
        (int) _tensor_update_configs.size());

    if (ret != AEE_SUCCESS) {
        LOG_ERROR("[%p]failed to set tensors in host_graph: 0x%x\n", (void *) this, (int) ret);
        return false;
    }

    LOG_DEBUG("[%p]host_graph::update, handle(%p), ggml_cgraph(%p), tensor count(%zu)\n", (void *) this,
              (void *) _graph_handle, (void *) cgraph, _tensor_handles.size());
    return true;
}

bool host_graph::compute() {
    if (!_graph_handle) {
        LOG_ERROR("host_graph not initialized\n");
        return false;
    }

    LOG_DEBUG("[%p]host_graph::compute started\n", (void *) this);
    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]compute, handle(%p)", (void *) this, (void *) _graph_handle);
    auto status = npu_device_graph_compute(_device_handle, _graph_handle);
    if (status != AEE_SUCCESS) {
        LOG_ERROR("Failed to compute host_graph: 0x%x\n", (int) status);
        LOG_DEBUG("[%p]host_graph::compute finished with failure\n", (void *) this);
        return false;
    }

    LOG_DEBUG("[%p]host_graph::compute finished\n", (void *) this);
    return true;
}

}  // namespace hexagon
