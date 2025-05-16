#include "graph.hpp"

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

    _tensor_handles.clear();
    _tensor_handles.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * node = cgraph->nodes[i];
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_RESHAPE) {
            // skip view liked ops
            LOG_DEBUG("node[%d]%s(%s), addr: %p, type: %s, skipped\n", i, ggml_get_name(node), ggml_op_desc(node),
                      (void *) node, ggml_type_name(node->type));
            continue;
        }

        auto * tensor_obj = host_tensor::from_ggml_tensor(node);
        if (!tensor_obj) {
            LOG_DEBUG("Unable to get host tensor from ggml tensor: %p\n", (void *) node);
            continue;
        }

        tensor_obj->set_op(node->op);
        _tensor_handles.push_back(tensor_obj->get_device_tensor_handle());
        LOG_DEBUG("node[%d]%s(%s), addr: %p, type: %s, tensor_handle: %p\n", i, ggml_get_name(node), ggml_op_desc(node),
                  (void *) node, ggml_type_name(node->type), (void *) tensor_obj->get_device_tensor_handle());
        for (size_t j = 0; j < GGML_MAX_SRC && node->src[j]; ++j) {
            auto * src = host_tensor::from_ggml_tensor(node->src[j]);
            tensor_obj->set_src(j, src);
        }
    }

    LOG_DEBUG("host_graph::update, host_graph(%p), handle(%p), ggml_cgraph(%p), tensor count(%zu)\n", (void *) this,
              (void *) _graph_handle, (void *) cgraph, _tensor_handles.size());
    if (!_tensor_handles.empty()) {
        npu_device_graph_set_tensor(_device_handle, _graph_handle, _tensor_handles.data(),
                                    (int) _tensor_handles.size());
    }
    return true;
}

bool host_graph::compute() {
    if (!_graph_handle) {
        LOG_ERROR("host_graph not initialized\n");
        return false;
    }

    auto status = npu_device_graph_compute(_device_handle, _graph_handle);
    if (status != AEE_SUCCESS) {
        LOG_ERROR("Failed to compute host_graph: 0x%x\n", (int) status);
        return false;
    }

    return true;
}

}  // namespace hexagon
