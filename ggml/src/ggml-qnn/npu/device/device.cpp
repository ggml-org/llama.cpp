
#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <hexagon_types.h>

#include <new>

#include "graph.hpp"
#include "hexagon_npu.h"
#include "op_impl.hpp"
#include "remote.h"
#include "tensor.hpp"
#include "util.hpp"

#define NPU_UNUSED(x) (void) (x)

namespace {

struct npu_device_context {
    int unused = 0;
    // TODO: should we add tensor context here?
};

inline hexagon::tensor * tensor_from_handle(npu_device_graph_handle_t h) {
    return reinterpret_cast<hexagon::tensor *>(h);
}

inline npu_device_graph_handle_t tensor_to_handle(hexagon::tensor * tensor) {
    return reinterpret_cast<npu_device_graph_handle_t>(tensor);
}

inline hexagon::graph * graph_from_handle(npu_device_tensor_handle_t h) {
    return reinterpret_cast<hexagon::graph *>(h);
}

inline npu_device_tensor_handle_t graph_to_handle(hexagon::graph * graph) {
    return reinterpret_cast<npu_device_tensor_handle_t>(graph);
}

}  // namespace

int npu_device_open(const char * uri, remote_handle64 * h) {
    // TODO: should we have a device context here?
    auto * context = new (std::nothrow) npu_device_context();
    if (!context) {
        DEVICE_LOG_ERROR("Failed to allocate memory for the npu_device_context");
        return AEE_ENOMEMORY;
    }

    *h = reinterpret_cast<remote_handle64>(context);
    return AEE_SUCCESS;
}

int npu_device_close(remote_handle64 h) {
    auto * context = reinterpret_cast<npu_device_context *>(h);
    if (!context) {
        DEVICE_LOG_ERROR("Invalid npu_device_context handle");
        return AEE_EINVHANDLE;
    }

    delete context;
    return AEE_SUCCESS;
}

AEEResult npu_device_device_get_alignment(remote_handle64 _h, uint32_t * alignment) {
    NPU_UNUSED(_h);
    *alignment = sizeof(HVX_Vector);
    return AEE_SUCCESS;
}

AEEResult npu_device_device_support_op(remote_handle64 _h, const npu_device_tensor_spec * src0,
                                       const npu_device_tensor_spec * src1, const npu_device_tensor_spec * dst,
                                       npu_device_tensor_op op, boolean * is_supported) {
    NPU_UNUSED(_h);
    *is_supported = hexagon::support_op(*src0, *src1, *dst, op);
    return AEE_SUCCESS;
}

AEEResult npu_device_tensor_init(remote_handle64 _h, const npu_device_tensor_config * info,
                                 npu_device_tensor_handle_t * tensor_handle) {
    NPU_UNUSED(_h);
    auto * tensor = new (std::nothrow) hexagon::tensor(*info);
    if (!tensor) {
        DEVICE_LOG_ERROR("Failed to allocate memory for the tensor");
        return AEE_ENOMEMORY;
    }

    *tensor_handle = tensor_to_handle(tensor);
    return AEE_SUCCESS;
}

AEEResult npu_device_tensor_set_src(remote_handle64 _h, npu_device_tensor_handle_t tensor_handle, uint64_t index,
                                    npu_device_tensor_handle_t src) {
    NPU_UNUSED(_h);
    auto * tensor = tensor_from_handle(tensor_handle);
    if (!tensor) {
        return AEE_EINVHANDLE;
    }

    auto * src_tensor = tensor_from_handle(src);
    tensor->set_src(index, src_tensor);
    return AEE_SUCCESS;
}

AEEResult npu_device_tensor_set_op(remote_handle64 _h, npu_device_tensor_handle_t tensor_handle,
                                   npu_device_tensor_op op) {
    NPU_UNUSED(_h);
    auto * tensor = tensor_from_handle(tensor_handle);
    if (!tensor) {
        return AEE_EINVHANDLE;
    }

    tensor->set_op(op);
    return AEE_SUCCESS;
}

AEEResult npu_device_tensor_free(remote_handle64 _h, npu_device_tensor_handle_t tensor_handle) {
    NPU_UNUSED(_h);
    auto * tensor = tensor_from_handle(tensor_handle);
    if (!tensor) {
        return AEE_EINVHANDLE;
    }

    delete tensor;
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_init(remote_handle64 _h, npu_device_graph_handle_t * graph_handle) {
    NPU_UNUSED(_h);
    auto * graph = new (std::nothrow) hexagon::graph();
    if (!graph) {
        return AEE_ENOMEMORY;
    }

    *graph_handle = graph_to_handle(graph);
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_set_tensor(remote_handle64 _h, npu_device_graph_handle_t graph_handle,
                                      const npu_device_tensor_handle_t * tensor_handles, int tensor_handlesLen) {
    NPU_UNUSED(_h);
    auto * graph = graph_from_handle(graph_handle);
    if (!graph || !tensor_handles || tensor_handlesLen <= 0) {
        return AEE_EINVHANDLE;
    }

    graph->set_tensor(tensor_handles, tensor_handlesLen);
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_compute(remote_handle64 _h, npu_device_graph_handle_t graph_handle) {
    NPU_UNUSED(_h);
    auto * graph = graph_from_handle(graph_handle);
    if (!graph) {
        return AEE_EINVHANDLE;
    }

    if (!graph->compute()) {
        return AEE_EFAILED;
    }

    return AEE_SUCCESS;
}

AEEResult npu_device_graph_free(remote_handle64 _h, npu_device_graph_handle_t graph_handle) {
    NPU_UNUSED(_h);
    auto * graph = graph_from_handle(graph_handle);
    if (graph) {
        delete graph;
    }

    return AEE_SUCCESS;
}
