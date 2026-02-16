
#include "graph.hpp"
#include "hexagon_npu.h"
#include "op_registry.hpp"
#include "remote.h"
#include "tensor.hpp"
#include "thread_pool.hpp"
#include "type_traits.hpp"
#include "util.hpp"

#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <hexagon_types.h>

#include <memory>

namespace {

struct npu_device_context {
    std::unique_ptr<hexagon::power_utils>         power_utils;       // Power management utilities
    std::unique_ptr<hexagon::default_thread_pool> thread_pool;
    std::unique_ptr<float[]>                      f16_to_f32_table;  // TODO: store vtcm?

    bool init() {
        if (!init_ltu()) {
            DEVICE_LOG_ERROR("Failed to initialize LTU\n");
            return false;
        }

        if (!init_thread_pool()) {
            DEVICE_LOG_ERROR("Failed to initialize thread pool\n");
            return false;
        }

        power_utils = std::make_unique<hexagon::power_utils>();
        if (power_utils && power_utils->is_valid()) {
            power_utils->set_dvcs_performance_mode(true);
            DEVICE_LOG_DEBUG("Power utilities initialized with DVCS performance mode enabled\n");
        } else {
            DEVICE_LOG_ERROR("Failed to initialize power utilities\n");
        }

        DEVICE_LOG_DEBUG("NPU device context initialized\n");
        return true;
    }

  private:
    bool init_ltu() {
        constexpr const size_t kLtuCount = 1U << 16;

        f16_to_f32_table = std::make_unique<float[]>(kLtuCount);
        if (!f16_to_f32_table) {
            DEVICE_LOG_ERROR("Failed to allocate memory for f16_to_f32 table\n");
            return false;
        }

        hexagon::init_f16_f32_table(f16_to_f32_table.get(), kLtuCount);
        DEVICE_LOG_DEBUG("f16_to_f32 table initialized\n");
        return true;
    }

    bool init_thread_pool() {
        if (thread_pool) {
            DEVICE_LOG_DEBUG("Thread pool already initialized\n");
            return true;
        }

        auto pool = std::make_unique<hexagon::default_thread_pool>();
        if (!pool) {
            DEVICE_LOG_ERROR("Failed to create thread pool\n");
            return false;
        }

        thread_pool = std::move(pool);
        DEVICE_LOG_DEBUG("Thread pool initialized\n");
        return true;
    }
};

inline hexagon::tensor * tensor_from_handle(npu_device_tensor_handle_t h) {
    if (h == npu_device_INVALID_DEVICE_TENSOR_HANDLE) {
        return nullptr;
    }

    return reinterpret_cast<hexagon::tensor *>(h);
}

inline npu_device_tensor_handle_t tensor_to_handle(hexagon::tensor * tensor) {
    return reinterpret_cast<npu_device_tensor_handle_t>(tensor);
}

inline hexagon::graph * graph_from_handle(npu_device_graph_handle_t h) {
    if (h == npu_device_INVALID_DEVICE_GRAPH_HANDLE) {
        return nullptr;
    }

    return reinterpret_cast<hexagon::graph *>(h);
}

inline npu_device_graph_handle_t graph_to_handle(hexagon::graph * graph) {
    return reinterpret_cast<npu_device_graph_handle_t>(graph);
}

inline npu_device_context * device_context_from_handle(remote_handle64 h) {
    return reinterpret_cast<npu_device_context *>(h);
}

}  // namespace

int npu_device_open(const char * uri, remote_handle64 * h) {
    // TODO: should we have a device context here?
    auto * context = new npu_device_context();
    if (!context->init()) {
        DEVICE_LOG_ERROR("Failed to initialize npu_device_context\n");
        delete context;
        return AEE_EFAILED;
    }

    *h = reinterpret_cast<remote_handle64>(context);
    DEVICE_LOG_INFO("NPU device context created: %p\n", (void *) *h);
    return AEE_SUCCESS;
}

int npu_device_close(remote_handle64 h) {
    auto * context = device_context_from_handle(h);
    if (!context) {
        DEVICE_LOG_ERROR("Invalid npu_device_context handle\n");
        return AEE_EINVHANDLE;
    }

    delete context;
    DEVICE_LOG_INFO("NPU device context destroyed: %p\n", (void *) h);
    return AEE_SUCCESS;
}

AEEResult npu_device_device_get_alignment(remote_handle64 _h, uint32_t * alignment) {
    NPU_UNUSED(_h);
    *alignment = sizeof(HVX_VectorPair);
    return AEE_SUCCESS;
}

AEEResult npu_device_device_support_op(remote_handle64                   _h,
                                       const npu_device_tensor_op_spec * op_spec,
                                       const npu_device_tensor_spec *    dst,
                                       const npu_device_tensor_spec *    srcs,
                                       int                               srcsLen,
                                       boolean *                         is_supported) {
    NPU_UNUSED(_h);

    if (!srcs || srcsLen <= 0 || !dst || !is_supported) {
        DEVICE_LOG_ERROR("npu_device_device_support_op: Invalid arguments\n");
        return AEE_EINVARGS;
    }

    *is_supported = hexagon::support_op(op_spec, dst, srcs, srcsLen);
    return AEE_SUCCESS;
}

AEEResult npu_device_tensor_init(remote_handle64                  _h,
                                 const npu_device_tensor_config * info,
                                 npu_device_tensor_handle_t *     tensor_handle) {
    NPU_UNUSED(_h);
    auto * tensor  = new hexagon::tensor(*info);
    *tensor_handle = tensor_to_handle(tensor);
    return AEE_SUCCESS;
}

AEEResult npu_device_tensor_update_params(remote_handle64                         _h,
                                          npu_device_tensor_handle_t              tensor_handle,
                                          const npu_device_tensor_update_config * config) {
    NPU_UNUSED(_h);
    auto * tensor = tensor_from_handle(tensor_handle);
    if (!tensor || !config) {
        return AEE_EINVHANDLE;
    }

    tensor->update_config(*config);
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

AEEResult npu_device_tensors_free(remote_handle64                    _h,
                                  const npu_device_tensor_handle_t * tensor_handles,
                                  int                                tensor_handlesLen) {
    NPU_UNUSED(_h);
    if (!tensor_handles || tensor_handlesLen < 0) {
        DEVICE_LOG_ERROR("npu_device_tensors_free: Invalid arguments\n");
        return AEE_EINVARGS;
    }

    for (int i = 0; i < tensor_handlesLen; ++i) {
        auto * tensor = tensor_from_handle(tensor_handles[i]);
        if (tensor) {
            delete tensor;
        } else {
            DEVICE_LOG_ERROR("npu_device_tensors_free: Invalid tensor handle at index %d\n", i);
        }
    }

    return AEE_SUCCESS;
}

AEEResult npu_device_graph_init(remote_handle64 _h, npu_device_graph_handle_t * graph_handle) {
    NPU_UNUSED(_h);
    auto * graph  = new hexagon::graph();
    *graph_handle = graph_to_handle(graph);
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_set_tensor(remote_handle64                    _h,
                                      npu_device_graph_handle_t          graph_handle,
                                      const npu_device_tensor_handle_t * tensor_handles,
                                      int                                tensor_handlesLen) {
    NPU_UNUSED(_h);
    auto * graph = graph_from_handle(graph_handle);
    if (!graph || !tensor_handles || tensor_handlesLen <= 0) {
        return AEE_EINVHANDLE;
    }

    graph->set_tensor(tensor_handles, tensor_handlesLen);
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_set_tensor_with_param(remote_handle64                         _h,
                                                 npu_device_graph_handle_t               graph_handle,
                                                 const npu_device_tensor_handle_t *      tensor_handles,
                                                 int                                     tensor_handlesLen,
                                                 const npu_device_tensor_update_config * tensor_params,
                                                 int                                     tensor_paramsLen) {
    NPU_UNUSED(_h);
    auto * graph = graph_from_handle(graph_handle);
    if (!graph || tensor_handlesLen != tensor_paramsLen || tensor_handlesLen < 0) {
        return AEE_EINVHANDLE;
    }

    if (tensor_params && tensor_handles) {
        for (int i = 0; i < tensor_handlesLen; ++i) {
            auto * tensor = tensor_from_handle(tensor_handles[i]);
            if (tensor) {
                tensor->update_config(tensor_params[i]);
            }
        }
    }

    graph->set_tensor(tensor_handles, tensor_handlesLen);
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_compute(remote_handle64 _h, npu_device_graph_handle_t graph_handle) {
    auto dev_ctx = device_context_from_handle(_h);
    if (!dev_ctx) {
        DEVICE_LOG_DEBUG("Invalid npu_device_context handle\n");
        return AEE_EINVHANDLE;
    }

    auto * graph = graph_from_handle(graph_handle);
    if (!graph) {
        DEVICE_LOG_ERROR("Invalid graph handle\n");
        return AEE_EINVHANDLE;
    }

    if (!graph->compute(dev_ctx->thread_pool.get(), dev_ctx->f16_to_f32_table.get())) {
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
