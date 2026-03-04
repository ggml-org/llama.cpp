#include "ggml-cxl-impl.h"
#include "cxl-graph-serialize.h"

#include <cstring>
#include <vector>

static const char * ggml_backend_cxl_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_CXL_NAME;
}

static void ggml_backend_cxl_free(ggml_backend_t backend) {
    delete backend;
}

static ggml_status ggml_backend_cxl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cxl_device_context * ctx = CXL_DEV_CTX(backend->device);

    // Serialize the compute graph
    std::vector<uint8_t> graph_data;
    if (!cxl_graph_serialize(cgraph, graph_data)) {
        GGML_LOG_ERROR(GGML_CXL_LOG "%s: failed to serialize graph\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // Send to device for execution
    std::lock_guard<std::mutex> lock(ctx->cmd_mutex);
    int status = cxl_device_graph_compute(&ctx->cxl_dev, graph_data.data(), graph_data.size());

    return (ggml_status)status;
}

static ggml_backend_i ggml_backend_cxl_interface = {
    /* .get_name                = */ ggml_backend_cxl_get_name,
    /* .free                    = */ ggml_backend_cxl_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cxl_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cxl_guid() {
    static ggml_guid guid = {
        0xc7, 0x78, 0x4c, 0x32, 0x14, 0x03, 0x86, 0x02,
        0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0xc1
    };
    return &guid;
}

ggml_backend_t ggml_backend_cxl_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    ggml_backend_t cxl_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cxl_guid(),
        /* .interface = */ ggml_backend_cxl_interface,
        /* .device    = */ dev,
        /* .context   = */ dev->context,
    };

    return cxl_backend;
}
