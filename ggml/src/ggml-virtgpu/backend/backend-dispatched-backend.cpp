#include "backend-dispatched.h"
#include "backend-virgl-apir.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "shared/apir_backend.h"

#include <cstdint>

// SECURITY: Essential validation for computation graph parameters
static uint32_t validate_graph_operation(size_t cgraph_size, const char* operation) {
    if (cgraph_size == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        return 1;
    }

    if (cgraph_size < sizeof(ggml_cgraph)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph too small: %zu bytes (min: %zu)\n",
                      operation, cgraph_size, sizeof(ggml_cgraph));
        return 1;
    }

    return 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    static bool async_backend_initialized = false;
    static bool async_backend;

    if (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = true;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    const void * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    if (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        return 1;
    }
    size_t cgraph_size;
    apir_decode_size_t(dec, &cgraph_size);

    // SECURITY: Validate graph size before processing
    if (validate_graph_operation(cgraph_size, __func__) != 0) {
        apir_decoder_set_fatal(dec);
        return 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((const char *) shmem_data, cgraph_size);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_size);

    // SECURITY: Validate graph deserialization succeeded
    if (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        return 1;
    }

    // SECURITY: Basic graph validation - no arbitrary limits
    if (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n",
                      __func__, cgraph->n_nodes, cgraph->n_leafs);
        return 1;
    }

    ggml_status status;
#if APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    for (int idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_node(cgraph, idx);
        if (dev->iface.supports_op(dev, op)) {
            continue;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", idx, ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        return 0;
    }
#endif
    status = bck->iface.graph_compute(bck, cgraph);

    if (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    return 0;
}
