#include "backend-dispatched.h"
#include "backend-internal.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "shared/apir_backend.h"

#include <cstdint>

struct timer_data graph_compute_timer = { 0, 0, 0, "compute_timer" };

uint32_t backend_backend_graph_compute(struct apir_encoder *       enc,
                                       struct apir_decoder *       dec,
                                       struct virgl_apir_context * ctx) {
    UNUSED(ctx);
    UNUSED(enc);

    static bool async_backend_initialized = false;
    static bool async_backend;

    if (!async_backend_initialized) {
        struct ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = true;
    }

    start_timer(&graph_compute_timer);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    const void * shmem_data = ctx->iface.get_shmem_ptr(ctx->virgl_ctx, shmem_res_id);
    if (!shmem_data) {
        FATAL("Couldn't get the shmem addr from virgl :/");
    }
    size_t cgraph_size;
    apir_decode_size_t(dec, &cgraph_size);

    struct apir_decoder secondary_dec = apir_new_decoder((const char *) shmem_data, cgraph_size);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_size);

    ggml_status status;
#if APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    for (int idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_node(cgraph, idx);
        if (dev->iface.supports_op(dev, op)) {
            continue;
        }
        ERROR("Graph node %d (%s) not supported by the backend :/", idx, ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        stop_timer(&graph_compute_timer);
        return 0;
    }
#endif
    status = bck->iface.graph_compute(bck, cgraph);

    if (async_backend) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    stop_timer(&graph_compute_timer);

    return 0;
}
