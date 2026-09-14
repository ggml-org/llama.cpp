#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 3);
    ggml_backend_load_all();

    const char * endpoint_a = argv[1];
    const char * endpoint_b = argv[2];

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_a);
    GGML_ASSERT(buffer != nullptr);

    // A remote pointer allocated by server A is not meaningful to server B.
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1, false);
    graph->nodes[0] = tensor;
    graph->n_nodes = 1;

    GGML_ASSERT(ggml_backend_graph_compute(backend_b, graph) == GGML_STATUS_SUCCESS);
    // Wait for server B to finish the graph before the script checks its log.
    size_t free_mem;
    size_t total_mem;
    ggml_backend_rpc_get_device_memory(endpoint_b, 0, &free_mem, &total_mem);
    GGML_ASSERT(total_mem > 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
