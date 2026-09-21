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

    // Alternate between two graph UIDs until both are promoted to the cache,
    // then reuse the first one. This exercises multi-graph RPC caching rather
    // than only the most recently computed graph.
    ggml_init_params cache_params = {
        /* .mem_size   = */ 2*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * cache_ctx = ggml_init(cache_params);
    GGML_ASSERT(cache_ctx != nullptr);

    ggml_tensor * cache_tensor_a = ggml_new_tensor_1d(cache_ctx, GGML_TYPE_F32, 1);
    ggml_tensor * cache_tensor_b = ggml_new_tensor_1d(cache_ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t cache_buffer = ggml_backend_alloc_ctx_tensors(cache_ctx, backend_b);
    GGML_ASSERT(cache_buffer != nullptr);

    ggml_cgraph * cache_graph_a = ggml_new_graph_custom(cache_ctx, 1, false);
    cache_graph_a->nodes[0] = cache_tensor_a;
    cache_graph_a->n_nodes = 1;
    ggml_cgraph * cache_graph_b = ggml_new_graph_custom(cache_ctx, 1, false);
    cache_graph_b->nodes[0] = cache_tensor_b;
    cache_graph_b->n_nodes = 1;
    const uint64_t reused_uid = cache_graph_a->uid;

    GGML_ASSERT(ggml_backend_graph_compute(backend_b, cache_graph_a) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_graph_compute(backend_b, cache_graph_b) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_graph_compute(backend_b, cache_graph_a) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_graph_compute(backend_b, cache_graph_b) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_graph_compute(backend_b, cache_graph_a) == GGML_STATUS_SUCCESS);
    ggml_backend_rpc_get_device_memory(endpoint_b, 0, &free_mem, &total_mem);

    // Freeing a backing buffer must invalidate cached server graphs. Reusing the
    // UID with a new buffer must therefore send and store a complete graph again.
    ggml_backend_buffer_free(cache_buffer);
    ggml_free(cache_ctx);

    ggml_init_params replacement_params = {
        /* .mem_size   = */ ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * replacement_ctx = ggml_init(replacement_params);
    GGML_ASSERT(replacement_ctx != nullptr);
    ggml_tensor * replacement_tensor = ggml_new_tensor_1d(replacement_ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t replacement_buffer = ggml_backend_alloc_ctx_tensors(replacement_ctx, backend_b);
    GGML_ASSERT(replacement_buffer != nullptr);
    ggml_cgraph * replacement_graph = ggml_new_graph_custom(replacement_ctx, 1, false);
    replacement_graph->nodes[0] = replacement_tensor;
    replacement_graph->n_nodes = 1;
    replacement_graph->uid = reused_uid;
    GGML_ASSERT(ggml_backend_graph_compute(backend_b, replacement_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_rpc_get_device_memory(endpoint_b, 0, &free_mem, &total_mem);

    ggml_backend_buffer_free(replacement_buffer);
    ggml_free(replacement_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
