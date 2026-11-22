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
        /* .mem_size   = */ 10*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
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

    // Two tensors with the same ne[] but different nb[] must not share a cached alloc size.
    // ref: https://github.com/ggml-org/llama.cpp/issues/28360
    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint_a, 0);
    GGML_ASSERT(buft != nullptr);

    // MUL_MAT may need extra memory, so the size is read from the server [TAG_ALLOC_SIZE_EXPAND]
    ggml_tensor * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    packed->op = GGML_OP_MUL_MAT;

    // same ne[], twice the row stride
    ggml_tensor * strided = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    strided->op = GGML_OP_MUL_MAT;
    strided->nb[1] = 2*strided->nb[1];
    strided->nb[2] = strided->ne[1]*strided->nb[1];
    strided->nb[3] = strided->nb[2];
    GGML_ASSERT(ggml_nbytes(strided) > ggml_nbytes(packed));

    // ask for the packed tensor first, so a cache keyed without nb[] holds the smaller size
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, packed)  >= ggml_nbytes(packed));
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, strided) >= ggml_nbytes(strided));

    // A cached alloc_size for a small src must not be reused for a larger src of the
    // same op/output-shape: the server-side allocation for TOP_K scales with
    // nelements(src0), so a stale smaller cached size would cause a server-side
    // buffer overrun.
    // ref: https://github.com/ggml-org/llama.cpp/issues/30794
    ggml_tensor * src_small = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * top_k_small = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    top_k_small->op = GGML_OP_TOP_K;
    top_k_small->src[0] = src_small;

    ggml_tensor * src_large = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096*8);
    ggml_tensor * top_k_large = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    top_k_large->op = GGML_OP_TOP_K;
    top_k_large->src[0] = src_large;

    // baseline: learn the correct, uncached alloc_size for the large src. Use a
    // distinct op_params value (which the Metal/CUDA get_alloc_size() for TOP_K
    // ignores) so this hits a fresh cache entry rather than the one primed below,
    // while still exercising the real backend's src-dependent size computation.
    ggml_tensor * top_k_large_probe = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    top_k_large_probe->op = GGML_OP_TOP_K;
    top_k_large_probe->src[0] = src_large;
    top_k_large_probe->op_params[0] = 1;
    size_t size_large_uncached = ggml_backend_buft_get_alloc_size(buft, top_k_large_probe);

    // prime the cache with the small src first, then ask again with a much larger src
    // of the same op/output ne[]/nb[] - the second call must not reuse the stale,
    // too-small cached size.
    size_t size_small = ggml_backend_buft_get_alloc_size(buft, top_k_small);
    size_t size_large = ggml_backend_buft_get_alloc_size(buft, top_k_large);
    GGML_ASSERT(size_large >= size_large_uncached);
    GGML_ASSERT(size_small <= size_large);

    // a subsequent request with a src no larger than the recorded high-water mark
    // must still hit the cache (and get a size that covers it).
    size_t size_small_again = ggml_backend_buft_get_alloc_size(buft, top_k_small);
    GGML_ASSERT(size_small_again >= size_small);

    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
