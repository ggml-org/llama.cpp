#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>

// same as fnv_hash in ggml-rpc.cpp
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 4);
    ggml_backend_load_all();

    const char * endpoint_a = argv[1];
    const char * endpoint_b = argv[2];
    const char * cache_dir  = argv[3];

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 4*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
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

    // An empty cache file (e.g. left by a failed write) must be a cache miss, not a hit.
    // ref: https://github.com/ggml-org/llama.cpp/issues/30633
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*1024*1024);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, weight));
    GGML_ASSERT(weight_buffer != nullptr);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    GGML_ASSERT(ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer)) == GGML_STATUS_SUCCESS);

    std::vector<float> weight_data(ggml_nelements(weight));
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = (float) i;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, fnv_hash((const uint8_t *) weight_data.data(), ggml_nbytes(weight)));
    FILE * f = fopen((std::string(cache_dir) + "/" + hash_str).c_str(), "wb");
    GGML_ASSERT(f != nullptr);
    fclose(f);

    ggml_backend_tensor_set(weight, weight_data.data(), 0, ggml_nbytes(weight));
    std::vector<float> weight_read(weight_data.size());
    ggml_backend_tensor_get(weight, weight_read.data(), 0, ggml_nbytes(weight));
    GGML_ASSERT(weight_read == weight_data);
    ggml_backend_buffer_free(weight_buffer);

    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
