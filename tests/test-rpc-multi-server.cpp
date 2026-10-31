#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cstdint>
#include <cstring>

static int test_zero_element_data(const char * endpoint, bool outside_buffer) {
    ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
    GGML_ASSERT(backend != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    void * data = tensor->data;
    // MUL_MAT makes get_alloc_size query the RPC server.
    tensor->op = GGML_OP_MUL_MAT;
    tensor->ne[3] = 0;
    if (outside_buffer) {
        tensor->data = reinterpret_cast<void *>(uintptr_t{1});
    } else {
        tensor->data = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer)) + ggml_backend_buffer_get_size(buffer));
    }

    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint, 0);
    GGML_ASSERT(buft != nullptr);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, tensor) >= ggml_nbytes(tensor));

    tensor->data = data;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}

static int test_get_rows(const char * endpoint, bool forged_stride) {
    ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
    GGML_ASSERT(backend != nullptr);

    ggml_init_params tensor_params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_a = ggml_init(tensor_params);
    ggml_context * ctx_b = ggml_init(tensor_params);
    GGML_ASSERT(ctx_a != nullptr);
    GGML_ASSERT(ctx_b != nullptr);

    ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx_a, GGML_TYPE_F32, 4, 2);
    ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx_b, GGML_TYPE_F32, 4, 2);
    ggml_backend_buffer_t buffer_a = ggml_backend_alloc_ctx_tensors(ctx_a, backend);
    ggml_backend_buffer_t buffer_b = ggml_backend_alloc_ctx_tensors(ctx_b, backend);
    GGML_ASSERT(buffer_a != nullptr);
    GGML_ASSERT(buffer_b != nullptr);

    ggml_tensor * source = tensor_a;
    ggml_tensor * target = tensor_b;
    if (reinterpret_cast<uintptr_t>(source->data) > reinterpret_cast<uintptr_t>(target->data)) {
        source = tensor_b;
        target = tensor_a;
    }
    const uintptr_t source_data = reinterpret_cast<uintptr_t>(source->data);
    const uintptr_t target_data = reinterpret_cast<uintptr_t>(target->data);
    GGML_ASSERT(target_data > source_data);
    const uintptr_t target_offset = target_data - source_data;
    GGML_ASSERT(target_offset <= UINT32_MAX/2);

    ggml_init_params graph_params = {
        /* .mem_size   = */ 2*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * graph_ctx = ggml_init(graph_params);
    GGML_ASSERT(graph_ctx != nullptr);

    ggml_tensor * rows = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, 1);
    ggml_tensor * result = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_F32, 4);
    result->op = GGML_OP_GET_ROWS;
    result->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result->src[0] = source;
    result->src[1] = rows;
    ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
    GGML_ASSERT(graph_buffer != nullptr);

    const char marker[] = "NE3-LEAK-OK-1234";
    static_assert(sizeof(marker) - 1 == 4*sizeof(float), "marker must fill one row");
    const int32_t row = forged_stride ? 1 : 0;
    ggml_backend_tensor_set(rows, &row, 0, sizeof(row));
    ggml_backend_tensor_set(forged_stride ? target : source, marker, 0, sizeof(marker) - 1);

    int64_t original_ne[GGML_MAX_DIMS];
    size_t original_nb[GGML_MAX_DIMS];
    memcpy(original_ne, source->ne, sizeof(original_ne));
    memcpy(original_nb, source->nb, sizeof(original_nb));
    if (forged_stride) {
        source->ne[3] = 0;
        source->nb[1] = target_offset;
        source->nb[2] = 2*target_offset;
        source->nb[3] = 2*target_offset;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 1, false);
    graph->nodes[0] = result;
    graph->n_nodes = 1;
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    char output[sizeof(marker) - 1] = {};
    ggml_backend_tensor_get(result, output, 0, sizeof(output));
    GGML_ASSERT(memcmp(output, marker, sizeof(output)) == 0);

    memcpy(source->ne, original_ne, sizeof(original_ne));
    memcpy(source->nb, original_nb, sizeof(original_nb));
    ggml_backend_buffer_free(graph_buffer);
    ggml_backend_buffer_free(buffer_b);
    ggml_backend_buffer_free(buffer_a);
    ggml_free(graph_ctx);
    ggml_free(ctx_b);
    ggml_free(ctx_a);
    ggml_backend_free(backend);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--zero-element-outside-buffer") == 0) {
        ggml_backend_load_all();
        return test_zero_element_data(argv[2], true);
    }
    if (argc == 3 && strcmp(argv[1], "--zero-element-buffer-end") == 0) {
        ggml_backend_load_all();
        return test_zero_element_data(argv[2], false);
    }
    if (argc == 3 && strcmp(argv[1], "--get-rows-in-buffer-stride") == 0) {
        ggml_backend_load_all();
        return test_get_rows(argv[2], true);
    }
    if (argc == 3 && strcmp(argv[1], "--get-rows-valid") == 0) {
        ggml_backend_load_all();
        return test_get_rows(argv[2], false);
    }

    GGML_ASSERT(argc == 3);
    ggml_backend_load_all();

    const char * endpoint_a = argv[1];
    const char * endpoint_b = argv[2];

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 3*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
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

    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
