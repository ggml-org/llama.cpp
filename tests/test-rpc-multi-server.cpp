#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <algorithm>
#include <vector>

static void test_uploads(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    const size_t count = 4 * 1024 * 1024 + 3;
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> input(count), output(count);
    for (size_t i = 0; i < count; i++) {
        input[i] = float(i % 113);
    }
    const auto expected = input;
    ggml_backend_tensor_set(tensor, input.data(), 0, input.size() * sizeof(float));
    std::fill(input.begin(), input.end(), -1.0f);
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
    GGML_ASSERT(output == expected);

    ggml_backend_event_t event = ggml_backend_event_new(ggml_backend_get_device(backend));
    GGML_ASSERT(event != nullptr);
    for (const size_t length : {count, size_t(17), count - 3, count - 3}) {
        const size_t offset = length == count ? 0 : 1;
        input = expected;
        ggml_backend_tensor_set_async(backend, tensor, input.data(), offset * sizeof(float), length * sizeof(float));
        ggml_backend_event_record(event, backend);
        ggml_backend_event_synchronize(event);
        std::fill(input.begin(), input.end(), -2.0f);
        ggml_backend_tensor_get(tensor, output.data(), offset * sizeof(float), length * sizeof(float));
        GGML_ASSERT(std::equal(output.begin(), output.begin() + length, expected.begin()));
    }
    ggml_backend_event_free(event);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

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
    test_uploads(backend_a);
    test_uploads(backend_b);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
