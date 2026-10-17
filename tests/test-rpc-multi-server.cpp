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
        /* .mem_size   = */ 11*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
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

    // malformed supports_op queries are answered as unsupported
    ggml_backend_dev_t dev_a = ggml_backend_get_device(backend_a);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);

    ggml_tensor * no_srcs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    no_srcs->op = GGML_OP_MUL_MAT;
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev_a, no_srcs));

    ggml_tensor * no_src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    no_src1->op = GGML_OP_MUL_MAT;
    no_src1->src[0] = x;
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev_a, no_src1));

    ggml_tensor * bad_op = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    bad_op->op = (ggml_op) (GGML_OP_COUNT + 7);
    bad_op->src[0] = x;
    bad_op->src[1] = y;
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev_a, bad_op));

    ggml_tensor * bad_type = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    bad_type->op = GGML_OP_ADD;
    bad_type->type = (ggml_type) 4; // removed Q4_2: no type traits
    bad_type->nb[0] = 1;
    bad_type->src[0] = x;
    bad_type->src[1] = y;
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev_a, bad_type));

    ggml_tensor * huge_ne = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    huge_ne->op = GGML_OP_ADD;
    huge_ne->src[0] = x;
    huge_ne->src[1] = y;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        huge_ne->ne[i] = UINT32_MAX;
    }
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev_a, huge_ne));

    ggml_tensor * mul_mat = ggml_mul_mat(ctx, x, y);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev_a, mul_mat));
    ggml_backend_rpc_get_device_memory(endpoint_a, 0, &free_mem, &total_mem);
    GGML_ASSERT(total_mem > 0);

    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
