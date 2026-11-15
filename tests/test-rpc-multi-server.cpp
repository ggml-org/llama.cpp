#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cstring>

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 3 || argc == 4);
    ggml_backend_load_all();

    const char * endpoint_a = argv[1];
    const char * endpoint_b = argv[2];

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);

    if (argc == 4) {
        // like a client older than 10.0.1, which does not ask which ops the device supports: send server B
        // a graph with an op its backend cannot run. The server refuses the graph and closes the connection,
        // so the next call aborts this process. The script checks the exit status and the server log.
        GGML_ASSERT(strcmp(argv[3], "unsupported") == 0);
        ggml_init_params uparams = {
            /* .mem_size   = */ 3*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * uctx = ggml_init(uparams);
        GGML_ASSERT(uctx != nullptr);
        ggml_tensor * src = ggml_new_tensor_1d(uctx, GGML_TYPE_F32, 256);
        ggml_tensor * dst = ggml_new_tensor_1d(uctx, GGML_TYPE_IQ2_XXS, 256);
        ggml_tensor * cpy = ggml_cpy(uctx, src, dst); // the CPU backend cannot convert to IQ2_XXS
        GGML_ASSERT(ggml_backend_alloc_ctx_tensors(uctx, backend_b) != nullptr);
        ggml_cgraph * ugraph = ggml_new_graph_custom(uctx, 1, false);
        ugraph->nodes[0] = cpy;
        ugraph->n_nodes  = 1;
        ggml_backend_graph_compute(backend_b, ugraph);
        size_t free_mem;
        size_t total_mem;
        ggml_backend_rpc_get_device_memory(endpoint_b, 0, &free_mem, &total_mem);
        return 0;
    }

    ggml_init_params params = {
        /* .mem_size   = */ 19*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false) + ggml_graph_overhead_custom(3, false),
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

    // a view is asked about with its view_src: a view that reaches past the end of its view_src is malformed
    ggml_tensor * base     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64*64);
    ggml_tensor * view     = ggml_view_2d(ctx, base, 64, 64, 64*sizeof(float), 0);
    ggml_tensor * bad_view = ggml_view_2d(ctx, base, 64, 64, 64*sizeof(float), 0);
    bad_view->view_offs = sizeof(float); // its last element is past the end of base
    GGML_ASSERT( ggml_backend_dev_supports_op(dev_a, ggml_add(ctx, view, y)));
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev_a, ggml_add(ctx, bad_view, y)));

    // a malformed op in a request is answered as unsupported and does not change the answers for the ops that do not
    // use it: the prefetch asks about the four nodes in one request, scale has a malformed src. add uses scale, so its
    // answer depends on what else is in the request, and is not checked
    ggml_tensor * bad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    bad->op = (ggml_op) (GGML_OP_COUNT + 9);
    bad->src[0] = x;
    ggml_tensor * scale = ggml_scale(ctx, bad, 2.0f);
    ggml_tensor * add   = ggml_add(ctx, scale, y);
    ggml_tensor * sub   = ggml_sub(ctx, x, y);
    ggml_cgraph * graph3 = ggml_new_graph_custom(ctx, 4, false);
    graph3->nodes[0] = bad;
    graph3->nodes[1] = scale;
    graph3->nodes[2] = add;
    graph3->nodes[3] = sub;
    graph3->n_nodes  = 4;
    auto prefetch = (ggml_backend_dev_supports_op_prefetch_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev_a), "ggml_backend_dev_supports_op_prefetch");
    GGML_ASSERT(prefetch != nullptr);
    prefetch(dev_a, graph3);
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev_a, scale));
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev_a, bad));
    GGML_ASSERT( ggml_backend_dev_supports_op(dev_a, sub));
    ggml_backend_rpc_get_device_memory(endpoint_a, 0, &free_mem, &total_mem);
    GGML_ASSERT(total_mem > 0);

    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
