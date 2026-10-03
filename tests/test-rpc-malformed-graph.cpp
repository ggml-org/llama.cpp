// Regression test: a remote graph whose PAD_REFLECT_1D node has op_params that would make
// the CPU kernel write outside the destination tensor must be rejected by the server
// instead of corrupting its heap.
//
// Before the fix the server aborts ("double free or corruption") under the same input.
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <csignal>
#include <cstdio>
#include <unistd.h>

int main(int argc, char ** argv) {
    signal(SIGPIPE, SIG_IGN);

    GGML_ASSERT(argc == 2);
    const char * endpoint = argv[1];

    ggml_init_params params = {
        /* .mem_size   = */ 16u*1024u*1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_dst = ggml_init(params);
    ggml_context * ctx_src = ggml_init(params);

    // destination: 8 floats (32 bytes), source: 64 floats; p0 = 4 puts the 64-float copy
    // past the end of the destination allocation.
    ggml_tensor * dst  = ggml_new_tensor_1d(ctx_dst, GGML_TYPE_F32, 8);
    ggml_tensor * src0 = ggml_new_tensor_1d(ctx_src, GGML_TYPE_F32, 64);

    dst->op     = GGML_OP_PAD_REFLECT_1D;
    dst->src[0] = src0;
    ((int32_t *) dst->op_params)[0] = 4;
    ((int32_t *) dst->op_params)[1] = 0;

    ggml_cgraph * graph = ggml_new_graph(ctx_src);
    ggml_build_forward_expand(graph, dst);

    ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_t buf_dst = ggml_backend_alloc_ctx_tensors(ctx_dst, backend);
    ggml_backend_buffer_t buf_src = ggml_backend_alloc_ctx_tensors(ctx_src, backend);
    GGML_ASSERT(buf_dst != nullptr);
    GGML_ASSERT(buf_src != nullptr);

    float values[64];
    for (size_t i = 0; i < 64; ++i) {
        values[i] = (float) i;
    }
    ggml_backend_tensor_set(src0, values, 0, sizeof(values));

    // server must refuse the graph; the connection may be torn down, so ignore the result
    (void) ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);

    // The server has already closed the connection after refusing the graph, so any
    // further RPC call (including the remote buffer frees) would abort the client.
    // The server owns no state for this connection anymore; just exit.
    (void) buf_dst;
    (void) buf_src;
    _exit(0);
}
