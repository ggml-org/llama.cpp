// Regression test for the RPC validation of PAD_REFLECT_1D.
//
// The server rebuilds a graph from client-supplied tensor descriptions, so the shape and the
// op_params of a node are not necessarily self-consistent.  A PAD_REFLECT_1D node whose
// destination is smaller than the reflected copy used to make the CPU kernel write outside
// the destination tensor and abort the server ("double free or corruption").
//
// Two cases are run against the same server:
//
//   good : a well-formed PAD_REFLECT_1D graph is accepted and computes the reflected output
//   bad  : a node that would write past the destination is rejected, and the server stays up
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <unistd.h>

static const int n_src = 64;
static const int n_dst = 72;
static const int pad   = 4;

// dst = [ 4, 3, 2, 1, 0 .. 63, 62, 61, 60, 59 ]
static float expected(int i) {
    if (i < pad) {
        return (float) (pad - i);
    }
    if (i < pad + n_src) {
        return (float) (i - pad);
    }
    return (float) (n_src - 2 - (i - pad - n_src));
}

int main(int argc, char ** argv) {
    signal(SIGPIPE, SIG_IGN);

    GGML_ASSERT(argc == 3);
    const char * endpoint = argv[1];
    const bool   good     = strcmp(argv[2], "good") == 0;

    ggml_init_params params = {
        /* .mem_size   = */ 16u*1024u*1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_dst = ggml_init(params);
    ggml_context * ctx_src = ggml_init(params);

    ggml_tensor * dst;
    ggml_tensor * src0;

    if (good) {
        // mirror of ggml_pad_reflect_1d(): dst->ne[0] == src0->ne[0] + p0 + p1
        dst  = ggml_new_tensor_1d(ctx_dst, GGML_TYPE_F32, n_dst);
        src0 = ggml_new_tensor_1d(ctx_src, GGML_TYPE_F32, n_src);
        ((int32_t *) dst->op_params)[0] = pad;
        ((int32_t *) dst->op_params)[1] = pad;
    } else {
        // p0 = 4 puts the 64-float copy past the end of the 8-float destination
        dst  = ggml_new_tensor_1d(ctx_dst, GGML_TYPE_F32, 8);
        src0 = ggml_new_tensor_1d(ctx_src, GGML_TYPE_F32, n_src);
        ((int32_t *) dst->op_params)[0] = pad;
        ((int32_t *) dst->op_params)[1] = 0;
    }

    dst->op     = GGML_OP_PAD_REFLECT_1D;
    dst->src[0] = src0;

    ggml_cgraph * graph = ggml_new_graph(ctx_src);
    ggml_build_forward_expand(graph, dst);

    ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_t buf_dst = ggml_backend_alloc_ctx_tensors(ctx_dst, backend);
    ggml_backend_buffer_t buf_src = ggml_backend_alloc_ctx_tensors(ctx_src, backend);
    GGML_ASSERT(buf_dst != nullptr);
    GGML_ASSERT(buf_src != nullptr);

    float values[n_src];
    for (int i = 0; i < n_src; ++i) {
        values[i] = (float) i;
    }
    ggml_backend_tensor_set(src0, values, 0, sizeof(values));

    (void) ggml_backend_graph_compute(backend, graph);

    if (!good) {
        // The server refuses the graph and closes the connection, so any further RPC call
        // (including the remote buffer frees) would abort the client.  The shell test checks
        // the server side: the rejection is logged and the server is still running.
        (void) buf_dst;
        (void) buf_src;
        _exit(0);
    }

    ggml_backend_synchronize(backend);

    float out[n_dst];
    ggml_backend_tensor_get(dst, out, 0, sizeof(out));

    int rc = 0;
    for (int i = 0; i < n_dst; ++i) {
        if (out[i] != expected(i)) {
            fprintf(stderr, "unexpected PAD_REFLECT_1D output at %d: got %g, want %g\n", i, out[i], expected(i));
            rc = 1;
            break;
        }
    }

    ggml_backend_buffer_free(buf_dst);
    ggml_backend_buffer_free(buf_src);
    ggml_free(ctx_dst);
    ggml_free(ctx_src);
    ggml_backend_free(backend);

    return rc;
}
