#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <signal.h>
#include <sys/types.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 3);
    ggml_backend_load_all();

    const char * endpoint   = argv[1];
    const pid_t  server_pid = (pid_t) atoi(argv[2]);

    ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint, 0);
    GGML_ASSERT(buft != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*2 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * b = ggml_add(ctx, a, a);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    GGML_ASSERT(buffer != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, b);

    // control: if this fails the rest of the test means nothing
    const float in[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float out[4];
    ggml_backend_tensor_set(a, in, 0, sizeof(in));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(b, out, 0, sizeof(out));
    GGML_ASSERT(out[0] == 2.0f && out[3] == 8.0f);

    kill(server_pid, SIGKILL);

    // GRAPH_COMPUTE has no response, so a send can still succeed until the peer resets
    ggml_status status = GGML_STATUS_SUCCESS;
    for (int i = 0; i < 500 && status != GGML_STATUS_FAILED; i++) {
        status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_FAILED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    GGML_ASSERT(status == GGML_STATUS_FAILED);

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);

    // a read cannot report an error, so it must at least not leave stale data behind
    memset(out, 0x5a, sizeof(out));
    ggml_backend_tensor_get(b, out, 0, sizeof(out));
    GGML_ASSERT(out[0] == 0.0f && out[3] == 0.0f);

    // tearing down a dead endpoint must not abort either
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
