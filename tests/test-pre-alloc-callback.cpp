#include <cstdio>

#include "llama.h"
#include "get-model.h"

struct callback_state {
    bool called;
    bool reassign_ok;
};

static void pre_alloc_cb(ggml_backend_sched_t sched, struct ggml_cgraph * gf, void * user_data) {
    auto * state = static_cast<callback_state *>(user_data);
    state->called = true;

    // reassign the first node to a different backend and verify
    int n_backends = ggml_backend_sched_get_n_backends(sched);
    if (n_backends < 1 || ggml_graph_n_nodes(gf) <= 0) {
        return;
    }

    struct ggml_tensor * node = ggml_graph_node(gf, 0);
    ggml_backend_t current = ggml_backend_sched_get_tensor_backend(sched, node);
    ggml_backend_t target  = current;

    for (int i = 0; i < n_backends; i++) {
        ggml_backend_t candidate = ggml_backend_sched_get_backend(sched, i);
        if (candidate != current) {
            target = candidate;
            break;
        }
    }

    if (target != current) {
        ggml_backend_sched_set_tensor_backend(sched, node, target);
        state->reassign_ok = (ggml_backend_sched_get_tensor_backend(sched, node) == target);
    } else {
        // only one backend available — can't test reassignment, just verify the callback was called
        state->reassign_ok = true;
    }
}

int main(int argc, char ** argv) {
    auto * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();
    auto * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "FAIL: could not load model\n");
        llama_backend_free();
        return 1;
    }

    callback_state state = { false, false };

    auto params = llama_context_default_params();
    params.n_ctx   = 64;
    params.n_batch = 1;
    params.cb_pre_alloc           = pre_alloc_cb;
    params.cb_pre_alloc_user_data = &state;

    auto * ctx = llama_init_from_model(model, params);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_token token = 0;
    if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
        fprintf(stderr, "FAIL: llama_decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    int ret = (state.called && state.reassign_ok) ? 0 : 1;

    if (ret != 0) {
        fprintf(stderr, "FAIL: called=%d reassign_ok=%d\n", state.called, state.reassign_ok);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ret;
}
