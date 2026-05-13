#include "arg.h"
#include "common.h"
#include "dist-comm.h"

#include <clocale>
#include <cstdio>

static void print_usage(int argc, char ** argv) {
    (void) argc;
    std::fprintf(stderr, "usage: %s [llama options]\n", argv[0]);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "required distributed configuration (via args or env):\n");
    std::fprintf(stderr, "  --world-size <N>=2+\n");
    std::fprintf(stderr, "  --world-rank <0..N-1>\n");
    std::fprintf(stderr, "  --dist-master <host:port>\n");
    std::fprintf(stderr, "  --dist-listen-port <port> (optional; default: master_port + world-rank)\n");
    std::fprintf(stderr, "optional model placement controls:\n");
    std::fprintf(stderr, "  LLAMA_DIST_TP_MODEL_SPLIT_MODE={none|layer|row|tensor}\n");
    std::fprintf(stderr, "  LLAMA_DIST_TP_MODEL_TENSOR_SPLIT=<csv-floats>\n");
    std::fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    common_dist_tp_env env;
    std::string err;
    if (!common_dist_tp_resolve(params, env, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    if (!env.enabled) {
        std::fprintf(stderr, "error: distributed TP is not enabled\n");
        return 1;
    }

    if (env.world_rank == 0) {
        std::fprintf(stderr, "error: dist-peer is worker-only; run rank 0 with llama-cli/llama-completion\n");
        return 1;
    }

    if (!common_dist_tp_apply_env_overrides(params, env, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    if (llama_init->context() == nullptr || llama_init->model() == nullptr) {
        std::fprintf(stderr, "error: failed to initialize model/context\n");
        return 1;
    }

    return common_dist_tp_run_worker(*llama_init, env);
}
