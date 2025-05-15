#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#define MAX_NARGS 2

int main(int argc, char *argv[]) {

    int n_threads = 4;
    int n_rounds  = 100;

    if (argc > 1) {
        n_threads = std::atoi(argv[1]);
    }

    if (argc > 2) {
        n_rounds  = std::atoi(argv[2]);
    }

    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Create graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Lots of small, parallel ops where barriers in between will dominate
    struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  64);
    for (int i = 0; i < 1000; i++) {
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 128);
        out = ggml_mul_mat(ctx, a, out);

        struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
        out = ggml_mul_mat(ctx, d, out);
    }

    ggml_build_forward_expand(gf, out);
    int n_nodes = ggml_graph_n_nodes(gf);

    // Create threadpool
    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool* threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        fprintf(stderr, "threadpool create failed : n_threads %d\n", n_threads);
        exit(1);
    }

    // Create compute plan
    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, threadpool);

    std::vector<uint8_t> work_data(cplan.work_size);
    cplan.work_data = work_data.data();

    std::cerr << "graph-compute with"
              << "\n n_threads: " << n_threads
              << "\n   n_nodes: " << n_nodes
              << "\n  n_rounds: " << n_rounds
              << "\n";
    // ggml_graph_print(gf);

    // Warmup
    ggml_graph_compute(gf, &cplan);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i=0; i < n_rounds; i++) {
        ggml_graph_compute(gf, &cplan);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    std::cerr << "graph-compute took " << usec << " usec "
              << "\n " << (float) usec / n_rounds << " usec per-iter"
              << "\n " << (float) nsec / (n_rounds * n_nodes) << " nsec per-node"
              << "\n";

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);

    return 0;
}
