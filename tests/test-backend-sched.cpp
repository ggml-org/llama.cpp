// Stress test for ggml_backend_sched: graphs built only from increments, placed on alternating backends.
// Since every node adds one, the expected result of a counter is just the number of times it was incremented,
// which makes the correct output independent of how the scheduler splits and copies the graph.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>


static ggml_backend_buffer_type_t sched_cpu_buft(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu, bool use_device_host_buft) {
    if (use_device_host_buft) {
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend_gpu));
        if (host != nullptr) {
            return host;
        }
    }
    return ggml_backend_get_default_buffer_type(backend_cpu);
}

static ggml_backend_sched_t create_test_scheduler(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu, size_t graph_size, bool use_device_host_buft) {
    ggml_backend_t backends[2] = { backend_gpu, backend_cpu };
    ggml_backend_buffer_type_t bufts[2] = {
        ggml_backend_get_default_buffer_type(backend_gpu),
        sched_cpu_buft(backend_gpu, backend_cpu, use_device_host_buft),
    };
    return ggml_backend_sched_new(backends, bufts, 2, graph_size, /*parallel =*/ false, /*op_offload =*/ false);
}

static bool test_ping_pong(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu, int n_nodes, int64_t ne, bool use_device_host_buft) {

    ggml_backend_t backends[2] = { backend_gpu, backend_cpu };

    const size_t graph_size = n_nodes + 16;

    ggml_backend_sched_t sched = create_test_scheduler(backend_gpu, backend_cpu, graph_size, use_device_host_buft);

    // the inputs are allocated separately so that they can be written before the graph is computed
    ggml_init_params params_static = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_static = ggml_init(params_static);

    ggml_tensor * x = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, ne);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_tensor * one = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, ne);
    ggml_set_name(one, "one");
    ggml_set_input(one);

    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, backend_cpu);

    std::vector<float> data(ne);

    std::fill(data.begin(), data.end(), 0.0f);
    ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));

    std::fill(data.begin(), data.end(), 1.0f);
    ggml_backend_tensor_set(one, data.data(), 0, ggml_nbytes(one));

    ggml_init_params params_compute = {
        /*.mem_size   =*/ (n_nodes + 2)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx_compute, graph_size, false);

    std::vector<ggml_tensor *> nodes;

    ggml_tensor * out = x;
    for (int i = 0; i < n_nodes; i++) {
        out = ggml_add(ctx_compute, out, one);
        nodes.push_back(out);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    // the assignments are only kept until the next reset, and ggml_backend_sched_graph_compute
    // resets the scheduler unless the graph has already been allocated
    ggml_backend_sched_reset(sched);
    for (int i = 0; i < n_nodes; i++) {
        ggml_backend_sched_set_tensor_backend(sched, nodes[i], backends[i % 2]);
    }

    bool ok = true;

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        printf("\n    failed to allocate the graph\n");
        ok = false;
    }

    if (ok && ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        printf("\n    failed to compute the graph\n");
        ok = false;
    }

    if (ok) {
        const int n_splits = ggml_backend_sched_get_n_splits(sched);
        if (n_splits != n_nodes) {
            printf("\n    n_splits = %d, expected %d - the backend assignments were not respected\n", n_splits, n_nodes);
            ok = false;
        }
    }

    if (ok) {
        ggml_backend_tensor_get(out, data.data(), 0, ggml_nbytes(out));
        for (int64_t i = 0; i < ne; i++) {
            if (data[i] != float(n_nodes)) {
                printf("\n    out[%" PRId64 "] = %f, expected %d\n", i, data[i], n_nodes);
                ok = false;
                break;
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);

    return ok;
}

// Same idea as test_ping_pong, but with 8 independent counters and a graph shape that covers
// three cases the plain chain does not:
//  - splits with no inputs at all, from lanes restarted out of constants that already live on the split's backend
//  - lanes with different histories, so a misrouted copy shows up as a wrong count in a single lane
//  - one split producing two values followed by two splits that each consume one of them and do not depend on each other
static bool test_multi_lane(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu, int n_rounds, int64_t ne, bool use_device_host_buft) {

    const int GPU = 0;
    const int CPU = 1;

    const int n_lanes = 8;

    ggml_backend_t backends[2] = { backend_gpu, backend_cpu };

    // the graph needs 2*n_lanes nodes per round plus one restart for half of the lanes, but the hash set of
    // the scheduler is sized from the same value and GGML_SCHED_DEBUG inserts every copy tensor into it as
    // well, so leave room for those
    const size_t n_nodes    = 2*n_lanes*n_rounds + n_lanes/2;
    const size_t graph_size = 3*n_nodes + 64;

    ggml_backend_sched_t sched = create_test_scheduler(backend_gpu, backend_cpu, graph_size, use_device_host_buft);

    ggml_init_params params_static = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    // the constants are kept on both backends so that a lane can be restarted without any cross-backend copy
    ggml_context * ctx_cpu = ggml_init(params_static);

    ggml_tensor * zero_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, ne);
    ggml_set_name(zero_cpu, "zero_cpu");

    ggml_tensor * one_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, ne);
    ggml_set_name(one_cpu, "one_cpu");

    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);

    ggml_context * ctx_gpu = ggml_init(params_static);

    ggml_tensor * zero_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, ne);
    ggml_set_name(zero_gpu, "zero_gpu");

    ggml_tensor * one_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, ne);
    ggml_set_name(one_gpu, "one_gpu");

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);

    std::vector<float> data(ne);

    std::fill(data.begin(), data.end(), 0.0f);
    ggml_backend_tensor_set(zero_cpu, data.data(), 0, ggml_nbytes(zero_cpu));
    ggml_backend_tensor_set(zero_gpu, data.data(), 0, ggml_nbytes(zero_gpu));

    std::fill(data.begin(), data.end(), 1.0f);
    ggml_backend_tensor_set(one_cpu, data.data(), 0, ggml_nbytes(one_cpu));
    ggml_backend_tensor_set(one_gpu, data.data(), 0, ggml_nbytes(one_gpu));

    ggml_init_params params_compute = {
        /*.mem_size   =*/ (graph_size + 8)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx_compute, graph_size, false);

    ggml_tensor * lane[n_lanes];
    int           expected[n_lanes];
    for (int l = 0; l < n_lanes; l++) {
        lane[l]     = zero_cpu;
        expected[l] = 0;
    }

    std::vector<ggml_tensor *> nodes;
    std::vector<int>           node_backend;

    // expanding each node as it is created keeps the graph order equal to the creation order,
    // which is what determines where the scheduler puts the split boundaries
    auto append = [&](int l, ggml_tensor * node, int b) {
        lane[l] = node;
        nodes.push_back(node);
        node_backend.push_back(b);
        ggml_build_forward_expand(gf, node);
    };

    auto increment = [&](int l, int b) {
        append(l, ggml_add(ctx_compute, lane[l], b == GPU ? one_gpu : one_cpu), b);
        expected[l]++;
    };

    // both sources already live on backend b, so the resulting split has no inputs to copy
    auto restart = [&](int l, int b) {
        append(l, ggml_add(ctx_compute, b == GPU ? zero_gpu : zero_cpu, b == GPU ? one_gpu : one_cpu), b);
        expected[l] = 1;
    };

    for (int round = 0; round < n_rounds; round++) {
        // halfway through, the upper half of the lanes starts over from scratch
        if (round == n_rounds/2) {
            // start on the backend the previous node did not use, so that no restart is absorbed into the split before it
            int b = node_backend.empty() || node_backend.back() == CPU ? GPU : CPU;
            for (int l = n_lanes/2; l < n_lanes; l++) {
                restart(l, b);
                b = b == CPU ? GPU : CPU;
            }
        }

        for (int p = 0; p < n_lanes/2; p++) {
            const int la = 2*p;
            const int lb = 2*p + 1;

            const int prod = p % 2 == 0 ? CPU : GPU;

            // one split producing two values ...
            increment(la, prod);
            increment(lb, prod);

            // ... and two splits that each consume one of them, with no dependency on each other
            increment(la, prod == CPU ? GPU : CPU);
            increment(lb, prod);
        }
    }

    for (int l = 0; l < n_lanes; l++) {
        ggml_set_output(lane[l]);
    }

    // the scheduler starts a new split on every backend change, so the assignments below fix the split count
    int n_splits_expected = nodes.empty() ? 0 : 1;
    for (size_t i = 1; i < node_backend.size(); i++) {
        n_splits_expected += node_backend[i] != node_backend[i - 1];
    }

    ggml_backend_sched_reset(sched);
    for (size_t i = 0; i < nodes.size(); i++) {
        ggml_backend_sched_set_tensor_backend(sched, nodes[i], backends[node_backend[i]]);
    }

    bool ok = true;

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        printf("\n    failed to allocate the graph\n");
        ok = false;
    }

    if (ok && ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        printf("\n    failed to compute the graph\n");
        ok = false;
    }

    if (ok) {
        const int n_splits = ggml_backend_sched_get_n_splits(sched);
        if (n_splits != n_splits_expected) {
            printf("\n    n_splits = %d, expected %d - the backend assignments were not respected\n", n_splits, n_splits_expected);
            ok = false;
        }
    }

    if (ok) {
        for (int l = 0; l < n_lanes && ok; l++) {
            ggml_backend_tensor_get(lane[l], data.data(), 0, ggml_nbytes(lane[l]));
            for (int64_t i = 0; i < ne; i++) {
                if (data[i] != float(expected[l])) {
                    printf("\n    lane %d [%" PRId64 "] = %f, expected %d\n", l, i, data[i], expected[l]);
                    ok = false;
                    break;
                }
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);
    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_cpu);

    return ok;
}

// Created in response to https://github.com/ggml-org/llama.cpp/issues/23321
// Currently, input copies to splits synchronize split execution.
// CPU split N+1 could overwrite output of split N if this was not copied to backend in time.
// Mechanism here: GPU split with long sleep to clog CUDA stream / vk command queue,
// followed by CPU splits without input, provoking a race condition in the current setup.
//   GPU: 100ms sleep to clog stream/queue
//   CPU: produce {55}
//   GPU: request copy of {55} in CUDA stream, ADD {44}; Output should be {99}
//   CPU: produce {66}
//   GPU: increment both {99} and {66} -> {100} and {67}
//   Correct result is thus {100}, incorrect output is {111} when {55} was overwritten by {66} before copy started.
// Note: currently only reproducible on async H2D copy (pinned memory in CUDA).
static bool test_sleep_race(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu, int64_t ne, int32_t sleep_us, bool use_device_host_buft) {

    const int GPU = 0;
    const int CPU = 1;

    ggml_backend_t backends[2] = { backend_gpu, backend_cpu };

    const size_t graph_size = 64;

    ggml_backend_sched_t sched = create_test_scheduler(backend_gpu, backend_cpu, graph_size, use_device_host_buft);

    ggml_init_params params_static = {
        /*.mem_size   =*/ 4*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx_cpu = ggml_init(params_static);

    ggml_tensor * zero_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, ne);
    ggml_set_name(zero_cpu, "zero_cpu");

    ggml_tensor * val55_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, ne);
    ggml_set_name(val55_cpu, "val55_cpu");

    ggml_tensor * val66_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, ne);
    ggml_set_name(val66_cpu, "val66_cpu");

    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);

    ggml_context * ctx_gpu = ggml_init(params_static);

    ggml_tensor * zero_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, ne);
    ggml_set_name(zero_gpu, "zero_gpu");

    ggml_tensor * one_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, ne);
    ggml_set_name(one_gpu, "one_gpu");

    ggml_tensor * val44_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, ne);
    ggml_set_name(val44_gpu, "val44_gpu");

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);

    std::vector<float> data(ne);

    std::fill(data.begin(), data.end(), 0.0f);
    ggml_backend_tensor_set(zero_cpu, data.data(), 0, ggml_nbytes(zero_cpu));
    ggml_backend_tensor_set(zero_gpu, data.data(), 0, ggml_nbytes(zero_gpu));

    std::fill(data.begin(), data.end(), 1.0f);
    ggml_backend_tensor_set(one_gpu, data.data(), 0, ggml_nbytes(one_gpu));

    std::fill(data.begin(), data.end(), 44.0f);
    ggml_backend_tensor_set(val44_gpu, data.data(), 0, ggml_nbytes(val44_gpu));

    std::fill(data.begin(), data.end(), 55.0f);
    ggml_backend_tensor_set(val55_cpu, data.data(), 0, ggml_nbytes(val55_cpu));

    std::fill(data.begin(), data.end(), 66.0f);
    ggml_backend_tensor_set(val66_cpu, data.data(), 0, ggml_nbytes(val66_cpu));

    ggml_init_params params_compute = {
        /*.mem_size   =*/ (graph_size + 8)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx_compute, graph_size, false);

    std::vector<ggml_tensor *> nodes;
    std::vector<int>           node_backend;

    auto append = [&](ggml_tensor * node, int b) {
        nodes.push_back(node);
        node_backend.push_back(b);
        ggml_build_forward_expand(gf, node);
    };

    // GPU: add then sleep - occupies the GPU while later splits run
    ggml_tensor * delayed = ggml_add(ctx_compute, zero_gpu, one_gpu);
    append(delayed, GPU);
    delayed = ggml_sleep(ctx_compute, delayed, sleep_us);
    append(delayed, GPU);

    // CPU: 0 + 55 -> 55
    ggml_tensor * v55 = ggml_add(ctx_compute, zero_cpu, val55_cpu);
    append(v55, CPU);

    // GPU: 55 + 44 -> 99
    ggml_tensor * v99 = ggml_add(ctx_compute, v55, val44_gpu);
    append(v99, GPU);

    // CPU: 0 + 66 -> 66
    ggml_tensor * v66 = ggml_add(ctx_compute, zero_cpu, val66_cpu);
    append(v66, CPU);

    // GPU: increment both of the previous split outputs
    ggml_tensor * out99 = ggml_add(ctx_compute, v99, one_gpu);
    append(out99, GPU);
    ggml_tensor * out66 = ggml_add(ctx_compute, v66, one_gpu);
    append(out66, GPU);

    ggml_set_output(delayed);
    ggml_set_output(out99);
    ggml_set_output(out66);

    const int n_splits_expected = 5;

    ggml_backend_sched_reset(sched);
    for (size_t i = 0; i < nodes.size(); i++) {
        ggml_backend_sched_set_tensor_backend(sched, nodes[i], backends[node_backend[i]]);
    }

    bool ok = true;

    if (!ggml_backend_supports_op(backend_gpu, delayed)) {
        printf("\n    GGML_OP_SLEEP not supported on GPU, skipping\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx_compute);
        ggml_backend_buffer_free(buf_gpu);
        ggml_free(ctx_gpu);
        ggml_backend_buffer_free(buf_cpu);
        ggml_free(ctx_cpu);
        return true;
    }

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        printf("\n    failed to allocate the graph\n");
        ok = false;
    }

    // checks if the the allocator reuses the same memory for output tensors between splits.
    if (ok && v55->data == v66->data) {
        printf("\n    NOTE: v55 and v66 alias after alloc (v55=%p v66=%p)\n", v55->data, v66->data);
    } else {
        printf("\n    NOTE: v55 and v66 do not alias after alloc (v55=%p v66=%p)\n", v55->data, v66->data);
    }

    if (ok && ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        printf("\n    failed to compute the graph\n");
        ok = false;
    }

    if (ok) {
        const int n_splits = ggml_backend_sched_get_n_splits(sched);
        if (n_splits != n_splits_expected) {
            printf("\n    n_splits = %d, expected %d - the backend assignments were not respected\n", n_splits, n_splits_expected);
            ok = false;
        }
    }

    if (ok) {
        const struct {
            ggml_tensor * tensor;
            float         expected;
            const char *  name;
        } checks[] = {
            { delayed, 1.0f,   "delayed" },
            { out99,   100.0f, "out99" },
            { out66,   67.0f,  "out66" },
        };

        for (const auto & c : checks) {
            ggml_backend_tensor_get(c.tensor, data.data(), 0, ggml_nbytes(c.tensor));
            for (int64_t i = 0; i < ne; i++) {
                if (data[i] != c.expected) {
                    printf("\n    %s[%" PRId64 "] = %f, expected %f\n", c.name, i, data[i], c.expected);
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);
    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_cpu);

    return ok;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev_gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            dev_gpu = dev;
            break;
        }
    }

    if (dev_gpu == nullptr) {
        printf("no GPU device found, skipping\n");
        return 0;
    }

    ggml_backend_t backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(backend_gpu != nullptr);
    GGML_ASSERT(backend_cpu != nullptr);

    printf("GPU: %s (%s)\n", ggml_backend_name(backend_gpu), ggml_backend_dev_description(dev_gpu));
    printf("CPU: %s\n", ggml_backend_name(backend_cpu));

    const bool have_device_host_buft = ggml_backend_dev_host_buffer_type(dev_gpu) != nullptr;
    if (!have_device_host_buft) {
        printf("GPU has no host buffer type; device_host cases will be skipped\n");
    }
    printf("\n");

    int n_ok   = 0;
    int n_test = 0;

    for (bool use_device_host_buft : { false, true }) {
        if (use_device_host_buft && !have_device_host_buft) {
            continue;
        }

        const char * buft = use_device_host_buft ? "device_host" : "pageable";
        printf("=== CPU sched buft: %s ===\n\n", buft);

        for (int n_nodes : { 1, 2, 3, 8, 33, 128, 1024 }) {
            for (int ne : { 1, 4096 }) {
                printf("  n_nodes = %4d, ne = %4d: ", n_nodes, ne);
                fflush(stdout);

                const bool ok = test_ping_pong(backend_gpu, backend_cpu, n_nodes, ne, use_device_host_buft);

                printf("%s\n", ok ? "OK" : "FAIL");

                n_ok += ok;
                n_test++;
            }
        }

        printf("\n");

        for (int n_rounds : { 1, 2, 3, 8, 33, 64 }) {
            for (int ne : { 1, 4096 }) {
                printf("  n_rounds = %4d, ne = %4d: ", n_rounds, ne);
                fflush(stdout);

                const bool ok = test_multi_lane(backend_gpu, backend_cpu, n_rounds, ne, use_device_host_buft);

                printf("%s\n", ok ? "OK" : "FAIL");

                n_ok += ok;
                n_test++;
            }
        }

        printf("\n");

        for (int32_t sleep_us : { 100000, 0 }) {
            for (int ne : { 1, 4096 }) {
                printf("  sleep_race sleep_us = %6d, ne = %4d: ", sleep_us, ne);
                fflush(stdout);

                const bool ok = test_sleep_race(backend_gpu, backend_cpu, ne, sleep_us, use_device_host_buft);

                printf("%s\n", ok ? "OK" : "FAIL");

                n_ok += ok;
                n_test++;
            }
        }

        printf("\n");
    }

    ggml_backend_free(backend_gpu);
    ggml_backend_free(backend_cpu);

    printf("%d/%d tests passed\n", n_ok, n_test);

    return n_ok == n_test ? 0 : 1;
}
