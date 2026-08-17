// Test suite for scheduler and backend behavior

// Towards formalized scheduling behavior:
// - a single inference pass can run on several backends. The subset of nodes running on a single backend is a split
// - synchronous backends:
//      - explicit synchronization (ggml_backend_synchronize()) is required between each operation
// - asynchronous backends:
//      - Activations from one split can be copied asynchronously to the next (output of split N -> input of split N+1)
//      - Several scheduling patterns must be supported by async backends. The scheduler may:
//          - not explicitly synchronize between CPU->backend memcpy and graph execution on backend
//          - dispatch several parallel memcpys to the same backend at once
//          - schedule splits without inputs at any point in the graph

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>


// global counters for pass/fail
static int n_ok   = 0;
static int n_test = 0;

// pretty-print helpers start
static int  case_len = 0;
static char case_label[256];

static void case_end(bool ok) {
    if (case_len == 0) {
        case_len = printf("  %s", case_label);
    }

    printf("%*s%s\n", std::max(1, 82 - case_len), "", ok ? "OK" : "FAIL");

    n_ok += ok;
    n_test++;
}

static void vnote(const char * fmt, va_list args) {
    if (case_len != 0) {
        printf("\n");
    }
    printf("    ");
    vprintf(fmt, args);
    printf("\n");

    case_len = 0;
}

GGML_ATTRIBUTE_FORMAT(1, 2)
static void note(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vnote(fmt, args);
    va_end(args);
}

GGML_ATTRIBUTE_FORMAT(1, 2)
static bool fail(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vnote(fmt, args);
    va_end(args);

    return false;
}

GGML_ATTRIBUTE_FORMAT(1, 2)
static void case_begin(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(case_label, sizeof(case_label), fmt, args);
    va_end(args);

    case_len = printf("  %s", case_label);

    fflush(stdout);
}
// pretty-print helpers end


static ggml_backend_buffer_type_t sched_cpu_buft(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu, bool use_device_host_buft) {
    if (use_device_host_buft) {
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend_gpu));
        if (host != nullptr) {
            return host;
        }
    }
    return ggml_backend_get_default_buffer_type(backend_cpu);
}

static ggml_backend_sched_t create_test_scheduler(std::vector<ggml_backend_t> backends, size_t graph_size,
        bool use_device_host_buft, bool parallel = false) {

    std::vector<ggml_backend_buffer_type_t> bufts(backends.size());
    for (size_t b = 0; b < backends.size(); b++) {
        bufts[b] = ggml_backend_get_default_buffer_type(backends[b]);
    }
    // CPU backend is always last
    bufts.back() = sched_cpu_buft(backends[0], backends.back(), use_device_host_buft);

    return ggml_backend_sched_new(backends.data(), bufts.data(), (int) backends.size(), graph_size, parallel, /*op_offload =*/ false);
}

// the nodes of the graph together with the backend each of them is assigned to
struct sched_graph {
    ggml_cgraph *              gf;
    std::vector<ggml_tensor *> nodes;
    std::vector<int>           backend_id;

    sched_graph(ggml_context * ctx, size_t graph_size) : gf(ggml_new_graph_custom(ctx, graph_size, false)) {}

    ggml_tensor * add(ggml_tensor * node, int backend) {
        nodes.push_back(node);
        backend_id.push_back(backend);
        ggml_build_forward_expand(gf, node);

        return node;
    }
};

struct sched_check {
    ggml_tensor * tensor;
    float         expected;
    std::string   name;
};

// executes the graph and evaluates results
static bool run_and_check(ggml_backend_sched_t sched, const sched_graph & g,
        const std::vector<ggml_backend_t> & backends, const std::vector<sched_check> & checks, int64_t ne) {

    int n_splits_expected = g.nodes.empty() ? 0 : 1;
    for (size_t i = 1; i < g.backend_id.size(); i++) {
        n_splits_expected += g.backend_id[i] != g.backend_id[i - 1];
    }

    ggml_backend_sched_reset(sched);
    for (size_t i = 0; i < g.nodes.size(); i++) {
        ggml_backend_sched_set_tensor_backend(sched, g.nodes[i], backends[g.backend_id[i]]);
    }

    if (!ggml_backend_sched_alloc_graph(sched, g.gf)) {
        return fail("failed to allocate the graph");
    }

    if (ggml_backend_sched_graph_compute(sched, g.gf) != GGML_STATUS_SUCCESS) {
        return fail("failed to compute the graph");
    }

    const int n_splits = ggml_backend_sched_get_n_splits(sched);
    if (n_splits != n_splits_expected) {
        return fail("n_splits = %d, expected %d - the backend assignments were not respected", n_splits, n_splits_expected);
    }

    std::vector<float> data(ne);
    for (const sched_check & c : checks) {
        ggml_backend_tensor_get(c.tensor, data.data(), 0, ggml_nbytes(c.tensor));
        for (int64_t i = 0; i < ne; i++) {
            if (data[i] != c.expected) {
                return fail("%s[%" PRId64 "] = %f, expected %f", c.name.c_str(), i, data[i], c.expected);
            }
        }
    }

    return true;
}

static bool backend_supports(ggml_backend_t backend, ggml_tensor * (*build)(ggml_context *, ggml_tensor *)) {
    ggml_init_params params = {
        /*.mem_size   =*/ 4*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    const bool ok = ggml_backend_supports_op(backend, build(ctx, a));

    ggml_free(ctx);

    return ok;
}

//helper struct for inputless splits, and constants like 0 and 1
struct backend_consts {
    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<ggml_tensor *>         zero;
    std::vector<ggml_tensor *>         one;

    backend_consts(const std::vector<ggml_backend_t> & backends, int64_t ne) {
        std::vector<float> data(ne);

        for (size_t b = 0; b < backends.size(); b++) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);

            ggml_tensor * z = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            ggml_format_name(z, "zero_%s", ggml_backend_name(backends[b]));

            ggml_tensor * o = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            ggml_format_name(o, "one_%s", ggml_backend_name(backends[b]));

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backends[b]);

            std::fill(data.begin(), data.end(), 0.0f);
            ggml_backend_tensor_set(z, data.data(), 0, ggml_nbytes(z));

            std::fill(data.begin(), data.end(), 1.0f);
            ggml_backend_tensor_set(o, data.data(), 0, ggml_nbytes(o));

            ctxs.push_back(ctx);
            bufs.push_back(buf);
            zero.push_back(z);
            one.push_back(o);
        }
    }

    ~backend_consts() {
        for (size_t b = 0; b < ctxs.size(); b++) {
            ggml_backend_buffer_free(bufs[b]);
            ggml_free(ctxs[b]);
        }
    }
};

// stress test: a tensor is incremented and sent back and forth between a backend and CPU in a ping-pong pattern
static bool stress_test_linked_list_cpu_device(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu, int n_nodes, int64_t tensor_len, bool use_device_host_buft) {

    const std::vector<ggml_backend_t> backends = { backend_gpu, backend_cpu };

    const size_t graph_size = n_nodes + 2; // see sched->hash_set FIXME, 2+ needed to account for leafs

    ggml_backend_sched_t sched = create_test_scheduler(backends, graph_size, use_device_host_buft);

    // the inputs are allocated separately so that they can be written before the graph is computed
    ggml_init_params params_static = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_static = ggml_init(params_static);

    ggml_tensor * x = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, tensor_len);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_tensor * one = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, tensor_len);
    ggml_set_name(one, "one");
    ggml_set_input(one);

    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, backend_cpu);

    std::vector<float> data(tensor_len);

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

    sched_graph g(ctx_compute, graph_size);

    ggml_tensor * out = x;
    for (int i = 0; i < n_nodes; i++) {
        out = g.add(ggml_add(ctx_compute, out, one), i % 2);
    }
    ggml_set_output(out);

    const bool ok = run_and_check(sched, g, backends, {{ out, float(n_nodes), "out" }}, tensor_len);

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);

    return ok;
}

// Same idea as stress_test_linked_list_cpu_device, but with the compute graph being a direct acyclic graph (DAG).
// n_lanes independent counters run through rounds of four shapes, so that one graph covers:
//  - lanes with different histories, so a misrouted copy shows up as a wrong count in a single lane
//  - one split producing two values followed by two splits that each consume one of them and do not depend on each other
//  - splits with no inputs at all, from lanes re-seeded out of constants that already live on the split's backend
//  - splits taking one activation per lane from the previous split, one of them from an older split instead
static bool stress_test_dag(const std::vector<ggml_backend_t> & backends, int n_lanes, int n_rounds,
        int64_t tensor_len, bool use_device_host_buft) {

    const int n_backends = (int) backends.size();

    GGML_ASSERT(n_lanes % 2 == 0);

    // sized for the worst case
    const size_t graph_size = n_lanes*(2*n_rounds + 1) + 2*n_backends;

    backend_consts consts(backends, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends, graph_size, use_device_host_buft);

    ggml_init_params params_compute = {
        /*.mem_size   =*/ (graph_size + 8)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);

    sched_graph g(ctx_compute, graph_size);

    std::vector<ggml_tensor *> lane(n_lanes);
    std::vector<int64_t>       value(n_lanes);
    std::vector<sched_check>   checks;

    // every lane starts at a different value, so that a misrouted copy changes a sum instead of going unnoticed
    auto seed = [&](int b, const char * prefix) {
        for (int l = 0; l < n_lanes; l++) {
            lane[l]  = g.add(ggml_add(ctx_compute, l == 0 ? consts.zero[b] : lane[l - 1], consts.one[b]), b);
            value[l] = l + 1;
            ggml_format_name(lane[l], "%s%d", prefix, l);
        }
    };

    auto check_lanes = [&]() {
        for (int l = 0; l < n_lanes; l++) {
            ggml_set_output(lane[l]);
            checks.push_back({ lane[l], float(value[l]), ggml_get_name(lane[l]) });
        }
    };

    seed(0, "seed");

    std::vector<ggml_tensor *> older       = lane;
    std::vector<int64_t>       value_older = value;

    for (int round = 0; round < n_rounds; round++) {
        // a backend the previous node did not use, so that the round is not absorbed into the split before it
        const int b = (g.backend_id.back() + 1) % n_backends;

        const std::vector<ggml_tensor *> prev       = lane;
        const std::vector<int64_t>       value_prev = value;

        switch (round % 5) {
            case 0:
                for (int p = 0; p < n_lanes/2; p++) {
                    const int la   = 2*p;
                    const int lb   = 2*p + 1;
                    const int prod = (b + p) % n_backends;
                    const int cons = (prod + 1) % n_backends;

                    // one split producing two values ...
                    for (int l : { la, lb }) {
                        lane[l] = g.add(ggml_add(ctx_compute, lane[l], consts.one[prod]), prod);
                        value[l]++;
                        ggml_format_name(lane[l], "r%d_prod%d", round, l);
                    }

                    // ... and two splits that each consume one of them, with no dependency on each other
                    lane[la] = g.add(ggml_add(ctx_compute, lane[la], consts.one[cons]), cons);
                    value[la]++;
                    ggml_format_name(lane[la], "r%d_cons%d", round, la);

                    lane[lb] = g.add(ggml_add(ctx_compute, lane[lb], consts.one[prod]), prod);
                    value[lb]++;
                    ggml_format_name(lane[lb], "r%d_cons%d", round, lb);
                }
                break;
            case 1:
                check_lanes();
                seed(b, "restart");
                break;
            case 4:
                lane[0]  = g.add(ggml_add(ctx_compute, prev[0], consts.one[b]), b);
                value[0] = value_prev[0] + 1;
                ggml_format_name(lane[0], "r%d_single", round);
                break;
            default:
                for (int l = 0; l < n_lanes; l++) {
                    const int  partner    = (l + 1 + round) % n_lanes;
                    const bool reach_back = round >= 2 && l == round % n_lanes;

                    ggml_tensor * a  = reach_back ? older[l]       : prev[l];
                    const int64_t va = reach_back ? value_older[l] : value_prev[l];

                    lane[l]  = g.add(ggml_add(ctx_compute, a, prev[partner]), b);
                    value[l] = va + value_prev[partner];
                    ggml_format_name(lane[l], "r%d_lane%d", round, l);
                }
                break;
        }

        older       = prev;
        value_older = value_prev;
    }

    check_lanes();

    for (const sched_check & c : checks) {
        GGML_ASSERT(c.expected < float(1 << 24)); // the counts have to stay exact in f32
    }

    const bool ok = run_and_check(sched, g, backends, checks, tensor_len);

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);

    return ok;
}

// Created in response to https://github.com/ggml-org/llama.cpp/issues/23321
// Currently, input copies to splits synchronize split execution.
// CPU split N+1 could overwrite output of split N if this was not copied to backend in time.
// Mechanism to test: GPU split with long sleep to clog CUDA stream / vk command queue,
// followed by CPU splits without input, provoking a race condition in the current setup.
//   GPU: 100ms sleep to clog stream/queue
//   CPU: produce {55}
//   GPU: request copy of {55} in CUDA stream, ADD {44}; Output should be {99}
//   CPU: produce {66}
//   GPU: increment both {99} and {66} -> {100} and {67}
//   Correct result is thus {100}, incorrect output is {111} when {55} was overwritten by {66} before the copy started.
// Note: currently only reproducible on async H2D copy (=pinned memory).
static bool test_inputless_splits_scheduling(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu, int64_t tensor_len, int32_t sleep_us, bool use_device_host_buft) {

    const int GPU = 0;
    const int CPU = 1;

    const std::vector<ggml_backend_t> backends = { backend_gpu, backend_cpu };

    const size_t graph_size = 64;

    ggml_backend_sched_t sched = create_test_scheduler(backends, graph_size, use_device_host_buft);

    ggml_init_params params_static = {
        /*.mem_size   =*/ 4*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx_cpu = ggml_init(params_static);

    ggml_tensor * zero_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(zero_cpu, "zero_cpu");

    ggml_tensor * val55_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(val55_cpu, "val55_cpu");

    ggml_tensor * val66_cpu = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(val66_cpu, "val66_cpu");

    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);

    ggml_context * ctx_gpu = ggml_init(params_static);

    ggml_tensor * zero_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(zero_gpu, "zero_gpu");

    ggml_tensor * one_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(one_gpu, "one_gpu");

    ggml_tensor * val44_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(val44_gpu, "val44_gpu");

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);

    std::vector<float> data(tensor_len);

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

    sched_graph g(ctx_compute, graph_size);

    // GPU: add then sleep - occupies the GPU while later splits run
    ggml_tensor * delayed = g.add(ggml_add(ctx_compute, zero_gpu, one_gpu), GPU);
    delayed = g.add(ggml_sleep(ctx_compute, delayed, sleep_us), GPU);

    // CPU: 0 + 55 -> 55
    ggml_tensor * v55 = g.add(ggml_add(ctx_compute, zero_cpu, val55_cpu), CPU);

    // GPU: 55 + 44 -> 99
    ggml_tensor * v99 = g.add(ggml_add(ctx_compute, v55, val44_gpu), GPU);

    // CPU: 0 + 66 -> 66
    ggml_tensor * v66 = g.add(ggml_add(ctx_compute, zero_cpu, val66_cpu), CPU);

    // GPU: increment both of the previous split outputs
    ggml_tensor * out99 = g.add(ggml_add(ctx_compute, v99, one_gpu), GPU);
    ggml_tensor * out66 = g.add(ggml_add(ctx_compute, v66, one_gpu), GPU);

    ggml_set_output(delayed);
    ggml_set_output(out99);
    ggml_set_output(out66);

    const bool ok = run_and_check(sched, g, backends, {
        { delayed, 1.0f,   "delayed" },
        { out99,   100.0f, "out99"   },
        { out66,   67.0f,  "out66"   },
    }, tensor_len);

    // the race needs the allocator to reuse the memory of the first CPU split for the second one
    note("v55 and v66 %s (v55=%p v66=%p)", v55->data == v66->data ? "alias" : "do not alias", v55->data, v66->data);

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);
    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_cpu);

    return ok;
}

// Test that all async backends transmit their activations to the following async backend correctly. No user inputs tested.
static bool test_chain_all_backends(const std::vector<ggml_backend_t> & backends, int64_t tensor_len, bool use_device_host_buft) {

    const int n_backends = (int) backends.size();

    std::vector<int> seq;

    auto push = [&](int b) {
        if (seq.empty() || seq.back() != b) {
            seq.push_back(b);
        }
    };

    for (int i = 0; i < n_backends; i++) {
        for (int j = 0; j < n_backends; j++) {
            if (i != j) {
                push(i);
                push(j);
            }
        }
    }

    const size_t n_nodes    = seq.size();
    const size_t graph_size = n_nodes + n_backends + 1; // see sched->hash_set FIXME, one "one" per backend plus the "zero" that starts the chain

    backend_consts consts(backends, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends, graph_size, use_device_host_buft);

    ggml_init_params params_compute = {
        /*.mem_size   =*/ (graph_size + 8)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);

    sched_graph g(ctx_compute, graph_size);

    ggml_tensor * out = nullptr;
    for (size_t k = 0; k < n_nodes; k++) {
        const int b = seq[k];

        out = g.add(ggml_add(ctx_compute, k == 0 ? consts.zero[b] : out, consts.one[b]), b);
    }
    ggml_set_output(out);

    const bool ok = run_and_check(sched, g, backends, {{ out, float(n_nodes), "out" }}, tensor_len);

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);

    return ok;
}

// Tests data transfer between all combinations of backend pairs
// Always tests between two backends only with a single activation and 4 parallel user inputs.
static bool test_pair_user_inputs(const std::vector<ggml_backend_t> & backends, int b_send, int b_recv, int64_t tensor_len,
        int n_inputs, bool inputs_on_sender, bool parallel, bool use_device_host_buft) {


    const size_t graph_size = 64;

    backend_consts consts(backends, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends, graph_size, use_device_host_buft, parallel);

    // placing the inputs on the sender makes the receiving split copy all of them, placing them on the
    // receiver leaves the activation as its only input
    const int b_inputs = inputs_on_sender ? b_send : b_recv;

    ggml_init_params params_inputs = {
        /*.mem_size   =*/ n_inputs*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_inputs = ggml_init(params_inputs);

    std::vector<ggml_tensor *> inputs;
    for (int k = 0; k < n_inputs; k++) {
        ggml_tensor * in = ggml_new_tensor_1d(ctx_inputs, GGML_TYPE_F32, tensor_len);
        ggml_format_name(in, "in%d", k);
        ggml_set_input(in);
        inputs.push_back(in);
    }

    ggml_backend_buffer_t buf_inputs = ggml_backend_alloc_ctx_tensors(ctx_inputs, backends[b_inputs]);

    std::vector<float> data(tensor_len);
    std::fill(data.begin(), data.end(), 1.0f);
    for (ggml_tensor * in : inputs) {
        ggml_backend_tensor_set(in, data.data(), 0, ggml_nbytes(in));
    }

    ggml_init_params params_compute = {
        /*.mem_size   =*/ (graph_size + 8)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);

    sched_graph g(ctx_compute, graph_size);

    ggml_tensor * out = g.add(ggml_add(ctx_compute, consts.zero[b_send], consts.one[b_send]), b_send);

    for (ggml_tensor * in : inputs) {
        out = g.add(ggml_add(ctx_compute, out, in), b_recv);
    }
    ggml_set_output(out);

    bool ok = run_and_check(sched, g, backends, {{ out, float(1 + n_inputs), "out" }}, tensor_len);

    for (ggml_tensor * in : inputs) {
        if (!ok) {
            break;
        }
        ggml_backend_tensor_get(in, data.data(), 0, ggml_nbytes(in));
        for (int64_t i = 0; i < tensor_len; i++) {
            if (data[i] != 1.0f) {
                ok = fail("%s[%" PRId64 "] = %f after compute, expected 1.000000 - the input was modified", ggml_get_name(in), i, data[i]);
                break;
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);
    ggml_backend_buffer_free(buf_inputs);
    ggml_free(ctx_inputs);

    return ok;
}

// Tests Y-shaped scheduling: two parallel lanes merging into 1. Lane A and B, merging into a single split.
// The lanes join in a final split on b_send that receives one activation from each of the two b_recv splits.
// TODO improve function signature + hoist backend_consts out of it?
static bool test_y_shaped_graph(const std::vector<ggml_backend_t> & backends, int backend_a, int backend_b, int64_t tensor_len,
        bool use_device_host_buft) {


    const size_t graph_size = 64;

    backend_consts consts(backends, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends, graph_size, use_device_host_buft);

    ggml_init_params params_compute = {
        /*.mem_size   =*/ (graph_size + 8)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);

    sched_graph g(ctx_compute, graph_size);

    ggml_tensor * seed_a = g.add(ggml_add(ctx_compute, consts.zero[backend_a], consts.one[backend_a]), backend_a);
    ggml_set_name(seed_a, "seed_a");

    ggml_tensor * lane_a = g.add(ggml_add(ctx_compute, seed_a, consts.one[backend_b]), backend_b);
    ggml_set_name(lane_a, "lane_a");

    ggml_tensor * seed_b = g.add(ggml_add(ctx_compute, consts.zero[backend_a], consts.one[backend_a]), backend_a);
    ggml_set_name(seed_b, "seed_b");

    ggml_tensor * lane_b = g.add(ggml_add(ctx_compute, seed_b, consts.one[backend_b]), backend_b);
    ggml_set_name(lane_b, "lane_b");

    // neither lane ends on b_send, so both activations are inputs of the last split
    ggml_tensor * out = g.add(ggml_add(ctx_compute, lane_a, lane_b), backend_a);
    ggml_set_name(out, "out");
    ggml_set_output(out);

    const bool ok = run_and_check(sched, g, backends, {{ out, 4.0f, "out" }}, tensor_len);

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);

    return ok;
}

static bool initialize_gpu_backends(std::vector<ggml_backend_t> & backends, bool & have_device_host_buft, bool & have_sleep) {

    // cf. GGML_SCHED_MAX_BACKENDS
    const size_t max_backends = 16;

    for (size_t i = 0; i < ggml_backend_dev_count() && backends.size() + 1 < max_backends; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        //GPU or IGPU only, for now
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            printf("failed to initialize %s, skipping it\n", ggml_backend_dev_name(dev));
            continue;
        }

        backends.push_back(backend);
    }

    if (backends.empty()) {
        printf("no GPU device found, skipping\n");
        return 0;
    }

    // ggml_backend_sched_new requires the CPU backend to be the last one
    backends.push_back(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(backends.back() != nullptr);
    // todo rework this, only selects a single GPU
    // create struct incorporating have_device_host_buft and have_sleep flags
    ggml_backend_t backend_gpu = backends[0];
    ggml_backend_t backend_cpu = backends.back();

    for (ggml_backend_t backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        printf("backend: %-10s (%s)\n", ggml_backend_name(backend), ggml_backend_dev_description(dev));
    }

    have_device_host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend_gpu)) != nullptr;
    if (!have_device_host_buft) {
        printf("GPU has no host buffer type, the device_host cases are skipped\n");
    }

    have_sleep = backend_supports(backend_gpu, [](ggml_context * ctx, ggml_tensor * a) { return ggml_sleep(ctx, a, 0); });
    if (!have_sleep) {
        printf("GPU does not support GGML_OP_SLEEP, some tests are skipped\n");
    }
    printf("\n");
    return 1;
}


int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends; // TODO merge into single struct
    bool have_device_host_buft = false;
    bool have_sleep = false;
    bool gpu_initialized = initialize_gpu_backends(backends, have_device_host_buft, have_sleep);
    ggml_backend_t backend_gpu = backends[0];
    ggml_backend_t backend_cpu = backends.back();

    if (!gpu_initialized) {
        return 0;
    }

    for (bool use_device_host_buft : { false, true }) {
        if (use_device_host_buft && !have_device_host_buft) {
            continue;
        }

        printf("=== CPU sched buft: %s ===\n\n", use_device_host_buft ? "device_host" : "pageable");

        for (int n_nodes : { 2, 5, 128, 1024 }) {
            for (int tensor_len : { 2, 4096 }) {
                case_begin("test_linked_list       n_nodes  = %4d, tensor_len = %4d", n_nodes, tensor_len);
                case_end(stress_test_linked_list_cpu_device(backend_gpu, backend_cpu, n_nodes, tensor_len, use_device_host_buft));
            }
        }

        printf("\n");

        for (int n_lanes : { 8, 16 }) {
            for (int n_rounds : { 5, 13, 64 }) {
                for (int tensor_len : { 2, 4096 }) {
                    case_begin("test_dag        n_lanes = %2d, n_rounds = %2d, tensor_len = %4d", n_lanes, n_rounds, tensor_len);
                    case_end(stress_test_dag(backends, n_lanes, n_rounds, tensor_len, use_device_host_buft));
                }
            }
        }

        printf("\n");

        if (have_sleep) {
            for (int32_t sleep_us : {50, 60, 70, 80,  400, 500, 600, 700, 1000, 10000}) {
                for (int tensor_len : { 2, 2048, 4096, 8192, 1600}) {
                    case_begin("test_inputless_splits_scheduling      sleep_us = %6d, tensor_len = %4d", sleep_us, tensor_len);
                    case_end(test_inputless_splits_scheduling(backend_gpu, backend_cpu, tensor_len, sleep_us, use_device_host_buft));
                }
            }

            printf("\n");
        }

        for (int tensor_len : { 2, 4096 }) {
            case_begin("test_chain_all_backends tensor_len = %4d", tensor_len);
            case_end(test_chain_all_backends(backends, tensor_len, use_device_host_buft));
        }

        printf("\n");

        // every ordered pair of backends is the sender and the receiver of a copy
        for (size_t b_send = 0; b_send < backends.size(); b_send++) {
            for (size_t b_recv = 0; b_recv < backends.size(); b_recv++) {
                if (b_send == b_recv) {
                    continue;
                }

                const char * name_send = ggml_backend_name(backends[b_send]);
                const char * name_recv = ggml_backend_name(backends[b_recv]);

                for (bool inputs_on_sender : { false, true }) {
                    for (bool parallel : { false, true }) {
                        for (int tensor_len : { 1, 4096 }) {
                            // todo aendk fix up.
                            case_begin("test_pair_user_inputs     %-8s -> %-8s inputs on %-8s parallel = %d, tensor_len = %4d",
                                    name_send, name_recv, inputs_on_sender ? "sender" : "receiver", parallel, tensor_len);
                            case_end(test_pair_user_inputs(backends, (int) b_send, (int) b_recv, tensor_len,
                                    /*n_inputs =*/ 4, inputs_on_sender, parallel, use_device_host_buft));
                        }
                    }
                }

                for (int tensor_len : { 1, 4096 }) {
                    case_begin("test_y_shaped        %-8s -> %-8s tensor_len = %4d", name_send, name_recv, tensor_len);
                    case_end(test_y_shaped_graph(backends, (int) b_send, (int) b_recv, tensor_len, use_device_host_buft));
                }
            }
        }

        printf("\n");
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }

    printf("%d/%d tests passed\n", n_ok, n_test);

    return n_ok == n_test ? 0 : 1;
}
