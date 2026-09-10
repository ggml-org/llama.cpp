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
#include <cstring>
#include <string>
#include <vector>


// global counters for pass/fail
static int n_ok   = 0;
static int n_test = 0;

// default silent on success; --verbose / -v prints progress
static bool print_log = false;

GGML_ATTRIBUTE_FORMAT(1, 2)
static void log_maybe(const char * fmt, ...) {
    if (!print_log) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

// pretty-print helpers start
static constexpr int status_column = 82; // for ok/fail column alignment
static int  case_len = 0;
static char case_label[256]; // max length of a case label

static void case_end(bool ok) {
    n_ok += ok;
    n_test++;

    if (ok && !print_log) {
        case_len = 0;
        return;
    }

    if (case_len == 0) {
        case_len = printf("  %s", case_label);
    }

    printf("%*s%s\n", std::max(1, status_column - case_len), "", ok ? "OK" : "FAIL");
    case_len = 0;
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
    if (!print_log) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    vnote(fmt, args);
    va_end(args);
}

GGML_ATTRIBUTE_FORMAT(1, 2)
static bool fail(const char * fmt, ...) {
    if (case_len == 0) {
        case_len = printf("  %s", case_label);
    }

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

    if (print_log) {
        case_len = printf("  %s", case_label);
        fflush(stdout);
    } else {
        case_len = 0;
    }
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

// a backend together with the capabilities that decide which tests can run on it
struct sched_backend_caps {
    ggml_backend_t backend               = nullptr;
    bool           have_device_host_buft = false;
    bool           have_sleep            = false;
};

static ggml_backend_sched_t create_test_scheduler(const std::vector<sched_backend_caps> & backends_w_caps, size_t graph_size,
        bool use_device_host_buft, bool parallel = false) {

    std::vector<ggml_backend_t>             backend_handles(backends_w_caps.size());
    std::vector<ggml_backend_buffer_type_t> bufts(backends_w_caps.size());
    for (size_t b = 0; b < backends_w_caps.size(); b++) {
        backend_handles[b] = backends_w_caps[b].backend;
        bufts[b]   = ggml_backend_get_default_buffer_type(backends_w_caps[b].backend);
    }
    // sets either pinned or paged memory
    bufts.back() = sched_cpu_buft(backends_w_caps[0].backend, backends_w_caps.back().backend, use_device_host_buft);

    return ggml_backend_sched_new(backend_handles.data(), bufts.data(), (int) backend_handles.size(), graph_size, parallel, /*op_offload =*/ false);
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
        const std::vector<sched_backend_caps> & backends_w_caps, const std::vector<sched_check> & checks, int64_t ne) {

    int n_splits_expected = g.nodes.empty() ? 0 : 1;
    for (size_t i = 1; i < g.backend_id.size(); i++) {
        n_splits_expected += g.backend_id[i] != g.backend_id[i - 1];
    }

    ggml_backend_sched_reset(sched);
    for (size_t i = 0; i < g.nodes.size(); i++) {
        ggml_backend_sched_set_tensor_backend(sched, g.nodes[i], backends_w_caps[g.backend_id[i]].backend);
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

    backend_consts(const std::vector<sched_backend_caps> & backends_w_caps, int64_t ne) {
        std::vector<float> data(ne);

        for (size_t b = 0; b < backends_w_caps.size(); b++) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);

            ggml_tensor * z = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            ggml_format_name(z, "zero_%s", ggml_backend_name(backends_w_caps[b].backend));

            ggml_tensor * o = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            ggml_format_name(o, "one_%s", ggml_backend_name(backends_w_caps[b].backend));

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backends_w_caps[b].backend);

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

// Creates a vector which holds a chain of backend ids. There, every backend is sender to and receiver from every other backend, to test all data transfers.
static std::vector<int> create_all_to_all_chain(int n_backends) {
    std::vector<int> chain;
    for (int i = 0; i < n_backends; i++) {
        for (int j = i + 1; j < n_backends; j++) {
            chain.push_back(i);
            chain.push_back(j);
        }
    }
    return chain;
}

// stress test: a tensor is incremented and sent back and forth between the backends in a ping-pong pattern.
// One lap tests data transfers from any backend to any other backend, with and without user inputs.
static bool stress_test_linked_list(const std::vector<sched_backend_caps> & backends_w_caps, int n_laps,
        int64_t tensor_len, bool use_device_host_buft) {
    const int n_backends = (int) backends_w_caps.size();

    const std::vector<int> chain     = create_all_to_all_chain(n_backends);
    const int              chain_len = (int) chain.size();

    // one node seeds the chain, two passes over the chain per lap follow. the extra node closes the
    // wrap-around pair of every pass
    const int n_nodes = 2*chain_len*n_laps + 1;

    // see sched->hash_set FIXME, the set has to hold the nodes and the leafs: two constants per backend
    // plus one user input per node of a second pass
    const size_t graph_size = n_nodes + 2*n_backends + chain_len*n_laps;

    backend_consts consts(backends_w_caps, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends_w_caps, graph_size, use_device_host_buft);

    // the user input of a second pass node is placed on the backend that sends it the activation, so that the
    // receiving split has to copy both of them
    std::vector<std::vector<int>> inputs_of(n_backends);
    std::vector<int>              input_value(n_nodes, 0);

    for (int i = 1; i < n_nodes; i++) {
        const int pass = (i - 1)/chain_len;
        if (pass % 2 == 0) {
            continue;
        }

        inputs_of[chain[(i - 1) % chain_len]].push_back(i);
        input_value[i] = (int) (chain_len*(pass/2) + (i - 1) % chain_len + 1);
    }

    // the inputs are allocated separately so that they can be written before the graph is computed
    std::vector<ggml_context *>        ctxs_input(n_backends);
    std::vector<ggml_backend_buffer_t> bufs_input(n_backends);
    std::vector<ggml_tensor *>         input(n_nodes, nullptr);

    std::vector<float> data(tensor_len);

    for (int b = 0; b < n_backends; b++) {
        ggml_init_params params_input = {
            /*.mem_size   =*/ inputs_of[b].size()*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctxs_input[b] = ggml_init(params_input);

        for (int i : inputs_of[b]) {
            input[i] = ggml_new_tensor_1d(ctxs_input[b], GGML_TYPE_F32, tensor_len);
            ggml_format_name(input[i], "input%d", i);
            ggml_set_input(input[i]);
        }

        bufs_input[b] = ggml_backend_alloc_ctx_tensors(ctxs_input[b], backends_w_caps[b].backend);

        for (int i : inputs_of[b]) {
            std::fill(data.begin(), data.end(), float(input_value[i]));
            ggml_backend_tensor_set(input[i], data.data(), 0, ggml_nbytes(input[i]));
        }
    }

    ggml_init_params params_compute = {
        /*.mem_size   =*/ (n_nodes + 2)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);

    sched_graph g(ctx_compute, graph_size);

    int64_t       sum = 0;
    ggml_tensor * out = nullptr;

    for (int i = 0; i < n_nodes; i++) {
        const int b = chain[i % chain_len];

        if (i == 0) {
            out  = ggml_add(ctx_compute, consts.zero[b], consts.one[b]);
            sum += 1;
        } else if (input[i] == nullptr) {
            out  = ggml_add(ctx_compute, out, consts.one[b]);
            sum += 1;
        } else {
            out  = ggml_add(ctx_compute, out, input[i]);
            sum += input_value[i];
        }

        g.add(out, b);
    }
    ggml_set_output(out);

    bool ok = sum < (1 << 24); // the counter has to stay exact in f32
    if (!ok) {
        fail("expected result %" PRId64 " does not fit into the f32 mantissa, lower n_laps", sum);
    } else {
        ok = run_and_check(sched, g, backends_w_caps, {{ out, float(sum), "out" }}, tensor_len);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);
    for (int b = 0; b < n_backends; b++) {
        ggml_backend_buffer_free(bufs_input[b]);
        ggml_free(ctxs_input[b]);
    }

    return ok;
}

// Same idea as stress_test_linked_list, but with the compute graph being a direct acyclic graph (DAG).
// n_lanes independent counters run through rounds of four shapes, so that one graph covers:
//  - lanes with different histories, so a misrouted copy shows up as a wrong count in a single lane
//  - one split producing two values followed by two splits that each consume one of them and do not depend on each other
//  - splits with no inputs at all, from lanes re-seeded out of constants that already live on the split's backend
//  - splits taking one activation per lane from the previous split, one of them from an older split instead
static bool stress_test_dag(const std::vector<sched_backend_caps> & backends_w_caps, int n_lanes, int n_rounds,
        int64_t tensor_len, bool use_device_host_buft) {

    const int n_backends = (int) backends_w_caps.size();

    GGML_ASSERT(n_lanes % 2 == 0);

    // sized for the worst case
    const size_t graph_size = n_lanes*(2*n_rounds + 1) + 2*n_backends;

    backend_consts consts(backends_w_caps, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends_w_caps, graph_size, use_device_host_buft);

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

    const bool ok = run_and_check(sched, g, backends_w_caps, checks, tensor_len);

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
static bool test_inputless_splits_scheduling(const sched_backend_caps & gpu, const sched_backend_caps & cpu, int64_t tensor_len, int32_t sleep_us, bool use_device_host_buft) {

    const int GPU = 0;
    const int CPU = 1;

    const std::vector<sched_backend_caps> backends_w_caps = { gpu, cpu };

    const size_t graph_size = 64;

    ggml_backend_sched_t sched = create_test_scheduler(backends_w_caps, graph_size, use_device_host_buft);

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

    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, cpu.backend);

    ggml_context * ctx_gpu = ggml_init(params_static);

    ggml_tensor * zero_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(zero_gpu, "zero_gpu");

    ggml_tensor * one_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(one_gpu, "one_gpu");

    ggml_tensor * val44_gpu = ggml_new_tensor_1d(ctx_gpu, GGML_TYPE_F32, tensor_len);
    ggml_set_name(val44_gpu, "val44_gpu");

    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, gpu.backend);

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

    const bool ok = run_and_check(sched, g, backends_w_caps, {
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
static bool test_chain_all_backends(const std::vector<sched_backend_caps> & backends_w_caps, int64_t tensor_len, bool use_device_host_buft) {

    const int n_backends = (int) backends_w_caps.size();

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

    backend_consts consts(backends_w_caps, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends_w_caps, graph_size, use_device_host_buft);

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

    const bool ok = run_and_check(sched, g, backends_w_caps, {{ out, float(n_nodes), "out" }}, tensor_len);

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);

    return ok;
}

// Tests data transfer between all combinations of backend pairs
// Always tests between two backends only with a single activation and 4 parallel user inputs.
static bool test_pair_user_inputs(const std::vector<sched_backend_caps> & backends_w_caps, int b_send, int b_recv, int64_t tensor_len,
        int n_inputs, bool inputs_on_sender, bool parallel, bool use_device_host_buft) {

    const size_t graph_size = 64;

    backend_consts consts(backends_w_caps, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends_w_caps, graph_size, use_device_host_buft, parallel);

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

    ggml_backend_buffer_t buf_inputs = ggml_backend_alloc_ctx_tensors(ctx_inputs, backends_w_caps[b_inputs].backend);

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

    bool ok = run_and_check(sched, g, backends_w_caps, {{ out, float(1 + n_inputs), "out" }}, tensor_len);

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
static bool test_y_shaped_graph(const std::vector<sched_backend_caps> & backends_w_caps, int backend_a, int backend_b, int64_t tensor_len,
        bool use_device_host_buft) {


    const size_t graph_size = 64;

    backend_consts consts(backends_w_caps, tensor_len);

    ggml_backend_sched_t sched = create_test_scheduler(backends_w_caps, graph_size, use_device_host_buft);

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

    const bool ok = run_and_check(sched, g, backends_w_caps, {{ out, 4.0f, "out" }}, tensor_len);

    ggml_backend_sched_free(sched);
    ggml_free(ctx_compute);

    return ok;
}

static const char * dev_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
    }
    return "UNKNOWN";
}

static bool initialize_backends(std::vector<sched_backend_caps> & backends_w_caps) {

    // cf. GGML_SCHED_MAX_BACKENDS
    const size_t max_backends = 16;

    for (size_t i = 0; i < ggml_backend_dev_count() && backends_w_caps.size() + 1 < max_backends; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        log_maybe("device %2zu: %-10s %-5s (%s)\n", i, ggml_backend_dev_name(dev),
                dev_type_name(ggml_backend_dev_type(dev)), ggml_backend_dev_description(dev));

        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            printf("failed to initialize %s, skipping it\n", ggml_backend_dev_name(dev));
            continue;
        }

        backends_w_caps.push_back({ backend });
    }

    if (backends_w_caps.empty()) {
        printf("no non-CPU backend found, skipping\n");
        return 0;
    }

    // ggml_backend_sched_new requires the CPU backend to be the last one
    backends_w_caps.push_back({ ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) });
    GGML_ASSERT(backends_w_caps.back().backend != nullptr);

    for (sched_backend_caps & sched_back_caps : backends_w_caps) {
        ggml_backend_dev_t dev = ggml_backend_get_device(sched_back_caps.backend);

        sched_back_caps.have_device_host_buft = ggml_backend_dev_host_buffer_type(dev) != nullptr;
        sched_back_caps.have_sleep = backend_supports(sched_back_caps.backend, [](ggml_context * ctx, ggml_tensor * a) { return ggml_sleep(ctx, a, 0); });

        log_maybe("backend: %-10s host_buft = %d, sleep = %d (%s)\n", ggml_backend_name(sched_back_caps.backend),
                sched_back_caps.have_device_host_buft, sched_back_caps.have_sleep, ggml_backend_dev_description(dev));
    }
    log_maybe("\n");
    return 1;
}


int main(int argc, char ** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            print_log = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-v|--verbose]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [-v|--verbose]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();

    std::vector<sched_backend_caps> backends_w_caps;
    bool non_cpu_backends_initialized = initialize_backends(backends_w_caps);

    // nothing to test with synchronous CPU backend only
    if (!non_cpu_backends_initialized || backends_w_caps.size() < 2) {
        return 0;
    }

    const sched_backend_caps & cpu  = backends_w_caps.back();
    const size_t n_non_cpu_backends = backends_w_caps.size() - 1;

    // Test cases for single backends against CPU backend
    for (size_t b_id = 0; b_id < n_non_cpu_backends; b_id++) {
        const sched_backend_caps & backend_w_caps      = backends_w_caps[b_id];
        const char *               name_backend = ggml_backend_name(backend_w_caps.backend);

        for (bool use_device_host_buft : { false, true }) {
            if (use_device_host_buft && !backend_w_caps.have_device_host_buft) {
                continue;
            }

            log_maybe("=== Backend: %s, CPU sched buft: %s ===\n\n", name_backend, use_device_host_buft ? "device_host" : "pageable");

            if (backend_w_caps.have_sleep) {
                for (int32_t sleep_us : {50, 60, 70, 80,  400, 500, 600, 700, 1000, 10000}) {
                    for (int tensor_len : { 2, 2048, 4096, 8192, 1600}) {
                        case_begin("test_inputless_splits_scheduling      %-8s sleep_us = %6d, tensor_len = %4d", name_backend, sleep_us, tensor_len);
                        case_end(test_inputless_splits_scheduling(backend_w_caps, cpu, tensor_len, sleep_us, use_device_host_buft));
                    }
                }

                log_maybe("\n");
            }
        }
    }

    // Test cases for all present backend subsets
    for (bool use_device_host_buft : { false, true }) {
        // create_test_scheduler takes the host buffer type from the first backend
        if (use_device_host_buft && !backends_w_caps[0].have_device_host_buft) {
            continue;
        }

        log_maybe("=== all backends, CPU sched buft: %s ===\n\n", use_device_host_buft ? "device_host" : "pageable");

        for (int n_lanes : { 8, 16 }) {
            for (int n_rounds : { 5, 13, 64 }) {
                for (int tensor_len : { 2, 4096 }) {
                    case_begin("stress_test_dag        n_lanes = %2d, n_rounds = %2d, tensor_len = %4d", n_lanes, n_rounds, tensor_len);
                    case_end(stress_test_dag(backends_w_caps, n_lanes, n_rounds, tensor_len, use_device_host_buft));
                }
            }
        }

        log_maybe("\n");

        for (int tensor_len : { 2, 4096 }) {
            case_begin("test_chain_all_backends tensor_len = %4d", tensor_len);
            case_end(test_chain_all_backends(backends_w_caps, tensor_len, use_device_host_buft));
        }

        log_maybe("\n");

        for (int n_laps : { 1, 4, 64 }) {
            for (int tensor_len : { 2, 4096 }) {
                case_begin("stress_test_linked_list n_laps = %3d, tensor_len = %4d", n_laps, tensor_len);
                case_end(stress_test_linked_list(backends_w_caps, n_laps, tensor_len, use_device_host_buft));
            }
        }

        log_maybe("\n");

        // every ordered pair of backends is the sender and the receiver of a copy
        for (size_t b_send = 0; b_send < backends_w_caps.size(); b_send++) {
            for (size_t b_recv = 0; b_recv < backends_w_caps.size(); b_recv++) {
                if (b_send == b_recv) {
                    continue;
                }

                const char * name_send = ggml_backend_name(backends_w_caps[b_send].backend);
                const char * name_recv = ggml_backend_name(backends_w_caps[b_recv].backend);

                for (bool inputs_on_sender : { false, true }) {
                    for (bool parallel : { false, true }) {
                        for (int tensor_len : { 1, 4096 }) {
                            // todo aendk fix up.
                            case_begin("test_pair_user_inputs     %-8s -> %-8s inputs on %-8s parallel = %d, tensor_len = %4d",
                                    name_send, name_recv, inputs_on_sender ? "sender" : "receiver", parallel, tensor_len);
                            case_end(test_pair_user_inputs(backends_w_caps, (int) b_send, (int) b_recv, tensor_len,
                                    /*n_inputs =*/ 4, inputs_on_sender, parallel, use_device_host_buft));
                        }
                    }
                }

                for (int tensor_len : { 1, 4096 }) {
                    case_begin("test_y_shaped        %-8s -> %-8s tensor_len = %4d", name_send, name_recv, tensor_len);
                    case_end(test_y_shaped_graph(backends_w_caps, (int) b_send, (int) b_recv, tensor_len, use_device_host_buft));
                }
            }
        }

        log_maybe("\n");
    }

    for (const sched_backend_caps & caps : backends_w_caps) {
        ggml_backend_free(caps.backend);
    }

    if (print_log || n_ok != n_test) {
        printf("%d/%d tests passed\n", n_ok, n_test);
    }

    return n_ok == n_test ? 0 : 1;
}
