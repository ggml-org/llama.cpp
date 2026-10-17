// decode trace (ggml-trace.h) on CUDA through the scheduler: a traced step runs node by node instead of replaying a
// CUDA graph. regression test for the bypass discarding a recapture that a traced step asked for, after which the
// next untraced step replayed the graph captured for an older shape: every step's output must equal the output of
// the same sequence run without the trace, also when the shape changes right before or inside a traced window.
// also checks that the traced steps produced device timings (OP_GPU) and scheduler events

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-trace.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #c); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

static void log_cb(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static constexpr int N = 512;

struct step_desc {
    int  n_cols; // the shape: columns of the activation (1 or 2 tokens)
    bool traced;
};

// the same steps, with or without the trace; one fresh set of backends per run so that no CUDA graph is shared
static std::vector<std::vector<float>> run(const std::vector<step_desc> & steps, const char * trace_path) {
    std::vector<std::vector<float>> out;

    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!gpu || !cpu) {
        fprintf(stderr, "no CUDA / CPU backend\n");
        exit(1);
    }

    ggml_init_params wp = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * cw = ggml_init(wp);
    ggml_tensor * wq   = ggml_new_tensor_2d(cw, GGML_TYPE_F32, N, N); ggml_set_name(wq,   "blk.3.attn_q.weight");
    ggml_tensor * wout = ggml_new_tensor_2d(cw, GGML_TYPE_F32, N, N); ggml_set_name(wout, "output.weight");
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, gpu);
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-0.05f, 0.05f);
    std::vector<float> w(N*N);
    for (ggml_tensor * t : { wq, wout }) {
        for (auto & v : w) {
            v = d(rng);
        }
        ggml_backend_tensor_set(t, w.data(), 0, ggml_nbytes(t));
    }

    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false, true);

    if (trace_path) {
        ggml_trace_params tp = {};
        tp.path = trace_path; tp.n_tokens = 0; tp.n_skip = 0; tp.max_batch = 1; tp.detail = 1;
        tp.poll_us = 1000; tp.buffer_mb = 16; tp.residency_valid = true; tp.meta = nullptr;
        if (!ggml_trace_start(&tp)) {
            fprintf(stderr, "trace start failed\n");
            exit(1);
        }
    }

    // the graph is rebuilt in the same memory every step, so that its nodes keep their addresses (as llama.cpp's
    // graph reuse does) and the CUDA graph of the split is the same one, only its properties change
    std::vector<uint8_t> mem(ggml_tensor_overhead()*16 + ggml_graph_overhead());
    for (size_t s = 0; s < steps.size(); ++s) {
        const step_desc & sd = steps[s];
        ggml_init_params gp = { mem.size(), mem.data(), true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, sd.n_cols); ggml_set_name(x, "inp_embd"); ggml_set_input(x);
        ggml_tensor * q  = ggml_mul_mat(ctx, wq, x);    ggml_set_name(q,  "Qcur-3");
        ggml_tensor * q2 = ggml_add(ctx, q, q);         ggml_set_name(q2, "attn_out-3");
        ggml_tensor * q3 = ggml_scale(ctx, q2, 0.5f);   ggml_set_name(q3, "ffn_out-3");
        ggml_tensor * o  = ggml_mul_mat(ctx, wout, q3); ggml_set_name(o,  "result_output"); ggml_set_output(o);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);

        std::vector<float> xin(N*sd.n_cols);
        for (size_t i = 0; i < xin.size(); ++i) {
            xin[i] = 0.01f*(float) ((i*7 + s*13) % 101) - 0.5f;
        }

        if (trace_path) {
            // untraced steps are prompt-sized batches: SUSPENDED, CUDA graphs as usual
            ggml_trace_batch_begin(sd.traced ? 1 : 40, (int32_t) s, (int32_t) s);
            CHECK(ggml_trace_active() == sd.traced, "step %zu state", s);
        }
        ggml_backend_sched_reset(sched);
        CHECK(ggml_backend_sched_alloc_graph(sched, gf), "alloc");
        ggml_backend_tensor_set(x, xin.data(), 0, ggml_nbytes(x));
        CHECK(ggml_backend_sched_graph_compute_async(sched, gf) == GGML_STATUS_SUCCESS, "compute");
        std::vector<float> r(ggml_nelements(o));
        ggml_backend_tensor_get_async(ggml_backend_sched_get_tensor_backend(sched, o), o, r.data(), 0, ggml_nbytes(o));
        if (trace_path) {
            ggml_trace_batch_end();
        }
        ggml_backend_sched_synchronize(sched);
        if (trace_path) {
            ggml_trace_stamp_token_out((int32_t) s + 1);
        }
        out.push_back(r);
        ggml_free(ctx);
    }

    if (trace_path) {
        ggml_trace_stop();
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(bw);
    ggml_free(cw);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return out;
}

int main() {
    ggml_log_set(log_cb, nullptr);
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("no CUDA device: SKIPPED\n");
        return 77;
    }

    // A = 1 column, B = 2 columns; traced steps right after a shape change and with a shape change inside
    const std::vector<step_desc> steps = {
        { 1, false }, { 1, false }, { 1, false }, // A captured
        { 2, false }, { 2, true  }, { 2, false }, { 2, false }, { 2, false }, // B, traced while B warms up
        { 1, false }, { 1, true  }, { 2, true  }, { 2, false }, { 2, false }, // change inside the traced window
        { 1, false }, { 1, false }, { 1, true  }, { 1, false }, { 1, false },
    };

    const char * tmp = getenv("TMPDIR");
    const std::string path = std::string(tmp && *tmp ? tmp : ".") + "/test-trace-cuda-" + std::to_string(std::random_device{}()) + ".json";

    const auto ref = run(steps, nullptr);
    const auto got = run(steps, path.c_str());

    for (size_t s = 0; s < steps.size(); ++s) {
        CHECK(got[s].size() == ref[s].size() && memcmp(got[s].data(), ref[s].data(), ref[s].size()*sizeof(float)) == 0,
                "step %zu (%d columns, %s): output differs from the untraced run", s, steps[s].n_cols, steps[s].traced ? "traced" : "untraced");
    }

    // the traced steps have device timings for all 4 GPU nodes (n_fused counts fused groups) and scheduler events
    std::ifstream in(path);
    json ev;
    try {
        ev = json::parse(in);
    } catch (const std::exception & e) {
        CHECK(false, "trace does not parse: %s", e.what());
    }
    size_t n_traced = 0;
    for (const auto & sd : steps) {
        n_traced += sd.traced;
    }
    std::vector<int> gpu_nodes(n_traced, 0);
    size_t n_tok = 0, n_split = 0;
    for (const auto & e : ev) {
        const std::string cat = e.value("cat", "");
        if (cat == "token") {
            n_tok++;
        } else if (cat == "split") {
            n_split++;
        } else if (cat == "op" && e.at("pid") == 2) {
            const int st = e.at("args").at("step").get<int>();
            CHECK(st >= 0 && st < (int) n_traced, "OP_GPU step %d", st);
            if (st >= 0 && st < (int) n_traced) {
                gpu_nodes[st] += e.at("args").at("n_fused").get<int>();
            }
            CHECK(e.at("dur").get<double>() >= 0.0, "OP_GPU duration");
        }
    }
    CHECK(n_tok == n_traced, "TOKEN events %zu, traced steps %zu", n_tok, n_traced);
    for (size_t st = 0; st < n_traced; ++st) {
        CHECK(gpu_nodes[st] == 4, "step %zu: OP_GPU covers %d nodes, expected 4", st, gpu_nodes[st]);
    }
    CHECK(n_split >= n_traced, "SPLIT events %zu", n_split);
    if (!g_fail) {
        std::remove(path.c_str());
    }

    printf("%s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
