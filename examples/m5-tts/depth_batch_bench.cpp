// Measures the fixed per-call cost of llama_decode on the depth core versus
// the per-row cost. The depth stage issues 15 serial 2-row decodes per audio
// frame and pays ~2.9 ms each for a model whose 1-token compute is <1 ms; this
// bench separates "cost of a submission" from "cost of a token" so we can tell
// whether coalescing submissions (batching / htp-fusion) is worth pursuing.
//
//   depth_batch_bench <depth.gguf> <backend> [reps]
//
// Scenarios all process 28 rows at positions 0..27 in one KV sequence:
//   1x28  one decode of 28 rows      (coalesced)
//   14x2  fourteen decodes of 2      (what the stage does today)
//   28x1  twenty-eight decodes of 1  (worst case)
//   1x2   one decode of 2 rows       (fixed-cost baseline)
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <depth.gguf> <backend> [reps]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * backend_name = argv[2];
    const int reps = argc > 3 ? std::atoi(argv[3]) : 7;

    ggml_backend_load_all();
    llama_backend_init();

    auto model_params = llama_model_default_params();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name(backend_name), nullptr};
    if (!devices[0]) {
        std::fprintf(stderr, "backend %s is unavailable\n", backend_name);
        return 1;
    }
    model_params.devices = devices;
    model_params.n_gpu_layers = std::strcmp(backend_name, "CPU") == 0 ? 0 : -1;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::fprintf(stderr, "cannot load %s\n", model_path);
        return 1;
    }

    auto params = llama_context_default_params();
    params.embeddings = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.n_ctx = 256;
    params.n_batch = 64;
    params.n_ubatch = 64;
    llama_context * ctx = llama_init_from_model(model, params);
    if (!ctx) {
        std::fprintf(stderr, "cannot init context\n");
        return 1;
    }

    const int width = llama_model_n_embd_inp(model);
    std::fprintf(stderr, "n_embd=%d backend=%s\n", width, backend_name);

    // Random rows; content does not affect cost.
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> embd(28 * width);
    for (float & f : embd) f = dist(rng);

    std::vector<llama_seq_id> seq(64, 0);
    std::vector<llama_seq_id *> seqs(64, seq.data());
    std::vector<int32_t> counts(64, 1);

    auto run = [&](int calls, int rows_per_call) -> double {
        llama_memory_clear(llama_get_memory(ctx), false);
        std::vector<llama_pos> pos(rows_per_call);
        std::vector<int8_t> want(rows_per_call, 0);
        want[rows_per_call - 1] = 1;
        const double t0 = now_ms();
        for (int c = 0; c < calls; ++c) {
            for (int i = 0; i < rows_per_call; ++i) pos[i] = c * rows_per_call + i;
            llama_batch batch{};
            batch.n_tokens = rows_per_call;
            batch.embd = embd.data() + (size_t) c * rows_per_call * width;
            batch.pos = pos.data();
            batch.n_seq_id = counts.data();
            batch.seq_id = seqs.data();
            batch.logits = want.data();
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "decode failed\n");
                std::exit(1);
            }
        }
        return now_ms() - t0;
    };

    struct Scenario { const char * name; int calls; int rows; };
    const Scenario scenarios[] = {
        {"1x28", 1, 28},
        {"14x2", 14, 2},
        {"28x1", 28, 1},
        {"1x2", 1, 2},
    };

    for (const auto & s : scenarios) run(s.calls, s.rows);   // warmup

    std::fprintf(stderr, "\n%-8s %10s %10s %12s\n", "shape", "median_ms", "ms/call", "ms/row");
    for (const auto & s : scenarios) {
        std::vector<double> samples;
        for (int r = 0; r < reps; ++r) samples.push_back(run(s.calls, s.rows));
        const double ms = median(samples);
        std::fprintf(stderr, "%-8s %10.2f %10.3f %12.3f\n", s.name, ms,
                     ms / s.calls, ms / s.rows);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
