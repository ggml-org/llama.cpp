// Profile DeepSeek4 expert routing frequencies during inference.
//
// Captures the `ffn_topk` tensor output for each layer, builds per-layer
// expert-id histograms, and emits a JSON-ish report at the end. Use this to
// see whether routing is skewed enough to make hot-expert pinning worthwhile.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

struct expert_profile {
    int n_layer = 0;
    int n_expert = 0;
    std::map<int, std::vector<uint64_t>> per_layer;
    uint64_t total_token_picks = 0;
    std::vector<uint8_t> scratch;
    std::regex topk_re{"^ffn_topk(?:-([0-9]+))?$"};
};

static bool ds4_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * prof = (expert_profile *) user_data;
    if (!t || !t->name) {
        return ask ? false : true;
    }

    std::cmatch m;
    if (!std::regex_match(t->name, m, prof->topk_re)) {
        return ask ? false : true;
    }

    if (ask) {
        return true;
    }

    int il = -1;
    if (m.size() >= 2 && m[1].matched) {
        il = std::atoi(m[1].str().c_str());
    }

    if (t->type != GGML_TYPE_I32) {
        return true;
    }

    auto nbytes = ggml_nbytes(t);
    prof->scratch.resize(nbytes);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(prof->scratch.data(), t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, prof->scratch.data(), 0, nbytes);
    }

    auto & hist = prof->per_layer[il];
    if ((int) hist.size() < prof->n_expert) {
        hist.assign(prof->n_expert, 0);
    }

    const int32_t * ids = (const int32_t *) prof->scratch.data();
    const size_t n_elems = nbytes / sizeof(int32_t);
    for (size_t i = 0; i < n_elems; ++i) {
        const int32_t e = ids[i];
        if (e >= 0 && e < prof->n_expert) {
            hist[e]++;
            prof->total_token_picks++;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    expert_profile prof;
    params.cb_eval = ds4_cb_eval;
    params.cb_eval_user_data = &prof;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }

    prof.n_layer = llama_model_n_layer(model);
    prof.n_expert = 256; // hardcoded for DS4-Flash; could read from model metadata

    LOG_INF("\nds4-expert-profile: model has %d layers, %d experts\n", prof.n_layer, prof.n_expert);
    LOG_INF("ds4-expert-profile: prompt length: %zu chars\n", params.prompt.size());

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    auto tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("no tokens; provide a prompt with -p\n");
        return 1;
    }
    LOG_INF("ds4-expert-profile: tokenized to %zu tokens\n", tokens.size());

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("decode failed\n");
        return 1;
    }

    LOG_INF("\n=== expert routing report ===\n");
    LOG_INF("total expert picks observed: %" PRIu64 "\n", prof.total_token_picks);

    std::vector<int> top_ks = {8, 16, 32, 64, 128};
    std::map<int, double> hot_coverage_max;
    std::map<int, double> hot_coverage_avg_sum;
    std::map<int, int> hot_coverage_avg_count;

    LOG_INF("\nper-layer routing summary:\n");
    for (auto & [il, hist] : prof.per_layer) {
        if (hist.empty()) continue;

        uint64_t total = 0;
        for (uint64_t v : hist) total += v;
        if (total == 0) continue;

        std::vector<std::pair<int, uint64_t>> sorted;
        sorted.reserve(hist.size());
        for (size_t e = 0; e < hist.size(); ++e) {
            if (hist[e] > 0) sorted.emplace_back((int) e, hist[e]);
        }
        std::sort(sorted.begin(), sorted.end(), [](auto & a, auto & b) {
            return a.second > b.second;
        });

        const uint64_t hottest = sorted.empty() ? 0 : sorted.front().second;
        const int unique_used = (int) sorted.size();

        LOG_INF("layer %2d: total=%" PRIu64 " unique=%d hottest=%" PRIu64 "(%.1f%%)\n",
                il, total, unique_used, hottest, 100.0 * hottest / total);

        for (int k : top_ks) {
            uint64_t sum = 0;
            for (int i = 0; i < k && i < (int) sorted.size(); ++i) {
                sum += sorted[i].second;
            }
            const double frac = 100.0 * sum / total;
            hot_coverage_max[k] = std::max(hot_coverage_max[k], frac);
            hot_coverage_avg_sum[k] += frac;
            hot_coverage_avg_count[k] += 1;
        }
    }

    LOG_INF("\n=== summary across layers ===\n");
    LOG_INF("top-K hot expert coverage:\n");
    for (int k : top_ks) {
        if (hot_coverage_avg_count[k] == 0) continue;
        const double avg = hot_coverage_avg_sum[k] / hot_coverage_avg_count[k];
        LOG_INF("  top-%-3d  avg=%.1f%%  max-layer=%.1f%%\n",
                k, avg, hot_coverage_max[k]);
    }

    LOG_INF("\nper-layer Pareto analysis (how many experts cover X%% of routings):\n");
    for (auto & [il, hist] : prof.per_layer) {
        if (hist.empty()) continue;
        std::vector<uint64_t> sorted_h(hist);
        std::sort(sorted_h.begin(), sorted_h.end(), std::greater<uint64_t>());
        uint64_t total = 0;
        for (uint64_t v : sorted_h) total += v;
        if (total == 0) continue;

        uint64_t cum = 0;
        int e50 = -1, e80 = -1, e90 = -1, e95 = -1, e99 = -1;
        for (size_t i = 0; i < sorted_h.size(); ++i) {
            cum += sorted_h[i];
            if (e50 < 0 && cum * 100 >= total * 50) e50 = (int)(i + 1);
            if (e80 < 0 && cum * 100 >= total * 80) e80 = (int)(i + 1);
            if (e90 < 0 && cum * 100 >= total * 90) e90 = (int)(i + 1);
            if (e95 < 0 && cum * 100 >= total * 95) e95 = (int)(i + 1);
            if (e99 < 0 && cum * 100 >= total * 99) e99 = (int)(i + 1);
        }
        LOG_INF("layer %2d: 50%%=top-%d  80%%=top-%d  90%%=top-%d  95%%=top-%d  99%%=top-%d\n",
                il, e50, e80, e90, e95, e99);
    }

    // Emit JSON profile to file (for runtime hot-expert pinning).
    // Set DS4_PROFILE_JSON_OUT=path.json to enable.
    if (const char * out_path = std::getenv("DS4_PROFILE_JSON_OUT")) {
        FILE * fp = std::fopen(out_path, "w");
        if (fp) {
            std::fprintf(fp, "{\n");
            std::fprintf(fp, "  \"n_layer\": %d,\n", prof.n_layer);
            std::fprintf(fp, "  \"n_expert\": %d,\n", prof.n_expert);
            std::fprintf(fp, "  \"total_picks\": %" PRIu64 ",\n", prof.total_token_picks);
            std::fprintf(fp, "  \"layers\": {\n");
            bool first_layer = true;
            for (auto & [il, hist] : prof.per_layer) {
                if (hist.empty()) continue;
                if (!first_layer) std::fprintf(fp, ",\n");
                first_layer = false;
                // Sort experts by frequency descending; emit pairs.
                std::vector<std::pair<int, uint64_t>> sorted;
                sorted.reserve(hist.size());
                for (size_t e = 0; e < hist.size(); ++e) {
                    if (hist[e] > 0) sorted.emplace_back((int)e, hist[e]);
                }
                std::sort(sorted.begin(), sorted.end(), [](auto & a, auto & b) {
                    return a.second > b.second;
                });
                std::fprintf(fp, "    \"%d\": [", il);
                for (size_t i = 0; i < sorted.size(); ++i) {
                    if (i) std::fprintf(fp, ",");
                    std::fprintf(fp, "[%d,%" PRIu64 "]", sorted[i].first, sorted[i].second);
                }
                std::fprintf(fp, "]");
            }
            std::fprintf(fp, "\n  }\n}\n");
            std::fclose(fp);
            LOG_INF("\nds4-expert-profile: wrote JSON to %s\n", out_path);
        } else {
            LOG_ERR("ds4-expert-profile: could not open %s\n", out_path);
        }
    }

    llama_backend_free();
    return 0;
}
