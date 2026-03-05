#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

// ---------------------------------------------------------------------------
// MoE Expert Selection Logger
//
// Phase A of the Morphological Analysis pipeline:
//   Instruments inference to capture per-layer, per-token expert selections.
//   Produces a binary log that offline analysis consumes to compute:
//     - Co-selection matrix (which experts are chosen together)
//     - Cross-layer transition matrix (layer L -> L+1 prediction accuracy)
//     - Periodicity spectrum (FFT of per-expert selection timeseries)
//     - Layer-wise prediction accuracy for multiple predictor types
// ---------------------------------------------------------------------------

// Binary log format:
//   Header: "MOELOG\x01\x00" (8 bytes)
//   uint32_t n_layers
//   uint32_t n_experts
//   uint32_t n_expert_used
//   Then per-token records:
//     For each MoE layer (n_moe_layers total):
//       int32_t expert_ids[n_expert_used]
//       float   expert_logits[n_experts]  (optional, if --save-logits)
static const char MOE_LOG_MAGIC[8] = {'M','O','E','L','O','G',0x01,0x00};

struct moe_log_header {
    char     magic[8];
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t n_expert_used;
    uint32_t flags;  // bit 0: has_logits
};

struct moe_collector_data {
    // config
    int n_layers      = 0;
    int n_experts     = 0;
    int n_expert_used = 0;
    bool save_logits  = false;

    // collection state
    int current_token = 0;
    std::vector<uint8_t> tmp_buf;  // for GPU->host copies

    // per-token, per-layer expert selections: [token][layer][expert_used]
    std::vector<std::vector<std::vector<int32_t>>> selections;

    // per-token, per-layer gating logits (optional): [token][layer][n_expert]
    std::vector<std::vector<std::vector<float>>> logits;

    // layer tracking: which layers have we seen this token?
    std::vector<bool> layer_seen;

    // output file
    std::string out_file;
    FILE * fp = nullptr;

    void init(int nl, int ne, int neu, bool sl, const std::string & of) {
        n_layers      = nl;
        n_experts     = ne;
        n_expert_used = neu;
        save_logits   = sl;
        out_file      = of;
        layer_seen.resize(nl, false);

        fp = fopen(out_file.c_str(), "wb");
        if (!fp) {
            LOG_ERR("Failed to open output file: %s\n", out_file.c_str());
            return;
        }

        // write placeholder header (will be updated in finish() with auto-detected values)
        moe_log_header hdr = {};
        memcpy(hdr.magic, MOE_LOG_MAGIC, 8);
        hdr.n_layers      = nl;
        hdr.n_experts     = ne;
        hdr.n_expert_used = neu;
        hdr.flags         = sl ? 1 : 0;
        fwrite(&hdr, sizeof(hdr), 1, fp);
    }

    void new_token() {
        current_token++;
        std::fill(layer_seen.begin(), layer_seen.end(), false);
    }

    void record_topk(int layer, const int32_t * ids, int n_ids) {
        if (layer < 0 || layer >= n_layers) return;
        layer_seen[layer] = true;

        // write to file immediately (streaming)
        if (fp) {
            fwrite(ids, sizeof(int32_t), n_ids, fp);
        }
    }

    void record_logits(int layer, const float * data, int n) {
        if (!save_logits || !fp) return;
        if (layer < 0 || layer >= n_layers) return;
        fwrite(data, sizeof(float), n, fp);
    }

    void finish() {
        if (fp) {
            // rewrite header with auto-detected values
            fseek(fp, 0, SEEK_SET);
            moe_log_header hdr = {};
            memcpy(hdr.magic, MOE_LOG_MAGIC, 8);
            hdr.n_layers      = n_layers;
            hdr.n_experts     = n_experts;
            hdr.n_expert_used = n_expert_used;
            hdr.flags         = save_logits ? 1u : 0u;
            fwrite(&hdr, sizeof(hdr), 1, fp);

            fclose(fp);
            fp = nullptr;
        }
        LOG_INF("MoE expert selections saved to %s (%d tokens, %d experts, top-%d)\n",
            out_file.c_str(), current_token, n_experts, n_expert_used);
    }
};

static moe_collector_data g_collector;

// Parse "ffn_moe_topk-42" -> ("ffn_moe_topk", 42)
static bool parse_moe_tensor_name(const char * name, std::string & op_name, int & layer) {
    if (!name) return false;
    const char * dash = strrchr(name, '-');
    if (!dash || dash[1] < '0' || dash[1] > '9') return false;
    op_name = std::string(name, dash - name);
    layer = atoi(dash + 1);
    return true;
}

static bool moe_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;

    std::string op_name;
    int layer = -1;

    if (!t->name[0]) return false;
    if (!parse_moe_tensor_name(t->name, op_name, layer)) return false;

    if (op_name == "ffn_moe_topk") {
        if (ask) return true;  // yes, we want this tensor's data

        // t shape: [n_expert_used, n_tokens] of type I32
        const int64_t neu = t->ne[0];
        const int64_t n_tokens = t->ne[1];

        // auto-detect n_expert_used on first encounter
        if (g_collector.n_expert_used == 0) {
            g_collector.n_expert_used = (int)neu;
            LOG_INF("Auto-detected n_expert_used = %d\n", (int)neu);
        }

        // copy data to host if needed
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        const int32_t * data;

        if (!is_host) {
            g_collector.tmp_buf.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, g_collector.tmp_buf.data(), 0, ggml_nbytes(t));
            data = (const int32_t *)g_collector.tmp_buf.data();
        } else {
            data = (const int32_t *)t->data;
        }

        // for each token in the batch
        for (int64_t tok = 0; tok < n_tokens; tok++) {
            const int32_t * ids = data + tok * neu;
            g_collector.record_topk(layer, ids, (int)neu);
        }
        return true;
    }

    if (op_name == "ffn_moe_logits") {
        if (ask) return g_collector.save_logits;

        const int64_t n_expert = t->ne[0];
        const int64_t n_tokens = t->ne[1];

        // auto-detect n_experts on first encounter
        if (g_collector.n_experts == 0) {
            g_collector.n_experts = (int)n_expert;
            LOG_INF("Auto-detected n_experts = %d\n", (int)n_expert);
        }

        if (!g_collector.save_logits) return true;

        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        const float * data;

        if (!is_host) {
            g_collector.tmp_buf.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, g_collector.tmp_buf.data(), 0, ggml_nbytes(t));
            data = (const float *)g_collector.tmp_buf.data();
        } else {
            data = (const float *)t->data;
        }

        for (int64_t tok = 0; tok < n_tokens; tok++) {
            g_collector.record_logits(layer, data + tok * n_expert, (int)n_expert);
        }
        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// Offline Analysis
// ---------------------------------------------------------------------------

struct moe_log_data {
    int n_layers;
    int n_experts;
    int n_expert_used;
    bool has_logits;
    int n_tokens;

    // selections[token][layer] = vector of expert ids
    std::vector<std::vector<std::vector<int32_t>>> selections;
    // logits[token][layer] = vector of logit values
    std::vector<std::vector<std::vector<float>>> logits;
};

static bool load_moe_log(const std::string & path, moe_log_data & out) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        LOG_ERR("Cannot open %s\n", path.c_str());
        return false;
    }

    moe_log_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        LOG_ERR("Failed to read header\n");
        fclose(fp);
        return false;
    }
    if (memcmp(hdr.magic, MOE_LOG_MAGIC, 8) != 0) {
        LOG_ERR("Invalid magic in %s\n", path.c_str());
        fclose(fp);
        return false;
    }

    out.n_layers      = hdr.n_layers;
    out.n_experts     = hdr.n_experts;
    out.n_expert_used = hdr.n_expert_used;
    out.has_logits    = (hdr.flags & 1) != 0;

    // each token has n_layers records
    std::vector<int32_t> ids(out.n_expert_used);
    std::vector<float>   lgt(out.n_experts);

    int layer_idx = 0;
    int token_idx = 0;
    out.selections.push_back(std::vector<std::vector<int32_t>>(out.n_layers));
    if (out.has_logits) {
        out.logits.push_back(std::vector<std::vector<float>>(out.n_layers));
    }

    while (true) {
        if (fread(ids.data(), sizeof(int32_t), out.n_expert_used, fp) != (size_t)out.n_expert_used) {
            break;
        }
        out.selections[token_idx][layer_idx] = ids;

        if (out.has_logits) {
            if (fread(lgt.data(), sizeof(float), out.n_experts, fp) != (size_t)out.n_experts) {
                break;
            }
            out.logits[token_idx][layer_idx] = lgt;
        }

        layer_idx++;
        if (layer_idx >= out.n_layers) {
            layer_idx = 0;
            token_idx++;
            out.selections.push_back(std::vector<std::vector<int32_t>>(out.n_layers));
            if (out.has_logits) {
                out.logits.push_back(std::vector<std::vector<float>>(out.n_layers));
            }
        }
    }

    // remove the last incomplete token if any
    if (layer_idx != 0) {
        out.selections.pop_back();
        if (out.has_logits) out.logits.pop_back();
    } else {
        // we added one extra at the end of the loop
        out.selections.pop_back();
        if (out.has_logits) out.logits.pop_back();
    }

    out.n_tokens = (int)out.selections.size();
    fclose(fp);
    return true;
}

// Analysis 1: Co-selection matrix
// co_select[i][j] = number of times experts i and j were selected in the same layer for the same token
static void analyze_co_selection(const moe_log_data & data) {
    const int ne = data.n_experts;
    std::vector<std::vector<int>> co_select(ne, std::vector<int>(ne, 0));
    std::vector<int> select_count(ne, 0);

    for (int t = 0; t < data.n_tokens; t++) {
        for (int l = 0; l < data.n_layers; l++) {
            const auto & sel = data.selections[t][l];
            for (int i = 0; i < (int)sel.size(); i++) {
                select_count[sel[i]]++;
                for (int j = i + 1; j < (int)sel.size(); j++) {
                    co_select[sel[i]][sel[j]]++;
                    co_select[sel[j]][sel[i]]++;
                }
            }
        }
    }

    // find top-20 co-selected pairs
    struct pair_count {
        int i, j, count;
    };
    std::vector<pair_count> pairs;
    for (int i = 0; i < ne; i++) {
        for (int j = i + 1; j < ne; j++) {
            if (co_select[i][j] > 0) {
                pairs.push_back({i, j, co_select[i][j]});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const pair_count & a, const pair_count & b) {
        return a.count > b.count;
    });

    LOG_INF("\n=== CO-SELECTION ANALYSIS ===\n");
    LOG_INF("Top 20 co-selected expert pairs (across all layers):\n");
    LOG_INF("  %-8s %-8s %-10s %-10s\n", "Exp_A", "Exp_B", "Co-count", "Rate");
    const int total_samples = data.n_tokens * data.n_layers;
    for (int i = 0; i < std::min(20, (int)pairs.size()); i++) {
        LOG_INF("  %-8d %-8d %-10d %.4f\n",
            pairs[i].i, pairs[i].j, pairs[i].count,
            (float)pairs[i].count / total_samples);
    }

    // expert usage distribution
    LOG_INF("\nExpert usage distribution (total selections across all layers/tokens):\n");
    LOG_INF("  %-8s %-10s %-10s\n", "Expert", "Count", "Rate");
    std::vector<int> sorted_idx(ne);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
        return select_count[a] > select_count[b];
    });
    const int total_expert_slots = data.n_tokens * data.n_layers * data.n_expert_used;
    for (int i = 0; i < std::min(20, ne); i++) {
        int idx = sorted_idx[i];
        LOG_INF("  %-8d %-10d %.4f\n", idx, select_count[idx],
            (float)select_count[idx] / total_expert_slots);
    }

    // compute clustering metric: how many unique expert "sets" exist?
    std::map<std::vector<int32_t>, int> unique_sets;
    for (int t = 0; t < data.n_tokens; t++) {
        for (int l = 0; l < data.n_layers; l++) {
            auto sel = data.selections[t][l];
            std::sort(sel.begin(), sel.end());
            unique_sets[sel]++;
        }
    }
    LOG_INF("\nUnique expert sets: %zu / %d total (%.1f%% diversity)\n",
        unique_sets.size(), total_samples,
        100.0f * unique_sets.size() / total_samples);

    // find top sets
    struct set_count {
        std::vector<int32_t> set;
        int count;
    };
    std::vector<set_count> top_sets;
    for (auto & [s, c] : unique_sets) {
        top_sets.push_back({s, c});
    }
    std::sort(top_sets.begin(), top_sets.end(), [](const set_count & a, const set_count & b) {
        return a.count > b.count;
    });
    LOG_INF("\nTop 10 most common expert sets:\n");
    for (int i = 0; i < std::min(10, (int)top_sets.size()); i++) {
        LOG_INF("  [");
        for (int j = 0; j < (int)top_sets[i].set.size(); j++) {
            LOG_INF("%d", top_sets[i].set[j]);
            if (j + 1 < (int)top_sets[i].set.size()) LOG_INF(",");
        }
        LOG_INF("] x%d (%.2f%%)\n", top_sets[i].count,
            100.0f * top_sets[i].count / total_samples);
    }
}

// Analysis 2: Cross-layer prediction accuracy
// For each layer L, use the experts selected at L to predict experts at L+1
static void analyze_cross_layer(const moe_log_data & data) {
    LOG_INF("\n=== CROSS-LAYER PREDICTION ANALYSIS ===\n");
    LOG_INF("Method: use layer L's selected experts to predict layer L+1\n\n");

    // per-layer accuracy
    std::vector<int> correct(data.n_layers - 1, 0);
    std::vector<int> total(data.n_layers - 1, 0);

    // also track: how many of L+1's experts appear in L's selection?
    std::vector<float> overlap_sum(data.n_layers - 1, 0.0f);

    for (int t = 0; t < data.n_tokens; t++) {
        for (int l = 0; l < data.n_layers - 1; l++) {
            const auto & sel_l  = data.selections[t][l];
            const auto & sel_l1 = data.selections[t][l + 1];

            // count how many of L+1's experts are in L's set
            int hits = 0;
            for (int e : sel_l1) {
                for (int f : sel_l) {
                    if (e == f) { hits++; break; }
                }
            }
            correct[l] += hits;
            total[l]   += (int)sel_l1.size();
            overlap_sum[l] += (float)hits / sel_l1.size();
        }
    }

    LOG_INF("  %-8s %-12s %-12s\n", "Layer", "Accuracy", "Avg Overlap");
    float total_acc = 0;
    for (int l = 0; l < data.n_layers - 1; l++) {
        float acc = total[l] > 0 ? (float)correct[l] / total[l] : 0;
        float avg_overlap = data.n_tokens > 0 ? overlap_sum[l] / data.n_tokens : 0;
        total_acc += acc;
        if (l < 10 || l >= data.n_layers - 3) {
            LOG_INF("  L%-6d %.4f       %.4f\n", l, acc, avg_overlap);
        } else if (l == 10) {
            LOG_INF("  ...     (see summary)\n");
        }
    }
    LOG_INF("\n  Mean cross-layer prediction accuracy: %.4f\n",
        total_acc / (data.n_layers - 1));

    // previous-token prediction: use token T-1's experts at layer L to predict token T's at L
    LOG_INF("\n--- Previous-token reuse accuracy ---\n");
    std::vector<int> prev_correct(data.n_layers, 0);
    std::vector<int> prev_total(data.n_layers, 0);

    for (int t = 1; t < data.n_tokens; t++) {
        for (int l = 0; l < data.n_layers; l++) {
            const auto & sel_prev = data.selections[t - 1][l];
            const auto & sel_curr = data.selections[t][l];
            int hits = 0;
            for (int e : sel_curr) {
                for (int f : sel_prev) {
                    if (e == f) { hits++; break; }
                }
            }
            prev_correct[l] += hits;
            prev_total[l]   += (int)sel_curr.size();
        }
    }

    float prev_mean = 0;
    for (int l = 0; l < data.n_layers; l++) {
        float acc = prev_total[l] > 0 ? (float)prev_correct[l] / prev_total[l] : 0;
        prev_mean += acc;
    }
    prev_mean /= data.n_layers;
    LOG_INF("  Mean previous-token reuse accuracy: %.4f\n", prev_mean);
}

// Analysis 3: Periodicity analysis
// For each expert in each layer, compute selection interval statistics
static void analyze_periodicity(const moe_log_data & data) {
    LOG_INF("\n=== PERIODICITY ANALYSIS ===\n");
    LOG_INF("Per-expert selection interval statistics (tokens between consecutive selections)\n\n");

    // for a few sample layers, compute interval distributions
    std::vector<int> sample_layers;
    if (data.n_layers <= 6) {
        for (int l = 0; l < data.n_layers; l++) sample_layers.push_back(l);
    } else {
        sample_layers = {0, data.n_layers/4, data.n_layers/2, 3*data.n_layers/4, data.n_layers-1};
    }

    for (int l : sample_layers) {
        LOG_INF("--- Layer %d ---\n", l);

        // track last-seen token for each expert
        std::vector<int> last_seen(data.n_experts, -1);
        // interval histograms per expert
        std::vector<std::vector<int>> intervals(data.n_experts);

        for (int t = 0; t < data.n_tokens; t++) {
            const auto & sel = data.selections[t][l];
            for (int e : sel) {
                if (last_seen[e] >= 0) {
                    intervals[e].push_back(t - last_seen[e]);
                }
                last_seen[e] = t;
            }
        }

        // summary stats for each expert
        struct expert_period {
            int expert;
            float mean_interval;
            float std_interval;
            int n_activations;
            float regularity;  // 1/CV = mean/std (higher = more periodic)
        };
        std::vector<expert_period> periods;

        for (int e = 0; e < data.n_experts; e++) {
            if (intervals[e].size() < 2) continue;
            float sum = 0;
            for (int iv : intervals[e]) sum += iv;
            float mean = sum / intervals[e].size();

            float var = 0;
            for (int iv : intervals[e]) var += (iv - mean) * (iv - mean);
            float std_dev = std::sqrt(var / intervals[e].size());

            periods.push_back({e, mean, std_dev, (int)intervals[e].size() + 1,
                std_dev > 0 ? mean / std_dev : 999.0f});
        }

        // sort by regularity (most periodic first)
        std::sort(periods.begin(), periods.end(), [](const expert_period & a, const expert_period & b) {
            return a.regularity > b.regularity;
        });

        LOG_INF("  %-8s %-8s %-10s %-10s %-10s\n",
            "Expert", "#Acts", "MeanIntv", "StdIntv", "Regularity");
        for (int i = 0; i < std::min(10, (int)periods.size()); i++) {
            LOG_INF("  %-8d %-8d %-10.1f %-10.1f %-10.2f\n",
                periods[i].expert, periods[i].n_activations,
                periods[i].mean_interval, periods[i].std_interval,
                periods[i].regularity);
        }

        // overall layer stats
        float total_activations = 0;
        for (auto & p : periods) total_activations += p.n_activations;
        int active_experts = (int)periods.size();
        LOG_INF("  Active experts: %d/%d, avg activations: %.1f\n\n",
            active_experts, data.n_experts,
            active_experts > 0 ? total_activations / active_experts : 0);
    }
}

// Analysis 4: Staleness analysis (SpecMD-inspired)
// Compute how a Least-Stale eviction policy would compare to LRU
static void analyze_staleness(const moe_log_data & data) {
    LOG_INF("\n=== STALENESS vs LRU ANALYSIS ===\n");
    LOG_INF("Simulated cache hit rates for different eviction policies\n");
    LOG_INF("(Cache size = N experts, simulated per-layer)\n\n");

    // simulate for different cache sizes
    std::vector<int> cache_sizes = {8, 16, 24, 32, 48, 64};

    for (int l : {0, data.n_layers/2, data.n_layers-1}) {
        LOG_INF("--- Layer %d ---\n", l);
        LOG_INF("  %-12s", "Cache size");
        for (int cs : cache_sizes) {
            if (cs > data.n_experts) break;
            LOG_INF("%-10d", cs);
        }
        LOG_INF("\n");

        // LRU simulation
        LOG_INF("  %-12s", "LRU hits");
        for (int cs : cache_sizes) {
            if (cs > data.n_experts) break;
            std::vector<int> lru_cache;
            int hits = 0, total = 0;

            for (int t = 0; t < data.n_tokens; t++) {
                const auto & sel = data.selections[t][l];
                for (int e : sel) {
                    total++;
                    auto it = std::find(lru_cache.begin(), lru_cache.end(), e);
                    if (it != lru_cache.end()) {
                        hits++;
                        lru_cache.erase(it);
                    } else if ((int)lru_cache.size() >= cs) {
                        lru_cache.erase(lru_cache.begin());
                    }
                    lru_cache.push_back(e);
                }
            }
            LOG_INF("%-10.4f", total > 0 ? (float)hits / total : 0);
        }
        LOG_INF("\n");

        // Frequency-based (EMA) simulation
        LOG_INF("  %-12s", "EMA hits");
        for (int cs : cache_sizes) {
            if (cs > data.n_experts) break;
            std::vector<float> ema(data.n_experts, 0.0f);
            const float alpha = 0.1f;
            int hits = 0, total = 0;

            for (int t = 0; t < data.n_tokens; t++) {
                const auto & sel = data.selections[t][l];

                // build current cache from top-cs by EMA
                std::vector<int> cache_idx(data.n_experts);
                std::iota(cache_idx.begin(), cache_idx.end(), 0);
                std::partial_sort(cache_idx.begin(), cache_idx.begin() + cs, cache_idx.end(),
                    [&](int a, int b) { return ema[a] > ema[b]; });

                for (int e : sel) {
                    total++;
                    for (int i = 0; i < cs; i++) {
                        if (cache_idx[i] == e) { hits++; break; }
                    }
                }

                // update EMA
                for (int e = 0; e < data.n_experts; e++) {
                    ema[e] *= (1.0f - alpha);
                }
                for (int e : sel) {
                    ema[e] += alpha;
                }
            }
            LOG_INF("%-10.4f", total > 0 ? (float)hits / total : 0);
        }
        LOG_INF("\n");

        // Oracle (perfect prediction) - upper bound
        LOG_INF("  %-12s", "Oracle");
        for (int cs : cache_sizes) {
            if (cs > data.n_experts) break;
            // oracle: for each access, the next cs unique experts to be accessed are in cache
            // simplified: just count how many of the current selection were also in the previous selection
            // actually, oracle = Least-Stale = evict the expert whose next access is furthest away
            // full oracle simulation:
            int hits = 0, total = 0;

            // precompute next-access times for each expert at each step
            // (iterate backwards to compute "next time expert e is accessed after step s")
            std::vector<std::vector<int>> access_times(data.n_experts);
            for (int t = 0; t < data.n_tokens; t++) {
                for (int e : data.selections[t][l]) {
                    access_times[e].push_back(t);
                }
            }
            // for each expert, reverse the list so we can binary search
            // next_access[e] after time t = lower_bound in access_times[e] for value > t

            std::vector<int> cache;  // current cache contents

            for (int t = 0; t < data.n_tokens; t++) {
                const auto & sel = data.selections[t][l];
                for (int e : sel) {
                    total++;
                    if (std::find(cache.begin(), cache.end(), e) != cache.end()) {
                        hits++;
                    } else {
                        if ((int)cache.size() >= cs) {
                            // evict expert with furthest next access (Belady's optimal)
                            int worst_idx = 0;
                            int worst_next = -1;
                            for (int ci = 0; ci < (int)cache.size(); ci++) {
                                int ce = cache[ci];
                                auto it = std::upper_bound(access_times[ce].begin(),
                                    access_times[ce].end(), t);
                                int next = (it != access_times[ce].end()) ? *it : data.n_tokens + 1;
                                if (next > worst_next) {
                                    worst_next = next;
                                    worst_idx = ci;
                                }
                            }
                            cache.erase(cache.begin() + worst_idx);
                        }
                        cache.push_back(e);
                    }
                }
            }
            LOG_INF("%-10.4f", total > 0 ? (float)hits / total : 0);
        }
        LOG_INF("\n\n");
    }
}

// Analysis 5: Cross-layer logit prediction (requires --save-logits)
// Use the gating logits at layer L to predict top-k at L+1
static void analyze_logit_prediction(const moe_log_data & data) {
    if (!data.has_logits || data.logits.empty()) {
        LOG_INF("\n=== LOGIT-BASED CROSS-LAYER PREDICTION ===\n");
        LOG_INF("Skipped: no logits saved (use --save-logits during collection)\n");
        return;
    }

    LOG_INF("\n=== LOGIT-BASED CROSS-LAYER PREDICTION ===\n");
    LOG_INF("Using layer L gating logits as predictors for layer L+1 expert selection\n\n");

    // for each layer pair (L, L+1), use top-K of L's logits to predict L+1's selection
    std::vector<float> accuracy(data.n_layers - 1, 0.0f);

    for (int t = 0; t < data.n_tokens; t++) {
        for (int l = 0; l < data.n_layers - 1; l++) {
            const auto & logits_l = data.logits[t][l];
            const auto & sel_l1   = data.selections[t][l + 1];

            if (logits_l.empty()) continue;

            // find top-K from logits at layer L
            std::vector<int> pred_idx(data.n_experts);
            std::iota(pred_idx.begin(), pred_idx.end(), 0);
            int k = data.n_expert_used;
            std::partial_sort(pred_idx.begin(), pred_idx.begin() + k, pred_idx.end(),
                [&](int a, int b) { return logits_l[a] > logits_l[b]; });

            // count hits
            int hits = 0;
            for (int e : sel_l1) {
                for (int i = 0; i < k; i++) {
                    if (pred_idx[i] == e) { hits++; break; }
                }
            }
            accuracy[l] += (float)hits / k;
        }
    }

    LOG_INF("  %-8s %-12s\n", "Layer", "Accuracy");
    float total = 0;
    for (int l = 0; l < data.n_layers - 1; l++) {
        float acc = data.n_tokens > 0 ? accuracy[l] / data.n_tokens : 0;
        total += acc;
        if (l < 10 || l >= data.n_layers - 3) {
            LOG_INF("  L%-6d %.4f\n", l, acc);
        } else if (l == 10) {
            LOG_INF("  ...     (see mean)\n");
        }
    }
    LOG_INF("\n  Mean logit-based cross-layer accuracy: %.4f\n",
        total / (data.n_layers - 1));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void print_usage(int, char ** argv) {
    LOG("\nMoE Expert Selection Analyzer\n");
    LOG("Phase A of the Morphological Analysis pipeline for MoE prefetch optimization.\n\n");
    LOG("Usage:\n");
    LOG("  Collect: %s -m model.gguf -f input.txt [-o moe_log.bin] [--save-logits]\n", argv[0]);
    LOG("  Analyze: %s --analyze moe_log.bin\n\n", argv[0]);
    LOG("Collection options:\n");
    LOG("  -m MODEL       Model file (GGUF format)\n");
    LOG("  -f FILE        Input text file for token generation\n");
    LOG("  -o FILE        Output log file (default: moe_selections.bin)\n");
    LOG("  --save-logits  Also save gating logits (larger file, enables logit-based analysis)\n\n");
    LOG("Analysis options:\n");
    LOG("  --analyze FILE   Run offline analysis on a collected log file\n");
    LOG("  --all            Run all analyses (default)\n");
    LOG("  --co-selection   Run co-selection matrix analysis\n");
    LOG("  --cross-layer    Run cross-layer prediction analysis\n");
    LOG("  --periodicity    Run periodicity/regularity analysis\n");
    LOG("  --staleness      Run staleness vs LRU cache simulation\n");
    LOG("  --logit-predict  Run logit-based cross-layer prediction\n\n");
}

int main(int argc, char ** argv) {
    // Check for --analyze mode first
    std::string analyze_file;
    bool do_co_selection  = false;
    bool do_cross_layer   = false;
    bool do_periodicity   = false;
    bool do_staleness     = false;
    bool do_logit_predict = false;
    bool do_all           = false;
    bool save_logits      = false;
    std::string out_file  = "moe_selections.bin";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--analyze") == 0 && i + 1 < argc) {
            analyze_file = argv[++i];
        } else if (strcmp(argv[i], "--all") == 0) {
            do_all = true;
        } else if (strcmp(argv[i], "--co-selection") == 0) {
            do_co_selection = true;
        } else if (strcmp(argv[i], "--cross-layer") == 0) {
            do_cross_layer = true;
        } else if (strcmp(argv[i], "--periodicity") == 0) {
            do_periodicity = true;
        } else if (strcmp(argv[i], "--staleness") == 0) {
            do_staleness = true;
        } else if (strcmp(argv[i], "--logit-predict") == 0) {
            do_logit_predict = true;
        } else if (strcmp(argv[i], "--save-logits") == 0) {
            save_logits = true;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_file = argv[++i];
        }
    }

    // Analysis mode
    if (!analyze_file.empty()) {
        LOG_INF("Loading MoE selection log: %s\n", analyze_file.c_str());

        moe_log_data data;
        if (!load_moe_log(analyze_file, data)) {
            return 1;
        }

        LOG_INF("Loaded: %d tokens, %d layers, %d experts, top-%d\n",
            data.n_tokens, data.n_layers, data.n_experts, data.n_expert_used);

        if (!do_co_selection && !do_cross_layer && !do_periodicity &&
            !do_staleness && !do_logit_predict) {
            do_all = true;
        }

        if (do_all || do_co_selection)  analyze_co_selection(data);
        if (do_all || do_cross_layer)   analyze_cross_layer(data);
        if (do_all || do_periodicity)   analyze_periodicity(data);
        if (do_all || do_staleness)     analyze_staleness(data);
        if (do_all || do_logit_predict) analyze_logit_prediction(data);

        return 0;
    }

    // Collection mode: use common_params for model/prompt handling
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY, print_usage)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // set up the eval callback
    params.cb_eval           = moe_cb_eval;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("Failed to initialize model/context\n");
        return 1;
    }

    // detect MoE parameters from model metadata
    const int n_layers = llama_model_n_layer(model);

    // read expert count from GGUF metadata
    char meta_buf[64] = {0};
    int n_experts = 0;
    int n_expert_used = 0;

    // try common metadata key patterns
    if (llama_model_meta_val_str(model, "general.expert_count", meta_buf, sizeof(meta_buf)) > 0) {
        n_experts = atoi(meta_buf);
    }
    if (llama_model_meta_val_str(model, "general.expert_used_count", meta_buf, sizeof(meta_buf)) > 0) {
        n_expert_used = atoi(meta_buf);
    }

    if (n_experts <= 1) {
        // will be auto-detected from first callback tensor
        LOG_INF("Expert count not found in metadata, will auto-detect from inference\n");
        n_experts     = 0;
        n_expert_used = 0;
    } else {
        LOG_INF("MoE model detected: %d layers, %d experts, top-%d\n",
            n_layers, n_experts, n_expert_used);
    }

    // init with potentially zero expert counts (auto-detected later)
    g_collector.init(n_layers, n_experts, n_expert_used, save_logits, out_file);

    // tokenize input
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;

    if (!params.prompt.empty()) {
        tokens = common_tokenize(vocab, params.prompt, true);
    } else if (!params.prompt_file.empty()) {
        // read from file
        std::ifstream ifs(params.prompt_file);
        if (!ifs) {
            LOG_ERR("Failed to open prompt file: %s\n", params.prompt_file.c_str());
            llama_backend_free();
            return 1;
        }
        std::string text((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
        tokens = common_tokenize(vocab, text, true);
    } else {
        LOG_ERR("No prompt provided. Use -p or -f to provide input text.\n");
        llama_backend_free();
        return 1;
    }

    LOG_INF("Input: %zu tokens\n", tokens.size());

    // process tokens in chunks
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = llama_n_batch(ctx);

    LOG_INF("Context: %d, batch: %d\n", n_ctx, n_batch);

    int n_processed = 0;
    const int n_total = std::min((int)tokens.size(), n_ctx);

    while (n_processed < n_total) {
        int n_eval = std::min(n_batch, n_total - n_processed);

        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        for (int i = 0; i < n_eval; i++) {
            common_batch_add(batch, tokens[n_processed + i], n_processed + i, {0}, false);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode batch at position %d\n", n_processed);
            llama_batch_free(batch);
            break;
        }

        g_collector.current_token = n_processed + n_eval;

        n_processed += n_eval;
        llama_batch_free(batch);

        if (n_processed % 100 == 0 || n_processed >= n_total) {
            LOG_INF("Progress: %d / %d tokens\n", n_processed, n_total);
        }
    }

    g_collector.finish();

    LOG_INF("\nCollection complete. Run analysis with:\n");
    LOG_INF("  %s --analyze %s\n", argv[0], out_file.c_str());

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
