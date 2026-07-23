/**
 * expert-profile: NemotronH MoE expert activation profiler (REAP implementation)
 *
 * Implements the REAP (Router-weighted Expert Activation Pruning) saliency criterion:
 *
 *   REAP(j) = mean over tokens routed to j of:  gate_weight(j,t) * ||expert_output(j,t)||_2
 *
 * where expert_output is ffn_moe_down (the FFN output BEFORE gate weighting),
 * and gate_weight is ffn_moe_weights (post-softmax routing probability).
 *
 * Intercepts three tensors per MoE layer via ggml eval callback:
 *   ffn_moe_topk-{il}    [n_expert_used, n_tokens] I32  — which experts were selected
 *   ffn_moe_weights-{il} [1, n_expert_used, n_tokens] F32 — gate weights (softmax probs)
 *   ffn_moe_down-{il}    [n_embd, n_expert_used, n_tokens] F32 — expert outputs (pre-weighting)
 *
 * Reference: "REAP: Router-weighted Expert Activation Pruning" (arXiv:2510.13999)
 *   score = mean_{x in X_j}[ g_j(x) * ||f_j(x)||_2 ]  (Equation 9)
 *
 * Usage:
 *   llama-expert-profile \
 *     -m model.gguf --jsonl training-data.jsonl --output expert_stats.json \
 *     [--n-experts 128] [--ctx-size 16384] [-ngl 32] [-t 24] [--save-every 1]
 */

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// ─── Per-layer stats ──────────────────────────────────────────────────────────

struct LayerStats {
    int64_t n_experts    = 0;
    int64_t total_tokens = 0;  // tokens processed through this layer

    // Frequency / weighted-frequency (kept for reference/comparison)
    std::vector<int64_t> activation_counts;   // [n_experts] — how many tokens routed here
    std::vector<double>  weighted_freq_sum;   // [n_experts] — sum of gate weights

    // REAP: running sum and count for computing mean(gate_weight * ||expert_out||_2)
    std::vector<double>  reap_sum;            // [n_experts] — sum of g_j(t)*||f_j(t)||_2
    std::vector<double>  ean_sum;             // [n_experts] — sum of ||f_j(t)||_2 (EAN, no gate)

    void init(int64_t n) {
        n_experts = n;
        activation_counts.assign(n, 0);
        weighted_freq_sum.assign(n, 0.0);
        reap_sum.assign(n, 0.0);
        ean_sum.assign(n, 0.0);
    }

    // Called once we have all three tensors for a batch.
    // expert_ids:  [n_expert_used * n_tokens]  I32  — flat, column-major: [k + t*n_expert_used]
    // gate_weights:[n_expert_used * n_tokens]  F32  — same layout
    // expert_outs: [n_embd * n_expert_used * n_tokens] F32 — layout: [e + k*n_embd + t*n_embd*n_expert_used]
    //              i.e. for token t, expert-slot k: out vector starts at t*n_embd*n_expert_used + k*n_embd
    void add_batch(const int32_t * expert_ids,
                   const float   * gate_weights,
                   const float   * expert_outs,
                   int64_t         n_expert_used,
                   int64_t         n_tok,
                   int64_t         n_embd) {
        total_tokens += n_tok;
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t k = 0; k < n_expert_used; ++k) {
                const int64_t flat = k + t * n_expert_used;
                const int32_t eid  = expert_ids[flat];
                if (eid < 0 || eid >= n_experts) continue;

                const float gw = gate_weights[flat];

                // L2 norm of expert output vector for this (token, expert-slot)
                const float * vec = expert_outs + t * n_embd * n_expert_used + k * n_embd;
                double norm2 = 0.0;
                for (int64_t d = 0; d < n_embd; ++d) {
                    norm2 += (double)vec[d] * (double)vec[d];
                }
                const double norm = std::sqrt(norm2);

                activation_counts [eid] += 1;
                weighted_freq_sum [eid] += gw;
                reap_sum          [eid] += gw * norm;   // REAP numerator
                ean_sum           [eid] += norm;         // EAN numerator
            }
        }
    }
};

// ─── Collector ────────────────────────────────────────────────────────────────

struct ExpertCollector {
    int64_t n_experts = 128;

    std::map<int, LayerStats> layer_stats;
    std::mutex                mtx;

    // We need all three tensors before we can compute REAP.
    // They arrive in order: topk → weights → down (per the graph build order).
    // Store pending topk+weights until down arrives.
    struct PendingBatch {
        int64_t              n_expert_used = 0;
        int64_t              n_tokens      = 0;
        std::vector<int32_t> expert_ids;    // [n_expert_used * n_tokens]
        std::vector<float>   gate_weights;  // [n_expert_used * n_tokens]
        bool                 has_topk    = false;
        bool                 has_weights = false;
    };
    std::map<int, PendingBatch> pending; // layer_idx → pending

    // Strip device prefix/suffix: "CUDA0#ffn_moe_down-5#0" → "ffn_moe_down-5"
    static std::string clean_name(const char * raw) {
        const char * p = strchr(raw, '#');
        if (p) {
            ++p;
            const char * q = strchr(p, '#');
            return q ? std::string(p, q - p) : std::string(p);
        }
        return raw;
    }

    bool wants(struct ggml_tensor * t) {
        if (!t->name[0]) return false;
        const std::string n = clean_name(t->name);
        return (n.compare(0, 13, "ffn_moe_topk-")    == 0 ||
                n.compare(0, 16, "ffn_moe_weights-") == 0 ||
                n.compare(0, 13, "ffn_moe_down-")    == 0);
    }

    bool on_tensor(struct ggml_tensor * t) {
        const std::string name = clean_name(t->name);

        // Identify tensor type and layer
        int  il         = -1;
        bool is_topk    = false;
        bool is_weights = false;
        bool is_down    = false;

        if      (name.compare(0, 13, "ffn_moe_topk-")    == 0) { il = atoi(name.c_str() + 13); is_topk    = true; }
        else if (name.compare(0, 16, "ffn_moe_weights-") == 0) { il = atoi(name.c_str() + 16); is_weights = true; }
        else if (name.compare(0, 13, "ffn_moe_down-")    == 0) { il = atoi(name.c_str() + 13); is_down    = true; }
        else return true;

        if (il < 0) return true;

        // Copy tensor data from (possibly GPU) buffer to host
        const size_t nbytes = ggml_nbytes(t);
        std::vector<char> buf(nbytes);
        ggml_backend_tensor_get(t, buf.data(), 0, nbytes);

        std::lock_guard<std::mutex> lk(mtx);
        PendingBatch & pb = pending[il];

        if (is_topk) {
            // [n_expert_used, n_tokens] I32
            pb.n_expert_used = t->ne[0];
            pb.n_tokens      = t->ne[1];
            pb.expert_ids.resize(pb.n_expert_used * pb.n_tokens);
            memcpy(pb.expert_ids.data(), buf.data(), pb.n_expert_used * pb.n_tokens * sizeof(int32_t));
            pb.has_topk    = true;
            pb.has_weights = false; // reset in case of re-use

        } else if (is_weights) {
            // [1, n_expert_used, n_tokens] F32 — flat layout same as topk
            if (!pb.has_topk) return true; // shouldn't happen
            pb.gate_weights.resize(pb.n_expert_used * pb.n_tokens);
            memcpy(pb.gate_weights.data(), buf.data(), pb.n_expert_used * pb.n_tokens * sizeof(float));
            pb.has_weights = true;

        } else if (is_down) {
            // [n_embd, n_expert_used, n_tokens] F32
            if (!pb.has_topk || !pb.has_weights) return true;

            const int64_t n_embd        = t->ne[0];
            const int64_t n_expert_used = t->ne[1];
            const int64_t n_tokens      = t->ne[2];

            // Sanity check
            if (n_expert_used != pb.n_expert_used || n_tokens != pb.n_tokens) {
                LOG_ERR("expert-profile: dimension mismatch at layer %d\n", il);
                pending.erase(il);
                return true;
            }

            // Ensure layer stats initialised
            auto & ls = layer_stats[il];
            if (ls.n_experts == 0) ls.init(n_experts);

            const float * expert_outs = reinterpret_cast<const float *>(buf.data());
            ls.add_batch(pb.expert_ids.data(), pb.gate_weights.data(),
                         expert_outs, n_expert_used, n_tokens, n_embd);

            // Done with this batch for this layer
            pending.erase(il);
        }

        return true;
    }
};

// ─── Global collector + C callback ───────────────────────────────────────────

static ExpertCollector g_collector;

static bool expert_eval_callback(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) return g_collector.wants(t);
    return g_collector.on_tensor(t);
}

// ─── JSON output ──────────────────────────────────────────────────────────────

static void save_stats(const std::string & path) {
    std::ofstream f(path);
    if (!f) {
        LOG_ERR("expert-profile: failed to open output file '%s'\n", path.c_str());
        return;
    }

    f << "{\n";
    bool first_layer = true;
    for (auto & [il, ls] : g_collector.layer_stats) {
        if (!first_layer) f << ",\n";
        first_layer = false;

        f << "  \"" << il << "\": {\n";
        f << "    \"total_tokens\": " << ls.total_tokens << ",\n";

        // activation_counts
        f << "    \"activation_counts\": [";
        for (int64_t i = 0; i < ls.n_experts; ++i) {
            if (i) f << ", ";
            f << ls.activation_counts[i];
        }
        f << "],\n";

        // activation_frequency
        f << "    \"activation_frequency\": [";
        for (int64_t i = 0; i < ls.n_experts; ++i) {
            if (i) f << ", ";
            f << ((ls.total_tokens > 0) ? (double)ls.activation_counts[i] / ls.total_tokens : 0.0);
        }
        f << "],\n";

        // avg_gate_weight  (weighted_freq_sum / activation_counts)
        f << "    \"avg_gate_weight\": [";
        for (int64_t i = 0; i < ls.n_experts; ++i) {
            if (i) f << ", ";
            f << ((ls.activation_counts[i] > 0) ? ls.weighted_freq_sum[i] / ls.activation_counts[i] : 0.0);
        }
        f << "],\n";

        // ean_mean  = ean_sum / activation_counts  (EAN criterion, no gate weight)
        f << "    \"ean_mean\": [";
        for (int64_t i = 0; i < ls.n_experts; ++i) {
            if (i) f << ", ";
            f << ((ls.activation_counts[i] > 0) ? ls.ean_sum[i] / ls.activation_counts[i] : 0.0);
        }
        f << "],\n";

        // reap  = reap_sum / activation_counts  (REAP criterion, Eq.9)
        f << "    \"reap\": [";
        for (int64_t i = 0; i < ls.n_experts; ++i) {
            if (i) f << ", ";
            f << ((ls.activation_counts[i] > 0) ? ls.reap_sum[i] / ls.activation_counts[i] : 0.0);
        }
        f << "],\n";

        // never_activated
        int64_t never = 0;
        for (int64_t i = 0; i < ls.n_experts; ++i) {
            if (ls.activation_counts[i] == 0) ++never;
        }
        f << "    \"never_activated\": " << never << "\n";
        f << "  }";
    }
    f << "\n}\n";

    LOG_INF("expert-profile: stats saved to '%s'  (%zu MoE layers)\n",
            path.c_str(), g_collector.layer_stats.size());
}

// ─── JSONL input ──────────────────────────────────────────────────────────────

struct JsonPair { std::string prompt, response; };

static bool json_get_string(const std::string & line, const std::string & key, std::string & out) {
    std::string search = "\"" + key + "\"";
    size_t kpos = line.find(search);
    if (kpos == std::string::npos) return false;
    size_t colon = line.find(':', kpos + search.size());
    if (colon == std::string::npos) return false;
    size_t q1 = line.find('"', colon + 1);
    if (q1 == std::string::npos) return false;
    out.clear();
    for (size_t i = q1 + 1; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size()) {
            ++i;
            switch (line[i]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                default:   out += line[i]; break;
            }
        } else if (line[i] == '"') {
            return true;
        } else {
            out += line[i];
        }
    }
    return false;
}

static std::vector<JsonPair> load_jsonl(const std::string & path) {
    std::vector<JsonPair> pairs;
    std::ifstream f(path);
    if (!f) { LOG_ERR("expert-profile: cannot open JSONL file '%s'\n", path.c_str()); return pairs; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        JsonPair p;
        json_get_string(line, "prompt",   p.prompt);
        json_get_string(line, "response", p.response);
        if (!p.prompt.empty() || !p.response.empty()) pairs.push_back(std::move(p));
    }
    return pairs;
}

// ─── Inference loop ───────────────────────────────────────────────────────────

static void run_inference(llama_context * ctx,
                          const llama_model * model,
                          const std::vector<JsonPair> & pairs,
                          int max_tokens,
                          const std::string & output_path,
                          int save_every) {
    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab);

    llama_batch batch = llama_batch_init(max_tokens, 0, 1);

    for (size_t pi = 0; pi < pairs.size(); ++pi) {
        const std::string text = pairs[pi].prompt + "\n" + pairs[pi].response;

        std::vector<llama_token> tokens = common_tokenize(ctx, text, add_bos, true);
        if ((int)tokens.size() > max_tokens) tokens.resize(max_tokens);
        if (tokens.empty()) continue;

        LOG_INF("  [%zu/%zu] %zu tokens\n", pi + 1, pairs.size(), tokens.size());

        llama_memory_clear(llama_get_memory(ctx), true);

        common_batch_clear(batch);
        for (int i = 0; i < (int)tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("  [%zu/%zu] llama_decode failed — skipping\n", pi + 1, pairs.size());
        }

        if (save_every > 0 && (pi + 1) % save_every == 0) {
            save_stats(output_path);
        }
    }

    llama_batch_free(batch);
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    std::string model_path;
    std::string jsonl_path;
    std::string output_path  = "expert_stats.json";
    int         n_experts    = 128;
    int         ctx_size     = 2048;
    int         n_gpu_layers = 99;
    int         n_threads    = 4;
    int         save_every   = 100;
    enum ggml_type kv_type_k = GGML_TYPE_F16;
    enum ggml_type kv_type_v = GGML_TYPE_F16;

    auto parse_ggml_type = [](const char * s) -> enum ggml_type {
        if (strcmp(s, "f32")  == 0) return GGML_TYPE_F32;
        if (strcmp(s, "f16")  == 0) return GGML_TYPE_F16;
        if (strcmp(s, "q8_0") == 0) return GGML_TYPE_Q8_0;
        if (strcmp(s, "q4_0") == 0) return GGML_TYPE_Q4_0;
        fprintf(stderr, "Unknown KV type '%s', using f16\n", s); return GGML_TYPE_F16;
    };

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i]); exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")           model_path  = next();
        else if (a == "--jsonl")                         jsonl_path  = next();
        else if (a == "--output")                        output_path = next();
        else if (a == "--n-experts")                     n_experts    = atoi(next());
        else if (a == "--ctx-size" || a == "-c")         ctx_size     = atoi(next());
        else if (a == "-ngl" || a == "--n-gpu-layers")   n_gpu_layers = atoi(next());
        else if (a == "-t" || a == "--threads")          n_threads    = atoi(next());
        else if (a == "--type-k")                        kv_type_k   = parse_ggml_type(next());
        else if (a == "--type-v")                        kv_type_v   = parse_ggml_type(next());
        else if (a == "--save-every")                    save_every  = atoi(next());
        else if (a == "-h" || a == "--help") {
            fprintf(stderr,
                "\nUsage: %s -m model.gguf --jsonl data.jsonl [options]\n"
                "  --output PATH       Output JSON (default: expert_stats.json)\n"
                "  --n-experts N       Experts per layer (default: 128)\n"
                "  --ctx-size N        Context length (default: 2048)\n"
                "  -ngl N              GPU layers (default: 99)\n"
                "  -t N                CPU threads (default: 4)\n"
                "  --type-k/v TYPE     KV cache type: f32/f16/q8_0/q4_0 (default: f16)\n"
                "  --save-every N      Checkpoint every N samples (default: 100)\n\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", a.c_str()); return 1;
        }
    }

    if (model_path.empty()) { fprintf(stderr, "Error: -m required\n"); return 1; }
    if (jsonl_path.empty()) { fprintf(stderr, "Error: --jsonl required\n"); return 1; }

    g_collector.n_experts = n_experts;

    LOG_INF("expert-profile: model     = %s\n", model_path.c_str());
    LOG_INF("expert-profile: jsonl     = %s\n", jsonl_path.c_str());
    LOG_INF("expert-profile: output    = %s\n", output_path.c_str());
    LOG_INF("expert-profile: n_experts = %d\n", n_experts);
    LOG_INF("expert-profile: ctx_size  = %d\n", ctx_size);
    LOG_INF("expert-profile: ngl       = %d\n", n_gpu_layers);
    LOG_INF("expert-profile: criterion = REAP (gate_weight * ||expert_out||_2)\n");

    auto pairs = load_jsonl(jsonl_path);
    if (pairs.empty()) { LOG_ERR("expert-profile: no pairs loaded\n"); return 1; }
    LOG_INF("expert-profile: loaded %zu pairs\n", pairs.size());

    llama_backend_init();

    // Suppress INFO/WARN spam (CUDA graph warmup etc.), only pass errors through
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { LOG_ERR("expert-profile: failed to load model\n"); return 1; }

    llama_context_params cparams  = llama_context_default_params();
    cparams.n_ctx                 = ctx_size;
    cparams.n_batch               = ctx_size;
    cparams.n_ubatch              = std::min(ctx_size, 512);
    cparams.n_threads             = n_threads;
    cparams.type_k                = kv_type_k;
    cparams.type_v                = kv_type_v;
    cparams.cb_eval               = expert_eval_callback;
    cparams.cb_eval_user_data     = nullptr;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { LOG_ERR("expert-profile: failed to create context\n"); return 1; }

    LOG_INF("expert-profile: running forward passes...\n");
    run_inference(ctx, model, pairs, ctx_size, output_path, save_every);
    save_stats(output_path);

    // Summary
    LOG_INF("\n  MoE layers profiled: %zu\n", g_collector.layer_stats.size());
    for (auto & [il, ls] : g_collector.layer_stats) {
        // Find top and bottom REAP expert
        int64_t top_e = 0, bot_e = 0;
        double  top_v = 0.0, bot_v = 1e18;
        for (int64_t i = 0; i < ls.n_experts; ++i) {
            double v = (ls.activation_counts[i] > 0) ? ls.reap_sum[i] / ls.activation_counts[i] : 0.0;
            if (v > top_v) { top_v = v; top_e = i; }
            if (v < bot_v) { bot_v = v; bot_e = i; }
        }
        int64_t never = 0;
        for (int64_t i = 0; i < ls.n_experts; ++i)
            if (ls.activation_counts[i] == 0) ++never;
        LOG_INF("  Layer %3d: tokens=%lld  never=%lld  reap_top=e%lld(%.4f)  reap_bot=e%lld(%.4f)\n",
                il, (long long)ls.total_tokens, (long long)never,
                (long long)top_e, top_v, (long long)bot_e, bot_v);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
