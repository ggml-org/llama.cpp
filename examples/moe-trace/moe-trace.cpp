// moe-trace: log which routed experts a MoE model selects, per layer, for every
// evaluated token.
//
// Motivation: expert-routing statistics decide whether hot-expert strategies
// (caching, pinning, offload placement) can work for a given model - a model
// with near-uniform routing has no exploitable hot set, while a skewed router
// rewards keeping its favourite experts resident. This tool measures that
// distribution on a real workload instead of guessing.
//
// It hooks the scheduler eval callback and captures only the "ffn_moe_topk"
// id tensors (one tiny i32 tensor per MoE layer per ubatch), so tracing adds
// negligible overhead to the evaluation itself.
//
// Output (MOE_TRACE_OUT, default moe_trace.csv): one "layer,expert" row per
// routed activation. Aggregate however you like, e.g.:
//   sort moe_trace.csv | uniq -c | sort -rn | head    # hottest (layer,expert)
//
// Usage:
//   llama-moe-trace -m model.gguf -f prompt.txt -n 64 [usual common flags]
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static FILE * g_out = nullptr;

// scheduler eval callback: called once with ask=true (do you want this
// tensor's data?) and, if accepted, once with ask=false after computation
static bool moe_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    const bool is_topk = strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (ask) {
        return is_topk;
    }
    if (!is_topk || t->type != GGML_TYPE_I32) {
        return true;
    }
    // layer index is the suffix after the last '-' in "ffn_moe_topk-<il>"
    int il = -1;
    if (const char * dash = strrchr(t->name, '-')) {
        il = atoi(dash + 1);
    }
    const int64_t n = ggml_nelements(t);
    std::vector<int32_t> ids(n);
    ggml_backend_tensor_get(t, ids.data(), 0, n * sizeof(int32_t));
    for (int64_t i = 0; i < n; ++i) {
        fprintf(g_out, "%d,%" PRId32 "\n", il, ids[i]);
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

    const char * out_path = getenv("MOE_TRACE_OUT");
    g_out = fopen(out_path ? out_path : "moe_trace.csv", "w");
    if (!g_out) {
        LOG_ERR("%s: cannot open trace output\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_cb;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt,
                                                      llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt - provide one with -p or -f\n", __func__);
        return 1;
    }
    LOG_INF("%s: %zu prompt tokens, n_predict = %d\n", __func__, tokens.size(), params.n_predict);

    // prefill in ubatch-sized chunks
    const int step = params.n_ubatch > 0 ? params.n_ubatch : 512;
    for (size_t i = 0; i < tokens.size(); i += step) {
        const int n = (int) std::min((size_t) step, tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("%s: decode failed at token %zu\n", __func__, i);
            return 1;
        }
        LOG_INF("  prefill %zu/%zu\n", i + n, tokens.size());
        fflush(g_out);
    }

    // short greedy decode so pure-decode routing is represented as well
    for (int i = 0; i < params.n_predict; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = 0;
        float best_v = logits[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > best_v) { best_v = logits[v]; best = v; }
        }
        if (llama_vocab_is_eog(vocab, best)) {
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(&best, 1))) {
            break;
        }
        if ((i + 1) % 8 == 0) {
            LOG_INF("  decode %d/%d\n", i + 1, params.n_predict);
            fflush(g_out);
        }
    }

    fclose(g_out);
    llama_backend_free();
    LOG_INF("%s: done\n", __func__);
    return 0;
}
