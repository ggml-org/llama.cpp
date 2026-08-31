// Standalone repro for the qwen35-mla / MLA long-context Vulkan flash-attention
// corruption: unlike test-backend-ops' single-op FLASH_ATTN_EXT test (which only
// shows a small ~0.1% numerical discrepancy at large kv), the real corruption only
// appears when the *same compiled graph* gets reused across *many separate
// llama_decode() submissions* as the KV cache grows incrementally -- exactly how
// llama-perplexity actually drives the model. This reproduces that exact pattern
// in a minimal driver, with a per-tensor eval callback (host-side, reads back from
// GPU via ggml_backend_tensor_get -- no GPU-side debugPrintfEXT needed, so no
// buffer-flush issues) watching a specific MLA attention tensor's running sum
// across every decode step.
#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include <clocale>
#include <cstdio>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p' or '-f')\n", __func__);
        return false;
    }

    const int n_batch = params.n_batch;
    LOG_INF("total tokens = %zu, n_batch = %d, decoding in %zu chunks\n",
            tokens.size(), n_batch, (tokens.size() + n_batch - 1) / n_batch);

    size_t n_done = 0;
    int chunk = 0;
    while (n_done < tokens.size()) {
        const size_t n_this = std::min((size_t) n_batch, tokens.size() - n_done);
        LOG_INF("=== chunk %d: tokens [%zu, %zu) ===\n", chunk, n_done, n_done + n_this);

        llama_batch batch = llama_batch_get_one(tokens.data() + n_done, n_this);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval chunk %d\n", __func__, chunk);
            return false;
        }

        n_done += n_this;
        ++chunk;
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

    // pass the callback to the backend scheduler; it fires for each node
    // during graph computation, for every llama_decode() call -- i.e. once
    // per chunk, not just once overall. Filter to the MLA attention output
    // tensor (named via cb(cur, "attn_pregate", il) in qwen35-mla.cpp) so the
    // dump stays readable across many chunks; override via MLA_REPRO_FILTER.
    // Constructed in place (not default-constructed then reassigned): the
    // constructor stashes `this` into params.cb_eval_user_data, so building a
    // temporary and move-assigning it into a longer-lived variable leaves
    // that pointer dangling the moment the temporary is destroyed -- crashes
    // on the very next callback invocation. Bit me once already.
    const char * filter_env = getenv("MLA_REPRO_FILTER");
    std::vector<std::string> filters = { filter_env ? filter_env : "attn_pregate" };
    base_callback_data cb_data(params, filters);
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
