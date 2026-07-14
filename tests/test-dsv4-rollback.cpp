// DSV4 bounded partial rollback.
//
// The DSV4 compressed caches are derived from a compressor-state ring addressed by pos%state_size.
// Rolling back the last k tokens is only sound if the rows those tokens wrote had not aliased onto
// rows an unfinished block still has to read. This checks exactly that: decode a prompt, decode k
// wrong tokens on top (a rejected draft), roll them back, decode the real k tokens, and require the
// resulting logits to match a context that decoded the real tokens all along.
//
// An end-to-end speculative-vs-greedy comparison cannot serve as this gate: the target evaluates a
// draft as one batch, which reorders float reductions, so greedy speculative output may legitimately
// differ from greedy non-speculative output on near-ties regardless of rollback correctness.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>

static constexpr uint32_t N_ROLLBACK = 5;

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = N_ROLLBACK;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_range(llama_context * ctx, const std::vector<llama_token> & tokens,
                         size_t first, size_t last, bool logits_last) {
    llama_batch batch = llama_batch_init(last - first, 0, 1);
    for (size_t pos = first; pos < last; ++pos) {
        common_batch_add(batch, tokens[pos], (llama_pos) pos, { 0 }, logits_last && pos + 1 == last);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    llama_context * ctx_ref = make_ctx(params, model);
    llama_context * ctx_rb  = make_ctx(params, model);
    if (ctx_ref == nullptr || ctx_rb == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_ref) == 0) {
        fprintf(stderr, "%s : skipping, the model does not support bounded rollback\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // The rolled-back span has to complete a block of every compressed cache, otherwise the rollback
    // of that cache is never exercised: CSA and the indexer compress every 4 positions, but HCA only
    // every 128. Decode N_TOKENS positions and remove the last N_ROLLBACK, so that the HCA boundary
    // at position DSV4_HCA_RATIO - 1 falls inside the removed span and has to be recomputed.
    constexpr size_t DSV4_HCA_RATIO = 128;
    constexpr size_t N_TOKENS       = DSV4_HCA_RATIO + 2;                 // 130
    constexpr size_t N_KEEP         = N_TOKENS - N_ROLLBACK;              // 125
    static_assert(N_KEEP <= DSV4_HCA_RATIO - 1 && DSV4_HCA_RATIO - 1 < N_TOKENS,
            "the HCA block boundary must fall inside the rolled-back span");

    std::vector<llama_token> tokens;
    while (tokens.size() < N_TOKENS) {
        std::vector<llama_token> chunk = common_tokenize(ctx_ref,
                "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
                "How vexingly quick daft zebras jump! The five boxing wizards jump quickly. ",
                tokens.empty());
        tokens.insert(tokens.end(), chunk.begin(), chunk.end());
    }
    tokens.resize(N_TOKENS);

    const size_t n_keep = N_KEEP;

    // reference: decode the prompt as prefix + the real tail
    if (!decode_range(ctx_ref, tokens, 0, n_keep, false) ||
        !decode_range(ctx_ref, tokens, n_keep, tokens.size(), true)) {
        fprintf(stderr, "%s : reference decode failed\n", __func__);
        return 1;
    }

    // rollback: same prefix, then a wrong tail (a rejected draft), rolled back, then the real tail
    std::vector<llama_token> wrong = tokens;
    for (size_t i = n_keep; i < wrong.size(); ++i) {
        wrong[i] = (wrong[i] + 1) % n_vocab;
    }

    if (!decode_range(ctx_rb, tokens, 0, n_keep, false) ||
        !decode_range(ctx_rb, wrong, n_keep, wrong.size(), false)) {
        fprintf(stderr, "%s : draft decode failed\n", __func__);
        return 1;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_rb), 0, (llama_pos) n_keep, -1)) {
        fprintf(stderr, "%s : rollback of %u tokens was refused\n", __func__, N_ROLLBACK);
        return 1;
    }

    if (!decode_range(ctx_rb, tokens, n_keep, tokens.size(), true)) {
        fprintf(stderr, "%s : replay decode failed\n", __func__);
        return 1;
    }

    const float * logits_ref = llama_get_logits_ith(ctx_ref, -1);
    const float * logits_rb  = llama_get_logits_ith(ctx_rb,  -1);
    if (logits_ref == nullptr || logits_rb == nullptr) {
        fprintf(stderr, "%s : missing logits\n", __func__);
        return 1;
    }

    constexpr float eps = 1e-3f;

    float max_diff = 0.0f;
    int   arg_ref = 0;
    int   arg_rb  = 0;
    for (int i = 0; i < n_vocab; ++i) {
        max_diff = std::max(max_diff, std::fabs(logits_ref[i] - logits_rb[i]));
        if (logits_ref[i] > logits_ref[arg_ref]) { arg_ref = i; }
        if (logits_rb [i] > logits_rb [arg_rb ]) { arg_rb  = i; }
    }

    fprintf(stderr, "%s : max logit diff = %g, argmax ref = %d, argmax rollback = %d\n",
            __func__, (double) max_diff, arg_ref, arg_rb);

    if (max_diff > eps || arg_ref != arg_rb) {
        fprintf(stderr, "%s : FAILED - rollback did not reconstruct the reference state\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : bounded rollback reconstructs the reference state\n", __func__);

    llama_free(ctx_ref);
    llama_free(ctx_rb);
    return 0;
}
