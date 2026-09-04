// Teacher-forced agreement between the MTP (NextN) head and the trunk.
//
// The trunk runs over a text with every token as an output. The head runs over the same
// tokens, fed with the trunk hidden state of the previous position, as the speculative
// driver feeds it. At position k both predict token k+1. The head's draft is accepted by a
// greedy trunk iff the two argmaxes agree, so the agreement rate is the depth-1 acceptance
// rate on a fixed text, free of the sampling and batch-shape effects of a live server.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "speculative.h"
#include "../../src/llama-ext.h" // staging API: llama_set_embeddings_nextn, llama_get_embeddings_nextn

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

static int argmax(const float * x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return best;
}

static double log_sum_exp(const float * x, int n, int i_max) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp((double) x[i] - x[i_max]);
    }
    return x[i_max] + std::log(sum);
}

struct agree_stats {
    int64_t n_pos      = 0; // positions scored
    int64_t n_agree    = 0; // head argmax == trunk argmax
    int64_t n_prop     = 0; // head confident enough to draft (p >= p_min)
    int64_t n_prop_ok  = 0; // ... and agrees with the trunk
    int64_t n_tgt_hit  = 0; // trunk argmax == corpus token
    int64_t n_mtp_hit  = 0; // head argmax == corpus token
    double  kld_sum    = 0; // sum of KL(trunk || head)

    void add(const float * lt, const float * lh, int n_vocab, llama_token truth, float p_min) {
        const int at = argmax(lt, n_vocab);
        const int ah = argmax(lh, n_vocab);

        const double lse_t = log_sum_exp(lt, n_vocab, at);
        const double lse_h = log_sum_exp(lh, n_vocab, ah);

        double kld = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            const double logp_t = lt[i] - lse_t;
            kld += std::exp(logp_t) * (logp_t - (lh[i] - lse_h));
        }

        const bool agree = at == ah;
        const bool prop  = std::exp(lh[ah] - lse_h) >= p_min;

        n_pos     += 1;
        n_agree   += agree;
        n_prop    += prop;
        n_prop_ok += prop && agree;
        n_tgt_hit += at == truth;
        n_mtp_hit += ah == truth;
        kld_sum   += kld;
    }

    void print(float p_min) const {
        const double n = (double) n_pos;
        LOG("\n");
        LOG("positions scored      : %" PRId64 "\n", n_pos);
        LOG("head == trunk (argmax): %.4f\n", n_agree / n);
        LOG("proposed at p_min %.2f : %.4f of positions, %.4f of those agree\n", p_min, n_prop / n, n_prop_ok / (double) std::max<int64_t>(1, n_prop));
        LOG("trunk hits corpus     : %.4f\n", n_tgt_hit / n);
        LOG("head  hits corpus     : %.4f\n", n_mtp_hit / n);
        LOG("mean KL(trunk || head): %.4f\n", kld_sum / n);
    }
};

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx = 1024;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 0; // nothing is drafted, so no rollback snapshots
    params.speculative.draft.p_min = 0.8f;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }

    if (llama_model_n_layer_nextn(model) <= 0) {
        LOG_ERR("%s: the model has no MTP (NextN) head\n", __func__);
        return 1;
    }

    common_speculative_init_result_ptr mtp_init = common_speculative_init_from_params(params, model, ctx);

    llama_context * ctx_mtp = mtp_init->context();
    if (ctx_mtp == nullptr) {
        LOG_ERR("%s: failed to create the MTP context\n", __func__);
        return 1;
    }

    // every token is an output, so the masked rows cover the whole batch in batch order
    llama_set_embeddings_nextn(ctx, true, /*masked*/ true);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_embd  = llama_model_n_embd_out(model);
    const int n_ctx   = llama_n_ctx(ctx);

    const float p_min = params.speculative.draft.p_min;

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    int n_chunk = tokens.size() / n_ctx;
    if (params.n_chunks > 0) {
        n_chunk = std::min(n_chunk, params.n_chunks);
    }

    if (n_chunk < 1) {
        LOG_ERR("%s: need at least %d tokens, got %zu\n", __func__, n_ctx, tokens.size());
        return 1;
    }

    LOG_INF("%s: %d chunks of %d tokens, p_min %.2f\n", __func__, n_chunk, n_ctx, p_min);

    llama_batch batch_tgt = llama_batch_init(n_ctx, 0, 1);

    // llama_batch_init allocates only one of token/embd; the head needs both
    llama_batch batch_mtp = llama_batch_init(n_ctx, n_embd, 1);
    batch_mtp.token = (llama_token *) malloc(sizeof(llama_token) * n_ctx);

    const size_t row_bytes = (size_t) n_embd * sizeof(float);

    agree_stats stats;

    for (int i = 0; i < n_chunk; ++i) {
        const llama_token * chunk = tokens.data() + (size_t) i * n_ctx;

        llama_memory_clear(llama_get_memory(ctx),     true);
        llama_memory_clear(llama_get_memory(ctx_mtp), true);

        common_batch_clear(batch_tgt);
        for (int k = 0; k < n_ctx; ++k) {
            common_batch_add(batch_tgt, chunk[k], k, { 0 }, true);
        }

        if (llama_decode(ctx, batch_tgt) != 0) {
            LOG_ERR("%s: trunk decode failed on chunk %d\n", __func__, i);
            return 1;
        }

        // row k of the head takes token k and the trunk state of position k-1; position 0 has no predecessor
        const float * h = llama_get_embeddings_nextn(ctx);

        common_batch_clear(batch_mtp);
        for (int k = 1; k < n_ctx; ++k) {
            std::memcpy(batch_mtp.embd + (size_t) (k - 1) * n_embd, h + (size_t) (k - 1) * n_embd, row_bytes);
            common_batch_add(batch_mtp, chunk[k], k, { 0 }, true);
        }

        if (llama_decode(ctx_mtp, batch_mtp) != 0) {
            LOG_ERR("%s: head decode failed on chunk %d\n", __func__, i);
            return 1;
        }

        const int64_t n_pos_before   = stats.n_pos;
        const int64_t n_agree_before = stats.n_agree;

        // both predict token k+1; the last position has no corpus token to check against
        for (int k = 1; k < n_ctx - 1; ++k) {
            stats.add(llama_get_logits_ith(ctx, k), llama_get_logits_ith(ctx_mtp, k - 1), n_vocab, chunk[k + 1], p_min);
        }

        // per-chunk counts, for paired tests between variants on the same text
        LOG_INF("%s: chunk %d/%d: agree %" PRId64 " of %" PRId64 ", cumulative %.4f\n", __func__, i + 1, n_chunk,
                stats.n_agree - n_agree_before, stats.n_pos - n_pos_before, stats.n_agree / (double) stats.n_pos);
    }

    stats.print(p_min);

    free(batch_mtp.token);
    batch_mtp.token = nullptr;
    llama_batch_free(batch_mtp);
    llama_batch_free(batch_tgt);

    llama_backend_free();

    return 0;
}
