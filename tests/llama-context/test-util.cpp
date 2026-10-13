#include "test-util.h"

#include "test-log.h"

#include "../../src/llama-ext.h" // llama_memory_get_buffers

#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

const double NMSE_THRESHOLD = 1e-5;

//
// comparison
//

// the distance of two logits, infinite when either of them is not finite
static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

void logits_diff::add(const float * a, const float * b, size_t n_elem) {
    for (size_t i = 0; i < n_elem; i++) {
        const float diff = logit_diff(a[i], b[i]);

        if (diff > 0.0f && i_first < 0) {
            i_first = (int64_t) (n + i);
        }

        if (diff > diff_max) {
            diff_max = diff;
        }

        if (std::isfinite(a[i]) && std::isfinite(b[i])) {
            const double d = (double) a[i] - (double) b[i];

            mse_ab += d*d;
            mse_a0 += (double) a[i] * (double) a[i];
        } else {
            n_nan++;
            mse_ab = std::numeric_limits<double>::infinity();
            mse_a0 = 1.0;
        }
    }

    n += n_elem;
}

double logits_diff::nmse() const {
    if (mse_a0 == 0.0) {
        return mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    }

    return mse_ab / mse_a0;
}

logits_diff compare_logits(const float * a, const float * b, size_t n) {
    logits_diff res;
    res.add(a, b, n);

    return res;
}

//
// decode
//

bool get_current_logits(llama_context * ctx, std::vector<float> & out) {
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return false;
    }
    out.assign(logits, logits + n_vocab);
    return true;
}

bool decode_seq(llama_context * ctx, const llama_tokens & tokens, llama_pos p_begin, llama_pos p_end,
                llama_seq_id seq_id, bool last_output) {
    common_batch batch(ctx);
    for (llama_pos pos = p_begin; pos < p_end; ++pos) {
        batch.add(tokens[pos], pos, seq_id, last_output && pos + 1 == p_end);
    }

    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

bool decode_tokens(llama_context * ctx, const llama_tokens & tokens, llama_seq_id seq_id) {
    return decode_seq(ctx, tokens, 0, (llama_pos) tokens.size(), seq_id, true);
}

llama_token gen_token(int32_t n_vocab, llama_seq_id seq_id, llama_pos pos) {
    return (llama_token) ((7*(uint32_t) pos + 31*(uint32_t) seq_id + 1) % (uint32_t) n_vocab);
}

bool decode_gen(llama_context * ctx, int32_t n_vocab, llama_seq_id seq_id, llama_pos p_begin, llama_pos p_end) {
    common_batch batch(ctx);
    for (llama_pos pos = p_begin; pos < p_end; ++pos) {
        batch.add(gen_token(n_vocab, seq_id, pos), pos, seq_id, false);
    }

    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq_id) {
    common_batch batch(ctx);
    batch.add(tok, pos, seq_id, true);

    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

bool generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
                     generation_result & result) {
    common_batch batch(ctx);

    // the tokens of the run as a single line - one entry per token would leave the next message
    //   of the log glued to the last one
    std::string tokens;

    for (int32_t i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            TLOG_ERR("%s: failed to get logits\n", __func__);
            return false;
        }

        const auto next_token = llama_sampler_sample(smpl, ctx, -1);

        tokens += std::to_string(next_token) + " ";
        result.tokens.push_back(next_token);
        result.logits.push_back(std::move(logits));

        batch.clear();
        batch.add(next_token, n_past, seq_id, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            TLOG_ERR("%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    TLOG_INF("%s: %s\n", __func__, tokens.c_str());

    return true;
}

bool generate_tokens_compare(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
                             const generation_result & expected, double nmse_eps) {
    if (expected.tokens.size() != expected.logits.size() || expected.tokens.size() < (size_t) n_predict) {
        TLOG_ERR("%s: invalid expected generation\n", __func__);
        return false;
    }

    common_batch batch(ctx);

    std::string tokens;

    for (int32_t i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            TLOG_ERR("%s: failed to get logits\n", __func__);
            return false;
        }
        if (logits.size() != expected.logits[i].size()) {
            TLOG_ERR("%s: logits size mismatch at step %d: %zu != %zu\n", __func__, i, logits.size(), expected.logits[i].size());
            return false;
        }

        const logits_diff diff = compare_logits(expected.logits[i].data(), logits.data(), logits.size());

        TLOG_TRC("%s: step %d nmse = %.6e\n", __func__, i, diff.nmse());
        if (diff.nmse() > nmse_eps) {
            TLOG_ERR("%s: error: NMSE at step %d is %.6e (threshold %.1e)\n", __func__, i, diff.nmse(), nmse_eps);
            return false;
        }

        const auto next_token = llama_sampler_sample(smpl, ctx, -1);
        const auto expected_token = expected.tokens[i];

        tokens += std::to_string(next_token) + " ";
        if (next_token != expected_token) {
            TLOG_TRC("%s: sampled token %d differs from expected %d, using expected token\n", __func__, next_token, expected_token);
        }

        batch.clear();
        batch.add(expected_token, n_past, seq_id, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            TLOG_ERR("%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    TLOG_INF("%s: %s\n", __func__, tokens.c_str());

    return true;
}

//
// state
//

bool get_seq_state(llama_context * ctx, llama_seq_id seq_id, uint32_t flags, std::vector<uint8_t> & state) {
    const size_t state_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);
    if (state_size == 0) {
        TLOG_ERR("%s: sequence state is empty\n", __func__);
        return false;
    }

    state.resize(state_size);
    const size_t ncopy = llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq_id, flags);
    if (ncopy != state.size()) {
        TLOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                __func__, ncopy, state.size());
        return false;
    }

    return true;
}

bool set_seq_state(llama_context * ctx, llama_seq_id seq_id, uint32_t flags, const std::vector<uint8_t> & state) {
    const size_t nset = llama_state_seq_set_data_ext(ctx, state.data(), state.size(), seq_id, flags);
    if (nset != state.size()) {
        TLOG_ERR("%s: seq set data length %zu does not match expected length %zu\n", __func__, nset, state.size());
        return false;
    }

    return true;
}

bool seq_rm(llama_context * ctx, llama_seq_id seq_id) {
    if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1)) {
        TLOG_ERR("%s: failed to remove sequence %d\n", __func__, seq_id);
        return false;
    }

    return true;
}

bool load_state_replay_last(llama_context * ctx, const std::string & path, const llama_tokens & tokens, int & n_past) {
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx, path.c_str(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        TLOG_ERR("%s: failed to load state\n", __func__);
        return false;
    }

    TLOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // the session is saved before the last token, so replay it at the position it was saved at
    n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx, tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    return true;
}

bool blobs_equal(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    if (a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size()) == 0)) {
        return true;
    }

    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) {
        ++i;
    }

    TLOG_ERR("%s: %zu bytes -> %zu bytes, first difference at byte %zu (%02x -> %02x)\n",
            __func__, a.size(), b.size(), i,
            i < a.size() ? a[i] : 0,
            i < b.size() ? b[i] : 0);

    return false;
}

bool corrupt_state(std::vector<uint8_t> & data) {
    if (data.size() < 3*4096) {
        TLOG_ERR("%s: state of %zu bytes is too small to corrupt\n", __func__, data.size());
        return false;
    }

    std::fill(data.begin() + 4096, data.end() - data.size()/4, 0xff);
    return true;
}

//
// context
//

// fill the buffers of the cache before a restore - a restore that reads cells it never wrote
// then produces garbage instead of silently passing
llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx.get());
    if (!decode_tokens(ctx.get(), llama_tokens(n_tokens, 0))) {
        return nullptr;
    }
    llama_synchronize(ctx.get());

    const auto buffers = llama_memory_get_buffers(llama_get_memory(ctx.get()));
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    if (buffers.empty()) {
        TLOG_ERR("%s: no cache buffers found\n", __func__);
        return nullptr;
    }
    for (auto * buffer : buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

llama_context_ptr make_ctx(llama_model * model, const common_params & params, const ctx_opts & opts) {
    auto cparams = common_context_params_to_llama(params);

    if (opts.n_ctx) {
        cparams.n_ctx = opts.n_ctx;
    }
    if (opts.n_seq_max) {
        cparams.n_seq_max = opts.n_seq_max;
    }
    if (opts.kv_unified >= 0) {
        cparams.kv_unified = opts.kv_unified != 0;
    }
    if (opts.n_rs_seq) {
        cparams.n_rs_seq = opts.n_rs_seq;
        cparams.n_batch  = std::max(cparams.n_batch,  (uint32_t) (opts.n_rs_seq + 1));
        cparams.n_ubatch = std::max(cparams.n_ubatch, (uint32_t) (opts.n_rs_seq + 1));
    }
    if (opts.n_batch) {
        cparams.n_batch = opts.n_batch;
    }
    if (opts.n_ubatch) {
        cparams.n_ubatch = opts.n_ubatch;
    }
    if (opts.fa != LLAMA_FLASH_ATTN_TYPE_AUTO) {
        cparams.flash_attn_type = opts.fa;
    }
    if (opts.type_k != GGML_TYPE_COUNT) {
        cparams.type_k = opts.type_k;
    }
    if (opts.type_v != GGML_TYPE_COUNT) {
        cparams.type_v = opts.type_v;
    }

    return init_ctx(model, cparams, opts.fill);
}

llama_sampler_ptr make_dist_sampler(uint32_t seed) {
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler_ptr smpl{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(seed));
    return smpl;
}

bool arch_reserves_single_seq(const llama_model * model) {
    char arch_str[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch_str, sizeof(arch_str));

    return strcmp(arch_str, "kimi-linear") == 0 || strcmp(arch_str, "minimax-01") == 0;
}

//
// model_run
//

llama_context_ptr model_run::ctx(const ctx_opts & opts) const {
    ctx_opts effective  = opts;
    effective.fill      = fill;

    auto ctx = make_ctx(model, params, effective);
    if (!ctx && !opts.quiet) {
        TLOG_ERR("%s: failed to create context for '%s'\n", __func__, name.c_str());
    }

    return ctx;
}

llama_sampler_ptr model_run::sampler() const {
    return make_dist_sampler(params.sampling.seed);
}

bool model_run::is_recurrent() const {
    return llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
}

bool model_run::has_vocab() const {
    return llama_vocab_type(llama_model_get_vocab(model)) != LLAMA_VOCAB_TYPE_NONE;
}
