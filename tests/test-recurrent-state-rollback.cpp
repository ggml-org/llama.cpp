#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "log.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <vector>

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[m";
        case test_status::SKIP: return "\033[[1;33mSKIP\033[m";
    }
    return "";
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx.get());
    if (!decode_tokens(ctx.get(), std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        return nullptr;
    }
    llama_synchronize(ctx.get());
    cache_buffer_collector collector;
    llama_get_memory(ctx.get())->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    if (collector.buffers.empty()) {
        LOG_ERR("%s: no cache buffers found\n", __func__);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context_ptr make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context_ptr ctx_roll = make_ctx_multi();
    llama_context_ptr ctx_ref  = make_ctx_multi();
    if (!ctx_roll || !ctx_ref) {
        LOG_ERR("%s: failed to init multi-seq contexts\n", __func__);
        return false;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll.get(), batch) == 0;
        ok = ok && llama_decode(ctx_ref.get(),  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll.get(), batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        LOG_ERR("%s: multi-seq prefill/rollback failed\n", __func__);
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll.get(), batch) == 0;
    ok = ok && llama_decode(ctx_ref.get(), batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        LOG_ERR("%s: multi-seq replay decode failed\n", __func__);
        return false;
    }

    // identical ubatch shapes should produce identical states, but the larger
    // stdev makes the model sensitive to backend scheduling/rounding noise
    constexpr float nmse_eps = 1e-5f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            LOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        LOG_ERR("%s: multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        return false;
    }

    LOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref.get(), batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll.get(), batch_one) == 0;
        ok = ok && llama_decode(ctx_ref.get(), batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        LOG_ERR("%s: seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        return false;
    }

    LOG_INF("%s: seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    return true;
}

static test_status test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_ptr ctx_src = make_ctx(params, model, fill);
    llama_context_ptr ctx_dst = make_ctx(params, model, fill);
    if (!ctx_src || !ctx_dst) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_src.get()) == 0) {
        LOG_INF("%s: skipping because n_rs_seq is disabled\n", __func__);
        return test_status::SKIP;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }
    if (tokens.empty()) {
        LOG_ERR("%s: not enough prompt tokens\n", __func__);
        return test_status::FAIL;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src.get(), tokens, n_tokens)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src.get(), 0, 0);
    ckpt.load_tgt(ctx_dst.get(), 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                LOG_ERR("%s: %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                LOG_ERR("%s: missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                LOG_ERR("%s: %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return test_status::FAIL;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context_ptr ctx_dirty = make_ctx(params, model, fill);
    if (!ctx_dirty) {
        LOG_ERR("%s: failed to init dirty ctx\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty.get(), noise, n_tokens)) {
        LOG_ERR("%s: dirty prompt decode failed\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: dirty rollback failed\n", __func__);
        return test_status::FAIL;
    }

    ckpt.load_tgt(ctx_dirty.get(), 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
            LOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
        if (logits_dirty == nullptr) {
            LOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            LOG_ERR("%s: dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return test_status::FAIL;
        }
    }

    LOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return test_status::FAIL;
    }

    return test_status::PASS;
}

// Run the rollback tests for a single model.
// Returns the per-model status.
static test_status run_rollback_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        return test_status::SKIP;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        LOG_INF("%s: skipping for non-recurrent model\n", __func__);
        return test_status::SKIP;
    }

    test_status status = test_status::SKIP;
    for (uint8_t fill : { 0, 0x3e }) {
        LOG_INF("%s: testing with cache fill 0x%02x\n", __func__, fill);
        const test_status fill_status = test_rollback(params, model, fill);
        if (fill_status == test_status::FAIL) {
            return test_status::FAIL;
        }
        if (fill_status == test_status::PASS) {
            status = test_status::PASS;
        }
    }

    return status;
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    ggml_backend_load_all();

    if (!models_dir.empty()) {
        // run the rollback tests over every dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        size_t name_width = 5; // "Model"
        for (const auto & model_path : models) {
            name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
        }

        // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
        common_log_set_verbosity_thold(0);

        LOG("%-*s  %s\n", (int) name_width, "Model", "rollback");
        common_log_flush(common_log_main());

        size_t n_pass = 0;
        size_t n_skip = 0;
        size_t n_fail = 0;
        for (const auto & model_path : models) {
            const auto name = std::filesystem::path(model_path).filename().string();

            LOG("%-*s", (int) name_width, name.c_str());

            const test_status status = run_rollback_tests_for_model(model_path, params);
            LOG("  %s", test_status_str(status));
            LOG("\n");
            common_log_flush(common_log_main());

            switch (status) {
                case test_status::PASS: n_pass++; break;
                case test_status::FAIL: n_fail++; break;
                case test_status::SKIP: n_skip++; break;
            }
        }

        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
        common_log_flush(common_log_main());

        LOG_INF("%s: summary: %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass, n_skip, n_fail, models.size());

        return n_fail == 0 ? 0 : 1;
    }

    // single-model mode
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        LOG_ERR("%s: failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        LOG_INF("%s: skipping for non-recurrent model\n", __func__);
        return 0;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        LOG_INF("%s: testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) == test_status::FAIL) {
            return 1;
        }
    }

    return 0;
}
