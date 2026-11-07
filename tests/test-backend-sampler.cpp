#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "common.h"
#include "sampling.h"
#include "../src/llama-ext.h"
#include "../src/llama-context.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <array>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct test_args {
    std::string model;
    std::string test;
    std::string device = "auto";
};

struct test_params {
    llama_model_ptr model;
    ggml_backend_dev_t device = nullptr;
};

static ggml_backend_dev_t sampling_test_gpu() {
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    return device ? device : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

static llama_model_ptr load_model(const test_args & args) {
    auto mparams = llama_model_default_params();

    ggml_backend_dev_t devs[2] = { nullptr, nullptr };

    if (args.device != "auto") {
        if (args.device == "gpu") {
            devs[0] = sampling_test_gpu();

            if (devs[0] == nullptr) {
                fprintf(stderr, "Error: GPU requested but not available\n");
                return nullptr;
            }

            mparams.n_gpu_layers = 999;
        } else if (args.device == "cpu") {
            devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

            mparams.n_gpu_layers = 0;
        } else {
            fprintf(stderr, "Error: invalid device '%s'\n", args.device.c_str());
            return nullptr;
        }

        mparams.devices = devs;

        fprintf(stderr, "Using device: %s\n", ggml_backend_dev_name(devs[0]));
    }

    llama_model_ptr res;

    res.reset(llama_model_load_from_file(args.model.c_str(), mparams));

    if (!res) {
        fprintf(stderr, "Warning: failed to load model '%s', skipping test\n", args.model.c_str());
        return nullptr;
    }

    return res;
}

struct test_context {
    llama_context_ptr ctx;

    int n_vocab = 0;

    const llama_vocab * vocab = nullptr;

    std::unordered_map<llama_seq_id, int32_t> seq_positions;
    std::unordered_map<llama_seq_id, int32_t> last_batch_info;

    test_context(
            const test_params & params,
            std::vector<llama_sampler_seq_config> & configs,
            int32_t n_seq_max = -1,
            uint32_t n_outputs_max = 0,
            uint32_t n_ubatch = 0,
            uint32_t n_outputs_max_per_seq = 1) {
        auto * model = params.model.get();

        GGML_ASSERT(model);
        GGML_ASSERT(!ctx);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        cparams.n_batch = 512;
        if (n_ubatch > 0) {
            cparams.n_ubatch = n_ubatch;
        }
        cparams.n_outputs_max = n_outputs_max;
        cparams.n_outputs_max_per_seq = n_outputs_max_per_seq;
        cparams.samplers = configs.data();
        cparams.n_samplers = configs.size();
        cparams.kv_unified = true;

        // If n_seq_max is not specified, calculate it from configs
        if (n_seq_max < 0) {
            int32_t max_seq_id = 0;
            for (const auto & config : configs) {
                max_seq_id = std::max(config.seq_id, max_seq_id);
            }
            cparams.n_seq_max = max_seq_id + 1;
        } else {
            cparams.n_seq_max = n_seq_max;
        }

        ctx.reset(llama_init_from_model(model, cparams));
        if (!ctx) {
            throw std::runtime_error("failed to create context");
        }

        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
    }

    bool decode(const std::map<llama_seq_id, std::string> & prompts) {
        GGML_ASSERT(ctx);

        last_batch_info.clear();
        common_batch batch(ctx.get());

        for (const auto & [seq_id, prompt] : prompts) {
            std::vector<llama_token> tokens;
            tokens.push_back(llama_vocab_bos(vocab));

            std::vector<llama_token> prompt_tokens(32);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                           prompt_tokens.data(), prompt_tokens.size(),
                                           false, false);
            if (n_tokens < 0) {
                fprintf(stderr, "Warning: tokenization failed for seq_id %d\n", seq_id);
                return false;
            }

            for (int i = 0; i < n_tokens; i++) {
                tokens.push_back(prompt_tokens[i]);
            }

            if (seq_positions.find(seq_id) == seq_positions.end()) {
                seq_positions[seq_id] = 0;
            }

            int32_t start_pos = seq_positions[seq_id];
            for (size_t i = 0; i < tokens.size(); i++) {
                batch.add(tokens[i], start_pos + i, seq_id, i == tokens.size() - 1);
            }

            seq_positions[seq_id] = start_pos + tokens.size();
        }


        printf("Batch contents:\n");
        printf("n_tokens: %d\n", batch.size());
        for (int i = 0; i < batch.size(); i++) {
            const auto & t = batch.tokens[i];
            printf("token[%d]: tok=%-5d, pos=%d, seq_id=%d, logits=%d\n", i, t.id, t.pos[0], t.seq_id, t.output);
        }

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "Warning: llama_decode failed\n");
            return false;
        }

        update_batch_info(batch);
        return true;
    }

    int32_t idx_for_seq(llama_seq_id seq_id) {
        auto it = last_batch_info.find(seq_id);
        if (it == last_batch_info.end()) {
            fprintf(stderr, "Error: no batch index found for seq_id %d\n", seq_id);
            return -1;
        }
        return it->second;
    }

    // build mapping from seq id to batch token idx
    void update_batch_info(const common_batch & batch) {
        last_batch_info.clear();
        for (int i = 0; i < batch.size(); i++) {
            if (batch.tokens[i].output) {
                llama_seq_id cur_seq = batch.tokens[i].seq_id;
                last_batch_info[cur_seq] = i;
            }
        }
    }

    bool decode_token(llama_token token, llama_seq_id seq_id = 0) {
        GGML_ASSERT(ctx);

        common_batch batch(ctx.get());
        int32_t pos = seq_positions[seq_id];
        batch.add(token, pos, seq_id, true);

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for token %d in seq %d\n", token, seq_id);
            return false;
        }

        update_batch_info(batch);

        seq_positions[seq_id]++;

        return true;
    }

    bool decode_tokens(const std::map<llama_seq_id, llama_token> & seq_tokens) {
        GGML_ASSERT(ctx);

        common_batch batch(ctx.get());

        for (const auto & [seq_id, token] : seq_tokens) {
            int32_t pos = seq_positions[seq_id];
            batch.add(token, pos, seq_id, true);
        }

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for batch tokens\n");
            return false;
        }

        for (const auto & [seq_id, _] : seq_tokens) {
            seq_positions[seq_id]++;
        }

        update_batch_info(batch);

        return true;
    }

    std::string token_to_piece(llama_token token, bool special) const {
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        } else {
            piece.resize(n_chars);
        }

        return piece;
    }
};

struct test_single_output_backend_sampler {
    bool backend_initialized = false;
    uint32_t backend_outputs_max_per_seq = 0;
    int backend_apply_count = 0;
    int apply_count = 0;
};

static const char * test_single_output_backend_sampler_name(const llama_sampler * /*smpl*/) {
    return "single-output-backend";
}

static void test_single_output_backend_sampler_apply(
        llama_sampler * smpl, llama_token_data_array * /*cur_p*/) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->apply_count++;
}

static void test_single_output_backend_sampler_free(llama_sampler * smpl) {
    delete (test_single_output_backend_sampler *) smpl->ctx;
}

static bool test_single_output_backend_sampler_backend_init(
        llama_sampler * smpl, ggml_backend_buffer_type_t /*buft*/, uint32_t n_outputs_max_per_seq) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->backend_outputs_max_per_seq = n_outputs_max_per_seq;
    if (n_outputs_max_per_seq > 1) {
        return false;
    }
    ctx->backend_initialized = true;
    return true;
}

static void test_single_output_backend_sampler_backend_apply(
        llama_sampler * smpl, ggml_context * /*ctx*/, ggml_cgraph * /*gf*/, llama_sampler_data * /*data*/) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->backend_apply_count++;
}

static llama_sampler_i test_single_output_backend_sampler_i = {
    /* .name              = */ test_single_output_backend_sampler_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ test_single_output_backend_sampler_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ nullptr,
    /* .free              = */ test_single_output_backend_sampler_free,
    /* .backend_init      = */ test_single_output_backend_sampler_backend_init,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ test_single_output_backend_sampler_backend_apply,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static llama_sampler * test_single_output_backend_sampler_init(
        test_single_output_backend_sampler ** sampler_ctx) {
    auto * ctx = new test_single_output_backend_sampler;
    *sampler_ctx = ctx;
    return llama_sampler_init(&test_single_output_backend_sampler_i, ctx);
}

static void test_greedy(const test_params & params) {
    const int seq_id = 0;

    struct llama_sampler_chain_params backend_sampler_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_sampler_params));

    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_greedy());
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    token = llama_get_sampled_token_ith(test_ctx.ctx.get(), -1);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), loop_idx);
        printf("Generation step %d: token id:%d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        if (!test_ctx.decode_token(token, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }
}


static void test_greedy_filtered_common(const test_params & params) {
    auto check = [&](common_params_sampling sp, bool greedy) {
        common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
        auto * chain = common_sampler_get(sampler.get());
        auto * last = llama_sampler_chain_get(chain, llama_sampler_chain_n(chain) - 1);
        GGML_ASSERT(strcmp(llama_sampler_name(last), greedy ? "greedy" : "dist") == 0);

        llama_token_data tokens[] = {{0, -2.0f, 0.0f}, {1, -1.0f, 0.0f}, {2, 0.0f, 0.0f}, {3, 1.0f, 0.0f}};
        llama_token_data_array candidates = {tokens, 4, -1, false};
        llama_sampler_apply(chain, &candidates);
        GGML_ASSERT(candidates.selected >= 0);
        if (greedy || sp.n_probs > 0) {
            GGML_ASSERT(candidates.data[candidates.selected].id == 3);
        }
        if (sp.n_probs > 0) {
            GGML_ASSERT(candidates.data[candidates.selected].p == 1.0f);
        }
        if (sp.dynatemp_range > 0.0f) {
            int positive = 0;
            for (size_t i = 0; i < candidates.size; ++i) {
                positive += candidates.data[i].p > 0.0f;
            }
            GGML_ASSERT(positive > 1);
        }
    };

    common_params_sampling sp;
    sp.temp = 0.0f;
    sp.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
    for (bool backend : {false, true}) {
        sp.backend_sampling = backend;
        check(sp, true);
        auto probabilities = sp;
        probabilities.n_probs = 4;
        check(probabilities, false);
        auto dynamic = sp;
        dynamic.dynatemp_range = 1.0f;
        check(dynamic, false);
    }
    auto grammar = sp;
    grammar.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= [a-z]+"};
    check(grammar, true);
    auto budget = sp;
    budget.reasoning_budget_start = {1};
    budget.reasoning_budget_end = {{2}};
    budget.reasoning_budget_tokens = 1;
    check(budget, true);
    sp.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K};
    check(sp, false);
    sp.samplers.clear();
    check(sp, false);

    sp.temp = 0.8f;
    sp.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K};
    for (bool backend : {false, true}) {
        sp.backend_sampling = backend;
        sp.top_k = 1;
        check(sp, true);
        auto probabilities = sp;
        probabilities.n_probs = 4;
        check(probabilities, false);
        sp.top_k = 8;
        check(sp, false);
    }
}

static void test_greedy_filtered(const test_params & params) {
    for (int k : {1, 8}) {
        llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        std::vector<llama_sampler_seq_config> configs = {{0, chain.get()}};
        test_context test_ctx(params, configs);
        GGML_ASSERT(test_ctx.decode({{0, "Write a Python function"}}));
        for (int step = 0; step < 4; ++step) {
            const int idx = test_ctx.idx_for_seq(0);
            const auto * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), idx);
            const auto * ids = llama_get_sampled_candidates_ith(test_ctx.ctx.get(), idx);
            GGML_ASSERT(llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), idx) == (uint32_t) k);
            GGML_ASSERT(llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), idx) == (uint32_t) k);
            const auto expected = ids[std::max_element(logits, logits + k) - logits];
            GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), idx) == expected);
            GGML_ASSERT(test_ctx.decode_token(expected));
        }
    }
}

static void test_top_k(const test_params & params) {
    const int seq_id = 0;
    const int32_t k = 8;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_top_k(k));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
    for (size_t i = 0; i < n_logits; ++i) {
        printf("top_k logit[%zu] = %.6f\n", i, logits[i]);
    }

    llama_token * candidates = llama_get_sampled_candidates_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_candidates = llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), batch_idx);
    for (size_t i = 0; i < n_candidates; ++i) {
        printf("top_k candidate[%zu] = %d : %s\n", i, candidates[i],
               test_ctx.token_to_piece(candidates[i], false).c_str());
    }

    // Sample using CPU sampler for verification that it is possible to do hybrid
    // sampling, first top_k on the backend and then dist on the CPU.
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    GGML_ASSERT(chain->iface->backend_apply != nullptr);

    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));
    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    printf("backend top-k hybrid sampling test PASSED\n");
}

static void test_temp(const test_params & params) {
    {
        const float temp_0 = 0.8f;
        struct llama_sampler_chain_params backend_chain_params_0 = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain_0(llama_sampler_chain_init(backend_chain_params_0));
        llama_sampler_chain_add(backend_sampler_chain_0.get(), llama_sampler_init_temp(temp_0));

        const float temp_1 = 0.1f;
        struct llama_sampler_chain_params backend_chain_params_1 = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain_1(llama_sampler_chain_init(backend_chain_params_1));
        llama_sampler_chain_add(backend_sampler_chain_1.get(), llama_sampler_init_temp(temp_1));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { 0, backend_sampler_chain_0.get() },
            { 1, backend_sampler_chain_1.get() }
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{0, "Some where over the"}, {1, "Once upon a"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        // Verify sequence 0
        {
            int32_t batch_idx = test_ctx.idx_for_seq(0);
            int n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
            GGML_ASSERT(n_logits == test_ctx.n_vocab);

            // Sample from sequence 0 using CPU sampler
            struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
            llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

            llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("Sequence 0 sampled token id:%d, string: '%s'\n", token, token_str.c_str());
            GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        }


        // Verify sequence 1
        {
            int32_t batch_idx = test_ctx.idx_for_seq(1);

            // Sample from sequence 1 using CPU sampler
            struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
            llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

            llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("Sequence 1 sampled token id:%d, string: '%s'\n", token, token_str.c_str());
            GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        }
    }

    // lambda for testing non-positive temperature values.
    auto test_argmax_temp = [&](float temp) {
        printf("\nTesting temperature = %.1f\n", temp);

        int seq_id = 0;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp(temp));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

        uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(n_logits == 1);
    };

    test_argmax_temp(0.0f);
    test_argmax_temp(-1.0f);

    printf("backend temp sampling test PASSED\n");
}

static void test_temp_ext(const test_params & params) {
    {
        int seq_id = 0;
        const float temp = 0.8f;
        const float delta = 0.5f;
        const float exponent = 1.5f;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp_ext(temp, delta, exponent));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once upon a"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        // Verify sequence 0
        {
            int32_t batch_idx = test_ctx.idx_for_seq(seq_id);
            int n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
            GGML_ASSERT(n_logits == test_ctx.n_vocab);
        }
    }

    // lambda for testing non-positive temp/delta/exponent values.
    auto test_argmax_temp = [&](float temp, float delta, float exponent) {
        printf("\nTesting temperature = %.1f, delta = %1.f, exponent = %1.f\n", temp, delta, exponent);

        int seq_id = 0;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp_ext(temp, delta, exponent));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

        uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

        if (temp <= 0.0f && delta >= 0.0f) {
            GGML_ASSERT(n_logits == 1);
        } else {
            GGML_ASSERT(n_logits == (uint32_t) test_ctx.n_vocab);
        }
    };

    test_argmax_temp(0.0f,  0.3f, 1.0f); // Greedy (temp=0)
    test_argmax_temp(-1.0f, 0.3f, 2.0f); // Greedy (temp<0)
    test_argmax_temp(0.8f,  0.0f, 2.0f); // Temperature scaling

    printf("backend temp_ext sampling test PASSED\n");
}

static void test_min_p(const test_params & params) {
    const int seq_id = 0;
    const float p = 0.1;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_min_p(p, 0));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

    // Print the logits that are above the min-p threshold
    std::vector<float> filtered_logits;
    for (size_t i = 0; i < n_logits; ++i) {
        if (logits[i] > -1e9f) {
            filtered_logits.push_back(logits[i]);
            //printf("min_p logit[%zu] = %.6f\n", i, logits[i]);
        }
    }
    GGML_ASSERT(filtered_logits.size() < (size_t) test_ctx.n_vocab);

    // Sample using CPU sampler for verification to inspect they are reasonable
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));

    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    const std::string token_str = test_ctx.token_to_piece(token, false);
    printf("min-p cpu sampled token id:%d, string: '%s'\n", token, token_str.c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    // Decode and sample 10 more tokens
    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), loop_idx);
        printf("min-p gen step %d: token id :%5.d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        if (!test_ctx.decode_token(token, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    printf("min-p sampling test PASSED\n");
}

static void test_top_p(const test_params & params) {
    const int seq_id = 0;
    const float p = 0.9;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_top_p(p, 0));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        return;
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

    // Print the logits that are above the min-p threshold
    std::vector<float> filtered_logits;
    for (size_t i = 0; i < n_logits; ++i) {
        if (logits[i] > -1e9f) {
            filtered_logits.push_back(logits[i]);
        }
    }
    GGML_ASSERT(filtered_logits.size() < (size_t) test_ctx.n_vocab);
    GGML_ASSERT(filtered_logits.size() > 0);

    // Sample using CPU sampler for verification to inspect they are reasonable
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));

    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    const std::string token_str = test_ctx.token_to_piece(token, false);
    printf("top-p cpu sampled token id:%d, string: '%s'\n", token, token_str.c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    // Decode and sample 10 more tokens
    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), loop_idx);
        printf("top-p gen step %d: token id :%5.d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        test_ctx.decode_token(token, 0);
    }

    printf("top-p sampling test PASSED\n");
}

static void test_multi_sequence(const test_params & params) {
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_greedy());

    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_1(llama_sampler_chain_init(chain_params_1));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_greedy());

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
        { 1, sampler_chain_1.get() }
    };

    test_context test_ctx(params, backend_sampler_configs);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"},
        {1, "Some"}
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 0 sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Verify sequence 1
    {
        int32_t batch_idx= test_ctx.idx_for_seq(1);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 1 sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Generate tokens for each sequence
    printf("\nMulti-sequence generation:\n");
    for (int step = 0; step < 4; step++) {
        std::map<llama_seq_id, llama_token> tokens;

        for (llama_seq_id seq_id : {0, 1}) {
            int32_t idx = test_ctx.idx_for_seq(seq_id);
            llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("  Seq %d, step %d: token id=%d, string='%s'\n", seq_id, step, token, token_str.c_str());
            tokens[seq_id] = token;
        }

        // Decode all tokens in a single batch
        if (!test_ctx.decode_tokens(tokens)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    printf("backend multi-sequence sampling test PASSED\n");
}

static void test_dist(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);
    llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    printf("dist sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    //GGML_ASSERT(llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx) == nullptr);

    token = llama_get_sampled_token_ith(test_ctx.ctx.get(), -1);
    printf("dist sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    printf("backend dist sampling test PASSED\n");
}

static void test_dist_and_cpu(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    // Sample using CPU sampler
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    llama_token cpu_token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    printf("dist & cpu sampled id:%d, string:'%s'\n", cpu_token, test_ctx.token_to_piece(cpu_token, false).c_str());
    GGML_ASSERT(backend_token == cpu_token);

    printf("backend dist & cpu sampling test PASSED\n");
}

static void test_logit_bias(const test_params & params) {
    const auto * model = params.model.get();
    const auto * vocab = llama_model_get_vocab(model);

    const int seq_id = 0;

    std::vector<llama_logit_bias> logit_bias;

    // Get the token for the piece "World".
    const std::string piece = "World";
    std::vector<llama_token> tokens(16);
    llama_tokenize(vocab, piece.c_str(), piece.size(), tokens.data(), tokens.size(), false, false);

    llama_token bias_token = tokens[0];
    // TODO: biasing too much here makes the Vulkan sampling fail - should be investigated further
    //       https://github.com/ggml-org/llama.cpp/actions/runs/20894267644/job/60030252675?pr=18753#step:3:23350
    //logit_bias.push_back({ bias_token, +100.0f });
    logit_bias.push_back({ bias_token, +10.0f });

    printf("biasing token piece '%s' -> token id %d\n", piece.c_str(), bias_token);

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                logit_bias.size(),
                logit_bias.data()));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(88));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { seq_id, backend_sampler_chain.get() },
    };

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), test_ctx.idx_for_seq(seq_id));
    printf("sampled token = %d, expected = %d\n", backend_token, bias_token);
    GGML_ASSERT(backend_token == bias_token);

    printf("backend logit bias sampling test PASSED\n");
}

static void accept_prompt(llama_sampler * smpl, const llama_vocab * vocab, const std::string & prompt) {
    const llama_token bos = llama_vocab_bos(vocab);
    if (bos != LLAMA_TOKEN_NULL) {
        llama_sampler_accept(smpl, bos);
    }

    std::vector<llama_token> tokens(64);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
            tokens.data(), (int32_t) tokens.size(), false, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                tokens.data(), (int32_t) tokens.size(), false, false);
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        llama_sampler_accept(smpl, tokens[i]);
    }
}

static std::vector<float> decode_raw_logits(const test_params & params, const std::string & prompt) {
    const int seq_id = 0;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(params.model.get()));
    std::vector<llama_sampler_seq_config> empty_configs;
    test_context ctx(params, empty_configs);

    GGML_ASSERT(ctx.decode({{ seq_id, prompt }}));

    float * logits = llama_get_logits_ith(ctx.ctx.get(), ctx.idx_for_seq(seq_id));
    GGML_ASSERT(logits != nullptr);
    return std::vector<float>(logits, logits + n_vocab);
}

static std::vector<llama_token_data> apply_cpu_sampler(
        const std::vector<float> & raw_logits,
        llama_sampler * sampler) {
    std::vector<llama_token_data> data;
    data.reserve(raw_logits.size());
    for (llama_token token = 0; token < (llama_token) raw_logits.size(); ++token) {
        data.push_back({ token, raw_logits[token], 0.0f });
    }

    llama_token_data_array cur_p = { data.data(), data.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    data.resize(cur_p.size);
    return data;
}

using sampler_setup_fn = std::function<void(llama_sampler *)>;
using sampler_init_fn = std::function<llama_sampler *()>;

enum class penalties_position {
    before_filter,
    after_filter,
};

static void add_filter_and_penalties(
        llama_sampler * chain,
        const sampler_init_fn & init_filter,
        int32_t n_vocab,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        penalties_position position) {
    const auto add_penalties = [&]() {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                    n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    };

    if (position == penalties_position::before_filter) {
        add_penalties();
        llama_sampler_chain_add(chain, init_filter());
    } else {
        llama_sampler_chain_add(chain, init_filter());
        add_penalties();
    }
}

static llama_sampler_ptr make_sampler_chain(
        const sampler_setup_fn & add_samplers,
        const sampler_setup_fn & accept_history) {
    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    add_samplers(chain.get());
    accept_history(chain.get());
    return chain;
}

struct backend_sampler_output {
    std::vector<float> logits;
    std::vector<llama_token> candidates;
};

static backend_sampler_output run_backend_sampler(
        const test_params & params,
        const std::string & prompt,
        llama_sampler * sampler) {
    const int seq_id = 0;
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, sampler }};
    test_context ctx(params, configs);

    GGML_ASSERT(ctx.decode({{ seq_id, prompt }}));
    llama_synchronize(ctx.ctx.get());

    const int32_t idx = ctx.idx_for_seq(seq_id);
    const uint32_t n_logits = llama_get_sampled_logits_count_ith(ctx.ctx.get(), idx);
    const uint32_t n_candidates = llama_get_sampled_candidates_count_ith(ctx.ctx.get(), idx);
    float * logits = llama_get_sampled_logits_ith(ctx.ctx.get(), idx);
    llama_token * candidates = llama_get_sampled_candidates_ith(ctx.ctx.get(), idx);
    GGML_ASSERT(logits != nullptr);

    backend_sampler_output result;
    result.logits.assign(logits, logits + n_logits);
    result.candidates.resize(n_logits);

    if (n_candidates == 0) {
        for (uint32_t i = 0; i < n_logits; ++i) {
            result.candidates[i] = (llama_token) i;
        }
    } else {
        GGML_ASSERT(candidates != nullptr);
        GGML_ASSERT(n_candidates == n_logits);
        std::memcpy(result.candidates.data(), candidates, n_candidates * sizeof(llama_token));
    }

    return result;
}

struct sampler_comparison_output {
    std::vector<llama_token_data> expected;
    backend_sampler_output actual;
};

static sampler_comparison_output run_sampler_comparison(
        const test_params & params,
        const std::string & prompt,
        const std::vector<float> & raw_logits,
        const sampler_setup_fn & add_samplers,
        const sampler_setup_fn & accept_history) {
    llama_sampler_ptr cpu_chain = make_sampler_chain(add_samplers, accept_history);
    llama_sampler_ptr backend_chain = make_sampler_chain(add_samplers, accept_history);
    return {
        apply_cpu_sampler(raw_logits, cpu_chain.get()),
        run_backend_sampler(params, prompt, backend_chain.get()),
    };
}

static std::unordered_map<llama_token, float> map_logits(const std::vector<llama_token_data> & data) {
    std::unordered_map<llama_token, float> result;
    result.reserve(data.size());
    for (const auto & item : data) {
        result[item.id] = item.logit;
    }
    return result;
}

struct sampler_comparison_stats {
    int n_mismatch = 0;
    int n_masked = 0;
    float max_diff = 0.0f;
};

static sampler_comparison_stats compare_sampler_outputs(
        const char * name,
        const std::unordered_map<llama_token, float> & expected,
        const backend_sampler_output & actual,
        bool allow_extra_candidates = false) {
    GGML_ASSERT(actual.logits.size() == actual.candidates.size());

    sampler_comparison_stats result;
    std::unordered_set<llama_token> seen;
    seen.reserve(actual.candidates.size());

    for (size_t i = 0; i < actual.logits.size(); ++i) {
        const llama_token token = actual.candidates[i];
        const float logit = actual.logits[i];
        if (!seen.insert(token).second || std::isnan(logit)) {
            if (result.n_mismatch < 5) {
                printf("%s token %d has invalid backend output\n", name, token);
            }
            ++result.n_mismatch;
            continue;
        }

        const auto it = expected.find(token);
        if (it == expected.end()) {
            if (std::isinf(logit) && logit < 0.0f) {
                ++result.n_masked;
            } else if (!allow_extra_candidates) {
                if (result.n_mismatch < 5) {
                    printf("%s token %d was not masked\n", name, token);
                }
                ++result.n_mismatch;
            }
            continue;
        }

        const float diff = fabsf(it->second - logit);
        result.max_diff = std::max(result.max_diff, diff);
        if (!std::isfinite(logit) || diff > 1e-3f) {
            if (result.n_mismatch < 5) {
                printf("%s mismatch token %d: cpu=%.6f backend=%.6f diff=%.6f\n",
                        name, token, it->second, logit, diff);
            }
            ++result.n_mismatch;
        }
    }

    for (const auto & item : expected) {
        if (seen.find(item.first) == seen.end()) {
            if (result.n_mismatch < 5) {
                printf("%s missing backend token %d\n", name, item.first);
            }
            ++result.n_mismatch;
        }
    }

    printf("%s logits: max_diff=%.6f n_masked=%d n_mismatch=%d\n",
            name, result.max_diff, result.n_masked, result.n_mismatch);
    return result;
}

static float find_backend_logit(const backend_sampler_output & output, llama_token token) {
    for (size_t i = 0; i < output.candidates.size(); ++i) {
        if (output.candidates[i] == token) {
            return output.logits[i];
        }
    }
    GGML_ABORT("backend token not found");
}

static sampler_comparison_output run_penalties_comparison(
        const test_params & params,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        const std::function<void(llama_sampler *)> & extra_accept = {}) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const auto add_samplers = [&](llama_sampler * chain) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                    llama_vocab_n_tokens(vocab), penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    };
    const auto accept_history = [&](llama_sampler * chain) {
        accept_prompt(chain, vocab, prompt);
        if (extra_accept) {
            extra_accept(chain);
        }
    };

    return run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);
}

static void compare_penalties_logits(
        const test_params & params,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        const std::function<void(llama_sampler *)> & extra_accept = {}) {
    const sampler_comparison_output output = run_penalties_comparison(
            params, penalty_last_n, penalty_repeat, penalty_freq, penalty_present, prompt, extra_accept);

    GGML_ASSERT(output.expected.size() == output.actual.logits.size());

    const sampler_comparison_stats stats = compare_sampler_outputs(
            "penalties", map_logits(output.expected), output.actual);
    GGML_ASSERT(stats.n_masked == 0);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static void check_penalty_parameter_values(const test_params & params) {
    struct penalty_test_case {
        const char * name;
        float repeat;
        float frequency;
        float presence;
    };

    const penalty_test_case cases[] = {
        { "frequency -1",   1.0f, -1.0f,     0.0f },
        { "frequency 0",    1.0f,  0.0f,     0.0f },
        { "frequency 1",    1.0f,  1.0f,     0.0f },
        { "presence -1",    1.0f,  0.0f,    -1.0f },
        { "presence 0",     1.0f,  0.0f,     0.0f },
        { "presence 1",     1.0f,  0.0f,     1.0f },
        { "repeat 1",       1.0f,  0.0f,     0.0f },
    };

    int n_failed = 0;
    for (const auto & test : cases) {
        const sampler_comparison_output output = run_penalties_comparison(
                params, 64, test.repeat, test.frequency, test.presence, "Hello Hello world");
        GGML_ASSERT(output.expected.size() == output.actual.logits.size());
        const sampler_comparison_stats stats = compare_sampler_outputs(
                test.name, map_logits(output.expected), output.actual);
        n_failed += stats.n_mismatch != 0;
    }

    GGML_ASSERT(n_failed == 0);
}

static void compare_top_k_penalties_logits(
        const test_params & params,
        int32_t k,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        penalties_position position) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const int n_vocab = (int) raw_logits.size();

    GGML_ASSERT(n_vocab > k);

    const sampler_init_fn init_top_k = [k]() {
        return llama_sampler_init_top_k(k);
    };
    llama_sampler_ptr top_k(init_top_k());
    const std::vector<llama_token_data> top_k_data = apply_cpu_sampler(raw_logits, top_k.get());
    GGML_ASSERT(top_k_data.size() == (size_t) k);
    const llama_token retained_history_token = top_k_data[0].id;

    llama_token excluded_history_token = LLAMA_TOKEN_NULL;
    for (llama_token token = 0; token < n_vocab; ++token) {
        const auto it = std::find_if(top_k_data.begin(), top_k_data.end(), [token](const llama_token_data & data) {
            return data.id == token;
        });
        if (it == top_k_data.end()) {
            excluded_history_token = token;
            break;
        }
    }
    GGML_ASSERT(excluded_history_token != LLAMA_TOKEN_NULL);

    const auto add_samplers = [&](llama_sampler * chain) {
        add_filter_and_penalties(chain, init_top_k, n_vocab,
                penalty_last_n, penalty_repeat, penalty_freq, penalty_present, position);
    };

    auto accept_history = [&](llama_sampler * smpl) {
        accept_prompt(smpl, vocab, prompt);
        llama_sampler_accept(smpl, excluded_history_token);
        llama_sampler_accept(smpl, excluded_history_token);
        llama_sampler_accept(smpl, retained_history_token);
        llama_sampler_accept(smpl, retained_history_token);
    };

    const sampler_comparison_output output = run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);

    GGML_ASSERT(output.expected.size() == (size_t) k);
    GGML_ASSERT(output.actual.logits.size() == (size_t) k);

    const std::unordered_map<llama_token, float> expected_logits = map_logits(output.expected);

    if (position == penalties_position::after_filter) {
        GGML_ASSERT(expected_logits.find(retained_history_token) != expected_logits.end());
        GGML_ASSERT(fabsf(expected_logits.at(retained_history_token) - raw_logits[retained_history_token]) > 1e-6f);
        GGML_ASSERT(expected_logits.find(excluded_history_token) == expected_logits.end());
        GGML_ASSERT(std::find(output.actual.candidates.begin(), output.actual.candidates.end(),
                    excluded_history_token) == output.actual.candidates.end());
    } else {
        const std::unordered_map<llama_token, float> unpenalized_logits = map_logits(top_k_data);
        bool changed = false;
        for (const auto & item : expected_logits) {
            const auto it = unpenalized_logits.find(item.first);
            if (it == unpenalized_logits.end() || fabsf(it->second - item.second) > 1e-6f) {
                changed = true;
                break;
            }
        }
        GGML_ASSERT(changed);
    }

    const char * name = position == penalties_position::before_filter
        ? "penalties top-k"
        : "top-k penalties";
    const sampler_comparison_stats stats = compare_sampler_outputs(
            name, expected_logits, output.actual);
    GGML_ASSERT(stats.n_masked == 0);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static void compare_masking_penalties_logits(
        const test_params & params,
        const char * filter_name,
        const sampler_init_fn & init_filter,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        penalties_position position,
        bool allow_extra_candidates,
        bool add_history = true) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const int n_vocab = (int) raw_logits.size();
    llama_sampler_ptr filter(init_filter());
    const std::vector<llama_token_data> filtered_data = apply_cpu_sampler(raw_logits, filter.get());
    GGML_ASSERT(!filtered_data.empty());
    GGML_ASSERT(filtered_data.size() < (size_t) n_vocab);

    const llama_token penalized_token = filtered_data[0].id;
    std::unordered_set<llama_token> retained_tokens;
    retained_tokens.reserve(filtered_data.size());
    for (const auto & data : filtered_data) {
        retained_tokens.insert(data.id);
    }

    llama_token masked_token = LLAMA_TOKEN_NULL;
    for (llama_token token = 0; token < n_vocab; ++token) {
        if (retained_tokens.find(token) == retained_tokens.end()) {
            masked_token = token;
            break;
        }
    }
    GGML_ASSERT(masked_token != LLAMA_TOKEN_NULL);

    const auto add_samplers = [&](llama_sampler * chain) {
        add_filter_and_penalties(chain, init_filter, n_vocab,
                penalty_last_n, penalty_repeat, penalty_freq, penalty_present, position);
    };
    auto accept_history = [&](llama_sampler * smpl) {
        if (!add_history) {
            return;
        }
        accept_prompt(smpl, vocab, prompt);
        llama_sampler_accept(smpl, penalized_token);
        llama_sampler_accept(smpl, penalized_token);
        llama_sampler_accept(smpl, masked_token);
        llama_sampler_accept(smpl, masked_token);
    };

    const sampler_comparison_output output = run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);

    GGML_ASSERT(output.actual.logits.size() == (size_t) n_vocab);

    const std::unordered_map<llama_token, float> expected_logits = map_logits(output.expected);

    GGML_ASSERT(expected_logits.find(masked_token) == expected_logits.end());
    if (add_history) {
        if (position == penalties_position::after_filter) {
            GGML_ASSERT(expected_logits.find(penalized_token) != expected_logits.end());
            GGML_ASSERT(fabsf(expected_logits.at(penalized_token) - raw_logits[penalized_token]) > 1e-6f);
        } else {
            llama_sampler_ptr penalties(llama_sampler_init_penalties(
                        n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
            accept_history(penalties.get());
            const std::unordered_map<llama_token, float> penalized_logits =
                map_logits(apply_cpu_sampler(raw_logits, penalties.get()));
            GGML_ASSERT(fabsf(penalized_logits.at(penalized_token) - raw_logits[penalized_token]) > 1e-6f);
        }
    }

    const std::string name = position == penalties_position::before_filter
        ? "penalties " + std::string(filter_name)
        : std::string(filter_name) + " penalties";
    const sampler_comparison_stats stats = compare_sampler_outputs(
            name.c_str(), expected_logits, output.actual, allow_extra_candidates);
    const float masked_logit = find_backend_logit(output.actual, masked_token);
    GGML_ASSERT(stats.n_masked > 0);
    GGML_ASSERT(std::isinf(masked_logit) && masked_logit < 0.0f);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static std::vector<llama_token> grammar_test_tokens(const llama_vocab * vocab) {
    std::vector<llama_token> tokens;
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab) && tokens.size() < 3; ++id) {
        const auto piece = common_token_to_piece(vocab, id);
        if (!llama_vocab_is_eog(vocab, id) && !piece.empty() && piece[0] != 0) {
            tokens.push_back(id);
        }
    }
    GGML_ASSERT(tokens.size() == 3);
    return tokens;
}

static std::string grammar_token(llama_token token) {
    return "<[" + std::to_string(token) + "]>";
}

static bool grammar_test_supported(const test_params & params) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    llama_sampler_ptr sampler(llama_sampler_init_grammar(vocab, "root ::= .", "root"));
    if (!sampler->iface->backend_init(sampler.get(), ggml_backend_dev_buffer_type(params.device), 1)) {
        printf("Skipping backend grammar test: TOKEN_MASK unsupported on %s\n", ggml_backend_dev_name(params.device));
        return false;
    }
    return true;
}

struct grammar_test_graph {
    llama_sampler * sampler;
    ggml_backend_t backend;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_backend_ptr cpu;
    ggml_backend_sched_ptr sched;
    ggml_cgraph * graph;
    ggml_tensor * logits;
    llama_sampler_data data;
    std::vector<llama_token_data> input;

    grammar_test_graph(llama_sampler * sampler, ggml_backend_t backend, int32_t n_vocab,
            const std::vector<llama_token> & candidates = {}, bool scheduled = false, bool parallel = false) : sampler(sampler), backend(backend) {
        sampler->iface->backend_reset(sampler);
        ggml_init_params params = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        graph = ggml_new_graph(ctx.get());
        const int64_t n = candidates.empty() ? n_vocab : candidates.size();
        logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
        ggml_set_input(logits);
        ggml_set_output(logits);
        ggml_tensor * ids = candidates.empty() ? nullptr : ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n);
        data = { logits, ggml_soft_max(ctx.get(), logits), nullptr, ids };
        sampler->iface->backend_apply(sampler, ctx.get(), graph, &data);
        GGML_ASSERT(data.candidates == ids);
        GGML_ASSERT(data.probs == nullptr);
        GGML_ASSERT(data.sampled == nullptr);
        ggml_build_forward_expand(graph, data.logits);
        if (scheduled) {
            if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
                GGML_ASSERT(cpu);
            }
            ggml_backend_t backends[] = {backend, cpu.get()};
            sched.reset(ggml_backend_sched_new(backends, nullptr, cpu ? 2 : 1, GGML_DEFAULT_GRAPH_SIZE, parallel, true));
            GGML_ASSERT(sched);
            ggml_backend_sched_set_tensor_backend(sched.get(), logits, backend);
            GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        } else {
            buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
            GGML_ASSERT(buffer);
        }

        std::vector<float> values(n);
        for (int64_t i = 0; i < n; ++i) {
            values[i] = float(i % 19) - 9.0f;
            input.push_back({ ids ? candidates[i] : (llama_token) i, values[i], 0.0f });
        }
        ggml_backend_tensor_set(logits, values.data(), 0, values.size()*sizeof(float));
        if (ids) {
            ggml_backend_tensor_set(ids, candidates.data(), 0, candidates.size()*sizeof(llama_token));
        }
    }

    ~grammar_test_graph() {
        sampler->iface->backend_reset(sampler);
    }

    void check(llama_sampler * reference) {
        auto expected = input;
        llama_token_data_array cur = { expected.data(), expected.size(), -1, false };
        llama_sampler_apply(reference, &cur);
        sampler->iface->backend_set_input(sampler, sched.get());
        compute();
        std::vector<float> actual(input.size()), original(input.size());
        ggml_backend_tensor_get(data.logits, actual.data(), 0, actual.size()*sizeof(float));
        ggml_backend_tensor_get(logits, original.data(), 0, original.size()*sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            GGML_ASSERT(actual[i] == expected[i].logit);
            GGML_ASSERT(original[i] == input[i].logit);
        }
    }

    void compute() {
        const auto status = sched ? ggml_backend_sched_graph_compute(sched.get(), graph) : ggml_backend_graph_compute(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
};

struct grammar_mask_upload_counter {
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    ggml_tensor * payload;
    decltype(ggml_backend_buffer_i::set_tensor) set_tensor;
    decltype(ggml_backend_buffer_i::set_tensor_n) set_tensor_n;
    decltype(ggml_backend_i::set_tensor_async) set_tensor_async;
    size_t uploads = 0;
    size_t batches = 0;
    size_t descriptors = 0;
    size_t payload_bytes = 0;
    static grammar_mask_upload_counter * active;

    grammar_mask_upload_counter(ggml_backend_t backend, ggml_tensor * payload)
        : backend(backend), buffer(payload->buffer), payload(payload), set_tensor(buffer->iface.set_tensor),
          set_tensor_n(buffer->iface.set_tensor_n), set_tensor_async(backend->iface.set_tensor_async) {
        GGML_ASSERT(!active);
        active = this;
        buffer->iface.set_tensor = [](ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                     const void * data, size_t offset, size_t size) {
            active->count(tensor, size);
            active->set_tensor(buffer, tensor, data, offset, size);
        };
        if (set_tensor_n) {
            buffer->iface.set_tensor_n = [](ggml_backend_buffer_t buffer, ggml_tensor * const * tensors,
                                           const void * const * data, const size_t * offsets, const size_t * sizes, size_t n) {
                ++active->batches;
                for (size_t i = 0; i < n; ++i) {
                    active->count(tensors[i], sizes[i]);
                }
                active->set_tensor_n(buffer, tensors, data, offsets, sizes, n);
            };
        }
        if (set_tensor_async) {
            backend->iface.set_tensor_async = [](ggml_backend_t backend, ggml_tensor * tensor,
                                                const void * data, size_t offset, size_t size) {
                active->count(tensor, size);
                active->set_tensor_async(backend, tensor, data, offset, size);
            };
        }
    }

    void count(ggml_tensor * tensor, size_t size) {
        if (tensor == payload) {
            ++uploads;
            payload_bytes += size;
        } else {
            ++descriptors;
        }
    }

    ~grammar_mask_upload_counter() {
        buffer->iface.set_tensor = set_tensor;
        buffer->iface.set_tensor_n = set_tensor_n;
        backend->iface.set_tensor_async = set_tensor_async;
        active = nullptr;
    }
};

grammar_mask_upload_counter * grammar_mask_upload_counter::active = nullptr;

struct grammar_async_upload_counter {
    ggml_backend_t backend;
    decltype(ggml_backend_i::set_tensor_async) set_tensor_async;
    size_t uploads = 0;
    static grammar_async_upload_counter * active;

    grammar_async_upload_counter(ggml_backend_t backend) : backend(backend), set_tensor_async(backend->iface.set_tensor_async) {
        GGML_ASSERT(!active);
        active = this;
        if (set_tensor_async) {
            backend->iface.set_tensor_async = [](ggml_backend_t backend, ggml_tensor * tensor,
                                                 const void * data, size_t offset, size_t size) {
                ++active->uploads;
                active->set_tensor_async(backend, tensor, data, offset, size);
            };
        }
    }

    ~grammar_async_upload_counter() {
        backend->iface.set_tensor_async = set_tensor_async;
        active = nullptr;
    }
};

grammar_async_upload_counter * grammar_async_upload_counter::active = nullptr;

static void test_grammar_async_upload(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const auto tokens = grammar_test_tokens(vocab);
    std::string rule = "root ::= ";
    for (int i = 0; i < 20; ++i) {
        rule += grammar_token(tokens[i % tokens.size()]) + " ";
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(params.device, nullptr));
    GGML_ASSERT(backend);
    llama_sampler_ptr sampler(llama_sampler_init_grammar(vocab, rule.c_str(), "root"));
    llama_sampler_ptr reference(llama_sampler_clone(sampler.get()));
    GGML_ASSERT(sampler->iface->backend_init(sampler.get(), ggml_backend_get_default_buffer_type(backend.get()), 1));
    grammar_async_upload_counter counter(backend.get());
    {
        grammar_test_graph graph(sampler.get(), backend.get(), n_vocab, {}, true);
        for (int i = 0; i < 20; ++i) {
            sampler->iface->backend_set_input(sampler.get(), graph.sched.get());
            GGML_ASSERT(ggml_backend_sched_graph_compute_async(graph.sched.get(), graph.graph) == GGML_STATUS_SUCCESS);
            llama_sampler_accept(sampler.get(), tokens[i % tokens.size()]);
            llama_sampler_accept(reference.get(), tokens[i % tokens.size()]);
        }
        graph.check(reference.get());
        llama_sampler_reset(sampler.get());
        sampler->iface->backend_set_input(sampler.get(), graph.sched.get());
        sampler->iface->backend_reset(sampler.get());
    }
    if (counter.set_tensor_async) {
        GGML_ASSERT(counter.uploads > 20);
    }
    llama_sampler_reset(sampler.get());
    llama_sampler_reset(reference.get());
    {
        grammar_test_graph graph(sampler.get(), backend.get(), n_vocab, {}, true);
        graph.check(reference.get());
    }

    const auto a = common_tokenize(vocab, "a", false, true);
    const auto b = common_tokenize(vocab, "b", false, true);
    GGML_ASSERT(a.size() == 1 && b.size() == 1);
    const auto c = common_tokenize(vocab, "c", false, true);
    GGML_ASSERT(c.size() == 1);
    sampler.reset(llama_sampler_init_grammar(vocab, "root ::= [ac]* \"b\"", "root"));
    reference.reset(llama_sampler_clone(sampler.get()));
    GGML_ASSERT(sampler->iface->backend_init(sampler.get(), ggml_backend_get_default_buffer_type(backend.get()), 1));
    {
        grammar_test_graph graph(sampler.get(), backend.get(), n_vocab, {}, true);
        for (int i = 0; i < 16; ++i) {
            graph.check(reference.get());
            const llama_token token = i % 2 ? a[0] : c[0];
            llama_sampler_accept(sampler.get(), token);
            llama_sampler_accept(reference.get(), token);
        }
        llama_sampler_accept(sampler.get(), b[0]);
        llama_sampler_accept(reference.get(), b[0]);
        graph.check(reference.get());
    }
    printf("backend grammar async upload, pending reset and unchanged state test PASSED\n");
}

static void test_grammar_resident_cache(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int64_t n_words = (int64_t(n_vocab) + 31)/32;
    std::vector<llama_token> tokens;
    for (llama_token id = 0; id < n_vocab && tokens.size() < 10; ++id) {
        const auto piece = common_token_to_piece(vocab, id);
        if (!llama_vocab_is_eog(vocab, id) && !piece.empty() && piece[0] != 0) {
            tokens.push_back(id);
        }
    }
    GGML_ASSERT(tokens.size() == 10);
    std::string rule = "root ::= ";
    for (auto token : tokens) {
        rule += grammar_token(token) + " ";
    }
    llama_sampler_ptr sampler(llama_sampler_init_grammar(vocab, rule.c_str(), "root"));
    llama_sampler_ptr reference(llama_sampler_clone(sampler.get()));
    std::vector<llama_sampler_ptr> states;
    for (auto token : tokens) {
        states.emplace_back(llama_sampler_clone(reference.get()));
        llama_sampler_accept(reference.get(), token);
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(params.device, nullptr));
    GGML_ASSERT(backend);
    GGML_ASSERT(sampler->iface->backend_init(sampler.get(), ggml_backend_get_default_buffer_type(backend.get()), 1));
    for (int mode = 0; mode < 3; ++mode) {
        llama_sampler_copy(states[0].get(), sampler.get());
        grammar_test_graph graph(sampler.get(), backend.get(), n_vocab, {}, mode != 0, mode == 2);
        auto * payload = graph.data.logits->src[2];
        auto * descriptor = graph.data.logits->src[3];
        if (graph.sched) {
            GGML_ASSERT(ggml_backend_sched_get_tensor_backend(graph.sched.get(), payload) == backend.get());
            GGML_ASSERT(ggml_backend_sched_get_tensor_backend(graph.sched.get(), descriptor) == backend.get());
            GGML_ASSERT(ggml_backend_sched_get_n_splits(graph.sched.get()) == 1);
        }
        grammar_mask_upload_counter counter(backend.get(), payload);
        const size_t capacity = payload->ne[0]/n_words;
        GGML_ASSERT(capacity >= 2 && capacity <= 8);
        const auto check_state = [&](size_t state, size_t slot) {
            llama_sampler_copy(states[state].get(), sampler.get());
            graph.check(states[state].get());
            std::array<int32_t, 4> desc;
            ggml_backend_tensor_get(descriptor, desc.data(), 0, sizeof(desc));
            GGML_ASSERT(desc[2] == int64_t(slot)*n_words);
        };
        for (size_t i = 0; i < capacity; ++i) {
            check_state(i, i);
        }
        GGML_ASSERT(counter.uploads == capacity);
        if (mode == 0 && counter.set_tensor_n) {
            GGML_ASSERT(counter.batches == capacity);
        }
        const size_t payload_bytes = counter.payload_bytes;
        GGML_ASSERT(payload_bytes <= capacity*n_words*sizeof(int32_t));
        graph.check(states[capacity - 1].get());
        GGML_ASSERT(counter.uploads == capacity && counter.descriptors == capacity);
        GGML_ASSERT(counter.payload_bytes == payload_bytes);
        check_state(0, 0);
        GGML_ASSERT(counter.uploads == capacity);
        check_state(capacity, 1);
        GGML_ASSERT(counter.uploads == capacity + 1);
        check_state(0, 0);
        GGML_ASSERT(counter.uploads == capacity + 1);
        check_state(1, 2);
        GGML_ASSERT(counter.uploads == capacity + 2);
        printf("grammar mask cache mode %d: %zu payload uploads, %zu bytes; unchanged hit uploads 0 bytes\n",
                mode, counter.uploads, counter.payload_bytes);
        llama_sampler_reset(sampler.get());
        graph.check(states[0].get());
        std::array<int32_t, 4> desc;
        ggml_backend_tensor_get(descriptor, desc.data(), 0, sizeof(desc));
        GGML_ASSERT(desc[2] == 0);
        const llama_sampler_row missing = {0, LLAMA_SAMPLER_PREFIX_LINEAR, nullptr, 1};
        GGML_ASSERT(!sampler->iface->backend_prepare(sampler.get(), &missing, 1));
        GGML_ASSERT(!sampler->iface->backend_prepare(sampler.get(), nullptr, 1));
    }
    {
        grammar_test_graph graph(sampler.get(), backend.get(), n_vocab, {}, true);
        grammar_mask_upload_counter counter(backend.get(), graph.data.logits->src[2]);
        graph.check(states[0].get());
        GGML_ASSERT(counter.uploads == 1 && counter.descriptors == 1);
        std::array<int32_t, 4> desc;
        ggml_backend_tensor_get(graph.data.logits->src[3], desc.data(), 0, sizeof(desc));
        GGML_ASSERT(desc[2] == 0);
    }
    printf("backend grammar resident cache hit, LRU eviction, reset and rebuild test PASSED\n");
}

#ifdef LLAMA_USE_LLGUIDANCE
static void test_llguidance_backend(const test_params & params) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const auto make_grammar = [&]() {
        llama_sampler_ptr result(llama_sampler_init_llg(vocab, "lark", "start: /[ab]/ /[bc]/ /[cd]/"));
        GGML_ASSERT(result);
        return result;
    };
    const auto a = common_tokenize(vocab, "a", false, true);
    const auto b = common_tokenize(vocab, "b", false, true);
    const auto z = common_tokenize(vocab, "z", false, true);
    GGML_ASSERT(a.size() == 1 && b.size() == 1 && z.size() == 1);
    ggml_backend_ptr backend(ggml_backend_dev_init(params.device, nullptr));
    GGML_ASSERT(backend);
    const auto buft = ggml_backend_get_default_buffer_type(backend.get());
    auto source = make_grammar();
    auto reference = make_grammar();
    GGML_ASSERT(source->iface->backend_init(source.get(), buft, 4));
    {
        grammar_test_graph graph(source.get(), backend.get(), n_vocab);
        graph.check(reference.get());
        graph.check(reference.get());
        llama_sampler_accept(source.get(), a[0]);
        llama_sampler_accept(reference.get(), a[0]);
        graph.check(reference.get());
    }
    llama_sampler_ptr clone(llama_sampler_clone(source.get()));
    {
        grammar_test_graph graph(clone.get(), backend.get(), n_vocab, {z[0], b[0], a[0], b[0], -1, n_vocab});
        graph.check(reference.get());
    }
    auto destination = make_grammar();
    {
        grammar_test_graph graph(destination.get(), backend.get(), n_vocab);
        llama_sampler_copy(source.get(), destination.get());
        graph.check(reference.get());
        llama_sampler_accept(source.get(), b[0]);
        graph.check(reference.get());
        llama_sampler_reset(destination.get());
        llama_sampler_reset(reference.get());
        graph.check(reference.get());
    }
    {
        grammar_test_graph rebuilt(destination.get(), backend.get(), n_vocab);
        rebuilt.check(reference.get());
    }
    const llama_sampler_row valid = {0, LLAMA_SAMPLER_PREFIX_LINEAR, a.data(), a.size()};
    GGML_ASSERT(destination->iface->backend_prepare(destination.get(), &valid, 1));
    GGML_ASSERT(destination->iface->backend_row_usable(destination.get(), 0));
    {
        auto provisional = make_grammar();
        llama_sampler_accept(provisional.get(), a[0]);
        grammar_test_graph graph(destination.get(), backend.get(), n_vocab);
        graph.check(provisional.get());
    }
    const llama_sampler_row invalid = {0, LLAMA_SAMPLER_PREFIX_TREE, z.data(), z.size()};
    GGML_ASSERT(destination->iface->backend_prepare(destination.get(), &invalid, 1));
    GGML_ASSERT(!destination->iface->backend_row_usable(destination.get(), 0));
    const llama_sampler_row missing = {0, LLAMA_SAMPLER_PREFIX_LINEAR, nullptr, 1};
    GGML_ASSERT(!destination->iface->backend_prepare(destination.get(), &missing, 1));
    GGML_ASSERT(!destination->iface->backend_prepare(destination.get(), nullptr, 1));
    GGML_ASSERT(destination->iface->backend_prepare(destination.get(), nullptr, 0));
    {
        grammar_test_graph graph(destination.get(), backend.get(), n_vocab);
        graph.check(reference.get());
    }
    source.reset();
    {
        auto expected = make_grammar();
        llama_sampler_accept(expected.get(), a[0]);
        grammar_test_graph graph(clone.get(), backend.get(), n_vocab);
        graph.check(expected.get());
    }
    auto broken = make_grammar();
    bool failed = false;
    try {
        llama_sampler_accept(broken.get(), z[0]);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    GGML_ASSERT(failed);
    llama_sampler_ptr broken_clone(llama_sampler_clone(broken.get()));
    llama_sampler_copy(broken.get(), destination.get());
    for (auto * sampler : {broken.get(), broken_clone.get(), destination.get()}) {
        failed = false;
        try {
            grammar_test_graph graph(sampler, backend.get(), n_vocab);
            sampler->iface->backend_set_input(sampler, nullptr);
        } catch (const std::runtime_error &) {
            failed = true;
        }
        GGML_ASSERT(failed);
        failed = false;
        llama_token_data value = {a[0], 17.0f, 0.0f};
        llama_token_data_array cur = {&value, 1, -1, false};
        try {
            llama_sampler_apply(sampler, &cur);
        } catch (const std::runtime_error &) {
            failed = true;
        }
        GGML_ASSERT(failed);
        failed = false;
        try {
            llama_sampler_reset(sampler);
        } catch (const std::runtime_error &) {
            failed = true;
        }
        GGML_ASSERT(failed);
    }
    GGML_ASSERT(!llama_sampler_init_llg(vocab, "lark", "start: ("));
    printf("llguidance backend masks, lifecycle, provisional rows and errors test PASSED\n");
}
#endif

static void test_grammar_lifecycle(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const auto tokens = grammar_test_tokens(vocab);
    const std::string grammar = "root ::= " + grammar_token(tokens[0]) + " " +
        grammar_token(tokens[1]) + " " + grammar_token(tokens[2]);
    const auto make_grammar = [&]() {
        return llama_sampler_ptr(llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
    };
    ggml_backend_ptr backend(ggml_backend_dev_init(params.device, nullptr));
    GGML_ASSERT(backend);
    const auto buft = ggml_backend_get_default_buffer_type(backend.get());
    auto source = make_grammar();
    auto reference = make_grammar();
    const llama_sampler_row invalid = {0, LLAMA_SAMPLER_PREFIX_LINEAR, &tokens[1], 1};
    GGML_ASSERT(source->iface->backend_prepare(source.get(), &invalid, 1));
    GGML_ASSERT(!source->iface->backend_row_usable(source.get(), 0));
    GGML_ASSERT(source->iface->backend_init(source.get(), buft, 2));
    GGML_ASSERT(source->iface->backend_row_usable(source.get(), 0));
    GGML_ASSERT(source->iface->backend_init(source.get(), buft, 1));
    source->iface->backend_set_input(source.get(), nullptr);
    {
        grammar_test_graph graph(source.get(), backend.get(), n_vocab);
        graph.check(reference.get());
        graph.check(reference.get());
        llama_sampler_accept(source.get(), tokens[0]);
        llama_sampler_accept(reference.get(), tokens[0]);
        graph.check(reference.get());
    }

    llama_sampler_ptr cloned(llama_sampler_clone(source.get()));
    GGML_ASSERT(cloned->iface->backend_init(cloned.get(), buft, 1));
    {
        grammar_test_graph graph(cloned.get(), backend.get(), n_vocab,
                { tokens[2], tokens[0], tokens[0], tokens[1] });
        graph.check(reference.get());
        llama_sampler_accept(cloned.get(), tokens[1]);
        llama_sampler_ptr advanced(llama_sampler_clone(reference.get()));
        llama_sampler_accept(advanced.get(), tokens[1]);
        graph.check(advanced.get());
    }

    auto destination = make_grammar();
    GGML_ASSERT(destination->iface->backend_init(destination.get(), buft, 1));
    {
        grammar_test_graph graph(destination.get(), backend.get(), n_vocab);
        auto initial = make_grammar();
        graph.check(initial.get());
        auto * destination_ctx = destination->ctx;
        llama_sampler_copy(source.get(), destination.get());
        GGML_ASSERT(destination->ctx == destination_ctx);
        source.reset();
        graph.check(reference.get());
        llama_sampler_accept(destination.get(), tokens[1]);
        llama_sampler_accept(reference.get(), tokens[1]);
        graph.check(reference.get());
    }
    {
        grammar_test_graph rebuilt(destination.get(), backend.get(), n_vocab);
        rebuilt.check(reference.get());
        llama_sampler_reset(destination.get());
        llama_sampler_reset(reference.get());
        rebuilt.check(reference.get());
    }

    for (const char * rule : { "root ::= [a-z]+", "root ::= .+" }) {
        llama_sampler_ptr sampler(llama_sampler_init_grammar(vocab, rule, "root"));
        llama_sampler_ptr cpu(llama_sampler_clone(sampler.get()));
        GGML_ASSERT(sampler->iface->backend_init(sampler.get(), buft, 1));
        grammar_test_graph graph(sampler.get(), backend.get(), n_vocab);
        graph.check(cpu.get());
    }

    const auto make_lazy = [&]() {
        return llama_sampler_ptr(llama_sampler_init_grammar_lazy_patterns(vocab, grammar.c_str(), "root",
                    nullptr, 0, tokens.data(), 1));
    };
    auto lazy = make_lazy();
    auto lazy_cpu = make_lazy();
    GGML_ASSERT(lazy->iface->backend_init(lazy.get(), buft, 1));
    {
        grammar_test_graph graph(lazy.get(), backend.get(), n_vocab);
        graph.check(lazy_cpu.get());
        llama_sampler_accept(lazy.get(), tokens[2]);
        llama_sampler_accept(lazy_cpu.get(), tokens[2]);
        graph.check(lazy_cpu.get());
        llama_sampler_ptr snapshot(llama_sampler_clone(lazy.get()));
        llama_sampler_accept(lazy.get(), tokens[0]);
        llama_sampler_accept(lazy_cpu.get(), tokens[0]);
        graph.check(lazy_cpu.get());
        llama_sampler_copy(snapshot.get(), lazy.get());
        lazy_cpu = make_lazy();
        llama_sampler_accept(lazy_cpu.get(), tokens[2]);
        graph.check(lazy_cpu.get());
        llama_sampler_accept(lazy.get(), tokens[0]);
        llama_sampler_accept(lazy_cpu.get(), tokens[0]);
        graph.check(lazy_cpu.get());
        llama_sampler_reset(lazy.get());
        llama_sampler_reset(lazy_cpu.get());
        graph.check(lazy_cpu.get());
    }

    const std::string trigger_word = common_token_to_piece(vocab, tokens[0]) + common_token_to_piece(vocab, tokens[1]);
    std::string pattern = "[\\s\\S]*?(";
    for (char c : trigger_word) {
        if (std::strchr(".^$|()*+?[]{}\\", c)) {
            pattern += '\\';
        }
        pattern += c;
    }
    pattern += ")[\\s\\S]*";
    const char * trigger = pattern.c_str();
    const auto make_regex = [&]() {
        return llama_sampler_ptr(llama_sampler_init_grammar_lazy_patterns(vocab, grammar.c_str(), "root", &trigger, 1, nullptr, 0));
    };
    auto regex = make_regex();
    auto regex_cpu = make_regex();
    GGML_ASSERT(regex->iface->backend_init(regex.get(), buft, 1));
    {
        grammar_test_graph graph(regex.get(), backend.get(), n_vocab);
        llama_sampler_accept(regex.get(), tokens[0]);
        llama_sampler_accept(regex_cpu.get(), tokens[0]);
        graph.check(regex_cpu.get());
        llama_sampler_ptr pending(llama_sampler_clone(regex.get()));
        llama_sampler_accept(regex.get(), tokens[1]);
        llama_sampler_accept(regex_cpu.get(), tokens[1]);
        graph.check(regex_cpu.get());
        llama_sampler_ptr triggered(llama_sampler_clone(regex.get()));
        llama_sampler_copy(pending.get(), regex.get());
        regex_cpu = make_regex();
        llama_sampler_accept(regex_cpu.get(), tokens[0]);
        graph.check(regex_cpu.get());
        llama_sampler_accept(regex.get(), tokens[1]);
        llama_sampler_accept(regex_cpu.get(), tokens[1]);
        graph.check(regex_cpu.get());
        llama_sampler_reset(regex.get());
        llama_sampler_copy(triggered.get(), regex.get());
        triggered.reset();
        graph.check(regex_cpu.get());
    }
    printf("backend grammar lifecycle test PASSED\n");
}

static void test_grammar(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    const std::string grammar = "root ::= (" + grammar_token(tokens[0]) + " | " + grammar_token(tokens[1]) + ")+";
    const auto raw = decode_raw_logits(params, "Hello");
    for (bool cpu_suffix : { false, true }) {
        auto add_prefix = [&](llama_sampler * chain) {
            llama_sampler_chain_add(chain, llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(0.75f));
        };
        auto add_samplers = [&](llama_sampler * chain) {
            add_prefix(chain);
            if (cpu_suffix) {
                llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(1.0f));
                llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            }
        };
        auto cpu = make_sampler_chain(add_samplers, [](llama_sampler *) {});
        auto cpu_prefix = make_sampler_chain(add_prefix, [](llama_sampler *) {});
        auto backend = make_sampler_chain(add_samplers, [](llama_sampler *) {});
        std::vector<llama_sampler_seq_config> configs = {{ 0, backend.get() }};
        test_context ctx(params, configs);
        GGML_ASSERT(ctx.decode({{ 0, "Hello" }}));
        const int32_t idx = ctx.idx_for_seq(0);
        const auto expected = apply_cpu_sampler(raw, cpu.get());
        const auto expected_prefix = apply_cpu_sampler(raw, cpu_prefix.get());
        const uint32_t count = llama_get_sampled_logits_count_ith(ctx.ctx.get(), idx);
        GGML_ASSERT(count == raw.size());
        const float * actual = llama_get_sampled_logits_ith(ctx.ctx.get(), idx);
        for (uint32_t i = 0; i < count; ++i) {
            if (std::isinf(expected_prefix[i].logit)) {
                GGML_ASSERT(actual[i] == expected_prefix[i].logit);
            } else {
                GGML_ASSERT(fabsf(actual[i] - expected_prefix[i].logit) < 1e-3f);
            }
        }
        GGML_ASSERT(llama_get_sampled_token_ith(ctx.ctx.get(), idx) == LLAMA_TOKEN_NULL);
        if (cpu_suffix) {
            const llama_token token = llama_sampler_sample(backend.get(), ctx.ctx.get(), idx);
            const auto best = std::max_element(expected.begin(), expected.end(),
                    [](const llama_token_data & a, const llama_token_data & b) { return a.logit < b.logit; });
            GGML_ASSERT(token == best->id);
        }
    }
    printf("backend grammar and CPU suffix test PASSED\n");
}

static void test_grammar_multi_sequence(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    std::vector<llama_sampler_ptr> chains;
    std::vector<llama_sampler_seq_config> configs;
    for (int i = 0; i < 3; ++i) {
        llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        const std::string term = grammar_token(tokens[i]);
        const std::string grammar = "root ::= " + term + " " + term + " " + term;
        llama_sampler_chain_add(chain.get(), llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        configs.push_back({ i, chain.get() });
        chains.push_back(std::move(chain));
    }
    test_context ctx(params, configs, 3, 3, 2);
    for (const std::vector<int> & order : { std::vector<int>{ 2, 0 }, { 1 }, { 0, 2, 1 } }) {
        common_batch batch(ctx.ctx.get());
        for (int seq : order) {
            batch.add(llama_vocab_bos(vocab), ctx.seq_positions[seq]++, seq, true);
        }
        GGML_ASSERT(llama_process(ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
        for (size_t row = 0; row < order.size(); ++row) {
            const int seq = order[row];
            const llama_token token = llama_sampler_sample(chains[seq].get(), ctx.ctx.get(), row);
            GGML_ASSERT(token == tokens[seq]);
        }
    }
    printf("backend different sequence grammars test PASSED\n");
}

static void test_grammar_fallback(const test_params & params) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    const std::string grammar = "root ::= " + grammar_token(tokens[0]) + " " + grammar_token(tokens[1]);
    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(0.75f));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    std::vector<llama_sampler_seq_config> configs = {{ 0, chain.get() }};
    test_context ctx(params, configs, 1, 2, 0, 2);
    common_batch batch(ctx.ctx.get());
    for (int i = 0; i < 2; ++i) {
        batch.add(llama_vocab_bos(vocab), i, 0, true);
    }
    GGML_ASSERT(llama_process(ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == -1);
    llama_sampler_row rows[] = {
        {0, LLAMA_SAMPLER_PREFIX_LINEAR, nullptr, 0},
        {1, LLAMA_SAMPLER_PREFIX_LINEAR, tokens.data(), 1},
    };
    GGML_ASSERT(llama_sampler_prepare(ctx.ctx.get(), 0, rows, 2));
    GGML_ASSERT(llama_process(ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    for (int row = 0; row < 2; ++row) {
        GGML_ASSERT(llama_sampler_sample(chain.get(), ctx.ctx.get(), row) == tokens[row]);
    }
    printf("backend grammar explicit multi-output prefix requirement test PASSED\n");
}

static void test_penalties(const test_params & params) {
    printf("Testing backend penalties (repeat + freq + presence)\n");
    compare_penalties_logits(params, 64, 1.1f, 0.5f, 0.25f, "Hello Hello world");

    printf("Testing backend penalties with penalty_last_n > 64\n");
    const auto * vocab = llama_model_get_vocab(params.model.get());
    std::vector<llama_token> tokens(8);
    int32_t n_tok = llama_tokenize(vocab, "a", 1, tokens.data(), (int32_t) tokens.size(), false, false);
    if (n_tok < 0) {
        tokens.resize(-n_tok);
        n_tok = llama_tokenize(vocab, "a", 1, tokens.data(), (int32_t) tokens.size(), false, false);
    }
    GGML_ASSERT(n_tok > 0);
    const llama_token tok = tokens[0];

    compare_penalties_logits(params, 80, 1.15f, 0.1f, 0.05f, "a", [tok](llama_sampler * smpl) {
        // accept_prompt already accepted BOS + one 'a'; fill the ring to n=80
        for (int i = 0; i < 78; ++i) {
            llama_sampler_accept(smpl, tok);
        }
    });

    printf("Testing backend penalties without filler entries\n");
    compare_penalties_logits(params, 64, 1.1f, 0.5f, 0.25f, "Hello", [](llama_sampler * smpl) {
        for (llama_token token = 0; token < 64; ++token) {
            llama_sampler_accept(smpl, token);
        }
    });

    printf("Testing backend top-k followed by penalties\n");
    compare_top_k_penalties_logits(params, 8, 64, 1.1f, 0.5f, 0.25f, "Hello",
            penalties_position::after_filter);

    printf("Testing backend penalties followed by top-k\n");
    compare_top_k_penalties_logits(params, 8, 64, 1.1f, 0.5f, 0.25f, "Hello",
            penalties_position::before_filter);

    printf("Testing backend top-p followed by penalties\n");
    compare_masking_penalties_logits(params, "top-p", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend top-p followed by penalties with a large history window\n");
    compare_masking_penalties_logits(params, "top-p large-window", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 4096, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend penalties followed by top-p\n");
    compare_masking_penalties_logits(params, "top-p", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::before_filter, true);

    printf("Testing backend min-p followed by penalties\n");
    compare_masking_penalties_logits(params, "min-p", []() {
        return llama_sampler_init_min_p(0.1f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, false);

    printf("Testing backend penalties followed by min-p\n");
    compare_masking_penalties_logits(params, "min-p", []() {
        return llama_sampler_init_min_p(0.1f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::before_filter, false);

    printf("Testing backend top-p followed by penalties with empty history\n");
    compare_masking_penalties_logits(params, "top-p empty", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true, false);

    printf("Testing backend top-p followed by individual penalties\n");
    compare_masking_penalties_logits(params, "top-p repeat", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.0f, 0.0f, "Hello", penalties_position::after_filter, true);
    compare_masking_penalties_logits(params, "top-p frequency", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.0f, 0.5f, 0.0f, "Hello", penalties_position::after_filter, true);
    compare_masking_penalties_logits(params, "top-p presence", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.0f, 0.0f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend penalty parameter values\n");
    check_penalty_parameter_values(params);

    printf("backend penalties sampling test PASSED\n");
}

// This test verifies that it is possible to have two different backend samplers,
// one that uses the backend dist sampler, and another that uses CPU dist sampler.
static void test_mixed(const test_params & params) {
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_dist(88));

    int k = 40;
    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_1(llama_sampler_chain_init(chain_params_1));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_top_k(k));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
        { 1, sampler_chain_1.get() }
    };

    test_context test_ctx(params, backend_sampler_configs);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"},
        {1, "Some"}
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0 that used the dist backend sampler.
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        //GGML_ASSERT(llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx) == nullptr);
        //GGML_ASSERT(llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx) == 0);
    }

    // Verify sequence 1 that used the top-k backend sampler.
    {
        int32_t batch_idx = test_ctx.idx_for_seq(1);
        float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(logits != nullptr);
        size_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(n_logits == (size_t) k);
        GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx) == LLAMA_TOKEN_NULL);
    }

    printf("backend mixed sampling test PASSED\n");
}

static void test_set_sampler(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    // Sample using backend sampler configured above
    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    const std::string backend_token_str = test_ctx.token_to_piece(backend_token, false);
    printf("dist sampled token = %d, string='%s'\n", backend_token, backend_token_str.c_str());

    // Now clear the backend sampler for this sequence.
    llama_set_sampler(test_ctx.ctx.get(), seq_id, nullptr);
    printf("Cleared backend sampler for seq_id %d\n", seq_id);

    // Sample using CPU sampler
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

    std::map<llama_seq_id, llama_token> tokens = { { seq_id, backend_token}, };
    if (!test_ctx.decode_tokens(tokens)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Should not have any sampled token or probs after clearing the backend sampler.
    const int32_t idx = test_ctx.idx_for_seq(seq_id);
    GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), idx) == LLAMA_TOKEN_NULL);
    GGML_ASSERT(llama_get_sampled_probs_ith(test_ctx.ctx.get(), idx) == nullptr);

    // Sample the token using the CPU sampler chain.
    llama_token token2 = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), seq_id);
    const std::string token2_str = test_ctx.token_to_piece(token2, false);
    printf("CPU sampled token after clearing backend sampler: id=%d, string='%s'\n", token2, token2_str.c_str());
    std::map<llama_seq_id, llama_token> tokens2 = { { seq_id, token2}, };

    // Set a new backend sampler for the sequence.
    struct llama_sampler_chain_params new_backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr new_backend_sampler_chain(llama_sampler_chain_init(new_backend_chain_params));
    llama_sampler_chain_add(new_backend_sampler_chain.get(), llama_sampler_init_top_k(20));
    llama_sampler_chain_add(new_backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    llama_set_sampler(test_ctx.ctx.get(), seq_id, new_backend_sampler_chain.get());

    if (!test_ctx.decode_tokens(tokens2)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    llama_token new_backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), test_ctx.idx_for_seq(seq_id));
    const std::string new_backend_token_str = test_ctx.token_to_piece(new_backend_token, false);
    printf("dist sampled token = %d, string='%s'\n", new_backend_token, new_backend_token_str.c_str());

    printf("backend set sampler test PASSED\n");
}

static void test_cpu_mixed(const test_params & params) {
    // Sequence 0 uses backend sampling
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_dist(88));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
    };

    // We need 2 sequences: seq 0 with backend sampling, seq 1 with CPU sampling
    test_context test_ctx(params, backend_sampler_configs, 2);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"}, // Will use backend sampling
        {1, "Some"}   // Will use CPU sampling
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0 (backend sampled)
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 0 (backend) sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Verify sequence 1 (CPU sampled)
    {
        int32_t batch_idx = test_ctx.idx_for_seq(1);

        llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(backend_token == LLAMA_TOKEN_NULL);

        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());

        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 1 (CPU) sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Clear/remove the backend sampler, and sample again
    {
        // clear the backend sampler for seq 0 so that there are no backend
        // samplers.
        llama_set_sampler(test_ctx.ctx.get(), 0, nullptr);

        // Create a CPU sampler and verify we can sample from it.
        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());

        int32_t batch_idx = test_ctx.idx_for_seq(1);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
        if (!test_ctx.decode_token(token, 1)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    // Set a backend sampler so that we can verify that it can be reset
    {
        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr sampler_chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_dist(88));

        llama_set_sampler(test_ctx.ctx.get(), 0, sampler_chain.get());

        if (!test_ctx.decode_token(3834, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("re-added backend sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    printf("backend-cpu mixed batch test PASSED\n");
}

static void test_multi_output_limit(const test_params & params) {
    const llama_seq_id seq_id = 0;

    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 3, 0, 2);

    common_batch batch(test_ctx.ctx.get());
    for (int i = 0; i < 3; ++i) {
        batch.add(llama_vocab_bos(test_ctx.vocab), i, seq_id, true);
    }

    printf(">>> test_multi_output_limit expected error start:\n");
    const int ret = llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
    GGML_ASSERT(ret != 0 && "llama_decode should reject outputs above the per-sequence limit");
    printf("<<< test_multi_output_limit expected error end.\n");

    printf("backend multi-output limit test PASSED\n");
}

static void test_multi_output_multi_sequence_dist(const test_params & params) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t seeds[] = { 88, 1337 };
    // reduce the chance that swapped random inputs select the same token
    const float temp = 10.0f;

    llama_sampler_ptr chain_0(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_ptr chain_1(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain_0.get(), llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain_0.get(), llama_sampler_init_dist(seeds[0]));
    llama_sampler_chain_add(chain_1.get(), llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain_1.get(), llama_sampler_init_dist(seeds[1]));
    std::vector<llama_sampler_seq_config> configs = {
        { 0, chain_0.get() },
        { 1, chain_1.get() },
    };
    test_context test_ctx(params, configs, 2, 4, 0, 2);

    std::vector<llama_sampler_seq_config> reference_configs;
    test_context reference_ctx(params, reference_configs, 2, 4);

    const llama_token seq_tokens[2][2] = {
        { llama_vocab_bos(vocab), llama_vocab_eos(vocab) },
        { llama_vocab_eos(vocab), llama_vocab_bos(vocab) },
    };

    // a batch belongs to one context, so it is built per context
    auto make_batch = [&](llama_context * ctx) {
        common_batch batch(ctx);
        for (int pos = 0; pos < 2; ++pos) {
            batch.add(seq_tokens[0][pos], pos, 0, true);
            batch.add(seq_tokens[1][pos], pos, 1, true);
        }
        return batch;
    };

    common_batch batch = make_batch(test_ctx.ctx.get());
    GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    {
        common_batch batch_ref = make_batch(reference_ctx.ctx.get());
        GGML_ASSERT(llama_process(reference_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_ref.get()) == 0);
    }

    std::mt19937 reference_rngs[] = {
        std::mt19937(seeds[0]),
        std::mt19937(seeds[1]),
    };
    std::uniform_real_distribution<double> reference_dist(0.0, 1.0);

    for (int i = 0; i < batch.size(); ++i) {
        const llama_seq_id seq_id = batch.tokens[i].seq_id;
        GGML_ASSERT(seq_id == 0 || seq_id == 1);

        llama_sampler * chain = seq_id == 0 ? chain_0.get() : chain_1.get();
        const llama_token backend_token = llama_sampler_sample(chain, test_ctx.ctx.get(), i);
        const float * sampled_logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), i);
        const float * sampled_probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), i);
        const uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_probs = llama_get_sampled_probs_count_ith(test_ctx.ctx.get(), i);
        const float * reference_logits = llama_get_logits_ith(reference_ctx.ctx.get(), i);

        GGML_ASSERT(backend_token >= 0 && backend_token < n_vocab);
        GGML_ASSERT(sampled_logits != nullptr);
        GGML_ASSERT(sampled_probs != nullptr);
        GGML_ASSERT(reference_logits != nullptr);
        GGML_ASSERT(n_logits == (uint32_t) n_vocab);
        GGML_ASSERT(n_probs == (uint32_t) n_vocab);

        float prob_sum = 0.0f;
        float cumsum_before = 0.0f;
        for (llama_token token = 0; token < n_vocab; ++token) {
            const float expected_logit = reference_logits[token] / temp;
            const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected_logit));
            GGML_ASSERT(std::fabs(sampled_logits[token] - expected_logit) <= tolerance);
            GGML_ASSERT(std::isfinite(sampled_probs[token]));
            GGML_ASSERT(sampled_probs[token] >= 0.0f);

            prob_sum += sampled_probs[token];
            if (token < backend_token) {
                cumsum_before += sampled_probs[token];
            }
        }

        GGML_ASSERT(std::fabs(prob_sum - 1.0f) <= 1e-3f);

        const float rnd = reference_dist(reference_rngs[seq_id]);
        const float cumsum_sampled = cumsum_before + sampled_probs[backend_token];
        GGML_ASSERT(rnd >= cumsum_before - 1e-4f);
        GGML_ASSERT(rnd <= cumsum_sampled + 1e-4f);
    }

    printf("backend multi-sequence multi-output dist test PASSED\n");
}

static void test_multi_output_dist_transaction(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const uint32_t seed = 95;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());

    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(10.0f));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 3, 2, 3);

    auto verify_random = [&](int32_t row, float rnd, bool accept = true) {
        const llama_token token = accept ?
            llama_sampler_sample(chain.get(), test_ctx.ctx.get(), row) :
            llama_get_sampled_token_ith(test_ctx.ctx.get(), row);
        const float * probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), row);

        GGML_ASSERT(token >= 0 && token < llama_vocab_n_tokens(vocab));
        GGML_ASSERT(probs != nullptr);

        float cumsum_before = 0.0f;
        for (llama_token i = 0; i < token; ++i) {
            cumsum_before += probs[i];
        }

        const float cumsum_sampled = cumsum_before + probs[token];
        GGML_ASSERT(rnd >= cumsum_before - 1e-4f);
        GGML_ASSERT(rnd <= cumsum_sampled + 1e-4f);
    };

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    float randoms[3];
    for (float & rnd : randoms) {
        rnd = dist(rng);
    }

    int32_t pos = 0;
    auto decode = [&]() {
        common_batch batch(test_ctx.ctx.get());
        for (int32_t i = 0; i < 3; ++i) {
            batch.add(llama_vocab_bos(vocab), pos++, seq_id, true);
        }
        GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    };

    decode();
    verify_random(0, randoms[0], false);

    decode();
    verify_random(0, randoms[0]);
    verify_random(1, randoms[1]);

    decode();
    llama_sampler_ptr saved(llama_sampler_clone(chain.get()));
    verify_random(0, randoms[2]);

    llama_sampler_copy(saved.get(), chain.get());

    decode();
    verify_random(0, randoms[2]);

    printf("backend multi-output dist transaction test PASSED\n");
}

static void test_multi_output_sampling_chain(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const uint32_t seed = 88;
    const float p = 0.9f;
    const float temp = 0.8f;
    const float cdf_epsilon = 1e-4f;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t k = std::min<uint32_t>(512, n_vocab);
    const llama_logit_bias bias = { llama_vocab_bos(vocab), -0.1f };

    auto make_filter_chain = [&]() {
        llama_sampler_ptr result(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(result.get(), llama_sampler_init_logit_bias(n_vocab, 1, &bias));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_p(p, 1));
        llama_sampler_chain_add(result.get(), llama_sampler_init_min_p(0.01f, 1));
        llama_sampler_chain_add(result.get(), llama_sampler_init_temp(temp));
        return result;
    };

    llama_sampler_ptr chain = make_filter_chain();
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 2, 2, 2);

    std::vector<llama_sampler_seq_config> reference_configs;
    test_context reference_ctx(params, reference_configs, 1, 2, 2);

    llama_sampler_ptr reference_bias(llama_sampler_init_logit_bias(n_vocab, 1, &bias));
    llama_sampler_ptr reference_top_k(llama_sampler_init_top_k(k));
    llama_sampler_ptr reference_top_p(llama_sampler_init_top_p(p, 1));
    llama_sampler_ptr reference_min_p(llama_sampler_init_min_p(0.01f, 1));
    llama_sampler_ptr reference_temp(llama_sampler_init_temp(temp));
    std::vector<llama_token_data> reference_data(n_vocab);

    // a batch belongs to one context, so it is built per context
    auto make_batch = [&](llama_context * ctx, int32_t pos) {
        common_batch batch(ctx);
        for (int i = 0; i < 2; ++i) {
            batch.add(llama_vocab_bos(vocab), pos + i, seq_id, true);
        }
        return batch;
    };

    common_batch batch = make_batch(test_ctx.ctx.get(), 0);
    GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    {
        common_batch batch_ref = make_batch(reference_ctx.ctx.get(), 0);
        GGML_ASSERT(llama_process(reference_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_ref.get()) == 0);
    }

    for (int i = 0; i < batch.size(); ++i) {
        const llama_token backend_token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), i);
        const float * sampled_logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), i);
        const float * sampled_probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), i);
        const llama_token * sampled_candidates = llama_get_sampled_candidates_ith(test_ctx.ctx.get(), i);
        const uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_probs = llama_get_sampled_probs_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_candidates = llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), i);
        const float * reference_logits = llama_get_logits_ith(reference_ctx.ctx.get(), i);

        GGML_ASSERT(backend_token >= 0 && backend_token < n_vocab);
        GGML_ASSERT(sampled_logits != nullptr);
        GGML_ASSERT(sampled_probs != nullptr);
        GGML_ASSERT(sampled_candidates != nullptr);
        GGML_ASSERT(reference_logits != nullptr);
        GGML_ASSERT(n_logits == k);
        GGML_ASSERT(n_probs == n_logits);
        GGML_ASSERT(n_candidates == n_logits);

        for (llama_token token = 0; token < n_vocab; ++token) {
            reference_data[token] = { token, reference_logits[token], 0.0f };
        }

        llama_token_data_array reference = {
            /* .data     = */ reference_data.data(),
            /* .size     = */ reference_data.size(),
            /* .selected = */ LLAMA_TOKEN_NULL,
            /* .sorted   = */ false,
        };

        llama_sampler_apply(reference_bias.get(), &reference);
        llama_sampler_apply(reference_top_k.get(), &reference);
        llama_sampler_apply(reference_top_p.get(), &reference);
        GGML_ASSERT(reference.size > 0);

        float cdf = 0.0f;
        for (size_t j = 0; j < reference.size; ++j) {
            cdf += reference.data[j].p;
        }
        const float cdf_before = cdf - reference.data[reference.size - 1].p;
        const float boundary_distance = std::min(std::fabs(cdf_before - p), std::fabs(cdf - p));

        llama_sampler_apply(reference_min_p.get(), &reference);
        llama_sampler_apply(reference_temp.get(), &reference);

        std::unordered_map<llama_token, float> reference_by_id;
        for (size_t j = 0; j < reference.size; ++j) {
            reference_by_id.emplace(reference.data[j].id, reference.data[j].logit);
        }
        size_t n_backend_only = 0;
        int32_t sampled_index = -1;
        float prob_sum = 0.0f;

        for (uint32_t j = 0; j < n_logits; ++j) {
            GGML_ASSERT(sampled_candidates[j] >= 0 && sampled_candidates[j] < n_vocab);
            GGML_ASSERT(std::isfinite(sampled_probs[j]));
            GGML_ASSERT(sampled_probs[j] >= 0.0f);
            prob_sum += sampled_probs[j];

            if (sampled_candidates[j] == backend_token) {
                sampled_index = j;
            }
            if (!std::isfinite(sampled_logits[j])) {
                GGML_ASSERT(std::isinf(sampled_logits[j]) && sampled_logits[j] < 0.0f);
                GGML_ASSERT(sampled_probs[j] == 0.0f);
                continue;
            }

            const auto match = reference_by_id.find(sampled_candidates[j]);
            if (match == reference_by_id.end()) {
                ++n_backend_only;
                continue;
            }

            const float tolerance = 1e-4f * std::max(1.0f, std::fabs(match->second));
            GGML_ASSERT(std::fabs(sampled_logits[j] - match->second) <= tolerance);
            reference_by_id.erase(match);
        }

        const size_t n_reference_only = reference_by_id.size();

        if (n_backend_only != 0 || n_reference_only != 0) {
            GGML_ASSERT(n_backend_only <= 1);
            GGML_ASSERT(n_reference_only <= 1);
            GGML_ASSERT(boundary_distance <= cdf_epsilon);
        }

        GGML_ASSERT(sampled_index >= 0);
        GGML_ASSERT(std::isfinite(sampled_logits[sampled_index]));
        GGML_ASSERT(sampled_probs[sampled_index] > 0.0f);
        GGML_ASSERT(std::fabs(prob_sum - 1.0f) <= 1e-3f);
    }

    batch = make_batch(test_ctx.ctx.get(), 2);
    GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);

    printf("backend multi-output sampling chain test PASSED\n");
}

static void test_multi_output_cpu_suffix(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const int32_t k = 8;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());

    auto make_chain = [&](test_single_output_backend_sampler ** sampler_ctx) {
        llama_sampler_ptr result(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(result.get(), test_single_output_backend_sampler_init(sampler_ctx));
        llama_sampler_chain_add(result.get(), llama_sampler_init_dist(88));
        return result;
    };

    {
        test_single_output_backend_sampler * sampler_ctx = nullptr;
        llama_sampler_ptr chain = make_chain(&sampler_ctx);
        std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
        test_context test_ctx(params, configs, 1, 1, 0, 4);

        common_batch batch(test_ctx.ctx.get());
        batch.add(llama_vocab_bos(vocab), 0, seq_id, true);
        GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);

        GGML_ASSERT(sampler_ctx->backend_initialized);
        GGML_ASSERT(sampler_ctx->backend_outputs_max_per_seq == 1);
        GGML_ASSERT(sampler_ctx->backend_apply_count > 0);
        GGML_ASSERT(sampler_ctx->apply_count == 0);
        GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), 0) != LLAMA_TOKEN_NULL);
    }

    {
        test_single_output_backend_sampler * sampler_ctx = nullptr;
        llama_sampler_ptr chain = make_chain(&sampler_ctx);
        std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
        test_context test_ctx(params, configs, 1, 2, 0, 0);

        common_batch batch(test_ctx.ctx.get());
        for (int i = 0; i < 2; ++i) {
            batch.add(llama_vocab_bos(vocab), i, seq_id, true);
        }
        GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);

        GGML_ASSERT(!sampler_ctx->backend_initialized);
        GGML_ASSERT(sampler_ctx->backend_outputs_max_per_seq == 2);
        GGML_ASSERT(sampler_ctx->backend_apply_count == 0);
        for (int i = 0; i < batch.size(); ++i) {
            GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), i) == LLAMA_TOKEN_NULL);
            GGML_ASSERT(llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i) == (uint32_t) k);
            GGML_ASSERT(llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), i) == (uint32_t) k);
            const llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), i);
            GGML_ASSERT(token >= 0 && token < llama_vocab_n_tokens(vocab));
        }
        GGML_ASSERT(sampler_ctx->apply_count == batch.size());
    }

    printf("backend multi-output CPU suffix test PASSED\n");
}

static common_params_sampling common_grammar_params(const std::string & grammar) {
    common_params_sampling sp;
    sp.grammar = { COMMON_GRAMMAR_TYPE_USER, grammar };
    sp.grammar_first = true;
    sp.backend_sampling = true;
    sp.seed = 1234;
    sp.temp = 0.0f;
    sp.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    return sp;
}

static void test_grammar_common(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto tokens = grammar_test_tokens(llama_model_get_vocab(params.model.get()));
    const std::string grammar = "root ::= (" + grammar_token(tokens[0]) + " | " + grammar_token(tokens[1]) + ")+";
    for (bool cpu_suffix : { false, true }) {
        auto sp = common_grammar_params(grammar);
        sp.logit_bias = {{tokens[2], 1000.0f}, {tokens[0], 100.0f}};
        if (cpu_suffix) {
            sp.samplers.insert(sp.samplers.begin(), COMMON_SAMPLER_TYPE_TOP_N_SIGMA);
            sp.top_n_sigma = 1.0f;
        }
        common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
        GGML_ASSERT(sp.backend_sampling);
        auto reference_params = sp;
        reference_params.backend_sampling = false;
        common_sampler_ptr reference(common_sampler_init(params.model.get(), reference_params));
        std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(sampler.get())}};
        std::vector<llama_sampler_seq_config> no_configs;
        test_context ctx(params, configs);
        test_context cpu(params, no_configs);
        GGML_ASSERT(ctx.decode({{0, "Hello"}}));
        GGML_ASSERT(cpu.decode({{0, "Hello"}}));
        for (int i = 0; i < 3; ++i) {
            const int idx = ctx.idx_for_seq(0);
            GGML_ASSERT((llama_get_sampled_token_ith(ctx.ctx.get(), idx) == LLAMA_TOKEN_NULL) == cpu_suffix);
            const auto actual = common_sampler_sample(sampler.get(), ctx.ctx.get(), idx);
            const auto expected = common_sampler_sample(reference.get(), cpu.ctx.get(), cpu.idx_for_seq(0));
            GGML_ASSERT(actual == expected && actual == tokens[0]);
            common_sampler_accept(sampler.get(), actual, true);
            common_sampler_accept(reference.get(), expected, true);
            GGML_ASSERT(ctx.decode_token(actual));
            GGML_ASSERT(cpu.decode_token(expected));
        }
    }
    printf("common grammar-first backend and CPU suffix test PASSED\n");
}

#ifdef LLAMA_USE_LLGUIDANCE
static void test_llguidance_prepared_rows(const test_params & params) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto a = common_tokenize(vocab, "a", false, true);
    const auto z = common_tokenize(vocab, "z", false, true);
    GGML_ASSERT(a.size() == 1 && z.size() == 1);
    auto cp = llama_sampler_chain_default_params();
    cp.output_flags = LLAMA_SAMPLER_OUTPUT_CANDIDATES;
    llama_sampler_ptr chain(llama_sampler_chain_init(cp));
    auto * grammar = llama_sampler_init_llg(vocab, "lark", "start: /[ab]/ /[bc]/ /[cd]/");
    GGML_ASSERT(grammar);
    llama_sampler_chain_add(chain.get(), grammar);
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(1234));
    std::vector<llama_sampler_seq_config> configs = {{0, chain.get()}};
    test_context ctx(params, configs, 1, 3, 2, 3);
    const llama_sampler_row rows[] = {
        {2, LLAMA_SAMPLER_PREFIX_TREE, z.data(), z.size()},
        {1, LLAMA_SAMPLER_PREFIX_TEACHER_FORCED, a.data(), a.size()},
        {0, LLAMA_SAMPLER_PREFIX_COMMITTED, nullptr, 0},
    };
    for (int round = 0; round < 2; ++round) {
        GGML_ASSERT(llama_sampler_prepare(ctx.ctx.get(), 0, rows, 3));
        common_batch batch(ctx.ctx.get());
        for (int i = 0; i < 3; ++i) {
            batch.add(a[0], round*3 + i, 0, true);
        }
        GGML_ASSERT(llama_process(ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
        for (int i = 0; i < 3; ++i) {
            GGML_ASSERT(llama_sampler_row_usable(ctx.ctx.get(), i) == (i != 2));
            if (i == 2) {
                GGML_ASSERT(llama_get_sampled_token_ith(ctx.ctx.get(), i) == LLAMA_TOKEN_NULL);
                continue;
            }
            llama_sampler_ptr reference(llama_sampler_clone(grammar));
            if (i == 1) {
                llama_sampler_accept(reference.get(), a[0]);
            }
            std::vector<llama_token_data> values(llama_vocab_n_tokens(vocab));
            for (size_t id = 0; id < values.size(); ++id) {
                values[id] = {llama_token(id), 0.0f, 0.0f};
            }
            llama_token_data_array cur = {values.data(), values.size(), -1, false};
            llama_sampler_apply(reference.get(), &cur);
            const auto * logits = llama_get_sampled_logits_ith(ctx.ctx.get(), i);
            GGML_ASSERT(logits && llama_get_sampled_logits_count_ith(ctx.ctx.get(), i) == values.size());
            for (size_t id = 0; id < values.size(); ++id) {
                GGML_ASSERT((logits[id] != -INFINITY) == (values[id].logit != -INFINITY));
            }
        }
    }
    printf("llguidance prepared row masks, rollback and ubatch splitting test PASSED\n");
}

static void test_llguidance_common(const test_params & params) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto a = common_tokenize(vocab, "a", false, true);
    const auto z = common_tokenize(vocab, "z", false, true);
    GGML_ASSERT(a.size() == 1 && z.size() == 1);
    for (bool first : {false, true}) {
        auto sp = common_grammar_params("%llguidance {}\nstart: /[ab]+/");
        sp.grammar_first = first;
        sp.logit_bias = {{z[0], 1000.0f}, {a[0], 100.0f}};
        common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
        GGML_ASSERT(sp.backend_sampling);
        auto rp = sp;
        rp.backend_sampling = false;
        common_sampler_ptr reference(common_sampler_init(params.model.get(), rp));
        std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(sampler.get())}};
        std::vector<llama_sampler_seq_config> none;
        test_context ctx(params, configs);
        test_context cpu(params, none);
        GGML_ASSERT(ctx.decode({{0, "Hello"}}));
        GGML_ASSERT(cpu.decode({{0, "Hello"}}));
        for (int i = 0; i < 3; ++i) {
            const auto actual = common_sampler_sample(sampler.get(), ctx.ctx.get(), ctx.idx_for_seq(0));
            const auto expected = common_sampler_sample(reference.get(), cpu.ctx.get(), cpu.idx_for_seq(0));
            GGML_ASSERT(actual == expected && actual == a[0]);
            common_sampler_accept(sampler.get(), actual, true);
            common_sampler_accept(reference.get(), expected, true);
            GGML_ASSERT(ctx.decode_token(actual));
            GGML_ASSERT(cpu.decode_token(expected));
        }
        common_sampler_ptr clone(common_sampler_clone(sampler.get()));
        const auto actual = common_sampler_sample(sampler.get(), ctx.ctx.get(), ctx.idx_for_seq(0));
        const auto cloned = common_sampler_sample(clone.get(), ctx.ctx.get(), ctx.idx_for_seq(0));
        GGML_ASSERT(actual == cloned);
    }
    printf("llguidance common grammar-first and rejection-policy test PASSED\n");
}
#endif

static void test_grammar_common_lifecycle(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const auto tokens = grammar_test_tokens(vocab);
    const std::string grammar = "root ::= " + grammar_token(tokens[0]) + " " + grammar_token(tokens[1]) + " " + grammar_token(tokens[2]);
    auto sp = common_grammar_params(grammar);
    common_sampler_ptr source(common_sampler_init(params.model.get(), sp));
    common_sampler_ptr destination(common_sampler_init(params.model.get(), sp));
    llama_sampler_ptr reference(llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
    ggml_backend_ptr backend(ggml_backend_dev_init(params.device, nullptr));
    const auto buft = ggml_backend_get_default_buffer_type(backend.get());
    auto adapter = [](common_sampler * s) { return llama_sampler_chain_get(common_sampler_get(s), 0); };
    auto * dst_adapter = adapter(destination.get());
    GGML_ASSERT(dst_adapter->iface->backend_init(dst_adapter, buft, 1));
    common_sampler_ptr cloned;
    {
        grammar_test_graph graph(dst_adapter, backend.get(), n_vocab);
        graph.check(reference.get());
        common_sampler_accept(source.get(), tokens[2], false);
        common_sampler_accept(source.get(), tokens[0], true);
        llama_sampler_accept(reference.get(), tokens[0]);
        cloned.reset(common_sampler_clone(source.get()));
        common_sampler_copy(source.get(), destination.get());
        source.reset();
        graph.check(reference.get());
        common_sampler_accept(destination.get(), tokens[2], false);
        graph.check(reference.get());
        common_sampler_reset(destination.get());
        graph.check(reference.get());
        common_sampler_accept(destination.get(), tokens[1], true);
        llama_sampler_accept(reference.get(), tokens[1]);
        graph.check(reference.get());
    }
    {
        grammar_test_graph rebuilt(dst_adapter, backend.get(), n_vocab);
        rebuilt.check(reference.get());
        common_sampler_copy(cloned.get(), destination.get());
        llama_sampler_reset(reference.get());
        llama_sampler_accept(reference.get(), tokens[0]);
        rebuilt.check(reference.get());
    }
    destination.reset();
    auto * clone_adapter = adapter(cloned.get());
    GGML_ASSERT(clone_adapter->iface->backend_init(clone_adapter, buft, 1));
    {
        grammar_test_graph graph(clone_adapter, backend.get(), n_vocab);
        graph.check(reference.get());
        common_sampler_accept(cloned.get(), tokens[1], true);
        llama_sampler_accept(reference.get(), tokens[1]);
        graph.check(reference.get());
    }
    printf("common grammar clone/copy, prompt acceptance and rebuild test PASSED\n");
}

static void test_grammar_common_prompt(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    const std::string prompt = " Hello";
    const auto prefill = common_tokenize(vocab, prompt, false, true);
    GGML_ASSERT(!prefill.empty());
    std::string grammar = "root ::= ";
    for (auto token : prefill) {
        grammar += grammar_token(token) + " ";
    }
    grammar += grammar_token(tokens[0]);
    ggml_backend_ptr backend(ggml_backend_dev_init(params.device, nullptr));
    for (auto type : { COMMON_GRAMMAR_TYPE_USER, COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, COMMON_GRAMMAR_TYPE_TOOL_CALLS }) {
        for (bool lazy : { false, true }) {
            auto sp = common_grammar_params(grammar);
            sp.grammar.type = type;
            sp.generation_prompt = prompt;
            sp.grammar_lazy = lazy;
            sp.grammar_triggers = {{ COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, "", prefill[0] }};
            common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
            llama_sampler_ptr reference(lazy ?
                llama_sampler_init_grammar_lazy_patterns(vocab, grammar.c_str(), "root", nullptr, 0, prefill.data(), 1) :
                llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
            if (!lazy && type != COMMON_GRAMMAR_TYPE_USER) {
                for (auto token : prefill) {
                    llama_sampler_accept(reference.get(), token);
                }
            }
            auto * adapter = llama_sampler_chain_get(common_sampler_get(sampler.get()), 0);
            GGML_ASSERT(adapter->iface->backend_init(adapter, ggml_backend_get_default_buffer_type(backend.get()), 1));
            grammar_test_graph graph(adapter, backend.get(), llama_vocab_n_tokens(vocab));
            graph.check(reference.get());
            common_sampler_accept(sampler.get(), prefill[0], false);
            graph.check(reference.get());
            if (lazy) {
                common_sampler_accept(sampler.get(), prefill[0], true);
                llama_sampler_accept(reference.get(), prefill[0]);
                graph.check(reference.get());
            }
        }
    }
    printf("common grammar generation-prompt prefill and lazy trigger test PASSED\n");
}

static void test_grammar_common_policy(const test_params & params) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    const std::string grammar = "root ::= " + grammar_token(tokens[0]) + "+";
    auto sp = common_grammar_params(grammar);
    sp.grammar_first = false;
    sp.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    sp.top_k = 1;
    sp.logit_bias = {{tokens[2], 1000.0f}, {tokens[0], 100.0f}};
    common_sampler_ptr rejection(common_sampler_init(params.model.get(), sp));
    GGML_ASSERT(sp.backend_sampling);
    std::vector<llama_sampler_seq_config> no_configs;
    test_context cpu(params, no_configs);
    GGML_ASSERT(cpu.decode({{0, "Hello"}}));
    const auto raw = decode_raw_logits(params, "Hello");
    auto proposal = apply_cpu_sampler(raw, common_sampler_get(rejection.get()));
    GGML_ASSERT(proposal.size() == 1 && proposal[0].id == tokens[2]);
    GGML_ASSERT(common_sampler_sample(rejection.get(), cpu.ctx.get(), cpu.idx_for_seq(0)) == tokens[0]);
    GGML_ASSERT(common_sampler_sample(rejection.get(), cpu.ctx.get(), cpu.idx_for_seq(0), true) == tokens[0]);

    auto budget = common_grammar_params(grammar);
    budget.reasoning_budget_start = {tokens[1]};
    budget.reasoning_budget_end = {{tokens[2]}};
    budget.reasoning_budget_tokens = 1;
    common_sampler_ptr restricted(common_sampler_init(params.model.get(), budget));
    GGML_ASSERT(!budget.backend_sampling);

    if (grammar_test_supported(params)) {
        auto eligible = common_grammar_params(grammar);
        common_sampler_ptr sampler(common_sampler_init(params.model.get(), eligible));
        std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(sampler.get())}};
        test_context fallback(params, configs, 1, 0, 0, 2);
        GGML_ASSERT(fallback.decode({{0, "Hello"}}));
        GGML_ASSERT(llama_get_sampled_token_ith(fallback.ctx.get(), fallback.idx_for_seq(0)) == tokens[0]);
        GGML_ASSERT(common_sampler_sample(sampler.get(), fallback.ctx.get(), fallback.idx_for_seq(0)) == tokens[0]);
        common_sampler_ptr shared(common_sampler_init(params.model.get(), eligible));
        configs = {{0, common_sampler_get(shared.get())}};
        test_context shared_ctx(params, configs, 2);
        common_batch batch(shared_ctx.ctx.get());
        batch.add(tokens[0], 0, std::vector<llama_seq_id>{0, 1}, true);
        GGML_ASSERT(llama_process(shared_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == -1);
    }
    printf("common rejection policy, reasoning budget and output-contract fallback test PASSED\n");
}

static void check_chain_continuation(common_sampler * actual, common_sampler * expected) {
    llama_sampler_ptr a(llama_sampler_clone(common_sampler_get(actual)));
    llama_sampler_ptr b(llama_sampler_clone(common_sampler_get(expected)));
    for (int draw = 0; draw < 32; ++draw) {
        llama_token_data da[] = {{0, 0.0f, 0.0f}, {1, 0.0f, 0.0f}, {2, 0.0f, 0.0f}};
        llama_token_data db[] = {{0, 0.0f, 0.0f}, {1, 0.0f, 0.0f}, {2, 0.0f, 0.0f}};
        llama_token_data_array ca = { da, 3, -1, false };
        llama_token_data_array cb = { db, 3, -1, false };
        auto * sa = llama_sampler_chain_get(a.get(), llama_sampler_chain_n(a.get()) - 1);
        auto * sb = llama_sampler_chain_get(b.get(), llama_sampler_chain_n(b.get()) - 1);
        llama_sampler_apply(sa, &ca);
        llama_sampler_apply(sb, &cb);
        GGML_ASSERT(ca.selected == cb.selected);
    }
}

static void test_grammar_compatibility(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    const std::string grammar = "root ::= (" + grammar_token(tokens[0]) + " | " + grammar_token(tokens[1]) + ")+";
    for (bool suffix : {false, true}) {
        for (bool late_first : {false, true}) {
            for (int n_probs : {0, 4}) {
                auto sp = common_grammar_params(grammar);
                sp.grammar_first = false;
                sp.n_probs = n_probs;
                sp.temp = 1.0f;
                sp.top_k = 1;
                sp.penalty_repeat = 1.2f;
                sp.logit_bias = {{tokens[2], 1000.0f}, {tokens[0], 100.0f}, {tokens[1], 99.0f}};
                sp.samplers = {COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
                if (suffix) {
                    sp.samplers.insert(sp.samplers.begin(), COMMON_SAMPLER_TYPE_TOP_N_SIGMA);
                    sp.top_n_sigma = 1000.0f;
                }
                common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
                auto rp = sp;
                rp.backend_sampling = false;
                common_sampler_ptr reference(common_sampler_init(params.model.get(), rp));
                std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(sampler.get())}};
                std::vector<llama_sampler_seq_config> empty;
                test_context backend(params, configs);
                test_context cpu(params, empty);
                GGML_ASSERT(backend.decode({{0, "Hello"}}));
                GGML_ASSERT(cpu.decode({{0, "Hello"}}));
                for (int step = 0; step < 3; ++step) {
                    const auto idx = backend.idx_for_seq(0);
                    const auto proposal = llama_get_sampled_token_ith(backend.ctx.get(), idx);
                    GGML_ASSERT(proposal == (suffix ? LLAMA_TOKEN_NULL : tokens[2]));
                    if (!suffix && n_probs == 0) {
                        GGML_ASSERT(llama_get_sampled_logits_count_ith(backend.ctx.get(), idx) == 0);
                        GGML_ASSERT(llama_get_sampled_probs_count_ith(backend.ctx.get(), idx) == 0);
                        GGML_ASSERT(llama_get_sampled_candidates_count_ith(backend.ctx.get(), idx) == 0);
                    }
                    llama_sampler_ptr mask(llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
                    std::vector<float> filtered(llama_vocab_n_tokens(vocab));
                    GGML_ASSERT(llama_sampler_apply_grammar(mask.get(), backend.ctx.get(), idx, filtered.data()));
                    GGML_ASSERT(filtered[tokens[2]] == -INFINITY);
                    const auto * raw = llama_get_logits_ith_raw(backend.ctx.get(), idx);
                    GGML_ASSERT(std::isfinite(raw[tokens[2]]));
                    GGML_ASSERT(filtered[tokens[0]] == raw[tokens[0]]);
                    const auto actual = common_sampler_sample(sampler.get(), backend.ctx.get(), idx, late_first);
                    const auto expected = common_sampler_sample(reference.get(), cpu.ctx.get(), cpu.idx_for_seq(0), late_first);
                    GGML_ASSERT(actual == expected && actual != tokens[2]);
                    check_chain_continuation(sampler.get(), reference.get());
                    const auto * ca = common_sampler_get_candidates(sampler.get(), false);
                    const auto * cb = common_sampler_get_candidates(reference.get(), false);
                    GGML_ASSERT(ca->size == cb->size && ca->selected == cb->selected);
                    GGML_ASSERT(ca->data[ca->selected].id == cb->data[cb->selected].id);
                    if (suffix || late_first || n_probs > 0) {
                        GGML_ASSERT(std::fabs(ca->data[ca->selected].logit - cb->data[cb->selected].logit) < 1e-4f);
                    }
                    GGML_ASSERT(std::fabs(ca->data[ca->selected].p - cb->data[cb->selected].p) < 1e-6f);
                    common_sampler_accept(sampler.get(), actual, true);
                    common_sampler_accept(reference.get(), expected, true);
                    common_sampler_ptr cloned(common_sampler_clone(sampler.get()));
                    common_sampler_ptr copied(common_sampler_init(params.model.get(), sp));
                    common_sampler_copy(sampler.get(), copied.get());
                    check_chain_continuation(cloned.get(), copied.get());
                    GGML_ASSERT(backend.decode_token(actual));
                    GGML_ASSERT(cpu.decode_token(expected));
                }
            }
        }
    }
    printf("compatibility full-logit retry, late policy, RNG and state test PASSED\n");
}

struct sampling_sync_counter {
    ggml_backend_t backend;
    decltype(ggml_backend_i::synchronize) synchronize;
    decltype(ggml_backend_i::cpy_tensor_async) copy;
    size_t calls = 0;
    size_t retained_copies = 0;
    static sampling_sync_counter * active;

    sampling_sync_counter(ggml_backend_t backend) : backend(backend), synchronize(backend->iface.synchronize),
        copy(backend->iface.cpy_tensor_async) {
        GGML_ASSERT(!active);
        active = this;
        backend->iface.synchronize = [](ggml_backend_t backend) {
            ++active->calls;
            if (active->synchronize) {
                active->synchronize(backend);
            }
        };
        if (copy) {
            backend->iface.cpy_tensor_async = [](ggml_backend_t src_backend, ggml_backend_t dst_backend,
                                                const ggml_tensor * src, ggml_tensor * dst) {
                if (std::strcmp(dst->name, "logits_retained") == 0) {
                    GGML_ASSERT(src_backend == dst_backend);
                    ++active->retained_copies;
                }
                return active->copy(src_backend, dst_backend, src, dst);
            };
        }
    }

    ~sampling_sync_counter() {
        backend->iface.cpy_tensor_async = copy;
        backend->iface.synchronize = synchronize;
        active = nullptr;
    }
};

sampling_sync_counter * sampling_sync_counter::active = nullptr;

struct retry_graph_counter {
    ggml_backend_t backend;
    decltype(ggml_backend_i::graph_compute) compute;
    ggml_cgraph * retry = nullptr;
    size_t calls = 0;
    static retry_graph_counter * active;

    retry_graph_counter(ggml_backend_t backend) : backend(backend), compute(backend->iface.graph_compute) {
        GGML_ASSERT(!active);
        active = this;
        backend->iface.graph_compute = [](ggml_backend_t backend, ggml_cgraph * graph) {
            bool mask = false;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                mask = mask || ggml_graph_node(graph, i)->op == GGML_OP_TOKEN_MASK;
            }
            if (mask) {
                auto * result = ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1);
                GGML_ASSERT(result->type == GGML_TYPE_I32 && ggml_nelements(result) == 1);
                GGML_ASSERT(!active->retry || active->retry == graph);
                active->retry = graph;
                ++active->calls;
            }
            return active->compute(backend, graph);
        };
    }

    ~retry_graph_counter() {
        backend->iface.graph_compute = compute;
        active = nullptr;
    }
};

retry_graph_counter * retry_graph_counter::active = nullptr;

static void test_grammar_proposal_retry(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    for (int mode : {0, 1, 2}) {
        const std::string grammar = "root ::= (" + grammar_token(tokens[0]) + " | " + grammar_token(tokens[1]) + ")+";
        for (bool rejected : {false, true}) {
            auto sp = common_grammar_params(grammar);
            sp.grammar_first = false;
            sp.temp = mode == 1 ? 0.0f : mode == 2 ? 0.7f : 1.0f;
            sp.top_k = mode == 2 ? 4 : 1;
            sp.samplers = {COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
            if (mode == 2) {
                sp.samplers = {COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P,
                               COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE};
                sp.top_p = 0.9f;
                sp.min_p = 0.05f;
            }
            sp.penalty_repeat = 1.2f;
            sp.logit_bias = {{tokens[0], 100.0f}, {tokens[2], rejected ? 1000.0f : -1000.0f}};
            if (mode == 2) {
                sp.logit_bias.push_back({tokens[1], 99.0f});
            }
            common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
            auto rp = sp;
            rp.backend_sampling = false;
            common_sampler_ptr reference(common_sampler_init(params.model.get(), rp));
            std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(sampler.get())}};
            std::vector<llama_sampler_seq_config> empty;
            test_context backend(params, configs);
            test_context cpu(params, empty);
            retry_graph_counter counter(ggml_backend_sched_get_backend(backend.ctx->get_sched(), 0));
            GGML_ASSERT(backend.decode({{0, "Hello"}}));
            GGML_ASSERT(cpu.decode({{0, "Hello"}}));
            for (int step = 0; step < 6; ++step) {
                const size_t before = counter.calls;
                const auto idx = backend.idx_for_seq(0);
                const auto proposal = llama_get_sampled_token_ith(backend.ctx.get(), idx);
                GGML_ASSERT(rejected ? proposal == tokens[2] : proposal == tokens[0] || proposal == tokens[1]);
                const auto actual = common_sampler_sample(sampler.get(), backend.ctx.get(), idx);
                const auto expected = common_sampler_sample(reference.get(), cpu.ctx.get(), cpu.idx_for_seq(0));
                GGML_ASSERT(actual == expected && (actual == tokens[0] || actual == tokens[1]));
                GGML_ASSERT(counter.calls == before + (rejected ? 1 : 0));
                check_chain_continuation(sampler.get(), reference.get());
                const auto * raw = llama_get_logits_ith_raw(backend.ctx.get(), idx);
                GGML_ASSERT(raw && std::isfinite(raw[tokens[2]]));
                common_sampler_accept(sampler.get(), actual, true);
                common_sampler_accept(reference.get(), expected, true);
                check_chain_continuation(sampler.get(), reference.get());
                if (step == 2) {
                    common_sampler_reset(sampler.get());
                    common_sampler_reset(reference.get());
                }
                const size_t after = counter.calls;
                GGML_ASSERT(backend.decode_token(actual));
                GGML_ASSERT(cpu.decode_token(expected));
                GGML_ASSERT(counter.calls == after);
            }
            GGML_ASSERT(counter.calls == (rejected ? 6 : 0));
        }
    }
    const std::string impossible = "root ::= " + grammar_token(llama_vocab_n_tokens(vocab));
    llama_sampler_ptr mask(llama_sampler_init_grammar(vocab, impossible.c_str(), "root"));
    GGML_ASSERT(mask);
    const llama_sampler_row row = {0, LLAMA_SAMPLER_PREFIX_COMMITTED, nullptr, 0};
    GGML_ASSERT(llama_sampler_grammar_prepare_rows(mask.get(), &row, 1, false));
    GGML_ASSERT(mask->iface->backend_row_usable(mask.get(), 0));
    GGML_ASSERT(llama_sampler_grammar_prepare_rows(mask.get(), &row, 1, true));
    GGML_ASSERT(!mask->iface->backend_row_usable(mask.get(), 0));
    for (int reporting : {0, 4}) {
        for (bool suffix : {false, true}) {
            for (bool late_first : {false, true}) {
                auto sp = common_grammar_params(impossible);
                sp.grammar_first = false;
                sp.n_probs = reporting;
                if (suffix) {
                    sp.samplers.insert(sp.samplers.begin(), COMMON_SAMPLER_TYPE_TOP_N_SIGMA);
                    sp.top_n_sigma = 1.0f;
                }
                common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
                std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(sampler.get())}};
                test_context backend(params, configs);
                GGML_ASSERT(backend.decode({{0, "Hello"}}));
                GGML_ASSERT(common_sampler_sample(sampler.get(), backend.ctx.get(), backend.idx_for_seq(0), late_first) == LLAMA_TOKEN_NULL);
            }
        }
    }

    printf("lazy proposal, cached device retry, reset and distinct RNG draws test PASSED\n");
}

static void test_sampling_output_transfers(const test_params & params) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    for (uint32_t flags : {uint32_t(LLAMA_SAMPLER_OUTPUT_TOKEN), uint32_t(LLAMA_SAMPLER_OUTPUT_CANDIDATES),
                           uint32_t(LLAMA_SAMPLER_OUTPUT_RAW), uint32_t(LLAMA_SAMPLER_OUTPUT_CANDIDATES | LLAMA_SAMPLER_OUTPUT_RAW)}) {
        auto cp = llama_sampler_chain_default_params();
        cp.output_flags = flags;
        llama_sampler_ptr chain(llama_sampler_chain_init(cp));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(1234));
        std::vector<llama_sampler_seq_config> configs = {{0, chain.get()}};
        test_context ctx(params, configs);
        sampling_sync_counter counter(ggml_backend_sched_get_backend(ctx.ctx->get_sched(), 0));
        GGML_ASSERT(ctx.decode({{0, "Hello"}}));
        if (counter.copy && (flags & LLAMA_SAMPLER_OUTPUT_RAW)) {
            GGML_ASSERT(counter.retained_copies == 1);
        }
        counter.calls = 0;
        const auto idx = ctx.idx_for_seq(0);
        GGML_ASSERT(llama_get_sampled_token_ith(ctx.ctx.get(), idx) != LLAMA_TOKEN_NULL);
        const auto count = flags & LLAMA_SAMPLER_OUTPUT_CANDIDATES ? llama_vocab_n_tokens(vocab) : 0;
        GGML_ASSERT(llama_get_sampled_logits_count_ith(ctx.ctx.get(), idx) == uint32_t(count));
        GGML_ASSERT(llama_get_sampled_probs_count_ith(ctx.ctx.get(), idx) == uint32_t(count));
        GGML_ASSERT(llama_get_sampled_candidates_count_ith(ctx.ctx.get(), idx) == 0);
        if (flags & LLAMA_SAMPLER_OUTPUT_RAW) {
            const auto raw = decode_raw_logits(params, "Hello");
            const auto * actual = llama_get_logits_ith_raw(ctx.ctx.get(), idx);
            for (size_t i = 0; i < raw.size(); ++i) {
                GGML_ASSERT(std::fabs(actual[i] - raw[i]) < 1e-5f);
            }
        }
        GGML_ASSERT(counter.calls == 1);
        GGML_ASSERT(ctx.decode_token(llama_vocab_bos(vocab)));
        const size_t before_read = counter.calls;
        GGML_ASSERT(llama_get_sampled_token_ith(ctx.ctx.get(), 0) != LLAMA_TOKEN_NULL);
        GGML_ASSERT(counter.calls == before_read + 1);
        llama_get_sampled_logits_count_ith(ctx.ctx.get(), 0);
        llama_get_sampled_probs_count_ith(ctx.ctx.get(), 0);
        GGML_ASSERT(counter.calls == before_read + 1);
    }
    for (bool need_candidates : {false, true}) {
        common_params_sampling sp;
        sp.backend_sampling = true;
        sp.backend_sampling_candidates = need_candidates;
        sp.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
        sp.temp = 1.0f;
        common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
        std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(sampler.get())}};
        test_context ctx(params, configs);
        GGML_ASSERT(ctx.decode({{0, "Hello"}}));
        sampling_sync_counter counter(ggml_backend_sched_get_backend(ctx.ctx->get_sched(), 0));
        llama_get_sampled_token_ith(ctx.ctx.get(), ctx.idx_for_seq(0));
        common_sampler_sample(sampler.get(), ctx.ctx.get(), ctx.idx_for_seq(0));
        GGML_ASSERT(counter.calls == 1);
        common_sampler_sample(sampler.get(), ctx.ctx.get(), ctx.idx_for_seq(0));
        GGML_ASSERT(counter.calls == 1);
        GGML_ASSERT(ctx.decode_token(llama_vocab_bos(vocab)));
        const size_t before_read = counter.calls;
        common_sampler_sample(sampler.get(), ctx.ctx.get(), 0);
        GGML_ASSERT(counter.calls == before_read + 1);
        common_sampler_sample(sampler.get(), ctx.ctx.get(), 0);
        GGML_ASSERT(counter.calls == before_read + 1);
        const auto * candidates = common_sampler_get_candidates(sampler.get(), false);
        GGML_ASSERT(candidates->size == (need_candidates ? size_t(llama_vocab_n_tokens(vocab)) : 1));
        GGML_ASSERT(candidates->selected >= 0);
    }
    {
        common_params_sampling sp;
        sp.backend_sampling = true;
        sp.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
        common_sampler_ptr a(common_sampler_init(params.model.get(), sp));
        common_sampler_ptr b(common_sampler_init(params.model.get(), sp));
        std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(a.get())}, {1, common_sampler_get(b.get())}};
        test_context ctx(params, configs, 2);
        GGML_ASSERT(ctx.decode({{1, "Hello"}, {0, "Hello"}}));
        sampling_sync_counter counter(ggml_backend_sched_get_backend(ctx.ctx->get_sched(), 0));
        for (int repeat = 0; repeat < 2; ++repeat) {
            GGML_ASSERT(common_sampler_sample(b.get(), ctx.ctx.get(), ctx.idx_for_seq(1)) != LLAMA_TOKEN_NULL);
            GGML_ASSERT(common_sampler_sample(a.get(), ctx.ctx.get(), ctx.idx_for_seq(0)) != LLAMA_TOKEN_NULL);
            GGML_ASSERT(counter.calls == 1);
        }
    }
    const auto raw = decode_raw_logits(params, "Hello");
    auto cp = llama_sampler_chain_default_params();
    cp.output_flags = LLAMA_SAMPLER_OUTPUT_RAW;
    llama_sampler_ptr a(llama_sampler_chain_init(cp));
    llama_sampler_ptr b(llama_sampler_chain_init(cp));
    llama_sampler_chain_add(a.get(), llama_sampler_init_dist(1234));
    llama_sampler_chain_add(b.get(), llama_sampler_init_dist(5678));
    std::vector<llama_sampler_seq_config> configs = {{0, a.get()}, {1, b.get()}};
    test_context ctx(params, configs, 2, 0, 2);
    GGML_ASSERT(ctx.decode({{0, "Hello"}, {1, "Hello"}}));
    GGML_ASSERT(llama_set_sampler(ctx.ctx.get(), 1, nullptr));
    GGML_ASSERT(llama_graph_reserve(ctx.ctx.get(), 4, 2, 2));
    for (int seq : {1, 0}) {
        const auto * actual = llama_get_logits_ith_raw(ctx.ctx.get(), ctx.idx_for_seq(seq));
        for (size_t i = 0; i < raw.size(); ++i) {
            GGML_ASSERT(std::fabs(actual[i] - raw[i]) < 1e-5f);
        }
    }
    printf("selected-token transfer suppression and retained raw output test PASSED\n");
}

static void test_grammar_prepared_rows(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    const std::string grammar = "root ::= " + grammar_token(tokens[0]) + " " + grammar_token(tokens[1]);
    {
        llama_sampler_ptr sampler(llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
        ggml_backend_ptr backend(ggml_backend_dev_init(params.device, nullptr));
        GGML_ASSERT(backend);
        GGML_ASSERT(sampler->iface->backend_init(sampler.get(), ggml_backend_get_default_buffer_type(backend.get()), 2));
        llama_token prefix = tokens[0];
        const llama_sampler_row rows[] = {
            {7, LLAMA_SAMPLER_PREFIX_LINEAR, &prefix, 1},
            {3, LLAMA_SAMPLER_PREFIX_COMMITTED, nullptr, 0},
        };
        GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), rows, 2));
        for (int index : {1, 0, 1}) {
            GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), nullptr, 0));
            GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), &rows[index], 1));
            llama_sampler_ptr reference(llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
            if (index == 0) {
                llama_sampler_accept(reference.get(), prefix);
            }
            grammar_test_graph graph(sampler.get(), backend.get(), llama_vocab_n_tokens(vocab));
            graph.check(reference.get());
        }
        prefix = tokens[1];
        GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), rows, 1));
        GGML_ASSERT(!sampler->iface->backend_row_usable(sampler.get(), 0));
        prefix = tokens[0];
        GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), rows, 1));
        GGML_ASSERT(sampler->iface->backend_row_usable(sampler.get(), 0));
        llama_sampler_accept(sampler.get(), prefix);
        GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), rows, 1));
        GGML_ASSERT(!sampler->iface->backend_row_usable(sampler.get(), 0));
        llama_sampler_reset(sampler.get());
        GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), rows, 1));
        GGML_ASSERT(sampler->iface->backend_row_usable(sampler.get(), 0));
    }
    for (bool lazy : {false, true}) {
        const std::string repeated = "root ::= (" + grammar_token(tokens[0]) + " " + grammar_token(tokens[1]) + ")+";
        llama_sampler_ptr sampler(lazy ?
                llama_sampler_init_grammar_lazy_patterns(vocab, repeated.c_str(), "root", nullptr, 0, &tokens[0], 1) :
                llama_sampler_init_grammar(vocab, repeated.c_str(), "root"));
        ggml_backend_ptr backend(ggml_backend_dev_init(params.device, nullptr));
        GGML_ASSERT(backend);
        GGML_ASSERT(sampler->iface->backend_init(sampler.get(), ggml_backend_get_default_buffer_type(backend.get()), 16));
        const std::vector<llama_tokens> prefixes = {
            {}, {tokens[0]}, {tokens[0], tokens[1]}, {tokens[0], tokens[1], tokens[0]},
            {tokens[0], tokens[1], tokens[0], tokens[0]},
            {tokens[0], tokens[1], tokens[0], tokens[0], tokens[1]},
            {tokens[0], tokens[1], tokens[0], tokens[1]},
            {tokens[0]}, {tokens[1]}, {}, {-1}, {-1, tokens[0]},
            {llama_vocab_n_tokens(vocab)}, {llama_vocab_eos(vocab)},
        };
        std::vector<llama_sampler_row> rows;
        for (size_t i = 0; i < prefixes.size(); ++i) {
            rows.push_back({int32_t(i), LLAMA_SAMPLER_PREFIX_LINEAR, prefixes[i].data(), prefixes[i].size()});
        }
        GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), rows.data(), rows.size()));
        for (size_t i = 0; i < rows.size(); ++i) {
            llama_sampler_ptr reference(llama_sampler_clone(sampler.get()));
            bool usable = true;
            for (auto token : prefixes[i]) {
                if (token < 0 || token >= llama_vocab_n_tokens(vocab) || llama_vocab_is_eog(vocab, token)) {
                    usable = false;
                    break;
                }
                llama_token_data value = {token, 0.0f, 0.0f};
                llama_token_data_array cur = {&value, 1, -1, false};
                llama_sampler_apply(reference.get(), &cur);
                if (value.logit == -INFINITY) {
                    usable = false;
                    break;
                }
                llama_sampler_accept(reference.get(), token);
            }
            GGML_ASSERT(sampler->iface->backend_prepare(sampler.get(), &rows[i], 1));
            GGML_ASSERT(sampler->iface->backend_row_usable(sampler.get(), 0) == usable);
            if (usable) {
                grammar_test_graph graph(sampler.get(), backend.get(), llama_vocab_n_tokens(vocab));
                graph.check(reference.get());
            }
        }
    }
    for (bool lazy : {false, true}) {
        std::vector<llama_sampler_ptr> chains;
        std::vector<llama_sampler *> grammars;
        std::vector<llama_sampler_seq_config> configs;
        const std::string active_grammar = lazy ? "root ::= " + grammar_token(tokens[2]) + " " + grammar_token(tokens[0]) + " " + grammar_token(tokens[1]) : grammar;
        for (int seq = 0; seq < 2; ++seq) {
            llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            auto * grmr = lazy ? llama_sampler_init_grammar_lazy_patterns(vocab, active_grammar.c_str(), "root", nullptr, 0, &tokens[2], 1) :
                                llama_sampler_init_grammar(vocab, active_grammar.c_str(), "root");
            grammars.push_back(grmr);
            llama_sampler_chain_add(chain.get(), grmr);
            llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(1234));
            configs.push_back({seq, chain.get()});
            chains.push_back(std::move(chain));
        }
        test_context ctx(params, configs, 2, 4, 2, 2);
        llama_sampler_row bad = {-1, LLAMA_SAMPLER_PREFIX_LINEAR, nullptr, 0};
        GGML_ASSERT(!llama_sampler_prepare(ctx.ctx.get(), 0, &bad, 1));
        bad = {0, LLAMA_SAMPLER_PREFIX_LINEAR, nullptr, 1};
        GGML_ASSERT(!llama_sampler_prepare(ctx.ctx.get(), 0, &bad, 1));
        bad = {0, LLAMA_SAMPLER_PREFIX_LINEAR, tokens.data(), size_t(-1)};
        GGML_ASSERT(!llama_sampler_prepare(ctx.ctx.get(), 0, &bad, 1));
        const llama_sampler_row duplicates[] = {{1, LLAMA_SAMPLER_PREFIX_LINEAR, nullptr, 0}, {1, LLAMA_SAMPLER_PREFIX_LINEAR, nullptr, 0}};
        GGML_ASSERT(!llama_sampler_prepare(ctx.ctx.get(), 0, duplicates, 2));
        std::vector<llama_tokens> prefixes = lazy ?
            std::vector<llama_tokens>{{tokens[2]}, {}, {tokens[2], tokens[0]}, {tokens[2], tokens[1]}} :
            std::vector<llama_tokens>{{tokens[0]}, {}, {}, {tokens[1]}};
        for (int round = 0; round < 2; ++round) {
            common_batch batch(ctx.ctx.get());
            batch.add(llama_vocab_bos(vocab), round*2, 1, true);
            batch.add(llama_vocab_bos(vocab), round*2, 0, true);
            batch.add(tokens[0], round*2 + 1, 1, true);
            batch.add(tokens[0], round*2 + 1, 0, true);
            for (int seq = 0; seq < 2; ++seq) {
                std::vector<llama_sampler_row> rows;
                for (int idx : seq == 0 ? std::vector<int>{3, 1} : std::vector<int>{2, 0}) {
                    rows.push_back({idx, seq == 0 ? LLAMA_SAMPLER_PREFIX_TEACHER_FORCED : LLAMA_SAMPLER_PREFIX_TREE,
                            prefixes[idx].data(), prefixes[idx].size()});
                }
                GGML_ASSERT(llama_sampler_prepare(ctx.ctx.get(), seq, rows.data(), rows.size()));
            }
            GGML_ASSERT(llama_process(ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
            for (int idx = 0; idx < 4; ++idx) {
                const int seq = idx % 2 == 0 ? 1 : 0;
                GGML_ASSERT(llama_sampler_row_usable(ctx.ctx.get(), idx) == (idx != 3));
                if (idx == 3) {
                    GGML_ASSERT(llama_get_sampled_token_ith(ctx.ctx.get(), idx) == LLAMA_TOKEN_NULL);
                    GGML_ASSERT(llama_get_sampled_logits_count_ith(ctx.ctx.get(), idx) == 0);
                    GGML_ASSERT(llama_sampler_sample(chains[seq].get(), ctx.ctx.get(), idx) == LLAMA_TOKEN_NULL);
                    continue;
                }
                llama_sampler_ptr reference(llama_sampler_clone(grammars[seq]));
                for (auto token : prefixes[idx]) {
                    llama_sampler_accept(reference.get(), token);
                }
                std::vector<llama_token_data> values(llama_vocab_n_tokens(vocab));
                for (size_t token = 0; token < values.size(); ++token) {
                    values[token] = {llama_token(token), 0.0f, 0.0f};
                }
                llama_token_data_array cur = {values.data(), values.size(), -1, false};
                llama_sampler_apply(reference.get(), &cur);
                const float * actual = llama_get_sampled_logits_ith(ctx.ctx.get(), idx);
                GGML_ASSERT(actual && llama_get_sampled_logits_count_ith(ctx.ctx.get(), idx) == values.size());
                for (size_t token = 0; token < values.size(); ++token) {
                    GGML_ASSERT((actual[token] != -INFINITY) == (values[token].logit != -INFINITY));
                }
            }
            for (auto * grmr : grammars) {
                llama_token_data value = {tokens[0], 0.0f, 0.0f};
                llama_token_data_array cur = {&value, 1, -1, false};
                llama_sampler_apply(grmr, &cur);
                GGML_ASSERT(value.logit == 0.0f);
            }
            GGML_ASSERT(llama_process(ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == -1);
            std::swap(prefixes[0], prefixes[2]);
        }
        for (const auto & chain : chains) {
            llama_sampler_ptr actual(llama_sampler_clone(llama_sampler_chain_get(chain.get(), 1)));
            llama_sampler_ptr expected(llama_sampler_init_dist(1234));
            for (int draw = 0; draw < 32; ++draw) {
                llama_token_data a[] = {{0, 0.0f, 0.0f}, {1, 0.0f, 0.0f}};
                llama_token_data b[] = {{0, 0.0f, 0.0f}, {1, 0.0f, 0.0f}};
                llama_token_data_array ca = {a, 2, -1, false};
                llama_token_data_array cb = {b, 2, -1, false};
                llama_sampler_apply(actual.get(), &ca);
                llama_sampler_apply(expected.get(), &cb);
                GGML_ASSERT(ca.selected == cb.selected);
            }
        }
    }
    printf("prepared row identity, teacher/tree masks, trigger rollback and ubatch mapping test PASSED\n");
}

static void test_grammar_speculative(const test_params & params) {
    if (!grammar_test_supported(params)) {
        return;
    }
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const auto tokens = grammar_test_tokens(vocab);
    const auto eog = llama_vocab_eos(vocab);
    GGML_ASSERT(llama_vocab_is_eog(vocab, eog));
    for (bool first : {true, false}) {
        for (bool late : {false, true}) {
            if (first && late) {
                continue;
            }
            for (int scenario = 0; scenario < 6; ++scenario) {
                if (scenario == 5 && (first || late)) {
                    continue;
                }
                for (bool rejection : {false, true}) {
                    const bool stop = scenario == 3 || scenario == 4;
                    const std::string grammar = "root ::= " + grammar_token(tokens[0]) + (stop ? "" :
                            " " + grammar_token(tokens[1]) + " " + grammar_token(tokens[0]));
                    auto sp = common_grammar_params(grammar);
                    sp.grammar_first = first;
                    sp.backend_sampling_candidates = rejection;
                    sp.temp = 1.0f;
                    sp.top_k = 1;
                    sp.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
                    sp.logit_bias = {{tokens[2], 1000.0f}, {tokens[0], 100.0f}, {tokens[1], 99.0f}};
                    if (scenario == 5) {
                        sp.logit_bias = {{tokens[0], 1000.0f}, {tokens[1], 99.0f}, {tokens[2], -1000.0f}};
                    }
                    common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
                    auto rp = sp;
                    rp.backend_sampling = false;
                    common_sampler_ptr reference(common_sampler_init(params.model.get(), rp));
                    std::vector<llama_sampler_seq_config> configs = {{0, common_sampler_get(sampler.get())}};
                    std::vector<llama_sampler_seq_config> empty;
                    test_context ctx(params, configs, 1, 4, 2, 4);
                    test_context cpu(params, empty, 1, 4, 2, 4);
                    llama_tokens draft = {tokens[0], tokens[1]};
                    if (scenario == 1) { draft[1] = tokens[2]; }
                    if (scenario == 2) { draft[0] = tokens[2]; }
                    if (scenario == 3) { draft = {tokens[0], eog}; }
                    if (scenario == 4) { draft = {tokens[0], eog, tokens[1]}; }
                    std::vector<int> idxs;
                    common_batch batch(ctx.ctx.get());
                    common_batch batch_cpu(cpu.ctx.get());
                    for (size_t i = 0; i <= draft.size(); ++i) {
                        const auto input = i == 0 ? llama_vocab_bos(vocab) : draft[i - 1];
                        idxs.push_back(i);
                        batch.add(input, i, 0, true);
                        batch_cpu.add(input, i, 0, true);
                    }
                    GGML_ASSERT(common_sampler_prepare(ctx.ctx.get(), 0, idxs, draft, 0, batch.size()));
                    GGML_ASSERT(llama_process(ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
                    GGML_ASSERT(llama_process(cpu.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_cpu.get()) == 0);
                    if (scenario == 1 || stop) {
                        GGML_ASSERT(!llama_sampler_row_usable(ctx.ctx.get(), 2));
                    }
                    if (scenario == 2) {
                        GGML_ASSERT(!llama_sampler_row_usable(ctx.ctx.get(), 1));
                    }
                    auto verify = [&](common_sampler * target, llama_context * context) {
                        if (!rejection) {
                            return common_sampler_sample_and_accept_n(target, context, idxs, draft, late);
                        }
                        std::vector<std::vector<llama_token_data>> q;
                        for (auto token : draft) {
                            q.push_back({{token, 0.0f, 1.0f}});
                        }
                        return common_sampler_sample_and_accept_n_rejection(target, context, idxs, draft, q, late);
                    };
                    const auto actual = verify(sampler.get(), ctx.ctx.get());
                    const auto expected = verify(reference.get(), cpu.ctx.get());
                    GGML_ASSERT(actual == expected);
                    GGML_ASSERT(actual.size() == size_t(scenario == 0 || scenario == 5 ? 3 : scenario == 2 ? 1 : 2));
                    check_chain_continuation(sampler.get(), reference.get());
                    ctx.seq_positions[0] = batch.size();
                    cpu.seq_positions[0] = batch.size();
                    GGML_ASSERT(ctx.decode_token(actual.back()));
                    GGML_ASSERT(cpu.decode_token(expected.back()));
                    if (!stop) {
                        const auto next = common_sampler_sample(sampler.get(), ctx.ctx.get(), -1, late);
                        const auto next_cpu = common_sampler_sample(reference.get(), cpu.ctx.get(), -1, late);
                        GGML_ASSERT(next == next_cpu);
                        common_sampler_accept(sampler.get(), next, true);
                        common_sampler_accept(reference.get(), next_cpu, true);
                        check_chain_continuation(sampler.get(), reference.get());
                    }
                }
            }
        }
    }
    printf("speculative acceptance, mismatch, EOG, retry/late-policy RNG and continuation test PASSED\n");
}


struct backend_test_case {
    std::string name;
    void (*fn)(const test_params &);
    bool enabled_by_default;
};

// note: test names are "test_<suffix>" and match the function implementing them
static const backend_test_case BACKEND_TESTS[] = {
    // single sampler
    { "test_greedy",                 test_greedy,                 true },
    { "test_greedy_filtered",        test_greedy_filtered,        true },
    { "test_greedy_filtered_common", test_greedy_filtered_common, true },
    { "test_temp",                   test_temp,                   true },
    { "test_temp_ext",               test_temp_ext,               true },
    { "test_top_k",                  test_top_k,                  true },
    { "test_top_p",                  test_top_p,                  true },
    { "test_min_p",                  test_min_p,                  true },
    { "test_logit_bias",             test_logit_bias,             true },
    { "test_penalties",              test_penalties,              true },
    { "test_grammar",                test_grammar,                true },
#ifdef LLAMA_USE_LLGUIDANCE
    { "test_llguidance_backend", test_llguidance_backend, true },
    { "test_llguidance_common", test_llguidance_common, true },
    { "test_llguidance_prepared_rows", test_llguidance_prepared_rows, true },
#endif
    { "test_grammar_resident_cache", test_grammar_resident_cache, true },
    { "test_grammar_async_upload", test_grammar_async_upload, true },
    { "test_grammar_lifecycle",      test_grammar_lifecycle,      true },
    { "test_grammar_multi_sequence", test_grammar_multi_sequence, true },
    { "test_grammar_prepared_rows", test_grammar_prepared_rows, true },
    { "test_grammar_speculative", test_grammar_speculative, true },
    { "test_grammar_fallback",       test_grammar_fallback,       true },
    { "test_grammar_compatibility", test_grammar_compatibility, true },
    { "test_sampling_output_transfers", test_sampling_output_transfers, true },
    { "test_grammar_proposal_retry", test_grammar_proposal_retry, true },
    { "test_grammar_common",           test_grammar_common,           true },
    { "test_grammar_common_lifecycle", test_grammar_common_lifecycle, true },
    { "test_grammar_common_prompt",    test_grammar_common_prompt,    true },
    { "test_grammar_common_policy",    test_grammar_common_policy,    true },
    // multiple sequences
    { "test_multi_sequence",         test_multi_sequence,         true },
    { "test_dist",                   test_dist,                   true },
    { "test_dist_and_cpu",           test_dist_and_cpu,           true },
    { "test_mixed",                  test_mixed,                  true },
    { "test_cpu_mixed",              test_cpu_mixed,              true },
    { "test_set_sampler",            test_set_sampler,            true },
    // multiple outputs per sequence
    { "test_multi_output_limit",                  test_multi_output_limit,               true },
    { "test_multi_output_multi_sequence_dist",    test_multi_output_multi_sequence_dist, true },
    { "test_multi_output_dist_transaction",       test_multi_output_dist_transaction,    true },
    { "test_multi_output_sampling_chain",         test_multi_output_sampling_chain,      true },
    { "test_multi_output_cpu_suffix",             test_multi_output_cpu_suffix,          true },
};

// TODO: add usage, examples
static test_args parse_cli(int argc, char ** argv) {
    test_args out;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];

        if (std::strcmp(arg, "--test") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--test expects a value\n");
                exit(EXIT_FAILURE);
            }
            out.test = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--test=", 7) == 0) {
            out.test = arg + 7;
            continue;
        }
        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--model expects a value\n");
                exit(EXIT_FAILURE);
            }
            out.model = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--model=", 8) == 0) {
            out.model = arg + 8;
            continue;
        }
        if (std::strcmp(arg, "--device") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--device expects a value (cpu or gpu)\n");
                exit(EXIT_FAILURE);
            }
            out.device = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--device=", 9) == 0) {
            out.device = arg + 9;
            continue;
        }
        if (out.model.empty()) {
            out.model = arg;
            continue;
        }

        fprintf(stderr, "Unexpected argument: %s\n", arg);
        exit(EXIT_FAILURE);
    }

    if (out.device != "cpu" && out.device != "gpu" && out.device != "auto") {
        fprintf(stderr, "Invalid device '%s'. Must be 'cpu', 'gpu' or 'auto'\n", out.device.c_str());
        exit(EXIT_FAILURE);
    }

    return out;
}

static std::vector<const backend_test_case *> collect_tests_to_run(const std::string & requested) {
    std::vector<const backend_test_case *> selected;

    if (!requested.empty()) {
        for (const auto & test : BACKEND_TESTS) {
            if (test.name == requested) {
                selected.push_back(&test);
                break;
            }
        }
        if (selected.empty()) {
            fprintf(stderr, "Unknown test '%s'. Available tests:\n", requested.c_str());
            for (const auto & test : BACKEND_TESTS) {
                fprintf(stderr, "  %s\n", test.name.c_str());
            }
            exit(EXIT_FAILURE);
        }
    } else {
        for (const auto & test : BACKEND_TESTS) {
            if (!test.enabled_by_default) {
                continue;
            }
#ifdef GGML_USE_HIP
            // TODO: remove this when https://github.com/ggml-org/llama.cpp/pull/26592 is merged
            if (test.name == "test_penalties"                   ||
                test.name == "test_set_sampler"                 ||
                test.name == "test_mixed"                       ||
                test.name == "test_top_p"                       ||
                test.name == "test_multi_output_sampling_chain" ||
                test.name == "test_multi_output_cpu_suffix") {
                fprintf(stderr, "Skipping test '%s' on HIP backend (no backend TOP_K support)\n", test.name.c_str());
                continue;
            }
#endif // GGML_USE_HIP
            selected.push_back(&test);
        }
    }

    if (selected.empty()) {
        fprintf(stderr, "No backend sampling tests selected. Use --test=<name> to pick one.\n");
    }

    return selected;
}

static void run_tests(const std::vector<const backend_test_case *> & tests, const test_params & args) {
    for (const auto & test : tests) {
        fprintf(stderr, "\n=== %s ===\n", test->name.c_str());
        try {
            test->fn(args);
        } catch (const std::exception & e) {
            fprintf(stderr, "Error running test '%s': %s\n", test->name.c_str(), e.what());
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char ** argv) {
    test_args args = parse_cli(argc, argv);

    if (args.model.empty()) {
        args.model = common_get_model_or_exit(1, argv);
    }

    {
        std::ifstream file(args.model);
        if (!file.is_open()) {
            fprintf(stderr, "no model '%s' found\n", args.model.c_str());
            return EXIT_FAILURE;
        }
    }

    fprintf(stderr, "using '%s'\n", args.model.c_str());

    llama_backend_init();

    test_params params = {
        /*.model =*/ load_model(args),
    };
    if (!params.model) {
        return EXIT_FAILURE;
    }
    params.device = args.device == "cpu" ? nullptr : sampling_test_gpu();
    if (!params.device) {
        params.device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }

    const std::vector<const backend_test_case *> tests = collect_tests_to_run(args.test);
    if (!tests.empty()) {
        run_tests(tests, params);
    }

    return 0;
}
