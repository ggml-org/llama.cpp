#include "ggml.h"
#include "llama.h"
#include "llama-cpp.h"
#include "get-model.h"
#include "common.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct test_args {
    std::string model;
    std::string test;
    std::string device = "auto";
};

struct test_params {
    llama_model_ptr model;
};

static llama_model_ptr load_model(const test_args & args) {
    auto mparams = llama_model_default_params();

    ggml_backend_dev_t devs[2] = { nullptr, nullptr };

    if (args.device != "auto") {
        if (args.device == "gpu") {
            devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);

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

    test_context(const test_params & params, std::vector<llama_sampler_seq_config> & configs, int32_t n_seq_max = -1) {
        auto * model = params.model.get();

        GGML_ASSERT(model);
        GGML_ASSERT(!ctx);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        cparams.n_batch = 512;
        cparams.samplers = configs.data();
        cparams.n_samplers = configs.size();

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

        llama_set_warmup(ctx.get(), false);

        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
    }

    bool decode(const std::map<llama_seq_id, std::string> & prompts) {
        GGML_ASSERT(ctx);

        last_batch_info.clear();
        llama_batch batch = llama_batch_init(512, 0, prompts.size());

        for (const auto & [seq_id, prompt] : prompts) {
            std::vector<llama_token> tokens;
            tokens.push_back(llama_vocab_bos(vocab));

            std::vector<llama_token> prompt_tokens(32);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                           prompt_tokens.data(), prompt_tokens.size(),
                                           false, false);
            if (n_tokens < 0) {
                fprintf(stderr, "Warning: tokenization failed for seq_id %d\n", seq_id);
                llama_batch_free(batch);
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
                common_batch_add(batch, tokens[i], start_pos + i, { seq_id }, i == tokens.size() - 1);
            }

            seq_positions[seq_id] = start_pos + tokens.size();
        }


        printf("Batch contents:\n");
        printf("n_tokens: %d\n", batch.n_tokens);
        for (int i = 0; i < batch.n_tokens; i++) {
            printf("token[%d]: tok=%-5d, pos=%d, n_seq_id=%d, seq_ids=[", i, batch.token[i], batch.pos[i], batch.n_seq_id[i]);

            for (int j = 0; j < batch.n_seq_id[i]; j++) {
                printf("%d%s", batch.seq_id[i][j], j < batch.n_seq_id[i]-1 ? ", " : "");
            }
            printf("], logits=%d\n", batch.logits[i]);
        }

        if (llama_decode(ctx.get(), batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed\n");
            llama_batch_free(batch);
            return false;
        }

        // Build mapping from seq id to batch token idx
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id seq_id = batch.seq_id[i][0];
                last_batch_info[seq_id] = i;
            }
        }

        llama_batch_free(batch);
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

    void update_batch_info(const llama_batch & batch) {
        last_batch_info.clear();
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id cur_seq = batch.seq_id[i][0];
                last_batch_info[cur_seq] = i;
            }
        }
    }

    bool decode_token(llama_token token, llama_seq_id seq_id = 0) {
        GGML_ASSERT(ctx);

        llama_batch batch = llama_batch_init(1, 0, 1);
        int32_t pos = seq_positions[seq_id];
        common_batch_add(batch, token, pos, { seq_id }, true);

        if (llama_decode(ctx.get(), batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for token %d in seq %d\n", token, seq_id);
            llama_batch_free(batch);
            return false;
        }

        update_batch_info(batch);

        seq_positions[seq_id]++;
        llama_batch_free(batch);

        return true;
    }

    bool decode_tokens(const std::map<llama_seq_id, llama_token> & seq_tokens) {
        GGML_ASSERT(ctx);

        llama_batch batch = llama_batch_init(seq_tokens.size(), 0, seq_tokens.size());

        for (const auto & [seq_id, token] : seq_tokens) {
            int32_t pos = seq_positions[seq_id];
            common_batch_add(batch, token, pos, { seq_id }, true);
        }

        if (llama_decode(ctx.get(), batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for batch tokens\n");
            llama_batch_free(batch);
            return false;
        }

        for (const auto & [seq_id, _] : seq_tokens) {
            seq_positions[seq_id]++;
        }

        update_batch_info(batch);

        llama_batch_free(batch);

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

static void test_backend_greedy_sampling(const test_params & params) {
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

static void test_backend_top_k_sampling(const test_params & params) {
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

static void test_backend_temp_sampling(const test_params & params) {
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

static void test_backend_temp_ext_sampling(const test_params & params) {
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

static void test_backend_min_p_sampling(const test_params & params) {
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

static void test_backend_top_p_sampling(const test_params & params) {
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

static void test_backend_multi_sequence_sampling(const test_params & params) {
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

static void test_backend_dist_sampling(const test_params & params) {
    const int seq_id = 189;
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

static void test_backend_dist_sampling_and_cpu(const test_params & params) {
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

static void test_backend_logit_bias_sampling(const test_params & params) {
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

// This test verifies that it is possible to have two different backend samplers,
// one that uses the backend dist sampler, and another that uses CPU dist sampler.
static void test_backend_mixed_sampling(const test_params & params) {
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

static void test_backend_set_sampler(const test_params & params) {
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

static void test_backend_cpu_mixed_batch(const test_params & params) {
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

static void test_backend_max_outputs(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    llama_batch batch = llama_batch_init(512, 0, 1);
    std::string prompt = "Hello";

    std::vector<llama_token> tokens;
    tokens.push_back(llama_vocab_bos(test_ctx.vocab));

    std::vector<llama_token> prompt_tokens(32);
    int n_tokens = llama_tokenize(test_ctx.vocab, prompt.c_str(), prompt.length(),
                                   prompt_tokens.data(), prompt_tokens.size(),
                                   false, false);
    for (int i = 0; i < n_tokens; i++) {
        tokens.push_back(prompt_tokens[i]);
    }

    for (size_t i = 0; i < tokens.size(); i++) {
        // set all tokens as output to trigger error
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }

    printf(">>> test_max_outputs expected error start:\n");
    const int ret = llama_decode(test_ctx.ctx.get(), batch);
    GGML_ASSERT(ret != 0 && "llama_decode should not succeed multiple outputs per sequence");
    printf("<<< test_max_outputs expected error end.\n");
    llama_batch_free(batch);

    printf("backend max outputs test PASSED\n");
}

struct backend_test_case {
    std::string name;
    void (*fn)(const test_params &);
    bool enabled_by_default;
};

static const backend_test_case BACKEND_TESTS[] = {
    { "greedy",          test_backend_greedy_sampling,         true  },
    { "logit_bias",      test_backend_logit_bias_sampling,     true  },
    { "temp",            test_backend_temp_sampling,           true  },
    { "temp_ext",        test_backend_temp_ext_sampling,       true  },
    { "top_k",           test_backend_top_k_sampling,          true  },
    { "multi_sequence",  test_backend_multi_sequence_sampling, true  },
    { "dist",            test_backend_dist_sampling,           true  },
    { "dist_and_cpu",    test_backend_dist_sampling_and_cpu,   true  },
    { "set_sampler",     test_backend_set_sampler,             true  },
    { "max_outputs",     test_backend_max_outputs,             true  },
    { "mixed",           test_backend_mixed_sampling,          true  },
    { "min_p",           test_backend_min_p_sampling,          true  },
    { "cpu_mixed",       test_backend_cpu_mixed_batch,         true  },
    { "top_p",           test_backend_top_p_sampling,          true  },
};

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
            if (test.enabled_by_default) {
                selected.push_back(&test);
            }
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
        args.model = get_model_or_exit(1, argv);
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

    const std::vector<const backend_test_case *> tests = collect_tests_to_run(args.test);
    if (!tests.empty()) {
        run_tests(tests, params);
    }

    return 0;
}
