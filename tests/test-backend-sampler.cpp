#include "ggml.h"
#include "llama.h"
#include "get-model.h"
#include "common.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdlib>
#include <cstring>
#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct test_model_context {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    std::unordered_map<llama_seq_id, int32_t> seq_positions;
    std::unordered_map<llama_seq_id, int32_t> last_batch_info;

    bool setup_model(const char * model_path) {
        if (model != nullptr) {
            return true;
        }

        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        model = llama_model_load_from_file(model_path, mparams);
        if (model == nullptr) {
            fprintf(stderr, "Warning: failed to load model '%s', skipping test\n", model_path);
            cleanup();
            return false;
        }
        vocab = llama_model_get_vocab(model);

        return true;
    }

    bool setup(const char * model_path, std::vector<llama_sampler_seq_config> & configs) {
        if (model == nullptr) {
            setup_model(model_path);
        }

        if (model != nullptr && ctx != nullptr) {
            return true;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        cparams.n_batch = 512;
        cparams.samplers = configs.data();
        cparams.n_samplers = configs.size();

        int32_t max_seq_id = 0;
        for (const auto & config : configs) {
            if (config.seq_id > max_seq_id) {
                max_seq_id = config.seq_id;
            }
        }
        cparams.n_seq_max = max_seq_id + 1;

        ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "Warning: failed to create context, skipping test\n");
            cleanup();
            return false;
        }
        llama_set_warmup(ctx, false);

        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        fprintf(stderr, "Vocabulary size: %d\n", n_vocab);

        return true;
    }

    bool decode(const std::map<llama_seq_id, std::string> & prompts) {
        if (ctx == nullptr || vocab == nullptr) {
            fprintf(stderr, "Error: context not initialized, call setup() first\n");
            return false;
        }

        last_batch_info.clear();
        llama_batch batch = llama_batch_init(512, 0, prompts.size());

        int n_tokens_per_prompt = 0;

        for (const auto & [seq_id, prompt] : prompts) {
            std::vector<llama_token> tokens;
            tokens.push_back(llama_vocab_bos(vocab));

            std::vector<llama_token> prompt_tokens(32);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                           prompt_tokens.data(), prompt_tokens.size(),
                                           false, false);
            //TODO: refactor this function to just handle a single prompt at a time
            //      to avoid this check and complexity.
            if (n_tokens_per_prompt == 0) {
                n_tokens_per_prompt = n_tokens;
            } else {
                if (n_tokens != n_tokens_per_prompt) {
                    fprintf(stderr, "Error: prompts must have the same number of tokens\n");
                    llama_batch_free(batch);
                    return false;
                }
                n_tokens_per_prompt = n_tokens;
            }
            if (n_tokens < 0) {
                fprintf(stderr, "Warning: tokenization failed for seq_id %d\n", seq_id);
                llama_batch_free(batch);
                return false;
            }

            for (int i = 0; i < n_tokens; i++) {
                tokens.push_back(prompt_tokens[i]);
            }

            for (size_t i = 0; i < tokens.size(); i++) {
                common_batch_add(batch, tokens[i], i, { seq_id }, i == tokens.size() - 1);
            }

            seq_positions[seq_id] = tokens.size();
        }


        printf("Batch contents:\n");
        printf("  n_tokens: %d\n", batch.n_tokens);
        for (int i = 0; i < batch.n_tokens; i++) {
            printf("  token[%d]: tok=%-5d, pos=%d, n_seq_id=%d, seq_ids=[", i, batch.token[i], batch.pos[i], batch.n_seq_id[i]);

        for (int j = 0; j < batch.n_seq_id[i]; j++) {
            printf("%d%s", batch.seq_id[i][j], j < batch.n_seq_id[i]-1 ? ", " : "");
        }
        printf("], logits=%d\n", batch.logits[i]);
}

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed\n");
            llama_batch_free(batch);
            return false;
        }

        // Build mapping from seq id to batch token idx
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id seq_id = batch.seq_id[i][0];
                last_batch_info[seq_id] = i;
                printf("seq %d : batch idx %d\n", seq_id, i);
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

    bool decode_token(llama_token token, llama_seq_id seq_id = 0) {
        if (ctx == nullptr) {
            fprintf(stderr, "Error: context not initialized, call setup() first\n");
            return false;
        }

        llama_batch batch = llama_batch_init(1, 0, 1);
        int32_t pos = seq_positions[seq_id];
        common_batch_add(batch, token, pos, { seq_id }, true);

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for token %d in seq %d\n", token, seq_id);
            llama_batch_free(batch);
            return false;
        }

        last_batch_info.clear();
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id cur_seq = batch.seq_id[i][0];
                last_batch_info[cur_seq] = i;
            }
        }

        seq_positions[seq_id]++;
        llama_batch_free(batch);
        return true;
    }

    bool decode_tokens(const std::map<llama_seq_id, llama_token> & seq_tokens) {
        if (ctx == nullptr) {
            fprintf(stderr, "Error: context not initialized, call setup() first\n");
            return false;
        }

        llama_batch batch = llama_batch_init(seq_tokens.size(), 0, seq_tokens.size());

        for (const auto & [seq_id, token] : seq_tokens) {
            int32_t pos = seq_positions[seq_id];
            common_batch_add(batch, token, pos, { seq_id }, true);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for batch tokens\n");
            llama_batch_free(batch);
            return false;
        }

        for (const auto & [seq_id, _] : seq_tokens) {
            seq_positions[seq_id]++;
        }

        last_batch_info.clear();
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id cur_seq = batch.seq_id[i][0];
                last_batch_info[cur_seq] = i;
            }
        }

        llama_batch_free(batch);
        return true;
    }

    std::string token_to_piece(llama_token token, bool special) {
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        }
        else {
            piece.resize(n_chars);
        }

        return piece;
    }

    void cleanup() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
        ctx = nullptr;
        model = nullptr;
        vocab = nullptr;
    }

    ~test_model_context() {
        cleanup();
    }
};

static void test_backend_greedy_sampling(const char * model_path) {
    test_model_context test_ctx;

    const int seq_id = 0;

    struct llama_sampler_chain_params backend_sampler_params = llama_sampler_chain_default_params();
    struct llama_sampler * backend_sampler_chain = llama_sampler_chain_init(backend_sampler_params);

    llama_sampler_chain_add(backend_sampler_chain, llama_sampler_backend_init_greedy());
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain }};

    if (!test_ctx.setup(model_path, backend_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        return;
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    llama_token token = llama_get_backend_sampled_token_ith(test_ctx.ctx, batch_idx);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    token = llama_get_backend_sampled_token_ith(test_ctx.ctx, -1);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_get_backend_sampled_token_ith(test_ctx.ctx, loop_idx);
        printf("Generation step %d: token id:%d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        test_ctx.decode_token(token, 0);
    }
}

static void test_backend_top_k_sampling(const char * model_path) {
    test_model_context test_ctx;

    const int seq_id = 0;
    const int32_t k = 8;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * backend_sampler_chain = llama_sampler_chain_init(backend_chain_params);
    llama_sampler_chain_add(backend_sampler_chain, llama_sampler_backend_init_top_k(k));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain }};

    if (!test_ctx.setup(model_path, backend_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        return;
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_backend_sampled_logits_ith(test_ctx.ctx, batch_idx);
    uint32_t n_logits = llama_get_backend_sampled_logits_count_ith(test_ctx.ctx, batch_idx);
    for (size_t i = 0; i < n_logits; ++i) {
        printf("top_k logit[%zu] = %.6f\n", i, logits[i]);
    }

    // Sample using CPU sampler for verification that it is possible to do hybrid
    // sampling, first top_k on the backend and then dist on the CPU.
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(chain_params);
    GGML_ASSERT(chain->iface->apply_ggml != nullptr);

    llama_sampler_chain_add(chain, llama_sampler_init_dist(18));
    llama_token token = llama_sampler_sample(chain, test_ctx.ctx, batch_idx);
    const std::string token_str = test_ctx.token_to_piece(token, false);
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    printf("backend top-k hybrid sampling test PASSED\n");

    llama_sampler_free(chain);
}

static void test_backend_temp_sampling(const char * model_path) {
    test_model_context test_ctx;

    const float temp_0 = 0.8f;
    struct llama_sampler_chain_params backend_chain_params_0 = llama_sampler_chain_default_params();
    struct llama_sampler * backend_sampler_chain_0 = llama_sampler_chain_init(backend_chain_params_0);
    llama_sampler_chain_add(backend_sampler_chain_0, llama_sampler_backend_init_temp(temp_0));

    const float temp_1 = 0.1f;
    struct llama_sampler_chain_params backend_chain_params_1 = llama_sampler_chain_default_params();
    struct llama_sampler * backend_sampler_chain_1 = llama_sampler_chain_init(backend_chain_params_1);
    llama_sampler_chain_add(backend_sampler_chain_1, llama_sampler_backend_init_temp(temp_1));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, backend_sampler_chain_0 },
        { 1, backend_sampler_chain_1 }
    };

    if (!test_ctx.setup(model_path, backend_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{0, "Some where over"}, {1, "Once upon a"}})) {
        return;
    }

    int32_t batch_idx_0 = test_ctx.idx_for_seq(0);
    int32_t batch_idx_1 = test_ctx.idx_for_seq(1);

    // Sample from sequence 0 using CPU sampler
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    struct llama_sampler * chain_0 = llama_sampler_chain_init(chain_params_0);
    llama_sampler_chain_add(chain_0, llama_sampler_init_dist(18));

    llama_token token_0 = llama_sampler_sample(chain_0, test_ctx.ctx, batch_idx_0);
    const std::string token_0_str = test_ctx.token_to_piece(token_0, false);
    printf("Sequence 0 sampled token id:%d, string: '%s'\n", token_0, token_0_str.c_str());
    GGML_ASSERT(token_0 >= 0 && token_0 < test_ctx.n_vocab);

    // Sample from sequence 1 using CPU sampler
    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    struct llama_sampler * chain_1 = llama_sampler_chain_init(chain_params_1);
    llama_sampler_chain_add(chain_1, llama_sampler_init_dist(18));

    llama_token token_1 = llama_sampler_sample(chain_1, test_ctx.ctx, batch_idx_1);
    const std::string token_1_str = test_ctx.token_to_piece(token_1, false);
    printf("Sequence 1 sampled token id:%d, string: '%s'\n", token_1, token_1_str.c_str());
    GGML_ASSERT(token_1 >= 0 && token_1 < test_ctx.n_vocab);

    printf("backend temp sampling test PASSED\n");

    llama_sampler_free(chain_0);
    llama_sampler_free(chain_1);
}

static void test_backend_multi_sequence_sampling(const char * model_path) {
    test_model_context test_ctx;

    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    struct llama_sampler * sampler_chain_0 = llama_sampler_chain_init(chain_params_0);
    llama_sampler_chain_add(sampler_chain_0, llama_sampler_backend_init_greedy());

    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    struct llama_sampler * sampler_chain_1 = llama_sampler_chain_init(chain_params_1);
    llama_sampler_chain_add(sampler_chain_1, llama_sampler_backend_init_temp(0.8f));
    llama_sampler_chain_add(sampler_chain_1, llama_sampler_backend_init_greedy());

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0 },
        { 1, sampler_chain_1 }
    };

    if (!test_ctx.setup(model_path, backend_sampler_configs)) {
        return;
    }

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"},
        {1, "Some"}
    };

    if (!test_ctx.decode(prompts)) {
        return;
    }

    int32_t batch_idx_0 = test_ctx.idx_for_seq(0);
    llama_token seq0_token = llama_get_backend_sampled_token_ith(test_ctx.ctx, batch_idx_0);
    const std::string seq0_token_str = test_ctx.token_to_piece(seq0_token, false);
    printf("Seq 0 sampled token id=%d, string='%s'\n", seq0_token, seq0_token_str.c_str());
    GGML_ASSERT(seq0_token >= 0 && seq0_token < test_ctx.n_vocab);

    int32_t batch_idx_1 = test_ctx.idx_for_seq(1);
    llama_token seq1_token = llama_get_backend_sampled_token_ith(test_ctx.ctx, batch_idx_1);
    const std::string seq1_token_str = test_ctx.token_to_piece(seq1_token, false);
    printf("Seq 1 sampled token id=%d, string='%s'\n", seq1_token, seq1_token_str.c_str());
    GGML_ASSERT(seq1_token >= 0 && seq1_token < test_ctx.n_vocab);

    // Generate tokens for each sequence
    printf("\nMulti-sequence generation:\n");
    for (int step = 0; step < 4; step++) {
        std::map<llama_seq_id, llama_token> tokens;

        for (llama_seq_id seq_id : {0, 1}) {
            int32_t idx = test_ctx.idx_for_seq(seq_id);
            llama_token token = llama_get_backend_sampled_token_ith(test_ctx.ctx, idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("  Seq %d, step %d: token id=%d, string='%s'\n", seq_id, step, token, token_str.c_str());
            tokens[seq_id] = token;
        }

        // Decode all tokens in a single batch
        if (!test_ctx.decode_tokens(tokens)) {
            break;
        }
    }

    printf("backend multi-sequence sampling test PASSED\n");
}

static void test_backend_dist_sampling(const char * model_path) {
    test_model_context test_ctx;

    const int32_t seed = 88;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * backend_sampler_chain = llama_sampler_chain_init(backend_chain_params);
    llama_sampler_chain_add(backend_sampler_chain, llama_sampler_backend_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ 0, backend_sampler_chain }};

    if (!test_ctx.setup(model_path, backend_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{0, "Hello"}})) {
        return;
    }

    llama_token token = llama_get_backend_sampled_token_ith(test_ctx.ctx, test_ctx.idx_for_seq(0));
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    token = llama_get_backend_sampled_token_ith(test_ctx.ctx, -1);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
}

static void test_backend_dist_sampling_and_cpu(const char * model_path) {
    test_model_context test_ctx;

    const int seq_id = 0;
    const int32_t seed = 88;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * backend_sampler_chain = llama_sampler_chain_init(backend_chain_params);
    llama_sampler_chain_add(backend_sampler_chain, llama_sampler_backend_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain }};

    if (!test_ctx.setup(model_path, backend_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        return;
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    // Sample using CPU sampler
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(chain, llama_sampler_init_dist(18));

    llama_token backend_token = llama_get_backend_sampled_token_ith(test_ctx.ctx, batch_idx);
    llama_token cpu_token = llama_sampler_sample(chain, test_ctx.ctx, batch_idx);
    GGML_ASSERT(backend_token == cpu_token);
}

static void test_backend_logit_bias_sampling(const char * model_path) {
    test_model_context test_ctx;

    // Calling setup_model to ensure vocab is loaded and can be accessed
    if (!test_ctx.setup_model(model_path)) {
        return;
    }

    const int seq_id = 0;

    // Create the logit biases vector.
    std::vector<llama_logit_bias> logit_bias;

    // Get the token for the piece "World".
    const std::string piece = "World";
    std::vector<llama_token> tokens(16);
    llama_tokenize(test_ctx.vocab, piece.c_str(), piece.size(), tokens.data(), tokens.size(), false, false);
    llama_token bias_token = tokens[0];
    logit_bias.push_back({ bias_token, +100.0f });
    printf("biasing token piece '%s' -> token id %d\n", piece.c_str(), bias_token);

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * backend_sampler_chain = llama_sampler_chain_init(backend_chain_params);
    llama_sampler_chain_add(backend_sampler_chain, llama_sampler_backend_init_logit_bias(
                llama_vocab_n_tokens(test_ctx.vocab),
                logit_bias.size(),
                logit_bias.data()));
    llama_sampler_chain_add(backend_sampler_chain, llama_sampler_backend_init_dist(88));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { seq_id, backend_sampler_chain },
    };

    if (!test_ctx.setup(model_path, backend_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        return;
    }

    llama_token backend_token = llama_get_backend_sampled_token_ith(test_ctx.ctx, test_ctx.idx_for_seq(seq_id));
    const std::string backend_token_str = test_ctx.token_to_piece(backend_token, false);
    printf("logit bias sampled token = %d, string='%s'\n", backend_token, backend_token_str.c_str());
    GGML_ASSERT(backend_token == bias_token);
}

static void test_backend_set_sampler(const char * model_path) {
    test_model_context test_ctx;

    const int32_t seed = 88;
    const int seq_id = 0;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * backend_sampler_chain = llama_sampler_chain_init(backend_chain_params);
    llama_sampler_chain_add(backend_sampler_chain, llama_sampler_backend_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain }};

    if (!test_ctx.setup(model_path, backend_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        return;
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    // Sample using backend sampler configured above
    llama_token backend_token = llama_get_backend_sampled_token_ith(test_ctx.ctx, batch_idx);
    const std::string backend_token_str = test_ctx.token_to_piece(backend_token, false);
    printf("dist sampled token = %d, string='%s'\n", backend_token, backend_token_str.c_str());

    // Now clear the backend sampler for this sequence.
    llama_set_backend_sampler(test_ctx.ctx, seq_id, nullptr);
    printf("Cleared backend sampler for seq_id %d\n", seq_id);

    // Sample using CPU sampler
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(chain, llama_sampler_init_dist(18));

    std::map<llama_seq_id, llama_token> tokens = { { seq_id, backend_token}, };
    if (!test_ctx.decode_tokens(tokens)) {
        return;
    }

    // Should not have any sampled token or probs after clearing the backend sampler.
    const int32_t idx = test_ctx.idx_for_seq(seq_id);
    GGML_ASSERT(llama_get_backend_sampled_token_ith(test_ctx.ctx, idx) == LLAMA_TOKEN_NULL);
    GGML_ASSERT(llama_get_backend_sampled_probs_ith(test_ctx.ctx, idx) == nullptr);

    // Sample the token using the CPU sampler chain.
    llama_token token2 = llama_sampler_sample(chain, test_ctx.ctx, seq_id);
    const std::string token2_str = test_ctx.token_to_piece(token2, false);
    printf("CPU sampled token after clearing backend sampler: id=%d, string='%s'\n", token2, token2_str.c_str());
    std::map<llama_seq_id, llama_token> tokens2 = { { seq_id, token2}, };

    // Set a new backend sampler for the sequence.
    struct llama_sampler_chain_params new_backend_chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * new_backend_sampler_chain = llama_sampler_chain_init(new_backend_chain_params);
    llama_sampler_chain_add(new_backend_sampler_chain, llama_sampler_backend_init_top_k(20));
    llama_sampler_chain_add(new_backend_sampler_chain, llama_sampler_backend_init_dist(seed));
    llama_set_backend_sampler(test_ctx.ctx, seq_id, new_backend_sampler_chain);

    if (!test_ctx.decode_tokens(tokens2)) {
        return;
    }

    llama_token new_backend_token = llama_get_backend_sampled_token_ith(test_ctx.ctx, test_ctx.idx_for_seq(seq_id));
    const std::string new_backend_token_str = test_ctx.token_to_piece(new_backend_token, false);
    printf("dist sampled token = %d, string='%s'\n", new_backend_token, new_backend_token_str.c_str());
}

struct backend_test_case {
    const char * name;
    void (*fn)(const char *);
    bool enabled_by_default;
};

static const backend_test_case BACKEND_TESTS[] = {
    { "greedy",          test_backend_greedy_sampling,         true  },
    { "logit_bias",      test_backend_logit_bias_sampling,     true  },
    { "temp",            test_backend_temp_sampling,           true  },
    { "top_k",           test_backend_top_k_sampling,          true  },
    { "multi_sequence",  test_backend_multi_sequence_sampling, true  },
    { "dist",            test_backend_dist_sampling,           true  },
    { "dist_and_cpu",    test_backend_dist_sampling_and_cpu,   true  },
    { "set_sampler",     test_backend_set_sampler,             true  },
};

struct backend_cli_args {
    const char * model = nullptr;
    const char * test = nullptr;
};

static backend_cli_args parse_backend_cli(int argc, char ** argv) {
    backend_cli_args out;

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
        if (!out.model) {
            out.model = arg;
            continue;
        }

        fprintf(stderr, "Unexpected argument: %s\n", arg);
        exit(EXIT_FAILURE);
    }

    return out;
}

static std::vector<const backend_test_case *> collect_tests_to_run(const char * requested) {
    std::vector<const backend_test_case *> selected;

    if (requested != nullptr) {
        for (const auto & test : BACKEND_TESTS) {
            if (std::strcmp(test.name, requested) == 0) {
                selected.push_back(&test);
                break;
            }
        }
        if (selected.empty()) {
            fprintf(stderr, "Unknown test '%s'. Available tests:\n", requested);
            for (const auto & test : BACKEND_TESTS) {
                fprintf(stderr, "  %s\n", test.name);
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

static void run_tests(const std::vector<const backend_test_case *> & tests, const char * model_path) {
    for (const auto * test : tests) {
        fprintf(stderr, "\n=== %s ===\n", test->name);
        test->fn(model_path);
    }
}


int main(int argc, char *argv[] ) {
    const backend_cli_args args = parse_backend_cli(argc, argv);

    std::array<char *, 2> model_argv { argv[0], const_cast<char *>(args.model) };
    const int model_argc = args.model ? 2 : 1;
    char * model_path = get_model_or_exit(model_argc, model_argv.data());

    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    ggml_time_init();

    const std::vector<const backend_test_case *> tests = collect_tests_to_run(args.test);
    if (!tests.empty()) {
        run_tests(tests, model_path);
    }

    return 0;
}
