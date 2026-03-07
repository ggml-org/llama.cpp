#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Dedicated sequence for learning entries. Seq 0 is used for conversation.
// Note: this is demo-only. If integrating into parallel serving, choose a
// higher seq_id to avoid collision with client sequences (1..N).
static const llama_seq_id LEARNING_SEQ_ID     = 1;
static const llama_seq_id CONVERSATION_SEQ_ID = 0;

struct learning_entry {
    std::string              text;
    std::vector<llama_token> tokens;
    llama_pos                pos_start;
    llama_pos                pos_end;   // exclusive
};

static void print_usage(int /* argc */, char ** argv) {
    fprintf(stderr, "\nexample usage:\n\n");
    fprintf(stderr, "  %s -m model.gguf --inject \"The user's name is Alice.\" -p \"What is my name?\" -n 64\n", argv[0]);
    fprintf(stderr, "  %s -m model.gguf --inject \"Fact 1.\" --inject \"Fact 2.\" --save /tmp/cache.bin\n", argv[0]);
    fprintf(stderr, "  %s -m model.gguf --load /tmp/cache.bin -p \"What do you know?\" -n 128\n", argv[0]);
    fprintf(stderr, "  %s -m model.gguf --inject \"Knowledge.\" -p \"Question?\" -n 64 --compare\n", argv[0]);
    fprintf(stderr, "\ncustom flags:\n");
    fprintf(stderr, "  --inject TEXT   knowledge text to encode (repeatable)\n");
    fprintf(stderr, "  --save PATH     save learning cache to file\n");
    fprintf(stderr, "  --load PATH     load learning cache from file\n");
    fprintf(stderr, "  --compare       generate with and without learning cache for comparison\n");
    fprintf(stderr, "\n");
}

// Decode tokens in chunks of n_batch, returning false on error.
static bool decode_tokens(
        llama_context * ctx,
        llama_batch   & batch,
        const std::vector<llama_token> & tokens,
        llama_pos       pos_start,
        llama_seq_id    seq_id,
        int32_t         n_batch,
        bool            logits_last) {

    const int32_t n_tokens = (int32_t) tokens.size();

    for (int32_t i = 0; i < n_tokens; i += n_batch) {
        const int32_t n_chunk = std::min(n_batch, n_tokens - i);

        common_batch_clear(batch);
        for (int32_t j = 0; j < n_chunk; j++) {
            bool want_logits = logits_last && (i + j == n_tokens - 1);
            common_batch_add(batch, tokens[i + j], pos_start + i + j, { seq_id }, want_logits);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed at pos %d\n", __func__, pos_start + i);
            return false;
        }
    }

    return true;
}

// Generate tokens and print them. Returns the generated text.
static std::string generate(
        llama_context         * ctx,
        llama_batch           & batch,
        common_sampler        * smpl,
        const llama_vocab     * vocab,
        llama_pos               n_past,
        int32_t                 n_predict) {

    std::string result;

    for (int32_t i = 0; i < n_predict; i++) {
        llama_token new_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token, true);

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        std::string piece = common_token_to_piece(ctx, new_token);
        printf("%s", piece.c_str());
        fflush(stdout);
        result += piece;

        common_batch_clear(batch);
        common_batch_add(batch, new_token, n_past, { CONVERSATION_SEQ_ID }, true);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed at pos %d\n", __func__, (int) n_past);
            break;
        }

        n_past++;
    }

    printf("\n");
    return result;
}

// Print a summary of the learning cache entries to stderr.
static void print_cache_summary(const std::vector<learning_entry> & entries, llama_pos learning_offset) {
    if (entries.empty()) {
        return;
    }

    fprintf(stderr, "\nLearning cache: %d tokens across %d entries\n",
            (int) learning_offset, (int) entries.size());

    for (size_t i = 0; i < entries.size(); i++) {
        std::string display = entries[i].text;
        if (display.size() > 60) {
            display = display.substr(0, 57) + "...";
        }
        fprintf(stderr, "  [pos %4d-%4d] \"%s\"\n",
                (int) entries[i].pos_start, (int) entries[i].pos_end - 1, display.c_str());
    }

    fprintf(stderr, "Conversation starts at position %d\n\n", (int) learning_offset);
}

int main(int argc, char ** argv) {
    // -------------------------------------------------------------------------
    // Pre-pass: extract custom flags before common_params_parse
    // -------------------------------------------------------------------------

    std::vector<std::string> inject_texts;
    std::string save_path;
    std::string load_path;
    bool        compare_mode = false;

    std::vector<char *> filtered_argv;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--inject") == 0 && i + 1 < argc) {
            inject_texts.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            load_path = argv[++i];
        } else if (strcmp(argv[i], "--compare") == 0) {
            compare_mode = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    int filtered_argc = (int) filtered_argv.size();

    common_params params;

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    // We need at least 2 sequences: one for learning (seq 1) and one for conversation (seq 0)
    if (params.n_parallel < 2) {
        params.n_parallel = 2;
    }

    // Use unified KV cache so seq_cp is a cheap metadata operation (bitset flip)
    // rather than a full buffer copy between separate streams
    params.kv_unified = true;

    common_init();

    if (inject_texts.empty() && load_path.empty() && params.prompt.empty()) {
        fprintf(stderr, "%s: error: provide at least --inject, --load, or -p\n", __func__);
        print_usage(argc, argv);
        return 1;
    }

    // -------------------------------------------------------------------------
    // Init backend + load model/context
    // -------------------------------------------------------------------------

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_context * ctx   = llama_init->context();
    llama_model   * model = llama_init->model();

    if (ctx == nullptr) {
        LOG_ERR("%s: error: unable to create context\n", __func__);
        return 1;
    }

    llama_memory_t        mem   = llama_get_memory(ctx);
    const llama_vocab   * vocab = llama_model_get_vocab(model);
    const int32_t         n_ctx = (int32_t) llama_n_ctx(ctx);
    const int32_t       n_batch = (int32_t) llama_n_batch(ctx);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // State tracking
    llama_pos                    learning_offset = 0;
    std::vector<llama_token>     all_learning_tokens;
    std::vector<learning_entry>  entries;

    // -------------------------------------------------------------------------
    // Load phase
    // -------------------------------------------------------------------------

    if (!load_path.empty()) {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<llama_token> loaded_tokens(n_ctx);
        size_t n_loaded = 0;

        size_t ret = llama_state_seq_load_file(ctx, load_path.c_str(), LEARNING_SEQ_ID,
                loaded_tokens.data(), loaded_tokens.size(), &n_loaded);

        if (ret == 0) {
            LOG_ERR("%s: error: failed to load learning cache from '%s'\n", __func__, load_path.c_str());
            llama_batch_free(batch);
            llama_backend_free();
            return 1;
        }

        learning_offset = (llama_pos) n_loaded;
        all_learning_tokens.assign(loaded_tokens.begin(), loaded_tokens.begin() + n_loaded);

        // We don't have the original text for loaded entries, so create a placeholder
        learning_entry entry;
        entry.text      = "(loaded from file)";
        entry.tokens.assign(loaded_tokens.begin(), loaded_tokens.begin() + n_loaded);
        entry.pos_start = 0;
        entry.pos_end   = learning_offset;
        entries.push_back(std::move(entry));

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        fprintf(stderr, "%s: loaded %zu tokens from '%s' (%.1f ms)\n",
                __func__, n_loaded, load_path.c_str(), ms);
    }

    // -------------------------------------------------------------------------
    // Inject phase
    // -------------------------------------------------------------------------

    for (size_t idx = 0; idx < inject_texts.size(); idx++) {
        const std::string & text = inject_texts[idx];

        if (text.empty()) {
            LOG_WRN("%s: warning: skipping empty inject text at index %zu\n", __func__, idx);
            continue;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // BOS only when the KV cache is empty (no prior load or inject)
        bool add_bos = (learning_offset == 0);
        std::vector<llama_token> tokens = common_tokenize(ctx, text, add_bos, false);

        if (learning_offset + (llama_pos) tokens.size() > n_ctx) {
            LOG_ERR("%s: error: inject would exceed context size (%d + %d > %d)\n",
                    __func__, (int) learning_offset, (int) tokens.size(), n_ctx);
            llama_batch_free(batch);
            llama_backend_free();
            return 1;
        }

        if (!decode_tokens(ctx, batch, tokens, learning_offset, LEARNING_SEQ_ID, n_batch, false)) {
            LOG_ERR("%s: error: failed to decode inject text %zu\n", __func__, idx);
            llama_batch_free(batch);
            llama_backend_free();
            return 1;
        }

        learning_entry entry;
        entry.text      = text;
        entry.tokens    = tokens;
        entry.pos_start = learning_offset;
        entry.pos_end   = learning_offset + (llama_pos) tokens.size();
        entries.push_back(std::move(entry));

        all_learning_tokens.insert(all_learning_tokens.end(), tokens.begin(), tokens.end());
        learning_offset += (llama_pos) tokens.size();

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        fprintf(stderr, "%s: injected %zu tokens (%.1f ms): \"%s\"\n",
                __func__, tokens.size(), ms, text.c_str());
    }

    // -------------------------------------------------------------------------
    // Save phase
    // -------------------------------------------------------------------------

    if (!save_path.empty()) {
        auto t_start = std::chrono::high_resolution_clock::now();

        size_t ret = llama_state_seq_save_file(ctx, save_path.c_str(), LEARNING_SEQ_ID,
                all_learning_tokens.data(), all_learning_tokens.size());

        if (ret == 0) {
            LOG_ERR("%s: error: failed to save learning cache to '%s'\n", __func__, save_path.c_str());
            llama_batch_free(batch);
            llama_backend_free();
            return 1;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        fprintf(stderr, "%s: saved %zu tokens to '%s' (%.1f ms, %zu bytes)\n",
                __func__, all_learning_tokens.size(), save_path.c_str(), ms, ret);
    }

    // -------------------------------------------------------------------------
    // Generate phase
    // -------------------------------------------------------------------------

    if (params.prompt.empty()) {
        // Nothing to generate - save-only mode
        llama_batch_free(batch);
        llama_backend_free();
        return 0;
    }

    const bool has_learning = (learning_offset > 0);

    // Warn if generation might evict learning entries via context shift
    if (has_learning && params.n_predict > 0) {
        // Rough estimate of total tokens needed
        std::vector<llama_token> prompt_check = common_tokenize(ctx, params.prompt, !has_learning, false);
        int32_t total_est = learning_offset + (int32_t) prompt_check.size() + params.n_predict;
        if (total_est > n_ctx) {
            LOG_WRN("%s: warning: estimated total tokens (%d) exceeds context size (%d), "
                    "learning entries may be evicted by context shift\n",
                    __func__, total_est, n_ctx);
        }
    }

    if (compare_mode && has_learning) {
        // -----------------------------------------------------------------
        // Compare mode: generate without learning, then with learning
        // -----------------------------------------------------------------

        print_cache_summary(entries, learning_offset);

        // --- Pass 1: without learning cache ---
        printf("=== Without learning cache ===\n");

        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, true, false);

        if (!decode_tokens(ctx, batch, prompt_tokens, 0, CONVERSATION_SEQ_ID, n_batch, true)) {
            LOG_ERR("%s: error: failed to decode prompt (pass 1)\n", __func__);
            llama_batch_free(batch);
            llama_backend_free();
            return 1;
        }

        common_sampler * smpl1 = common_sampler_init(model, params.sampling);
        llama_pos n_past1 = (llama_pos) prompt_tokens.size();

        generate(ctx, batch, smpl1, vocab, n_past1, params.n_predict);
        common_sampler_free(smpl1);

        // Clear conversation seq for pass 2
        llama_memory_seq_rm(mem, CONVERSATION_SEQ_ID, -1, -1);

        // --- Pass 2: with learning cache ---
        printf("\n=== With learning cache (%d tokens, %d entries) ===\n",
               (int) learning_offset, (int) entries.size());

        // Make learning entries visible to conversation
        llama_memory_seq_cp(mem, LEARNING_SEQ_ID, CONVERSATION_SEQ_ID, -1, -1);

        // Tokenize without BOS (learning region has it)
        prompt_tokens = common_tokenize(ctx, params.prompt, false, false);

        llama_pos conv_start = learning_offset;

        if (!decode_tokens(ctx, batch, prompt_tokens, conv_start, CONVERSATION_SEQ_ID, n_batch, true)) {
            LOG_ERR("%s: error: failed to decode prompt (pass 2)\n", __func__);
            llama_batch_free(batch);
            llama_backend_free();
            return 1;
        }

        common_sampler * smpl2 = common_sampler_init(model, params.sampling);
        llama_pos n_past2 = conv_start + (llama_pos) prompt_tokens.size();

        generate(ctx, batch, smpl2, vocab, n_past2, params.n_predict);
        common_sampler_free(smpl2);

    } else {
        // -----------------------------------------------------------------
        // Normal mode: single generation pass
        // -----------------------------------------------------------------

        if (has_learning) {
            print_cache_summary(entries, learning_offset);

            // Make learning entries visible to conversation
            llama_memory_seq_cp(mem, LEARNING_SEQ_ID, CONVERSATION_SEQ_ID, -1, -1);
        }

        // BOS only when no learning entries
        bool add_bos = !has_learning;
        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, false);

        llama_pos conv_start = has_learning ? learning_offset : 0;

        auto t_prompt_start = std::chrono::high_resolution_clock::now();

        if (!decode_tokens(ctx, batch, prompt_tokens, conv_start, CONVERSATION_SEQ_ID, n_batch, true)) {
            LOG_ERR("%s: error: failed to decode prompt\n", __func__);
            llama_batch_free(batch);
            llama_backend_free();
            return 1;
        }

        auto t_prompt_end = std::chrono::high_resolution_clock::now();
        double prompt_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();
        fprintf(stderr, "%s: prompt decoded (%zu tokens, %.1f ms)\n",
                __func__, prompt_tokens.size(), prompt_ms);

        common_sampler * smpl = common_sampler_init(model, params.sampling);
        llama_pos n_past = conv_start + (llama_pos) prompt_tokens.size();

        generate(ctx, batch, smpl, vocab, n_past, params.n_predict);
        common_sampler_free(smpl);
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
