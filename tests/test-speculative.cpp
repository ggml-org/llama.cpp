#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr,
                "  --model-draft, -md <path> : Path to separate draft model (enables 2-model speculative decoding)\n");
        fprintf(stderr, "  --mtp                     : Enable Multi-Token Prediction (MTP) mode testing\n");
        fprintf(stderr, "  --n-predict, -n <int>     : Number of tokens to generate (default: 20)\n");
        fprintf(stderr, "  --n-draft, -nd <int>      : Number of draft tokens per burst (default: 3)\n");
        fprintf(stderr, "  --no-synthetic-vision     : Skip the synthetic multimodal embedding bypass test\n");
        return 1;
    }

    const std::string model_tgt_path = argv[1];
    std::string       model_dft_path = "";
    bool              bIsMtp         = false;
    int32_t           max_generation = 20;
    int32_t           n_draft        = 3;
    bool              test_vision    = true;

    // Parse Command-Line Arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mtp") {
            bIsMtp = true;
        } else if ((arg == "--model-draft" || arg == "-md") && i + 1 < argc) {
            model_dft_path = argv[++i];
        } else if ((arg == "--n-predict" || arg == "-n") && i + 1 < argc) {
            max_generation = std::atoi(argv[++i]);
        } else if ((arg == "--n-draft" || arg == "-nd") && i + 1 < argc) {
            n_draft = std::atoi(argv[++i]);
        } else if (arg == "--no-synthetic-vision") {
            test_vision = false;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
            return 1;
        }
    }

    const bool is_two_model = !model_dft_path.empty();

    // 1. Initialize Backend
    llama_backend_init();

    // 2. Load Models with STRICT GPU 0 OFFLOADING
    llama_model_params mparams_tgt = llama_model_default_params();
    // Explicitly command the loader to read MTP projection tensors into VRAM if testing MTP mode
    mparams_tgt.load_mtp           = bIsMtp;
    mparams_tgt.split_mode         = LLAMA_SPLIT_MODE_NONE;  // Disable multi-GPU layer splitting
    mparams_tgt.main_gpu           = 0;                      // Force to device CUDA0
    mparams_tgt.n_gpu_layers       = 999;                    // Ensure full offload

    fprintf(stderr, "[INIT] Loading target model: %s\n", model_tgt_path.c_str());
    llama_model * model_tgt = llama_model_load_from_file(model_tgt_path.c_str(), mparams_tgt);
    GGML_ASSERT(model_tgt != nullptr && "Failed to load target model");

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    GGML_ASSERT(vocab_tgt != nullptr && "Failed to load target vocab");

    llama_model * model_dft = nullptr;
    if (is_two_model) {
        fprintf(stderr, "[INIT] Loading separate draft model: %s\n", model_dft_path.c_str());
        llama_model_params mparams_dft = llama_model_default_params();
        mparams_dft.split_mode         = LLAMA_SPLIT_MODE_NONE;  // Disable multi-GPU layer splitting
        mparams_dft.main_gpu           = 0;                      // Force to device CUDA0
        mparams_dft.n_gpu_layers       = 999;                    // Ensure full offload

        model_dft = llama_model_load_from_file(model_dft_path.c_str(), mparams_dft);
        GGML_ASSERT(model_dft != nullptr && "Failed to load draft model");
    } else {
        fprintf(stderr, "[INIT] Operating in single-model mode (%s)\n",
                bIsMtp ? "Multi-Token Prediction" : "Self-Speculation");
        // For CI testing without a separate draft model, we reuse the SAME model for both target and draft (Self-Speculation).
        // This validates the engine's speculative mechanics without downloading two separate physical models.
        model_dft = model_tgt;
    }

    // 3. Initialize Speculative Parameters
    llama_speculative_params spec_params = llama_speculative_default_params();
    // A budget of 3. For MTP, this optimally matches the model's `nextn_predict_layers` count.
    spec_params.n_draft                  = n_draft;
    spec_params.is_mtp                   = bIsMtp;

    // 4. Create Contexts Dynamically
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 1024;
    cparams.n_batch              = 512;
    // Speculative decoding dynamically forks sequences during the draft phase.
    // 1 Base Sequence + 3 Draft Tokens = 4 concurrent sequences max (Prompt 0 + 3 Draft Branches).
    cparams.n_seq_max            = 4;

    llama_context * ctx_tgt = llama_init_from_model(model_tgt, cparams);
    GGML_ASSERT(ctx_tgt != nullptr && "Failed to create target context");

    if (spec_params.is_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
    GGML_ASSERT(ctx_dft != nullptr && "Failed to create draft context");

    // 5. Initialize Greedy Sampler
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler *            sampler = llama_sampler_chain_init(sparams);
    // Speculative verification relies on strict mathematical equality. Stochastic samplers will fail validation here.
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // 6. Initialize Speculative Context
    llama_speculative_context * spec_ctx = llama_speculative_init(ctx_tgt, ctx_dft, sampler, &spec_params);
    GGML_ASSERT(spec_ctx != nullptr && "Failed to initialize speculative context");

    // 7. Synthetic Multimodal Embedding Bypass Test
    if (test_vision) {
        fprintf(stderr, "\n--- [TEST] Synthetic Multimodal Embedding Bypass ---\n");
        const int32_t n_embd_tgt      = llama_model_n_embd(model_tgt);
        const int32_t n_vision_tokens = 8;

        llama_batch        batch_vision = llama_batch_init(n_vision_tokens, n_embd_tgt, 1);
        std::vector<float> dummy_embd((size_t) n_vision_tokens * n_embd_tgt, 0.0f);
        std::memcpy(batch_vision.embd, dummy_embd.data(), dummy_embd.size() * sizeof(float));

        for (int i = 0; i < n_vision_tokens; i++) {
            batch_vision.pos[i]       = i;
            batch_vision.n_seq_id[i]  = 1;
            batch_vision.seq_id[i][0] = 0;
            batch_vision.logits[i]    = false;
        }
        batch_vision.n_tokens = n_vision_tokens;

        llama_speculative_result vision_results[1];
        int32_t                  vision_out = llama_speculative_decode(spec_ctx, &batch_vision, vision_results, 1);

        GGML_ASSERT(vision_out == 0 && "Vision bypass failed: expected 0 speculative results for embedding batch");
        fprintf(stderr,
                ">> PASS: Synthetic multimodal batch successfully bypassed draft model and emitted 0 results.\n");

        llama_batch_free(batch_vision);

        // Reset memory
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
        if (ctx_dft && ctx_dft != ctx_tgt) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);
        }
    }

    // 8. Tokenize Prompt
    llama_batch              batch  = llama_batch_init(512, 0, 1);
    const char *             prompt = "The capital of France is";
    std::vector<llama_token> tokens(32);

    int32_t n_tokens = llama_tokenize(vocab_tgt, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
    GGML_ASSERT(n_tokens > 0 && "Failed to tokenize prompt");

    for (int i = 0; i < n_tokens; i++) {
        batch.token[batch.n_tokens]     = tokens[i];
        batch.pos[batch.n_tokens]       = i;
        batch.n_seq_id[batch.n_tokens]  = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens]    = false;
        batch.n_tokens++;
    }
    batch.logits[n_tokens - 1] = true;  // Request logits for final prompt token

    // -----------------------------------------------------------------------
    // 9. BASELINE BENCHMARK (Pure Autoregressive Target Generation)
    // -----------------------------------------------------------------------
    fprintf(stderr, "\n--- [BENCHMARK] Baseline Pure Autoregressive Generation ---\n");

    // Evaluate the prompt manually
    llama_decode(ctx_tgt, batch);

    auto t_base_start = std::chrono::high_resolution_clock::now();

    llama_token base_token = llama_sampler_sample(sampler, ctx_tgt, batch.n_tokens - 1);
    llama_sampler_accept(sampler, base_token);

    for (int i = 0; i < max_generation; i++) {
        llama_batch b  = llama_batch_init(1, 0, 1);
        b.token[0]     = base_token;
        b.pos[0]       = n_tokens + i;
        b.n_seq_id[0]  = 1;
        b.seq_id[0][0] = 0;
        b.logits[0]    = true;
        b.n_tokens     = 1;

        llama_decode(ctx_tgt, b);
        base_token = llama_sampler_sample(sampler, ctx_tgt, 0);
        llama_sampler_accept(sampler, base_token);

        llama_batch_free(b);
    }

    auto   t_base_end       = std::chrono::high_resolution_clock::now();
    double base_ms          = std::chrono::duration<double, std::milli>(t_base_end - t_base_start).count();
    double base_tok_per_sec = max_generation / (base_ms / 1000.0);

    fprintf(stderr, ">> Baseline Speed: %.2f tokens/second\n", base_tok_per_sec);

    // Clean up context completely so the Speculative Test can start fresh
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
    llama_sampler_reset(sampler);
    // -----------------------------------------------------------------------

    // 10. Execute Continuous Speculative Decoding Loop with Benchmarking
    llama_speculative_result results[1];
    int32_t                  generated_tokens = 0;
    int32_t                  n_iterations     = 0;
    llama_pos                current_pos      = n_tokens;

    fprintf(stderr, "\n--- [BENCHMARK] Speculative Decoding Generation ---\n");
    fprintf(stderr, "Prompt: \"%s\"\n", prompt);
    fprintf(stderr, "Generating: ");

    auto t_start = std::chrono::high_resolution_clock::now();

    while (generated_tokens < max_generation) {
        // Run the full pipeline (Draft -> Verify -> Zero-Copy Rollback) internally.
        // llama_speculative_decode evaluates the prompt batch on its very first run, then evaluates draft bursts.
        int32_t n_results = llama_speculative_decode(spec_ctx, &batch, results, 1);
        GGML_ASSERT(n_results == 1 && "Expected exactly 1 sequence result");
        GGML_ASSERT(results[0].count > 0 && "Expected at least 1 accepted token");

        // Print accepted tokens cleanly
        for (int i = 0; i < results[0].count; i++) {
            char    buf[128] = { 0 };
            int32_t n_chars  = llama_token_to_piece(vocab_tgt, results[0].tokens[i], buf, sizeof(buf) - 1, 0, true);
            if (n_chars >= 0) {
                buf[n_chars] = '\0';
                fprintf(stderr, "%s", buf);
            }
        }

        generated_tokens += results[0].count;
        n_iterations++;

        // Prepare batch for next step
        llama_token last_token = results[0].tokens[results[0].count - 1];
        // Prepare the batch for the next step.
        // Because the speculative engine handles its own KV cache and context synchronization,
        // we only need to pass the final accepted token back in to trigger the next loop.
        current_pos += results[0].count;

        batch.n_tokens     = 1;
        batch.token[0]     = last_token;
        batch.pos[0]       = current_pos - 1;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    // Compute Benchmark Statistics
    double elapsed_ms   = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double elapsed_sec  = elapsed_ms / 1000.0;
    double tok_per_sec  = (elapsed_sec > 0.0) ? (generated_tokens / elapsed_sec) : 0.0;
    double avg_accepted = (n_iterations > 0) ? ((double) generated_tokens / n_iterations) : 0.0;
    double speedup      = (base_tok_per_sec > 0.0) ? (tok_per_sec / base_tok_per_sec) : 0.0;

    fprintf(stderr, "\n\n");
    fprintf(stderr, "================ BENCHMARK RESULTS ================\n");
    fprintf(stderr, " Mode                  : %s\n",
            bIsMtp ? "Multi-Token Prediction (MTP)" : (is_two_model ? "Two-Model Speculative" : "Self-Speculative"));
    fprintf(stderr, " Target Model          : %s\n", model_tgt_path.c_str());
    if (is_two_model) {
        fprintf(stderr, " Draft Model           : %s\n", model_dft_path.c_str());
    }
    fprintf(stderr, " Draft Burst Budget (K): %d\n", n_draft);
    fprintf(stderr, " Total Generated       : %d tokens\n", generated_tokens);
    fprintf(stderr, " --------------------------------------------------\n");
    fprintf(stderr, " Baseline Speed        : %.2f tokens/second\n", base_tok_per_sec);
    fprintf(stderr, " Speculative Speed     : %.2f tokens/second\n", tok_per_sec);
    fprintf(stderr, " Net Speedup           : %.2fx\n", speedup);
    fprintf(stderr, " --------------------------------------------------\n");
    fprintf(stderr, " Speculative Rounds    : %d bursts\n", n_iterations);
    fprintf(stderr, " Avg Tokens / Round    : %.2f tokens/burst (Acceptance Efficiency)\n", avg_accepted);
    fprintf(stderr, "===================================================\n\n");

    GGML_ASSERT(generated_tokens >= max_generation && "Failed to generate the target number of tokens");

    // 11. Memory Cleanup
    llama_batch_free(batch);
    llama_speculative_free(spec_ctx);
    llama_sampler_free(sampler);

    // Unconditionally free ctx_dft, since we allocate a draft context in both two-model and self-speculation modes.
    llama_free(ctx_dft);
    llama_free(ctx_tgt);

    if (is_two_model && model_dft != nullptr) {
        llama_free_model(model_dft);
    }
    llama_free_model(model_tgt);
    llama_backend_free();

    fprintf(stderr, "All speculative tests and benchmarks completed successfully.\n");
    return 0;
}
