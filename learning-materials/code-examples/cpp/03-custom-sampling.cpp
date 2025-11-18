/**
 * Custom Sampling Example - llama.cpp API
 *
 * This example demonstrates advanced sampling techniques using llama.cpp.
 * It shows how to:
 * - Build custom sampler chains with various strategies
 * - Combine multiple sampling methods (temperature, top-k, top-p, etc.)
 * - Implement a custom sampler from scratch
 * - Apply penalties for repetition control
 * - Compare different sampling strategies
 *
 * Sampling is crucial for controlling the quality and diversity of generated text.
 * Different applications require different sampling strategies:
 * - Creative writing: Higher temperature, nucleus sampling
 * - Code generation: Lower temperature, top-k sampling
 * - Chatbots: Balanced settings with repetition penalties
 *
 * Build instructions:
 *   cmake -B build && cmake --build build
 *   ./build/03-custom-sampling -m /path/to/model.gguf -p "Once upon a time"
 *
 * For more information, see:
 * - https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h
 * - https://github.com/ggerganov/llama.cpp/blob/master/common/sampling.h
 */

#include "llama.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <algorithm>
#include <cmath>

// RAII wrappers for automatic cleanup
struct ModelDeleter {
    void operator()(llama_model* model) const {
        if (model) llama_model_free(model);
    }
};

struct ContextDeleter {
    void operator()(llama_context* ctx) const {
        if (ctx) llama_free(ctx);
    }
};

struct SamplerDeleter {
    void operator()(llama_sampler* smpl) const {
        if (smpl) llama_sampler_free(smpl);
    }
};

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

/**
 * Tokenize text into tokens
 */
std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                   const std::string& text,
                                   bool add_special = true) {
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(),
                                          nullptr, 0, add_special, true);

    if (n_tokens <= 0) {
        std::cerr << "Error: Failed to tokenize text" << std::endl;
        return {};
    }

    std::vector<llama_token> tokens(n_tokens);
    const int result = llama_tokenize(vocab, text.c_str(), text.size(),
                                      tokens.data(), tokens.size(),
                                      add_special, true);

    if (result < 0) {
        std::cerr << "Error: Tokenization failed" << std::endl;
        return {};
    }

    return tokens;
}

/**
 * Convert token to text
 */
std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);

    if (n < 0) {
        return "";
    }

    return std::string(buf, n);
}

// =============================================================================
// Custom Sampler Implementation
// =============================================================================

/**
 * Custom sampler context - holds sampler-specific state
 *
 * This example implements a "top-logprob" sampler that only considers
 * tokens within a certain log probability threshold of the best token.
 */
struct custom_sampler_context {
    float logprob_threshold;  // Log probability threshold
    size_t min_keep;          // Minimum number of tokens to keep
};

/**
 * Custom sampler: name function
 */
static const char* custom_sampler_name(const struct llama_sampler* /*smpl*/) {
    return "custom-top-logprob";
}

/**
 * Custom sampler: apply function
 *
 * This is where the actual sampling logic happens.
 * It filters the token probability distribution based on log probabilities.
 */
static void custom_sampler_apply(struct llama_sampler* smpl,
                                  llama_token_data_array* cur_p) {
    auto* ctx = static_cast<custom_sampler_context*>(smpl->ctx);

    if (cur_p->size <= ctx->min_keep) {
        return;  // Not enough tokens to filter
    }

    // Sort by logit (descending) to find the best token
    std::sort(cur_p->data, cur_p->data + cur_p->size,
              [](const llama_token_data& a, const llama_token_data& b) {
                  return a.logit > b.logit;
              });

    // Find the log probability threshold relative to the best token
    const float max_logit = cur_p->data[0].logit;
    const float threshold = max_logit + ctx->logprob_threshold;

    // Find how many tokens are above the threshold
    size_t n_keep = 1;
    for (size_t i = 1; i < cur_p->size; i++) {
        if (cur_p->data[i].logit >= threshold) {
            n_keep++;
        } else {
            break;
        }
    }

    // Ensure we keep at least min_keep tokens
    n_keep = std::max(n_keep, ctx->min_keep);
    n_keep = std::min(n_keep, cur_p->size);

    // Update the size to reflect the filtered tokens
    cur_p->size = n_keep;

    // Mark as sorted
    cur_p->sorted = true;
}

/**
 * Custom sampler: clone function
 */
static struct llama_sampler* custom_sampler_clone(const struct llama_sampler* smpl) {
    const auto* ctx = static_cast<const custom_sampler_context*>(smpl->ctx);
    auto* new_ctx = new custom_sampler_context(*ctx);

    return llama_sampler_init(smpl->iface, new_ctx);
}

/**
 * Custom sampler: free function
 */
static void custom_sampler_free(struct llama_sampler* smpl) {
    delete static_cast<custom_sampler_context*>(smpl->ctx);
}

/**
 * Custom sampler: interface definition
 */
static struct llama_sampler_i custom_sampler_iface = {
    /* .name   = */ custom_sampler_name,
    /* .accept = */ nullptr,  // We don't need to track accepted tokens
    /* .apply  = */ custom_sampler_apply,
    /* .reset  = */ nullptr,  // No state to reset
    /* .clone  = */ custom_sampler_clone,
    /* .free   = */ custom_sampler_free,
};

/**
 * Initialize custom sampler
 */
struct llama_sampler* custom_sampler_init(float logprob_threshold, size_t min_keep) {
    auto* ctx = new custom_sampler_context{logprob_threshold, min_keep};
    return llama_sampler_init(&custom_sampler_iface, ctx);
}

// =============================================================================
// Sampler Chain Builders
// =============================================================================

/**
 * Create a conservative sampler chain (for factual, deterministic output)
 */
SamplerPtr create_conservative_sampler(uint32_t seed) {
    std::cout << "Creating conservative sampler chain:\n";
    std::cout << "  - Temperature: 0.3 (low randomness)\n";
    std::cout << "  - Top-K: 20 (narrow focus)\n";
    std::cout << "  - Top-P: 0.85 (nucleus sampling)\n";
    std::cout << "  - Repetition penalty: 1.1\n" << std::endl;

    auto params = llama_sampler_chain_default_params();
    params.no_perf = false;

    SamplerPtr sampler(llama_sampler_chain_init(params));

    // Apply penalties first to modify logits
    llama_sampler_chain_add(sampler.get(),
        llama_sampler_init_penalties(
            /* n_vocab           = */ 32000,  // Will be updated by context
            /* special_eos_id    = */ LLAMA_TOKEN_NULL,
            /* linefeed_id       = */ LLAMA_TOKEN_NULL,
            /* penalty_last_n    = */ 64,
            /* penalty_repeat    = */ 1.1f,
            /* penalty_freq      = */ 0.0f,
            /* penalty_present   = */ 0.0f,
            /* penalize_nl       = */ false,
            /* ignore_eos        = */ false
        ));

    // Apply top-k filtering
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(20));

    // Apply top-p (nucleus) filtering
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.85f, 1));

    // Apply temperature
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.3f));

    // Final sampling
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed));

    return sampler;
}

/**
 * Create a creative sampler chain (for diverse, imaginative output)
 */
SamplerPtr create_creative_sampler(uint32_t seed) {
    std::cout << "Creating creative sampler chain:\n";
    std::cout << "  - Temperature: 0.9 (high randomness)\n";
    std::cout << "  - Top-K: 50 (wide selection)\n";
    std::cout << "  - Top-P: 0.95 (broader nucleus)\n";
    std::cout << "  - Min-P: 0.05 (minimum probability)\n";
    std::cout << "  - Repetition penalty: 1.05\n" << std::endl;

    auto params = llama_sampler_chain_default_params();
    params.no_perf = false;

    SamplerPtr sampler(llama_sampler_chain_init(params));

    // Apply penalties
    llama_sampler_chain_add(sampler.get(),
        llama_sampler_init_penalties(
            32000, LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL,
            64, 1.05f, 0.0f, 0.0f, false, false
        ));

    // Apply top-k filtering
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(50));

    // Apply top-p (nucleus) filtering
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.95f, 1));

    // Apply min-p filtering (removes low probability tokens)
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_min_p(0.05f, 1));

    // Apply temperature
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.9f));

    // Final sampling
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed));

    return sampler;
}

/**
 * Create a custom sampler chain (uses our custom sampler)
 */
SamplerPtr create_custom_sampler(uint32_t seed) {
    std::cout << "Creating custom sampler chain:\n";
    std::cout << "  - Custom top-logprob filter: -2.0 threshold\n";
    std::cout << "  - Temperature: 0.7\n";
    std::cout << "  - Repetition penalty: 1.1\n" << std::endl;

    auto params = llama_sampler_chain_default_params();
    params.no_perf = false;

    SamplerPtr sampler(llama_sampler_chain_init(params));

    // Apply penalties
    llama_sampler_chain_add(sampler.get(),
        llama_sampler_init_penalties(
            32000, LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL,
            64, 1.1f, 0.0f, 0.0f, false, false
        ));

    // Apply our custom sampler!
    // Keeps tokens within 2.0 log probability of the best token
    llama_sampler_chain_add(sampler.get(), custom_sampler_init(-2.0f, 5));

    // Apply temperature
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.7f));

    // Final sampling
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed));

    return sampler;
}

/**
 * Generate text with a given sampler
 */
std::string generate_text(llama_context* ctx,
                          const llama_vocab* vocab,
                          llama_sampler* sampler,
                          const std::vector<llama_token>& prompt_tokens,
                          int n_predict) {
    std::string result;

    // Prepare initial batch
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token*>(prompt_tokens.data()),
        prompt_tokens.size()
    );

    // Handle encoder-decoder models
    if (llama_model_has_encoder(llama_get_model(ctx))) {
        if (llama_encode(ctx, batch)) {
            std::cerr << "Error: Failed to encode" << std::endl;
            return "";
        }

        llama_token decoder_start = llama_model_decoder_start_token(llama_get_model(ctx));
        if (decoder_start == LLAMA_TOKEN_NULL) {
            decoder_start = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start, 1);
    }

    // Generation loop
    int n_generated = 0;
    for (int n_pos = 0; n_pos + batch.n_tokens < prompt_tokens.size() + n_predict; ) {
        // Decode
        if (llama_decode(ctx, batch)) {
            std::cerr << "Error: Failed to decode" << std::endl;
            return result;
        }

        n_pos += batch.n_tokens;

        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Append to result
        result += token_to_piece(vocab, new_token);

        // Prepare next batch
        batch = llama_batch_get_one(&new_token, 1);
        n_generated++;
    }

    return result;
}

/**
 * Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << "\nUsage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model PATH      Path to the GGUF model file (required)\n";
    std::cout << "  -p, --prompt TEXT     Text prompt (default: \"Once upon a time\")\n";
    std::cout << "  -n, --n-predict N     Number of tokens to generate (default: 50)\n";
    std::cout << "  -s, --strategy STR    Sampling strategy: conservative, creative, custom, all (default: all)\n";
    std::cout << "  --seed N              Random seed (default: 42)\n";
    std::cout << "  -ngl, --n-gpu-layers N  GPU layers (default: 99)\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " -m model.gguf -p \"The future of AI\" -s creative\n";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Configuration
    std::string model_path;
    std::string prompt = "Once upon a time";
    int n_predict = 50;
    std::string strategy = "all";
    uint32_t seed = 42;
    int n_gpu_layers = 99;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--strategy") && i + 1 < argc) {
            strategy = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: Model path is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== Custom Sampling Example ===\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Prompt: \"" << prompt << "\"\n";
    std::cout << "Tokens to generate: " << n_predict << "\n";
    std::cout << "Strategy: " << strategy << "\n";
    std::cout << "Seed: " << seed << "\n" << std::endl;

    // Load backends
    ggml_backend_load_all();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    std::cout << "Loading model..." << std::endl;
    ModelPtr model(llama_model_load_from_file(model_path.c_str(), model_params));

    if (!model) {
        std::cerr << "Error: Failed to load model" << std::endl;
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    // Tokenize prompt
    std::cout << "Tokenizing prompt..." << std::endl;
    auto prompt_tokens = tokenize(vocab, prompt, true);

    if (prompt_tokens.empty()) {
        std::cerr << "Error: Failed to tokenize prompt" << std::endl;
        return 1;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = prompt_tokens.size() + n_predict;
    ctx_params.n_batch = prompt_tokens.size();
    ctx_params.no_perf = false;

    std::cout << "Creating context..." << std::endl;
    ContextPtr ctx(llama_init_from_model(model.get(), ctx_params));

    if (!ctx) {
        std::cerr << "Error: Failed to create context" << std::endl;
        return 1;
    }

    // Print prompt
    std::cout << "\n=== Prompt ===\n" << prompt << "\n" << std::endl;

    // Generate with different strategies
    if (strategy == "conservative" || strategy == "all") {
        std::cout << "=== Conservative Sampling ===\n";
        auto sampler = create_conservative_sampler(seed);
        std::string output = generate_text(ctx.get(), vocab, sampler.get(),
                                          prompt_tokens, n_predict);
        std::cout << prompt << output << "\n" << std::endl;
        llama_perf_sampler_print(sampler.get());
        std::cout << std::endl;
    }

    if (strategy == "creative" || strategy == "all") {
        std::cout << "=== Creative Sampling ===\n";
        auto sampler = create_creative_sampler(seed + 1);
        std::string output = generate_text(ctx.get(), vocab, sampler.get(),
                                          prompt_tokens, n_predict);
        std::cout << prompt << output << "\n" << std::endl;
        llama_perf_sampler_print(sampler.get());
        std::cout << std::endl;
    }

    if (strategy == "custom" || strategy == "all") {
        std::cout << "=== Custom Sampling (Our Implementation) ===\n";
        auto sampler = create_custom_sampler(seed + 2);
        std::string output = generate_text(ctx.get(), vocab, sampler.get(),
                                          prompt_tokens, n_predict);
        std::cout << prompt << output << "\n" << std::endl;
        llama_perf_sampler_print(sampler.get());
        std::cout << std::endl;
    }

    // Print context performance
    std::cout << "=== Context Performance ===\n";
    llama_perf_context_print(ctx.get());

    std::cout << "\nDone!" << std::endl;

    return 0;
}
