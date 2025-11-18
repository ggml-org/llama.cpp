/**
 * Simple Inference Example - llama.cpp API
 *
 * This example demonstrates basic text generation using the llama.cpp API.
 * It shows how to:
 * - Load a GGUF model
 * - Initialize the context
 * - Tokenize a prompt
 * - Generate text with greedy sampling
 * - Properly clean up resources
 *
 * Build instructions:
 *   cmake -B build && cmake --build build
 *   ./build/01-simple-inference -m /path/to/model.gguf -p "Hello, my name is"
 *
 * For more information, see:
 * - https://github.com/ggerganov/llama.cpp/blob/master/examples/simple/simple.cpp
 * - https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h
 */

#include "llama.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

// RAII wrapper for llama_model to ensure proper cleanup
struct ModelDeleter {
    void operator()(llama_model* model) const {
        if (model) llama_model_free(model);
    }
};

// RAII wrapper for llama_context to ensure proper cleanup
struct ContextDeleter {
    void operator()(llama_context* ctx) const {
        if (ctx) llama_free(ctx);
    }
};

// RAII wrapper for llama_sampler to ensure proper cleanup
struct SamplerDeleter {
    void operator()(llama_sampler* smpl) const {
        if (smpl) llama_sampler_free(smpl);
    }
};

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

/**
 * Tokenize a text prompt into llama tokens
 *
 * @param vocab The vocabulary from the model
 * @param text The text to tokenize
 * @param add_special Add special tokens (BOS, etc.)
 * @return Vector of tokens, or empty vector on error
 */
std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                   const std::string& text,
                                   bool add_special = true) {
    // First call with NULL to get the required size
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(),
                                          nullptr, 0, add_special, true);

    if (n_tokens <= 0) {
        std::cerr << "Error: Failed to tokenize text" << std::endl;
        return {};
    }

    // Allocate and tokenize
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
 * Convert a token back to text
 *
 * @param vocab The vocabulary from the model
 * @param token The token to convert
 * @return The text representation of the token
 */
std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);

    if (n < 0) {
        std::cerr << "Warning: Failed to convert token to piece" << std::endl;
        return "";
    }

    return std::string(buf, n);
}

/**
 * Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << "\nUsage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model PATH      Path to the GGUF model file (required)\n";
    std::cout << "  -p, --prompt TEXT     Text prompt for generation (default: \"Hello, my name is\")\n";
    std::cout << "  -n, --n-predict N     Number of tokens to generate (default: 32)\n";
    std::cout << "  -ngl, --n-gpu-layers N  Number of layers to offload to GPU (default: 99)\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " -m model.gguf -p \"The quick brown fox\" -n 50\n";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Configuration parameters
    std::string model_path;
    std::string prompt = "Hello, my name is";
    int n_predict = 32;
    int n_gpu_layers = 99;

    // Parse command line arguments
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
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (model_path.empty()) {
        std::cerr << "Error: Model path is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Configuration:\n";
    std::cout << "  Model: " << model_path << "\n";
    std::cout << "  Prompt: \"" << prompt << "\"\n";
    std::cout << "  Tokens to generate: " << n_predict << "\n";
    std::cout << "  GPU layers: " << n_gpu_layers << "\n" << std::endl;

    // Step 1: Load dynamic backends (enables GPU support)
    ggml_backend_load_all();
    std::cout << "Loaded backends" << std::endl;

    // Step 2: Initialize and load the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    std::cout << "Loading model from: " << model_path << std::endl;
    ModelPtr model(llama_model_load_from_file(model_path.c_str(), model_params));

    if (!model) {
        std::cerr << "Error: Failed to load model from " << model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // Get vocabulary for tokenization
    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    // Step 3: Tokenize the prompt
    std::cout << "Tokenizing prompt..." << std::endl;
    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt, true);

    if (prompt_tokens.empty()) {
        std::cerr << "Error: Failed to tokenize prompt" << std::endl;
        return 1;
    }
    std::cout << "Prompt tokenized: " << prompt_tokens.size() << " tokens" << std::endl;

    // Step 4: Create the context
    llama_context_params ctx_params = llama_context_default_params();

    // Set context size to accommodate prompt + generation
    ctx_params.n_ctx = prompt_tokens.size() + n_predict;

    // Batch size should be at least the size of the prompt
    ctx_params.n_batch = prompt_tokens.size();

    // Enable performance tracking
    ctx_params.no_perf = false;

    std::cout << "Creating context (n_ctx=" << ctx_params.n_ctx << ")..." << std::endl;
    ContextPtr ctx(llama_init_from_model(model.get(), ctx_params));

    if (!ctx) {
        std::cerr << "Error: Failed to create context" << std::endl;
        return 1;
    }
    std::cout << "Context created successfully" << std::endl;

    // Step 5: Initialize the sampler (using greedy sampling for simplicity)
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;

    SamplerPtr sampler(llama_sampler_chain_init(sampler_params));

    // Add greedy sampler (always picks the most likely token)
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

    std::cout << "Sampler initialized (greedy)" << std::endl;

    // Step 6: Print the original prompt
    std::cout << "\n=== Generation ===\n";
    for (auto token : prompt_tokens) {
        std::cout << token_to_piece(vocab, token);
    }
    std::cout << std::flush;

    // Step 7: Prepare the initial batch with the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            prompt_tokens.size());

    // Handle encoder-decoder models (like T5, BART)
    if (llama_model_has_encoder(model.get())) {
        if (llama_encode(ctx.get(), batch)) {
            std::cerr << "\nError: Failed to encode prompt" << std::endl;
            return 1;
        }

        // For encoder-decoder, start decoding with the decoder start token
        llama_token decoder_start = llama_model_decoder_start_token(model.get());
        if (decoder_start == LLAMA_TOKEN_NULL) {
            decoder_start = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start, 1);
    }

    // Step 8: Main generation loop
    int n_generated = 0;

    for (int n_pos = 0; n_pos + batch.n_tokens < prompt_tokens.size() + n_predict; ) {
        // Decode the current batch
        if (llama_decode(ctx.get(), batch)) {
            std::cerr << "\nError: Failed to decode" << std::endl;
            return 1;
        }

        n_pos += batch.n_tokens;

        // Sample the next token
        llama_token new_token = llama_sampler_sample(sampler.get(), ctx.get(), -1);

        // Check if we've reached the end of generation
        if (llama_vocab_is_eog(vocab, new_token)) {
            std::cout << std::flush;
            break;
        }

        // Convert token to text and print
        std::cout << token_to_piece(vocab, new_token) << std::flush;

        // Prepare next batch with the newly generated token
        batch = llama_batch_get_one(&new_token, 1);
        n_generated++;
    }

    std::cout << "\n\n=== Statistics ===\n";
    std::cout << "Generated " << n_generated << " tokens" << std::endl;

    // Step 9: Print performance statistics
    llama_perf_sampler_print(sampler.get());
    llama_perf_context_print(ctx.get());

    // Step 10: Cleanup is automatic via smart pointers
    std::cout << "\nCleanup complete" << std::endl;

    return 0;
}
