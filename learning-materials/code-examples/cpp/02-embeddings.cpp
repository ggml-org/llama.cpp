/**
 * Embeddings Example - llama.cpp API
 *
 * This example demonstrates generating text embeddings using llama.cpp.
 * Embeddings are useful for:
 * - Retrieval-Augmented Generation (RAG)
 * - Semantic search
 * - Text similarity comparisons
 * - Clustering and classification
 *
 * Features:
 * - Batch processing of multiple texts
 * - Different pooling strategies (mean, cls, last)
 * - Normalization options
 * - Performance optimizations
 *
 * Build instructions:
 *   cmake -B build && cmake --build build
 *   ./build/02-embeddings -m /path/to/embedding-model.gguf -t "Hello world" "Another text"
 *
 * For more information, see:
 * - https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp
 * - https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h
 */

#include "llama.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <cmath>
#include <iomanip>

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

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;

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
 * Add a sequence of tokens to a batch
 */
void batch_add_seq(llama_batch& batch,
                   const std::vector<llama_token>& tokens,
                   llama_seq_id seq_id) {
    for (size_t i = 0; i < tokens.size(); i++) {
        // Add each token to the batch
        // seq_id allows processing multiple sequences in parallel
        // logits flag set to true for the last token to get embeddings
        llama_batch_add(batch, tokens[i], i, {seq_id}, i == tokens.size() - 1);
    }
}

/**
 * Normalize an embedding vector
 */
void normalize_embedding(const float* input, float* output, int n_embd) {
    // Calculate L2 norm
    float norm = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        norm += input[i] * input[i];
    }
    norm = std::sqrt(norm);

    // Normalize
    if (norm > 0.0f) {
        for (int i = 0; i < n_embd; i++) {
            output[i] = input[i] / norm;
        }
    } else {
        // Handle zero vector
        std::fill_n(output, n_embd, 0.0f);
    }
}

/**
 * Calculate cosine similarity between two embeddings
 */
float cosine_similarity(const float* a, const float* b, int n_embd) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < n_embd; i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a > 0.0f && norm_b > 0.0f) {
        return dot_product / (norm_a * norm_b);
    }
    return 0.0f;
}

/**
 * Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << "\nUsage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model PATH      Path to the GGUF embedding model (required)\n";
    std::cout << "  -t, --texts TEXT...   Texts to embed (space-separated, at least 1 required)\n";
    std::cout << "  -ngl, --n-gpu-layers N  Number of layers to offload to GPU (default: 99)\n";
    std::cout << "  --normalize           Normalize embeddings to unit length (default: true)\n";
    std::cout << "  --no-normalize        Don't normalize embeddings\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " -m model.gguf -t \"Hello world\" \"Goodbye world\"\n";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Configuration
    std::string model_path;
    std::vector<std::string> texts;
    int n_gpu_layers = 99;
    bool normalize = true;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--texts") && i + 1 < argc) {
            // Collect all remaining arguments as texts
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                texts.push_back(argv[++i]);
            }
        } else if (arg == "--normalize") {
            normalize = true;
        } else if (arg == "--no-normalize") {
            normalize = false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate arguments
    if (model_path.empty()) {
        std::cerr << "Error: Model path is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (texts.empty()) {
        std::cerr << "Error: At least one text is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Configuration:\n";
    std::cout << "  Model: " << model_path << "\n";
    std::cout << "  Number of texts: " << texts.size() << "\n";
    std::cout << "  GPU layers: " << n_gpu_layers << "\n";
    std::cout << "  Normalize: " << (normalize ? "yes" : "no") << "\n" << std::endl;

    // Step 1: Load backends
    ggml_backend_load_all();

    // Step 2: Load the model with embedding mode enabled
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    std::cout << "Loading model..." << std::endl;
    ModelPtr model(llama_model_load_from_file(model_path.c_str(), model_params));

    if (!model) {
        std::cerr << "Error: Failed to load model" << std::endl;
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    const int n_embd = llama_model_n_embd(model.get());

    std::cout << "Model loaded (embedding dimension: " << n_embd << ")" << std::endl;

    // Step 3: Tokenize all texts
    std::cout << "\nTokenizing texts..." << std::endl;
    std::vector<std::vector<llama_token>> all_tokens;
    int max_tokens = 0;

    for (size_t i = 0; i < texts.size(); i++) {
        auto tokens = tokenize(vocab, texts[i], true);
        if (tokens.empty()) {
            std::cerr << "Error: Failed to tokenize text " << i << std::endl;
            return 1;
        }
        std::cout << "  Text " << i << ": " << tokens.size() << " tokens" << std::endl;
        max_tokens = std::max(max_tokens, static_cast<int>(tokens.size()));
        all_tokens.push_back(std::move(tokens));
    }

    // Step 4: Create context with embedding mode enabled
    llama_context_params ctx_params = llama_context_default_params();

    // IMPORTANT: Enable embedding mode
    ctx_params.embeddings = true;

    // Set context size to accommodate the longest text
    ctx_params.n_ctx = max_tokens + 16; // Add some padding

    // Batch size to accommodate all sequences
    ctx_params.n_batch = max_tokens * texts.size();

    // Enable performance tracking
    ctx_params.no_perf = false;

    std::cout << "\nCreating context (n_ctx=" << ctx_params.n_ctx
              << ", n_batch=" << ctx_params.n_batch << ")..." << std::endl;

    ContextPtr ctx(llama_init_from_model(model.get(), ctx_params));

    if (!ctx) {
        std::cerr << "Error: Failed to create context" << std::endl;
        return 1;
    }

    // Get the pooling type used by the model
    const llama_pooling_type pooling_type = llama_pooling_type(ctx.get());
    std::cout << "Pooling type: ";
    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            std::cout << "NONE (per-token embeddings)"; break;
        case LLAMA_POOLING_TYPE_MEAN:
            std::cout << "MEAN"; break;
        case LLAMA_POOLING_TYPE_CLS:
            std::cout << "CLS"; break;
        case LLAMA_POOLING_TYPE_LAST:
            std::cout << "LAST"; break;
        default:
            std::cout << "UNKNOWN"; break;
    }
    std::cout << std::endl;

    // Step 5: Create and populate batch with all sequences
    // Each text gets its own sequence ID for parallel processing
    llama_batch batch = llama_batch_init(ctx_params.n_batch, 0, texts.size());

    std::cout << "\nAdding sequences to batch..." << std::endl;
    for (size_t i = 0; i < all_tokens.size(); i++) {
        batch_add_seq(batch, all_tokens[i], i);
    }

    std::cout << "Batch prepared: " << batch.n_tokens << " total tokens, "
              << texts.size() << " sequences" << std::endl;

    // Step 6: Process the batch to get embeddings
    std::cout << "\nGenerating embeddings..." << std::endl;

    // Clear KV cache (not needed for embeddings, but good practice)
    llama_memory_clear(llama_get_memory(ctx.get()), true);

    // Decode the batch
    if (llama_decode(ctx.get(), batch) < 0) {
        std::cerr << "Error: Failed to decode batch" << std::endl;
        llama_batch_free(batch);
        return 1;
    }

    // Step 7: Extract embeddings
    std::vector<std::vector<float>> embeddings(texts.size());

    for (size_t i = 0; i < texts.size(); i++) {
        const float* embd = nullptr;

        // Get embeddings based on pooling type
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // For per-token embeddings, get the last token's embedding
            // Find the last token of this sequence in the batch
            int last_token_idx = -1;
            for (int j = batch.n_tokens - 1; j >= 0; j--) {
                if (batch.seq_id[j][0] == static_cast<llama_seq_id>(i)) {
                    last_token_idx = j;
                    break;
                }
            }

            if (last_token_idx >= 0) {
                embd = llama_get_embeddings_ith(ctx.get(), last_token_idx);
            }
        } else {
            // For pooled embeddings, get the sequence embedding
            embd = llama_get_embeddings_seq(ctx.get(), i);
        }

        if (!embd) {
            std::cerr << "Error: Failed to get embeddings for sequence " << i << std::endl;
            llama_batch_free(batch);
            return 1;
        }

        // Copy and optionally normalize the embedding
        embeddings[i].resize(n_embd);
        if (normalize) {
            normalize_embedding(embd, embeddings[i].data(), n_embd);
        } else {
            std::copy_n(embd, n_embd, embeddings[i].begin());
        }
    }

    std::cout << "Embeddings generated successfully" << std::endl;

    // Step 8: Display results
    std::cout << "\n=== Results ===\n" << std::endl;

    for (size_t i = 0; i < texts.size(); i++) {
        std::cout << "Text " << i << ": \"" << texts[i] << "\"" << std::endl;
        std::cout << "  Embedding (first 10 dimensions): ";
        for (int j = 0; j < std::min(10, n_embd); j++) {
            std::cout << std::fixed << std::setprecision(6) << embeddings[i][j];
            if (j < std::min(10, n_embd) - 1) std::cout << ", ";
        }
        std::cout << (n_embd > 10 ? "..." : "") << std::endl;

        // Calculate L2 norm
        float norm = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            norm += embeddings[i][j] * embeddings[i][j];
        }
        norm = std::sqrt(norm);
        std::cout << "  L2 norm: " << std::fixed << std::setprecision(6) << norm << "\n" << std::endl;
    }

    // Step 9: Calculate pairwise similarities if multiple texts
    if (texts.size() > 1) {
        std::cout << "=== Pairwise Cosine Similarities ===\n" << std::endl;
        for (size_t i = 0; i < texts.size(); i++) {
            for (size_t j = i + 1; j < texts.size(); j++) {
                float similarity = cosine_similarity(
                    embeddings[i].data(),
                    embeddings[j].data(),
                    n_embd
                );
                std::cout << "Text " << i << " <-> Text " << j << ": "
                          << std::fixed << std::setprecision(4) << similarity << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Step 10: Print performance statistics
    std::cout << "=== Performance ===\n";
    llama_perf_context_print(ctx.get());

    // Cleanup
    llama_batch_free(batch);
    std::cout << "\nCleanup complete" << std::endl;

    return 0;
}
