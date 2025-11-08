// State save/load functions for main.cpp
// This file contains the simplified implementation using llama_state_get_data/set_data

#include "llama.h"
#include "log.h"
#include "gguf.h"
#include <vector>
#include <fstream>

// Save complete LLM state to GGUF file
// This includes: KV cache, logits, embeddings, RNG state
static bool save_llm_state_to_gguf(llama_context * ctx, const std::string & filename) {
    LOG("\nSaving LLM state to %s...\n", filename.c_str());

    // Get the size of the state
    const size_t state_size = llama_state_get_size(ctx);
    LOG("State size: %zu bytes (%.2f MB)\n", state_size, state_size / (1024.0 * 1024.0));

    // Allocate buffer and get state data
    std::vector<uint8_t> state_data(state_size);
    const size_t written = llama_state_get_data(ctx, state_data.data(), state_size);

    if (written != state_size) {
        LOG_ERR("Failed to get state data: got %zu bytes, expected %zu\n", written, state_size);
        return false;
    }

    // Create GGUF context
    struct gguf_context * gguf_ctx = gguf_init_empty();

    // Add metadata
    gguf_set_val_u32(gguf_ctx, "llm_state.version", 1);
    gguf_set_val_u64(gguf_ctx, "llm_state.size", state_size);
    gguf_set_val_str(gguf_ctx, "llm_state.type", "kv_cache_rng_logits_embeddings");

    // For GGUF, we need to add the state as a tensor
    // Create a ggml context for the tensor
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_size + 1024*1024,  // Extra space for tensor metadata
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,  // We already have the data
    };

    struct ggml_context * ggml_ctx = ggml_init(params);

    // Create a 1D tensor to hold the state data
    int64_t ne[4] = {(int64_t)state_size, 1, 1, 1};
    struct ggml_tensor * state_tensor = ggml_new_tensor(ggml_ctx, GGML_TYPE_I8, 1, ne);
    ggml_set_name(state_tensor, "llm_state_data");
    state_tensor->data = state_data.data();

    // Add tensor to GGUF
    gguf_add_tensor(gguf_ctx, state_tensor);

    // Write to file
    gguf_write_to_file(gguf_ctx, filename.c_str(), false);

    LOG("Successfully saved LLM state (%zu bytes)\n", written);

    // Cleanup
    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);

    return true;
}

// Load complete LLM state from GGUF file
static bool load_llm_state_from_gguf(llama_context * ctx, const std::string & filename) {
    LOG("\nLoading LLM state from %s...\n", filename.c_str());

    struct ggml_context * ggml_ctx = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ggml_ctx,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(filename.c_str(), params);

    if (!gguf_ctx) {
        LOG_ERR("Failed to load state file: %s\n", filename.c_str());
        return false;
    }

    // Read metadata
    const int n_kv = gguf_get_n_kv(gguf_ctx);
    uint32_t version = 0;
    uint64_t state_size = 0;

    for (int i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(gguf_ctx, i);
        const enum gguf_type type = gguf_get_kv_type(gguf_ctx, i);

        if (strcmp(key, "llm_state.version") == 0 && type == GGUF_TYPE_UINT32) {
            version = gguf_get_val_u32(gguf_ctx, i);
        } else if (strcmp(key, "llm_state.size") == 0 && type == GGUF_TYPE_UINT64) {
            state_size = gguf_get_val_u64(gguf_ctx, i);
        }
    }

    LOG("State version: %u, size: %lu bytes (%.2f MB)\n", version, state_size, state_size / (1024.0 * 1024.0));

    // Get the state tensor
    struct ggml_tensor * state_tensor = ggml_get_tensor(ggml_ctx, "llm_state_data");
    if (!state_tensor) {
        LOG_ERR("State tensor not found in file\n");
        gguf_free(gguf_ctx);
        return false;
    }

    // Set the state
    const size_t loaded = llama_state_set_data(ctx, (const uint8_t*)state_tensor->data, ggml_nbytes(state_tensor));

    if (loaded == 0) {
        LOG_ERR("Failed to set state data\n");
        gguf_free(gguf_ctx);
        return false;
    }

    LOG("Successfully loaded LLM state (%zu bytes)\n", loaded);

    gguf_free(gguf_ctx);

    return true;
}
