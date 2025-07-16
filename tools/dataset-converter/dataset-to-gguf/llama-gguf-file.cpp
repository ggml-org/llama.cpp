#include <iostream>
#include <stdexcept>
#include <vector>

#include "llama-gguf-file.h"

// Default constructor: Initializes an empty GGUF context for writing.
llama_gguf_file::llama_gguf_file() : m_ctx(nullptr) {
    m_ctx = gguf_init_empty();
    if (!m_ctx) {
        throw std::runtime_error("Failed to initialize empty GGUF context.");
    }
}

// Constructor: Initializes a GGUF context from an existing file for reading.
// path: Path to the GGUF file to open.
llama_gguf_file::llama_gguf_file(const std::string & path) : m_ctx(nullptr) {
    struct gguf_init_params params = {};
    // When reading, we do NOT want gguf_init_from_file to allocate a ggml_context
    // for tensors, as we will manage data reading manually using file offsets.
    params.no_alloc = true;  // Ensure no allocation for tensor data by gguf_init_from_file
    m_ctx = gguf_init_from_file(path.c_str(), params);
    if (!m_ctx) {
        throw std::runtime_error("Failed to initialize GGUF context from file: " + path);
    }
}

// Destructor: Frees the GGUF context.
llama_gguf_file::~llama_gguf_file() {
    if (m_ctx) {
        gguf_free(m_ctx);
        m_ctx = nullptr;
    }
}

// Checks if the GGUF context is initialized.
// Returns true if initialized, false otherwise.
bool llama_gguf_file::llama_gguf_file_is_initialized() const {
    return m_ctx != nullptr;
}

// --- Methods for working with metadata (KV-pairs) ---

// Sets a string value for a given key.
void llama_gguf_file::llama_gguf_file_set_val_str(const std::string & key, const std::string & value) {
    if (!m_ctx) {
        throw std::runtime_error("GGUF context not initialized.");
    }
    gguf_set_val_str(m_ctx, key.c_str(), value.c_str());
}

// Sets a uint64_t value for a given key.
void llama_gguf_file::llama_gguf_file_set_val_u64(const std::string & key, uint64_t value) {
    if (!m_ctx) {
        throw std::runtime_error("GGUF context not initialized.");
    }
    gguf_set_val_u64(m_ctx, key.c_str(), value);
}

// Sets an array of strings for a given key.
void llama_gguf_file::llama_gguf_file_set_arr_str(const std::string & key, const std::vector<const char *> & values) {
    if (!m_ctx) {
        throw std::runtime_error("GGUF context not initialized.");
    }
    gguf_set_arr_str(m_ctx, key.c_str(), const_cast<const char **>(values.data()), values.size());
}

// Sets an array of data of a specified GGUF type for a given key.
void llama_gguf_file::llama_gguf_file_set_arr_data(const std::string & key, gguf_type type, const void * data,
                                                   size_t n) {
    if (!m_ctx) {
        throw std::runtime_error("GGUF context not initialized.");
    }
    gguf_set_arr_data(m_ctx, key.c_str(), type, data, n);
}

// Gets a string value by key.
std::string llama_gguf_file::llama_gguf_file_get_val_str(const std::string & key,
                                                         const std::string & defaultValue) const {
    if (!m_ctx) {
        return defaultValue;
    }
    int64_t key_id = llama_gguf_file_find_key(key);
    if (key_id == -1 || gguf_get_kv_type(m_ctx, key_id) != GGUF_TYPE_STRING) {
        return defaultValue;
    }
    return gguf_get_val_str(m_ctx, key_id);
}

// Gets a uint64_t value by key.
uint64_t llama_gguf_file::llama_gguf_file_get_val_u64(const std::string & key, uint64_t defaultValue) const {
    if (!m_ctx) {
        return defaultValue;
    }
    int64_t key_id = llama_gguf_file_find_key(key);
    if (key_id == -1 || gguf_get_kv_type(m_ctx, key_id) != GGUF_TYPE_UINT64) {
        return defaultValue;
    }
    return gguf_get_val_u64(m_ctx, key_id);
}

// --- Methods for working with tensors ---

// Adds a ggml_tensor to the GGUF context.
void llama_gguf_file::llama_gguf_file_add_tensor(struct ggml_tensor * tensor) {
    if (!m_ctx) {
        throw std::runtime_error("GGUF context not initialized.");
    }
    gguf_add_tensor(m_ctx, tensor);
}

// Sets the data for a tensor by its name.
void llama_gguf_file::llama_gguf_file_set_tensor_data(const std::string & name, const void * data) {
    if (!m_ctx) {
        throw std::runtime_error("GGUF context not initialized.");
    }
    gguf_set_tensor_data(m_ctx, name.c_str(), data);
}

// Gets the number of tensors in the GGUF file.
int64_t llama_gguf_file::llama_gguf_file_get_n_tensors() const {
    if (!m_ctx) {
        return 0;
    }
    return gguf_get_n_tensors(m_ctx);
}

std::string llama_gguf_file::llama_gguf_file_get_tensor_name(int64_t idx) const {
    return gguf_get_tensor_name(m_ctx, idx);
}

enum ggml_type llama_gguf_file::llama_gguf_file_get_tensor_type(int64_t idx) const {
    return gguf_get_tensor_type(m_ctx, idx);
}

size_t llama_gguf_file::llama_gguf_file_get_tensor_size(int64_t idx) const {
    return gguf_get_tensor_size(m_ctx, idx);
}

// Reads tensor data into a vector of llama_token.
// This is specific for sequence tensors (GGML_TYPE_I32).
bool llama_gguf_file::llama_gguf_file_write_to_file(const std::string & output_path, bool only_meta) {
    if (!m_ctx) {
        std::cerr << "Error: GGUF context is not initialized. Cannot write to file." << std::endl;
        return false;
    }
    if (!gguf_write_to_file(m_ctx, output_path.c_str(), only_meta)) {
        std::cerr << "Error: Failed to write GGUF file to " << output_path << std::endl;
        return false;
    }
    return true;
}

struct gguf_context * llama_gguf_file::get_gguf_context() {
    return m_ctx;
}

int64_t llama_gguf_file::llama_gguf_file_find_key(const std::string & key) const {
    if (!m_ctx) {
        return -1;
    }
    return gguf_find_key(m_ctx, key.c_str());
}
