#pragma once

#include <cstdint>    // For uint64_t, int64_t
#include <stdexcept>  // For std::runtime_error
#include <string>     // For std::string
#include <vector>     // For std::vector

#include "ggml.h"     // For struct ggml_tensor
#include "gguf.h"     // For struct gguf_context, enum gguf_type
#include "llama.h"

// Opaque type for the GGUF file handler.
typedef struct llama_gguf_file llama_gguf_file_t;

// Class for encapsulating GGUF file operations.
// It manages the underlying gguf_context and provides a higher-level API
// for setting metadata, adding tensors, and reading/writing files.
struct llama_gguf_file {
    // Default constructor: Initializes an empty GGUF context for writing.
    llama_gguf_file();

    // Constructor: Initializes a GGUF context from an existing file for reading.
    // path: Path to the GGUF file to open.
    // Throws std::runtime_error if file cannot be opened or context cannot be initialized.
    llama_gguf_file(const std::string & path);

    // Destructor: Frees the GGUF context and the associated ggml_context.
    ~llama_gguf_file();

    // Checks if the GGUF context is initialized.
    // Returns true if initialized, false otherwise.
    bool llama_gguf_file_is_initialized() const;

    // --- Methods for working with metadata (KV-pairs) ---

    // Sets a string value for a given key.
    // key: The metadata key.
    // value: The string value to set.
    void llama_gguf_file_set_val_str(const std::string & key, const std::string & value);

    // Sets a uint64_t value for a given key.
    // key: The metadata key.
    // value: The uint64_t value to set.
    void llama_gguf_file_set_val_u64(const std::string & key, uint64_t value);

    // Sets an array of strings for a given key.
    // key: The metadata key.
    // values: A vector of C-style strings (const char*) to set.
    void llama_gguf_file_set_arr_str(const std::string & key, const std::vector<const char *> & values);

    // Sets an array of data of a specified GGUF type for a given key.
    // key: The metadata key.
    // type: The GGUF type of the data (e.g., GGUF_TYPE_INT32).
    // data: Pointer to the data array.
    // n: Number of elements in the data array.
    void llama_gguf_file_set_arr_data(const std::string & key, gguf_type type, const void * data, size_t n);

    // Gets a string value by key.
    // key: The metadata key.
    // default_value: The value to return if the key is not found or has a different type.
    // Returns the string value or the default_value.
    std::string llama_gguf_file_get_val_str(const std::string & key, const std::string & default_value = "") const;

    // Gets a uint64_t value by key.
    // key: The metadata key.
    // default_value: The value to return if the key is not found or has a different type.
    // Returns the uint64_t value or the default_value.
    uint64_t llama_gguf_file_get_val_u64(const std::string & key, uint64_t default_value = 0) const;

    // --- Methods for working with tensors ---

    // Adds a ggml_tensor to the GGUF context.
    // tensor: Pointer to the ggml_tensor to add.
    void llama_gguf_file_add_tensor(struct ggml_tensor * tensor);

    // Sets the data for a tensor by its name.
    // name: The name of the tensor.
    // data: Pointer to the tensor data.
    void llama_gguf_file_set_tensor_data(const std::string & name, const void * data);

    // Gets the number of tensors in the GGUF file.
    // Returns the count of tensors.
    int64_t llama_gguf_file_get_n_tensors(void) const;

    // Gets the name of a tensor by index.
    // idx: The index of the tensor.
    // Returns the tensor name or an empty string if not found.
    std::string llama_gguf_file_get_tensor_name(int64_t idx) const;

    // Gets the type of a tensor by index.
    // idx: The index of the tensor.
    // Returns the ggml_type of the tensor.
    enum ggml_type llama_gguf_file_get_tensor_type(int64_t idx) const;

    // Gets the size of a tensor in bytes by index.
    // idx: The index of the tensor.
    // Returns the size of the tensor in bytes.
    size_t llama_gguf_file_get_tensor_size(int64_t idx) const;

    // --- Methods for saving/loading the file ---

    // Writes the entire GGUF context to a file.
    // output_path: Path to the output GGUF file.
    // only_meta: If true, only metadata is written (no tensor data).
    // Returns true on success, false on error.
    bool llama_gguf_file_write_to_file(const std::string & output_path, bool only_meta);
    struct gguf_context * get_gguf_context();

  private:
    struct gguf_context * m_ctx;       // The underlying GGUF context

    // Private helper function to find a key by name.
    // key: The key name to find.
    // Returns the key ID or -1 if not found.
    int64_t llama_gguf_file_find_key(const std::string & key) const;
};
