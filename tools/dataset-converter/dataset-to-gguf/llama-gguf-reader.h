#pragma once

#include <cstdint>            // For int64_t, uint64_t
#include <memory>             // For std::unique_ptr
#include <string>             // For std::string
#include <vector>             // For std::vector

#include "ggml.h"             // For ggml_type
#include "llama-gguf-file.h"  // For llama_gguf_file_t
#include "llama.h"            // For llama_token

// Class for reading GGUF files, providing access to metadata and tensor data.
struct llama_gguf_reader {
    // Constructor: Initializes the reader to read from the specified GGUF file.
    // path: Path to the GGUF file.
    // Throws std::runtime_error if the file cannot be opened or context cannot be initialized.
    llama_gguf_reader(const std::string & path);

    // Destructor.
    ~llama_gguf_reader() = default;

    // Checks if the reader has been successfully initialized.
    bool llama_gguf_reader_is_initialized(void) const;

    // Gets a string metadata value by key.
    std::string llama_gguf_reader_get_metadata_str(const std::string & key,
                                                   const std::string & default_value = "") const;

    // Gets a uint64_t metadata value by key.
    uint64_t llama_gguf_reader_get_metadata_u64(const std::string & key, uint64_t default_value = 0) const;

    // Gets the number of tensors in the file.
    int64_t llama_gguf_reader_get_tensor_count(void) const;

    // Gets the name of a tensor by index.
    std::string llama_gguf_reader_get_tensor_name(int64_t index) const;

    // Gets the type of a tensor by index.
    // Returns ggml_type.
    enum ggml_type llama_gguf_reader_get_tensor_type(int64_t index) const;

    // Gets the size of a tensor in bytes by index.
    size_t llama_gguf_reader_get_tensor_size(int64_t index) const;

    // Reads tensor data by index into a vector of llama_token.
    // index: Index of the tensor.
    // tokens: Vector where tokens will be read into.
    // Returns true on success, false on error (e.g., tensor not found,
    // or its type is not GGML_TYPE_I32, or size mismatch).
    bool llama_gguf_reader_read_tensor_data(int64_t index, std::vector<llama_token> & tokens) const;

  private:
    std::unique_ptr<llama_gguf_file> m_gguf_file_ptr;  // Pointer to the llama_gguf_file object
    std::string                      m_file_path;      // Path to the file from which GGUF is read
};
