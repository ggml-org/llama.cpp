#pragma once

#include <string>  // For std::string
#include <vector>  // For std::vector

#include "llama-gguf-file.h"
#include "llama.h"  // For llama_token

// Class for encapsulating GGUF file writing logic.
// It now uses llama_gguf_file for low-level operations.
struct llama_gguf_writer {
    // Constructor, takes a pointer to a llama_gguf_file object.
    // m_gguf_file: pointer to an initialized llama_gguf_file object,
    //              which will be used for writing.
    llama_gguf_writer(llama_gguf_file * m_gguf_file);

    // Destructor (does not free m_gguf_file, as it is managed externally).
    ~llama_gguf_writer() = default;

    // Initializes the GGUF file metadata.
    // model: pointer to the loaded llama model to get tokenizer information.
    // input_path: path to the input file, used for the dataset name.
    // sequence_count: total number of sequences.
    void llama_gguf_writer_init_metadata(const struct llama_model * model, const std::string & input_path,
                                         uint64_t sequence_count);

    // Adds a sequence of tokens to the GGUF file as a tensor.
    // index: sequence index (used for tensor name).
    // tokens: vector of tokens representing the sequence.
    void llama_gguf_writer_add_sequence_tensor(uint64_t index, const std::vector<llama_token> & tokens);

    // Writes the entire GGUF context (metadata and tensors) to the specified file.
    // output_path: path to the output GGUF file.
    // Returns true on success, false on error.
    bool llama_gguf_writer_write_to_file(const std::string & output_path);

  private:
    llama_gguf_file * m_gguf_file;  // Pointer to the llama_gguf_file object
};
