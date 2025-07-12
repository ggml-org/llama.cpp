#include "llama-gguf-writer.h"
#include "llama-gguf-file.h" // Include llama-gguf-file.h
#include "llama.h"           // For llama_model_get_vocab, llama_vocab_n_tokens, llama_vocab_get_text, llama_model_meta_val_str

#include <cinttypes>          // For PRIu64
#include <cstdio>             // For snprintf
#include <cstring>            // For memcpy
#include <ctime>              // For time, gmtime, strftime
#include <iostream>           // For std::cerr
#include <stdexcept>          // For std::runtime_error
#include <vector>             // For std::vector

// Constructor: takes a pointer to a llama_gguf_file object
llama_gguf_writer::llama_gguf_writer(llama_gguf_file * m_gguf_file_ptr) : m_gguf_file(m_gguf_file_ptr) {
    if (!m_gguf_file) {
        throw std::runtime_error("llama_gguf_file pointer provided to llama_gguf_writer is null.");
    }
    if (!m_gguf_file->llama_gguf_file_is_initialized()) {
        throw std::runtime_error("llama_gguf_file provided to llama_gguf_writer is not initialized.");
    }
}

// Initializes the GGUF file metadata
void llama_gguf_writer::llama_gguf_writer_init_metadata(const struct llama_model * model,
                                                        const std::string & input_path, uint64_t sequence_count) {
    if (!m_gguf_file) {
        std::cerr << "Error: llama_gguf_file is not set. Cannot set metadata." << std::endl;
        return;
    }

    m_gguf_file->llama_gguf_file_set_val_str("training.format.version", "1.0");
    m_gguf_file->llama_gguf_file_set_val_str("training.dataset.name", input_path);

    // Set file creation date
    time_t now = time(0);
    char   buf[sizeof "2011-10-08T07:07:09Z"];
    strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    m_gguf_file->llama_gguf_file_set_val_str("training.file.creation_date", buf);

    // Set tokenizer information
    char arch_name_buffer[128];
    int  res = llama_model_meta_val_str(model, "general.architecture", arch_name_buffer, sizeof(arch_name_buffer));
    if (res >= 0) {
        m_gguf_file->llama_gguf_file_set_val_str("training.tokenizer.gguf.model", arch_name_buffer);
    } else {
        m_gguf_file->llama_gguf_file_set_val_str("training.tokenizer.gguf.model", "unknown");
    }

    // Set tokenizer vocabulary
    const struct llama_vocab * vocab      = llama_model_get_vocab(model);
    int                        vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<const char *>  vocab_list;
    vocab_list.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        vocab_list.push_back(llama_vocab_get_text(vocab, i));
    }
    m_gguf_file->llama_gguf_file_set_arr_str("training.tokenizer.gguf.vocab", vocab_list);

    // Set total sequence count
    m_gguf_file->llama_gguf_file_set_val_u64("training.sequence.count", sequence_count);
}

// Adds a sequence of tokens to the GGUF file as a tensor
void llama_gguf_writer::llama_gguf_writer_add_sequence_tensor(uint64_t index, const std::vector<llama_token> & tokens) {
    if (!m_gguf_file) {
        std::cerr << "Error: llama_gguf_file is not set. Cannot add sequence tensor." << std::endl;
        return;
    }

    if (tokens.empty()) {
        return;
    }

    char tensor_name[128];
    snprintf(tensor_name, sizeof(tensor_name), "training.tensor.%" PRIu64, index);

    // Allocate enough memory for a temporary ggml_context to hold the tensor
    size_t n_tokens = tokens.size();
    size_t tensor_mem_size = ggml_tensor_overhead() + n_tokens * sizeof(int32_t);

    struct ggml_init_params ggml_params = {};
    ggml_params.mem_size = tensor_mem_size;
    ggml_params.mem_buffer = nullptr;
    ggml_params.no_alloc = false;

    struct ggml_context * ggml_ctx = ggml_init(ggml_params);
    if (!ggml_ctx) {
        std::cerr << "Error: Failed to initialize ggml context for tensor " << index << std::endl;
        return;
    }

    // Create a 1D tensor of type GGML_TYPE_I32
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(tensor, tensor_name);

    // Copy token data to the tensor buffer
    memcpy(tensor->data, tokens.data(), n_tokens * sizeof(int32_t));

    // Add the tensor to the GGUF context via llama_gguf_file
    m_gguf_file->llama_gguf_file_add_tensor(tensor);

    // Set tensor data in the GGUF context via llama_gguf_file
    m_gguf_file->llama_gguf_file_set_tensor_data(tensor_name, tokens.data());

    ggml_free(ggml_ctx);  // Free the temporary ggml context
}

// Writes the entire GGUF context (metadata and tensors) to the specified file
bool llama_gguf_writer::llama_gguf_writer_write_to_file(const std::string & output_path) {
    if (!m_gguf_file) {
        std::cerr << "Error: llama_gguf_file is not set. Cannot write to file." << std::endl;
        return false;
    }
    return m_gguf_file->llama_gguf_file_write_to_file(output_path, false);
}
