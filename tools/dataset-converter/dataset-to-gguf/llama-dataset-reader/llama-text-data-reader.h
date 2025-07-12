#pragma once

#include <fstream>  // For std::ifstream
#include <string>   // For std::string
#include <vector>   // For std::vector

#include "llama-dataset-reader.h"
#include "llama.h"  // For llama_token and llama_model

// Implementation of DatasetReader for reading text files.
// Supports both raw text and pre-tokenized data.
struct llama_text_dataset_reader : public llama_dataset_reader {
    // Constructor.
    // model: pointer to the llama model for tokenization (can be nullptr if pre_tokenized is true).
    // max_seq_len: maximum sequence length for truncation.
    // pre_tokenized: if true, input data is already tokenized (token IDs as numbers).
    llama_text_dataset_reader(const struct llama_model * model, int32_t max_seq_len, bool pre_tokenized);

    // Destructor.
    ~llama_text_dataset_reader();

    // Opens the text file for reading.
    bool open(const std::string & path) override;

    // Reads the next sequence of tokens from the file.
    // If pre_tokenized is true, parses numbers from the string.
    // If pre_tokenized is false, tokenizes the string using llama_model.
    bool read_next_sequence(std::vector<llama_token> & tokens) override;

    // Closes the file.
    void close() override;

    // Resets the file pointer to the beginning of the file.
    bool reset() override;

    // Method to get the total number of sequences in the dataset.
    // For text files, this will be the number of lines.
    uint64_t total_sequences() const override;

  private:
    const struct llama_model * m_model;          // Model for tokenization
    int32_t                    m_max_seq_len;    // Maximum sequence length
    bool                       m_pre_tokenized;  // Flag for pre-tokenized data
    std::ifstream              m_input_file;     // File stream object
    std::string                m_file_path;      // File path for reset and total_sequences
    std::vector<llama_token>   m_tokens_buffer;  // Internal buffer for tokens
};
