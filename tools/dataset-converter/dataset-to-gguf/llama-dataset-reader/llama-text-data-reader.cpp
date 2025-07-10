#include "llama-text-data-reader.h"

#include <algorithm>  // For std::min
#include <iostream>   // For std::cerr
#include <sstream>

#include "llama.h"  // For llama_tokenize, llama_model_get_vocab

// Constructor
llama_text_dataset_reader::llama_text_dataset_reader(const struct llama_model * model, int32_t max_seq_len,
                                                     bool pre_tokenized) :
    m_model(model),
    m_max_seq_len(max_seq_len),
    m_pre_tokenized(pre_tokenized),
    m_tokens_buffer(max_seq_len) {}

// Destructor
llama_text_dataset_reader::~llama_text_dataset_reader() {
    close();
}

// Opens the text file for reading.
bool llama_text_dataset_reader::open(const std::string & path) {
    m_file_path = path;  // Store the file path
    m_input_file.open(path);
    if (!m_input_file.is_open()) {
        std::cerr << "Error: Failed to open input file " << path << std::endl;
        return false;
    }
    return true;
}

// Reads the next sequence of tokens from the file.
bool llama_text_dataset_reader::read_next_sequence(std::vector<llama_token> & tokens) {
    std::string line;
    if (!std::getline(m_input_file, line)) {
        return false;  // End of file or read error
    }

    tokens.clear();  // Clear the vector for a new sequence
    int n_tokens = 0;

    if (line.empty()) {
        // Empty line, return an empty sequence
        return true;
    }

    if (m_pre_tokenized) {
        // Pre-tokenized data mode: parse tokens from the string
        std::istringstream iss(line);
        llama_token token_id;
        while (iss >> token_id) {
            if (n_tokens < m_max_seq_len) {
                tokens.push_back(token_id);
                n_tokens++;
            } else {
                // Truncate if it exceeds m_max_seq_len
                break;
            }
        }
    } else {
        // Raw text data mode: tokenize the string
        if (!m_model) {
            std::cerr << "Error: Llama model not provided for tokenization of raw text." << std::endl;
            return false;
        }
        // Ensure the buffer is large enough
        if (m_tokens_buffer.size() < (size_t) m_max_seq_len) {
            m_tokens_buffer.resize(m_max_seq_len);
        }
        n_tokens = llama_tokenize(llama_model_get_vocab(m_model), line.c_str(), line.length(), m_tokens_buffer.data(),
                                  m_max_seq_len, false, true);
        if (n_tokens < 0) {
            std::cerr << "Error: Tokenization failed for line: " << line << std::endl;
            // Return an empty sequence in case of tokenization error
            return true;
        }
        tokens.assign(m_tokens_buffer.begin(), m_tokens_buffer.begin() + n_tokens);
    }
    return true;
}

// Closes the file.
void llama_text_dataset_reader::close() {
    if (m_input_file.is_open()) {
        m_input_file.close();
    }
}

// Resets the file pointer to the beginning of the file.
bool llama_text_dataset_reader::reset() {
    if (m_input_file.is_open()) {
        m_input_file.clear();                  // Clear any error flags (e.g., EOF)
        m_input_file.seekg(0, std::ios::beg);  // Move pointer to the beginning
        return true;
    }
    // If not open, try to open it again using the stored path
    return open(m_file_path);
}

// Method to get the total number of sequences in the dataset.
// For text files, this will be the number of lines.
// Note: This method will be slow for very large files,
// as it reads the entire file to count lines.
uint64_t llama_text_dataset_reader::total_sequences() const {
    if (m_file_path.empty()) {
        std::cerr << "Error (llama_text_dataset_reader::total_sequences): File path not set." << std::endl;
        return 0;
    }

    std::ifstream temp_file(m_file_path);
    if (!temp_file.is_open()) {
        std::cerr << "Error (llama_text_dataset_reader::total_sequences): Failed to open file '" << m_file_path
                  << "' for counting lines." << std::endl;
        return 0;
    }

    uint64_t count = 0;
    std::string line;
    while (std::getline(temp_file, line)) {
        count++;
    }
    temp_file.close();
    return count;
}
