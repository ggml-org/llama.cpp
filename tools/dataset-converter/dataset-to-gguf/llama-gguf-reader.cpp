#include "llama-gguf-reader.h"  // Include the new header name

#include <fstream>              // For reading tensor data directly
#include <iostream>             // For std::cerr
#include <stdexcept>            // For std::runtime_error

// Constructor: Initializes the reader to read from the specified GGUF file.
llama_gguf_reader::llama_gguf_reader(const std::string & path) : m_gguf_file_ptr(nullptr), m_file_path(path) {
    try {
        // Initialize llama_gguf_file in read mode (with ggml_context allocation)
        m_gguf_file_ptr = std::make_unique<llama_gguf_file>(path);
    } catch (const std::runtime_error & e) {
        std::cerr << "Error: llama_gguf_reader constructor failed to initialize llama_gguf_file from path '" << path
                  << "': " << e.what() << std::endl;
        // Re-throw the exception as initialization failed
        throw;
    }
}

// Checks if the reader has been successfully initialized.
bool llama_gguf_reader::llama_gguf_reader_is_initialized(void) const {
    return m_gguf_file_ptr != nullptr && m_gguf_file_ptr->llama_gguf_file_is_initialized();
}

// Gets a string metadata value by key.
std::string llama_gguf_reader::llama_gguf_reader_get_metadata_str(const std::string & key,
                                                                  const std::string & default_value) const {
    if (!llama_gguf_reader_is_initialized()) {
        std::cerr << "Error (llama_gguf_reader::llama_gguf_reader_get_metadata_str): GGUFReader is not initialized. "
                     "Cannot retrieve metadata for key '"
                  << key << "'." << std::endl;
        return default_value;
    }
    return m_gguf_file_ptr->llama_gguf_file_get_val_str(key, default_value);
}

// Gets a uint64_t metadata value by key.
uint64_t llama_gguf_reader::llama_gguf_reader_get_metadata_u64(const std::string & key, uint64_t default_value) const {
    if (!llama_gguf_reader_is_initialized()) {
        std::cerr << "Error (llama_gguf_reader::llama_gguf_reader_get_metadata_u64): GGUFReader is not initialized. "
                     "Cannot retrieve metadata for key '"
                  << key << "'." << std::endl;
        return default_value;
    }
    return m_gguf_file_ptr->llama_gguf_file_get_val_u64(key, default_value);
}

// Gets the number of tensors in the file.
int64_t llama_gguf_reader::llama_gguf_reader_get_tensor_count(void) const {
    if (!llama_gguf_reader_is_initialized()) {
        std::cerr << "Error (llama_gguf_reader::llama_gguf_reader_get_tensor_count): GGUFReader is not initialized. "
                     "Cannot get tensor count."
                  << std::endl;
        return 0;
    }
    return m_gguf_file_ptr->llama_gguf_file_get_n_tensors();
}

// Gets the name of a tensor by index.
std::string llama_gguf_reader::llama_gguf_reader_get_tensor_name(int64_t index) const {
    if (!llama_gguf_reader_is_initialized()) {
        std::cerr << "Error (llama_gguf_reader::llama_gguf_reader_get_tensor_name): GGUFReader is not initialized. "
                     "Cannot get tensor name for index "
                  << index << "." << std::endl;
        return "";
    }
    return m_gguf_file_ptr->llama_gguf_file_get_tensor_name(index);
}

// Gets the type of a tensor by index.
enum ggml_type llama_gguf_reader::llama_gguf_reader_get_tensor_type(int64_t index) const {
    if (!llama_gguf_reader_is_initialized()) {
        std::cerr << "Error (llama_gguf_reader::llama_gguf_reader_get_tensor_type): GGUFReader is not initialized. "
                     "Cannot get tensor type for index "
                  << index << "." << std::endl;
        return GGML_TYPE_COUNT;  // Unknown type
    }
    return m_gguf_file_ptr->llama_gguf_file_get_tensor_type(index);
}

// Gets the size of a tensor in bytes by index.
size_t llama_gguf_reader::llama_gguf_reader_get_tensor_size(int64_t index) const {
    if (!llama_gguf_reader_is_initialized()) {
        std::cerr << "Error (llama_gguf_reader::llama_gguf_reader_get_tensor_size): GGUFReader is not initialized. "
                     "Cannot get tensor size for index "
                  << index << "." << std::endl;
        return 0;
    }
    return m_gguf_file_ptr->llama_gguf_file_get_tensor_size(index);
}

// Reads tensor data by index into a vector of llama_token.
bool llama_gguf_reader::llama_gguf_reader_read_tensor_data(int64_t index, std::vector<llama_token> & tokens) const {
    if (!llama_gguf_reader_is_initialized()) {
        std::cerr << "Error (GGUFReader::read_tensor_data): GGUFReader is not initialized. Cannot read tensor data."
                  << std::endl;
        return false;
    }

    struct gguf_context * ctx_internal = m_gguf_file_ptr->get_gguf_context();
    if (!ctx_internal) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Internal GGUF context is null in GGUFFile." << std::endl;
        return false;
    }

    if (index < 0 || index >= gguf_get_n_tensors(ctx_internal)) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Tensor with index " << index
                  << " not found or out of bounds." << std::endl;
        return false;
    }

    // Получаем ggml_type напрямую и сравниваем с GGML_TYPE_I32
    ggml_type tensor_ggml_type = gguf_get_tensor_type(ctx_internal, index);
    if (tensor_ggml_type != GGML_TYPE_I32) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Tensor type for '"
                  << gguf_get_tensor_name(ctx_internal, index)
                  << "' is not GGML_TYPE_I32. Actual type: " << ggml_type_name(tensor_ggml_type) << std::endl;
        return false;
    }

    size_t expected_bytes = gguf_get_tensor_size(ctx_internal, index);
    if (expected_bytes == 0) {
        // Если тензор пустой, просто возвращаем пустой вектор токенов
        tokens.clear();
        return true;
    }

    size_t num_tokens = expected_bytes / sizeof(llama_token);
    if (expected_bytes % sizeof(llama_token) != 0) {
        std::cerr << "Warning (GGUFReader::read_tensor_data): Tensor size " << expected_bytes
                  << " bytes is not a multiple of llama_token size (" << sizeof(llama_token) << " bytes) for tensor '"
                  << gguf_get_tensor_name(ctx_internal, index) << "'. Data might be corrupted." << std::endl;
    }

    tokens.resize(num_tokens);

    size_t data_offset_in_file = gguf_get_data_offset(ctx_internal) + gguf_get_tensor_offset(ctx_internal, index);

    // Открываем файл для чтения данных, используя сохраненный путь
    std::ifstream file(m_file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Could not open GGUF file '" << m_file_path
                  << "' for reading tensor data." << std::endl;
        return false;
    }

    // Seek to the calculated offset
    file.seekg(data_offset_in_file, std::ios::beg);
    if (file.fail()) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Failed to seek to offset " << data_offset_in_file
                  << " in file '" << m_file_path << "'. Stream state: good=" << file.good() << " eof=" << file.eof()
                  << " fail=" << file.fail() << " bad=" << file.bad() << std::endl;
        file.close();
        return false;
    }

    // Read the tensor data into the tokens vector
    file.read(reinterpret_cast<char *>(tokens.data()), expected_bytes);

    if (!file) {  // Check if the read operation failed or reached EOF before reading all bytes
        std::cerr << "Error (GGUFReader::read_tensor_data): Failed to read " << expected_bytes << " bytes for tensor '"
                  << gguf_get_tensor_name(ctx_internal, index) << "' from file '" << m_file_path << "'." << std::endl;
        std::cerr << "  Stream state after read: good=" << file.good() << " eof=" << file.eof()
                  << " fail=" << file.fail() << " bad=" << file.bad() << std::endl;
        std::cerr << "  Bytes actually read: " << file.gcount() << std::endl;
        file.close();
        return false;
    }
    // Verify that the number of bytes read matches the expected bytes
    if (file.gcount() != (std::streamsize) expected_bytes) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Mismatch in bytes read for tensor '"
                  << gguf_get_tensor_name(ctx_internal, index) << "'. Expected " << expected_bytes << ", but read "
                  << file.gcount() << "." << std::endl;
        file.close();
        return false;
    }

    file.close();
    return true;
}
