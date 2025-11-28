#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../vendor/nlohmann/json.hpp"

// Safetensors data types
enum class safetensors_dtype {
    F64,    // float64
    F32,    // float32
    F16,    // float16
    BF16,   // bfloat16
    I64,    // int64
    I32,    // int32
    I16,    // int16
    I8,     // int8
    U8,     // uint8
    BOOL,   // bool
    UNKNOWN
};

// Convert safetensors dtype string to enum
safetensors_dtype safetensors_dtype_from_string(const std::string & dtype_str);

// Get size in bytes for a given dtype
size_t safetensors_dtype_size(safetensors_dtype dtype);

// Get dtype name
const char * safetensors_dtype_name(safetensors_dtype dtype);

// Information about a single tensor in the safetensors file
struct safetensors_tensor_info {
    std::string name;
    safetensors_dtype dtype;
    std::vector<int64_t> shape;
    size_t offset_start;    // offset in data buffer (not file position)
    size_t offset_end;      // end offset in data buffer

    size_t size() const {
        return offset_end - offset_start;
    }

    int64_t n_elements() const {
        int64_t n = 1;
        for (auto dim : shape) {
            n *= dim;
        }
        return n;
    }
};

// Represents a safetensors file (single file or one shard)
class safetensors_file {
public:
    safetensors_file() = default;
    ~safetensors_file() = default;

    // Open and parse a safetensors file
    // Returns true on success, false on error (check get_error())
    bool open(const std::string & filename);

    // Close the file
    void close();

    // Get list of all tensor names
    std::vector<std::string> get_tensor_names() const;

    // Get information about a specific tensor
    // Returns nullptr if tensor not found
    const safetensors_tensor_info * get_tensor_info(const std::string & name) const;

    // Read tensor data into a pre-allocated buffer
    // Buffer must be at least tensor_info->size() bytes
    // Returns true on success
    bool read_tensor_data(const std::string & name, void * buffer, size_t buffer_size);

    // Get metadata (optional __metadata__ field)
    const nlohmann::json * get_metadata() const;

    // Get last error message
    const std::string & get_error() const { return error_msg; }

    // Get file size
    size_t get_file_size() const { return file_size; }

    // Get data buffer offset (where tensor data starts in file)
    size_t get_data_offset() const { return data_start_offset; }

private:
    std::string filename;
    FILE * file = nullptr;
    size_t file_size = 0;
    size_t data_start_offset = 0;
    std::string error_msg;

    std::map<std::string, safetensors_tensor_info> tensors;
    std::unique_ptr<nlohmann::json> metadata;

    bool parse_header();
};

// Represents a collection of safetensors files (for sharded models)
class safetensors_loader {
public:
    safetensors_loader() = default;
    ~safetensors_loader() = default;

    // Load a single safetensors file
    bool load_single(const std::string & filename);

    // Load sharded model using index.json
    // index_path should be "model.safetensors.index.json"
    // base_dir is the directory containing the shard files
    bool load_sharded(const std::string & index_path, const std::string & base_dir);

    // Get list of all tensor names across all shards
    std::vector<std::string> get_tensor_names() const;

    // Get information about a specific tensor
    const safetensors_tensor_info * get_tensor_info(const std::string & name) const;

    // Read tensor data (handles finding the right shard)
    bool read_tensor_data(const std::string & name, void * buffer, size_t buffer_size);

    // Get last error message
    const std::string & get_error() const { return error_msg; }

    // Get total number of tensors
    size_t get_tensor_count() const { return tensor_to_file.size(); }

private:
    std::vector<std::unique_ptr<safetensors_file>> files;
    std::map<std::string, size_t> tensor_to_file;  // maps tensor name to file index
    std::string error_msg;
};
