#include "llama-safetensors.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include "../vendor/nlohmann/json.hpp"

using json = nlohmann::json;

// RAII file handle wrapper
struct file_handle {
    FILE * file = nullptr;

    file_handle(const char * filename, const char * mode) {
        file = fopen(filename, mode);
    }

    ~file_handle() {
        if (file) {
            fclose(file);
        }
    }

    operator FILE*() { return file; }
    operator bool() const { return file != nullptr; }
};

safetensors_dtype safetensors_dtype_from_string(const std::string & dtype_str) {
    if (dtype_str == "F64") return safetensors_dtype::F64;
    if (dtype_str == "F32") return safetensors_dtype::F32;
    if (dtype_str == "F16") return safetensors_dtype::F16;
    if (dtype_str == "BF16") return safetensors_dtype::BF16;
    if (dtype_str == "I64") return safetensors_dtype::I64;
    if (dtype_str == "I32") return safetensors_dtype::I32;
    if (dtype_str == "I16") return safetensors_dtype::I16;
    if (dtype_str == "I8") return safetensors_dtype::I8;
    if (dtype_str == "U8") return safetensors_dtype::U8;
    if (dtype_str == "BOOL") return safetensors_dtype::BOOL;
    return safetensors_dtype::UNKNOWN;
}

size_t safetensors_dtype_size(safetensors_dtype dtype) {
    switch (dtype) {
        case safetensors_dtype::F64:  return 8;
        case safetensors_dtype::F32:  return 4;
        case safetensors_dtype::F16:  return 2;
        case safetensors_dtype::BF16: return 2;
        case safetensors_dtype::I64:  return 8;
        case safetensors_dtype::I32:  return 4;
        case safetensors_dtype::I16:  return 2;
        case safetensors_dtype::I8:   return 1;
        case safetensors_dtype::U8:   return 1;
        case safetensors_dtype::BOOL: return 1;
        default: return 0;
    }
}

const char * safetensors_dtype_name(safetensors_dtype dtype) {
    switch (dtype) {
        case safetensors_dtype::F64:  return "F64";
        case safetensors_dtype::F32:  return "F32";
        case safetensors_dtype::F16:  return "F16";
        case safetensors_dtype::BF16: return "BF16";
        case safetensors_dtype::I64:  return "I64";
        case safetensors_dtype::I32:  return "I32";
        case safetensors_dtype::I16:  return "I16";
        case safetensors_dtype::I8:   return "I8";
        case safetensors_dtype::U8:   return "U8";
        case safetensors_dtype::BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

bool safetensors_file::open(const std::string & fname) {
    close();  // close any existing file

    filename = fname;
    file = fopen(filename.c_str(), "rb");
    if (!file) {
        error_msg = "Failed to open file: " + filename;
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size < 8) {
        error_msg = "File too small to be a valid safetensors file (< 8 bytes)";
        close();
        return false;
    }

    return parse_header();
}

void safetensors_file::close() {
    if (file) {
        fclose(file);
        file = nullptr;
    }
    tensors.clear();
    metadata.reset();
    error_msg.clear();
    file_size = 0;
    data_start_offset = 0;
}

bool safetensors_file::parse_header() {
    // Read 8-byte header (u64 little-endian)
    uint8_t header_bytes[8];
    if (fread(header_bytes, 1, 8, file) != 8) {
        error_msg = "Failed to read header";
        return false;
    }

    // Parse as little-endian u64
    uint64_t metadata_size = 0;
    for (int i = 0; i < 8; i++) {
        metadata_size |= (uint64_t)header_bytes[i] << (i * 8);
    }

    // Sanity check
    if (metadata_size > file_size - 8) {
        error_msg = "Invalid metadata size: " + std::to_string(metadata_size);
        return false;
    }

    if (metadata_size > 100 * 1024 * 1024) {  // 100 MB max for metadata
        error_msg = "Metadata size too large: " + std::to_string(metadata_size);
        return false;
    }

    // Read metadata JSON
    std::vector<char> metadata_bytes(metadata_size + 1);  // +1 for null terminator
    if (fread(metadata_bytes.data(), 1, metadata_size, file) != metadata_size) {
        error_msg = "Failed to read metadata";
        return false;
    }
    metadata_bytes[metadata_size] = '\0';

    // Calculate data start offset (with alignment)
    data_start_offset = 8 + metadata_size;
    constexpr size_t ALIGNMENT = 8;
    if (data_start_offset % ALIGNMENT != 0) {
        data_start_offset += ALIGNMENT - (data_start_offset % ALIGNMENT);
    }

    // Parse JSON
    json j;
    try {
        j = json::parse(metadata_bytes.data());
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to parse metadata JSON: ") + e.what();
        return false;
    }

    // Extract tensors
    for (auto & [key, value] : j.items()) {
        if (key == "__metadata__") {
            // Store optional metadata
            metadata = std::make_unique<json>(value);
            continue;
        }

        // Parse tensor info
        if (!value.is_object()) {
            continue;
        }

        safetensors_tensor_info info;
        info.name = key;

        try {
            // Get dtype
            if (!value.contains("dtype") || !value["dtype"].is_string()) {
                error_msg = "Missing or invalid dtype for tensor: " + key;
                return false;
            }
            info.dtype = safetensors_dtype_from_string(value["dtype"].get<std::string>());
            if (info.dtype == safetensors_dtype::UNKNOWN) {
                error_msg = "Unknown dtype for tensor: " + key;
                return false;
            }

            // Get shape
            if (!value.contains("shape") || !value["shape"].is_array()) {
                error_msg = "Missing or invalid shape for tensor: " + key;
                return false;
            }
            for (auto & dim : value["shape"]) {
                if (!dim.is_number_integer()) {
                    error_msg = "Invalid shape dimension for tensor: " + key;
                    return false;
                }
                info.shape.push_back(dim.get<int64_t>());
            }

            // Get data_offsets
            if (!value.contains("data_offsets") || !value["data_offsets"].is_array() ||
                value["data_offsets"].size() != 2) {
                error_msg = "Missing or invalid data_offsets for tensor: " + key;
                return false;
            }
            info.offset_start = value["data_offsets"][0].get<size_t>();
            info.offset_end = value["data_offsets"][1].get<size_t>();

            // Validate offsets
            if (info.offset_end < info.offset_start) {
                error_msg = "Invalid offsets for tensor: " + key;
                return false;
            }

            // Validate size matches shape and dtype
            size_t expected_size = info.n_elements() * safetensors_dtype_size(info.dtype);
            if (info.size() != expected_size) {
                error_msg = "Size mismatch for tensor " + key + ": expected " +
                           std::to_string(expected_size) + ", got " + std::to_string(info.size());
                return false;
            }

            // Validate offset is within file bounds
            if (data_start_offset + info.offset_end > file_size) {
                error_msg = "Tensor data extends beyond file bounds: " + key;
                return false;
            }

            tensors[key] = info;

        } catch (const std::exception & e) {
            error_msg = std::string("Error parsing tensor ") + key + ": " + e.what();
            return false;
        }
    }

    return true;
}

std::vector<std::string> safetensors_file::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors.size());
    for (const auto & [name, _] : tensors) {
        names.push_back(name);
    }
    // Sort for consistency
    std::sort(names.begin(), names.end());
    return names;
}

const safetensors_tensor_info * safetensors_file::get_tensor_info(const std::string & name) const {
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        return &it->second;
    }
    return nullptr;
}

bool safetensors_file::read_tensor_data(const std::string & name, void * buffer, size_t buffer_size) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        error_msg = "Tensor not found: " + name;
        return false;
    }

    const auto & info = it->second;
    if (buffer_size < info.size()) {
        error_msg = "Buffer too small for tensor " + name + ": need " +
                   std::to_string(info.size()) + ", got " + std::to_string(buffer_size);
        return false;
    }

    if (!file) {
        error_msg = "File not open";
        return false;
    }

    // Seek to tensor data position
    size_t file_offset = data_start_offset + info.offset_start;
    if (fseek(file, file_offset, SEEK_SET) != 0) {
        error_msg = "Failed to seek to tensor data for: " + name;
        return false;
    }

    // Read tensor data
    if (fread(buffer, 1, info.size(), file) != info.size()) {
        error_msg = "Failed to read tensor data for: " + name;
        return false;
    }

    return true;
}

const nlohmann::json * safetensors_file::get_metadata() const {
    return metadata.get();
}

// safetensors_loader implementation

bool safetensors_loader::load_single(const std::string & filename) {
    auto file = std::make_unique<safetensors_file>();
    if (!file->open(filename)) {
        error_msg = file->get_error();
        return false;
    }

    // Map all tensors to this file
    size_t file_idx = files.size();
    for (const auto & name : file->get_tensor_names()) {
        tensor_to_file[name] = file_idx;
    }

    files.push_back(std::move(file));
    return true;
}

bool safetensors_loader::load_sharded(const std::string & index_path, const std::string & base_dir) {
    // Read index.json
    std::ifstream f(index_path);
    if (!f.is_open()) {
        error_msg = "Failed to open index file: " + index_path;
        return false;
    }

    json index_json;
    try {
        f >> index_json;
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to parse index JSON: ") + e.what();
        return false;
    }

    // Get weight_map
    if (!index_json.contains("weight_map") || !index_json["weight_map"].is_object()) {
        error_msg = "Index file missing or invalid weight_map";
        return false;
    }

    const auto & weight_map = index_json["weight_map"];

    // Collect unique shard files
    std::map<std::string, size_t> shard_file_to_idx;
    for (auto & [tensor_name, shard_file] : weight_map.items()) {
        if (!shard_file.is_string()) {
            continue;
        }
        std::string shard_path = base_dir + "/" + shard_file.get<std::string>();

        // Load shard if not already loaded
        if (shard_file_to_idx.find(shard_path) == shard_file_to_idx.end()) {
            size_t idx = files.size();
            auto file = std::make_unique<safetensors_file>();
            if (!file->open(shard_path)) {
                error_msg = "Failed to load shard " + shard_path + ": " + file->get_error();
                return false;
            }
            files.push_back(std::move(file));
            shard_file_to_idx[shard_path] = idx;
        }

        tensor_to_file[tensor_name] = shard_file_to_idx[shard_path];
    }

    return true;
}

std::vector<std::string> safetensors_loader::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_to_file.size());
    for (const auto & [name, _] : tensor_to_file) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

const safetensors_tensor_info * safetensors_loader::get_tensor_info(const std::string & name) const {
    auto it = tensor_to_file.find(name);
    if (it == tensor_to_file.end()) {
        return nullptr;
    }
    return files[it->second]->get_tensor_info(name);
}

bool safetensors_loader::read_tensor_data(const std::string & name, void * buffer, size_t buffer_size) {
    auto it = tensor_to_file.find(name);
    if (it == tensor_to_file.end()) {
        error_msg = "Tensor not found: " + name;
        return false;
    }

    if (!files[it->second]->read_tensor_data(name, buffer, buffer_size)) {
        error_msg = files[it->second]->get_error();
        return false;
    }

    return true;
}
