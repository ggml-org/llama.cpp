#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

// header-only utilities to showcase how to directly load a model from memory
#include "uint8-buff-stream-wrapper.h"

namespace {
bool is_split_file(const char * const model_path) {
    if (!model_path) {
        fprintf(stderr, "No model file provided\n");
        exit(EXIT_FAILURE);
    }

    std::string path(model_path);
    return path.find("-of-") != std::string::npos;
}

std::vector<uint8_t> load_file_into_buffer(const char * const model_path) {
    std::ifstream file_stream(model_path, std::ios::binary | std::ios::ate);
    if (!file_stream) {
        fprintf(stderr, "Failed to open file %s for reading into streambuf\n", model_path);
        exit(EXIT_FAILURE);
    }

    const size_t file_size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);

    static_assert(sizeof(std::uint8_t) == sizeof(char), "uint8_t must be same size as char");
    std::vector<std::uint8_t> buffer(file_size);
    if (!file_stream.read((char *) buffer.data(), file_size)) {
        fprintf(stderr, "Failed to read entire file into buffer\n");
        exit(EXIT_FAILURE);
    }

    return buffer;
}

std::unique_ptr<std::basic_streambuf<uint8_t>> load_file_into_streambuf(const char * const model_path) {
    return std::make_unique<Uint8BufferStreamBuf>(load_file_into_buffer(model_path));
}

struct file_entry {
    std::string                                    path;
    std::unique_ptr<std::basic_streambuf<uint8_t>> streambuf;
};

std::vector<file_entry> load_files_into_streambuf(const char * const model_path) {
    std::vector<file_entry> files;

    // Extract pattern from first file path
    std::string path(model_path);

    // Split by '-'
    std::vector<std::string> parts;
    std::stringstream        ss(path);
    std::string              item;
    while (std::getline(ss, item, '-')) {
        parts.push_back(item);
    }

    // Split the last part by '.'
    std::string last_part = parts.back();
    parts.pop_back();
    size_t dot_pos = last_part.find('.');
    if (dot_pos != std::string::npos) {
        parts.push_back(last_part.substr(0, dot_pos));
        parts.push_back(last_part.substr(dot_pos + 1));  // extension
    } else {
        parts.push_back(last_part);
    }

    // Check if we have enough parts
    if (parts.size() < 4) {
        fprintf(stderr, "Model path does not contain expected pattern\n");
        exit(EXIT_FAILURE);
    }

    // Get total files from [-2] position (before the extension)
    int total_files = std::stoi(parts[parts.size() - 2]);

    // Get base path by joining all parts except -start-of-end.gguf
    std::string base_path;
    for (size_t i = 0; i < parts.size() - 4; i++) {
        if (i > 0) {
            base_path += "-";
        }
        base_path += parts[i];
    }

    for (int i = 1; i <= total_files; i++) {
        char numbered_path[1024];
        snprintf(numbered_path, sizeof(numbered_path), "%s-%05d-of-%05d.gguf", base_path.c_str(), i, total_files);

        files.push_back({ numbered_path, load_file_into_streambuf(numbered_path) });
    }

    return files;
}

file_entry load_tensor_list_file(const char * const model_path) {
    std::string path(model_path);

    // Split by '-'
    std::vector<std::string> parts;
    std::stringstream        ss(path);
    std::string              item;
    while (std::getline(ss, item, '-')) {
        parts.push_back(item);
    }

    // Split the last part by '.'
    std::string last_part = parts.back();
    parts.pop_back();
    size_t dot_pos = last_part.find('.');
    if (dot_pos != std::string::npos) {
        parts.push_back(last_part.substr(0, dot_pos));
        parts.push_back(last_part.substr(dot_pos + 1));  // extension
    } else {
        parts.push_back(last_part);
    }

    // Check if we have enough parts
    if (parts.size() < 4) {
        fprintf(stderr, "Model path does not contain expected pattern\n");
        exit(EXIT_FAILURE);
    }

    // Get base path by joining all parts except -start-of-end.gguf
    std::string base_path;
    for (size_t i = 0; i < parts.size() - 4; i++) {
        if (i > 0) {
            base_path += "-";
        }
        base_path += parts[i];
    }

    // Construct tensor list file path
    std::string tensor_list_path = base_path + ".tensors.txt";

    printf("Loading tensor list file: %s\n", tensor_list_path.c_str());
    return { tensor_list_path, load_file_into_streambuf(tensor_list_path.c_str()) };
}

llama_model * load_model_from_memory_configuration(const char * model_path, llama_model_params & model_params) {
    llama_model *                         model;
    std::chrono::steady_clock::time_point load_start_time;
    if (getenv("LLAMA_EXAMPLE_MEMORY_BUFFER")) {
        std::vector<uint8_t> buffer = load_file_into_buffer(model_path);
        fprintf(stdout, "%s: loading model from memory buffer\n", __func__);
        load_start_time = std::chrono::steady_clock::now();
        model           = llama_model_load_from_buffer(std::move(buffer), model_params);
    } else if (getenv("LLAMA_EXAMPLE_MEMORY_BUFFER_SPLIT")) {
        file_entry              tensor_list_file = load_tensor_list_file(model_path);
        std::vector<file_entry> files            = load_files_into_streambuf(model_path);
        fprintf(stdout, "%s: loading model from %zu file streambufs\n", __func__, files.size());

        std::vector<const char *> file_paths;
        for (const auto & file : files) {
            printf("Found file %s with streambuf\n", file.path.c_str());
            file_paths.push_back(file.path.c_str());
        }

        load_start_time                 = std::chrono::steady_clock::now();
        const char * async_load_context = "test-model-load";
        std::thread  fulfill_thread([&files, &tensor_list_file, &async_load_context]() {
            const bool success = llama_model_load_fulfill_split_future(
                tensor_list_file.path.c_str(), async_load_context, std::move(tensor_list_file.streambuf));
            printf("Fulfilling tensor list file %s: %s\n", tensor_list_file.path.c_str(),
                   success ? "success" : "failure");
            if (!success) {
                exit(EXIT_FAILURE);
            }

            for (auto & file : files) {
                const bool success = llama_model_load_fulfill_split_future(file.path.c_str(), async_load_context,
                                                                            std::move(file.streambuf));
                printf("Fulfilling file %s with streambuf: %s\n", file.path.c_str(), success ? "success" : "failure");
                if (!success) {
                    exit(EXIT_FAILURE);
                }
            }
        });
        fprintf(stderr, "Loading model from splits\n");
        model = llama_model_load_from_split_futures(file_paths.data(), file_paths.size(), async_load_context,
                                                    tensor_list_file.path.c_str(), model_params);
        fulfill_thread.join();
    } else if (getenv("LLAMA_EXAMPLE_FROM_FILE")) {
        load_start_time = std::chrono::steady_clock::now();
        model           = llama_model_load_from_file(model_path, model_params);
    } else {
        return nullptr;
    }

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        exit(1);
    }
    std::chrono::steady_clock::time_point load_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double>         load_duration = load_end_time - load_start_time;
    fprintf(stdout, "%s: loading model took %f seconds\n", __func__, load_duration.count());
    return model;
}

bool memory_configuration_env_is_set() {
    return getenv("LLAMA_EXAMPLE_MEMORY_BUFFER") || getenv("LLAMA_EXAMPLE_MEMORY_BUFFER_SPLIT") ||
           getenv("LLAMA_EXAMPLE_FROM_FILE");
}
}  // namespace
