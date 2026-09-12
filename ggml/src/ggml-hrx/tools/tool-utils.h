#pragma once

#include "../hrx-interop-utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace ggml::hrx::tool {

inline std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

inline void write_file(const std::filesystem::path & path, const void * data, size_t size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create " + path.string());
    }
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

inline void write_file(const std::filesystem::path & path, const std::string & contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create " + path.string());
    }
    output << contents;
    if (contents.empty() || contents.back() != '\n') {
        output << '\n';
    }
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

inline void check_status(hrx_status_t status, const std::string & operation) {
    if (ErrorResult error = take_status(status)) {
        throw std::runtime_error(operation + ": " + *error);
    }
}

inline bool report_status(hrx_status_t status, const std::string & operation) {
    if (ErrorResult error = take_status(status)) {
        std::cerr << operation << ": " << *error << '\n';
        return false;
    }
    return true;
}

}  // namespace ggml::hrx::tool
