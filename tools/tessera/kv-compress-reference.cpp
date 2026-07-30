#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static ggml_type parse_type(const std::string & name) {
    for (int value = 0; value < GGML_TYPE_COUNT; ++value) {
        const auto type = static_cast<ggml_type>(value);
        if (name == ggml_type_name(type)) {
            return type;
        }
    }
    throw std::invalid_argument("unsupported GGML type: " + name);
}

static void hadamard(float * row, int64_t columns, int64_t block_size) {
    if (block_size == 0) {
        return;
    }
    if (block_size < 2 || (block_size & (block_size - 1)) != 0 || columns % block_size != 0) {
        throw std::invalid_argument("Hadamard block size must be a power of two that divides columns");
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(block_size));
    for (int64_t base = 0; base < columns; base += block_size) {
        for (int64_t stride = 1; stride < block_size; stride *= 2) {
            for (int64_t offset = 0; offset < block_size; offset += 2 * stride) {
                for (int64_t index = 0; index < stride; ++index) {
                    const float left  = row[base + offset + index];
                    const float right = row[base + offset + stride + index];
                    row[base + offset + index]          = left + right;
                    row[base + offset + stride + index] = left - right;
                }
            }
        }
        for (int64_t index = 0; index < block_size; ++index) {
            row[base + index] *= scale;
        }
    }
}

static std::vector<float> read_f32(const char * path, int64_t elements) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open input");
    }
    if (input.tellg() != static_cast<std::streamoff>(elements * sizeof(float))) {
        throw std::runtime_error("input byte count does not match shape");
    }
    input.seekg(0);
    std::vector<float> values(elements);
    input.read(reinterpret_cast<char *>(values.data()), elements * sizeof(float));
    if (!input) {
        throw std::runtime_error("failed to read input");
    }
    return values;
}

static void write_f32(const char * path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output");
    }
    output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    if (!output) {
        throw std::runtime_error("failed to write output");
    }
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s TYPE ROWS COLS HADAMARD INPUT_F32 OUTPUT_F32\n", argv[0]);
        return 2;
    }
    try {
        const ggml_type type = parse_type(argv[1]);
        const int64_t rows = std::stoll(argv[2]);
        const int64_t columns = std::stoll(argv[3]);
        const int64_t rotation = std::stoll(argv[4]);
        const auto * traits = ggml_get_type_traits(type);
        if (!traits->from_float_ref || !traits->to_float || rows <= 0 || columns <= 0 ||
                columns % traits->blck_size != 0) {
            throw std::invalid_argument("type or shape cannot be quantized by the reference path");
        }
        auto input = read_f32(argv[5], rows * columns);
        std::vector<float> output(input.size());
        const size_t row_bytes = ggml_row_size(type, columns);
        std::vector<uint8_t> quantized(row_bytes);
        for (int64_t row = 0; row < rows; ++row) {
            float * source = input.data() + row * columns;
            hadamard(source, columns, rotation);
            traits->from_float_ref(source, quantized.data(), columns);
            traits->to_float(quantized.data(), output.data() + row * columns, columns);
        }
        write_f32(argv[6], output);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "kv-compress-reference: %s\n", error.what());
        return 1;
    }
}
