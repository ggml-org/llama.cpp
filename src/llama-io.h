#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_io_tensor_conversion {
    ggml_type type_src = GGML_TYPE_COUNT; // source type, COUNT means no conversion
    uint32_t n_rot = 0; // Hadamard block size, 0 means no rotation
};

class llama_io_tensor_converter {
public:
    // offset is in destination bytes; size is in source bytes
    void set_tensor(ggml_tensor * tensor, const void * data, size_t offset, size_t size, llama_io_tensor_conversion conversion = {});

private:
    std::vector<uint8_t> data_src;
    std::vector<float> data_f32;
    std::vector<uint8_t> data_dst;
};

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    // offset is in destination bytes; size is in source bytes
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size, llama_io_tensor_conversion conversion = {}) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
