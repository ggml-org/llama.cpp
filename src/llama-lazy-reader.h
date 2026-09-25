#pragma once

#include "ggml.h"
#include "llama-mmap.h"

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

struct llama_lazy_reader;

struct llama_lazy_reader_factory {
    void add(const ggml_tensor * tensor, const llama_file & source, size_t offs);
    bool has(const ggml_tensor * tensor) const;
    std::unique_ptr<llama_lazy_reader> create(int n_readers) const;

private:
    friend struct llama_lazy_reader;

    struct tensor_info {
        size_t file;
        size_t offs;
        size_t rsize;
        int64_t nrows;
    };

    std::map<const ggml_tensor *, tensor_info> tensors;
    std::vector<std::unique_ptr<llama_file>> sources;
};

struct llama_lazy_reader {
    llama_lazy_reader(const llama_lazy_reader_factory & factory, int n_readers);

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader();

    bool has(const ggml_tensor * tensor) const { return factory.has(tensor); }
    void gather(const ggml_tensor * tensor, const int32_t * rows, int64_t n, uint8_t * dst) const;

private:
    void read_range(const llama_lazy_reader_factory::tensor_info & info,
                    const std::pair<int32_t, int32_t> * pairs, int64_t begin, int64_t end,
                    size_t fi, uint8_t * dst) const;

    const llama_lazy_reader_factory & factory;
    std::vector<std::vector<std::unique_ptr<llama_file>>> files;
};
