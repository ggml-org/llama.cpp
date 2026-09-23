#pragma once

#include "ggml.h"
#include "llama-mmap.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_lazy_reader {
    // path holds the table, whose row 0 starts at file offset offs
    llama_lazy_reader(const llama_file & source, size_t offs, enum ggml_type type,
                      int64_t row_elems, int64_t n_rows, int n_readers);

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader();

    // fill dst with the n gathered rows, dequantized to F32; thread-safe
    void gather(const int32_t * rows, int64_t n, float * dst) const;
    std::unique_ptr<llama_lazy_reader> clone(int n_readers) const;

    int64_t n_rows()    const { return nrows; }
    int64_t row_elems() const { return relems; }
    size_t  row_size()  const { return rsize;  }
    int     n_readers() const { return (int) files.size(); }

private:
    // read the rows of pairs[begin, end) through files[fi], writing each to its slot
    void read_range(const std::pair<int32_t, int32_t> * pairs, int64_t begin, int64_t end,
                    size_t fi, float * dst) const;

    // one buffered file per reader thread: read_at is not thread-safe, and the loader's own descriptor may be direct I/O
    std::vector<std::unique_ptr<llama_file>> files;
    const std::string path;
    const enum ggml_type type;

    const size_t   offs;   // file offset of row 0
    const size_t   rsize;  // bytes per stored row
    const int64_t  relems; // F32 elements per row
    const int64_t  nrows;

    ggml_to_float_t to_float; // the dequantizer the ggml_get_rows CPU kernel uses; null for F32
};
