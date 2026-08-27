#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>

// A weight table that stays in the GGUF file. Nothing is mapped or loaded for it at
// model load; each batch reads exactly the rows it gathers, with pread, and hands
// them over dequantized.
//
// Built for the qwen4exp n-gram hash embedding (per_layer_token_embd): 320 M rows of
// 90 bytes, a third of the model's bytes, of which one token touches 16 at unrelated
// hashed offsets. The working set is a few thousand rows per batch, so the table has
// no business being resident -- on a unified-memory machine it competes with the
// weights and the KV cache for the same RAM.
struct llama_ple_disk {
    struct params {
        int32_t n_threads   = 64;         // parallel readers: random reads on NVMe need queue depth
        size_t  cache_bytes = 256u << 20; // direct-mapped cache of raw rows; 0 disables
        bool    direct_io   = true;       // O_DIRECT, so the rows never enter the page cache either
    };

    llama_ple_disk(const std::string & fname, size_t offs, ggml_type type, int64_t ne0, int64_t nrows, const params & p);
    ~llama_ple_disk();

    // dequantize rows idx[0..n) into dst, laid out [ne0, n]; aborts on an out-of-range index
    void gather(const int32_t * idx, size_t n, float * dst);

    int64_t     n_rows()   const;
    int64_t     ne0()      const;
    size_t      row_size() const;
    std::string describe() const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};
