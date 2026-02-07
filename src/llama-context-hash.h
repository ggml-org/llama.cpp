#pragma once

// CPUOPTI: Deterministic block hashing for token sequences
// Uses FNV-1a (64-bit) — zero dependencies, deterministic, platform-independent

#include "llama.h"

#include <cstdint>
#include <vector>

// FNV-1a constants
static constexpr uint64_t LLAMA_OPT_FNV_OFFSET = 14695981039346656037ULL;
static constexpr uint64_t LLAMA_OPT_FNV_PRIME  = 1099511628211ULL;

//
// Hash a block of tokens (position-independent)
//
uint64_t llama_opt_hash_block(const llama_token * tokens, uint32_t n_tokens);

//
// Hash a block of tokens with positional context
//
uint64_t llama_opt_hash_block_pos(const llama_token * tokens, uint32_t n_tokens, llama_pos pos_start);

//
// Hash raw bytes (for tensor data hashing)
//
uint64_t llama_opt_hash_bytes(const void * data, size_t n_bytes);

//
// Incremental hasher for streaming token feeds
//
struct llama_opt_hasher {
    uint64_t state;

    llama_opt_hasher();

    void     feed(llama_token token);
    void     feed(const llama_token * tokens, uint32_t n);
    uint64_t finalize() const;
    void     reset();
};

//
// Block segmentation: split a token sequence into fixed-size blocks and hash each
//
struct llama_opt_block_hash {
    uint64_t hash;
    uint32_t offset;    // Start index in the token sequence
    uint32_t n_tokens;  // Number of tokens in this block (may be < block_size for the last block)
};

// Segment a token sequence into blocks and compute hashes
// Returns a vector of (hash, offset, n_tokens) for each block
std::vector<llama_opt_block_hash> llama_opt_segment_and_hash(
    const llama_token * tokens,
    uint32_t            n_tokens,
    uint32_t            block_size);
