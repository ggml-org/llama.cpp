// CPUOPTI: Deterministic block hashing for token sequences

#include "llama-context-hash.h"

#include <cstring>

//
// FNV-1a hash implementation
//

static uint64_t fnv1a_feed_bytes(uint64_t state, const void * data, size_t n_bytes) {
    const uint8_t * bytes = (const uint8_t *) data;
    for (size_t i = 0; i < n_bytes; i++) {
        state ^= (uint64_t) bytes[i];
        state *= LLAMA_OPT_FNV_PRIME;
    }
    return state;
}

static uint64_t fnv1a_feed_u32(uint64_t state, uint32_t val) {
    // Feed 4 bytes in little-endian order for platform independence
    state ^= (uint64_t)(val & 0xFF);
    state *= LLAMA_OPT_FNV_PRIME;
    state ^= (uint64_t)((val >> 8) & 0xFF);
    state *= LLAMA_OPT_FNV_PRIME;
    state ^= (uint64_t)((val >> 16) & 0xFF);
    state *= LLAMA_OPT_FNV_PRIME;
    state ^= (uint64_t)((val >> 24) & 0xFF);
    state *= LLAMA_OPT_FNV_PRIME;
    return state;
}

//
// Public API
//

uint64_t llama_opt_hash_block(const llama_token * tokens, uint32_t n_tokens) {
    uint64_t h = LLAMA_OPT_FNV_OFFSET;
    for (uint32_t i = 0; i < n_tokens; i++) {
        h = fnv1a_feed_u32(h, (uint32_t) tokens[i]);
    }
    return h;
}

uint64_t llama_opt_hash_block_pos(const llama_token * tokens, uint32_t n_tokens, llama_pos pos_start) {
    uint64_t h = LLAMA_OPT_FNV_OFFSET;
    // Mix in the starting position
    h = fnv1a_feed_u32(h, (uint32_t) pos_start);
    for (uint32_t i = 0; i < n_tokens; i++) {
        h = fnv1a_feed_u32(h, (uint32_t) tokens[i]);
    }
    return h;
}

uint64_t llama_opt_hash_bytes(const void * data, size_t n_bytes) {
    return fnv1a_feed_bytes(LLAMA_OPT_FNV_OFFSET, data, n_bytes);
}

//
// Incremental hasher
//

llama_opt_hasher::llama_opt_hasher() : state(LLAMA_OPT_FNV_OFFSET) {}

void llama_opt_hasher::feed(llama_token token) {
    state = fnv1a_feed_u32(state, (uint32_t) token);
}

void llama_opt_hasher::feed(const llama_token * tokens, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        state = fnv1a_feed_u32(state, (uint32_t) tokens[i]);
    }
}

uint64_t llama_opt_hasher::finalize() const {
    return state;
}

void llama_opt_hasher::reset() {
    state = LLAMA_OPT_FNV_OFFSET;
}

//
// Block segmentation
//

std::vector<llama_opt_block_hash> llama_opt_segment_and_hash(
        const llama_token * tokens,
        uint32_t            n_tokens,
        uint32_t            block_size) {

    if (block_size == 0 || n_tokens == 0) {
        return {};
    }

    const uint32_t n_full_blocks = n_tokens / block_size;
    const uint32_t remainder     = n_tokens % block_size;
    const uint32_t n_blocks      = n_full_blocks + (remainder > 0 ? 1 : 0);

    std::vector<llama_opt_block_hash> result;
    result.reserve(n_blocks);

    for (uint32_t b = 0; b < n_full_blocks; b++) {
        const uint32_t offset = b * block_size;
        llama_opt_block_hash bh;
        bh.hash     = llama_opt_hash_block(tokens + offset, block_size);
        bh.offset   = offset;
        bh.n_tokens = block_size;
        result.push_back(bh);
    }

    if (remainder > 0) {
        const uint32_t offset = n_full_blocks * block_size;
        llama_opt_block_hash bh;
        bh.hash     = llama_opt_hash_block(tokens + offset, remainder);
        bh.offset   = offset;
        bh.n_tokens = remainder;
        result.push_back(bh);
    }

    return result;
}
