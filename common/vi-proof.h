#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Minimal utilities for the commit-and-open style "verifiable inference" demo:
// - SHA-256 hashing
// - Merkle tree commitment and inclusion paths
// - Leaf hashing for GGML tensor trace entries
// - Base64 encoding for transporting opened bytes in JSON

namespace vi_proof {

static constexpr size_t kSha256Size = 32;
using sha256_digest = std::array<uint8_t, kSha256Size>;

struct trace_entry_meta {
    std::string name;
    std::string op_desc;
    std::string type_name;
    std::array<int64_t, 4> ne{};
    size_t nbytes = 0;
};

// ---- encoding helpers ----
std::string hex_encode(const uint8_t * data, size_t n);
std::string base64_encode(const uint8_t * data, size_t n);

// ---- sha256 helpers ----
sha256_digest sha256_bytes(const void * data, size_t n);
sha256_digest sha256_concat(const sha256_digest & a, const sha256_digest & b);

// ---- merkle ----
struct merkle_tree {
    // levels[0] = leaves, levels.back()[0] = root
    std::vector<std::vector<sha256_digest>> levels;
    sha256_digest root() const;
};

merkle_tree build_merkle(std::vector<sha256_digest> leaves);
std::vector<sha256_digest> merkle_proof(const merkle_tree & mt, size_t leaf_idx);
sha256_digest merkle_recompute_root(size_t leaf_index, sha256_digest leaf, const std::vector<sha256_digest> & siblings);

// ---- leaf hashing ----
sha256_digest hash_trace_leaf(const trace_entry_meta & m, const uint8_t * bytes, size_t nbytes);

} // namespace vi_proof

