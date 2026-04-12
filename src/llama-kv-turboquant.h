#pragma once

// TurboQuant KV cache compression — scaffolding (Sprint 1)
//
// This header declares the public API for TurboQuant KV cache
// compression. Implementation in subsequent sprints.
//
// See: docs/turboquant-kv-design.md
// Reference: https://github.com/ahb-sjsu/turboquant-pro

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace llama_kv_tq {

// Bit widths supported. Maps to GGML_TYPE_TQ_KV{2,3,4}.
enum bits {
    BITS_2 = 2,
    BITS_3 = 3,
    BITS_4 = 4,
};

// Random orthogonal rotation matrix.
// For head_dim <= 4096: full QR factorization (memory: 4 * D^2 bytes).
// For head_dim  > 4096: structured sign-flip + permutation (O(D) memory).
struct rotation_matrix {
    int  head_dim;
    bool structured;

    // QR mode (head_dim <= 4096)
    std::vector<float> Pi;     // (D, D) row-major
    std::vector<float> Pi_T;   // (D, D) row-major, precomputed transpose

    // Structured mode (head_dim > 4096)
    std::vector<int8_t>  sign_flip;  // (D,) elements in {-1, +1}
    std::vector<int32_t> perm;       // (D,) permutation
    std::vector<int32_t> inv_perm;   // (D,) inverse permutation
};

// Compressed vector: bit-packed indices + L2 norm.
// One block per quantized vector of length head_dim.
struct compressed_block {
    float                norm;     // L2 norm of original vector
    std::vector<uint8_t> indices;  // bit-packed b-bit indices
};

// Initialize rotation matrix for a given head_dim.
// Cached per (head_dim, seed) — call once per model context.
//
// TODO(sprint-2): implement
rotation_matrix init_rotation(int head_dim, uint32_t seed = 42);

// Compress a single vector of length head_dim.
//
// TODO(sprint-2): implement
compressed_block compress_vector(
    const float *           vec,
    int                     head_dim,
    bits                    b,
    const rotation_matrix & rot
);

// Decompress a single vector into out (length head_dim).
//
// TODO(sprint-2): implement
void decompress_vector(
    const compressed_block & block,
    int                      head_dim,
    bits                     b,
    const rotation_matrix &  rot,
    float *                  out
);

// Compute storage size (bytes) for a compressed vector of given head_dim.
inline std::size_t block_size_bytes(int head_dim, bits b) {
    return sizeof(float) + (head_dim * static_cast<int>(b) + 7) / 8;
}

// Compute compression ratio vs fp16 storage.
inline double compression_ratio(int head_dim, bits b) {
    const auto compressed = static_cast<double>(block_size_bytes(head_dim, b));
    const auto fp16       = static_cast<double>(head_dim * 2);
    return fp16 / compressed;
}

// Map a TurboQuant ggml_type to its bit width.
// Returns -1 if not a TurboQuant type.
//
// TODO(sprint-2): wire up after adding GGML_TYPE_TQ_KV{2,3,4} to
// ggml/include/ggml.h and registering them in ggml_type_traits.
inline int ggml_type_to_bits(ggml_type /*t*/) {
    // Will return 2/3/4 for GGML_TYPE_TQ_KV{2,3,4}, -1 otherwise.
    return -1;
}

inline bool is_turboquant_kv_type(ggml_type t) {
    return ggml_type_to_bits(t) > 0;
}

}  // namespace llama_kv_tq
