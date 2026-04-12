// TurboQuant KV cache compression — stub implementation (Sprint 1)
//
// This file contains TODO stubs for the TurboQuant KV cache API.
// All functions throw runtime_error("not implemented") for now.
// Real implementation lands in Sprint 2 (CPU) and Sprint 3 (CUDA).
//
// See: docs/turboquant-kv-design.md

#include "llama-kv-turboquant.h"

#include <stdexcept>

namespace llama_kv_tq {

rotation_matrix init_rotation(int /*head_dim*/, uint32_t /*seed*/) {
    // TODO(sprint-2): implement QR decomposition for head_dim <= 4096,
    // structured sign-flip + permutation for head_dim > 4096.
    // Reference: turboquant-pro/turboquant_pro/_kv_cache.py:init_rotation
    throw std::runtime_error(
        "llama_kv_tq::init_rotation not implemented (Sprint 2)"
    );
}

compressed_block compress_vector(
    const float *           /*vec*/,
    int                     /*head_dim*/,
    bits                    /*b*/,
    const rotation_matrix & /*rot*/
) {
    // TODO(sprint-2): rotate vec, compute L2 norm, scalar quantize,
    // bit-pack indices.
    throw std::runtime_error(
        "llama_kv_tq::compress_vector not implemented (Sprint 2)"
    );
}

void decompress_vector(
    const compressed_block & /*block*/,
    int                      /*head_dim*/,
    bits                     /*b*/,
    const rotation_matrix &  /*rot*/,
    float *                  /*out*/
) {
    // TODO(sprint-2): unpack indices, look up centroids, unrotate,
    // scale by norm.
    throw std::runtime_error(
        "llama_kv_tq::decompress_vector not implemented (Sprint 2)"
    );
}

}  // namespace llama_kv_tq
