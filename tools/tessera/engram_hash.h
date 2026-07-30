#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tessera {

struct engram_hash_spec {
    int64_t pad_id;
    std::vector<uint64_t> multipliers;
    std::vector<std::vector<int64_t>> moduli;
};

inline int64_t engram_positive_mod(uint64_t bits, int64_t modulus) {
    if (modulus <= 0) {
        throw std::invalid_argument("Engram hash modulus must be positive");
    }
    int64_t value;
    static_assert(sizeof(value) == sizeof(bits), "Engram hash requires 64-bit integers");
    std::memcpy(&value, &bits, sizeof(value));
    const int64_t remainder = value % modulus;
    return remainder < 0 ? remainder + modulus : remainder;
}

inline std::vector<int64_t> engram_hash_position(
        const int64_t * tokens,
        size_t          position,
        const engram_hash_spec & spec) {
    if (spec.multipliers.size() < 2 || spec.moduli.size() + 1 != spec.multipliers.size()) {
        throw std::invalid_argument("invalid Engram hash specification");
    }
    std::vector<int64_t> result;
    for (size_t order = 2; order <= spec.multipliers.size(); ++order) {
        uint64_t mixed = 0;
        for (size_t offset = 0; offset < order; ++offset) {
            const int64_t token = position >= offset ? tokens[position - offset] : spec.pad_id;
            const uint64_t term = static_cast<uint64_t>(token) * spec.multipliers[offset];
            mixed = offset == 0 ? term : mixed ^ term;
        }
        for (const int64_t modulus : spec.moduli[order - 2]) {
            result.push_back(engram_positive_mod(mixed, modulus));
        }
    }
    return result;
}

}
