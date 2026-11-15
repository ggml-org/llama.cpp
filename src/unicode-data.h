#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

extern const std::unordered_map<uint32_t, uint8_t> unicode_combining_class;
extern const std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> unicode_decomposition;
extern const std::unordered_map<uint64_t, uint32_t> unicode_composition;

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase;
extern const std::initializer_list<range_nfd> unicode_ranges_nfd;
