#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
using namespace std;
struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const initializer_list<pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const unordered_set<uint32_t> unicode_set_whitespace;
extern const initializer_list<pair<uint32_t, uint32_t>> unicode_map_lowercase;
extern const initializer_list<pair<uint32_t, uint32_t>> unicode_map_uppercase;
extern const initializer_list<range_nfd> unicode_ranges_nfd;
