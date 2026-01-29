#pragma once

#include <cstdint>
#include <vector>
#include <memory>

#define COMMON_NGRAM_MOD_MAX_CHOICES 4

struct common_ngram_mod_entry {
    uint16_t head = 0;
    uint16_t n_choices = 0;

    int32_t choices[COMMON_NGRAM_MOD_MAX_CHOICES];
};

// basic n-gram hasher
struct common_ngram_mod {
    common_ngram_mod(uint16_t n, uint64_t size);

    uint64_t idx(const int32_t * tokens) const;

    void    add(const int32_t * tokens);
    int32_t get(const int32_t * tokens, int32_t offs) const; // return -1 if not found

    uint16_t n; // ngram size to hash

    std::vector<common_ngram_mod_entry> entries;
};

using common_ngram_mod_ptr = std::unique_ptr<common_ngram_mod>;
