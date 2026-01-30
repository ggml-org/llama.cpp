#pragma once

#include <cstdint>
#include <vector>
#include <memory>

//
// common_ngram_mod
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
};

using common_ngram_mod_ptr = std::unique_ptr<common_ngram_mod>;

//
// common_ngram_mod_ext (for experiments)
//

#define COMMON_NGRAM_MOD_MAX_CHOICES 4

struct common_ngram_mod_ext_entry {
    uint16_t head = 0;
    uint16_t n_choices = 0;

    int32_t choices[COMMON_NGRAM_MOD_MAX_CHOICES];
};

struct common_ngram_mod_ext {
    common_ngram_mod_ext(uint16_t n, size_t size);

    size_t idx(const int32_t * tokens) const;

    void    add(const int32_t * tokens);
    int32_t get(const int32_t * tokens, int32_t offs) const; // return -1 if not found

    size_t n; // ngram size to hash

    std::vector<common_ngram_mod_ext_entry> entries;
};

using common_ngram_mod_ext_ptr = std::unique_ptr<common_ngram_mod_ext>;
