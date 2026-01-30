#include "ngram-mod.h"

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    std::fill(entries.begin(), entries.end(), EMPTY);
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = (res * 6364136223846793005ULL + tokens[i]);
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] != EMPTY) {
        used++;
    }

    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i];
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(entries[0]);
}

//
// common_ngram_mod_ext
//

common_ngram_mod_ext::common_ngram_mod_ext(uint16_t n, size_t size) : n(n) {
    entries.resize(size);
}

size_t common_ngram_mod_ext::idx(const int32_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = (res * 6364136223846793005ULL + tokens[i]);
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod_ext::add(const int32_t * tokens) {
    const size_t i = idx(tokens);

    common_ngram_mod_ext_entry & entry = entries[i];

    if (entry.n_choices < COMMON_NGRAM_MOD_MAX_CHOICES) {
        entry.n_choices++;
    }

    entry.choices[entry.head] = tokens[n];
    entry.head = (entry.head + 1) % COMMON_NGRAM_MOD_MAX_CHOICES;
}

int32_t common_ngram_mod_ext::get(const int32_t * tokens, int32_t offs) const {
    const size_t i = idx(tokens);

    const common_ngram_mod_ext_entry & entry = entries[i];

    if (entry.n_choices == 0) {
        return -1;
    }

    const int32_t k = (offs + entry.head) % entry.n_choices;

    return entry.choices[k];
}
