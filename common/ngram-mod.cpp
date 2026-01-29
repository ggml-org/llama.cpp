#include "ngram-mod.h"

common_ngram_mod::common_ngram_mod(uint16_t n, uint64_t size) : n(n) {
    entries.resize(size);
}

uint64_t common_ngram_mod::idx(const int32_t * tokens) const {
    uint64_t res = 0;

    for (uint64_t i = 0; i < n; ++i) {
        res = (res * 6364136223846793005ULL + tokens[i]);
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const int32_t * tokens) {
    const uint64_t i = idx(tokens);

    common_ngram_mod_entry & entry = entries[i];

    if (entry.n_choices < COMMON_NGRAM_MOD_MAX_CHOICES) {
        entry.n_choices++;
    }

    entry.choices[entry.head] = tokens[n];
    entry.head = (entry.head + 1) % COMMON_NGRAM_MOD_MAX_CHOICES;
}

int32_t common_ngram_mod::get(const int32_t * tokens, int32_t offs) const {
    const uint64_t i = idx(tokens);

    const common_ngram_mod_entry & entry = entries[i];

    if (entry.n_choices == 0) {
        return -1;
    }

    const int32_t k = (offs + entry.head) % entry.n_choices;

    return entry.choices[k];
}
