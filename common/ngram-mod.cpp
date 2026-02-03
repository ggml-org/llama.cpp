#include "ngram-mod.h"

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);
    scores.resize(size, SCORE_INIT);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
        used++;
        scores[i] = SCORE_INS;
    } else if (entries[i] != tokens[n]) {
        // a different token hashes to the same bucket
        ++collisions;
    }
    // keep existing score if entry already occupied
    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i];
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    std::fill(scores.begin(), scores.end(), SCORE_INIT);
    used = 0;
    collisions = 0;
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
    return entries.size() * sizeof(entries[0]) + scores.size() * sizeof(scores[0]);
}

size_t common_ngram_mod::index(const entry_t * tokens) const {
    return idx(tokens);
}

void common_ngram_mod::inc_score(const entry_t * tokens) {
    const size_t i = idx(tokens);
    if (scores[i] < common_ngram_mod::SCORE_MAX) {
        ++scores[i];
    }
}

void common_ngram_mod::dec_score(const entry_t * tokens) {
    const size_t i = idx(tokens);
    if (scores[i] > common_ngram_mod::SCORE_MIN) {
        --scores[i];
    }
}

void common_ngram_mod::inc_score_by_index(size_t i) {
    if (i < scores.size() && scores[i] < common_ngram_mod::SCORE_MAX) {
        ++scores[i];
    }
}

void common_ngram_mod::dec_score_by_index(size_t i) {
    if (i < scores.size() && scores[i] > common_ngram_mod::SCORE_MIN) {
        --scores[i];
    }
}

void common_ngram_mod::prune_low_score() {
    used = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (scores[i] < common_ngram_mod::SCORE_THR) {
            entries[i] = EMPTY;
            scores[i] = 0;
        } else {
            ++used;
        }
    }
}

size_t common_ngram_mod::get_collisions() const {
    return collisions;
}

size_t common_ngram_mod::get_below_thr() const {
    return count_below_thr;
}

size_t common_ngram_mod::get_at_min() const {
    return count_at_min;
}

size_t common_ngram_mod::get_at_max() const {
    return count_at_max;
}

size_t common_ngram_mod::get_at_ins() const {
    return count_at_ins;
}

void common_ngram_mod::update_score_stats() {
    // reset counters
    count_below_thr = 0;
    count_at_min   = 0;
    count_at_max   = 0;
    count_at_ins  = 0;

    for (size_t i = 0; i < scores.size(); ++i) {
        const int8_t s = scores[i];
        if (s < SCORE_THR) ++count_below_thr;
        if (s == SCORE_MIN) ++count_at_min;
        if (s == SCORE_MAX) ++count_at_max;
        if (s == SCORE_INS) ++count_at_ins;
    }
}
