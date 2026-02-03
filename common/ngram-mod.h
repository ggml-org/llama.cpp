#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    static constexpr int8_t SCORE_INIT = 0;
    static constexpr int8_t SCORE_MIN  = -5;
    static constexpr int8_t SCORE_MAX  = 20;
    static constexpr int8_t SCORE_THR  = 0; // keep equal or lower than SCORE_INIT
    static constexpr int8_t SCORE_INS  = 3;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    // expose the hash index for external bookkeeping
    size_t index(const entry_t * tokens) const;

    // score handling
    void inc_score(const entry_t * tokens);
    void dec_score(const entry_t * tokens);
    void inc_score_by_index(size_t i);
    void dec_score_by_index(size_t i);
    void prune_low_score(); // remove entries below SCORE_THR

    size_t get_n()    const;
    size_t get_used() const;

    void update_score_stats();

    size_t get_collisions() const;
    size_t get_below_thr()  const;
    size_t get_at_min()     const;
    size_t get_at_max()     const;
    size_t get_at_ins()    const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
    // per-entry score, range SCORE_MIN .. SCORE_MAX
    std::vector<int8_t> scores;

    // stats
    // count of hash collisions
    size_t collisions = 0;
    // counts for score
    size_t count_below_thr = 0;
    size_t count_at_min   = 0;
    size_t count_at_max   = 0;
    size_t count_at_ins  = 0;
};
