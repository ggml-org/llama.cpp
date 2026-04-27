#include "ngram-mod.h"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

//
// common_ngram_mod
//

static bool is_pow2(size_t v) {
    return v > 0 && (v & (v - 1)) == 0;
}

static uint64_t pow_u64(uint64_t base, size_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    // Power-of-two fast path
    mask = is_pow2(size) ? (size - 1) : 0;

    // Precompute M^(n-1) for rolling hash
    m_pow_n_minus_1 = pow_u64(NGRAM_HASH_M, n > 0 ? n - 1 : 0);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*NGRAM_HASH_M + tokens[i];
    }

    res = mask ? (res & mask) : (res % entries.size());

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
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

// --- Rolling hash implementation ---

uint64_t common_ngram_mod::hash_full(const entry_t * tokens) const {
    uint64_t h = 0;
    for (size_t i = 0; i < n; ++i) {
        h = h * NGRAM_HASH_M + (uint64_t)tokens[i];
    }
    return h;
}

uint64_t common_ngram_mod::hash_roll(uint64_t h_prev, entry_t old_tok, entry_t new_tok) const {
    // Remove contribution of old_tok (leftmost), shift, add new_tok (rightmost)
    // h = t[0]*M^(n-1) + t[1]*M^(n-2) + ... + t[n-1]
    // h' = (h - old_tok * M^(n-1)) * M + new_tok
    return (h_prev - (uint64_t)old_tok * m_pow_n_minus_1) * NGRAM_HASH_M
           + (uint64_t)new_tok;
}

common_ngram_mod::entry_t common_ngram_mod::get_by_hash(uint64_t h) const {
    const size_t i = mask ? (h & mask) : (h % entries.size());
    return entries[i];
}

int common_ngram_mod::draft_rolling(const entry_t * ctx, int max_draft, entry_t * out) const {
    if (max_draft <= 0 || n == 0) return 0;

    // Full hash for the first window: ctx[0..n-1]
    uint64_t h = hash_full(ctx);
    entry_t tok = get_by_hash(h);
    if (tok == EMPTY) return 0;
    out[0] = tok;

    // Rolling hash for subsequent windows
    // Window i: ctx[i..i+n-1], where ctx[n+i-1] = out[i-1] (previous drafted token)
    // We store drafted tokens into out[], but the ctx array must also contain them
    // at positions ctx[n], ctx[n+1], ... for the rolling hash to reference.
    //
    // Caller must provide ctx with enough space: ctx[0..n-1+max_draft]
    // and ctx[n+i] will be filled by this function.
    //
    // Actually — caller passes ctx as mutable via const_cast or by design.
    // For safety, we use out[] and reconstruct: the "new_tok" entering the window
    // at position i is out[i-1], and the "old_tok" leaving is ctx[i-1].

    int drafted = 1;
    for (int i = 1; i < max_draft; ++i) {
        entry_t old_tok = ctx[i - 1];
        entry_t new_tok = out[i - 1];

        h = hash_roll(h, old_tok, new_tok);

        // Prefetch the NEXT entry while we process current result
        if (i + 1 < max_draft) {
            // Speculative prefetch: assume out[i] will be valid, pre-hash next
            const size_t next_idx = mask ? (h & mask) : (h % entries.size());
#if defined(_MSC_VER) && !defined(__clang__)
            _mm_prefetch((const char *)&entries[next_idx], _MM_HINT_T0);
#else
            __builtin_prefetch(&entries[next_idx], 0, 1);
#endif
        }

        tok = get_by_hash(h);
        if (tok == EMPTY) break;
        out[i] = tok;
        drafted++;
    }

    return drafted;
}
