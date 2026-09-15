// Unit tests for the online suffix tree used by suffix decoding
// (--spec-type ngram-suffix). The tree is a self-contained, model-free
// component, so these tests exercise it directly without loading a model.
//
// ref: Suffix Decoding, arXiv:2411.04975

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "suffix-tree.h"

#include <cassert>
#include <cstdio>

// generous factor / low probability floor so the knobs under test are the only
// thing that bounds the draft length.
static common_suffix_draft speculate(const common_suffix_tree & tree, const llama_tokens & ctx, int32_t n_max) {
    return tree.speculate(ctx, n_max, 100.0f, 0.0f);
}

// the most frequent continuation of a repeated pattern is drafted, and the
// draft wraps around the pattern for as many tokens as requested.
static void test_greedy_continuation() {
    common_suffix_tree tree(24);
    // pattern 10,20,30,40 repeated, ending mid-pattern.
    tree.extend({10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30});
    assert(tree.size() == 11);

    // context ending at 30 -> 40, then wrap to 10, 20, 30, ...
    const auto d = speculate(tree, {10, 20, 30}, 6);
    assert(d.tokens == (llama_tokens{40, 10, 20, 30, 40, 10}));
    assert(d.match_len == 3);
    assert(d.score > 0.0f);
}

// when the same context is followed by different tokens, the most frequent one
// wins (frequency-weighted greedy selection).
static void test_frequency_selection() {
    common_suffix_tree tree(24);
    // [1,2] is followed by 3 three times and by 4 once.
    tree.extend({1, 2, 3, 1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2});

    const auto d = speculate(tree, {1, 2}, 1);
    assert(!d.tokens.empty());
    assert(d.tokens[0] == 3);
}

// building the tree one token at a time must be equivalent to building it from
// the whole sequence at once.
static void test_append_equals_extend() {
    const llama_tokens seq = {1, 2, 3, 1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2};

    common_suffix_tree a(24);
    common_suffix_tree b(24);
    for (const llama_token tok : seq) {
        a.append(tok);
    }
    b.extend(seq);

    assert(a.size() == b.size());

    const llama_tokens ctx = {1, 2};
    const auto da = a.speculate(ctx, 4, 100.0f, 0.0f);
    const auto db = b.speculate(ctx, 4, 100.0f, 0.0f);
    assert(da.tokens == db.tokens);
    assert(da.match_len == db.match_len);
}

// a context that never appeared yields an empty draft.
static void test_unseen_context() {
    common_suffix_tree tree(24);
    tree.extend({10, 20, 30, 40, 10, 20, 30, 40});

    const auto d = speculate(tree, {99, 98}, 4);
    assert(d.tokens.empty());
    assert(d.match_len == 0);
}

// max_spec_tokens is a hard cap on the number of drafted tokens.
static void test_max_tokens_cap() {
    common_suffix_tree tree(64);
    // a long, perfectly repetitive sequence so drafting could run indefinitely.
    llama_tokens seq;
    for (int i = 0; i < 20; ++i) {
        seq.push_back(7);
        seq.push_back(8);
    }
    tree.extend(seq);

    const auto d = tree.speculate({7, 8}, 5, 100.0f, 0.0f);
    assert((int32_t) d.tokens.size() == 5);
}

// max_spec_factor scales the draft length by the matched context length:
// a match of length m may draft at most m * factor tokens.
static void test_spec_factor_scaling() {
    common_suffix_tree tree(64);
    llama_tokens seq;
    for (int i = 0; i < 20; ++i) {
        seq.push_back(7);
        seq.push_back(8);
    }
    tree.extend(seq);

    // match_len 2, factor 1.0 -> at most 2 tokens, even with a large n_max.
    const auto d = tree.speculate({7, 8}, 100, 1.0f, 0.0f);
    assert((int32_t) d.tokens.size() <= 2);
    assert(!d.tokens.empty());
}

// min_token_prob stops drafting once the frequency-based probability of the
// next token falls below the threshold.
static void test_min_prob_cutoff() {
    common_suffix_tree tree(24);
    // after [1], token 2 always follows, but after [1,2] the continuation
    // branches (3 twice, 4 once), so the per-step probability drops.
    tree.extend({1, 2, 3, 1, 2, 4, 1, 2, 3});

    // a probability floor of 1.0 only accepts certain (p == 1) continuations.
    const auto d = tree.speculate({1}, 8, 100.0f, 1.0f);
    // 2 is certain after 1; the branch after that is < 1.0 so drafting stops.
    assert(d.tokens == (llama_tokens{2}));
}

// ties in child frequency are broken by the smallest token id, so the draft is
// deterministic regardless of hash-map iteration order.
static void test_tie_break_determinism() {
    common_suffix_tree tree(24);
    // after [5], both 9 and 3 follow exactly once (a tie).
    tree.extend({5, 9, 5, 3});

    const auto d = tree.speculate({5}, 1, 100.0f, 0.0f);
    assert(!d.tokens.empty());
    assert(d.tokens[0] == 3); // smaller token id wins the tie
}

// reset() drops all state.
static void test_reset() {
    common_suffix_tree tree(24);
    tree.extend({5, 6, 7, 5, 6, 7});
    assert(tree.size() == 6);

    tree.reset();
    assert(tree.size() == 0);

    const auto d = speculate(tree, {5, 6}, 4);
    assert(d.tokens.empty());
}

int main() {
    test_greedy_continuation();
    test_frequency_selection();
    test_append_equals_extend();
    test_unseen_context();
    test_max_tokens_cap();
    test_spec_factor_scaling();
    test_min_prob_cutoff();
    test_tie_break_determinism();
    test_reset();

    printf("All suffix-tree tests passed.\n");
    return 0;
}
