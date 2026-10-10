#pragma once
//
// common/suffix-tree.h: online path-compressed suffix tree for suffix decoding
//
// Suffix decoding is a model-free (self-speculative) drafting method: it builds
// a suffix tree over the tokens seen so far (prompt + previously generated
// tokens) and, given the current context, proposes the most frequent
// continuation that followed the same context earlier in the history.
//
// The tree is path-compressed (each node covers a run of tokens) and built
// online in amortized O(1) per appended token, truncated to a maximum depth.
// Only a single token sequence is tracked per tree and the corpus only grows,
// so no sequence removal / eviction is needed.
//
// ref: Suffix Decoding, arXiv:2411.04975
//

#include "llama.h"
#include "common.h"

#include <deque>
#include <memory>
#include <unordered_map>

// a single node of the suffix tree.
struct common_suffix_node {
    // number of suffixes from the root that end at or pass through this node.
    // this is the frequency signal used to score continuations.
    int64_t count = 0;

    // the node covers seq[ref_idx .. ref_idx + length) (path compression).
    // token is the first token of that run (also the key in the parent's map).
    llama_token token   = 0;
    int32_t     length  = 0;
    int32_t     ref_idx = -1;

    common_suffix_node * parent = nullptr;

    // children keyed by their first token.
    std::unordered_map<llama_token, std::unique_ptr<common_suffix_node>> children;

    common_suffix_node() = default;
    common_suffix_node(int64_t count, llama_token token, int32_t length, int32_t ref_idx)
        : count(count), token(token), length(length), ref_idx(ref_idx) {}
};

// the flat (linear) draft produced by a speculation query.
struct common_suffix_draft {
    llama_tokens tokens;         // the drafted continuation
    float        score     = 0;  // sum of per-token probabilities
    int32_t      match_len = 0;  // length of the context prefix that matched
};

// online path-compressed suffix tree over a single, append-only token sequence.
struct common_suffix_tree {
    explicit common_suffix_tree(int32_t max_depth);

    // append a single token / a run of tokens to the sequence.
    void append(llama_token token);
    void extend(const llama_tokens & tokens);

    // drop all tokens and start over (e.g. at the beginning of a new generation).
    void reset();

    // number of tokens currently stored in the tree.
    size_t size() const { return seq.size(); }

    // given a context, greedily draft the most frequent continuation.
    //   max_spec_tokens: hard cap on the number of drafted tokens.
    //   max_spec_factor: draft up to match_len * max_spec_factor tokens, so that
    //                    longer context matches are allowed to speculate deeper.
    //   min_token_prob:  stop drafting once the frequency-based probability of
    //                    the next token drops below this threshold.
    common_suffix_draft speculate(
            const llama_tokens & context,
            int32_t max_spec_tokens,
            float   max_spec_factor,
            float   min_token_prob) const;

private:
    int32_t max_depth;

    std::unique_ptr<common_suffix_node> root;

    // the full token sequence; node runs reference slices of this vector.
    llama_tokens seq;

    // sliding window of at most max_depth active insertion points, one per
    // still-growing suffix. shifted forward as tokens are appended.
    std::deque<common_suffix_node *> active;

    // walk the tree from the root along context, returning the node and the
    // offset within it where the match ends (or {nullptr, -1} on mismatch).
    std::pair<common_suffix_node *, int32_t> match_context(const llama_token * ctx, int32_t n) const;

    // highest-count child of a node (nullptr if it has no children).
    static common_suffix_node * best_child(const common_suffix_node * node);
};
