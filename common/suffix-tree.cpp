//
// common/suffix-tree.cpp: online path-compressed suffix tree for suffix decoding
//
// This is a clean-room reimplementation of the online suffix-tree construction
// used by suffix decoding (arXiv:2411.04975). It tracks a single, append-only
// token sequence, so the sequence-removal / eviction machinery and the
// count-sorted sibling lists of the original are not needed here: the highest
// frequency child is found by a small scan at speculation time instead.
//

#include "suffix-tree.h"

#include <algorithm>

common_suffix_tree::common_suffix_tree(int32_t max_depth)
    : max_depth(max_depth), root(std::make_unique<common_suffix_node>()) {
}

void common_suffix_tree::reset() {
    root = std::make_unique<common_suffix_node>();
    seq.clear();
    active.clear();
}

void common_suffix_tree::extend(const llama_tokens & tokens) {
    for (const llama_token token : tokens) {
        append(token);
    }
}

void common_suffix_tree::append(llama_token token) {
    // a fresh length-1 suffix starts at the root with every new token.
    active.push_back(root.get());
    root->count += 1;

    // keep at most max_depth active suffixes; this bounds the tree depth.
    if ((int32_t) active.size() > max_depth) {
        active.pop_front();
    }

    seq.push_back(token);
    const int32_t seq_len = (int32_t) seq.size();

    // advance every active suffix by the new token.
    for (auto & active_node : active) {
        common_suffix_node * node = active_node;

        auto it = node->children.find(token);
        common_suffix_node * child = it != node->children.end() ? it->second.get() : nullptr;

        if (child == nullptr) {
            if (node->count == 1 && node != root.get()) {
                // case 1a: the current suffix is the only one ending here, so
                // this is a leaf we can simply grow by one token.
                node->length += 1;
            } else {
                // case 1b: branch off a new length-1 leaf for the new token.
                auto new_child = std::make_unique<common_suffix_node>(1, token, 1, seq_len - 1);
                new_child->parent = node;
                active_node = new_child.get();
                node->children.emplace(token, std::move(new_child));
            }
        } else if (node->count == child->count + 1 && node != root.get()) {
            // case 2: a child exists for the new token and, since the current
            // suffix ends here, every other suffix through this node continues
            // into that single child.
            if (child->length == 1) {
                // case 2a: appending the token makes this node overlap the
                // child perfectly, so fuse the node into the child.
                const llama_token   node_key = node->token;
                common_suffix_node * parent  = node->parent;

                child->count  += 1;
                child->token   = node->token;
                child->length  = node->length + 1;
                child->ref_idx = seq_len - child->length;
                child->parent  = parent;

                // move the child up to take the node's slot in its parent; this
                // frees the old node (its only child was just moved out).
                std::unique_ptr<common_suffix_node> child_owned = std::move(node->children[token]);
                active_node = child_owned.get();
                parent->children[node_key] = std::move(child_owned);
            } else {
                // case 2b: the child is longer than one token, so grow the node
                // by one token into the child and shrink the child accordingly.
                node->length  += 1;
                node->ref_idx  = seq_len - node->length;

                child->length  -= 1;
                child->ref_idx += 1;

                const llama_token new_first = seq[child->ref_idx];
                if (new_first != token) {
                    // the child's first token changed, so rekey it.
                    std::unique_ptr<common_suffix_node> moved = std::move(node->children[token]);
                    node->children.erase(token);
                    moved->token = new_first;
                    node->children.emplace(new_first, std::move(moved));
                } else {
                    child->token = new_first;
                }
            }
        } else {
            // case 3: a child exists for the new token and the active suffix
            // simply descends into it.
            if (child->length == 1) {
                // case 3a: descend into the length-1 child.
                child->count += 1;
                active_node   = child;
            } else {
                // case 3b: split the child into a length-1 head and a tail.
                std::unique_ptr<common_suffix_node> child_owned = std::move(node->children[token]);
                node->children.erase(token);

                auto new_node = std::make_unique<common_suffix_node>(child->count, token, 1, seq_len - 1);
                new_node->parent = node;

                child_owned->length  -= 1;
                child_owned->ref_idx += 1;
                child_owned->token    = seq[child_owned->ref_idx];
                child_owned->parent   = new_node.get();

                const llama_token child_key = child_owned->token;
                new_node->children.emplace(child_key, std::move(child_owned));
                new_node->count += 1; // the active suffix now also ends here

                active_node = new_node.get();
                node->children.emplace(token, std::move(new_node));
            }
        }
    }
}

common_suffix_node * common_suffix_tree::best_child(const common_suffix_node * node) {
    common_suffix_node * best = nullptr;
    for (const auto & [token, child] : node->children) {
        // pick the most frequent child; break ties by smallest token id so the
        // draft is deterministic regardless of hash-map iteration order.
        if (best == nullptr ||
                child->count > best->count ||
                (child->count == best->count && token < best->token)) {
            best = child.get();
        }
    }
    return best;
}

std::pair<common_suffix_node *, int32_t> common_suffix_tree::match_context(const llama_token * ctx, int32_t n) const {
    common_suffix_node * node = root.get();
    int32_t idx = 0;

    for (int32_t i = 0; i < n; ++i) {
        const llama_token token = ctx[i];

        if (idx >= node->length) {
            auto it = node->children.find(token);
            if (it == node->children.end()) {
                return { nullptr, -1 };
            }
            node = it->second.get();
            idx  = 0;
        }

        if (seq[node->ref_idx + idx] != token) {
            return { nullptr, -1 };
        }
        idx++;
    }

    return { node, idx };
}

common_suffix_draft common_suffix_tree::speculate(
        const llama_tokens & context,
        int32_t max_spec_tokens,
        float   max_spec_factor,
        float   min_token_prob) const {
    common_suffix_draft best;

    const int32_t n_ctx = (int32_t) context.size();

    // try every context suffix length; a longer match is more reliable and is
    // allowed (via max_spec_factor) to speculate more tokens. the tree does not
    // contain the most recent (still-uncommitted) token, so matching the whole
    // context is meaningful here, unlike the original which self-matches it.
    for (int32_t match_len = 1; match_len <= n_ctx; ++match_len) {
        auto [node, idx] = match_context(context.data() + n_ctx - match_len, match_len);
        if (node == nullptr) {
            // if a short suffix does not match, no longer one will either.
            break;
        }

        int32_t max_tokens = std::min(max_spec_tokens, (int32_t) (match_len * max_spec_factor + 1e-6f));
        max_tokens = std::max(max_tokens, 0);

        // greedily follow the most frequent continuation (flat path draft).
        common_suffix_draft draft;
        float prob = 1.0f;
        while ((int32_t) draft.tokens.size() < max_tokens && prob >= min_token_prob) {
            if (idx < node->length) {
                draft.tokens.push_back(seq[node->ref_idx + idx]);
                draft.score += prob;
                idx++;
            } else {
                common_suffix_node * child = best_child(node);
                if (child == nullptr) {
                    break;
                }
                prob *= (float) child->count / (float) node->count;
                node  = child;
                idx   = 0;
            }
        }

        if (draft.score >= best.score) {
            best           = std::move(draft);
            best.match_len = match_len;
        }
    }

    return best;
}
