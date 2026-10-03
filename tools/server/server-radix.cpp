#include "server-radix.h"

#include "ggml.h"

#include <algorithm>
#include <climits>

server_radix_cache::server_radix_cache(size_t page_size) : page_size(std::max<size_t>(1, page_size)) {}

size_t server_radix_cache::align_len(size_t n) const {
    if (page_size <= 1) {
        return n;
    }
    return (n / page_size) * page_size;
}

size_t server_radix_cache::n_nodes_helper(const server_radix_node & node) const {
    size_t n = node.children.size();
    for (const auto & kv : node.children) {
        n += n_nodes_helper(*kv.second);
    }
    return n;
}

size_t server_radix_cache::n_nodes() const {
    return n_nodes_helper(root);
}

static size_t edge_lcp(const std::vector<llama_token> & a, const llama_tokens & b, size_t b_off) {
    const size_t n = std::min(a.size(), b.size() - b_off);
    size_t i = 0;
    for (; i < n; ++i) {
        if (a[i] != b[b_off + i]) {
            break;
        }
    }
    return i;
}

size_t server_radix_cache::match_helper(
        const server_radix_node & node,
        const llama_tokens & tokens,
        size_t pos,
        int32_t & donor_slot) const {
    if (pos >= tokens.size()) {
        return pos;
    }

    const auto it = node.children.find(tokens[pos]);
    if (it == node.children.end()) {
        return pos;
    }

    const server_radix_node & child = *it->second;
    const size_t matched = edge_lcp(child.key, tokens, pos);
    if (matched == 0) {
        return pos;
    }

    if (child.donor_slot >= 0) {
        donor_slot = child.donor_slot;
    }

    pos += matched;

    if (matched < child.key.size()) {
        // ended inside edge - structural split not done on match (read-only)
        return pos;
    }

    return match_helper(child, tokens, pos, donor_slot);
}

server_radix_match server_radix_cache::match_prefix(const server_tokens & tokens) const {
    server_radix_match res;
    if (tokens.empty() || tokens.has_mtmd) {
        return res;
    }

    int32_t donor = -1;
    size_t n = match_helper(root, tokens.get_tokens(), 0, donor);
    n = align_len(n);

    res.n_shared   = n;
    res.donor_slot = (n > 0) ? donor : -1;

    return res;
}

void server_radix_cache::insert_helper(
        server_radix_node & node,
        const llama_tokens & tokens,
        size_t pos,
        int32_t slot_id) {
    if (pos >= tokens.size()) {
        return;
    }

    auto it = node.children.find(tokens[pos]);
    if (it == node.children.end()) {
        auto child = std::make_unique<server_radix_node>();
        child->key.assign(tokens.begin() + pos, tokens.end());
        child->donor_slot = slot_id;
        child->last_access_us = ggml_time_us();
        node.children.emplace(tokens[pos], std::move(child));
        return;
    }

    server_radix_node & child = *it->second;
    const size_t matched = edge_lcp(child.key, tokens, pos);

    if (matched < child.key.size()) {
        // split child edge
        auto split = std::make_unique<server_radix_node>();
        split->key.assign(child.key.begin() + matched, child.key.end());
        split->donor_slot = child.donor_slot;
        split->lock_ref = child.lock_ref;
        split->last_access_us = child.last_access_us;
        split->children = std::move(child.children);

        child.key.resize(matched);
        child.children.clear();
        child.lock_ref = 0;
        if (!split->key.empty()) {
            child.children.emplace(split->key[0], std::move(split));
        }
    }

    child.donor_slot = slot_id;
    child.last_access_us = ggml_time_us();
    pos += matched;

    if (pos < tokens.size()) {
        insert_helper(child, tokens, pos, slot_id);
    }
}

void server_radix_cache::insert(const server_tokens & tokens, int32_t slot_id) {
    size_t n = align_len(tokens.size());
    if (n == 0 || slot_id < 0) {
        return;
    }

    llama_tokens prefix(tokens.get_tokens().begin(), tokens.get_tokens().begin() + n);
    insert_helper(root, prefix, 0, slot_id);
    stats.n_insert++;
}

void server_radix_cache::lock_helper(
        server_radix_node & node,
        const llama_tokens & tokens,
        size_t pos,
        size_t end,
        int delta) {
    if (pos >= end) {
        return;
    }

    auto it = node.children.find(tokens[pos]);
    if (it == node.children.end()) {
        return;
    }

    server_radix_node & child = *it->second;
    const size_t matched = edge_lcp(child.key, tokens, pos);
    if (matched == 0) {
        return;
    }

    const size_t next = pos + matched;
    if (next > end && matched > 0) {
        // partial edge lock still pins the node
        child.lock_ref = std::max(0, child.lock_ref + delta);
        child.last_access_us = ggml_time_us();
        return;
    }

    child.lock_ref = std::max(0, child.lock_ref + delta);
    child.last_access_us = ggml_time_us();

    if (matched == child.key.size()) {
        lock_helper(child, tokens, next, end, delta);
    }
}

void server_radix_cache::lock_prefix(const server_tokens & tokens, size_t n_shared) {
    n_shared = align_len(std::min(n_shared, tokens.size()));
    if (n_shared == 0) {
        return;
    }
    lock_helper(root, tokens.get_tokens(), 0, n_shared, +1);
}

void server_radix_cache::unlock_prefix(const server_tokens & tokens, size_t n_shared) {
    n_shared = align_len(std::min(n_shared, tokens.size()));
    if (n_shared == 0) {
        return;
    }
    lock_helper(root, tokens.get_tokens(), 0, n_shared, -1);
}

void server_radix_cache::invalidate_helper(server_radix_node & node, int32_t slot_id) {
    if (node.donor_slot == slot_id) {
        node.donor_slot = -1;
    }
    for (auto & kv : node.children) {
        invalidate_helper(*kv.second, slot_id);
    }
}

void server_radix_cache::invalidate_slot(int32_t slot_id) {
    invalidate_helper(root, slot_id);
}

void server_radix_cache::refresh_helper(
        server_radix_node & node,
        const llama_tokens & tokens,
        size_t pos,
        int32_t slot_id) {
    if (pos >= tokens.size()) {
        return;
    }

    auto it = node.children.find(tokens[pos]);
    if (it == node.children.end()) {
        return;
    }

    server_radix_node & child = *it->second;
    const size_t matched = edge_lcp(child.key, tokens, pos);
    if (matched == 0) {
        return;
    }

    if (matched == child.key.size()) {
        child.donor_slot = slot_id;
        child.last_access_us = ggml_time_us();
        refresh_helper(child, tokens, pos + matched, slot_id);
    }
}

void server_radix_cache::refresh_donor(const server_tokens & tokens, int32_t slot_id) {
    size_t n = align_len(tokens.size());
    if (n == 0) {
        return;
    }
    llama_tokens prefix(tokens.get_tokens().begin(), tokens.get_tokens().begin() + n);
    refresh_helper(root, prefix, 0, slot_id);
}

bool server_radix_cache::evict_helper(server_radix_node & node, size_t & budget) {
    // evict unlocked leaf children first (LRU among candidates)
    while (budget == 0 && !node.children.empty()) {
        llama_token worst_tok = 0;
        int64_t worst_t = LLONG_MAX;
        bool found = false;

        for (auto & kv : node.children) {
            server_radix_node & child = *kv.second;
            if (child.lock_ref > 0) {
                continue;
            }
            if (child.children.empty() && child.last_access_us <= worst_t) {
                worst_t = child.last_access_us;
                worst_tok = kv.first;
                found = true;
            }
        }

        if (!found) {
            // try recurse into children
            for (auto it = node.children.begin(); it != node.children.end(); ) {
                if (evict_helper(*it->second, budget)) {
                    if (it->second->children.empty() && it->second->lock_ref == 0 && it->second->key.empty()) {
                        it = node.children.erase(it);
                        stats.n_evict++;
                    } else {
                        ++it;
                    }
                    if (budget == 0) {
                        return true;
                    }
                } else {
                    ++it;
                }
            }
            return false;
        }

        node.children.erase(worst_tok);
        stats.n_evict++;
        // budget stays 0 -> keep until caller raises; here we evict one and return
        return true;
    }

    for (auto it = node.children.begin(); it != node.children.end(); ) {
        if (budget > 0 && n_nodes_helper(node) <= budget) {
            return false;
        }
        if (evict_helper(*it->second, budget)) {
            if (it->second->children.empty() && it->second->lock_ref == 0) {
                // drop empty unlocked child after nested eviction
            }
            ++it;
            return true;
        }
        ++it;
    }

    return false;
}

void server_radix_cache::evict_unlocked(size_t max_nodes) {
    if (max_nodes == 0) {
        return;
    }
    size_t budget = max_nodes;
    while (n_nodes() > max_nodes) {
        size_t before = n_nodes();
        if (!evict_helper(root, budget)) {
            break;
        }
        if (n_nodes() >= before) {
            break;
        }
    }
}
