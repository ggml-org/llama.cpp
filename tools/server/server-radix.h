#pragma once

#include "server-common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Server-side radix prefix index for physical KV sharing via seq_cp (unified KV).
// Nodes map token edges to a donor slot id that currently holds those cells.
// Requires --kv-unified. Opt-in via --radix-cache.
//
// Phase 3 (paged / block pool) is intentionally NOT implemented here. Ship radix
// aliasing first; revisit paged KV only when capacity benchmarks prove a bottleneck
// after Phase 1/2 (see tools/server/bench/radix-prefix/README.md).

struct server_radix_stats {
    uint64_t n_match       = 0;
    uint64_t n_hit         = 0;
    uint64_t n_matched_tok = 0;
    uint64_t n_insert      = 0;
    uint64_t n_evict       = 0;
    uint64_t n_alias       = 0;
};

struct server_radix_match {
    size_t  n_shared   = 0;
    int32_t donor_slot = -1;
};

struct server_radix_node {
    std::vector<llama_token> key;
    int32_t  donor_slot = -1;
    int32_t  lock_ref   = 0;
    int64_t  last_access_us = 0;

    // first token of child edge -> child
    std::unordered_map<llama_token, std::unique_ptr<server_radix_node>> children;
};

struct server_radix_cache {
    explicit server_radix_cache(size_t page_size = 1);

    size_t page_size = 1;

    server_radix_node root;
    server_radix_stats stats;

    // Align length down to page_size (page_size==1 => identity).
    size_t align_len(size_t n) const;

    // Longest prefix match. donor_slot may be -1 if only structural match remains.
    server_radix_match match_prefix(const server_tokens & tokens) const;

    // Register / refresh path for tokens owned by slot_id. Splits edges as needed.
    void insert(const server_tokens & tokens, int32_t slot_id);

    // Pin / unpin matched path (flight requests).
    void lock_prefix(const server_tokens & tokens, size_t n_shared);
    void unlock_prefix(const server_tokens & tokens, size_t n_shared);

    // Drop unlocked leaves until node count is under budget (0 = no limit).
    void evict_unlocked(size_t max_nodes);

    size_t n_nodes() const;

    // Invalidate donor pointers when a slot is cleared or reassigned.
    void invalidate_slot(int32_t slot_id);

    // Update donor for nodes still matching this slot's retained tokens.
    void refresh_donor(const server_tokens & tokens, int32_t slot_id);

private:
    size_t n_nodes_helper(const server_radix_node & node) const;

    void insert_helper(server_radix_node & node, const llama_tokens & tokens, size_t pos, int32_t slot_id);

    size_t match_helper(const server_radix_node & node, const llama_tokens & tokens, size_t pos,
                        int32_t & donor_slot) const;

    void lock_helper(server_radix_node & node, const llama_tokens & tokens, size_t pos, size_t end, int delta);

    void invalidate_helper(server_radix_node & node, int32_t slot_id);

    void refresh_helper(server_radix_node & node, const llama_tokens & tokens, size_t pos, int32_t slot_id);

    bool evict_helper(server_radix_node & node, size_t & budget);
};
