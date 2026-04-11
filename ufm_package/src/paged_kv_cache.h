// paged_kv_cache.h — Paged KV Cache, Piece 1 (CPU block allocator)
//
// FIX vs previous version: O(log n) LRU eviction
// ──────────────────────────────────────────────────────────────────
// Previous pkv_lru_victim() was a linear scan: O(total_blocks).
// Fine for <200 blocks, painful at 1000+ (32K+ context at 16KB/block).
//
// Now: lru_heap is a min-heap keyed on last_access_seq.
// pkv_lru_victim() is O(1) (peek top of heap).
// Heap maintenance:
//   - pkv_alloc_block(): push new block → O(log n)
//   - pkv_evict_block():  remove from heap → O(log n) via lazy deletion
//   - pkv_touch_block():  update last_access and re-heapify → O(log n)
//
// Lazy deletion: rather than remove mid-heap, we mark evicted blocks with
// ref_count==0 and skip them on heap pop. Standard heap eviction pattern.

#pragma once
#include <vector>
#include <unordered_map>
#include <queue>
#include <functional>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <cstdio>

#define PKV_TOKENS_PER_BLOCK   16u
#define PKV_MAX_SEQS           64u
#define PKV_INVALID_BLOCK      0xFFFFFFFFu

struct pkv_block_t {
    uint32_t block_id;
    uint32_t ref_count;
    uint32_t last_access_seq;
    bool     in_vram;
    uint64_t vram_offset;
    uint64_t ram_offset;
};

struct pkv_seq_t {
    uint32_t seq_id;
    uint32_t n_tokens;
    uint32_t n_layers;
    std::vector<uint32_t> block_table;
    uint32_t max_blocks;
};

// Min-heap entry for LRU
struct pkv_heap_entry_t {
    uint32_t last_access_seq;
    uint32_t block_id;
    bool operator>(const pkv_heap_entry_t& o) const {
        return last_access_seq > o.last_access_seq;
    }
};

struct pkv_allocator_t {
    uint32_t              total_blocks;
    std::vector<pkv_block_t> blocks;
    std::queue<uint32_t>  free_list;
    uint32_t              n_layers;
    uint32_t              head_dim;
    uint32_t              n_heads_kv;
    uint64_t              bytes_per_block;

    std::unordered_map<uint32_t, pkv_seq_t> seqs;
    uint32_t              access_counter;

    // Min-heap for O(1) LRU victim lookup, O(log n) update
    // Uses lazy deletion: stale entries are ignored on pop.
    std::priority_queue<pkv_heap_entry_t,
                        std::vector<pkv_heap_entry_t>,
                        std::greater<pkv_heap_entry_t>> lru_heap;
};

// ── Init ─────────────────────────────────────────────────────────────────────

inline void pkv_init(pkv_allocator_t* pkv,
                     uint32_t total_blocks,
                     uint32_t n_layers,
                     uint32_t n_heads_kv,
                     uint32_t head_dim,
                     uint64_t vram_pool_base,
                     uint64_t ram_pool_base) {
    pkv->total_blocks   = total_blocks;
    pkv->n_layers       = n_layers;
    pkv->head_dim       = head_dim;
    pkv->n_heads_kv     = n_heads_kv;
    pkv->access_counter = 0;
    pkv->bytes_per_block= PKV_TOKENS_PER_BLOCK * n_heads_kv * head_dim * sizeof(uint16_t) * 2;

    pkv->blocks.resize(total_blocks);
    for (uint32_t i = 0; i < total_blocks; i++) {
        pkv->blocks[i] = {i, 0, 0, false,
            vram_pool_base + i * pkv->bytes_per_block,
            ram_pool_base  + i * pkv->bytes_per_block};
        pkv->free_list.push(i);
    }
}

// ── Sequence management ───────────────────────────────────────────────────────

inline uint32_t pkv_create_seq(pkv_allocator_t* pkv, uint32_t seq_id) {
    assert(pkv->seqs.find(seq_id) == pkv->seqs.end());
    pkv_seq_t seq{};
    seq.seq_id     = seq_id;
    seq.n_tokens   = 0;
    seq.n_layers   = pkv->n_layers;
    seq.max_blocks = 8;
    seq.block_table.assign(pkv->n_layers * seq.max_blocks, PKV_INVALID_BLOCK);
    pkv->seqs[seq_id] = seq;
    return seq_id;
}

inline void pkv_destroy_seq(pkv_allocator_t* pkv, uint32_t seq_id) {
    auto it = pkv->seqs.find(seq_id);
    if (it == pkv->seqs.end()) return;
    pkv_seq_t& seq = it->second;
    uint32_t n_slots = (seq.n_tokens + PKV_TOKENS_PER_BLOCK - 1) / PKV_TOKENS_PER_BLOCK;
    for (uint32_t layer = 0; layer < pkv->n_layers; layer++) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            uint32_t bid = seq.block_table[layer * seq.max_blocks + slot];
            if (bid != PKV_INVALID_BLOCK) {
                pkv->blocks[bid].ref_count--;
                if (pkv->blocks[bid].ref_count == 0) {
                    pkv->blocks[bid].in_vram = false;
                    pkv->free_list.push(bid);
                    // Lazy deletion: stale heap entries ignored on pop
                }
            }
        }
    }
    pkv->seqs.erase(it);
}

// ── Block allocation ──────────────────────────────────────────────────────────

inline uint32_t pkv_alloc_block(pkv_allocator_t* pkv) {
    if (pkv->free_list.empty()) return PKV_INVALID_BLOCK;
    uint32_t bid = pkv->free_list.front();
    pkv->free_list.pop();
    uint32_t seq = pkv->access_counter++;
    pkv->blocks[bid].ref_count      = 1;
    pkv->blocks[bid].last_access_seq= seq;
    pkv->blocks[bid].in_vram        = true;
    // Push to LRU heap
    pkv->lru_heap.push({seq, bid});
    return bid;
}

inline uint32_t pkv_ensure_block(pkv_allocator_t* pkv, uint32_t seq_id,
                                  uint32_t layer, uint32_t token_pos) {
    auto it = pkv->seqs.find(seq_id);
    assert(it != pkv->seqs.end());
    pkv_seq_t& seq = it->second;
    uint32_t slot  = token_pos / PKV_TOKENS_PER_BLOCK;

    if (slot >= seq.max_blocks) {
        uint32_t new_max = seq.max_blocks * 2;
        seq.block_table.resize(pkv->n_layers * new_max, PKV_INVALID_BLOCK);
        for (int32_t l = (int32_t)pkv->n_layers - 1; l >= 0; l--) {
            for (int32_t s = (int32_t)seq.max_blocks - 1; s >= 0; s--)
                seq.block_table[l * new_max + s] = seq.block_table[l * seq.max_blocks + s];
            for (uint32_t s = seq.max_blocks; s < new_max; s++)
                seq.block_table[l * new_max + s] = PKV_INVALID_BLOCK;
        }
        seq.max_blocks = new_max;
    }

    uint32_t& bid_ref = seq.block_table[layer * seq.max_blocks + slot];
    if (bid_ref == PKV_INVALID_BLOCK) {
        bid_ref = pkv_alloc_block(pkv);
        if (bid_ref == PKV_INVALID_BLOCK) return PKV_INVALID_BLOCK;
    } else {
        // Touch: update access time and push new heap entry (lazy update)
        uint32_t new_seq = pkv->access_counter++;
        pkv->blocks[bid_ref].last_access_seq = new_seq;
        pkv->lru_heap.push({new_seq, bid_ref});  // old entry becomes stale
    }
    return bid_ref;
}

// ── LRU victim — O(1), using min-heap with lazy deletion ─────────────────────
// Pops stale entries (blocks that have been freed or have a newer heap entry)
// until the top is a valid in-VRAM block.

inline uint32_t pkv_lru_victim(pkv_allocator_t* pkv) {
    while (!pkv->lru_heap.empty()) {
        auto top = pkv->lru_heap.top();
        uint32_t bid = top.block_id;
        const pkv_block_t& b = pkv->blocks[bid];

        // Stale if: freed (ref_count==0), not in VRAM, or has newer access
        if (b.ref_count == 0 || !b.in_vram ||
            b.last_access_seq != top.last_access_seq) {
            pkv->lru_heap.pop();   // discard stale entry
            continue;
        }
        return bid;   // top is a valid victim, O(1) amortised
    }
    return PKV_INVALID_BLOCK;   // nothing to evict
}

// ── Query helpers ─────────────────────────────────────────────────────────────

struct pkv_token_addr_t { uint64_t kv_offset; uint32_t token_in_block; bool valid; };

inline pkv_token_addr_t pkv_get_token_addr(pkv_allocator_t* pkv, uint32_t seq_id,
                                             uint32_t layer, uint32_t token_pos, bool vram) {
    auto it = pkv->seqs.find(seq_id);
    if (it == pkv->seqs.end()) return {0,0,false};
    pkv_seq_t& seq = it->second;
    uint32_t slot = token_pos / PKV_TOKENS_PER_BLOCK;
    if (slot >= seq.max_blocks) return {0,0,false};
    uint32_t bid = seq.block_table[layer * seq.max_blocks + slot];
    if (bid == PKV_INVALID_BLOCK) return {0,0,false};
    uint64_t base = vram ? pkv->blocks[bid].vram_offset : pkv->blocks[bid].ram_offset;
    uint32_t tok  = token_pos % PKV_TOKENS_PER_BLOCK;
    return {base + tok * pkv->n_heads_kv * pkv->head_dim * sizeof(uint16_t) * 2, tok, true};
}

inline uint32_t pkv_count_evicted(const pkv_allocator_t* pkv, uint32_t seq_id, uint32_t layer) {
    auto it = pkv->seqs.find(seq_id);
    if (it == pkv->seqs.end()) return 0;
    const pkv_seq_t& seq = it->second;
    uint32_t n_slots = (seq.n_tokens + PKV_TOKENS_PER_BLOCK - 1) / PKV_TOKENS_PER_BLOCK;
    uint32_t evicted = 0;
    for (uint32_t s = 0; s < n_slots; s++) {
        uint32_t bid = seq.block_table[layer * seq.max_blocks + s];
        if (bid != PKV_INVALID_BLOCK && !pkv->blocks[bid].in_vram) evicted++;
    }
    return evicted;
}

inline void pkv_print_stats(const pkv_allocator_t* pkv) {
    uint32_t used = pkv->total_blocks - (uint32_t)pkv->free_list.size();
    uint32_t in_vram = 0;
    for (uint32_t i = 0; i < pkv->total_blocks; i++)
        if (pkv->blocks[i].in_vram && pkv->blocks[i].ref_count > 0) in_vram++;
    printf("[PKV] %u/%u blocks | %u VRAM | %u seqs | %.1f MB/block | heap_size=%zu\n",
           used, pkv->total_blocks, in_vram, (uint32_t)pkv->seqs.size(),
           pkv->bytes_per_block / 1e6, pkv->lru_heap.size());
}
