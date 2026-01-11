#pragma once

#include "llama.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>

struct llama_kv_block_config {
    uint32_t tokens_per_block = 64;
    uint32_t min_free_blocks = 4;
    bool enable_cow = false;
    size_t block_size_bytes = 0;  // computed during init
};

struct llama_kv_block {
    uint32_t ref_count = 0;      // 0=free, 1=exclusive, >1=shared
    uint32_t n_tokens = 0;       // valid tokens in block
    llama_seq_id seq_id = -1;
    uint32_t logical_idx = UINT32_MAX;

    bool is_free() const {
        return ref_count == 0;
    }

    bool is_shared() const {
        return ref_count > 1;
    }

    void reset() {
        ref_count = 0;
        n_tokens = 0;
        seq_id = -1;
        logical_idx = UINT32_MAX;
    }
};

struct llama_kv_block_stats {
    uint32_t n_blocks_total = 0;
    uint32_t n_blocks_used = 0;
    uint32_t n_blocks_free = 0;
    uint32_t n_sequences = 0;

    uint32_t n_tokens_total = 0;
    uint32_t n_tokens_used = 0;
    uint32_t n_tokens_wasted = 0;

    float fragmentation = 0.0f;
    float utilization = 0.0f;

    size_t memory_allocated = 0;
    size_t memory_used = 0;
    size_t memory_wasted = 0;
};

// Stack-based block pool with O(1) allocation/deallocation.
// NOT thread-safe: designed for single-threaded llama_context access.
class llama_kv_block_pool {
public:
    llama_kv_block_pool() = default;

    void init(uint32_t n_blocks) {
        blocks.resize(n_blocks);
        free_stack.reserve(n_blocks);
        for (uint32_t i = n_blocks; i > 0; --i) {
            blocks[i - 1].reset();
            free_stack.push_back(i - 1);
        }
    }

    // Returns block index, or -1 if unavailable
    int32_t allocate() {
        if (free_stack.empty()) {
            return -1;
        }

        const uint32_t idx = free_stack.back();
        free_stack.pop_back();

        assert(blocks[idx].is_free());
        blocks[idx].ref_count = 1;

        return static_cast<int32_t>(idx);
    }

    std::vector<int32_t> allocate_batch(uint32_t n) {
        if (n > free_stack.size()) {
            return {};
        }

        std::vector<int32_t> result;
        result.reserve(n);

        for (uint32_t i = 0; i < n; ++i) {
            int32_t idx = allocate();
            if (idx < 0) {
                // Rollback on failure
                for (int32_t allocated_idx : result) {
                    deallocate(allocated_idx);
                }
                return {};
            }
            result.push_back(idx);
        }

        return result;
    }

    void deallocate(int32_t idx) {
        assert(idx >= 0 && static_cast<size_t>(idx) < blocks.size());
        assert(!blocks[idx].is_free());

        blocks[idx].ref_count--;

        if (blocks[idx].ref_count == 0) {
            blocks[idx].reset();
            free_stack.push_back(static_cast<uint32_t>(idx));
        }
    }

    void add_ref(int32_t idx) {
        assert(idx >= 0 && static_cast<size_t>(idx) < blocks.size());
        assert(!blocks[idx].is_free());

        blocks[idx].ref_count++;
    }

    llama_kv_block & get(int32_t idx) {
        assert(idx >= 0 && static_cast<size_t>(idx) < blocks.size());
        return blocks[idx];
    }

    const llama_kv_block & get(int32_t idx) const {
        assert(idx >= 0 && static_cast<size_t>(idx) < blocks.size());
        return blocks[idx];
    }

    uint32_t n_free() const {
        return static_cast<uint32_t>(free_stack.size());
    }

    uint32_t n_total() const {
        return static_cast<uint32_t>(blocks.size());
    }

    uint32_t n_used() const {
        return n_total() - n_free();
    }

    llama_kv_block_stats compute_stats(uint32_t tokens_per_block, size_t bytes_per_token = 0) const {
        llama_kv_block_stats stats;

        stats.n_blocks_total = n_total();
        stats.n_blocks_used = n_used();
        stats.n_blocks_free = n_free();

        uint32_t total_tokens_in_blocks = 0;
        uint32_t actual_tokens_stored = 0;

        for (const auto & block : blocks) {
            if (!block.is_free()) {
                total_tokens_in_blocks += tokens_per_block;
                actual_tokens_stored += block.n_tokens;
            }
        }

        stats.n_tokens_total = stats.n_blocks_total * tokens_per_block;
        stats.n_tokens_used = actual_tokens_stored;
        stats.n_tokens_wasted = total_tokens_in_blocks - actual_tokens_stored;

        if (total_tokens_in_blocks > 0) {
            stats.fragmentation = static_cast<float>(stats.n_tokens_wasted) /
                                  static_cast<float>(total_tokens_in_blocks);
        }

        if (stats.n_tokens_total > 0) {
            stats.utilization = static_cast<float>(stats.n_tokens_used) /
                                static_cast<float>(stats.n_tokens_total);
        }

        if (bytes_per_token > 0) {
            stats.memory_allocated = stats.n_blocks_used * tokens_per_block * bytes_per_token;
            stats.memory_used = stats.n_tokens_used * bytes_per_token;
            stats.memory_wasted = stats.n_tokens_wasted * bytes_per_token;
        }

        return stats;
    }

    float get_fragmentation(uint32_t tokens_per_block) const {
        if (blocks.empty()) {
            return 0.0f;
        }

        uint32_t total_allocated = 0;
        uint32_t total_used = 0;

        for (const auto & block : blocks) {
            if (!block.is_free()) {
                total_allocated += tokens_per_block;
                total_used += block.n_tokens;
            }
        }

        if (total_allocated == 0) {
            return 0.0f;
        }

        return static_cast<float>(total_allocated - total_used) /
               static_cast<float>(total_allocated);
    }

    void clear() {
        for (auto & block : blocks) {
            block.reset();
        }
        free_stack.clear();
        free_stack.reserve(blocks.size());
        for (uint32_t i = static_cast<uint32_t>(blocks.size()); i > 0; --i) {
            free_stack.push_back(i - 1);
        }
    }

private:
    std::vector<llama_kv_block> blocks;
    std::vector<uint32_t> free_stack;
};

// Maps logical blocks to physical blocks per sequence.
// NOT thread-safe: same design as llama_kv_block_pool.
class llama_kv_block_table {
public:
    llama_kv_block_table() = default;

    // Returns -1 if mapping doesn't exist
    int32_t get_physical(llama_seq_id seq_id, uint32_t logical_idx) const {
        auto it = table.find(seq_id);
        if (it == table.end()) {
            return -1;
        }

        const auto & seq_blocks = it->second;
        if (logical_idx >= seq_blocks.size()) {
            return -1;
        }

        return seq_blocks[logical_idx];
    }

    void set_mapping(llama_seq_id seq_id, uint32_t logical_idx, int32_t physical_idx) {
        auto & seq_blocks = table[seq_id];
        if (logical_idx >= seq_blocks.size()) {
            seq_blocks.resize(logical_idx + 1, -1);
        }

        seq_blocks[logical_idx] = physical_idx;
    }

    uint32_t append_block(llama_seq_id seq_id, int32_t physical_idx) {
        auto & seq_blocks = table[seq_id];
        uint32_t logical_idx = static_cast<uint32_t>(seq_blocks.size());
        seq_blocks.push_back(physical_idx);
        return logical_idx;
    }

    const std::vector<int32_t> * get_sequence_blocks(llama_seq_id seq_id) const {
        auto it = table.find(seq_id);
        if (it == table.end()) {
            return nullptr;
        }
        return &it->second;
    }

    uint32_t get_sequence_n_blocks(llama_seq_id seq_id) const {
        auto it = table.find(seq_id);
        if (it == table.end()) {
            return 0;
        }
        return static_cast<uint32_t>(it->second.size());
    }

    void clear_sequence(llama_seq_id seq_id) {
        table.erase(seq_id);
    }

    std::vector<int32_t> truncate_sequence(llama_seq_id seq_id, uint32_t n_blocks_to_remove) {
        std::vector<int32_t> removed;

        auto it = table.find(seq_id);
        if (it == table.end()) {
            return removed;
        }

        auto & seq_blocks = it->second;
        uint32_t n_to_remove = std::min(n_blocks_to_remove, static_cast<uint32_t>(seq_blocks.size()));

        removed.reserve(n_to_remove);
        for (uint32_t i = 0; i < n_to_remove; ++i) {
            removed.push_back(seq_blocks.back());
            seq_blocks.pop_back();
        }

        if (seq_blocks.empty()) {
            table.erase(it);
        }

        return removed;
    }

    void copy_sequence(llama_seq_id seq_src, llama_seq_id seq_dst) {
        auto it = table.find(seq_src);
        if (it == table.end()) {
            return;
        }

        table[seq_dst] = it->second;
    }

    bool has_sequence(llama_seq_id seq_id) const {
        return table.find(seq_id) != table.end();
    }

    std::vector<llama_seq_id> get_sequences() const {
        std::vector<llama_seq_id> seqs;
        seqs.reserve(table.size());
        for (const auto & [seq_id, _] : table) {
            seqs.push_back(seq_id);
        }
        return seqs;
    }

    void clear() {
        table.clear();
    }

    size_t total_mappings() const {
        size_t total = 0;
        for (const auto & [_, blocks] : table) {
            total += blocks.size();
        }
        return total;
    }

    uint32_t n_sequences() const {
        return static_cast<uint32_t>(table.size());
    }

private:
    std::unordered_map<llama_seq_id, std::vector<int32_t>> table;
};

inline uint32_t cell_to_block(uint32_t cell_idx, uint32_t tokens_per_block) {
    return cell_idx / tokens_per_block;
}

inline uint32_t block_to_cell(uint32_t block_idx, uint32_t tokens_per_block) {
    return block_idx * tokens_per_block;
}

inline uint32_t cell_offset_in_block(uint32_t cell_idx, uint32_t tokens_per_block) {
    return cell_idx % tokens_per_block;
}

inline uint32_t tokens_to_blocks(uint32_t n_tokens, uint32_t tokens_per_block) {
    return (n_tokens + tokens_per_block - 1) / tokens_per_block;
}

inline std::string llama_kv_block_stats_to_string(const llama_kv_block_stats & stats) {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "blocks: %u/%u (%.1f%% used), tokens: %u/%u (%.1f%% util), "
        "waste: %u tokens (%.1f%% frag), seqs: %u",
        stats.n_blocks_used, stats.n_blocks_total,
        stats.n_blocks_total > 0 ? 100.0f * stats.n_blocks_used / stats.n_blocks_total : 0.0f,
        stats.n_tokens_used, stats.n_tokens_total,
        100.0f * stats.utilization,
        stats.n_tokens_wasted,
        100.0f * stats.fragmentation,
        stats.n_sequences);
    return std::string(buf);
}

inline std::string llama_kv_block_stats_detailed(const llama_kv_block_stats & stats) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "KV Block Stats:\n"
        "  Blocks: %u total, %u used, %u free\n"
        "  Tokens: %u capacity, %u stored, %u wasted\n"
        "  Utilization: %.2f%%, Fragmentation: %.2f%%\n"
        "  Sequences: %u\n"
        "  Memory: %.2f MiB allocated, %.2f MiB used, %.2f MiB wasted",
        stats.n_blocks_total, stats.n_blocks_used, stats.n_blocks_free,
        stats.n_tokens_total, stats.n_tokens_used, stats.n_tokens_wasted,
        100.0f * stats.utilization, 100.0f * stats.fragmentation,
        stats.n_sequences,
        stats.memory_allocated / (1024.0 * 1024.0),
        stats.memory_used / (1024.0 * 1024.0),
        stats.memory_wasted / (1024.0 * 1024.0));
    return std::string(buf);
}
