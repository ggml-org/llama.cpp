// paged_kv_eviction.h — Paged KV Cache, Piece 3 (Vulkan eviction/reload)
//
// FIX vs previous version: async eviction
// ──────────────────────────────────────────────────────────────────
// Previous: vkWaitForFences() immediately after every eviction copy.
//   CPU blocks until GPU copy completes. For long-context generation
//   where evictions happen between decode steps, this adds latency
//   proportional to the copy time (e.g. 256KB over PCIe ≈ 0.1ms).
//
// Now: ring of EVICTION_RING_SIZE fences.
//   pkv_evict_block_async(): submit copy + fence, return immediately.
//     Marks block as "eviction pending" (in_vram=false, pending_fence set).
//   pkv_flush_evictions(): wait on all outstanding fences before dispatch.
//     Called once per layer before uploading block table.
//   pkv_reload_block(): still synchronous (we NEED the data before dispatch).
//
// Pattern matches how vLLM handles async swapping: initiate evictions
// during the gap between dispatch submissions, flush before the next
// attention dispatch that needs the freed VRAM.
//
// RING SIZE: 4 is sufficient for typical eviction patterns (rarely need
// more than 1-2 concurrent evictions per decode step).

#pragma once
#include "paged_kv_cache.h"
#include <vulkan/vulkan.h>
#include <array>
#include <cstdio>
#include <cassert>

#define PKV_EVICTION_RING_SIZE 4

struct pkv_pending_eviction_t {
    VkFence   fence;
    uint32_t  block_id;
    bool      in_use;
};

struct pkv_vulkan_t {
    VkDevice        device;
    VkQueue         compute_queue;
    VkCommandPool   cmd_pool;
    VkBuffer        vram_buffer;
    VkBuffer        ram_buffer;
    void*           ram_mapped;

    // Async eviction ring
    std::array<pkv_pending_eviction_t, PKV_EVICTION_RING_SIZE> eviction_ring;
    uint32_t ring_head;   // next slot to use

    // Synchronous fence for reloads (must complete before dispatch)
    VkFence reload_fence;
};

// ── Init / teardown ───────────────────────────────────────────────────────────

inline bool pkv_vulkan_init(pkv_vulkan_t* vk, VkDevice device, VkQueue queue,
                             VkCommandPool pool, VkBuffer vram_buf, VkBuffer ram_buf,
                             void* ram_mapped) {
    vk->device        = device;
    vk->compute_queue = queue;
    vk->cmd_pool      = pool;
    vk->vram_buffer   = vram_buf;
    vk->ram_buffer    = ram_buf;
    vk->ram_mapped    = ram_mapped;
    vk->ring_head     = 0;

    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr,
                          VK_FENCE_CREATE_SIGNALED_BIT};
    for (auto& slot : vk->eviction_ring) {
        slot.in_use = false;
        if (vkCreateFence(device, &fi, nullptr, &slot.fence) != VK_SUCCESS)
            return false;
    }
    return vkCreateFence(device, &fi, nullptr, &vk->reload_fence) == VK_SUCCESS;
}

inline void pkv_vulkan_destroy(pkv_vulkan_t* vk) {
    for (auto& slot : vk->eviction_ring)
        vkDestroyFence(vk->device, slot.fence, nullptr);
    vkDestroyFence(vk->device, vk->reload_fence, nullptr);
}

// ── Internal: submit a buffer copy ───────────────────────────────────────────

static inline VkResult pkv_submit_copy(pkv_vulkan_t* vk,
                                        VkBuffer src, VkDeviceSize src_off,
                                        VkBuffer dst, VkDeviceSize dst_off,
                                        VkDeviceSize size, VkFence fence) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, vk->cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd;
    VkResult r = vkAllocateCommandBuffers(vk->device, &ai, &cmd);
    if (r != VK_SUCCESS) return r;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy region{src_off, dst_off, size};
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    vkEndCommandBuffer(cmd);

    vkResetFences(vk->device, 1, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0,nullptr,nullptr, 1,&cmd, 0,nullptr};
    r = vkQueueSubmit(vk->compute_queue, 1, &si, fence);
    vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &cmd);
    return r;
}

// ── Async eviction ────────────────────────────────────────────────────────────
// Submits VRAM→RAM copy. Returns immediately. Block marked in_vram=false.
// Call pkv_flush_evictions() before any dispatch that needs the freed slot.

inline bool pkv_evict_block_async(pkv_vulkan_t* vk, pkv_allocator_t* pkv,
                                   uint32_t block_id) {
    assert(block_id < pkv->total_blocks);
    pkv_block_t& b = pkv->blocks[block_id];
    if (!b.in_vram) return true;

    // Acquire the next ring slot — wait if it's still in use
    pkv_pending_eviction_t& slot = vk->eviction_ring[vk->ring_head];
    if (slot.in_use) {
        // Wait on the previous occupant of this slot
        vkWaitForFences(vk->device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        slot.in_use = false;
    }

    VkResult r = pkv_submit_copy(vk,
        vk->vram_buffer, b.vram_offset,
        vk->ram_buffer,  b.ram_offset,
        pkv->bytes_per_block, slot.fence);

    if (r != VK_SUCCESS) {
        fprintf(stderr, "[PKV] async evict block %u failed (%d)\n", block_id, (int)r);
        return false;
    }

    slot.block_id = block_id;
    slot.in_use   = true;
    vk->ring_head = (vk->ring_head + 1) % PKV_EVICTION_RING_SIZE;

    // Mark block as evicted immediately — GPU copy is in flight
    b.in_vram = false;
    return true;
}

// ── Flush all pending evictions ───────────────────────────────────────────────
// Wait until all in-flight eviction copies have completed.
// Call this once per decode step before uploading block tables.

inline bool pkv_flush_evictions(pkv_vulkan_t* vk) {
    for (auto& slot : vk->eviction_ring) {
        if (!slot.in_use) continue;
        VkResult r = vkWaitForFences(vk->device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "[PKV] flush_evictions wait failed (%d)\n", (int)r);
            return false;
        }
        slot.in_use = false;
    }
    return true;
}

// ── Synchronous reload ────────────────────────────────────────────────────────
// Copies RAM→VRAM. Blocks until complete — must finish before attention dispatch.

inline bool pkv_reload_block(pkv_vulkan_t* vk, pkv_allocator_t* pkv,
                               uint32_t block_id) {
    assert(block_id < pkv->total_blocks);
    pkv_block_t& b = pkv->blocks[block_id];
    if (b.in_vram) return true;

    VkResult r = pkv_submit_copy(vk,
        vk->ram_buffer,  b.ram_offset,
        vk->vram_buffer, b.vram_offset,
        pkv->bytes_per_block, vk->reload_fence);

    if (r != VK_SUCCESS) {
        fprintf(stderr, "[PKV] reload block %u failed (%d)\n", block_id, (int)r);
        return false;
    }
    vkWaitForFences(vk->device, 1, &vk->reload_fence, VK_TRUE, UINT64_MAX);
    b.in_vram = true;
    b.last_access_seq = pkv->access_counter++;
    pkv->lru_heap.push({b.last_access_seq, block_id});
    return true;
}

// ── Ensure all blocks for seq+layer are in VRAM ───────────────────────────────
// Full pre-dispatch check: evict LRU if needed, then reload evicted blocks.
// Flushes pending evictions before returning.

inline bool pkv_ensure_all_in_vram(pkv_vulkan_t* vk, pkv_allocator_t* pkv,
                                    uint32_t seq_id, uint32_t layer) {
    auto it = pkv->seqs.find(seq_id);
    if (it == pkv->seqs.end()) return false;
    const pkv_seq_t& seq = it->second;
    uint32_t n_slots = (seq.n_tokens + PKV_TOKENS_PER_BLOCK - 1) / PKV_TOKENS_PER_BLOCK;

    for (uint32_t s = 0; s < n_slots; s++) {
        uint32_t bid = seq.block_table[layer * seq.max_blocks + s];
        if (bid == PKV_INVALID_BLOCK || pkv->blocks[bid].in_vram) continue;

        // Need VRAM space — evict LRU asynchronously if pool full
        if (pkv->free_list.empty()) {
            uint32_t victim = pkv_lru_victim(pkv);
            if (victim == PKV_INVALID_BLOCK) {
                fprintf(stderr, "[PKV] ensure_all_in_vram: OOM\n");
                return false;
            }
            if (!pkv_evict_block_async(vk, pkv, victim)) return false;
            pkv->blocks[victim].in_vram = false;
            pkv->free_list.push(victim);
        }
        if (!pkv_reload_block(vk, pkv, bid)) return false;
    }

    // Wait for any queued evictions before this decode step proceeds
    return pkv_flush_evictions(vk);
}

inline uint64_t pkv_ram_pool_size(const pkv_allocator_t* pkv) {
    return (uint64_t)pkv->total_blocks * pkv->bytes_per_block;
}
