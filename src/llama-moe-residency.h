#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

// Per-tensor residency: compact Metal pool with N slots + LRU + pread from GGUF.
// One instance per original weights tensor. Lives outside scheduler buffers.
struct moe_residency_slot {
    ggml_tensor * orig_weights = nullptr;  // original tensor (CPU, MADV_FREE'd)
    ggml_tensor * pool         = nullptr;  // [n_embd, n_ff, n_slots] on Metal
    int           layer        = -1;
    int64_t       n_slots      = 0;
    int64_t       n_expert     = 0;
    size_t        stride       = 0;  // bytes per expert (== nb[2])

    // File source.
    uint64_t file_offset  = 0;  // byte offset of the tensor in GGUF
    uint64_t total_nbytes = 0;  // == n_expert * stride

    // slot_table: expert_id -> slot_id (-1 if not resident).
    std::vector<int32_t> expert_to_slot;

    // LRU: slot_id list, front = MRU, back = LRU.
    // slot_to_expert[slot_id] = expert_id.
    std::list<int64_t>                        lru;
    std::vector<std::list<int64_t>::iterator> lru_iter;        // [slot_id]
    std::vector<int32_t>                      slot_to_expert;  // [slot_id], -1 if empty

    // Statistics.
    uint64_t total_misses  = 0;
    uint64_t total_hits    = 0;
    bool     orig_released = false;

    // Owned allocation for `pool`.
    ggml_context *        own_ctx = nullptr;
    ggml_backend_buffer_t own_buf = nullptr;

    ~moe_residency_slot();
};

class moe_residency_t {
  public:
    // Look up or create a slot pool for this original weights tensor.
    // target_backend = Metal backend where the pool lives.
    moe_residency_slot * get_or_create(ggml_tensor *  orig_weights,
                                       int64_t        n_slots,
                                       int            layer,
                                       ggml_backend_t target_backend,
                                       uint64_t       file_offset,
                                       uint64_t       total_nbytes);

    // Main entry point from the custom op. For each expert_id in ids:
    //   - hit: return the existing slot
    //   - miss: evict via LRU, pread into the selected slot, update slot_table, return the new slot
    // Writes slot_ids_out[i] for every ids[i].
    // Called from a CPU thread and can access pool->data directly through Shared storage.
    void resolve_ids(moe_residency_slot * slot, const int32_t * ids, int32_t * slot_ids_out, int64_t n_ids);

    // Warm up from an offline profile: top-N experts for the layer are loaded in advance.
    void prefill_from_profile(moe_residency_slot * slot, const std::vector<int32_t> & top_experts);

    void clear();

    // Singleton.
    static moe_residency_t & instance();

  private:
    int     pread_into_pool(moe_residency_slot * slot, int32_t expert_id, int64_t dst_slot);
    void    touch_lru(moe_residency_slot * slot, int64_t slot_id);
    int64_t evict_lru(moe_residency_slot * slot);  // returns a slot_id to overwrite

    std::mutex                                                             mu;
    std::unordered_map<ggml_tensor *, std::unique_ptr<moe_residency_slot>> slots;
};

inline moe_residency_t & moe_residency() {
    return moe_residency_t::instance();
}
