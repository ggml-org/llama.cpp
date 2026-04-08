#pragma once

#include "llama-kv-cache-paged.h"

#include <clocale>
#include <vector>

enum class llama_scheduler_status {
    OK,
    DEADLOCK,  // cannot make progress
};

class llama_paged_scheduler_impl {
  public:
    llama_paged_scheduler_impl(uint32_t n_ctx, uint32_t block_sz, int32_t n_batch, llama_kv_cache_paged * kv_manager);

    llama_scheduler_status step(llama_batch & batch);
    bool                   queue_request(llama_sequence_group group);
    void update(const llama_batch & batch, const std::vector<llama_token> & new_tokens, const int8_t * stop_flags);
    void set_on_finish(llama_paged_on_finish_cb cb, void * user_data);
    llama_sequence_group *         get_group_from_id(int32_t request_id) const;
    const llama_paged_batch_info * get_curr_batch_info() const;

  private:
    void insert_sorted_by_arrival_time(llama_sequence_group_ptr new_group, llama_sequence_group_list & list);

    bool check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const;
    bool check_livelock(uint32_t n_swapped, uint32_t prev_n_swapped);

    void set_running(llama_sequence_group_ptr group);
    void set_swapped(llama_sequence_group_ptr group);
    void set_waiting(llama_sequence_group_ptr group);

    void finish(llama_sequence_group & group);

    int32_t get_curr_decode_tokens() const;

    void evict();
    void process_running_list(llama_sequence_group_raw_list & candidates);
    void process_swapped_list(llama_sequence_group_raw_list & candidates);
    void process_waiting_list(llama_sequence_group_raw_list & candidates, int32_t remaining_token_bugdet);

    void swap_out_or_recompute(llama_sequence_group_ptr group_ptr);

    int32_t calculate_global_slot_index(int32_t token_pos, std::vector<uint32_t> & block_table);

    void clear_batch(llama_batch & batch);
    void populate_batch_from(const llama_sequence_group_raw_list & candidates, llama_batch & batch);

    llama_sequence_group_list running;
    llama_sequence_group_list swapped;
    llama_sequence_group_list waiting;

    // Used for fast lookups
    std::unordered_map<int32_t, llama_sequence_group *> id_to_group;

    const uint32_t         n_seq_max_ctx;
    const uint32_t         block_size;
    const int32_t          n_batch;
    llama_kv_cache_paged * kv_cache_manager = nullptr;
    llama_paged_batch_info curr_info;

    uint32_t n_livelock_steps   = 0;
    uint32_t prev_n_swapped     = 0;
    uint32_t max_livelock_steps = 20;

    // Callback for output tracking
    llama_paged_on_finish_cb on_finish_cb        = nullptr;
    void *                   on_finish_user_data = nullptr;
};
