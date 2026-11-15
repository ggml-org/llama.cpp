#pragma once

#include "server-common.h"
#include "server-task.h"

#include <list>
#include <memory>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    std::shared_ptr<const common_prompt_checkpoint> ckpt;
};

struct server_prompt {
    server_tokens tokens;

    // a checkpoint does not change after it is created, so the prompt cache shares it with the slot instead of a copy
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const;
};

struct server_prompt_data {
    common_state_data main;
    common_state_data drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();
};

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;

    // the slot runs a task whose prompt a new task can copy, see select_donor()
    bool can_donate = false;

    // tokens of the task that are processed, -1 if the task has not started yet
    int32_t n_task_done = -1;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;
};

// a busy slot that has, or will have, the start of the prompt of a new task
struct server_slot_donor {
    int id_slot = -1; // -1 if there is none

    // tokens shared with the new task, the task waits until the donor has them
    int32_t n_tokens = 0;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // a donor must give at least this many tokens more than the slot of the task already has
    int32_t n_donor_min = 64;

    // the memory can only be copied as a whole: the donor must end a batch exactly at the end of the shared tokens
    bool donor_exact = false;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task) const;

    // id_slot: the slot of the task
    // n_have: tokens of the task that its slot already has
    server_slot_donor select_donor(const std::vector<server_slot_info> & slots, const server_task & task, int id_slot, int32_t n_have) const;
};
