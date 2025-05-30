#pragma once

#include "llama.h"

#include <memory>
#include <vector>

struct llama_ubatch;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;
};

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
class llama_memory_i {
public:
    virtual ~llama_memory_i() = default;

    virtual void clear() = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual bool get_can_edit() const = 0;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// the interface for managing the memory state during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_unified_state
//   - llama_kv_cache_unified_iswa_state
//   ...
//
// the only method that can mutate the memory and the memory state is llama_memory_i::apply()
//
// TODO: rename to llama_memory_context_i ?
class llama_memory_state_i {
public:
    virtual ~llama_memory_state_i() = default;

    // consume the current ubatch from the state and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // TODO: this might get reworked in the future when refactoring llama_batch
    virtual std::vector<int64_t> & out_ids() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory state
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_state_ptr = std::unique_ptr<llama_memory_state_i>;
