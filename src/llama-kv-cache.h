#pragma once

#include "llama.h"
#include "llama-memory.h"

class llama_io_write_i;
class llama_io_read_i;

struct llama_kv_cache : public llama_memory_i {
    virtual ~llama_kv_cache() = default;

    // TODO: move the init_ interfaces to llama_memory_i

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a state object containing the ubatches and KV cache state required to process them
    // check the llama_memory_state_i::get_status() for the result
    virtual llama_memory_state_ptr init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_state_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, defrags, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_state_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    bool get_can_edit() const override { return get_can_shift(); }

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) = 0;
};
