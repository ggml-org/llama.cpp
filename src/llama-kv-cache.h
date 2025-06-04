#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-memory.h"

struct llama_kv_cache : public llama_memory_i {
    virtual ~llama_kv_cache() = default;

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

    // process any pending defrag/shift/etc. operations
    // optionally call once before processing a new batch
    // return true if any operations were performed
    virtual bool update(llama_context & lctx) = 0;

    // schedule a defrag if the fragmentation threshold is exceeded. otherwise, do nothing
    // TODO: change to
    //   llama_memory_state_ptr init_defrag(float thold) = 0;
    //
    virtual void defrag_sched(float thold) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    bool get_can_edit() const override { return get_can_shift(); }

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) = 0;
};
