#include "llama-memory-idx-iswa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <cassert>

//
// llama_memory_idx_iswa
//

llama_memory_idx_iswa::llama_memory_idx_iswa(
        const llama_model & model,
                     bool   no_alloc,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
    const layer_filter_cb & filter,
    const layer_filter_cb & filter_idx) :
    mem_attn(new llama_kv_cache_iswa(
        model, no_alloc, type_k, type_v, v_trans, offload, swa_full, unified,
        kv_size, n_seq_max, n_ubatch,
        nullptr, filter, nullptr, nullptr)),
    mem_idx([&] {
        // the indexer mirrors the base child, so a SWA layer has no cells of its own to index
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            GGML_ASSERT(!(filter_idx(il) && model.hparams.is_swa(il)) && "an indexer layer cannot be a SWA layer");
        }

        return new llama_kv_cache_kpool(
            model, no_alloc, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, 0, LLAMA_SWA_TYPE_NONE, filter_idx);
    }()) {}

llama_memory_context_ptr llama_memory_idx_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_kv_cache_iswa::init_batch, as the indexer needs the base slot infos that its context hides
    const bool unified = mem_attn->get_base()->get_n_stream() == 1;

    // first try simple split, if it fails, try equal split
    for (const bool simple : { true, false }) {
        if (simple && !unified) {
            // requires equal splits, so we skip the simple split
            continue;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = simple ? balloc.split_simple(n_ubatch) :
                embd_all ? balloc.split_seq(n_ubatch) : balloc.split_equal(n_ubatch, !unified, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            continue;
        }

        auto sinfos_base = mem_attn->get_base()->prepare(ubatches);
        if (sinfos_base.empty()) {
            continue;
        }

        auto sinfos_swa = mem_attn->get_swa()->prepare(ubatches);
        if (sinfos_swa.empty()) {
            continue;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        // the indexer uses the base cache's slot layout; a separate one can drift from it
        auto sinfos_idx = sinfos_base;

        return std::make_unique<llama_memory_idx_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(sinfos_idx), std::move(ubatches));
    }

    return std::make_unique<llama_memory_idx_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_idx_iswa::init_full() {
    return std::make_unique<llama_memory_idx_iswa_context>(this);
}

llama_memory_context_ptr llama_memory_idx_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_idx_iswa_context>(this, lctx, optimize);
}

bool llama_memory_idx_iswa::get_can_shift() const {
    // the indexer holds raw keys, so a shift never has to rotate them: only attention can refuse one
    return mem_attn->get_can_shift();
}

void llama_memory_idx_iswa::clear(bool data) {
    mem_attn->clear(data);
    mem_idx ->clear(data);
}

bool llama_memory_idx_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;

    res = res & mem_attn->seq_rm(seq_id, p0, p1);
    res = res & mem_idx ->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_memory_idx_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // only whole sequences are copied: a shared cell holds a single pool grouping
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported");

    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_idx ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_idx_iswa::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_idx ->seq_keep(seq_id);
}

void llama_memory_idx_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_idx ->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_idx_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_idx ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_idx_iswa::seq_pos_min(llama_seq_id seq_id) const {
    return mem_attn->seq_pos_min(seq_id);
}

llama_pos llama_memory_idx_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return mem_attn->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_idx_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();

    for (const auto & buft_size : mem_idx->get_cache()->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }

    return mb;
}

void llama_memory_idx_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    mem_attn->state_write(io, seq_id, flags);

    // the indexer section goes last, so it is a pure suffix: an iSWA-only reader stops early instead of misparsing it
    mem_idx->get_cache()->state_write(io, seq_id, flags);
}

void llama_memory_idx_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_kv_cache_iswa::state_read, as the indexer restore adopts the base cache's layout
    llama_kv_cache::slot_info_vec_t sinfos_base;

    try {
        mem_attn->get_base()->state_read_sinfo(io, seq_id, flags, &sinfos_base, nullptr);
        mem_attn->get_swa ()->state_read(io, seq_id, flags);

        // must mirror the write order in state_write
        mem_idx->state_read(io, seq_id, flags, sinfos_base);
    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from both of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_idx_iswa::state_drop(llama_seq_id seq_id) {
    if (seq_id < 0) {
        clear(true);

        return;
    }

    mem_attn->get_base()->state_clear(seq_id);
    mem_attn->get_swa ()->state_clear(seq_id);
    mem_idx->state_clear(seq_id);
}

llama_kv_cache_iswa * llama_memory_idx_iswa::get_mem_attn() const {
    return mem_attn.get();
}

llama_kv_cache_kpool * llama_memory_idx_iswa::get_mem_idx() const {
    return mem_idx.get();
}

//
// llama_memory_idx_iswa_context
//

llama_memory_idx_iswa_context::llama_memory_idx_iswa_context(llama_memory_status status) :
    llama_kv_cache_iswa_context(status), status(status) {}

llama_memory_idx_iswa_context::llama_memory_idx_iswa_context(llama_memory_idx_iswa * mem) :
    llama_kv_cache_iswa_context(mem->get_mem_attn()),
    kpool(mem->get_mem_idx()),
    status(llama_memory_status_combine(llama_kv_cache_iswa_context::get_status(), kpool.get_status())) {}

llama_memory_idx_iswa_context::llama_memory_idx_iswa_context(
        llama_memory_idx_iswa * mem,
                llama_context * lctx,
                         bool   optimize) :
    llama_kv_cache_iswa_context(mem->get_mem_attn(), lctx, optimize),
    kpool(mem->get_mem_idx(), lctx, optimize),
    status(llama_memory_status_combine(llama_kv_cache_iswa_context::get_status(), kpool.get_status())) {}

llama_memory_idx_iswa_context::llama_memory_idx_iswa_context(
        llama_memory_idx_iswa * mem,
              slot_info_vec_t   sinfos_base,
              slot_info_vec_t   sinfos_swa,
              slot_info_vec_t   sinfos_idx,
    std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; the indexer gets a copy of its own
    llama_kv_cache_iswa_context(mem->get_mem_attn(), std::move(sinfos_base), std::move(sinfos_swa), ubatches),
    kpool(mem->get_mem_idx(), std::move(sinfos_idx), std::move(ubatches)),
    status(llama_memory_status_combine(llama_kv_cache_iswa_context::get_status(), kpool.get_status())) {}

bool llama_memory_idx_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    kpool.next();

    return llama_kv_cache_iswa_context::next();
}

bool llama_memory_idx_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    const bool res = llama_kv_cache_iswa_context::apply();

    return kpool.apply(res);
}

llama_memory_status llama_memory_idx_iswa_context::get_status() const {
    return status;
}
