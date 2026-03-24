#include "llama-kv-cache-dsa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>

//
// llama_kv_cache_dsa
//

llama_kv_cache_dsa::llama_kv_cache_dsa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    n_stream(unified ? 1 : n_seq_max) {

    LLAMA_LOG_INFO("%s: creating main KV cache, size = %u cells\n", __func__, kv_size);

    kv_base = std::make_unique<llama_kv_cache>(
            model, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, filter, reuse);

    LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

    kv_ik = std::make_unique<llama_ik_cache>(
            model, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, filter, reuse);
}

void llama_kv_cache_dsa::clear(bool data) {
    kv_base->clear(data);
    kv_ik ->clear(data);
}

bool llama_kv_cache_dsa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_ik ->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_kv_cache_dsa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_ik ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_dsa::seq_keep(llama_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_ik ->seq_keep(seq_id);
}

void llama_kv_cache_dsa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_ik ->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_dsa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_ik ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_dsa::seq_pos_min(llama_seq_id seq_id) const {
    return kv_base->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_dsa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_base->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_ik->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llama_memory_context_ptr llama_kv_cache_dsa::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_ik = kv_ik->prepare(ubatches);
        if (sinfos_ik.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_ik.size());

        return std::make_unique<llama_kv_cache_dsa_context>(
                this, std::move(sinfos_base), std::move(sinfos_ik), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_dsa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_dsa::init_full() {
    return std::make_unique<llama_kv_cache_dsa_context>(this);
}

llama_memory_context_ptr llama_kv_cache_dsa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_dsa_context>(this, lctx, optimize);
}

bool llama_kv_cache_dsa::get_can_shift() const {
    return kv_base->get_can_shift() &&
           kv_ik->get_can_shift() &&
           kv_base->get_size() == kv_ik->get_size();
}

void llama_kv_cache_dsa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv_base->state_write(io, seq_id, flags);
    kv_ik->state_write(io, seq_id, flags);
}

void llama_kv_cache_dsa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    kv_base->state_read(io, seq_id, flags);
    kv_ik->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_kv_cache_dsa::get_base() const {
    return kv_base.get();
}

llama_ik_cache * llama_kv_cache_dsa::get_ik() const {
    return kv_ik.get();
}

//
// llama_kv_cache_dsa_context
//

llama_kv_cache_dsa_context::llama_kv_cache_dsa_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsa_context::llama_kv_cache_dsa_context(
        llama_kv_cache_dsa * kv) :
    ctx_base(kv->get_base()->init_full()),
    ctx_ik(kv->get_ik()->init_full()),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_ik->get_status())) {
}

llama_kv_cache_dsa_context::llama_kv_cache_dsa_context(
        llama_kv_cache_dsa * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_ik(kv->get_ik()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_ik->get_status())) {
}

llama_kv_cache_dsa_context::llama_kv_cache_dsa_context(
        llama_kv_cache_dsa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_ik,
        std::vector<llama_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_base(new llama_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_ik(new llama_ik_cache_context(kv->get_ik(), std::move(sinfos_ik), this->ubatches)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_ik->get_status())) {
}

llama_kv_cache_dsa_context:: ~llama_kv_cache_dsa_context() = default;

bool llama_kv_cache_dsa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_ik ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_ik ->apply();

    return res;
}

llama_memory_status llama_kv_cache_dsa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_context * llama_kv_cache_dsa_context::get_base() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_base.get());
}

const llama_ik_cache_context * llama_kv_cache_dsa_context::get_ik()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_ik_cache_context *>(ctx_ik.get());
}
