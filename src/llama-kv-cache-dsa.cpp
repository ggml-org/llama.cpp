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
    hparams_lid(model.hparams), n_stream(unified ? 1 : n_seq_max) {

    LLAMA_LOG_INFO("%s: creating main KV cache, size = %u cells\n", __func__, kv_size);

    kv_mla = std::make_unique<llama_kv_cache>(
            model, model.hparams, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, nullptr, filter, reuse, nullptr);

    // we use llama_kv_cache for caching indexer keys
    // by hand-tweaking some hparams we fool it to create
    // indexer key cache tensors with correct dimensions
    // https://github.com/ggml-org/llama.cpp/pull/21149#discussion_r3015940823

    // DSA lightning indexer uses MQA with single key head
    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;

    LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

    kv_lid = std::make_unique<llama_kv_cache>(
            model, hparams_lid, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, nullptr, filter, reuse, nullptr);
}

void llama_kv_cache_dsa::clear(bool data) {
    kv_mla->clear(data);
    kv_lid->clear(data);
}

bool llama_kv_cache_dsa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;

    res = res & kv_mla->seq_rm(seq_id, p0, p1);
    res = res & kv_lid->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_kv_cache_dsa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_mla->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_lid->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_dsa::seq_keep(llama_seq_id seq_id) {
    kv_mla->seq_keep(seq_id);
    kv_lid->seq_keep(seq_id);
}

void llama_kv_cache_dsa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_mla->seq_add(seq_id, p0, p1, shift);
    kv_lid->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_dsa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_mla->seq_div(seq_id, p0, p1, d);
    kv_lid->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_dsa::seq_pos_min(llama_seq_id seq_id) const {
    return kv_mla->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_dsa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_mla->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_mla->memory_breakdown();
    for (const auto & buft_size : kv_lid->memory_breakdown()) {
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
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_mla = kv_mla->prepare(ubatches);
        if (sinfos_mla.empty()) {
            break;
        }

        auto sinfos_lid = kv_lid->prepare(ubatches);
        if (sinfos_lid.empty()) {
            break;
        }

        assert(sinfos_mla.size() == sinfos_lid.size());

        return std::make_unique<llama_kv_cache_dsa_context>(
                this, std::move(sinfos_mla), std::move(sinfos_lid), std::move(ubatches));
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
    return kv_mla->get_can_shift() &&
           kv_lid->get_can_shift() &&
           kv_mla->get_size() == kv_lid->get_size();
}

void llama_kv_cache_dsa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv_mla->state_write(io, seq_id, flags);
    kv_lid->state_write(io, seq_id, flags);
}

void llama_kv_cache_dsa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    kv_mla->state_read(io, seq_id, flags);
    kv_lid->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_kv_cache_dsa::get_mla() const {
    return kv_mla.get();
}

llama_kv_cache * llama_kv_cache_dsa::get_lid() const {
    return kv_lid.get();
}

//
// llama_kv_cache_dsa_context
//

llama_kv_cache_dsa_context::llama_kv_cache_dsa_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsa_context::llama_kv_cache_dsa_context(
        llama_kv_cache_dsa * kv) :
    ctx_mla(kv->get_mla()->init_full()),
    ctx_lid(kv->get_lid()->init_full()),
    status(llama_memory_status_combine(ctx_mla->get_status(), ctx_lid->get_status())) {
}

llama_kv_cache_dsa_context::llama_kv_cache_dsa_context(
        llama_kv_cache_dsa * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_mla(kv->get_mla()->init_update(lctx, optimize)),
    ctx_lid(kv->get_lid()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_mla->get_status(), ctx_lid->get_status())) {
}

llama_kv_cache_dsa_context::llama_kv_cache_dsa_context(
        llama_kv_cache_dsa * kv,
        slot_info_vec_t sinfos_mla,
        slot_info_vec_t sinfos_lid,
        std::vector<llama_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_mla(new llama_kv_cache_context(kv->get_mla(), std::move(sinfos_mla), this->ubatches)),
    ctx_lid(new llama_kv_cache_context(kv->get_lid(), std::move(sinfos_lid), this->ubatches)),
    status(llama_memory_status_combine(ctx_mla->get_status(), ctx_lid->get_status())) {
}

llama_kv_cache_dsa_context:: ~llama_kv_cache_dsa_context() = default;

bool llama_kv_cache_dsa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_mla->next();
    ctx_lid->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_mla->apply();
    res = res & ctx_lid->apply();

    return res;
}

llama_memory_status llama_kv_cache_dsa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_context * llama_kv_cache_dsa_context::get_mla() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_mla.get());
}

const llama_kv_cache_context * llama_kv_cache_dsa_context::get_lid()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_lid.get());
}
