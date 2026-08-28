#include "llama-kv-cache-dsa-iswa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>

//
// llama_kv_cache_dsa_iswa
//

llama_kv_cache_dsa_iswa::llama_kv_cache_dsa_iswa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
    const layer_filter_cb & filter_mla,
    const layer_filter_cb & filter_lid,
    const  layer_reuse_cb & reuse) : unified(unified) {

    const auto & hparams = model.hparams;

    // chain filters
    const layer_filter_cb filter_dsa = [&](int32_t il) {
        if (filter_mla && !filter_mla(il)) {
            return false;
        }

        return !hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa = [&](int32_t il) {
        if (filter_mla && !filter_mla(il)) {
            return false;
        }

        return hparams.is_swa(il);
    };

    const uint32_t size_dsa = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_dsa, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_dsa;
    }

    LLAMA_LOG_INFO("%s: creating DSA KV cache, size = %u cells\n", __func__, size_dsa);

    kv_dsa = std::make_unique<llama_kv_cache_dsa>(
            model, type_k, type_v,
            v_trans, offload, unified, size_dsa, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, filter_dsa, filter_lid, reuse);

    LLAMA_LOG_INFO("%s: creating SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, nullptr, filter_swa, reuse, nullptr);
}

void llama_kv_cache_dsa_iswa::clear(bool data) {
    kv_dsa->clear(data);
    kv_swa->clear(data);
}

bool llama_kv_cache_dsa_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;

    res = res & kv_dsa->seq_rm(seq_id, p0, p1);
    res = res & kv_swa->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_kv_cache_dsa_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_dsa->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_dsa_iswa::seq_keep(llama_seq_id seq_id) {
    kv_dsa->seq_keep(seq_id);
    kv_swa->seq_keep(seq_id);
}

void llama_kv_cache_dsa_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_dsa->seq_add(seq_id, p0, p1, shift);
    kv_swa->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_dsa_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_dsa->seq_div(seq_id, p0, p1, d);
    kv_swa->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_dsa_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the DSA cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_dsa_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsa_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_dsa->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llama_memory_context_ptr llama_kv_cache_dsa_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_mla = kv_dsa->get_mla()->prepare(ubatches);
        if (sinfos_mla.empty()) {
            break;
        }

        auto sinfos_lid = kv_dsa->get_lid()->prepare(ubatches);
        if (sinfos_lid.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_mla.size() == sinfos_swa.size());

        return std::make_unique<llama_kv_cache_dsa_iswa_context>(
                this, std::move(sinfos_mla), std::move(sinfos_lid), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_mla = kv_dsa->get_mla()->prepare(ubatches);
        if (sinfos_mla.empty()) {
            break;
        }

        auto sinfos_lid = kv_dsa->get_lid()->prepare(ubatches);
        if (sinfos_lid.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_mla.size() == sinfos_swa.size());

        return std::make_unique<llama_kv_cache_dsa_iswa_context>(
                this, std::move(sinfos_mla), std::move(sinfos_lid), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_dsa_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_dsa_iswa::init_full() {
    return std::make_unique<llama_kv_cache_dsa_iswa_context>(this);
}

llama_memory_context_ptr llama_kv_cache_dsa_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_dsa_iswa_context>(this, lctx, optimize);
}

bool llama_kv_cache_dsa_iswa::get_can_shift() const {
    return kv_dsa->get_can_shift() &&
           kv_swa->get_can_shift() &&
           kv_dsa->get_mla()->get_size() == kv_swa->get_size();
}

void llama_kv_cache_dsa_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_dsa->state_write(io, seq_id, flags);
    }

    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_dsa_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_dsa->state_read(io, seq_id, flags);
    }

    kv_swa->state_read(io, seq_id, flags);
}

llama_kv_cache_dsa * llama_kv_cache_dsa_iswa::get_dsa() const {
    return kv_dsa.get();
}

llama_kv_cache * llama_kv_cache_dsa_iswa::get_swa() const {
    return kv_swa.get();
}

//
// llama_kv_cache_dsa_iswa_context
//

llama_kv_cache_dsa_iswa_context::llama_kv_cache_dsa_iswa_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsa_iswa_context::llama_kv_cache_dsa_iswa_context(
        llama_kv_cache_dsa_iswa * kv) :
    ctx_dsa(kv->get_dsa()->init_full()),
    ctx_swa(kv->get_swa()->init_full()),
    status(llama_memory_status_combine(ctx_dsa->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_dsa_iswa_context::llama_kv_cache_dsa_iswa_context(
        llama_kv_cache_dsa_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_dsa(kv->get_dsa()->init_update(lctx, optimize)),
    ctx_swa(kv->get_swa()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_dsa->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_dsa_iswa_context::llama_kv_cache_dsa_iswa_context(
        llama_kv_cache_dsa_iswa * kv,
        slot_info_vec_t sinfos_mla,
        slot_info_vec_t sinfos_lid,
        slot_info_vec_t sinfos_swa,
        std::vector<llama_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_dsa(new llama_kv_cache_dsa_context(kv->get_dsa(), std::move(sinfos_mla), std::move(sinfos_lid), this->ubatches)),
    ctx_swa(new llama_kv_cache_context(kv->get_swa(), std::move(sinfos_swa), this->ubatches)),
    status(llama_memory_status_combine(ctx_dsa->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_dsa_iswa_context:: ~llama_kv_cache_dsa_iswa_context() = default;

bool llama_kv_cache_dsa_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_dsa->next();
    ctx_swa->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsa_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_dsa->apply();
    res = res & ctx_swa->apply();

    return res;
}

llama_memory_status llama_kv_cache_dsa_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsa_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_dsa_context * llama_kv_cache_dsa_iswa_context::get_dsa() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_dsa_context *>(ctx_dsa.get());
}

const llama_kv_cache_context * llama_kv_cache_dsa_iswa_context::get_swa() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_swa.get());
}
