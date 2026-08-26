#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_lid,
    const layer_filter_cb & filter_ple) :
    hparams(model.hparams),
    hparams_lid(model.hparams),
    hparams_ple(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        model.hparams,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {
    if (filter_lid) {
        // the indexer stores one key per token, indexer_head_size wide
        std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
        hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
        // the indexer has no values, but llama_kv_cache always allocates V - keep it narrow
        hparams_lid.n_embd_head_v_full = model.hparams.indexer_head_size;

        mem_lid = std::make_unique<llama_kv_cache>(
                model, hparams_lid, type_k, type_v, v_trans, offload, unified,
                kv_size, n_seq_max, n_pad, n_swa, swa_type, nullptr, filter_lid, nullptr, nullptr);
    }

    if (filter_ple) {
        // its geometry is its own, so override what n_embd_r()/n_embd_s() would derive from the
        // ssm sizes. Same trick as hparams_lid above
        GGML_ASSERT(hparams.n_embd_r_2nd != 0 && hparams.n_embd_s_2nd != 0 &&
                "a second recurrent state needs n_embd_r_2nd/n_embd_s_2nd");
        hparams_ple.n_embd_r_override = hparams.n_embd_r_2nd;
        hparams_ple.n_embd_s_override = hparams.n_embd_s_2nd;

        mem_ple = std::make_unique<llama_memory_recurrent>(
                model, hparams_ple, type_r, type_s, offload, rs_size, n_seq_max, n_rs_seq, filter_ple);
    }
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        if (mem_ple && !mem_ple->prepare(ubatches)) {
            LLAMA_LOG_ERROR("%s: failed to prepare the second recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer cache mirrors the attention cache, so it uses the same ubatch splits
        llama_kv_cache::slot_info_vec_t heads_lid;
        if (mem_lid) {
            heads_lid = mem_lid->prepare(ubatches);
            if (heads_lid.empty()) {
                LLAMA_LOG_ERROR("%s: failed to prepare indexer ubatches\n", __func__);
                return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }
            GGML_ASSERT(heads_lid.size() == heads_attn.size());
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(heads_lid), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
    if (mem_lid) {
        mem_lid->clear(data);
    }
    if (mem_ple) {
        mem_ple->clear(data);
    }
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (mem_lid) {
        mem_lid->seq_rm(seq_id, p0, p1);
    }
    if (mem_ple) {
        mem_ple->seq_rm(seq_id, p0, p1);
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_lid) {
        mem_lid->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    if (mem_ple) {
        mem_ple->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
    if (mem_lid) {
        mem_lid->seq_keep(seq_id);
    }
    if (mem_ple) {
        mem_ple->seq_keep(seq_id);
    }
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
    if (mem_lid) {
        mem_lid->seq_add(seq_id, p0, p1, shift);
    }
    if (mem_ple) {
        mem_ple->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
    if (mem_lid) {
        mem_lid->seq_div(seq_id, p0, p1, d);
    }
    if (mem_ple) {
        mem_ple->seq_div(seq_id, p0, p1, d);
    }
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    if (mem_ple) {
        for (const auto & buft_size : mem_ple->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    if (mem_lid) {
        for (const auto & buft_size : mem_lid->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
        if (mem_lid) {
            mem_lid->state_write(io, seq_id, flags);
        }
    }
    if (mem_ple) {
        mem_ple->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
        if (mem_lid) {
            mem_lid->state_read(io, seq_id, flags);
        }
    }
    if (mem_ple) {
        mem_ple->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_kv_cache * llama_memory_hybrid::get_mem_lid() const {
    return mem_lid.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_ple() const {
    return mem_ple.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    ctx_lid(mem->get_mem_lid() ? mem->get_mem_lid()->init_full() : nullptr),
    ctx_ple(mem->get_mem_ple() ? mem->get_mem_ple()->init_full() : nullptr),
    status(llama_memory_status_combine(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status()),
                ctx_lid ? ctx_lid->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    ctx_lid(mem->get_mem_lid() ? mem->get_mem_lid()->init_update(lctx, optimize) : nullptr),
    ctx_ple(mem->get_mem_ple() ? mem->get_mem_ple()->init_update(lctx, optimize) : nullptr),
    status(llama_memory_status_combine(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status()),
                ctx_lid ? ctx_lid->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
                  slot_info_vec_t   sinfos_lid,
        std::vector<llama_ubatch>   ubatches) :
    mem(mem),
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    ctx_lid(mem->get_mem_lid() ?
        std::make_unique<llama_kv_cache_context>(mem->get_mem_lid(), std::move(sinfos_lid), this->ubatches) : nullptr),
    ctx_ple(mem->get_mem_ple() ?
        std::make_unique<llama_memory_recurrent_context>(mem->get_mem_ple(), this->ubatches) : nullptr),
    status(llama_memory_status_combine(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status()),
                ctx_lid ? ctx_lid->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();
    if (ctx_lid) {
        ctx_lid->next();
    }
    if (ctx_ple) {
        ctx_ple->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();
    if (ctx_lid) {
        res = res & ctx_lid->apply();
    }
    if (ctx_ple) {
        res = res & ctx_ple->apply();
    }

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_lid() const {
    return static_cast<const llama_kv_cache_context *>(ctx_lid.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_ple() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_ple.get());
}

uint32_t llama_memory_hybrid_context::get_n_pos(uint32_t n_pad, uint32_t n_seq_max) const {
    llama_pos pos_max = -1;

    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq_max; ++seq_id) {
        pos_max = std::max(pos_max, mem->get_mem_attn()->seq_pos_max(seq_id));
    }

    return std::max(n_pad, GGML_PAD((uint32_t) (pos_max + 1), n_pad));
}

void llama_memory_hybrid_context::set_input_pos_cell(ggml_tensor * dst, const llama_ubatch * ubatch, bool lid, int64_t n_rows) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32 || dst->type == GGML_TYPE_F32);

    const llama_kv_cache * kv = lid ? mem->get_mem_lid() : mem->get_mem_attn();
    GGML_ASSERT(kv);

    const int64_t n_pos = dst->ne[0];
    const int64_t ns    = dst->ne[1];

    GGML_ASSERT(ubatch->n_tokens % ns == 0);
    const int64_t n_tps = ubatch->n_tokens/ns;

    for (int64_t s = 0; s < ns; ++s) {
        const llama_seq_id seq_id = ubatch->seq_id[s*n_tps][0];

        const auto & cells = kv->get_cells(seq_id);

        std::vector<int32_t> map(n_pos, 0);

        for (uint32_t j = 0; j < cells.size(); ++j) {
            if (cells.is_empty(j) || !cells.seq_has(j, seq_id)) {
                continue;
            }

            const llama_pos p0 = cells.pos_get(j);

            if (p0 >= 0 && p0 < n_pos && (int64_t) j < n_rows) {
                map[p0] = (int32_t) j;
            }
        }

        if (dst->type == GGML_TYPE_I32) {
            std::copy(map.begin(), map.end(), (int32_t *) dst->data + s*n_pos);
        } else {
            float * data = (float *) dst->data + s*n_pos;
            for (int64_t p = 0; p < n_pos; ++p) {
                data[p] = (float) map[p];
            }
        }
    }
}

void llama_memory_hybrid_context::set_input_cell_blk(ggml_tensor * dst, const llama_ubatch * ubatch, int32_t div) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    const int64_t n_kv = dst->ne[0];
    const int64_t ns   = dst->ne[1];

    GGML_ASSERT(ubatch->n_tokens % ns == 0);
    const int64_t n_tps = ubatch->n_tokens/ns;

    int32_t * data = (int32_t *) dst->data;

    for (int64_t s = 0; s < ns; ++s) {
        const llama_seq_id seq_id = ubatch->seq_id[s*n_tps][0];

        const auto & cells = mem->get_mem_attn()->get_cells(seq_id);

        for (int64_t j = 0; j < n_kv; ++j) {
            data[s*n_kv + j] =
                cells.is_empty(j) || !cells.seq_has(j, seq_id)
                    ? 0
                    : (int32_t) (cells.pos_get(j)/div);
        }
    }
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
