#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
    const llama_model & model,
                         /* attn */
            ggml_type    attn_type_k,
            ggml_type    attn_type_v,
                 bool    attn_v_trans,
             uint32_t    attn_kv_size,
             uint32_t    attn_n_pad,
             uint32_t    attn_n_swa,
       llama_swa_type    attn_swa_type,
                         /* recurrent */
            ggml_type    recurrent_type_k,
            ggml_type    recurrent_type_v,
             uint32_t    recurrent_kv_size,
                         /* common */
             uint32_t    n_seq_max,
                 bool    offload,
                         /* layer filters */
      layer_filter_cb && attn_filter,
      layer_filter_cb && recurrent_filter) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_unified(
        model,
        attn_filter == nullptr ?
            [&](int32_t il) { return !model.hparams.recurrent_layer(il); }
            : attn_filter,
        attn_type_k,
        attn_type_v,
        attn_v_trans,
        offload,
        attn_kv_size,
        n_seq_max,
        attn_n_pad,
        attn_n_swa,
        attn_swa_type
    )),
    mem_recurrent(new llama_memory_recurrent(
        model,
        recurrent_filter == nullptr ?
            [&](int32_t il) { return model.hparams.recurrent_layer(il); }
            : recurrent_filter,
        recurrent_type_k,
        recurrent_type_v,
        offload,
        recurrent_kv_size,
        n_seq_max
    )) {}

llama_memory_state_ptr llama_memory_hybrid::init_batch(const llama_batch & batch, uint32_t n_ubatch, bool embd_pooled) {

    // since this includes a recurrent cache, we cannot use split_simple
    auto sbatch = llama_sbatch(batch, hparams.n_embd, false);

    // follow the recurrent pattern for creating the ubatch splits
    std::vector<llama_ubatch> ubatches;
    while (sbatch.n_tokens > 0) {
        llama_ubatch ubatch;

        if (embd_pooled) {
            // Pooled embeddings cannot be split across ubatches (yet)
            ubatch = sbatch.split_seq(n_ubatch);
        } else {
            ubatch = sbatch.split_equal(n_ubatch);
        }

        ubatches.push_back(ubatch);
    }

    // prepare the recurrent batches first
    if (!mem_recurrent->prepare(ubatches)) {
        // TODO: will the recurrent cache be in an undefined state at this point?
        LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
        return std::make_unique<llama_memory_hybrid_state>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    // prepare the attention cache
    auto heads_attn = mem_attn->prepare(ubatches);
    if (heads_attn.empty()) {
        LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
        return std::make_unique<llama_memory_hybrid_state>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_memory_hybrid_state>(
        this, std::move(sbatch), std::move(heads_attn), std::move(ubatches));
}

llama_memory_state_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_state>(this);
}

llama_memory_state_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_state>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn     ->clear(data);
    mem_recurrent->clear(data);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recurrent->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn     ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recurrent->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn     ->seq_keep(seq_id);
    mem_recurrent->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recurrent->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn     ->seq_div(seq_id, p0, p1, d);
    mem_recurrent->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recurrent->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recurrent->seq_pos_max(seq_id));
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    mem_attn     ->state_write(io, seq_id);
    mem_recurrent->state_write(io, seq_id);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    mem_attn     ->state_read(io, seq_id);
    mem_recurrent->state_read(io, seq_id);
}

llama_kv_cache_unified * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recurrent() const {
    return mem_recurrent.get();
}

llama_memory_hybrid_state::llama_memory_hybrid_state(llama_memory_status status) : status(status) {}

llama_memory_hybrid_state::llama_memory_hybrid_state(llama_memory_hybrid * mem) :
    state_attn     (mem->get_mem_attn     ()->init_full()),
    state_recurrent(mem->get_mem_recurrent()->init_full()),
    status(llama_memory_status_combine(state_attn->get_status(), state_recurrent->get_status())) {
}

llama_memory_hybrid_state::llama_memory_hybrid_state(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    state_attn     (mem->get_mem_attn     ()->init_update(lctx, optimize)),
    state_recurrent(mem->get_mem_recurrent()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(state_attn->get_status(), state_recurrent->get_status())) {
}

llama_memory_hybrid_state::llama_memory_hybrid_state(
              llama_memory_hybrid * mem,
                     llama_sbatch   sbatch,
            std::vector<uint32_t>   heads_attn,
        std::vector<llama_ubatch>   ubatches) :
    sbatch(std::move(sbatch)),
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    state_attn     (new llama_kv_cache_unified_state(mem->get_mem_attn(),      {}, std::move(heads_attn), this->ubatches)),
    state_recurrent(new llama_memory_recurrent_state(mem->get_mem_recurrent(), {},                        this->ubatches)),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
}

bool llama_memory_hybrid_state::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    state_attn     ->next();
    state_recurrent->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_state::apply() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    bool res = true;

    res = res & state_attn     ->apply();
    res = res & state_recurrent->apply();

    return res;
}

std::vector<int64_t> & llama_memory_hybrid_state::out_ids() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return sbatch.out_ids;
}

llama_memory_status llama_memory_hybrid_state::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_state::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_unified_state * llama_memory_hybrid_state::get_state_attn() const {
    return static_cast<const llama_kv_cache_unified_state *>(state_attn.get());
}

const llama_memory_recurrent_state * llama_memory_hybrid_state::get_state_recurrent() const {
    return static_cast<const llama_memory_recurrent_state *>(state_recurrent.get());
}
