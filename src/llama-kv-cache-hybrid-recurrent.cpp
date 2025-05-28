#include "llama-kv-cache-hybrid-recurrent.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

//
// llama_kv_cache_hybrid_recurrent
//

llama_kv_cache_hybrid_recurrent::llama_kv_cache_hybrid_recurrent(
    const llama_model & model,
                        /* attn */
            ggml_type   attn_type_k,
            ggml_type   attn_type_v,
                 bool   attn_v_trans,
             uint32_t   attn_kv_size,
             uint32_t   attn_n_pad,
             uint32_t   attn_n_swa,
       llama_swa_type   attn_swa_type,
                        /* recurrent */
            ggml_type   recurrent_type_k,
            ggml_type   recurrent_type_v,
             uint32_t   recurrent_kv_size,
                        /* common */
             uint32_t   n_seq_max,
                 bool   offload) :
    hparams(model.hparams),
    kv_attn(new llama_kv_cache_unified(
        model,
        [&](int32_t il) { return !model.hparams.recurrent_layer(il); },
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
    kv_recurrent(new llama_kv_cache_recurrent(
        model,
        [&](int32_t il) { return model.hparams.recurrent_layer(il); },
        recurrent_type_k,
        recurrent_type_v,
        offload,
        recurrent_kv_size,
        n_seq_max
    )) {}

void llama_kv_cache_hybrid_recurrent::clear() {
    kv_attn     ->clear();
    kv_recurrent->clear();
}

bool llama_kv_cache_hybrid_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!kv_recurrent->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return kv_attn->seq_rm(seq_id, p0, p1);
}

void llama_kv_cache_hybrid_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_attn     ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_recurrent->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_hybrid_recurrent::seq_keep(llama_seq_id seq_id) {
    kv_attn     ->seq_keep(seq_id);
    kv_recurrent->seq_keep(seq_id);
}

void llama_kv_cache_hybrid_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_attn->seq_add(seq_id, p0, p1, shift);
    kv_recurrent->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_hybrid_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_attn     ->seq_div(seq_id, p0, p1, d);
    kv_recurrent->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_hybrid_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(kv_attn->seq_pos_min(seq_id), kv_recurrent->seq_pos_min(seq_id));
}

llama_pos llama_kv_cache_hybrid_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(kv_attn->seq_pos_max(seq_id), kv_recurrent->seq_pos_max(seq_id));
}

llama_memory_state_ptr llama_kv_cache_hybrid_recurrent::init_batch(const llama_batch & batch, uint32_t n_ubatch, bool embd_pooled, bool logits_all) {

    // since this includes a recurrent cache, we cannot use split_simple
    auto sbatch = llama_sbatch(batch, hparams.n_embd, false, logits_all);

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
    if (!kv_recurrent->prepare(ubatches)) {
        // TODO: will the recurrent cache be in an undefined state at this point?
        LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
        return std::make_unique<llama_kv_cache_hybrid_recurrent_state>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    // prepare the attention cache
    auto heads_attn = kv_attn->prepare(ubatches);
    if (heads_attn.empty()) {
        LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
        return std::make_unique<llama_kv_cache_hybrid_recurrent_state>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_kv_cache_hybrid_recurrent_state>(
        this, std::move(sbatch), std::move(heads_attn), std::move(ubatches));
}

llama_memory_state_ptr llama_kv_cache_hybrid_recurrent::init_full() {
    return std::make_unique<llama_kv_cache_hybrid_recurrent_state>(this);
}

bool llama_kv_cache_hybrid_recurrent::update(llama_context & lctx) {
    bool res = false;

    res = res | kv_attn     ->update(lctx);
    res = res | kv_recurrent->update(lctx);

    return res;
}

void llama_kv_cache_hybrid_recurrent::defrag_sched(float thold) {
    kv_attn     ->defrag_sched(thold);
    kv_recurrent->defrag_sched(thold);
}

bool llama_kv_cache_hybrid_recurrent::get_can_shift() const {
    // TODO: Should this return true if the attention cache can shift?
    return false;
}

void llama_kv_cache_hybrid_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    kv_attn     ->state_write(io, seq_id);
    kv_recurrent->state_write(io, seq_id);
}

void llama_kv_cache_hybrid_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    kv_attn     ->state_read(io, seq_id);
    kv_recurrent->state_read(io, seq_id);
}

llama_kv_cache_unified * llama_kv_cache_hybrid_recurrent::get_kv_attn() const {
    return kv_attn.get();
}

llama_kv_cache_recurrent * llama_kv_cache_hybrid_recurrent::get_kv_recurrent() const {
    return kv_recurrent.get();
}

llama_kv_cache_hybrid_recurrent_state::llama_kv_cache_hybrid_recurrent_state(llama_memory_status status)
    : status(status), state_attn(status), state_recurrent(status) {}

llama_kv_cache_hybrid_recurrent_state::llama_kv_cache_hybrid_recurrent_state(llama_kv_cache_hybrid_recurrent * kv)
    : status(LLAMA_MEMORY_STATUS_SUCCESS),
      kv(kv),
      state_attn(status, kv->get_kv_attn()),
      state_recurrent(status, kv->get_kv_recurrent()) {}

llama_kv_cache_hybrid_recurrent_state::llama_kv_cache_hybrid_recurrent_state(
    llama_kv_cache_hybrid_recurrent * kv,
                       llama_sbatch   sbatch,
              std::vector<uint32_t>   heads_attn,
          std::vector<llama_ubatch>   ubatches)
    : status(LLAMA_MEMORY_STATUS_SUCCESS),
      kv(kv),
      sbatch(std::move(sbatch)),
      heads_attn(std::move(heads_attn)),
      ubatches(std::move(ubatches)),
      // NOTE: these child states are only used as wrapper APIs for the
      //    const methods, so we use the "init full" signature since the
      //    actual state is not used.
      state_attn(LLAMA_MEMORY_STATUS_SUCCESS, kv->get_kv_attn()),
      state_recurrent(LLAMA_MEMORY_STATUS_SUCCESS, kv->get_kv_recurrent()) {}


bool llama_kv_cache_hybrid_recurrent_state::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_hybrid_recurrent_state::apply() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    kv->get_kv_attn()     ->apply_ubatch(heads_attn[i_next], ubatches[i_next]);
    kv->get_kv_recurrent()->find_slot(ubatches[i_next]);

    return true;
}

std::vector<int64_t> & llama_kv_cache_hybrid_recurrent_state::out_ids() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return sbatch.out_ids;
}

llama_memory_status llama_kv_cache_hybrid_recurrent_state::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_hybrid_recurrent_state::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_unified_state * llama_kv_cache_hybrid_recurrent_state::get_state_attn () const {
    return &state_attn;
}

const llama_kv_cache_recurrent_state * llama_kv_cache_hybrid_recurrent_state::get_state_recurrent() const {
    return &state_recurrent;
}
