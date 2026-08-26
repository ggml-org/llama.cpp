#include "llama-memory-hybrid-idx.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams(model.hparams),
    n_seq_max(n_seq_max),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "cache_idx");
    }()),
    mem_ple(model.hparams.ple_n_heads == 0 ? nullptr : new llama_memory_recurrent(
            model, GGML_TYPE_F32, GGML_TYPE_F32, offload, rs_size, n_seq_max, n_rs_seq,
            [&model](uint32_t il) { return model.hparams.is_ple(il); },
            model.hparams.ple_conv_state(), 1, "cache_ple")) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(
        llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // Repeated from llama_memory_hybrid::init_batch because the indexer must
    // receive the attention cache's exact slot infos.
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (embd_all) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                const bool unified = get_mem_attn()->get_n_stream() == 1;
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;
                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }
        if (!get_mem_recr()->prepare(ubatches) || (mem_ple && !mem_ple->prepare(ubatches))) {
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

bool llama_memory_hybrid_idx::get_can_shift() const {
    return llama_memory_hybrid::get_can_shift() &&
           (!mem_idx || (mem_idx->get_can_shift() && mem_idx->get_size() == get_mem_attn()->get_size())) &&
           (!mem_ple || mem_ple->get_can_shift());
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);
    if (mem_idx) {
        mem_idx->clear(data);
    }
    if (mem_ple) {
        mem_ple->clear(data);
    }
    ple_history.clear();
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Recurrent removal is the participant that may refuse. Do it first so a
    // failed partial rollback leaves attention, indexer, and PLE state intact.
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (mem_ple && !mem_ple->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (!get_mem_attn()->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }
    ple_seq_rm(seq_id, p0, p1);
    return true;
}

void llama_memory_hybrid_idx::seq_cp(
        llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    if (mem_ple) {
        mem_ple->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    ple_seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);
    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
    if (mem_ple) {
        mem_ple->seq_keep(seq_id);
    }
    ple_seq_keep(seq_id);
}

void llama_memory_hybrid_idx::seq_add(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);
    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
    if (mem_ple) {
        mem_ple->seq_add(seq_id, p0, p1, shift);
    }
    ple_seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_idx::seq_div(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
    if (mem_ple) {
        mem_ple->seq_div(seq_id, p0, p1, d);
    }
    ple_seq_div(seq_id, p0, p1, d);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    auto mb = llama_memory_hybrid::memory_breakdown();
    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    if (mem_ple) {
        for (const auto & buft_size : mem_ple->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    return mb;
}

void llama_memory_hybrid_idx::state_write(
        llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        get_mem_attn()->state_write(io, seq_id, flags);
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }
    get_mem_recr()->state_write(io, seq_id, flags);
    if (mem_ple) {
        mem_ple->state_write(io, seq_id, flags);
    }
    if (hparams.ple_n_heads > 0) {
        ple_state_write(io, seq_id);
    }
}

void llama_memory_hybrid_idx::state_read(
        llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        get_mem_attn()->state_read(io, seq_id, flags);
        if (mem_idx) {
            mem_idx->state_read(io, seq_id, flags);
        }
    }
    get_mem_recr()->state_read(io, seq_id, flags);
    if (mem_ple) {
        mem_ple->state_read(io, seq_id, flags);
    }
    if (hparams.ple_n_heads > 0) {
        ple_state_read(io, seq_id);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

llama_memory_recurrent * llama_memory_hybrid_idx::get_mem_ple() const {
    return mem_ple.get();
}

void llama_memory_hybrid_idx::ple_append(
        llama_seq_id seq_id, llama_pos pos, llama_token token) {
    if (hparams.ple_n_heads == 0 || seq_id < 0 || pos < 0 || token < 0) {
        return;
    }

    auto & history = ple_history[seq_id];
    history.push_back({pos, token});

    const uint64_t max_entries_u64 =
            (uint64_t) get_mem_attn()->get_size() + hparams.ple_ngram_size + 1;
    const size_t max_entries = (size_t) std::min<uint64_t>(
            max_entries_u64, std::numeric_limits<size_t>::max());
    if (history.size() > max_entries) {
        history.erase(history.begin(), history.begin() + (history.size() - max_entries));
    }
}

void llama_memory_hybrid_idx::ple_apply_ubatch(
        const llama_ubatch & ubatch, ple_prefix_map & prefix) {
    prefix.clear();
    if (hparams.ple_n_heads == 0) {
        return;
    }

    const size_t lookback = hparams.ple_ngram_size > 0 ? hparams.ple_ngram_size - 1 : 0;
    for (int64_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            if (seq_id < 0 || prefix.count(seq_id) != 0) {
                continue;
            }

            auto & tail = prefix[seq_id];
            const auto found = ple_history.find(seq_id);
            if (found == ple_history.end()) {
                continue;
            }
            const auto & history = found->second;
            const size_t first = history.size() > lookback ? history.size() - lookback : 0;
            tail.reserve(history.size() - first);
            for (size_t j = first; j < history.size(); ++j) {
                tail.push_back(history[j].token);
            }
        }
    }

    const llama_token embd_token = (llama_token) hparams.ple_image_token_id;
    for (int64_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_token token = ubatch.token ? ubatch.token[i] : embd_token;
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            bool duplicate = false;
            for (int32_t prev = 0; prev < s; ++prev) {
                duplicate |= ubatch.seq_id[i][prev] == seq_id;
            }
            if (!duplicate) {
                ple_append(seq_id, ubatch.pos[i], token);
            }
        }
    }
}

void llama_memory_hybrid_idx::ple_seq_rm(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (hparams.ple_n_heads == 0) {
        return;
    }
    p0 = std::max<llama_pos>(p0, 0);
    p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;

    auto erase_range = [p0, p1](ple_seq_history & history) {
        history.erase(std::remove_if(history.begin(), history.end(), [p0, p1](const ple_history_entry & entry) {
            return p0 <= entry.pos && entry.pos < p1;
        }), history.end());
    };

    if (seq_id >= 0) {
        auto it = ple_history.find(seq_id);
        if (it != ple_history.end()) {
            erase_range(it->second);
            if (it->second.empty()) {
                ple_history.erase(it);
            }
        }
    } else {
        for (auto it = ple_history.begin(); it != ple_history.end();) {
            erase_range(it->second);
            it = it->second.empty() ? ple_history.erase(it) : std::next(it);
        }
    }
}

void llama_memory_hybrid_idx::ple_seq_cp(
        llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (hparams.ple_n_heads == 0 || seq_id_src == seq_id_dst) {
        return;
    }
    const auto src = ple_history.find(seq_id_src);
    if (src == ple_history.end()) {
        return;
    }
    p0 = std::max<llama_pos>(p0, 0);
    p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;

    ple_seq_history copied;
    for (const auto & entry : src->second) {
        if (p0 <= entry.pos && entry.pos < p1) {
            copied.push_back(entry);
        }
    }
    if (copied.empty()) {
        return;
    }

    auto & dst = ple_history[seq_id_dst];
    auto first = std::find_if(dst.begin(), dst.end(), [p0](const ple_history_entry & entry) {
        return entry.pos >= p0;
    });
    const size_t insert_at = first - dst.begin();
    dst.erase(std::remove_if(dst.begin(), dst.end(), [p0, p1](const ple_history_entry & entry) {
        return p0 <= entry.pos && entry.pos < p1;
    }), dst.end());
    dst.insert(dst.begin() + std::min(insert_at, dst.size()), copied.begin(), copied.end());

    const uint64_t max_entries_u64 =
            (uint64_t) get_mem_attn()->get_size() + hparams.ple_ngram_size + 1;
    const size_t max_entries = (size_t) std::min<uint64_t>(
            max_entries_u64, std::numeric_limits<size_t>::max());
    if (dst.size() > max_entries) {
        dst.erase(dst.begin(), dst.begin() + (dst.size() - max_entries));
    }
}

void llama_memory_hybrid_idx::ple_seq_keep(llama_seq_id seq_id) {
    if (hparams.ple_n_heads == 0) {
        return;
    }
    for (auto it = ple_history.begin(); it != ple_history.end();) {
        it = it->first == seq_id ? std::next(it) : ple_history.erase(it);
    }
}

void llama_memory_hybrid_idx::ple_seq_add(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (hparams.ple_n_heads == 0 || shift == 0) {
        return;
    }
    const auto found = ple_history.find(seq_id);
    if (found == ple_history.end()) {
        return;
    }
    p0 = std::max<llama_pos>(p0, 0);
    p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;

    ple_seq_history shifted;
    shifted.reserve(found->second.size());
    for (auto entry : found->second) {
        if (p0 <= entry.pos && entry.pos < p1) {
            entry.pos += shift;
        }
        if (entry.pos >= 0) {
            shifted.push_back(entry);
        }
    }
    found->second = std::move(shifted);
}

void llama_memory_hybrid_idx::ple_seq_div(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (hparams.ple_n_heads == 0 || d == 1) {
        return;
    }
    GGML_ASSERT(d > 0);
    const auto found = ple_history.find(seq_id);
    if (found == ple_history.end()) {
        return;
    }
    p0 = std::max<llama_pos>(p0, 0);
    p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;

    for (auto & entry : found->second) {
        if (p0 <= entry.pos && entry.pos < p1) {
            entry.pos /= d;
        }
    }
}

void llama_memory_hybrid_idx::ple_state_write(
        llama_io_write_i & io, llama_seq_id seq_id) const {
    constexpr uint32_t version = 2;
    io.write(&version, sizeof(version));

    auto write_history = [&io](const ple_seq_history & history) {
        if (history.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Qwen4Exp PLE history is too large to serialize");
        }
        const uint32_t count = static_cast<uint32_t>(history.size());
        io.write(&count, sizeof(count));
        for (const auto & entry : history) {
            io.write(&entry.pos,   sizeof(entry.pos));
            io.write(&entry.token, sizeof(entry.token));
        }
    };

    if (seq_id >= 0) {
        const auto found = ple_history.find(seq_id);
        const ple_seq_history empty;
        write_history(found == ple_history.end() ? empty : found->second);
        return;
    }

    uint32_t n_seq = 0;
    for (const auto & entry : ple_history) {
        n_seq += !entry.second.empty();
    }
    io.write(&n_seq, sizeof(n_seq));
    for (const auto & entry : ple_history) {
        if (entry.second.empty()) {
            continue;
        }
        io.write(&entry.first, sizeof(entry.first));
        write_history(entry.second);
    }
}

void llama_memory_hybrid_idx::ple_state_read(
        llama_io_read_i & io, llama_seq_id seq_id) {
    const uint64_t max_entries_u64 =
            (uint64_t) get_mem_attn()->get_size() + hparams.ple_ngram_size + 1;
    const uint32_t max_entries = (uint32_t) std::min<uint64_t>(
            max_entries_u64, std::numeric_limits<uint32_t>::max());
    const uint32_t max_sequences = std::max<uint32_t>(n_seq_max, 1);

    uint32_t version = 0;
    io.read(&version, sizeof(version));
    if (version != 1 && version != 2) {
        throw std::runtime_error("unsupported Qwen4Exp PLE history state version");
    }

    auto read_history = [&io, max_entries](ple_seq_history & history) {
        uint32_t count = 0;
        io.read(&count, sizeof(count));
        if (count > max_entries) {
            throw std::runtime_error("invalid Qwen4Exp PLE history size");
        }
        history.clear();
        history.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            llama_pos pos;
            llama_token token;
            io.read(&pos,   sizeof(pos));
            io.read(&token, sizeof(token));
            if (pos < 0 || token < 0) {
                throw std::runtime_error("invalid Qwen4Exp PLE history entry");
            }
            history.push_back({pos, token});
        }
    };

    if (seq_id >= 0) {
        auto & history = ple_history[seq_id];
        read_history(history);
        if (history.empty()) {
            ple_history.erase(seq_id);
        }
        return;
    }

    uint32_t n_seq = 0;
    io.read(&n_seq, sizeof(n_seq));
    if (n_seq > max_sequences) {
        throw std::runtime_error("invalid Qwen4Exp PLE sequence count");
    }
    ple_history.clear();
    std::set<llama_seq_id> seen_sequences;
    for (uint32_t i = 0; i < n_seq; ++i) {
        llama_seq_id stored_seq;
        io.read(&stored_seq, sizeof(stored_seq));
        if (stored_seq < 0 || (uint32_t) stored_seq >= n_seq_max ||
                !seen_sequences.insert(stored_seq).second) {
            throw std::runtime_error("invalid or duplicate Qwen4Exp PLE sequence id");
        }
        auto & history = ple_history[stored_seq];
        read_history(history);
        if (history.empty()) {
            ple_history.erase(stored_seq);
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

static std::vector<std::vector<llama_seq_id>> llama_memory_hybrid_idx_stream_ids(
        const llama_kv_cache::slot_info_vec_t & sinfos, size_t n_ubatches) {
    GGML_ASSERT(sinfos.empty() || sinfos.size() == n_ubatches);
    std::vector<std::vector<llama_seq_id>> result;
    result.reserve(std::max(sinfos.size(), n_ubatches));
    for (const auto & sinfo : sinfos) {
        result.push_back(sinfo.strm);
    }
    // PLE also uses this vector as the marker for actual batch contexts. Keep
    // one entry per ubatch even if an older/dense-only model has no index cache.
    result.resize(n_ubatches);
    return result;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    ctx_idx(mem->get_mem_idx() ? mem->get_mem_idx()->init_full() : nullptr),
    ctx_ple(mem->get_mem_ple() ? mem->get_mem_ple()->init_full() : nullptr) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    ctx_idx(mem->get_mem_idx() ? mem->get_mem_idx()->init_update(lctx, optimize) : nullptr),
    ctx_ple(mem->get_mem_ple() ? mem->get_mem_ple()->init_update(lctx, optimize) : nullptr) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    stream_ids_ubatch(llama_memory_hybrid_idx_stream_ids(sinfos_idx, ubatches.size())),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)),
    ctx_ple(mem->get_mem_ple() == nullptr ? nullptr :
        new llama_memory_recurrent_context(mem->get_mem_ple(), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    const bool idx_more = ctx_idx ? ctx_idx->next() : false;
    const bool ple_more = ctx_ple ? ctx_ple->next() : false;
    const bool base_more = llama_memory_hybrid_context::next();
    if (ctx_idx) {
        GGML_ASSERT(idx_more == base_more);
    }
    if (ctx_ple) {
        GGML_ASSERT(ple_more == base_more);
    }
    ++i_cur;
    return base_more;
}

bool llama_memory_hybrid_idx_context::apply() {
    bool result = llama_memory_hybrid_context::apply();
    if (ctx_idx) {
        result = result & ctx_idx->apply();
    }
    if (ctx_ple) {
        result = result & ctx_ple->apply();
    }
    if (result && !stream_ids_ubatch.empty()) {
        GGML_ASSERT(mem != nullptr);
        mem->ple_apply_ubatch(get_ubatch(), ple_prefix);
    }
    return result;
}

llama_memory_status llama_memory_hybrid_idx_context::get_status() const {
    const auto base_idx = llama_memory_status_combine(
            llama_memory_hybrid_context::get_status(),
            ctx_idx ? ctx_idx->get_status() : LLAMA_MEMORY_STATUS_NO_UPDATE);
    return llama_memory_status_combine(
            base_idx, ctx_ple ? ctx_ple->get_status() : LLAMA_MEMORY_STATUS_NO_UPDATE);
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_idx_context::get_ple() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_ple.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    if (!stream_ids_ubatch.empty()) {
        GGML_ASSERT(i_cur < stream_ids_ubatch.size());
        return stream_ids_ubatch[i_cur].size();
    }
    return mem && mem->get_mem_idx() ? mem->get_mem_idx()->get_n_stream() : 0;
}

bool llama_memory_hybrid_idx_context::qsa_compatible(
        const llama_ubatch * ubatch, uint32_t ratio) const {
    if (ubatch == nullptr || ratio == 0 || mem == nullptr || get_idx() == nullptr ||
            i_cur >= stream_ids_ubatch.size()) {
        return false;
    }

    const auto & stream_ids = stream_ids_ubatch[i_cur];
    const uint32_t n_stream = stream_ids.size();
    const uint32_t n_kv = get_idx()->get_n_kv();
    if (n_stream == 0 || n_kv == 0 || ubatch->n_tokens <= 0 ||
            ubatch->n_tokens % n_stream != 0) {
        return false;
    }
    const int64_t n_tps = ubatch->n_tokens / n_stream;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const int64_t first = (int64_t) s * n_tps;
        if (ubatch->n_seq_id[first] != 1) {
            return false;
        }
        const llama_seq_id seq_id = ubatch->seq_id[first][0];
        if (seq_id < 0 || stream_ids[s] < 0 ||
                (n_stream > 1 && seq_id != stream_ids[s])) {
            return false;
        }
        for (int64_t i = first; i < first + n_tps; ++i) {
            if (ubatch->n_seq_id[i] != 1 || ubatch->seq_id[i][0] != seq_id) {
                return false;
            }
        }

        const auto & cells = mem->get_mem_idx()->get_cells(seq_id);
        if (n_kv > cells.size()) {
            return false;
        }
        std::vector<bool> seen(n_kv, false);
        uint32_t used = 0;
        llama_pos max_pos = -1;
        for (uint32_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }
            if (!cells.seq_has(j, seq_id)) {
                return false;
            }
            const llama_pos pos = cells.pos_get(j);
            if (pos < 0 || (uint64_t) pos >= n_kv || seen[(size_t) pos]) {
                return false;
            }
            seen[(size_t) pos] = true;
            used++;
            max_pos = std::max(max_pos, pos);
        }
        if (used == 0 || (uint64_t) max_pos + 1 != used) {
            return false;
        }
    }

    return true;
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio) const {
    GGML_ASSERT(qsa_compatible(ubatch, ratio));
    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];
    const int64_t n_blocks = blk_pos->ne[0] / (4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;
    const int64_t n_tps    = n_tokens / n_ns;

    GGML_ASSERT((uint32_t) n_ns == get_n_stream());

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    for (int64_t sec = 0; sec < 4; ++sec) {
        for (int64_t s = 0; s < n_ns; ++s) {
            for (int64_t b = 0; b < n_blocks; ++b) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = (int32_t) (b*r);
            }
        }
    }

    std::vector<int32_t> blk_of(n_kv);
    std::vector<int32_t> filled(n_blocks);

    for (int64_t s = 0; s < n_ns; ++s) {
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(blk_of.begin(), blk_of.end(), -1);
        std::fill(filled.begin(), filled.end(), 0);
        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }
            const llama_pos p = cells.pos_get(j);
            const int64_t b = p/r;
            GGML_ASSERT(b >= 0 && b < n_blocks);
            blk_of[j] = (int32_t) b;
            cur_blk_cells[b*r + (p%r)] = (int32_t) j;
            filled[b]++;
        }

        for (int64_t j = 0; j < n_kv; ++j) {
            if (blk_of[j] >= 0 && filled[blk_of[j]] < r) {
                blk_of[j] = -1;
            }
            cur_cell_blk[j] = blk_of[j] < 0 ? 0 : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const llama_pos q = ubatch->pos[i];
            const llama_pos tail_start = (q + 1)/r*r;
            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float value = -INFINITY;
                if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                    value = cells.pos_get(j) >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                }
                cur_bias[j] = value;
            }
        }
    }
}

llama_token llama_memory_hybrid_idx_context::ple_prefix_token(
        llama_seq_id seq_id, size_t lookback, llama_token fallback) const {
    if (lookback == 0) {
        return fallback;
    }
    const auto found = ple_prefix.find(seq_id);
    if (found == ple_prefix.end() || lookback > found->second.size()) {
        return fallback;
    }
    return found->second[found->second.size() - lookback];
}
