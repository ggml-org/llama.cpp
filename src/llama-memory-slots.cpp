#include "llama-memory-slots.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <cstring>
#include <stdexcept>

llama_memory_slots::llama_memory_slots(const llama_model & model, const std::vector<slot_desc> & slots, uint32_t n_seq_max, bool offload)
    : n_seq_max(n_seq_max), descs(slots), n_pos(n_seq_max, -1), no_alloc(model.no_alloc()), seqs(n_seq_max) {
    // the slots live where the model output lives
    buft = ggml_backend_cpu_buffer_type();
    if (offload) {
        if (ggml_backend_dev_t dev = model.dev_output()) {
            buft = ggml_backend_dev_buffer_type(dev);
        }
    }
    size_t per_seq = 0;
    for (const auto & d : slots) {
        per_seq += (size_t) d.ne0 * d.ne1 * sizeof(float);
    }
    LLAMA_LOG_INFO("%s: %zu state slots, %.2f MiB per sequence (allocated by its first use), up to %u sequences\n",
            __func__, slots.size(), per_seq / 1024.0 / 1024.0, n_seq_max);
}

llama_memory_slots::seq_slots & llama_memory_slots::alloc_seq(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);
    seq_slots & s = seqs[seq_id];
    if (s.buf) {
        return s;
    }
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (descs.size() + 1),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    s.ctx.reset(ggml_init(params));
    for (const auto & d : descs) {
        ggml_tensor * t = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, d.ne0, d.ne1);
        ggml_format_name(t, "slot_%s_%d", d.name.c_str(), seq_id);
        s.tensors[d.name] = t;
    }
    s.buf = llama_memory_bufs::alloc_buf(s.ctx.get(), buft, no_alloc, "state slots");
    if (!no_alloc) {
        ggml_backend_buffer_clear(s.buf.get(), 0);
    }
    return s;
}

void llama_memory_slots::free_seq(llama_seq_id seq_id) {
    seqs[seq_id] = seq_slots();
    n_pos[seq_id] = -1;
}

ggml_tensor * llama_memory_slots::view(ggml_context * ctx0, const std::string & name, llama_seq_id seq_id) const {
    ggml_tensor * t = alloc_seq(seq_id).tensors.at(name);
    return ggml_view_2d(ctx0, t, t->ne[0], t->ne[1], t->nb[1], 0);
}

void llama_memory_slots::advance(llama_seq_id seq_id, llama_pos n) {
    n_pos[seq_id] = (n_pos[seq_id] < 0 ? 0 : n_pos[seq_id] + 1) + n - 1;
}

llama_memory_context_ptr llama_memory_slots::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);
    balloc.split_reset();
    std::vector<llama_ubatch> ubatches;
    while (true) {
        auto ubatch = balloc.split_seq(n_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(std::move(ubatch));
    }
    return std::make_unique<llama_memory_slots_context>(this, std::move(ubatches));
}

llama_memory_context_ptr llama_memory_slots::init_full() {
    return std::make_unique<llama_memory_slots_context>(this, LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_memory_slots::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_memory_slots_context>(this, LLAMA_MEMORY_STATUS_NO_UPDATE);
}

void llama_memory_slots::clear(bool data) {
    GGML_UNUSED(data); // a sequence starts from zeroed slots either way
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        free_seq((llama_seq_id) s);
    }
}

bool llama_memory_slots::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id >= 0 && (uint32_t) seq_id >= n_seq_max) {
        return false;
    }
    // a slot is the result of the whole history: only removing all of it is possible; a sequence without history
    // (slots allocated by a reservation or by a call that failed) goes whole
    auto untouched = [&](llama_seq_id s) { return n_pos[s] >= 0 && p0 > n_pos[s]; };
    auto whole     = [&](llama_seq_id s) { return n_pos[s] < 0 || (p0 <= 0 && (p1 < 0 || p1 > n_pos[s])); };
    if (seq_id < 0) {
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if (!untouched((llama_seq_id) s) && !whole((llama_seq_id) s)) {
                return false;
            }
        }
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if (!untouched((llama_seq_id) s)) {
                free_seq((llama_seq_id) s);
            }
        }
        return true;
    }
    if (untouched(seq_id)) {
        return true;
    }
    if (!whole(seq_id)) {
        return false;
    }
    free_seq(seq_id);
    return true;
}

void llama_memory_slots::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    if (seq_id_src < 0 || (uint32_t) seq_id_src >= n_seq_max || seq_id_dst < 0 || (uint32_t) seq_id_dst >= n_seq_max ||
            seq_id_src == seq_id_dst) {
        return;
    }
    if (!seqs[seq_id_src].buf) {
        free_seq(seq_id_dst);
        return;
    }
    const seq_slots & src = seqs[seq_id_src];
    seq_slots       & dst = alloc_seq(seq_id_dst);
    if (!no_alloc) { // the tensors of an estimate have no data
        for (const auto & [name, t] : src.tensors) {
            std::vector<uint8_t> tmp(ggml_nbytes(t));
            ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
            ggml_backend_tensor_set(dst.tensors.at(name), tmp.data(), 0, tmp.size());
        }
    }
    n_pos[seq_id_dst] = n_pos[seq_id_src];
}

void llama_memory_slots::seq_keep(llama_seq_id seq_id) {
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if ((llama_seq_id) s != seq_id) {
            free_seq((llama_seq_id) s);
        }
    }
}

void llama_memory_slots::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {}
void llama_memory_slots::seq_div(llama_seq_id, llama_pos, llama_pos, int) {}

llama_pos llama_memory_slots::seq_pos_min(llama_seq_id seq_id) const {
    return seq_id < 0 || (uint32_t) seq_id >= n_seq_max || n_pos[seq_id] < 0 ? -1 : 0;
}

llama_pos llama_memory_slots::seq_pos_max(llama_seq_id seq_id) const {
    return seq_id < 0 || (uint32_t) seq_id >= n_seq_max ? -1 : n_pos[seq_id];
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_slots::memory_breakdown() const {
    size_t size = 0;
    for (const auto & s : seqs) {
        if (!s.buf) {
            continue;
        }
        if (no_alloc) {
            size += ggml_backend_alloc_ctx_tensors_from_buft_size(s.ctx.get(), buft);
        } else {
            size += ggml_backend_buffer_get_size(s.buf.get());
        }
    }
    return { { buft, size } };
}

void llama_memory_slots::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    throw std::runtime_error("state slots: state save is not supported yet");
}

void llama_memory_slots::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    throw std::runtime_error("state slots: state load is not supported yet");
}

llama_memory_slots_context::llama_memory_slots_context(llama_memory_slots * mem, std::vector<llama_ubatch> ubatches)
    : mem(mem), status(LLAMA_MEMORY_STATUS_SUCCESS), ubatches(std::move(ubatches)) {}

bool llama_memory_slots_context::next() {
    return ++i_next < ubatches.size();
}

bool llama_memory_slots_context::apply() {
    return true;
}

const llama_ubatch & llama_memory_slots_context::get_ubatch() const {
    GGML_ASSERT(i_next < ubatches.size());
    return ubatches[i_next];
}
