#include "llama-memory-slots.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <cstring>
#include <stdexcept>

llama_memory_slots::llama_memory_slots(const llama_model & model, const std::vector<slot_desc> & slots, uint32_t n_seq_max, bool offload)
    : n_seq_max(n_seq_max), descs(slots), n_pos(n_seq_max, -1) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (slots.size() + 1),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(params));

    for (const auto & d : slots) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d.ne0, d.ne1 * n_seq_max);
        ggml_format_name(t, "slot_%s", d.name.c_str());
        tensors[d.name] = t;
    }

    // the slots live where the model output lives
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (offload) {
        if (ggml_backend_dev_t dev = model.dev_output()) {
            buft = ggml_backend_dev_buffer_type(dev);
        }
    }
    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    if (!buf) {
        throw std::runtime_error("failed to allocate the state slots");
    }
    ggml_backend_buffer_clear(buf.get(), 0);
    LLAMA_LOG_INFO("%s: %zu state slots, %.2f MiB for %u sequences\n", __func__, slots.size(),
            ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0, n_seq_max);
}

ggml_tensor * llama_memory_slots::view(ggml_context * ctx0, const std::string & name, llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);
    ggml_tensor * t = tensors.at(name);
    const int64_t ne1 = t->ne[1] / n_seq_max;
    return ggml_view_2d(ctx0, t, t->ne[0], ne1, t->nb[1], (size_t) seq_id * ne1 * t->nb[1]);
}

void llama_memory_slots::advance(llama_seq_id seq_id, llama_pos n) {
    n_pos[seq_id] = (n_pos[seq_id] < 0 ? 0 : n_pos[seq_id] + 1) + n - 1;
}

void llama_memory_slots::zero_seq(llama_seq_id seq_id) {
    for (const auto & [name, t] : tensors) {
        const int64_t ne1   = t->ne[1] / n_seq_max;
        const size_t  bytes = (size_t) ne1 * t->nb[1];
        std::vector<uint8_t> zeros(bytes, 0);
        ggml_backend_tensor_set(t, zeros.data(), (size_t) seq_id * bytes, bytes);
    }
    n_pos[seq_id] = -1;
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
    if (data) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
    std::fill(n_pos.begin(), n_pos.end(), -1);
}

bool llama_memory_slots::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // a slot is the result of the whole history: only removing all of it is possible
    if (p0 > 0 || (p1 >= 0 && p1 <= seq_pos_max(seq_id))) {
        return false;
    }
    if (seq_id < 0) {
        clear(true);
        return true;
    }
    zero_seq(seq_id);
    return true;
}

void llama_memory_slots::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    for (const auto & [name, t] : tensors) {
        const int64_t ne1   = t->ne[1] / n_seq_max;
        const size_t  bytes = (size_t) ne1 * t->nb[1];
        std::vector<uint8_t> tmp(bytes);
        ggml_backend_tensor_get(t, tmp.data(), (size_t) seq_id_src * bytes, bytes);
        ggml_backend_tensor_set(t, tmp.data(), (size_t) seq_id_dst * bytes, bytes);
    }
    n_pos[seq_id_dst] = n_pos[seq_id_src];
}

void llama_memory_slots::seq_keep(llama_seq_id seq_id) {
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if ((llama_seq_id) s != seq_id) {
            zero_seq(s);
        }
    }
}

void llama_memory_slots::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {}
void llama_memory_slots::seq_div(llama_seq_id, llama_pos, llama_pos, int) {}

llama_pos llama_memory_slots::seq_pos_min(llama_seq_id seq_id) const {
    return n_pos.at(seq_id) < 0 ? -1 : 0;
}

llama_pos llama_memory_slots::seq_pos_max(llama_seq_id seq_id) const {
    return n_pos.at(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_slots::memory_breakdown() const {
    return { { ggml_backend_buffer_get_type(buf.get()), ggml_backend_buffer_get_size(buf.get()) } };
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
