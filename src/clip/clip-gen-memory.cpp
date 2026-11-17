#include "clip-gen-memory.h"
#include "clip-gen.h"

#include "llama-impl.h"

#include <stdexcept>

namespace {

// the gen pipelines do not run ubatches of this memory, the context only checks the status
class llama_memory_clip_gen_context : public llama_memory_context_i {
public:
    explicit llama_memory_clip_gen_context(llama_memory_status status) : status(status) {}

    bool next() override { return false; }
    bool apply() override { return true; }

    const llama_ubatch & get_ubatch() const override {
        GGML_ABORT("the gen memory has no ubatches");
    }

    llama_memory_status get_status() const override { return status; }

private:
    llama_memory_status status;
};

}

llama_memory_clip_gen::~llama_memory_clip_gen() {
    clear(true);
}

clip_gen_state * llama_memory_clip_gen::get(clip_gen_ctx * host, llama_seq_id seq_id) {
    auto it = states.find(seq_id);
    if (it != states.end()) {
        return it->second;
    }
    const auto its = seeds.find(seq_id);
    clip_gen_state * st = clip_gen_state_init(host, its == seeds.end() ? UINT32_MAX : its->second);
    if (st) {
        states[seq_id] = st;
    }
    return st;
}

void llama_memory_clip_gen::set_seed(llama_seq_id seq_id, uint32_t seed) {
    seeds[seq_id] = seed;
    seq_rm(seq_id, -1, -1);
}

llama_memory_context_ptr llama_memory_clip_gen::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(balloc);
    GGML_UNUSED(n_ubatch);
    GGML_UNUSED(embd_all);
    return std::make_unique<llama_memory_clip_gen_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_clip_gen::init_full() {
    return std::make_unique<llama_memory_clip_gen_context>(LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_memory_clip_gen::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_memory_clip_gen_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

void llama_memory_clip_gen::clear(bool data) {
    GGML_UNUSED(data);
    for (auto & it : states) {
        clip_gen_state_free(it.second);
    }
    states.clear();
}

bool llama_memory_clip_gen::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if ((p0 > 0) || (p1 >= 0 && p1 != std::numeric_limits<llama_pos>::max())) {
        return false; // only whole sequences
    }
    if (seq_id < 0) {
        clear(true);
        return true;
    }
    auto it = states.find(seq_id);
    if (it != states.end()) {
        clip_gen_state_free(it->second);
        states.erase(it);
    }
    return true;
}

void llama_memory_clip_gen::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(seq_id_src);
    GGML_UNUSED(seq_id_dst);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    LLAMA_LOG_WARN("%s: the gen state of a sequence cannot be copied\n", __func__);
}

void llama_memory_clip_gen::seq_keep(llama_seq_id seq_id) {
    for (auto it = states.begin(); it != states.end();) {
        if (it->first != seq_id) {
            clip_gen_state_free(it->second);
            it = states.erase(it);
        } else {
            ++it;
        }
    }
}

void llama_memory_clip_gen::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(shift);
}

void llama_memory_clip_gen::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(d);
}

llama_pos llama_memory_clip_gen::seq_pos_min(llama_seq_id seq_id) const {
    GGML_UNUSED(seq_id);
    return -1;
}

llama_pos llama_memory_clip_gen::seq_pos_max(llama_seq_id seq_id) const {
    GGML_UNUSED(seq_id);
    return -1;
}

void llama_memory_clip_gen::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    throw std::runtime_error("the gen state cannot be saved");
}

void llama_memory_clip_gen::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    throw std::runtime_error("the gen state cannot be loaded");
}
