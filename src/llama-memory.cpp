#include "llama-memory.h"

#include "llama-ext.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

//
// llama_memory_bufs
//

ggml_backend_buffer_ptr llama_memory_bufs::alloc_buf(ggml_context * ctx, ggml_backend_buffer_type_t buft, bool no_alloc, const char * name) {
    ggml_backend_buffer_ptr buf;
    if (no_alloc) {
        buf.reset(ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0)); // dummy buffer
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            t->buffer = buf.get(); // set dummy buffer so that the backend scheduler won't try to allocate the tensors
        }
    } else {
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)); // real buffer
    }
    if (!buf) {
        throw std::runtime_error(std::string("failed to allocate buffer for ") + name);
    }
    return buf;
}

ggml_backend_buffer_t llama_memory_bufs::emplace(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name) {
    ggml_backend_buffer_ptr buf = alloc_buf(ctx.get(), buft, no_alloc, name);

    // initialize the buffer to avoid NaNs in the padding
    ggml_backend_buffer_clear(buf.get(), 0);

    bufs.emplace_back(std::move(ctx), std::move(buf));

    return bufs.back().second.get();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_bufs::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }
    return ret;
}

// collects the buffers of the tensors that a state write visits, without keeping the data
struct llama_memory_buffers_writer : public llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem) {
    if (mem == nullptr) {
        return {};
    }

    llama_memory_buffers_writer writer;

    mem->state_write(writer);

    return std::vector<ggml_backend_buffer_t>(writer.buffers.begin(), writer.buffers.end());
}

static bool inspect_attention_cells(const llama_memory_i * memory, uint64_t & free, uint64_t & total) {
    if (!memory || dynamic_cast<const llama_memory_recurrent *>(memory)) {
        return true;
    }
    if (auto * kv = dynamic_cast<const llama_kv_cache *>(memory)) {
        if (kv->get_n_stream() != 1 || kv->get_n_seq_max() == 0) {
            return false;
        }
        const auto & cells = kv->get_cells(0);
        const uint64_t pool_total = cells.size();
        const uint64_t pool_free = pool_total - std::min<uint64_t>(cells.get_used(), pool_total);
        if (pool_total > 0 && (total == 0 || pool_free * total < free * pool_total)) {
            free = pool_free;
            total = pool_total;
        }
        return true;
    }
    if (auto * hybrid = dynamic_cast<const llama_memory_hybrid *>(memory)) {
        return inspect_attention_cells(hybrid->get_mem_attn(), free, total);
    }
    if (auto * hybrid = dynamic_cast<const llama_memory_hybrid_iswa *>(memory)) {
        return inspect_attention_cells(hybrid->get_mem_attn(), free, total);
    }
    if (auto * iswa = dynamic_cast<const llama_kv_cache_iswa *>(memory)) {
        const bool base = inspect_attention_cells(iswa->get_base(), free, total);
        const bool swa = inspect_attention_cells(iswa->get_swa(), free, total);
        return base && swa;
    }
    if (auto * dsa = dynamic_cast<const llama_kv_cache_dsa *>(memory)) {
        const bool mla = inspect_attention_cells(dsa->get_mla(), free, total);
        const bool lid = inspect_attention_cells(dsa->get_lid(), free, total);
        return mla && lid;
    }
    if (auto * dsa = dynamic_cast<const llama_kv_cache_dsa_iswa *>(memory)) {
        const bool base = inspect_attention_cells(dsa->get_dsa(), free, total);
        const bool swa = inspect_attention_cells(dsa->get_swa(), free, total);
        return base && swa;
    }
    if (auto * msa = dynamic_cast<const llama_kv_cache_msa *>(memory)) {
        const bool base = inspect_attention_cells(msa->get_base(), free, total);
        const bool idx = inspect_attention_cells(msa->get_idx(), free, total);
        return base && idx;
    }
    return false;
}

bool llama_memory_get_attention_cells(llama_memory_t mem, uint64_t & free, uint64_t & total) {
    free = 0;
    total = 0;
    return inspect_attention_cells(mem, free, total);
}
