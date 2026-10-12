#include "llama-kv-stream.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

llama_kv_stream::llama_kv_stream(const std::vector<layer> & layers, uint32_t n_slots, uint32_t kv_size, ggml_backend_dev_t dev, bool no_alloc) :
    layers(layers), n_slots(n_slots), kv_size(kv_size), n_pages((kv_size + page_size - 1)/page_size), dev(dev) {
    GGML_ASSERT(!layers.empty());

    size_t size = 0;
    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        host_offs.push_back(size);
        size += (size_t) n_pages*page_size*(row_size(ikv, false) + row_size(ikv, true));
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    if (!buft) {
        buft = ggml_backend_cpu_buffer_type();
    }
    host_buf.reset(ggml_backend_buft_alloc_buffer(buft, no_alloc ? 0 : size));
    if (!host_buf) {
        throw std::runtime_error("failed to allocate the KV stream host store");
    }

    // two scratch slots for gathers, sized for the widest layer
    {
        int64_t ne0_k = 0;
        int64_t ne0_v = 0;
        for (const auto & l : layers) {
            ne0_k = std::max(ne0_k, l.k->ne[0]);
            ne0_v = std::max(ne0_v, l.v->ne[0]);
        }

        ggml_init_params params = {
            /*.mem_size   =*/ 4*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        scratch_ctx.reset(ggml_init(params));

        for (int s = 0; s < 2; ++s) {
            scratch[s].k = ggml_new_tensor_2d(scratch_ctx.get(), layers[0].k->type, ne0_k, (int64_t) n_pages*page_size);
            scratch[s].v = ggml_new_tensor_2d(scratch_ctx.get(), layers[0].v->type, ne0_v, (int64_t) n_pages*page_size);
            ggml_format_name(scratch[s].k, "kv_stream_scratch_k%d", s);
            ggml_format_name(scratch[s].v, "kv_stream_scratch_v%d", s);
        }

        ggml_backend_buffer_type_t buft_dev = ggml_backend_buffer_get_type(layers[0].k->buffer);
        if (no_alloc) {
            // dummy buffer, so that the scheduler does not try to allocate the slots
            scratch_buf.reset(ggml_backend_buft_alloc_buffer(buft_dev, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(scratch_ctx.get()); t; t = ggml_get_next_tensor(scratch_ctx.get(), t)) {
                t->buffer = scratch_buf.get();
            }
        } else {
            scratch_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(scratch_ctx.get(), buft_dev));
        }
        if (!scratch_buf) {
            throw std::runtime_error("failed to allocate the KV stream scratch slots");
        }
        ggml_backend_buffer_clear(scratch_buf.get(), 0);
    }

    page_slot.assign(n_pages, -1);
    slot_page.assign(n_slots, -1);
    dirty.assign(n_pages, 0);
    saved.assign(n_pages, 0);

    const char * debug = getenv("LLAMA_KV_STREAM_DEBUG");
    debug_rotate = debug && strcmp(debug, "rotate") == 0;

    LLAMA_LOG_INFO("%s: %u of %u pages resident, %10s host store = %8.2f MiB, scratch = %8.2f MiB%s\n", __func__,
            n_slots, n_pages, ggml_backend_buffer_name(host_buf.get()), size/1024.0/1024.0,
            ggml_backend_buffer_get_size(scratch_buf.get())/1024.0/1024.0, debug_rotate ? " (debug: rotate)" : "");
}

uint32_t llama_kv_stream::slots_for(size_t bytes, size_t bytes_per_cell) {
    return (uint32_t) std::min<size_t>(bytes/(bytes_per_cell*page_size), UINT32_MAX);
}

void llama_kv_stream::set_backend(ggml_backend_t backend) {
    this->backend = backend;
}

size_t llama_kv_stream::row_size(int32_t ikv, bool is_v) const {
    const ggml_tensor * t = is_v ? layers[ikv].v : layers[ikv].k;
    return ggml_row_size(t->type, t->ne[0]);
}

char * llama_kv_stream::host(int32_t ikv, bool is_v, uint32_t page) const {
    char * base = (char *) ggml_backend_buffer_get_base(host_buf.get()) + host_offs[ikv];
    if (is_v) {
        base += (size_t) n_pages*page_size*row_size(ikv, false);
    }
    return base + (size_t) page*page_size*row_size(ikv, is_v);
}

void llama_kv_stream::assign(uint32_t page, uint32_t slot) {
    GGML_ASSERT(page_slot[page] < 0 && slot_page[slot] < 0);

    page_slot[page] = slot;
    slot_page[slot] = page;
    dirty[page] = 0;
    layout_id++;

    if (saved[page]) {
        GGML_ASSERT(backend && "llama_kv_stream: no backend for page moves");
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (bool is_v : { false, true }) {
                const size_t bytes = page_size*row_size(ikv, is_v);
                ggml_backend_tensor_set_async(backend, is_v ? layers[ikv].v : layers[ikv].k, host(ikv, is_v, page), slot*bytes, bytes);
            }
        }
    }
}

void llama_kv_stream::evict(uint32_t page) {
    const int32_t slot = page_slot[page];
    GGML_ASSERT(slot >= 0);

    if (dirty[page]) {
        GGML_ASSERT(backend && "llama_kv_stream: no backend for page moves");
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (bool is_v : { false, true }) {
                const size_t bytes = page_size*row_size(ikv, is_v);
                ggml_backend_tensor_get_async(backend, is_v ? layers[ikv].v : layers[ikv].k, host(ikv, is_v, page), slot*bytes, bytes);
            }
        }
        saved[page] = 1;
        dirty[page] = 0;
    }

    page_slot[page] = -1;
    slot_page[slot] = -1;
    layout_id++;
}

void llama_kv_stream::prepare(const std::vector<uint32_t> & cells, uint32_t n_kv) {
    GGML_ASSERT(n_kv % page_size == 0);

    std::vector<uint8_t> need(n_pages, 0);
    for (uint32_t p = 0; p < std::min(n_kv/page_size, n_pages); ++p) {
        need[p] = 1;
    }
    for (uint32_t c : cells) {
        need[c/page_size] = 1;
    }

    if (debug_rotate) {
        // every page moves: write back the resident ones, then bring the needed ones into rotated slots
        rotation++;
        for (uint32_t p = 0; p < n_pages; ++p) {
            if (page_slot[p] >= 0) {
                evict(p);
            }
        }
        for (uint32_t p = 0; p < n_pages; ++p) {
            if (need[p]) {
                assign(p, (p + rotation) % n_slots);
            }
        }
    } else {
        uint32_t free_slot = 0;
        for (uint32_t p = 0; p < n_pages; ++p) {
            if (!need[p] || page_slot[p] >= 0) {
                continue;
            }
            while (free_slot < n_slots && slot_page[free_slot] >= 0) {
                free_slot++;
            }
            uint32_t slot = free_slot;
            if (slot == n_slots) {
                // evict the oldest resident page that is not needed
                uint32_t victim = 0;
                while (victim < n_pages && (page_slot[victim] < 0 || need[victim])) {
                    victim++;
                }
                if (victim == n_pages) {
                    GGML_ABORT("llama_kv_stream: the attended pages do not fit in the pool");
                }
                slot = page_slot[victim];
                evict(victim);
            }
            assign(p, slot);
        }
    }

    for (uint32_t c : cells) {
        dirty[c/page_size] = 1;
    }
}

int64_t llama_kv_stream::row(uint32_t cell) const {
    const int32_t slot = page_slot[cell/page_size];
    GGML_ASSERT(slot >= 0);
    return (int64_t) slot*page_size + cell % page_size;
}

void llama_kv_stream::set_input_pages(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && dst->ne[1] == (int64_t) layers.size());

    int32_t * data = (int32_t *) dst->data;
    for (int64_t p = 0; p < dst->ne[0]; ++p) {
        GGML_ASSERT(page_slot[p] >= 0);
    }
    for (int64_t l = 0; l < dst->ne[1]; ++l) {
        std::copy(page_slot.begin(), page_slot.begin() + dst->ne[0], data + l*dst->ne[0]);
    }
}

std::vector<llama_kv_stream::run> llama_kv_stream::runs(uint32_t n_kv) const {
    std::vector<run> res;
    for (uint32_t p = 0; p < std::min(n_kv/page_size, n_pages); ++p) {
        const int32_t slot = page_slot[p];
        if (slot < 0) {
            continue; // only when reserving a graph: nothing is resident yet
        }
        if (!res.empty() && res.back().page + res.back().n == p && res.back().slot + res.back().n == (uint32_t) slot) {
            res.back().n++;
        } else {
            res.push_back({ p, (uint32_t) slot, 1 });
        }
    }
    return res;
}

void llama_kv_stream::clear() {
    page_slot.assign(n_pages, -1);
    slot_page.assign(n_slots, -1);
    dirty.assign(n_pages, 0);
    saved.assign(n_pages, 0);
    layout_id++;
}

void llama_kv_stream::get_rows(int32_t ikv, bool is_v, uint32_t c0, uint32_t c1, void * dst) const {
    if (backend) {
        ggml_backend_synchronize(backend);
    }

    const size_t row = row_size(ikv, is_v);
    ggml_tensor * pool = is_v ? layers[ikv].v : layers[ikv].k;

    for (uint32_t c = c0; c < c1; ) {
        const uint32_t page = c/page_size;
        const uint32_t n    = std::min(c1, (page + 1)*page_size) - c;
        char * out = (char *) dst + (size_t) (c - c0)*row;

        if (page_slot[page] >= 0) {
            ggml_backend_tensor_get(pool, out, (size_t) (page_slot[page]*page_size + c % page_size)*row, n*row);
        } else if (saved[page]) {
            memcpy(out, host(ikv, is_v, page) + (size_t) (c % page_size)*row, n*row);
        } else {
            memset(out, 0, n*row);
        }
        c += n;
    }
}

void llama_kv_stream::set_rows(int32_t ikv, bool is_v, uint32_t c0, uint32_t c1, const void * src) {
    if (backend) {
        ggml_backend_synchronize(backend);
    }

    const size_t row = row_size(ikv, is_v);
    ggml_tensor * pool = is_v ? layers[ikv].v : layers[ikv].k;

    for (uint32_t c = c0; c < c1; ) {
        const uint32_t page = c/page_size;
        const uint32_t n    = std::min(c1, (page + 1)*page_size) - c;
        const char * in = (const char *) src + (size_t) (c - c0)*row;

        if (page_slot[page] >= 0) {
            ggml_backend_tensor_set(pool, in, (size_t) (page_slot[page]*page_size + c % page_size)*row, n*row);
            dirty[page] = 1;
        } else {
            if (!saved[page]) {
                // the rest of the page is masked, but must not hold NaNs (0*NaN in P*V)
                for (size_t l = 0; l < layers.size(); ++l) {
                    memset(host(l, false, page), 0, page_size*row_size(l, false));
                    memset(host(l, true,  page), 0, page_size*row_size(l, true));
                }
                saved[page] = 1;
            }
            memcpy(host(ikv, is_v, page) + (size_t) (c % page_size)*row, in, n*row);
        }
        c += n;
    }
}
