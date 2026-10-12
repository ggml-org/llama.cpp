#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <vector>

// KV streaming [EXPERIMENTAL]
//
// The attention layers of one sequence keep their K/V rows in pages of page_size cells. A device pool (the KV cache
// tensors, shrunk to n_slots pages) holds the resident pages, and pinned host memory holds the pages that have left the
// pool. Decode attention reads the pool through a page table (ggml_flash_attn_ext_set_pages); larger ubatches attend a
// contiguous gather of the pages into one of two scratch slots, using the stock kernels.
//
// Pages are assigned to pool slots on demand. A page is written back to the host when it is evicted, and read back if it
// is needed again (e.g. after the sequence was truncated into it). Page moves are asynchronous copies on the compute
// backend's stream, so they are ordered with the graphs that read and write the pool.
//
// For now every attended page must be resident (the pool must hold the whole context); streaming the pages that do not
// fit is the next step.
//
// Debug: LLAMA_KV_STREAM_DEBUG=rotate moves every resident page through host memory into a rotated slot before every
// ubatch.
class llama_kv_stream {
public:
    static constexpr uint32_t page_size = 256;

    // pool tensors of one layer: [n_embd_gqa, n_slots*page_size]
    struct layer {
        ggml_tensor * k;
        ggml_tensor * v;
    };

    // a run of logical pages held by consecutive pool slots
    struct run {
        uint32_t page;
        uint32_t slot;
        uint32_t n;
    };

    // no_alloc: size the buffers only (memory fitting), as the cache does
    llama_kv_stream(const std::vector<layer> & layers, uint32_t n_slots, uint32_t kv_size, ggml_backend_dev_t dev, bool no_alloc);

    // pool pages that fit in `bytes` when each cell takes bytes_per_cell (K and V of every layer)
    static uint32_t slots_for(size_t bytes, size_t bytes_per_cell);

    ggml_backend_dev_t device() const { return dev; }

    // the backend whose stream runs the graphs: page moves are queued on it
    void set_backend(ggml_backend_t backend);

    // before a ubatch writes `cells`: make the written and the attended pages ([0, n_kv)) resident
    void prepare(const std::vector<uint32_t> & cells, uint32_t n_kv);

    // pool row of a cell whose page is resident
    int64_t row(uint32_t cell) const;

    // page table input: I32 [n_kv/page_size, n_layer], the pool slot of each attended page
    void set_input_pages(ggml_tensor * dst) const;

    // runs of the attended pages, for gathers built into a graph
    std::vector<run> runs(uint32_t n_kv) const;

    // scratch slot of layer ikv: [n_embd_gqa, kv_size]
    ggml_tensor * scratch_k(int32_t ikv) const { return scratch[ikv % 2].k; }
    ggml_tensor * scratch_v(int32_t ikv) const { return scratch[ikv % 2].v; }

    // changes whenever a page moves, so graphs built around the layout (gathers) are not reused across moves
    uint64_t layout() const { return layout_id; }

    // forget every page (the cache was cleared)
    void clear();

    // rows [c0, c1) of layer ikv, wherever they live; for state save/restore, with the backend idle
    void get_rows(int32_t ikv, bool is_v, uint32_t c0, uint32_t c1, void * dst) const;
    void set_rows(int32_t ikv, bool is_v, uint32_t c0, uint32_t c1, const void * src);

private:
    size_t row_size(int32_t ikv, bool is_v) const;
    char * host(int32_t ikv, bool is_v, uint32_t page) const;

    void assign(uint32_t page, uint32_t slot);
    void evict(uint32_t page);

    const std::vector<layer> layers;
    const uint32_t n_slots;
    const uint32_t kv_size;
    const uint32_t n_pages;

    ggml_backend_dev_t dev;
    ggml_backend_t     backend = nullptr;

    // host store: per layer, K pages then V pages, in page order
    ggml_backend_buffer_ptr host_buf;
    std::vector<size_t>     host_offs; // [n_layer] start of the layer's K pages

    ggml_context_ptr        scratch_ctx;
    ggml_backend_buffer_ptr scratch_buf;
    layer                   scratch[2] = {};

    std::vector<int32_t> page_slot; // [n_pages] pool slot, -1 if not resident
    std::vector<int32_t> slot_page; // [n_slots] page, -1 if free
    std::vector<uint8_t> dirty;     // resident page written since it was loaded
    std::vector<uint8_t> saved;     // the host store holds the page

    uint64_t layout_id = 0;

    bool     debug_rotate = false;
    uint32_t rotation     = 0;
};
