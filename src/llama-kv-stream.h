#pragma once

#include "ggml-backend.h"
#include "ggml-copy-engine.h"
#include "ggml-cpp.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

// KV streaming [EXPERIMENTAL]
//
// The attention layers of one sequence keep their K/V rows in pages of page_size cells. A device pool (the KV cache
// tensors, shrunk to n_slots pages) holds the resident pages, and pinned host memory holds the pages that have left the
// pool. Pages are assigned to pool slots on demand: the pages a ubatch writes are always resident, free slots are
// filled, and the oldest page is evicted when a written page needs a slot. A page is written back to the host when it
// is evicted, and read back if it gets a slot again. These page moves are asynchronous copies on the compute backend's
// stream, so they are ordered with the graphs that read and write the pool.
//
// Attended pages that are not resident stream in every ubatch: a copy engine (ggml-copy-engine.h) copies them from the
// host store into one of two device slots per layer (alternating by layer), beside the compute stream. A copier thread
// queues each layer's copies once the slot's previous reader has released it, and a publisher thread sets the layer's
// ready flag when they have landed; the attention waits for the flag and releases the slot (ggml_flash_attn_ext_set_sync).
//
// Decode ubatches (up to 4 tokens, the paged decode kernel) read resident pages from the pool and streamed pages from the
// slot through a page table (ggml_flash_attn_ext_set_pages); streamed pages are packed at the start of the slot. So do
// larger ubatches if the backend's attention takes paged K/V for them (paged_prefill()); otherwise they attend the slot
// as a contiguous copy of the whole layer for the stock kernels: streamed pages are copied to their logical place by the
// copy engine, resident pages by the graph.
//
// Between decode ubatches the host samples and drafts while the copy engine would idle, so after a decode ubatch the
// copier prefetches the first two layers of the next one: the same streamed pages into the two slots, once their last
// readers released them. They are published only when the next ubatch streams exactly the same copies (pages move
// about once per page of tokens); otherwise they are copied again over the prefetched data.
//
// Debug: LLAMA_KV_STREAM_DEBUG=rotate (when the pool holds the whole context) moves every resident page through host
// memory into a rotated slot before every ubatch; LLAMA_KV_STREAM_DEBUG=trace logs the copy queue;
// LLAMA_KV_STREAM_DEBUG=nocopy runs the copy queue without copying (wrong results; for the cost of the copies);
// LLAMA_KV_STREAM_DEBUG=noprefetch disables the prefetch.
//
// Stalls: every wait of the copy queue is bounded (LLAMA_KV_STREAM_TIMEOUT seconds, default 30) and aborts with the queue
// state; the attention's waits trap after 60 s.
// what one layer attends in a graph (llama_kv_cache::get_kv_stream_attn)
struct llama_kv_stream_attn {
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;

    int32_t table = -1; // row of the page table input, -1 when attending a gather

    // prefill: the page table as runs (ggml_flash_attn_ext_set_page_runs)
    const ggml_fa_page_runs * runs = nullptr;

    // when pages stream: views of the layer's device slot for the page table, and the sync flags
    ggml_tensor *    ring_k   = nullptr;
    ggml_tensor *    ring_v   = nullptr;
    const uint32_t * ready    = nullptr;
    uint32_t *       consumed = nullptr;
};

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
    ~llama_kv_stream();

    // pool pages that fit in `bytes` when each cell takes bytes_per_cell (K and V of every layer)
    static uint32_t slots_for(size_t bytes, size_t bytes_per_cell);

    ggml_backend_dev_t device() const { return dev; }

    // the backend whose stream runs the graphs: page moves are queued on it
    // n_head: query heads, to check whether the backend's attention takes paged K/V for prefill ubatches too
    void set_backend(ggml_backend_t backend, int64_t n_head);


    // prefill ubatches read the pages through the page table like decode, instead of attending a gather
    bool paged_prefill() const { return prefill_paged; }

    // the attended pages of the current ubatch as runs of consecutive pool or ring pages
    const ggml_fa_page_runs * page_runs() const { return &runs_desc; }

    // before a ubatch writes `cells`, with [0, n_kv) attended: place the pages, and queue the copies of the streamed ones
    void prepare(const std::vector<uint32_t> & cells, uint32_t n_kv);

    // pool row of a cell whose page is resident
    int64_t row(uint32_t cell) const;

    // the current ubatch attends pages that stream (its attention waits for the copy engine)
    bool synced() const { return n_streamed > 0; }

    // page table input: I32 [n_kv/page_size, n_layer]; slots of resident pages, -1 - ring index of streamed ones
    void set_input_pages(ggml_tensor * dst) const;

    // step tag input: I32 [1]
    void set_input_tag(ggml_tensor * dst) const;

    // runs of the resident attended pages, for gathers built into a graph
    std::vector<run> runs(uint32_t n_kv) const;

    // device slot of layer ikv: [n_embd_gqa, kv_size]
    ggml_tensor * slot_k(int32_t ikv) const { return slots[ikv % 2].k; }
    ggml_tensor * slot_v(int32_t ikv) const { return slots[ikv % 2].v; }

    // sync flags of layer ikv
    const uint32_t * ready   (int32_t ikv) const { return flags + ikv; }
    uint32_t       * consumed(int32_t ikv) const { return flags + layers.size() + ikv; }

    // changes whenever a page moves, so graphs built around the layout (gathers) are not reused across moves
    uint64_t layout() const { return layout_id; }

    // forget every page (the cache was cleared)
    void clear();

    // pool pages
    uint32_t n_pool() const { return n_slots; }

    // pages each device slot holds
    uint32_t n_ring() const { return slots[0].k ? (uint32_t) (slots[0].k->ne[1]/page_size) : 0; }

    // the slots hold the streamed pages packed (paged prefill and decode), so they can be smaller than the context
    bool ring_packed() const { return prefill_paged && !debug_rotate; }

    // bytes of one page in both slots
    size_t ring_page_size() const { return 2*page_size*(row_size(0, false) + row_size(0, true)); }

    // resizing the pool (the cache reallocates the pool tensors in between): release_pool writes every resident page
    // back to the host store if it was written since it was loaded, frees the slots, and leaves the pool empty and the
    // backend idle; set_pool then takes the new pool size and allocates slots of n_ring pages (0: every page that is
    // not resident), and the pages come back on demand
    void release_pool();
    void set_pool(uint32_t n_slots, uint32_t n_ring = 0);

    // rows [c0, c1) of layer ikv, wherever they live; for state save/restore, with the backend idle
    void get_rows(int32_t ikv, bool is_v, uint32_t c0, uint32_t c1, void * dst);
    void set_rows(int32_t ikv, bool is_v, uint32_t c0, uint32_t c1, const void * src);

private:
    void   alloc_slots(int64_t rows, bool no_alloc);
    size_t row_size(int32_t ikv, bool is_v) const;
    char * host(int32_t ikv, bool is_v, uint32_t page) const;

    void assign(uint32_t page, uint32_t slot);
    void evict(uint32_t page);
    void zero_host(uint32_t page);

    // copies of one ubatch: runs of host pages to slot pages, for every layer
    struct job {
        uint32_t tag;
        std::vector<run> copies; // page: host page, slot: page index in the device slot
        ggml_backend_event_t wb = nullptr; // write-backs the copies read, recorded on the compute stream
        bool small = false;                // a decode-sized ubatch: prefetch the next one
    };

    // copier: the layers copied ahead for the next job
    struct prefetch {
        bool             valid = false;
        uint32_t         tag   = 0;
        uint64_t         epoch = 0;
        uint64_t         g_end = 0; // copy events up to the prefetched layers
        std::vector<run> copies;
    };
    static constexpr int32_t n_prefetch = 2; // one layer per device slot

    void copier_main();
    void publisher_main();
    void drain();

    // spin until cond() holds; false if stopping. A wait longer than the timeout is a bug: log the queue state and abort
    template <typename F>
    bool wait_for(F cond, const char * what, int32_t ikv, uint32_t wait_tag);
    void log_state() const;

    const std::vector<layer> layers;
    uint32_t       n_slots;
    const uint32_t kv_size;
    const uint32_t n_pages;

    ggml_backend_dev_t dev;
    ggml_backend_t     backend = nullptr;

    // host store: per layer, K pages then V pages, in page order
    ggml_backend_buffer_ptr host_buf;
    std::vector<size_t>     host_offs; // [n_layer] start of the layer's K pages

    ggml_context_ptr        slots_ctx;
    ggml_backend_buffer_ptr slots_buf;
    layer                   slots[2] = {};

    std::vector<int32_t> page_slot; // [n_pages] pool slot, -1 if not resident
    std::vector<int32_t> slot_page; // [n_slots] page, -1 if free
    std::vector<int32_t> page_ring; // [n_pages] ring index of a streamed page in the current ubatch, -1 otherwise
    std::vector<uint8_t> dirty;     // resident page written since it was loaded
    std::vector<uint8_t> saved;     // the host store holds the page

    std::vector<ggml_fa_page_run> run_list;
    ggml_fa_page_runs             runs_desc = {};

    uint64_t layout_id  = 0;
    uint32_t n_streamed = 0; // pages streamed in the current ubatch
    uint32_t tag        = 0; // tag of the current ubatch, counts the ubatches that streamed

    // write-backs since the last job, for the next one to wait for
    ggml_backend_event_t wb_pending = nullptr;

    // copy engine
    const ggml_backend_copy_engine_i * ce_iface = nullptr;
    ggml_backend_copy_engine_t         ce       = nullptr;
    uint32_t *                         flags    = nullptr; // ready[n_layer], consumed[n_layer]

    static constexpr int n_events = 64;

    std::thread             copier;
    std::thread             publisher;
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<job>         jobs;            // queued by prepare, taken by the copier
    struct span { int32_t ikv; uint32_t tag; bool prefetch = false; }; // a prefetch is published by the copier
    span                    spans[n_events]; // by event: the layer copy it marks
    std::atomic<uint64_t>   n_enqueued{0};   // layer copies queued on the copy engine
    std::atomic<uint64_t>   n_published{0};  // layer copies published
    uint64_t                n_jobs = 0;      // jobs queued (main thread)
    std::atomic<uint64_t>   n_jobs_done{0};  // jobs fully queued on the copy engine
    std::atomic<bool>       stop{false};
    std::atomic<uint64_t>   epoch{0};        // bumped by drain: the host store or the slots may change, prefetches are stale

    bool     prefill_paged = false;

    bool     debug_rotate = false;
    bool     debug_trace  = false;
    bool     debug_nocopy = false;
    bool     no_prefetch  = false;

    int64_t  timeout_us = 30*1000*1000; // LLAMA_KV_STREAM_TIMEOUT, seconds
    uint32_t rotation     = 0;
};
