#include "llama-moe-stream.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 64;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

// saturating increment - route-hotness counters accumulate over a whole run and must not wrap
static uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

// page-aligned allocation, required both for O_DIRECT reads and for Metal private-buffer uploads
static void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); staging must have
// room for len (+ 2*MOE_STREAM_DIRECT_ALIGN when direct). returns a pointer to the len bytes
// within staging, or nullptr on failure
static const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // no positional read primitive; serialize the seek+read pairs
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

// true iff all of the given exps tensors are this layer's cache tensors - guards against a second,
// non-streamed expert group on the same layer index (e.g. grovemoe chexps)
bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct) : n_slots(n_slots) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    use_direct_io = direct;
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    ggml_context * ctx = nullptr;
    for (auto & [cur_buft, cur_ctx] : ctxs) {
        if (cur_buft == buft) {
            ctx = cur_ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size()*4 + 1),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        ctxs.emplace_back(buft, ctx);
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_claimed .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->route_hotness.resize(n_expert, 0);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);
    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    for (auto & [buft, ctx_ptr] : ctxs) {
        ggml_context * ctx = ctx_ptr.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buf);

        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

    open_all(use_direct_io);

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        bool ok = !files.empty() && files.front()->has_direct_io();
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: O_DIRECT not usable, falling back to buffered streaming reads\n", __func__);
            use_direct_io = false;
            open_all(false);
        }
    }

    if (use_direct_io) {
        LLAMA_LOG_INFO("%s: MoE expert streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }

    // one token drives ~one remap per streamed layer, so decaying every 64 tokens is
    //   64 * n_streamed_layers remap calls (computed once here, off the hot path)
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    hot_decay_interval = MOE_STREAM_HOT_DECAY_TOKENS * n_streamed;
}

// spawn the I/O thread pool on first use (from the remap callback, under mtx)
void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

// I/O worker: pops a reserved load, reads its expert slab(s) from the GGUF file into the cache
// slot, and marks the slot RESIDENT (or flags load_failed); stale/duplicate items are skipped
void llama_moe_stream::worker_loop() {
    // page-aligned staging (Metal private buffers require page-aligned source + page-multiple
    // length; O_DIRECT needs the extra head/tail slack for its aligned reads)
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(max_nb_expert + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&]{ return shutting_down || !q_demand.empty(); });
        if (shutting_down) {
            break;
        }

        llama_moe_stream_work w = q_demand.front();
        q_demand.pop_front();

        auto & sl = *w.sl;
        if (w.gen != sl.slot_gen[w.slot] ||
            sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
            sl.slot_expert[w.slot] != w.expert ||
            sl.slot_claimed[w.slot]) {
            continue; // stale or duplicate item
        }
        sl.slot_claimed[w.slot] = 1;

        lk.unlock();

        bool ok = true;
        for (const auto & wt : sl.weights) {
            const uint8_t * data = llama_moe_stream_pread(*files[wt.file_idx], staging, wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, use_direct_io);
            if (data == nullptr) {
                ok = false;
                break;
            }
            ggml_backend_tensor_set(wt.cache, data, (size_t) w.slot*wt.nb_expert, wt.nb_expert);
        }

        lk.lock();

        sl.slot_claimed[w.slot] = 0;
        if (!ok) {
            load_failed = true;
        } else {
            sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        }
        cv_done.notify_all();
    }
    lk.unlock();

    moe_aligned_free(staging);
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when no candidate exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the slot's prior occupant, bump slot_gen (so any
// in-flight load for the old occupant is recognized as stale), and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot.erase(sl.slot_expert[slot]);
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
}

size_t llama_moe_stream::size_bufs() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    LLAMA_LOG_INFO("%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_INFO("%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
}

// custom-op callback (single-threaded on ith 0): given the router's expert ids, ensure every touched
// expert is resident - reserving cache slots and demand-loading misses, stalling until they commit -
// then rewrite each id to its cache slot. this only relabels ids, so the same experts are computed
// in the same order; the result matches a non-streamed run (bit-exact when both paths use the same
// kernels, as on CUDA; a CPU build that repacks the non-streamed weights can differ in the last bits).
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly-hot expert ages out
    for (const int32_t e : sl->uniq) {
        sat_inc(sl->route_hotness[e]);
    }
    if (mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & sl2 : mgr->layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }

    // classify the touched experts; reserve and enqueue demand loads in deterministic order
    std::fill(sl->keep.begin(), sl->keep.end(), 0);
    sl->demand_slots.clear();

    bool waited = false;
    for (const int32_t e : sl->uniq) {
        const auto it = sl->expert_slot.find(e);
        if (it != sl->expert_slot.end()) {
            const int32_t s = it->second;
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                mgr->q_demand.push_back({ sl, e, s, sl->slot_gen[s] });
                mgr->cv_work.notify_one();
                waited = true;
            }
            mgr->stats.n_hit++;
            sl->keep[s] = 1;
            sl->demand_slots.push_back(s);
        } else {
            int32_t v;
            while ((v = mgr->pick_victim_locked(*sl, sl->keep.data())) < 0) {
                // every allowed slot is loading; wait for a commit and retry
                mgr->cv_done.wait(lk);
                if (mgr->load_failed) {
                    GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                }
            }
            if (!sl->seen[e]) {
                mgr->stats.n_miss_cold++;
            }
            mgr->reserve_slot_locked(*sl, e, v);
            mgr->q_demand.push_back({ sl, e, v, sl->slot_gen[v] });
            mgr->cv_work.notify_one();
            mgr->stats.n_miss++;
            waited = true;
            sl->keep[v] = 1;
            sl->demand_slots.push_back(v);
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        mgr->cv_done.wait(lk, [&]{
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t s : sl->demand_slots) {
                if (sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->stats.t_stall_us += ggml_time_us() - t0;
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t s = sl->expert_slot.at(ids[i]);
        sl->slot_last_use[s] = ++sl->use_counter;
        out[i] = s;
    }
}
