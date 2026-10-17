// per-action decode trace, see ggml-trace.h
//
// emit path: the acting thread appends a record to its own 64 KiB chunk and release-stores the count; no lock, the
// chunk pool mutex once per chunk. the writer thread scans all chunks every 100 ms and serializes new records.
// the poller thread (Linux) samples page cache residency: expert slices at compute start (requests from compute
// threads through a lock-free ring, woken by an eventfd) and the ranges of pending POSIX_MADV_WILLNEED pieces.
//
// file: '[' then one event per line, so that a truncated file still loads. the first event is the metadata record;
// its counters that are only known at close are fixed-width (space-padded) and rewritten in place. the writer merges the per-thread
// EXPERT_THREAD records into one EXPERT span, and at detail 1 the pieces of one slice into one I/O span.
// tracks: pid 1 host (sched, predictor, poller, sampler, worker N), pid 2.. one per device (device, launch),
// pid 100 expert compute (gate / up / down), pid 101 expert I/O (tensor x source).

#include "ggml-trace.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#define GGML_TRACE_LINUX
#elif defined(__APPLE__)
#include <time.h>
#endif

#define GGML_TRACE_CHUNK_BYTES (64*1024)
#define GGML_TRACE_RING_SIZE   4096
#define GGML_TRACE_WRITER_MS   100
#define GGML_TRACE_GPU_GAP_NS  100000 // simple layout: GPU ops of a layer further apart are separate spans

enum { TRACE_OFF = 0, TRACE_IDLE = 1, TRACE_SUSPENDED = 2, TRACE_TRACED = 3 };

//
// clocks
//

int64_t ggml_trace_now_ns(void) {
#if defined(GGML_TRACE_LINUX) || defined(__APPLE__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec*1000000000LL + ts.tv_nsec;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

int64_t ggml_trace_thread_cpu_ns(void) {
#if defined(GGML_TRACE_LINUX) || defined(__APPLE__)
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) {
        return 0;
    }
    return (int64_t) ts.tv_sec*1000000000LL + ts.tv_nsec;
#else
    return 0;
#endif
}

bool ggml_trace_thread_faults(int64_t * minflt, int64_t * majflt) {
#if defined(GGML_TRACE_LINUX)
    struct rusage ru;
    if (getrusage(RUSAGE_THREAD, &ru) != 0) {
        return false;
    }
    *minflt = ru.ru_minflt;
    *majflt = ru.ru_majflt;
    return true;
#else
    *minflt = 0;
    *majflt = 0;
    return false;
#endif
}

int64_t ggml_trace_read_bytes(void) {
#if defined(GGML_TRACE_LINUX)
    // the fd refers to this thread's io file; a pread at 0 re-reads it. closed when the thread exits
    struct io_fd {
        int fd = -2;
        ~io_fd() {
            if (fd >= 0) {
                close(fd);
            }
        }
    };
    static thread_local io_fd f;
    if (f.fd == -2) {
        f.fd = open("/proc/thread-self/io", O_RDONLY | O_CLOEXEC);
    }
    const int fd = f.fd;
    if (fd < 0) {
        return -1;
    }
    char buf[512];
    const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) {
        return -1;
    }
    buf[n] = 0;
    const char * p = strstr(buf, "read_bytes:");
    return p ? strtoll(p + 11, nullptr, 10) : -1;
#else
    return -1;
#endif
}

//
// state
//

struct ggml_trace_chunk {
    std::atomic<uint32_t> count;   // records written by the owner (release), read by the writer (acquire)
    uint32_t written;              // serialized so far, writer only
    ggml_trace_event events[(GGML_TRACE_CHUNK_BYTES - 64) / sizeof(ggml_trace_event)];
};

static constexpr uint32_t GGML_TRACE_CHUNK_EVENTS = sizeof(ggml_trace_chunk::events) / sizeof(ggml_trace_event);

struct ggml_trace_io_pending {
    int64_t   cookie;
    uintptr_t addr;
    size_t    len;
    int64_t   t_not_resident; // last time the range was seen not resident (issue t1 at first)
    int32_t   layer;
    int32_t   expert;
    int32_t   piece;
    int32_t   source;
    char      name[32];
};

struct ggml_trace_residency_req {
    int32_t   layer;
    int32_t   expert;
    int32_t   step;
    char      name[32];
    uintptr_t addr;
    size_t    len;
    int64_t   t_first_touch;
};

struct ggml_trace_ring_slot {
    std::atomic<uint32_t>    seq;
    ggml_trace_residency_req req;
};

struct ggml_trace_state {
    std::string path;
    ggml_trace_params params;
    std::string meta;
    FILE *  f = nullptr;
    int64_t t_base = 0;
    uint32_t gen = 0;

    // state machine, under g_ctl_mtx
    int  n_traced   = 0;
    int  n_skipped  = 0;
    bool token_pending = false;
    ggml_trace_event token_ev;

    // chunk pool
    std::mutex pool_mtx;
    std::vector<ggml_trace_chunk *> chunks;   // all allocated
    std::vector<ggml_trace_chunk *> free_chunks;
    size_t max_chunks = 0;
    uint64_t              emitted = 0; // records that reached a chunk, counted by the writer (+ dropped = emitted)
    std::atomic<uint64_t> dropped{0};

    // pending I/O ranges
    std::mutex io_mtx;
    std::map<std::pair<uintptr_t, size_t>, ggml_trace_io_pending> io_pending; // by (addr, len)
    std::atomic<int64_t> next_cookie{1};

    // residency requests, compute threads -> poller
    ggml_trace_ring_slot ring[GGML_TRACE_RING_SIZE];
    std::atomic<uint32_t> ring_enq{0};
    uint32_t              ring_deq = 0;
    std::atomic<uint64_t> ring_drops{0};
    std::atomic<uint64_t> poll_adaptations{0};
    int efd = -1;
    std::atomic<bool> poller_sleeping{false};

    std::atomic<bool> stop_threads{false}; // poller
    std::atomic<bool> stop_writer{false};  // set after the poller has been joined
    std::thread poller;
    std::thread writer;
    std::mutex writer_mtx;
    std::condition_variable writer_cv;

    // metadata rewrite
    long   meta_offset = 0;
    size_t meta_pos_emitted = 0, meta_pos_dropped = 0, meta_pos_ring = 0, meta_pos_adapt = 0, meta_pos_written = 0, meta_pos_tokens = 0, meta_pos_late = 0;
};

static std::atomic<int> g_state{TRACE_OFF};
static std::atomic<int> g_step{-1};      // traced step while TRACED, else -1
static std::atomic<int> g_last_step{-1}; // last traced step
static ggml_trace_state * g_trace = nullptr;
static std::mutex g_ctl_mtx;             // start / stop / state machine
static int        g_refcount = 0;

// flush callbacks and backend ids live for the process: backends register before the trace starts
static std::mutex g_reg_mtx;
static std::vector<std::pair<void (*)(void *), void *>> g_flush;
static std::vector<std::string> g_backends = { "CPU" };

// emitters hold a reference while they use g_trace; finish() sets OFF and waits until all stripes are zero.
// the counter is striped per thread so that 14 compute threads do not contend on one cache line
#define GGML_TRACE_STRIPES 64
struct alignas(64) ggml_trace_stripe {
    std::atomic<int> n{0};
};
static ggml_trace_stripe   g_inflight[GGML_TRACE_STRIPES];
static std::atomic<int>    g_next_stripe{0};
static thread_local int    tls_stripe = -1;

static std::atomic<int> & ggml_trace_my_stripe(void) {
    if (tls_stripe < 0) {
        tls_stripe = g_next_stripe.fetch_add(1, std::memory_order_relaxed) % GGML_TRACE_STRIPES;
    }
    return g_inflight[tls_stripe].n;
}

static bool ggml_trace_any_inflight(void) {
    for (auto & s : g_inflight) {
        if (s.n.load(std::memory_order_seq_cst) > 0) {
            return true;
        }
    }
    return false;
}

struct ggml_trace_guard {
    ggml_trace_state * st = nullptr;
    std::atomic<int> * n  = nullptr;
    ggml_trace_guard() {
        if (g_state.load(std::memory_order_relaxed) == TRACE_OFF) {
            return;
        }
        n = &ggml_trace_my_stripe();
        n->fetch_add(1, std::memory_order_seq_cst);
        if (g_state.load(std::memory_order_seq_cst) == TRACE_OFF) {
            n->fetch_sub(1, std::memory_order_seq_cst);
            n = nullptr;
            return;
        }
        st = g_trace;
    }
    ~ggml_trace_guard() {
        if (n) {
            n->fetch_sub(1, std::memory_order_seq_cst);
        }
    }
};

bool ggml_trace_active(void) {
    return g_state.load(std::memory_order_relaxed) == TRACE_TRACED;
}

bool ggml_trace_enabled(void) {
    return g_state.load(std::memory_order_relaxed) != TRACE_OFF;
}

bool ggml_trace_io_enabled(void) {
    const int s = g_state.load(std::memory_order_relaxed);
    return s == TRACE_IDLE || s == TRACE_TRACED;
}

int ggml_trace_detail(void) {
    ggml_trace_guard g;
    return g.st ? g.st->params.detail : 0;
}

int ggml_trace_step(void) {
    return g_step.load(std::memory_order_relaxed);
}

int ggml_trace_last_step(void) {
    return g_last_step.load(std::memory_order_relaxed);
}

int ggml_trace_backend_id(const char * name) {
    std::lock_guard<std::mutex> lock(g_reg_mtx);
    for (size_t i = 0; i < g_backends.size(); ++i) {
        if (g_backends[i] == name) {
            return (int) i;
        }
    }
    g_backends.push_back(name);
    return (int) g_backends.size() - 1;
}

static std::string ggml_trace_backend_name(int id) {
    std::lock_guard<std::mutex> lock(g_reg_mtx);
    return id >= 0 && (size_t) id < g_backends.size() ? g_backends[id] : std::to_string(id);
}

void ggml_trace_register_flush(void (*fn)(void *), void * ud) {
    std::lock_guard<std::mutex> lock(g_reg_mtx);
    g_flush.emplace_back(fn, ud);
}

void ggml_trace_unregister_flush(void (*fn)(void *), void * ud) {
    std::lock_guard<std::mutex> lock(g_reg_mtx);
    for (size_t i = 0; i < g_flush.size(); ++i) {
        if (g_flush[i].first == fn && g_flush[i].second == ud) {
            g_flush.erase(g_flush.begin() + i);
            return;
        }
    }
}

// under g_reg_mtx, so that a backend cannot unregister (and free its context) while its callback runs
static void ggml_trace_run_flush(void) {
    std::lock_guard<std::mutex> lock(g_reg_mtx);
    for (auto & f : g_flush) {
        f.first(f.second);
    }
}

//
// emit
//

static thread_local ggml_trace_chunk * tls_chunk = nullptr;
static thread_local uint32_t           tls_gen   = 0;

void ggml_trace_event_init(struct ggml_trace_event * ev, enum ggml_trace_type type, int tid) {
    memset(ev, 0, sizeof(*ev));
    ev->type   = type;
    ev->tid    = tid;
    ev->step   = ggml_trace_step();
    ev->layer  = -1;
    ev->expert = -1;
    ev->rank   = -1;
}

void ggml_trace_event_set_name(struct ggml_trace_event * ev, const char * name) {
    const size_t cap = sizeof(ev->name) - 1;
    const size_t n   = name ? strlen(name) : 0;
    if (n <= cap) {
        memcpy(ev->name, name ? name : "", n);
        ev->name[n] = 0;
        return;
    }
    // cut on UTF-8 character boundaries: no continuation byte (10xxxxxx) right after the head or at the tail start
    auto cont = [](char c) { return ((unsigned char) c & 0xC0) == 0x80; };
    size_t head = cap/2;
    while (head > 0 && cont(name[head])) {
        head--;
    }
    size_t tail = cap - head - 1;
    while (tail > 0 && cont(name[n - tail])) {
        tail--;
    }
    memcpy(ev->name, name, head);
    ev->name[head] = '~';
    memcpy(ev->name + head + 1, name + n - tail, tail);
    ev->name[head + 1 + tail] = 0;
}

static void ggml_trace_emit_locked(ggml_trace_state * st, const ggml_trace_event * ev) {
    if (tls_chunk == nullptr || tls_gen != st->gen) {
        std::lock_guard<std::mutex> lock(st->pool_mtx);
        if (!st->free_chunks.empty()) {
            tls_chunk = st->free_chunks.back();
            st->free_chunks.pop_back();
        } else if (st->chunks.size() < st->max_chunks) {
            tls_chunk = new ggml_trace_chunk;
            tls_chunk->count.store(0, std::memory_order_relaxed);
            tls_chunk->written = 0;
            st->chunks.push_back(tls_chunk);
        } else {
            st->dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        tls_gen = st->gen;
    }

    const uint32_t n = tls_chunk->count.load(std::memory_order_relaxed);
    tls_chunk->events[n] = *ev;
    tls_chunk->count.store(n + 1, std::memory_order_release);
    if (n + 1 == GGML_TRACE_CHUNK_EVENTS) {
        tls_chunk = nullptr; // the writer recycles it once serialized; never touch it again
    }
}

// the layer of a node event, parsed from its name, else the previous one of the same kind on this thread (in
// emission order, which is the graph order; the writer sees the chunks of different threads in no useful order)
static int ggml_trace_parse_layer(const char * name);

static void ggml_trace_tag_layer(ggml_trace_event * ev) {
    static thread_local int32_t last_layer[GGML_TRACE_TYPE_COUNT];
    static thread_local int32_t last_step[GGML_TRACE_TYPE_COUNT];
    static thread_local bool    init = false;
    if (!init) {
        for (int i = 0; i < GGML_TRACE_TYPE_COUNT; ++i) {
            last_layer[i] = -1;
            last_step[i]  = INT32_MIN;
        }
        init = true;
    }
    const int t = ev->type;
    if (last_step[t] != ev->step) {
        last_step[t]  = ev->step; // a new step starts without a layer
        last_layer[t] = -1;
    }
    if (ev->layer < 0) {
        ev->layer = ggml_trace_parse_layer(ev->name);
    }
    if (ev->layer >= 0) {
        last_layer[t] = ev->layer;
    } else if (last_layer[t] >= 0) {
        ev->layer  = last_layer[t];
        ev->flags |= GGML_TRACE_FLAG_LAYER_INFERRED;
    }
}

void ggml_trace_emit(const struct ggml_trace_event * ev) {
    ggml_trace_guard g;
    if (!g.st) {
        return;
    }
    switch (ev->type) {
        case GGML_TRACE_OP_CPU:
        case GGML_TRACE_OP_GPU:
        case GGML_TRACE_SPLIT:
        case GGML_TRACE_COPY:
            {
                ggml_trace_event e = *ev;
                ggml_trace_tag_layer(&e);
                ggml_trace_emit_locked(g.st, &e);
            } break;
        default:
            ggml_trace_emit_locked(g.st, ev);
            break;
    }
}

void ggml_trace_flush(void) {
    if (ggml_trace_enabled()) {
        ggml_trace_run_flush();
    }
}

//
// state machine
//

static void ggml_trace_finish_locked(void);

static void ggml_trace_flush_token(ggml_trace_state * st) {
    if (st->token_pending) {
        st->token_pending = false;
        ggml_trace_emit_locked(st, &st->token_ev);
    }
}

// TRACED or SUSPENDED -> IDLE; g_ctl_mtx held
static void ggml_trace_batch_end_locked(ggml_trace_state * st) {
    if (g_state.load(std::memory_order_relaxed) == TRACE_TRACED) {
        st->token_ev.t1_ns = ggml_trace_now_ns();
        st->token_pending  = true; // emitted once the sampled token is known
    }
    g_step.store(-1, std::memory_order_relaxed);
    g_state.store(TRACE_IDLE, std::memory_order_seq_cst);
}

void ggml_trace_batch_begin(int n_tokens, int32_t token, int32_t pos) {
    std::lock_guard<std::mutex> ctl(g_ctl_mtx);
    ggml_trace_state * st = g_trace;
    if (st == nullptr || g_state.load(std::memory_order_relaxed) == TRACE_OFF) {
        return;
    }
    if (g_state.load(std::memory_order_relaxed) != TRACE_IDLE) {
        ggml_trace_batch_end_locked(st);
    }
    ggml_trace_flush_token(st);

    // the GPU timings of the previous batch are complete by now: let the backends resolve them
    ggml_trace_run_flush();

    if (st->params.n_tokens > 0 && st->n_traced >= st->params.n_tokens) {
        ggml_trace_finish_locked();
        return;
    }

    if (n_tokens > st->params.max_batch || st->n_skipped < st->params.n_skip) {
        if (n_tokens <= st->params.max_batch) {
            st->n_skipped++;
        }
        g_state.store(TRACE_SUSPENDED, std::memory_order_seq_cst);
        return;
    }

    const int step = st->n_traced++;
    ggml_trace_event_init(&st->token_ev, GGML_TRACE_TOKEN, GGML_TRACE_TID_SCHED);
    st->token_ev.step  = step;
    st->token_ev.token = token;
    st->token_ev.pos   = pos;
    st->token_ev.aux   = -1;
    st->token_ev.t0_ns = ggml_trace_now_ns();
    g_step.store(step, std::memory_order_relaxed);
    g_last_step.store(step, std::memory_order_relaxed);
    g_state.store(TRACE_TRACED, std::memory_order_seq_cst);
}

void ggml_trace_batch_end(void) {
    std::lock_guard<std::mutex> ctl(g_ctl_mtx);
    ggml_trace_state * st = g_trace;
    if (st == nullptr || g_state.load(std::memory_order_relaxed) == TRACE_OFF) {
        return;
    }
    ggml_trace_batch_end_locked(st);
}

int ggml_trace_stamp_token_out(int32_t token) {
    std::lock_guard<std::mutex> ctl(g_ctl_mtx);
    ggml_trace_state * st = g_trace;
    if (st == nullptr || g_state.load(std::memory_order_relaxed) == TRACE_OFF || !st->token_pending) {
        return -1;
    }
    st->token_ev.aux = token;
    ggml_trace_flush_token(st);
    // the step's outputs have been waited for: its GPU timings are complete, resolve them before an idle gap
    ggml_trace_run_flush();
    return st->token_ev.step;
}

//
// I/O issue and residency requests
//

static std::pair<uintptr_t, size_t> ggml_trace_range_key(uintptr_t addr, size_t len) {
    return { addr, len };
}

static void ggml_trace_wake_poller(ggml_trace_state * st) {
#if defined(GGML_TRACE_LINUX)
    if (st->efd >= 0 && st->poller_sleeping.load(std::memory_order_seq_cst)) {
        const uint64_t one = 1;
        (void) !write(st->efd, &one, sizeof(one));
    }
#else
    GGML_UNUSED(st);
#endif
}

#if defined(GGML_TRACE_LINUX)
// the last page of a range is in the page cache (a hinted range is read in order, so it arrives last)
static bool ggml_trace_tail_resident(uintptr_t addr, size_t len) {
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t tail = (addr + len - 1) & ~(page - 1);
    unsigned char v = 0;
    return mincore((void *) tail, page, &v) == 0 && (v & 1);
}
#endif

int64_t ggml_trace_io_issue(int step, const void * addr, size_t len, int source, int rank, int layer, int expert,
        const char * name, int piece, int n_pieces, int64_t t0, int64_t t1, int64_t read_bytes_delta, int tid) {
    ggml_trace_guard g;
    ggml_trace_state * st = g.st;
    if (!st || !ggml_trace_io_enabled()) {
        return 0;
    }

    if (layer < 0 && name != nullptr) {
        layer = ggml_trace_parse_layer(name);
    }

    // no read caused by this hint: cached, unless another thread's read of these pages is still in flight
    bool cached = read_bytes_delta == 0;
#if defined(GGML_TRACE_LINUX)
    cached = cached && ggml_trace_tail_resident((uintptr_t) addr, len);
#endif

    ggml_trace_event ev;
    ggml_trace_event_init(&ev, GGML_TRACE_IO_ISSUE, tid);
    ev.step    = step;
    ev.pos     = step >= 0 ? step : g_last_step.load(std::memory_order_relaxed) + 1; // the step a hint in IDLE is for
    ev.t0_ns   = t0;
    ev.t1_ns   = t1;
    ev.detail  = source;
    ev.layer   = layer;
    ev.expert  = expert;
    ev.piece   = piece;
    ev.n_parts = n_pieces;
    ev.rank    = rank;
    ev.bytes   = (int64_t) len;
    ev.aux2    = read_bytes_delta;
    ggml_trace_event_set_name(&ev, name);

    const auto key = ggml_trace_range_key((uintptr_t) addr, len);
    bool pending = false;
    {
        std::lock_guard<std::mutex> lock(st->io_mtx);
        auto it = st->io_pending.find(key);
        if (it != st->io_pending.end()) {
            ev.detail |= GGML_TRACE_IO_DUPLICATE;
            ev.aux     = it->second.cookie;
        } else {
            ev.aux = st->next_cookie.fetch_add(1, std::memory_order_relaxed);
            if (!cached) {
                ggml_trace_io_pending p = { ev.aux, (uintptr_t) addr, len, t1, layer, expert, piece, source, {} };
                memcpy(p.name, ev.name, sizeof(p.name));
                st->io_pending[key] = p;
                pending = true;
            }
        }
    }

    if (!pending && !(ev.detail & GGML_TRACE_IO_DUPLICATE)) {
        ev.detail |= GGML_TRACE_IO_CACHED_FLAG;
    }
    ggml_trace_emit_locked(st, &ev);

    if (ev.detail & GGML_TRACE_IO_CACHED_FLAG) {
        ggml_trace_event res;
        ggml_trace_event_init(&res, GGML_TRACE_IO_RESIDENT, tid);
        res.step   = step;
        res.t0_ns  = t1;
        res.detail = GGML_TRACE_IO_CACHED;
        res.layer  = layer;
        res.expert = expert;
        res.piece  = piece;
        res.rank   = source;
        res.aux    = ev.aux;
        res.bytes  = (int64_t) len;
        memcpy(res.name, ev.name, sizeof(res.name));
        ggml_trace_emit_locked(st, &res);
    }
    return ev.aux;
}

void ggml_trace_residency_request(int layer, int expert, const char * name, const void * addr, size_t len, int64_t t_first_touch) {
    ggml_trace_guard g;
    ggml_trace_state * st = g.st;
    if (!st) {
        return;
    }
    // bounded MPSC queue (Vyukov): a slot is free for position pos when seq == pos
    uint32_t pos = st->ring_enq.load(std::memory_order_relaxed);
    ggml_trace_ring_slot * slot;
    for (;;) {
        slot = &st->ring[pos % GGML_TRACE_RING_SIZE];
        const uint32_t seq = slot->seq.load(std::memory_order_acquire);
        const int32_t  dif = (int32_t) (seq - pos);
        if (dif == 0) {
            if (st->ring_enq.compare_exchange_weak(pos, pos + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                break;
            }
        } else if (dif < 0) {
            st->ring_drops.fetch_add(1, std::memory_order_relaxed);
            return;
        } else {
            pos = st->ring_enq.load(std::memory_order_relaxed);
        }
    }
    ggml_trace_residency_req & r = slot->req;
    r.layer  = layer;
    r.expert = expert;
    r.step   = g_step.load(std::memory_order_relaxed);
    r.addr   = (uintptr_t) addr;
    r.len    = len;
    r.t_first_touch = t_first_touch;
    {
        ggml_trace_event tmp;
        ggml_trace_event_set_name(&tmp, name);
        memcpy(r.name, tmp.name, sizeof(r.name));
    }
    slot->seq.store(pos + 1, std::memory_order_release);

    ggml_trace_wake_poller(st);
}

//
// poller (Linux)
//

#if defined(GGML_TRACE_LINUX)

// resident pages of [addr, addr + len), page-aligned; -1 if mincore fails
static int64_t ggml_trace_mincore(uintptr_t addr, size_t len, std::vector<unsigned char> & vec, int64_t * n_pages) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const uintptr_t beg = addr & ~(uintptr_t) (page - 1);
    const uintptr_t end = (addr + len + page - 1) & ~(uintptr_t) (page - 1);
    const size_t n = (end - beg) / page;
    *n_pages = (int64_t) n;
    if (n == 0) {
        return 0;
    }
    if (vec.size() < n) {
        vec.resize(n);
    }
    if (mincore((void *) beg, end - beg, vec.data()) != 0) {
        return -1;
    }
    int64_t resident = 0;
    for (size_t i = 0; i < n; ++i) {
        resident += vec[i] & 1;
    }
    return resident;
}

static bool ggml_trace_ring_pop(ggml_trace_state * st, ggml_trace_residency_req & out) {
    ggml_trace_ring_slot * slot = &st->ring[st->ring_deq % GGML_TRACE_RING_SIZE];
    const uint32_t seq = slot->seq.load(std::memory_order_acquire);
    if ((int32_t) (seq - (st->ring_deq + 1)) != 0) {
        return false;
    }
    out = slot->req;
    slot->seq.store(st->ring_deq + GGML_TRACE_RING_SIZE, std::memory_order_release);
    st->ring_deq++;
    return true;
}

static void ggml_trace_poller_residency(ggml_trace_state * st, std::vector<unsigned char> & vec) {
    ggml_trace_residency_req r;
    while (ggml_trace_ring_pop(st, r)) {
        int64_t n_pages = 0;
        const int64_t resident = ggml_trace_mincore(r.addr, r.len, vec, &n_pages);
        const int64_t now = ggml_trace_now_ns();

        ggml_trace_event ev;
        ggml_trace_event_init(&ev, GGML_TRACE_EXPERT_RESIDENCY, GGML_TRACE_TID_POLLER);
        ev.t0_ns  = now;
        ev.step   = r.step;
        ev.layer  = r.layer;
        ev.expert = r.expert;
        ev.bytes  = resident;
        ev.aux    = n_pages;
        ev.aux2   = now - r.t_first_touch;
        memcpy(ev.name, r.name, sizeof(ev.name));
        ggml_trace_emit_locked(st, &ev);
    }
}

// one pass over the pending I/O ranges; resolved ranges leave the table
static void ggml_trace_poller_io(ggml_trace_state * st, std::vector<unsigned char> & vec, int64_t t_poll) {
    std::vector<ggml_trace_io_pending> snapshot;
    {
        std::lock_guard<std::mutex> lock(st->io_mtx);
        snapshot.reserve(st->io_pending.size());
        for (const auto & kv : st->io_pending) {
            snapshot.push_back(kv.second);
        }
    }

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    std::vector<ggml_trace_io_pending> resolved;
    std::vector<std::pair<uintptr_t, size_t>> not_resident;
    for (auto & p : snapshot) {
        // the tail page first: readahead fills the range in order, so a non-resident tail skips the full probe
        int64_t n_pages = 0;
        const uintptr_t tail = (p.addr + p.len - 1) & ~(uintptr_t) (page - 1);
        if (ggml_trace_mincore(tail, 1, vec, &n_pages) == 1 && ggml_trace_mincore(p.addr, p.len, vec, &n_pages) == n_pages) {
            resolved.push_back(p);
        } else {
            not_resident.push_back(ggml_trace_range_key(p.addr, p.len));
        }
    }
    const int64_t t_done = ggml_trace_now_ns();

    {
        // only the probed ranges: the ones issued during this pass were never seen not resident by it
        std::lock_guard<std::mutex> lock(st->io_mtx);
        for (const auto & p : resolved) {
            st->io_pending.erase(ggml_trace_range_key(p.addr, p.len));
        }
        for (const auto & key : not_resident) {
            auto it = st->io_pending.find(key);
            if (it != st->io_pending.end()) {
                it->second.t_not_resident = std::max(it->second.t_not_resident, t_poll);
            }
        }
    }

    for (const auto & p : resolved) {
        ggml_trace_event ev;
        ggml_trace_event_init(&ev, GGML_TRACE_IO_RESIDENT, GGML_TRACE_TID_POLLER);
        ev.t0_ns  = t_done;
        ev.detail = GGML_TRACE_IO_RESOLVED;
        ev.layer  = p.layer;
        ev.expert = p.expert;
        ev.piece  = p.piece;
        ev.rank   = p.source;
        ev.aux    = p.cookie;
        ev.aux2   = p.t_not_resident;
        ev.bytes  = (int64_t) p.len;
        memcpy(ev.name, p.name, sizeof(ev.name));
        ggml_trace_emit_locked(st, &ev);
    }
}

static void ggml_trace_poller_unresolved(ggml_trace_state * st, std::vector<unsigned char> & vec) {
    std::vector<ggml_trace_io_pending> snapshot;
    {
        std::lock_guard<std::mutex> lock(st->io_mtx);
        for (const auto & kv : st->io_pending) {
            snapshot.push_back(kv.second);
        }
        st->io_pending.clear();
    }
    const int64_t now = ggml_trace_now_ns();
    for (const auto & p : snapshot) {
        int64_t n_pages = 0;
        const int64_t resident = ggml_trace_mincore(p.addr, p.len, vec, &n_pages);
        ggml_trace_event ev;
        ggml_trace_event_init(&ev, GGML_TRACE_IO_RESIDENT, GGML_TRACE_TID_POLLER);
        ev.t0_ns  = now;
        ev.detail = GGML_TRACE_IO_UNRESOLVED;
        ev.layer  = p.layer;
        ev.expert = p.expert;
        ev.piece  = p.piece;
        ev.rank   = p.source;
        ev.aux    = p.cookie;
        ev.aux2   = resident;
        ev.bytes  = n_pages;
        memcpy(ev.name, p.name, sizeof(ev.name));
        ggml_trace_emit_locked(st, &ev);
    }
}

static void ggml_trace_poller_loop(ggml_trace_state * st) {
    std::vector<unsigned char> vec(4096);
    int64_t interval  = (int64_t) st->params.poll_us*1000;
    int64_t next_poll = ggml_trace_now_ns() + interval;

    while (!st->stop_threads.load(std::memory_order_acquire)) {
        const int64_t now = ggml_trace_now_ns();
        if (now < next_poll) {
            struct pollfd pfd = { st->efd, POLLIN, 0 };
            st->poller_sleeping.store(true, std::memory_order_seq_cst);
            // a request may have arrived between the last drain and the sleep flag
            if ((int32_t) (st->ring_enq.load(std::memory_order_seq_cst) - st->ring_deq) <= 0) {
                const int64_t wait = next_poll - now;
                struct timespec ts = { (time_t) (wait / 1000000000), (long) (wait % 1000000000) };
                (void) ppoll(&pfd, 1, &ts, nullptr);
            }
            st->poller_sleeping.store(false, std::memory_order_seq_cst);
            if (pfd.revents & POLLIN) {
                uint64_t v;
                (void) !read(st->efd, &v, sizeof(v));
            }
        }

        ggml_trace_poller_residency(st, vec);

        const int64_t t_poll = ggml_trace_now_ns();
        if (t_poll >= next_poll) {
            ggml_trace_poller_io(st, vec, t_poll);
            const int64_t took = ggml_trace_now_ns() - t_poll;
            if (took > interval) {
                interval *= 2;
                st->poll_adaptations.fetch_add(1, std::memory_order_relaxed);
                ggml_trace_event ev;
                ggml_trace_event_init(&ev, GGML_TRACE_META, GGML_TRACE_TID_POLLER);
                ev.t0_ns = t_poll;
                ev.aux   = interval/1000;
                ev.aux2  = took;
                ggml_trace_emit_locked(st, &ev);
            }
            next_poll = t_poll + interval;
        }
    }

    ggml_trace_poller_residency(st, vec);
    ggml_trace_poller_unresolved(st, vec);
}

#endif // GGML_TRACE_LINUX

//
// writer: records -> Chrome Trace Event JSON
//

enum { PID_HOST = 1, PID_DEVICE0 = 2, PID_EXPERT = 100, PID_IO = 101 };
enum { TID_SCHED = 1, TID_PREDICTOR = 2, TID_POLLER = 3, TID_SAMPLER = 4, TID_PREDICTOR1 = 5, TID_WORKER0 = 100 };

static const char * ggml_trace_type_name(int type) {
    static const char * names[GGML_TRACE_TYPE_COUNT] = {
        "TOKEN", "SYNC", "SAMPLE", "XFER", "SPLIT", "COPY", "OP_CPU", "OP_GPU", "EXPERT_THREAD", "EXPERT",
        "EXPERT_RESIDENCY", "IO_ISSUE", "IO_RESIDENT", "PREDICT", "META",
    };
    return type >= 0 && type < GGML_TRACE_TYPE_COUNT ? names[type] : "?";
}

static const char * ggml_trace_source_name(int source) {
    switch (source & GGML_TRACE_IO_SOURCE_MASK) {
        case GGML_TRACE_IO_PREFETCH:     return "prefetch";
        case GGML_TRACE_IO_PREDICT_HIGH: return "predict_high";
        case GGML_TRACE_IO_PREDICT_LOW:  return "predict_low";
        default:                         return "other";
    }
}

static int ggml_trace_host_tid(int tid) {
    switch (tid) {
        case GGML_TRACE_TID_SCHED:     return TID_SCHED;
        case GGML_TRACE_TID_PREDICTOR: return TID_PREDICTOR;
        case GGML_TRACE_TID_POLLER:    return TID_POLLER;
        case GGML_TRACE_TID_SAMPLER:   return TID_SAMPLER;
        case GGML_TRACE_TID_PREDICTOR1: return TID_PREDICTOR1;
        default:                       return tid >= 0 ? TID_WORKER0 + tid : TID_SCHED;
    }
}

// 1 gate, 2 up, 3 down, 4 other: from node names (ffn_moe_up-12) and tensor names (blk.12.ffn_up_exps.weight)
static int ggml_trace_tensor_kind(const char * name) {
    if (strstr(name, "gate")) {
        return 1;
    }
    if (strstr(name, "up")) {
        return 2;
    }
    if (strstr(name, "down")) {
        return 3;
    }
    return 4;
}

static const char * ggml_trace_kind_name(int kind) {
    static const char * names[] = { "?", "gate", "up", "down", "other" };
    return kind >= 1 && kind <= 4 ? names[kind] : "?";
}

// layer tag: "-<il>" suffix, "blk.<il>.", "_l<il>"; -1 if none
static int ggml_trace_parse_layer(const char * name) {
    const char * p = strstr(name, "blk.");
    if (p && p[4] >= '0' && p[4] <= '9') {
        return atoi(p + 4);
    }
    const char * dash = strrchr(name, '-');
    if (dash && dash[1] >= '0' && dash[1] <= '9') {
        const char * q = dash + 1;
        while (*q >= '0' && *q <= '9') {
            q++;
        }
        if (*q == 0) {
            return atoi(dash + 1);
        }
    }
    p = strstr(name, "_l");
    if (p && p[2] >= '0' && p[2] <= '9') {
        return atoi(p + 2);
    }
    return -1;
}

struct ggml_trace_json {
    std::string s;

    void raw(const char * fmt, ...) GGML_ATTRIBUTE_FORMAT(2, 3) {
        char buf[512];
        va_list args;
        va_start(args, fmt);
        va_list args2;
        va_copy(args2, args);
        const int n = vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        if (n < 0) {
            va_end(args2);
            return;
        }
        if ((size_t) n < sizeof(buf)) {
            s.append(buf, (size_t) n);
        } else {
            std::vector<char> big((size_t) n + 1);
            vsnprintf(big.data(), big.size(), fmt, args2);
            s.append(big.data(), (size_t) n);
        }
        va_end(args2);
    }

    void str(const char * v) {
        s += '"';
        for (; *v; ++v) {
            const unsigned char c = (unsigned char) *v;
            if (c == '"' || c == '\\') {
                s += '\\';
                s += (char) c;
            } else if (c < 0x20) {
                raw("\\u%04x", c);
            } else {
                s += (char) c;
            }
        }
        s += '"';
    }
};

// EXPERT_THREAD records of one (step, name, expert), merged when all threads reported
struct ggml_trace_expert_agg {
    int     n = 0;
    int     n_parts = 0;
    int64_t t0 = INT64_MAX, t1 = 0;
    int64_t stall = 0, minflt = 0, majflt = 0, bytes = 0;
    int32_t layer = -1;
    int     hold = 0;  // writer passes since completion, waiting for the residency sample
    bool    complete = false;
    bool    have_res = false;
    int64_t res_pages = 0, res_total = 0, res_delay = 0;
};

// one piece of the detailed layout: its I/O span from the issue to resident
struct ggml_trace_slice_agg {
    int     n_pieces = 0, n_issued = 0, n_resident = 0;
    int64_t t0 = INT64_MAX, t1 = 0, issue_dur = 0, bytes = 0, read_bytes = 0;
    int64_t t_res = 0, t_prev_poll = INT64_MAX;
    int64_t res_pages = 0, res_total = 0;
    bool    all_cached = true, any_dup = false, any_unresolved = false, all_issued = false;
    int64_t id = 0;
    int32_t piece = -1; // the piece, with cookie = id
    int32_t layer = -1, expert = -1, source = 0, rank = -1, step = -1;
    char    name[32] = {};
    std::vector<int64_t> cookies;
};

struct ggml_trace_resident {
    int64_t t = 0, t_prev = 0, pages = 0, resident = 0;
    int     detail = 0;
};

// spans that may overlap (expert compute, issue -> resident) go on lanes: Perfetto shows "X" events of one thread
// as a stack, so each logical track (e.g. gate, or up/prefetch) gets as many thread tracks as it needs at once.
// a span takes the first lane where it overlaps nothing; spans arrive out of order, so a lane keeps its intervals
struct ggml_trace_lanes {
    std::vector<std::map<int64_t, int64_t>> lanes; // t0 -> t1

    static bool overlaps(const std::map<int64_t, int64_t> & l, int64_t t0, int64_t t1) {
        auto it = l.lower_bound(t0);
        if (it != l.end() && it->first < t1) {
            return true;
        }
        return it != l.begin() && std::prev(it)->second > t0;
    }

    int place(int64_t t0, int64_t t1) {
        t1 = std::max(t1, t0 + 1); // a zero-length span still occupies its instant
        for (size_t i = 0; i < lanes.size(); ++i) {
            if (!overlaps(lanes[i], t0, t1)) {
                lanes[i][t0] = t1;
                return (int) i;
            }
        }
        lanes.emplace_back();
        lanes.back()[t0] = t1;
        return (int) lanes.size() - 1;
    }

    // intervals that ended before horizon cannot meet new spans any more (the writer's lag is far below that)
    void prune(int64_t horizon) {
        for (auto & l : lanes) {
            for (auto it = l.begin(); it != l.end() && it->first < horizon; ) {
                it = it->second < horizon ? l.erase(it) : std::next(it);
            }
        }
    }
};

// tids on the expert compute / expert I/O processes: logical track*100 + lane, residency instants 1000 + kind
enum { TID_LANES = 100, TID_RESIDENCY = 1000 };

struct ggml_trace_writer {
    ggml_trace_state * st;
    ggml_trace_json    j;
    uint64_t n_written = 0; // JSON events
    uint64_t n_records = 0; // records read from the chunks
    int64_t  next_id = 1;
    std::set<std::pair<int, int>> named_tracks;
    std::set<int>                 named_pids;
    std::map<std::string, ggml_trace_expert_agg> experts;
    std::map<std::string, ggml_trace_slice_agg>  slices;
    std::unordered_map<int64_t, std::vector<std::string>> cookie_slices;
    std::unordered_map<int64_t, ggml_trace_resident>      resolved; // by cookie, for issues read after their resolution
    int64_t t_newest = 0; // latest resolution seen
    int64_t t_span_newest = 0; // latest span start placed on a lane
    std::map<std::pair<int, int>, ggml_trace_lanes> lanes; // (pid, logical track)

    // the tid of a span on its logical track
    int lane_tid(int pid, int track, int64_t t0, int64_t t1) {
        t_span_newest = std::max(t_span_newest, t0);
        return track*TID_LANES + lanes[{ pid, track }].place(t0, t1);
    }

    explicit ggml_trace_writer(ggml_trace_state * st) : st(st) {}

    double us(int64_t t_ns) const { return (double) (t_ns - st->t_base)/1000.0; }

    void name_pid(int pid, const char * name) {
        if (named_pids.insert(pid).second) {
            j.raw("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":%d,\"args\":{\"name\":", pid);
            j.str(name);
            j.raw("}},\n");
        }
    }

    void name_track(int pid, int tid) {
        if (!named_tracks.insert({ pid, tid }).second) {
            return;
        }
        std::string pname, tname;
        if (pid == PID_HOST) {
            pname = "host";
            switch (tid) {
                case TID_SCHED:     tname = "sched";     break;
                case TID_PREDICTOR: tname = "predictor"; break;
                case TID_POLLER:    tname = "poller";    break;
                case TID_SAMPLER:   tname = "sampler";   break;
                case TID_PREDICTOR1: tname = "predictor 1"; break;
                default:            tname = "worker " + std::to_string(tid - TID_WORKER0); break;
            }
        } else if (pid == PID_EXPERT) {
            pname = "expert compute";
            if (tid >= TID_RESIDENCY) {
                tname = std::string(ggml_trace_kind_name(tid - TID_RESIDENCY)) + " residency";
            } else {
                const int lane = tid % TID_LANES;
                tname = ggml_trace_kind_name(tid / TID_LANES);
                if (lane > 0) {
                    tname += " " + std::to_string(lane);
                }
            }
        } else if (pid == PID_IO) {
            pname = "expert I/O";
            const int track = tid / TID_LANES;
            tname = std::string(ggml_trace_kind_name(track / 4)) + "/" + ggml_trace_source_name(track % 4) + " " + std::to_string(tid % TID_LANES);
        } else {
            pname = ggml_trace_backend_name(pid - PID_DEVICE0 + 1);
            tname = tid == 1 ? "device" : "launch";
        }
        name_pid(pid, pname.c_str());
        j.raw("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":", pid, tid);
        j.str(tname.c_str());
        j.raw("}},\n");
    }

    // common head of an event: name, cat, ph, ts[, dur][, id], pid, tid, then "args":{ ... opened
    void head(const char * name, const char * cat, char ph, int64_t t0, int64_t t1, int pid, int tid, int64_t id = 0) {
        name_track(pid, tid);
        j.raw("{\"name\":");
        j.str(name);
        j.raw(",\"cat\":\"%s\",\"ph\":\"%c\",\"ts\":%.3f", cat, ph, us(t0));
        if (ph == 'X') {
            j.raw(",\"dur\":%.3f", (double) (t1 - t0)/1000.0);
        } else if (ph == 'i') {
            j.raw(",\"s\":\"t\"");
        } else {
            j.raw(",\"id\":%" PRId64, id);
        }
        j.raw(",\"pid\":%d,\"tid\":%d,\"args\":{", pid, tid);
    }

    void tail() {
        j.raw("}},\n");
        n_written++;
    }

    void arg_step(const ggml_trace_event & ev) {
        j.raw("\"step\":%d", ev.step);
    }

    // layer of a node event, tagged at emission (ggml_trace_tag_layer)
    int layer_of(const ggml_trace_event & ev, bool & inferred) {
        inferred = (ev.flags & GGML_TRACE_FLAG_LAYER_INFERRED) != 0;
        return ev.layer;
    }

    // ",\"rank\":\"r3\"" for the 4th router guess, "p0" for the first popularity guess, null for no guess
    void arg_rank(int rank) {
        if (rank < 0) {
            j.raw(",\"rank\":null");
        } else if (rank >= GGML_TRACE_RANK_POPULARITY) {
            j.raw(",\"rank\":\"p%d\"", rank - GGML_TRACE_RANK_POPULARITY);
        } else {
            j.raw(",\"rank\":\"r%d\"", rank);
        }
    }

    void arg_layer(int layer, bool inferred) {
        j.raw(",\"layer\":%d", layer);
        if (inferred) {
            j.raw(",\"layer_inferred\":true");
        }
    }

    void write_expert(const std::string & key, ggml_trace_expert_agg & a, const char * name, int expert, int step, bool partial) {
        const int kind = ggml_trace_tensor_kind(name);
        char label[64];
        snprintf(label, sizeof(label), "L%d e%d", a.layer, expert);
        head(label, "expert", 'X', a.t0, a.t1, PID_EXPERT, lane_tid(PID_EXPERT, kind, a.t0, a.t1));
        j.raw("\"step\":%d,\"layer\":%d,\"expert\":%d,\"name\":", step, a.layer, expert);
        j.str(name);
        j.raw(",\"bytes\":%" PRId64 ",\"stall_ns\":%" PRId64 ",\"t_first_touch_ns\":%" PRId64 ",\"n_threads\":%d",
                a.bytes, a.stall, a.t0 - st->t_base, a.n);
        if (st->params.detail >= 2) {
            j.raw(",\"minflt\":%" PRId64 ",\"majflt\":%" PRId64, a.minflt, a.majflt);
        }
        if (a.have_res) {
            j.raw(",\"resident_frac\":%.4f,\"sample_delay_ns\":%" PRId64, a.res_total > 0 && a.res_pages >= 0 ? (double) a.res_pages/a.res_total : -1.0, a.res_delay);
        } else {
            j.raw(",\"resident_frac\":null");
        }
        if (partial) {
            j.raw(",\"partial\":true");
        }
        tail();
        experts.erase(key);
    }

    void on_expert_thread(const ggml_trace_event & ev) {
        if (st->params.detail >= 2) {
            const int tid = ggml_trace_host_tid(ev.tid);
            char label[64];
            snprintf(label, sizeof(label), "e%d", ev.expert);
            head(label, "expert_thread", 'X', ev.t0_ns, ev.t1_ns, PID_HOST, tid);
            j.raw("\"step\":%d,\"layer\":%d,\"expert\":%d,\"name\":", ev.step, ggml_trace_parse_layer(ev.name), ev.expert);
            j.str(ev.name);
            j.raw(",\"stall_ns\":%" PRId64 ",\"minflt\":%" PRId64 ",\"majflt\":%" PRId64,
                    ev.aux, (int64_t) (ev.aux2 & 0xffffffff), (int64_t) (ev.aux2 >> 32));
            tail();
        }
        const std::string key = std::to_string(ev.step) + "|" + ev.name + "|" + std::to_string(ev.expert);
        ggml_trace_expert_agg & a = experts[key];
        a.n++;
        a.n_parts = ev.n_parts;
        a.t0      = std::min(a.t0, ev.t0_ns);
        a.t1      = std::max(a.t1, ev.t1_ns);
        a.stall  += ev.aux;
        a.minflt += (int64_t) (ev.aux2 & 0xffffffff);
        a.majflt += (int64_t) (ev.aux2 >> 32);
        a.bytes   = ev.bytes;
        a.layer   = ggml_trace_parse_layer(ev.name);
        if (a.n >= a.n_parts) {
            a.complete = true;
        }
    }

    void on_expert_residency(const ggml_trace_event & ev) {
        const int kind = ggml_trace_tensor_kind(ev.name);
        const int layer = ggml_trace_parse_layer(ev.name);
        char label[64];
        snprintf(label, sizeof(label), "RES L%d e%d", layer, ev.expert);
        head(label, "residency", 'i', ev.t0_ns, 0, PID_EXPERT, TID_RESIDENCY + kind);
        j.raw("\"step\":%d,\"layer\":%d,\"expert\":%d,\"name\":", ev.step, layer, ev.expert);
        j.str(ev.name);
        j.raw(",\"resident_pages\":%" PRId64 ",\"pages\":%" PRId64 ",\"resident_frac\":%.4f,\"sample_delay_ns\":%" PRId64,
                ev.bytes, ev.aux, ev.aux > 0 && ev.bytes >= 0 ? (double) ev.bytes/ev.aux : -1.0, ev.aux2);
        tail();

        const std::string key = std::to_string(ev.step) + "|" + ev.name + "|" + std::to_string(ev.expert);
        ggml_trace_expert_agg & a = experts[key];
        a.have_res  = true;
        a.res_pages = ev.bytes;
        a.res_total = ev.aux;
        a.res_delay = ev.aux2;
    }

    // completed EXPERT aggregates are held for one pass so that the residency sample can attach
    void flush_experts(bool final) {
        for (auto it = experts.begin(); it != experts.end(); ) {
            ggml_trace_expert_agg & a = it->second;
            const std::string key = it->first;
            ++it;
            if (!a.complete && !final) {
                continue;
            }
            if (a.complete && !a.have_res && !final && a.hold++ < 1) {
                continue;
            }
            if (!a.complete && a.n == 0) {
                experts.erase(key); // a residency sample without compute records
                continue;
            }
            // key = step|name|expert
            const size_t p1 = key.find('|');
            const size_t p2 = key.rfind('|');
            const int step   = atoi(key.c_str());
            const int expert = atoi(key.c_str() + p2 + 1);
            const std::string name = key.substr(p1 + 1, p2 - p1 - 1);
            write_expert(key, a, name.c_str(), expert, step, !a.complete);
        }
    }

    void write_io_issue_x(const ggml_trace_event & ev) {
        const int tid = ggml_trace_host_tid(ev.tid);
        char label[64];
        snprintf(label, sizeof(label), "ISSUE L%d e%d p%d", ev.layer, ev.expert, ev.piece);
        head(label, "io_issue", 'X', ev.t0_ns, ev.t1_ns, PID_HOST, tid);
        j.raw("\"step\":%d,\"layer\":%d,\"expert\":%d,\"name\":", ev.step, ev.layer, ev.expert);
        j.str(ev.name);
        j.raw(",\"source\":\"%s\"", ggml_trace_source_name(ev.detail));
        arg_rank(ev.rank);
        j.raw(",\"piece\":%d,\"n_pieces\":%d,\"bytes\":%" PRId64 ",\"cookie\":%" PRId64
              ",\"read_bytes\":%" PRId64 ",\"cached\":%s,\"duplicate\":%s",
                ev.piece, ev.n_parts, ev.bytes, ev.aux, ev.aux2,
                (ev.detail & GGML_TRACE_IO_CACHED_FLAG) ? "true" : "false", (ev.detail & GGML_TRACE_IO_DUPLICATE) ? "true" : "false");
        tail();
    }

    int io_tid(const char * name, int source) {
        return ggml_trace_tensor_kind(name)*4 + (source & GGML_TRACE_IO_SOURCE_MASK);
    }

    // the issue -> resident span of a slice (detail 1) or piece (detail 2), once its end is known
    void write_slice(const std::string & key, ggml_trace_slice_agg & a, bool partial) {
        const int64_t t_end = a.n_resident > 0 ? std::max(a.t_res, a.t1) : a.t1;
        char label[64];
        if (a.piece >= 0) {
            snprintf(label, sizeof(label), "L%d e%d p%d", a.layer, a.expert, a.piece);
        } else {
            snprintf(label, sizeof(label), "L%d e%d", a.layer, a.expert);
        }
        const int track = io_tid(a.name, a.source);
        head(label, "io", 'X', a.t0, t_end, PID_IO, lane_tid(PID_IO, track, a.t0, t_end));
        j.raw("\"step\":%d,\"layer\":%d,\"expert\":%d,\"name\":", a.step, a.layer, a.expert);
        j.str(a.name);
        j.raw(",\"source\":\"%s\"", ggml_trace_source_name(a.source));
        arg_rank(a.rank);
        j.raw(",\"id\":%" PRId64 ",\"n_pieces\":%d,\"bytes\":%" PRId64 ",\"issue_dur_ns\":%" PRId64
              ",\"read_bytes\":%" PRId64 ",\"cached\":%s,\"duplicate\":%s,\"unresolved\":%s",
                a.id, a.n_pieces, a.bytes, a.issue_dur, a.read_bytes,
                a.all_cached ? "true" : "false", a.any_dup ? "true" : "false", a.any_unresolved ? "true" : "false");
        if (a.piece >= 0) {
            j.raw(",\"piece\":%d,\"cookie\":%" PRId64 ",\"resolution\":\"%s\"", a.piece, a.id,
                    a.n_resident == 0 ? "none" : a.any_unresolved ? "unresolved" : a.all_cached ? "cached" : "resolved");
        }
        if (a.n_resident > 0) {
            j.raw(",\"t_resident_ns\":%" PRId64, a.t_res - st->t_base);
        }
        if (a.any_unresolved) {
            j.raw(",\"resident_frac\":%.4f", a.res_total > 0 ? (double) a.res_pages/a.res_total : -1.0);
        }
        if (a.t_prev_poll != INT64_MAX) {
            j.raw(",\"t_prev_poll_ns\":%" PRId64, a.t_prev_poll - st->t_base);
        }
        if (partial) {
            j.raw(",\"partial\":true");
        }
        tail();
        for (int64_t c : a.cookies) {
            auto it = cookie_slices.find(c);
            if (it != cookie_slices.end()) {
                it->second.erase(std::remove(it->second.begin(), it->second.end(), key), it->second.end());
                if (it->second.empty()) {
                    cookie_slices.erase(it);
                }
            }
        }
        slices.erase(key);
    }

    void slice_add_resident(ggml_trace_slice_agg & a, const ggml_trace_resident & r) {
        a.n_resident++;
        a.t_res = std::max(a.t_res, r.t);
        if (r.detail == GGML_TRACE_IO_RESOLVED) {
            a.t_prev_poll = std::min(a.t_prev_poll, r.t_prev);
        } else if (r.detail == GGML_TRACE_IO_UNRESOLVED) {
            a.any_unresolved = true;
            a.res_pages += std::max<int64_t>(0, r.resident);
            a.res_total += r.pages;
        }
    }

    void on_io_issue(const ggml_trace_event & ev) {
        write_io_issue_x(ev);

        // the detailed layout: one span per piece, keyed by its cookie (a duplicate only shows on the issuing
        // thread; the simple layout merges the pieces per layer, expert and step instead)
        if (ev.detail & GGML_TRACE_IO_DUPLICATE) {
            return;
        }
        const std::string key = "c|" + std::to_string(ev.aux);
        ggml_trace_slice_agg & a = slices[key];
        if (a.n_issued == 0) {
            a.id     = ev.aux;
            a.piece  = ev.piece;
            a.layer  = ev.layer;
            a.expert = ev.expert;
            a.source = ev.detail & GGML_TRACE_IO_SOURCE_MASK;
            a.rank   = ev.rank;
            a.step   = ev.step;
            memcpy(a.name, ev.name, sizeof(a.name));
        }
        a.n_pieces   = 1;
        a.n_issued++;
        a.t0         = std::min(a.t0, ev.t0_ns);
        a.t1         = std::max(a.t1, ev.t1_ns);
        a.issue_dur += ev.t1_ns - ev.t0_ns;
        a.bytes     += ev.bytes;
        a.read_bytes += ev.aux2 > 0 ? ev.aux2 : 0;
        a.all_cached = a.all_cached && (ev.detail & GGML_TRACE_IO_CACHED_FLAG);
        a.any_dup    = a.any_dup || (ev.detail & GGML_TRACE_IO_DUPLICATE);
        a.cookies.push_back(ev.aux);

        auto it = resolved.find(ev.aux);
        if (it != resolved.end()) {
            slice_add_resident(a, it->second);
        } else {
            cookie_slices[ev.aux].push_back(key);
        }
        if (a.n_issued >= a.n_pieces) {
            a.all_issued = true;
            if (a.n_resident >= a.n_pieces) {
                write_slice(key, a, false);
            }
        }
    }

    void on_io_resident(const ggml_trace_event & ev) {
        ggml_trace_resident r;
        r.t        = ev.t0_ns;
        r.detail   = ev.detail;
        r.t_prev   = ev.detail == GGML_TRACE_IO_RESOLVED ? ev.aux2 : 0;
        r.pages    = ev.detail == GGML_TRACE_IO_UNRESOLVED ? ev.bytes : 0;
        r.resident = ev.detail == GGML_TRACE_IO_UNRESOLVED ? ev.aux2 : 0;
        resolved[ev.aux] = r;
        t_newest = std::max(t_newest, r.t);

        auto it = cookie_slices.find(ev.aux);
        if (it == cookie_slices.end()) {
            return;
        }
        const std::vector<std::string> keys = it->second;
        cookie_slices.erase(it);
        for (const std::string & key : keys) {
            auto sit = slices.find(key);
            if (sit == slices.end()) {
                continue;
            }
            ggml_trace_slice_agg & a = sit->second;
            slice_add_resident(a, r);
            if (a.all_issued && a.n_resident >= a.n_pieces) {
                write_slice(key, a, false);
            }
        }
    }


    //
    // simple layout (detail 1): one process "simple" with the rows token, GPU, worker N, load N, copy N; every span
    // of a layer and expert is named exactly "L<layer> e<expert>", every other span of a layer "L<layer>", so that
    // Perfetto gives them the same color in every row. the records of a step are collected until the step is
    // complete (its TOKEN record has arrived and two more writer passes have run), then its rows are written;
    // loads are written once all their pieces are resident. the per-step counts the checks need are in the args
    // of the token span
    //

    struct simple_step {
        bool    have_token = false;
        int     hold       = 0;
        ggml_trace_event token;
        std::vector<ggml_trace_event> expert_threads;
        std::vector<ggml_trace_event> ops;    // worker 0, not MUL_MAT_ID
        std::vector<ggml_trace_event> gpu;
        std::vector<ggml_trace_event> copies; // COPY and XFER
        std::map<std::pair<std::string, int>, std::pair<double, int64_t>> residency; // (node, expert) -> frac, delay
        int     n_residency = 0, n_sync = 0, n_out_sync = 0, n_sample = 0, n_split = 0, n_predict = 0;
        int64_t sync_ns = 0, out_t0 = 0, out_t1 = 0, sample_t0 = 0, sample_t1 = 0, sample_token = -1;
        int64_t n_pieces[3] = { 0, 0, 0 }, n_pieces_cached = 0, n_pieces_dup = 0;
    };

    struct simple_load {
        int64_t t0 = INT64_MAX, t_issue_end = 0, t_res = 0, bytes = 0;
        int     n_pieces = 0, n_cached = 0, n_dup = 0;
        unsigned sources = 0;
        bool    unresolved = false;
        std::set<std::string> tensors;
        std::set<int64_t>     pending; // cookies not resident yet
    };

    using load_key = std::tuple<int, int, int>; // (step, layer, expert)

    std::map<int, simple_step> s_steps;
    std::set<int>              s_done; // steps whose rows have been written
    std::map<load_key, simple_load> s_loads;
    std::unordered_map<int64_t, std::vector<load_key>> s_cookie_loads;
    ggml_trace_lanes s_load_lanes, s_copy_lanes;
    int64_t s_newest = 0; // end of the latest written TOKEN

    enum { S_TID_TOKEN = 1, S_TID_GPU = 2, S_TID_WORKER0 = 100, S_TID_LOAD0 = 1000, S_TID_COPY0 = 2000 };

    void simple_track(int tid) {
        if (!named_tracks.insert({ PID_HOST, tid }).second) {
            return;
        }
        std::string tname;
        if (tid == S_TID_TOKEN) {
            tname = "token";
        } else if (tid == S_TID_GPU) {
            tname = "GPU";
        } else if (tid >= S_TID_COPY0) {
            tname = "copy " + std::to_string(tid - S_TID_COPY0);
        } else if (tid >= S_TID_LOAD0) {
            tname = "load " + std::to_string(tid - S_TID_LOAD0);
        } else {
            tname = "worker " + std::to_string(tid - S_TID_WORKER0);
        }
        name_pid(PID_HOST, "simple");
        j.raw("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":", PID_HOST, tid);
        j.str(tname.c_str());
        j.raw("}},\n");
        j.raw("{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"args\":{\"sort_index\":%d}},\n", PID_HOST, tid, tid);
    }

    // a span of the simple layout; args are written by the caller, then tail()
    void simple_head(const std::string & name, int64_t t0, int64_t t1, int tid) {
        simple_track(tid);
        j.raw("{\"name\":");
        j.str(name.c_str());
        j.raw(",\"cat\":\"simple\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":%d,\"args\":{",
                us(t0), (double) std::max<int64_t>(0, t1 - t0)/1000.0, PID_HOST, tid);
    }

    static std::string simple_name(int layer, int expert) {
        char buf[48];
        if (expert >= 0) {
            snprintf(buf, sizeof(buf), "L%d e%d", layer, expert);
        } else {
            snprintf(buf, sizeof(buf), "L%d", layer);
        }
        return buf;
    }

    void simple_on_event(const ggml_trace_event & ev) {
        // a record of a step whose rows are written already (e.g. a residency sample that came very late): dropped
        // and counted (late_records in the metadata); I/O records go to the loads, which are written separately
        if (ev.type != GGML_TRACE_IO_ISSUE && ev.type != GGML_TRACE_IO_RESIDENT && ev.step >= 0 && s_done.count(ev.step)) {
            s_late++;
            return;
        }
        switch (ev.type) {
            case GGML_TRACE_TOKEN:
                {
                    simple_step & s = s_steps[ev.step];
                    s.have_token = true;
                    s.token      = ev;
                } break;
            case GGML_TRACE_SYNC:
                {
                    if (ev.step < 0) {
                        break;
                    }
                    simple_step & s = s_steps[ev.step];
                    if (ev.detail == GGML_TRACE_OUTPUT) {
                        s.n_out_sync++;
                        s.out_t0 = ev.t0_ns;
                        s.out_t1 = ev.t1_ns;
                    } else {
                        s.n_sync++;
                        s.sync_ns += ev.t1_ns - ev.t0_ns;
                    }
                } break;
            case GGML_TRACE_SAMPLE:
                {
                    if (ev.step < 0) {
                        break;
                    }
                    simple_step & s = s_steps[ev.step];
                    s.n_sample++;
                    s.sample_t0    = ev.t0_ns;
                    s.sample_t1    = ev.t1_ns;
                    s.sample_token = ev.aux;
                } break;
            case GGML_TRACE_XFER:
            case GGML_TRACE_COPY:
                if (ev.step >= 0) {
                    s_steps[ev.step].copies.push_back(ev);
                } break;
            case GGML_TRACE_SPLIT:
                if (ev.step >= 0) {
                    s_steps[ev.step].n_split++;
                } break;
            case GGML_TRACE_OP_CPU:
                if (ev.step >= 0 && ev.op != GGML_OP_MUL_MAT_ID) {
                    s_steps[ev.step].ops.push_back(ev);
                } break;
            case GGML_TRACE_OP_GPU:
                if (ev.step >= 0) {
                    s_steps[ev.step].gpu.push_back(ev);
                } break;
            case GGML_TRACE_EXPERT_THREAD:
                if (ev.step >= 0) {
                    s_steps[ev.step].expert_threads.push_back(ev);
                } break;
            case GGML_TRACE_EXPERT_RESIDENCY:
                if (ev.step >= 0) {
                    simple_step & s = s_steps[ev.step];
                    s.n_residency++;
                    s.residency[{ ev.name, ev.expert }] = { ev.aux > 0 && ev.bytes >= 0 ? (double) ev.bytes/ev.aux : -1.0, ev.aux2 };
                } break;
            case GGML_TRACE_PREDICT:
                if (ev.step >= 0) {
                    s_steps[ev.step].n_predict++;
                } break;
            case GGML_TRACE_IO_ISSUE:   simple_io_issue(ev);    break;
            case GGML_TRACE_IO_RESIDENT: simple_io_resident(ev); break;
            default:
                break;
        }
    }

    void simple_io_issue(const ggml_trace_event & ev) {
        const int src = ev.detail & GGML_TRACE_IO_SOURCE_MASK;
        const bool cached = (ev.detail & GGML_TRACE_IO_CACHED_FLAG) != 0;
        const bool dup    = (ev.detail & GGML_TRACE_IO_DUPLICATE) != 0;
        if (ev.pos >= 0) {
            simple_step & s = s_steps[ev.pos];
            s.n_pieces[std::min(src, 2)]++;
            s.n_pieces_cached += cached;
            s.n_pieces_dup    += dup;
        }
        const load_key key { ev.pos, ev.layer, ev.expert };
        simple_load & l = s_loads[key];
        l.t0          = std::min(l.t0, ev.t0_ns);
        l.t_issue_end = std::max(l.t_issue_end, ev.t1_ns);
        l.bytes      += ev.bytes;
        l.n_pieces++;
        l.n_cached   += cached;
        l.n_dup      += dup;
        l.sources    |= 1u << src;
        l.tensors.insert(ev.name);
        if (cached) {
            return;
        }
        auto it = resolved.find(ev.aux);
        if (it != resolved.end()) {
            simple_load_resident(l, it->second);
        } else {
            l.pending.insert(ev.aux);
            s_cookie_loads[ev.aux].push_back(key);
        }
    }

    void simple_load_resident(simple_load & l, const ggml_trace_resident & r) {
        l.t_res = std::max(l.t_res, r.t);
        l.unresolved = l.unresolved || r.detail == GGML_TRACE_IO_UNRESOLVED;
    }

    void simple_io_resident(const ggml_trace_event & ev) {
        ggml_trace_resident r;
        r.t      = ev.t0_ns;
        r.detail = ev.detail;
        resolved[ev.aux] = r;
        t_newest = std::max(t_newest, r.t);
        auto it = s_cookie_loads.find(ev.aux);
        if (it == s_cookie_loads.end()) {
            return;
        }
        for (const load_key & key : it->second) {
            auto lit = s_loads.find(key);
            if (lit != s_loads.end()) {
                simple_load_resident(lit->second, r);
                lit->second.pending.erase(ev.aux);
            }
        }
        s_cookie_loads.erase(it);
    }

    void simple_write_step(int step, simple_step & s) {
        // token: the counts the checks need are in its args
        int n_experts = 0;
        {
            std::set<std::pair<std::string, int>> ex;
            for (const auto & e : s.expert_threads) {
                ex.insert({ e.name, e.expert });
            }
            n_experts = (int) ex.size();
        }
        int64_t copy_bytes = 0;
        for (const auto & c : s.copies) {
            copy_bytes += c.bytes;
        }
        int n_loads = 0, n_loads_cached = 0;
        for (auto it = s_loads.lower_bound(load_key { step, INT_MIN, INT_MIN }); it != s_loads.end() && std::get<0>(it->first) == step; ++it) {
            n_loads++;
            n_loads_cached += it->second.n_cached == it->second.n_pieces;
        }
        // the experts this step computed, for the loads written later ("computed" in their args)
        {
            auto & comp = s_computed[step];
            for (const auto & e : s.expert_threads) {
                comp.insert({ ggml_trace_parse_layer(e.name), e.expert });
            }
        }
        if (s.have_token) {
            const ggml_trace_event & t = s.token;
            char name[64];
            snprintf(name, sizeof(name), "token %d -> %" PRId64, step, t.aux);
            simple_head(name, t.t0_ns, t.t1_ns, S_TID_TOKEN);
            j.raw("\"step\":%d,\"token\":%d,\"pos\":%d,\"token_out\":%" PRId64, step, t.token, t.pos, t.aux);
            // the output SYNC and the SAMPLE of the step (the last one if there were several; counted)
            j.raw(",\"n_out_sync\":%d,\"n_sample\":%d,\"sample_token\":%" PRId64, s.n_out_sync, s.n_sample, s.sample_token);
            if (s.n_out_sync > 0) {
                j.raw(",\"out_sync_ts\":%.3f,\"out_sync_dur\":%.3f", us(s.out_t0), (s.out_t1 - s.out_t0)/1000.0);
            }
            if (s.n_sample > 0) {
                j.raw(",\"sample_ts\":%.3f,\"sample_dur\":%.3f", us(s.sample_t0), (s.sample_t1 - s.sample_t0)/1000.0);
            }
            j.raw(",\"n_sync\":%d,\"sync_ms\":%.3f,\"n_split\":%d,\"n_predict\":%d", s.n_sync, s.sync_ns/1e6, s.n_split, s.n_predict);
            j.raw(",\"n_experts\":%d,\"n_expert_threads\":%zu,\"n_residency\":%d", n_experts, s.expert_threads.size(), s.n_residency);
            j.raw(",\"n_pieces_prefetch\":%" PRId64 ",\"n_pieces_predict_high\":%" PRId64 ",\"n_pieces_predict_low\":%" PRId64
                  ",\"n_pieces_cached\":%" PRId64 ",\"n_pieces_dup\":%" PRId64 ",\"n_loads\":%d,\"n_loads_cached\":%d",
                    s.n_pieces[0], s.n_pieces[1], s.n_pieces[2], s.n_pieces_cached, s.n_pieces_dup, n_loads, n_loads_cached);
            j.raw(",\"n_copies\":%zu,\"copy_bytes\":%" PRId64 ",\"n_gpu_ops\":%zu", s.copies.size(), copy_bytes, s.gpu.size());
            tail();
        }

        // workers: the expert each thread computes; on worker 0 also the layer's other ops, merged per layer run
        struct item { int64_t t0, t1; int layer, expert; const ggml_trace_event * ev; };
        std::map<int, std::vector<item>> rows;
        for (const auto & e : s.expert_threads) {
            rows[e.tid].push_back({ e.t0_ns, e.t1_ns, ggml_trace_parse_layer(e.name), e.expert, &e });
        }
        for (const auto & o : s.ops) {
            if (o.layer >= 0) {
                rows[o.tid].push_back({ o.t0_ns, o.t1_ns, o.layer, -1, &o });
            }
        }
        for (auto & kv : rows) {
            auto & v = kv.second;
            std::sort(v.begin(), v.end(), [](const item & a, const item & b) { return a.t0 < b.t0; });
            std::vector<item> merged;
            std::vector<int> n_merged;
            for (const item & it : v) {
                if (it.expert < 0 && !merged.empty() && merged.back().expert < 0 && merged.back().layer == it.layer) {
                    merged.back().t1 = std::max(merged.back().t1, it.t1);
                    n_merged.back()++;
                    continue;
                }
                merged.push_back(it);
                n_merged.push_back(1);
            }
            int64_t prev_end = INT64_MIN;
            for (size_t i = 0; i < merged.size(); ++i) {
                item & it = merged[i];
                const int64_t t0 = std::max(it.t0, prev_end);
                if (it.t1 < t0) {
                    continue;
                }
                prev_end = it.t1;
                simple_head(simple_name(it.layer, it.expert), t0, it.t1, S_TID_WORKER0 + kv.first);
                if (it.expert >= 0) {
                    const ggml_trace_event & e = *it.ev;
                    j.raw("\"layer\":%d,\"expert\":%d,\"step\":%d,\"tensor\":", it.layer, it.expert, step);
                    j.str(e.name);
                    j.raw(",\"bytes\":%" PRId64 ",\"stall_ns\":%" PRId64 ",\"n_threads\":%d", e.bytes, e.aux, e.n_parts);
                    auto r = s.residency.find({ e.name, e.expert });
                    if (r != s.residency.end()) {
                        j.raw(",\"resident_frac\":%.4f,\"sample_delay_ns\":%" PRId64, r->second.first, r->second.second);
                    }
                } else {
                    j.raw("\"layer\":%d,\"step\":%d,\"n_ops\":%d", it.layer, step, n_merged[i]);
                }
                tail();
            }
        }

        // GPU: device time per layer run (consecutive ops of a layer less than GGML_TRACE_GPU_GAP_NS apart); the
        // output ops (result_*) as "output"
        {
            auto & v = s.gpu;
            std::sort(v.begin(), v.end(), [](const ggml_trace_event & a, const ggml_trace_event & b) { return a.t0_ns < b.t0_ns; });
            struct run { int64_t t0, t1; int key, n, n_weight; };
            std::vector<run> runs;
            for (const auto & g : v) {
                const int key = strncmp(g.name, "result_", 7) == 0 ? -2 : g.layer;
                if (key == -1) {
                    continue;
                }
                const int w = (g.flags & GGML_TRACE_FLAG_WEIGHT_SRC) != 0;
                // the same layer, and the GPU did not idle in between (a gap would show a busy GPU that waits)
                if (!runs.empty() && runs.back().key == key && g.t0_ns - runs.back().t1 < GGML_TRACE_GPU_GAP_NS) {
                    runs.back().t1 = std::max(runs.back().t1, g.t1_ns);
                    runs.back().n++;
                    runs.back().n_weight += w;
                    continue;
                }
                runs.push_back({ g.t0_ns, g.t1_ns, key, 1, w });
            }
            int64_t prev_end = INT64_MIN;
            for (const run & r : runs) {
                const int64_t t0 = std::max(r.t0, prev_end);
                if (r.t1 < t0) {
                    continue;
                }
                prev_end = r.t1;
                simple_head(r.key == -2 ? std::string("output") : simple_name(r.key, -1), t0, r.t1, S_TID_GPU);
                j.raw("\"layer\":%d,\"step\":%d,\"n_ops\":%d,\"n_weight_ops\":%d", r.key, step, r.n, r.n_weight);
                tail();
            }
        }

        // copies between host and device
        for (const auto & c : s.copies) {
            const bool xfer = c.type == GGML_TRACE_XFER;
            const bool h2d  = xfer ? c.detail == GGML_TRACE_INPUT : c.backend != 0;
            const bool async = !xfer && c.detail == GGML_TRACE_COPY_ASYNC;
            const int layer = xfer ? ggml_trace_parse_layer(c.name) : c.layer;
            const std::string name = layer >= 0 ? simple_name(layer, -1) : std::string(h2d ? "input" : "output");
            const int64_t t1 = c.t1_ns > 0 ? c.t1_ns : c.t0_ns;
            s_newest = std::max(s_newest, c.t0_ns);
            simple_head(name, c.t0_ns, t1, S_TID_COPY0 + s_copy_lanes.place(c.t0_ns, t1));
            j.raw("\"layer\":%d,\"step\":%d,\"direction\":\"%s\",\"bytes\":%" PRId64 ",\"kind\":\"%s\",\"async\":%s,\"weight\":%s,\"tensor\":",
                    layer, step, h2d ? "h2d" : "d2h", c.bytes, xfer ? "xfer" : "copy", async ? "true" : "false",
                    (c.flags & GGML_TRACE_FLAG_WEIGHT_SRC) ? "true" : "false");
            j.str(c.name);
            j.raw(",\"backend\":");
            j.str(ggml_trace_backend_name(c.backend).c_str());
            tail();
        }
    }

    void simple_write_load(const load_key & key, simple_load & l, const std::map<int, std::pair<int64_t, int64_t>> & token_span) {
        if (l.n_cached == l.n_pieces) {
            return; // no read at all: counted in the token's n_loads_cached
        }
        int64_t t1 = std::max(l.t_res, l.t_issue_end);
        if (l.unresolved) {
            // still not resident when the trace stopped: up to the end of its step, or the end of the issue if the
            // step was never traced (the guesses after the last traced token)
            auto it = token_span.find(std::get<0>(key));
            t1 = it != token_span.end() ? std::min(t1, std::max(it->second.second, l.t_issue_end)) : l.t_issue_end;
        }
        auto cit = s_computed.find(std::get<0>(key));
        const bool computed = cit != s_computed.end() && cit->second.count({ std::get<1>(key), std::get<2>(key) }) > 0;
        s_newest = std::max(s_newest, l.t0);
        simple_head(simple_name(std::get<1>(key), std::get<2>(key)), l.t0, t1, S_TID_LOAD0 + s_load_lanes.place(l.t0, t1));
        std::string src;
        for (int i = 0; i < 3; ++i) {
            if (l.sources & (1u << i)) {
                src += src.empty() ? "" : ",";
                src += ggml_trace_source_name(i);
            }
        }
        j.raw("\"layer\":%d,\"expert\":%d,\"step\":%d,\"pieces\":%d,\"cached_pieces\":%d,\"duplicate_pieces\":%d,\"bytes\":%" PRId64
              ",\"tensors\":%zu,\"sources\":\"%s\",\"unresolved\":%s,\"computed\":%s",
                std::get<1>(key), std::get<2>(key), std::get<0>(key), l.n_pieces, l.n_cached, l.n_dup, l.bytes,
                l.tensors.size(), src.c_str(), l.unresolved ? "true" : "false", computed ? "true" : "false");
        if (l.t_res > 0) {
            j.raw(",\"t_resident_ns\":%" PRId64, l.t_res - st->t_base);
        }
        tail();
    }

    std::map<int, std::pair<int64_t, int64_t>> s_token_span; // written steps: TOKEN t0, t1
    std::map<int, std::set<std::pair<int, int>>> s_computed; // written steps: (layer, expert) computed
    uint64_t s_late = 0; // records of a step that had already been written (dropped)

    void simple_flush(bool final) {
        for (auto it = s_steps.begin(); it != s_steps.end(); ) {
            simple_step & s = it->second;
            const bool ready = final || (s.have_token && ++s.hold > 2);
            if (!ready) {
                ++it;
                continue;
            }
            if (s.have_token) {
                s_token_span[it->first] = { s.token.t0_ns, s.token.t1_ns };
                s_newest = std::max(s_newest, s.token.t1_ns);
            }
            simple_write_step(it->first, s);
            s_done.insert(it->first);
            it = s_steps.erase(it);
        }
        for (auto it = s_loads.begin(); it != s_loads.end(); ) {
            const bool step_done = s_done.count(std::get<0>(it->first)) > 0;
            if (final || (step_done && it->second.pending.empty())) {
                simple_write_load(it->first, it->second, s_token_span);
                it = s_loads.erase(it);
            } else {
                ++it;
            }
        }
        if (final) {
            s_cookie_loads.clear();
        }
    }

    void on_event(const ggml_trace_event & ev) {
        if (st->params.detail == 1) {
            simple_on_event(ev);
            return;
        }
        switch (ev.type) {
            case GGML_TRACE_TOKEN:
                {
                    head("TOKEN", "token", 'X', ev.t0_ns, ev.t1_ns, PID_HOST, TID_SCHED);
                    j.raw("\"step\":%d,\"token\":%d,\"pos\":%d,\"token_out\":%" PRId64, ev.step, ev.token, ev.pos, ev.aux);
                    tail();
                } break;
            case GGML_TRACE_SYNC:
                {
                    const std::string be = ggml_trace_backend_name(ev.backend);
                    head(ev.detail == GGML_TRACE_OUTPUT ? "SYNC output" : "SYNC", "sync", 'X', ev.t0_ns, ev.t1_ns, PID_HOST, ggml_trace_host_tid(ev.tid));
                    j.raw("\"step\":%d,\"backend\":", ev.step);
                    j.str(be.c_str());
                    j.raw(",\"output\":%s", ev.detail == GGML_TRACE_OUTPUT ? "true" : "false");
                    tail();
                } break;
            case GGML_TRACE_SAMPLE:
                {
                    head("SAMPLE", "sample", 'X', ev.t0_ns, ev.t1_ns, PID_HOST, TID_SAMPLER);
                    j.raw("\"step\":%d,\"token_out\":%" PRId64, ev.step, ev.aux);
                    tail();
                } break;
            case GGML_TRACE_XFER:
                {
                    head(ev.name, "xfer", ev.t1_ns ? 'X' : 'i', ev.t0_ns, ev.t1_ns, PID_HOST, ggml_trace_host_tid(ev.tid));
                    j.raw("\"step\":%d,\"direction\":\"%s\",\"bytes\":%" PRId64 ",\"name\":", ev.step, ev.detail == GGML_TRACE_INPUT ? "input" : "output", ev.bytes);
                    j.str(ev.name);
                    tail();
                } break;
            case GGML_TRACE_SPLIT:
                {
                    const std::string be = ggml_trace_backend_name(ev.backend);
                    const int tid = ggml_trace_host_tid(ev.tid);
                    bool inferred;
                    const int layer = layer_of(ev, inferred);
                    head(("SPLIT " + be).c_str(), "split", 'X', ev.t0_ns, ev.t1_ns, PID_HOST, tid);
                    j.raw("\"step\":%d,\"backend\":", ev.step);
                    j.str(be.c_str());
                    j.raw(",\"n_nodes\":%" PRId64 ",\"n_inputs\":%" PRId64 ",\"name\":", ev.aux, ev.bytes);
                    j.str(ev.name);
                    arg_layer(layer, inferred);
                    tail();
                } break;
            case GGML_TRACE_COPY:
                {
                    const std::string be = ggml_trace_backend_name(ev.backend);
                    const int tid = ggml_trace_host_tid(ev.tid);
                    bool inferred;
                    const int layer = layer_of(ev, inferred);
                    head((std::string("COPY ") + ev.name).c_str(), "copy", ev.t1_ns ? 'X' : 'i', ev.t0_ns, ev.t1_ns, PID_HOST, tid);
                    j.raw("\"step\":%d,\"backend\":", ev.step);
                    j.str(be.c_str());
                    j.raw(",\"bytes\":%" PRId64 ",\"async\":%s,\"weight\":%s,\"name\":", ev.bytes, ev.detail == GGML_TRACE_COPY_ASYNC ? "true" : "false",
                            (ev.flags & GGML_TRACE_FLAG_WEIGHT_SRC) ? "true" : "false");
                    j.str(ev.name);
                    arg_layer(layer, inferred);
                    tail();
                } break;
            case GGML_TRACE_OP_CPU:
                {
                    const int tid = ggml_trace_host_tid(ev.tid);
                    bool inferred;
                    const int layer = layer_of(ev, inferred);
                    head(ev.name, "op", 'X', ev.t0_ns, ev.t1_ns, PID_HOST, tid);
                    j.raw("\"step\":%d,\"op\":\"%s\",\"n_fused\":%d,\"name\":", ev.step, ggml_op_name((enum ggml_op) ev.op), ev.detail);
                    j.str(ev.name);
                    arg_layer(layer, inferred);
                    tail();
                } break;
            case GGML_TRACE_OP_GPU:
                {
                    const int pid = PID_DEVICE0 + ev.backend - 1;
                    bool inferred;
                    const int layer = layer_of(ev, inferred);
                    head(ev.name, "op", 'X', ev.t0_ns, ev.t1_ns, pid, 1);
                    j.raw("\"step\":%d,\"op\":\"%s\",\"n_fused\":%d,\"stream\":%d,\"weight\":%s,\"name\":", ev.step,
                            ggml_op_name((enum ggml_op) ev.op), ev.detail, ev.piece, (ev.flags & GGML_TRACE_FLAG_WEIGHT_SRC) ? "true" : "false");
                    j.str(ev.name);
                    arg_layer(layer, inferred);
                    tail();
                    head(ev.name, "launch", 'X', ev.aux, ev.aux2, pid, 2);
                    j.raw("\"step\":%d,\"op\":\"%s\",\"name\":", ev.step, ggml_op_name((enum ggml_op) ev.op));
                    j.str(ev.name);
                    arg_layer(layer, inferred);
                    tail();
                } break;
            case GGML_TRACE_EXPERT_THREAD:    on_expert_thread(ev);    break;
            case GGML_TRACE_EXPERT_RESIDENCY: on_expert_residency(ev); break;
            case GGML_TRACE_IO_ISSUE:         on_io_issue(ev);         break;
            case GGML_TRACE_IO_RESIDENT:      on_io_resident(ev);      break;
            case GGML_TRACE_PREDICT:
                {
                    char label[64];
                    snprintf(label, sizeof(label), "PREDICT L%d", ev.layer);
                    head(label, "predict", 'X', ev.t0_ns, ev.t1_ns, PID_HOST, ggml_trace_host_tid(ev.tid));
                    j.raw("\"step\":%d,\"layer\":%d,\"n_guesses\":%" PRId64 ",\"layer_next\":%" PRId64 ",\"n_selected\":%" PRId64,
                            ev.step, ev.layer, ev.bytes, ev.aux, ev.aux2);
                    tail();
                } break;
            case GGML_TRACE_META:
                {
                    head("poll interval", "meta", 'i', ev.t0_ns, 0, PID_HOST, TID_POLLER);
                    j.raw("\"step\":%d,\"poll_us\":%" PRId64 ",\"poll_took_ns\":%" PRId64, ev.step, ev.aux, ev.aux2);
                    tail();
                } break;
            default:
                break;
        }
    }

    // serialize the new records of every chunk, recycle the full ones
    void pass(bool final) {
        std::vector<ggml_trace_chunk *> chunks;
        {
            std::lock_guard<std::mutex> lock(st->pool_mtx);
            chunks = st->chunks;
        }
        for (ggml_trace_chunk * c : chunks) {
            const uint32_t n = c->count.load(std::memory_order_acquire);
            for (uint32_t i = c->written; i < n; ++i) {
                on_event(c->events[i]);
            }
            n_records += n - c->written;
            c->written = n;
            if (n == GGML_TRACE_CHUNK_EVENTS) {
                std::lock_guard<std::mutex> lock(st->pool_mtx);
                c->count.store(0, std::memory_order_relaxed);
                c->written = 0;
                st->free_chunks.push_back(c);
            }
            if (j.s.size() > (1u << 20)) {
                flush();
            }
        }
        if (st->params.detail == 1) {
            simple_flush(final);
        }
        flush_experts(final);
        prune();
        if (final) {
            for (auto it = slices.begin(); it != slices.end(); ) {
                const std::string key = it->first;
                ggml_trace_slice_agg & a = it->second;
                ++it;
                write_slice(key, a, true);
            }
        }
        flush();
    }

    // the records of all threads of one moment are serialized within a few writer passes: resolutions older than
    // 10 s have met their issues, and cookies whose slice is gone point nowhere
    // lane intervals: a span still held (a slice, expert or load waiting for its end, a step not written yet)
    // may start as early as its t0, so no interval that ends at or after the earliest such t0 may go
    void prune() {
        int64_t held = INT64_MAX;
        for (const auto & kv : slices) {
            held = std::min(held, kv.second.t0);
        }
        for (const auto & kv : experts) {
            held = std::min(held, kv.second.t0);
        }
        for (auto & kv : lanes) {
            kv.second.prune(std::min<int64_t>(t_span_newest - 10000000000LL, held));
        }
        int64_t s_held = INT64_MAX;
        for (const auto & kv : s_loads) {
            s_held = std::min(s_held, kv.second.t0);
        }
        for (const auto & kv : s_steps) {
            for (const auto & c : kv.second.copies) {
                s_held = std::min(s_held, c.t0_ns);
            }
        }
        s_load_lanes.prune(std::min<int64_t>(s_newest - 10000000000LL, s_held));
        // the computed sets are needed until the loads of their step are written
        const int min_load_step = s_loads.empty() ? INT_MAX : std::get<0>(s_loads.begin()->first);
        while (!s_computed.empty() && s_computed.begin()->first < min_load_step) {
            s_computed.erase(s_computed.begin());
        }
        s_copy_lanes.prune(std::min<int64_t>(s_newest - 10000000000LL, s_held));
        for (auto it = s_cookie_loads.begin(); it != s_cookie_loads.end(); ) {
            bool live = false;
            for (const auto & k : it->second) {
                live = live || s_loads.count(k) > 0;
            }
            it = live ? std::next(it) : s_cookie_loads.erase(it);
        }
        const int64_t horizon = t_newest - 10000000000LL;
        for (auto it = resolved.begin(); it != resolved.end(); ) {
            it = it->second.t < horizon ? resolved.erase(it) : std::next(it);
        }
        for (auto it = cookie_slices.begin(); it != cookie_slices.end(); ) {
            auto & keys = it->second;
            keys.erase(std::remove_if(keys.begin(), keys.end(), [this](const std::string & k) { return slices.count(k) == 0; }), keys.end());
            it = keys.empty() ? cookie_slices.erase(it) : std::next(it);
        }
    }

    void flush() {
        if (!j.s.empty()) {
            fwrite(j.s.data(), 1, j.s.size(), st->f);
            j.s.clear();
        }
    }
};

static void ggml_trace_writer_loop(ggml_trace_state * st) {
    ggml_trace_writer w(st);
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(st->writer_mtx);
            st->writer_cv.wait_for(lock, std::chrono::milliseconds(GGML_TRACE_WRITER_MS), [st] {
                return st->stop_writer.load(std::memory_order_acquire);
            });
        }
        if (st->stop_writer.load(std::memory_order_acquire)) {
            break;
        }
        w.pass(false);
        fflush(st->f);
    }
    // the poller has been joined: all records are in the chunks
    w.pass(true);

    // close the array with an event that carries no trailing comma
    w.j.raw("{\"name\":\"trace_end\",\"cat\":\"meta\",\"ph\":\"i\",\"s\":\"g\",\"ts\":%.3f,\"pid\":%d,\"tid\":%d,\"args\":{\"events\":%" PRIu64 "}}\n]\n",
            w.us(ggml_trace_now_ns()), PID_HOST, TID_SCHED, w.n_written);
    w.flush();

    // the counters only known now, into the fixed-width fields of the metadata record
    auto patch = [st](size_t pos, uint64_t v) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%10" PRIu64, (uint64_t) (v % 10000000000ull));
        fseek(st->f, st->meta_offset + (long) pos, SEEK_SET);
        fwrite(buf, 1, 10, st->f);
    };
    st->emitted = w.n_records + st->dropped.load();
    patch(st->meta_pos_emitted, st->emitted);
    patch(st->meta_pos_dropped, st->dropped.load());
    patch(st->meta_pos_ring,    st->ring_drops.load());
    patch(st->meta_pos_adapt,   st->poll_adaptations.load());
    patch(st->meta_pos_written, w.n_written);
    patch(st->meta_pos_tokens,  (uint64_t) st->n_traced);
    patch(st->meta_pos_late,    w.s_late);
    fflush(st->f);
}

//
// start / stop
//

static void ggml_trace_write_meta(ggml_trace_state * st) {
    ggml_trace_json j;
    j.raw("[\n");
    st->meta_offset = (long) j.s.size();

    j.raw("{\"name\":\"meta\",\"cat\":\"meta\",\"ph\":\"i\",\"s\":\"g\",\"ts\":0,\"pid\":%d,\"tid\":%d,\"args\":{", PID_HOST, TID_SCHED);
    if (!st->meta.empty()) {
        j.raw("%s,", st->meta.c_str());
    }
    j.raw("\"config\":{\"path\":");
    j.str(st->path.c_str());
    j.raw(",\"tokens\":%d,\"skip\":%d,\"max_batch\":%d,\"detail\":%d,\"poll_us\":%d,\"buffer_mb\":%d}",
            st->params.n_tokens, st->params.n_skip, st->params.max_batch, st->params.detail, st->params.poll_us, st->params.buffer_mb);
    j.raw(",\"t_base_ns\":%" PRId64, st->t_base);
#if defined(GGML_TRACE_LINUX)
    j.raw(",\"linux\":true,\"residency_valid\":%s", st->params.residency_valid ? "true" : "false");
#else
    j.raw(",\"linux\":false,\"residency_valid\":false");
#endif
    j.raw(",\"cuda_graphs_disabled_while_traced\":true");
    j.raw(",\"stall_metric\":\"per thread wall - CLOCK_THREAD_CPUTIME_ID: page fault and I/O wait plus preemption (the poller and writer share hyperthreads with the compute threads)\"");
    j.raw(",\"gpu_time_note\":\"device times are anchored by one synchronized event per flush: constant shift of about 5-20 us\"");
    j.raw(",\"ts_unit\":\"us since t_base_ns\"");
    j.raw(",\"layout\":\"%s\"", st->params.detail == 1 ? "simple" : "detailed");
    j.raw(",\"io_notes\":\"cached = no read_bytes caused by the hint and the tail page resident; the windowed expert prefetch (more used experts than GGML_CPU_MOE_PREFETCH_WINDOW_MB, prompt batches) is not traced\"");
    // fixed width, right-aligned with spaces (JSON allows no leading zeros); valid before the rewrite too
    const char * tags = ",\"records_emitted\":         0,\"dropped\":         0,\"ring_drops\":         0,\"poll_adaptations\":         0,\"events_written\":         0,\"tokens_traced\":         0,\"late_records\":         0}},\n";
    const size_t base = j.s.size() - st->meta_offset;
    j.raw("%s", tags);
    const std::string t(tags);
    st->meta_pos_emitted = base + t.find("\"records_emitted\":")  + strlen("\"records_emitted\":");
    st->meta_pos_dropped = base + t.find("\"dropped\":")          + strlen("\"dropped\":");
    st->meta_pos_ring    = base + t.find("\"ring_drops\":")       + strlen("\"ring_drops\":");
    st->meta_pos_adapt   = base + t.find("\"poll_adaptations\":") + strlen("\"poll_adaptations\":");
    st->meta_pos_written = base + t.find("\"events_written\":")   + strlen("\"events_written\":");
    st->meta_pos_tokens  = base + t.find("\"tokens_traced\":")    + strlen("\"tokens_traced\":");
    st->meta_pos_late    = base + t.find("\"late_records\":")     + strlen("\"late_records\":");
    fwrite(j.s.data(), 1, j.s.size(), st->f);
    fflush(st->f);
}

bool ggml_trace_start(const struct ggml_trace_params * params) {
    std::lock_guard<std::mutex> lock(g_ctl_mtx);
    if (g_refcount > 0) {
        g_refcount++;
        return true;
    }
    if (params == nullptr || params->path == nullptr) {
        return false;
    }

    static uint32_t gen = 0;

    ggml_trace_state * st = new ggml_trace_state;
    st->path   = params->path;
    st->params = *params;
    st->params.path = st->path.c_str();
    st->meta   = params->meta ? params->meta : "";
    st->params.meta = nullptr;
    st->params.detail    = std::max(1, std::min(2, params->detail));
    st->params.poll_us   = std::max(50, params->poll_us);
    st->params.buffer_mb = std::max(1, params->buffer_mb);
    st->params.max_batch = std::max(1, params->max_batch);
    st->params.n_skip    = std::max(0, params->n_skip);
    st->params.n_tokens  = std::max(0, params->n_tokens);
    st->max_chunks = (size_t) st->params.buffer_mb*(1024*1024)/GGML_TRACE_CHUNK_BYTES;
    st->gen = ++gen;
    st->t_base = ggml_trace_now_ns();
    for (uint32_t i = 0; i < GGML_TRACE_RING_SIZE; ++i) {
        st->ring[i].seq.store(i, std::memory_order_relaxed);
    }

    st->f = fopen(st->path.c_str(), "wb");
    if (st->f == nullptr) {
        GGML_LOG_ERROR("%s: cannot open %s\n", __func__, st->path.c_str());
        delete st;
        return false;
    }
    ggml_trace_write_meta(st);

#if defined(GGML_TRACE_LINUX)
    st->efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    st->poller = std::thread(ggml_trace_poller_loop, st);
#endif
    st->writer = std::thread(ggml_trace_writer_loop, st);

    g_step.store(-1, std::memory_order_relaxed);
    g_last_step.store(-1, std::memory_order_relaxed);
    g_trace = st;
    g_refcount = 1;
    g_state.store(TRACE_IDLE, std::memory_order_seq_cst);
    // at warning level: servers do not show info messages by default, and where the file goes matters
    GGML_LOG_WARN("%s: tracing to %s (tokens %d, skip %d, detail %d, poll %d us, buffer %d MB)\n", __func__,
            st->path.c_str(), st->params.n_tokens, st->params.n_skip, st->params.detail, st->params.poll_us, st->params.buffer_mb);
    return true;
}

// closes the file; the state stays allocated until the last ggml_trace_stop. g_ctl_mtx held
static void ggml_trace_finish_locked(void) {
    ggml_trace_state * st = g_trace;
    if (st == nullptr || g_state.load(std::memory_order_relaxed) == TRACE_OFF) {
        return;
    }
    if (g_state.load(std::memory_order_relaxed) != TRACE_IDLE) {
        ggml_trace_batch_end_locked(st);
    }
    ggml_trace_flush_token(st);
    ggml_trace_run_flush();

    // no new emitters after this; wait for the ones in flight (this thread holds no guard here)
    g_state.store(TRACE_OFF, std::memory_order_seq_cst);
    while (ggml_trace_any_inflight()) {
        std::this_thread::yield();
    }

    st->stop_threads.store(true, std::memory_order_release);
#if defined(GGML_TRACE_LINUX)
    if (st->poller.joinable()) {
        const uint64_t one = 1;
        (void) !write(st->efd, &one, sizeof(one));
        st->poller.join();
    }
#endif
    {
        std::lock_guard<std::mutex> lock(st->writer_mtx);
        st->stop_writer.store(true, std::memory_order_release);
    }
    st->writer_cv.notify_all();
    if (st->writer.joinable()) {
        st->writer.join();
    }
    fclose(st->f);
    st->f = nullptr;
#if defined(GGML_TRACE_LINUX)
    if (st->efd >= 0) {
        close(st->efd);
        st->efd = -1;
    }
#endif
    GGML_LOG_WARN("%s: trace closed: %s, %d tokens, %" PRIu64 " records emitted, %" PRIu64 " dropped, %" PRIu64 " residency requests dropped\n",
            __func__, st->path.c_str(), st->n_traced, st->emitted, st->dropped.load(), st->ring_drops.load());
}

void ggml_trace_stop(void) {
    std::lock_guard<std::mutex> lock(g_ctl_mtx);
    if (g_refcount == 0) {
        return;
    }
    if (--g_refcount > 0) {
        return;
    }
    ggml_trace_finish_locked();
    ggml_trace_state * st = g_trace;
    g_trace = nullptr;
    for (ggml_trace_chunk * c : st->chunks) {
        delete c;
    }
    delete st;
}
