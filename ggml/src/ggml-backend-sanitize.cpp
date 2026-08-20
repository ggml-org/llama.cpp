#include "ggml-backend-sanitize.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

int ggml_san_level(void) {
    static const int level = []() {
        const char * env = getenv("GGML_SCHED_SANITIZE");
        return env ? atoi(env) : 0;
    }();
    return level;
}

namespace {

constexpr int    SAN_MAX_ACTORS      = 64;
constexpr size_t SAN_MAX_RANGES      = 65536;

using vclock = std::vector<uint64_t>;

struct san_access {
    int          actor  = -1;
    uint64_t     clock  = 0;
    int          split  = -1;
    const char * what   = ""; // always a string literal
    char         tensor[GGML_MAX_NAME] = { 0 }; // copied: source tensor is recycled per graph
};

struct san_entry {
    size_t                  end;
    san_access              last_write;
    std::vector<san_access> reads; // at most one per actor
};

struct san_state {
    std::mutex mutex;

    std::unordered_map<const void *, int> actors;
    std::vector<std::string>              actor_names;
    std::unordered_map<std::string, int>  name_counts;
    std::vector<vclock>                   vc; // vc[a][b] = a's knowledge of b's clock

    std::unordered_map<ggml_backend_buffer_t, std::map<size_t, san_entry>> mem;

    std::unordered_map<const void *, vclock> ev_vc;
    std::unordered_map<const void *, int>    events;
};

san_state & state() {
    static san_state * s = []() {
        san_state * st = new san_state();
        st->actor_names.push_back("HOST");
        st->vc.emplace_back(SAN_MAX_ACTORS, 0);
        return st;
    }();
    return *s;
}

// every helper below assumes the caller holds state().mutex

int actor_id(san_state & s, ggml_backend_t backend) {
    if (backend == nullptr) {
        return 0;
    }

    auto it = s.actors.find(backend);
    if (it != s.actors.end()) {
        return it->second;
    }

    // two backends can share a device and therefore a name, disambiguate
    std::string base = ggml_backend_name(backend);
    const int   idx  = s.name_counts[base]++;

    const int id = (int) s.actor_names.size();
    GGML_ASSERT(id < SAN_MAX_ACTORS);
    s.actor_names.push_back(idx == 0 ? base : base + "#" + std::to_string(idx));
    s.vc.emplace_back(SAN_MAX_ACTORS, 0);
    s.actors[backend] = id;
    return id;
}

const char * actor_name(san_state & s, int id) {
    return id >= 0 && id < (int) s.actor_names.size() ? s.actor_names[id].c_str() : "?";
}

void join(vclock & dst, const vclock & src) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = std::max(dst[i], src[i]);
    }
}

uint64_t tick(san_state & s, int a) {
    return ++s.vc[a][a];
}

uint64_t issue(san_state & s, int a) {
    join(s.vc[a], s.vc[0]);
    return tick(s, a);
}

void maybe_flush(san_state & s) {
    if (s.mem.empty()) {
        return;
    }

    // once the host knows every actor's current clock, the issue rule guarantees that any
    // future operation joins vc[0] first, so the shadow state can no longer report a race
    for (size_t b = 0; b < s.vc.size(); b++) {
        if (s.vc[0][b] < s.vc[b][b]) {
            return;
        }
    }

    s.mem.clear();
}

bool is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

struct mem_range {
    ggml_backend_buffer_t buf;
    size_t                off;
    size_t                len;
};

bool resolve(const ggml_tensor * t, size_t offset, size_t size, mem_range & out) {
    if (t == nullptr || t->data == nullptr) {
        return false;
    }

    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (buf == nullptr) {
        return false;
    }

    // weights are written once at load and never recycled by the allocator
    if (ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }

    void * base = ggml_backend_buffer_get_base(buf);
    if (base == nullptr || size == 0) {
        return false;
    }

    out.buf = buf;
    out.off = (size_t) ((const char *) t->data - (const char *) base) + offset;
    out.len = size;
    return true;
}

thread_local int          tl_split    = -1;
thread_local int          tl_split_actor = -1;

void report(san_state & s, const mem_range & mr, const san_access & cur, const san_access & prev, const char * kind) {
    fprintf(stderr, "\nggml-sched-sanitize: RACE (%s) on %s[%zu, %zu)\n",
            kind, ggml_backend_buffer_name(mr.buf), mr.off, mr.off + mr.len);
    fprintf(stderr, "ggml-sched-sanitize:   %-5s %-12s @%-4llu split %-4d %s (%s)\n",
            strcmp(kind, "write-after-read") == 0 ? "read" : "write",
            actor_name(s, prev.actor), (unsigned long long) prev.clock, prev.split, prev.tensor, prev.what);
    fprintf(stderr, "ggml-sched-sanitize:   %-5s %-12s @%-4llu split %-4d %s (%s)\n",
            strcmp(kind, "read-after-write") == 0 ? "read" : "write",
            actor_name(s, cur.actor), (unsigned long long) cur.clock, cur.split, cur.tensor, cur.what);
    fprintf(stderr, "ggml-sched-sanitize:   no happens-before edge: %s knows %s@%llu, needs >=%llu\n\n",
            actor_name(s, cur.actor), actor_name(s, prev.actor),
            (unsigned long long) s.vc[cur.actor][prev.actor], (unsigned long long) prev.clock);

    GGML_ABORT("ggml-sched-sanitize: race detected");
}

// make sure a range boundary exists at pos, so [off,off+len) lands on whole entries
void split_at(std::map<size_t, san_entry> & m, size_t pos) {
    auto it = m.upper_bound(pos);
    if (it == m.begin()) {
        return;
    }
    --it;
    if (it->first < pos && it->second.end > pos) {
        san_entry tail = it->second;
        it->second.end = pos;
        m.emplace(pos, std::move(tail));
    }
}

void access_range(san_state & s, const mem_range & mr, bool write, const san_access & info) {
    auto & m = s.mem[mr.buf];
    const int a = info.actor;

    // the shadow state is bounded by the work issued before the host observes every actor,
    // not by run length. reaching this means maybe_flush stopped clearing it -
    // most likely an operation was enqueued without going through issue(), which breaks
    // the argument maybe_flush relies on. that is a sanitizer bug, not a memory limit.
    GGML_ASSERT(m.size() <= SAN_MAX_RANGES && "shadow state is not being flushed - see issue()/maybe_flush()");

    const size_t begin = mr.off;
    const size_t end   = mr.off + mr.len;

    split_at(m, begin);
    split_at(m, end);

    // fill gaps so the whole span is covered by entries
    size_t cur = begin;
    while (cur < end) {
        auto it = m.lower_bound(cur);
        if (it == m.end() || it->first >= end) {
            san_entry e;
            e.end = end;
            m.emplace(cur, std::move(e));
            break;
        }
        if (it->first > cur) {
            san_entry e;
            e.end = it->first;
            m.emplace(cur, std::move(e));
        }
        cur = it->second.end;
    }

    for (auto it = m.lower_bound(begin); it != m.end() && it->first < end; ++it) {
        san_entry & e = it->second;

        if (e.last_write.actor >= 0 && e.last_write.actor != a) {
            if (e.last_write.clock > s.vc[a][e.last_write.actor]) {
                report(s, mr, info, e.last_write, write ? "write-after-write" : "read-after-write");
            }
        }

        if (write) {
            for (const san_access & r : e.reads) {
                if (r.actor != a && r.clock > s.vc[a][r.actor]) {
                    report(s, mr, info, r, "write-after-read");
                }
            }
        }

        if (write) {
            e.last_write = info;
            e.reads.clear();
        } else {
            bool found = false;
            for (san_access & r : e.reads) {
                if (r.actor == a) {
                    r     = info;
                    found = true;
                    break;
                }
            }
            if (!found) {
                e.reads.push_back(info);
            }
        }
    }

}

// returns false if the tensor is not in tracked memory
bool touch(san_state & s, int a, uint64_t c, const ggml_tensor * t,
           size_t offset, size_t size, bool write, const char * what) {
    mem_range mr;
    if (!resolve(t, offset, size, mr)) {
        return false;
    }

    san_access info;
    info.actor = a;
    info.clock = c;
    info.what  = what;
    info.split = tl_split;
    snprintf(info.tensor, sizeof(info.tensor), "%s", t->name);

    if (ggml_san_level() >= 2) {
        GGML_LOG_DEBUG("san [split %3d %-10s] %-10s %-5s %s[%zu+%zu] %s (%s)\n",
                       tl_split, tl_split_actor >= 0 ? actor_name(s, tl_split_actor) : "-", actor_name(s, a),
                       write ? "write" : "read",
                       ggml_backend_buffer_name(mr.buf), mr.off, mr.len, t->name, what);
    }

    access_range(s, mr, write, info);
    return true;
}

void trace(san_state & s, int a, const char * fmt, ...) {
    if (ggml_san_level() < 2) {
        return;
    }

    char body[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);

    GGML_LOG_DEBUG("san [split %3d %-10s] %-10s %s",
                   tl_split, tl_split_actor >= 0 ? actor_name(s, tl_split_actor) : "-", actor_name(s, a), body);
}

} // namespace

void ggml_san_sync(ggml_backend_t backend) {
    if (ggml_san_level() == 0) {
        return;
    }
    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);

    const int a = actor_id(s, backend);
    join(s.vc[0], s.vc[a]); // host now knows everything this backend knew
    trace(s, a, "EDGE  synchronize -> HOST\n");
    maybe_flush(s);
}

void ggml_san_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    if (ggml_san_level() == 0) {
        return;
    }
    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);

    const int a = actor_id(s, backend);
    s.ev_vc[event] = s.vc[a];
    trace(s, a, "EDGE  event_record  ev=%d\n", (int) s.events.emplace(event, (int) s.events.size()).first->second);
}

void ggml_san_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    if (ggml_san_level() == 0) {
        return;
    }
    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);

    const int a  = actor_id(s, backend);
    auto      it = s.ev_vc.find(event);
    if (it != s.ev_vc.end()) {
        join(s.vc[a], it->second);
    }
    trace(s, a, "EDGE  event_wait    ev=%d\n", (int) s.events.emplace(event, (int) s.events.size()).first->second);
}

void ggml_san_event_sync(ggml_backend_event_t event) {
    if (ggml_san_level() == 0) {
        return;
    }
    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);

    auto it = s.ev_vc.find(event);
    if (it != s.ev_vc.end()) {
        join(s.vc[0], it->second);
    }
    trace(s, 0, "EDGE  event_sync    ev=%d\n", (int) s.events.emplace(event, (int) s.events.size()).first->second);
    maybe_flush(s);
}

void ggml_san_compute(ggml_backend_t backend, const ggml_cgraph * cgraph) {
    if (ggml_san_level() == 0 || cgraph == nullptr) {
        return;
    }
    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);

    const int a = actor_id(s, backend);

    // a backend with no synchronize runs inline on the calling thread, so it both
    // inherits and publishes the host's knowledge - without this the CPU backend's
    // writes are never ordered against anything and no race can be detected
    const bool synchronous = backend->iface.synchronize == nullptr;

    const uint64_t c = issue(s, a);

    size_t n_read = 0;
    size_t n_write = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (is_view_op(node->op)) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->op == GGML_OP_CPY && j == 1) {
                continue;
            }
            if (node->src[j] && touch(s, a, c, node->src[j], 0, ggml_nbytes(node->src[j]), false, "compute")) {
                n_read++;
            }
        }
        if (touch(s, a, c, node, 0, ggml_nbytes(node), true, "compute")) {
            n_write++;
        }
    }

    if (synchronous) {
        join(s.vc[0], s.vc[a]);
        maybe_flush(s);
    }

    trace(s, a, "compute       %d nodes, r=%zu w=%zu ranges\n", cgraph->n_nodes, n_read, n_write);
}

void ggml_san_access(ggml_backend_t backend, const ggml_tensor * tensor,
                     size_t offset, size_t size, bool write, const char * what) {
    if (ggml_san_level() == 0) {
        return;
    }
    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);

    const int      a = actor_id(s, backend);
    const uint64_t c = issue(s, a);
    touch(s, a, c, tensor, offset, size, write, what);

    // a synchronous access has completed by the time the call returns
    if (backend == nullptr || backend->iface.synchronize == nullptr) {
        join(s.vc[0], s.vc[a]);
        maybe_flush(s);
    }
}

void ggml_san_cpy_async(ggml_backend_t src_be, ggml_backend_t dst_be,
                        const ggml_tensor * src, const ggml_tensor * dst) {
    if (ggml_san_level() == 0) {
        return;
    }
    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);

    const int a_src = actor_id(s, src_be);
    const int a_dst = actor_id(s, dst_be);

    // TODO: clarify in ggml
    // which stream actually performs the read of src is backend specific. host memory has
    // no queue of its own, so a host->device transfer runs on dst's queue - that is what
    // makes the source racy against a later host write. a device->device transfer is
    // issued on the src stream (see "copy on src stream" in ggml-cuda), where src's own
    // later work is already ordered against it.
    ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
    const int read_actor = (src_buf && ggml_backend_buffer_is_host(src_buf)) ? a_dst : a_src;

    uint64_t read_clock;
    uint64_t write_clock;

    if (read_actor == a_dst) {
        // the dst queue performs both sides of the copy after src's pending work
        join(s.vc[a_dst], s.vc[a_src]);
        read_clock = write_clock = issue(s, a_dst);
    } else {
        // issue the src read first, then carry its completion into the dst write
        read_clock = issue(s, read_actor);
        join(s.vc[a_dst], s.vc[read_actor]);
        write_clock = issue(s, a_dst);
    }

    touch(s, read_actor, read_clock,  src, 0, ggml_nbytes(src), false, "cpy_async src");
    touch(s, a_dst,      write_clock, dst, 0, ggml_nbytes(dst), true,  "cpy_async dst");

    trace(s, a_dst, "cpy_async     %s@%s -> %s\n", src->name, actor_name(s, read_actor), dst->name);
}

void ggml_san_buffer_free(ggml_backend_buffer_t buffer) {
    if (ggml_san_level() == 0) {
        return;
    }
    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);

    s.mem.erase(buffer);
}

void ggml_san_split(int split_id, ggml_backend_t backend, int n_inputs) {
    if (ggml_san_level() == 0) {
        return;
    }
    tl_split = split_id;

    if (split_id < 0) {
        tl_split_actor = -1;
        return;
    }

    san_state & s = state();
    std::lock_guard<std::mutex> lk(s.mutex);
    const int a = actor_id(s, backend);
    tl_split_actor = a;

    if (n_inputs == 0) {
        trace(s, a, "split         0 inputs (no sync path)\n");
    }
}
