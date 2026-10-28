#include "llama-context.h"
#include "moe_predictor.h"
#include "moe_doorbell.h"
#include "ggml-cuda.h"
#include <thread>
#include <atomic>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <cinttypes>
#include <chrono>
#include <algorithm>
#include <climits>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <liburing.h>

static thread_local double moe_diag_publish_ms = 0.0;
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <cstdlib>
#include <vector>


static inline int moe_get_layer_slots(int il, int n_layer);
static inline int moe_get_layer_base_slot(int il, int n_layer);
static void moe_cache_scan(llama_context * ctx, const char * tag, bool drop_dup);

static bool moe_doorbell_on(const llama_context * ctx) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("MOE_DOORBELL");
        on = (!e || atoi(e) != 0) ? 1 : 0;
    }
    return on && ctx && ctx->moe_enabled();
}

static llama_context * g_moe_resolve_ctx = nullptr;

// Per-layer batch plan published by the resolve kernel. Keep in sync with
// src/moe_doorbell.cu.
#define MOE_DOORBELL_PLAN_MAX 256
struct DoorbellPlan {
    volatile uint32_t state;
    volatile uint32_t gpu_ready;
    uint32_t n;
    uint32_t n_miss;
    uint32_t n_copy;
    uint32_t stage_stride;
    uint32_t gate_bytes;
    uint32_t up_bytes;
    uint32_t down_bytes;
    uint32_t s_pl;
    uint32_t pad0;
    uint32_t pad1;
    int32_t  sel[MOE_DOORBELL_PLAN_MAX];
    int32_t  assigned[MOE_DOORBELL_PLAN_MAX];
    int32_t  evicted[MOE_DOORBELL_PLAN_MAX];
    int32_t  copy_slot[MOE_DOORBELL_PLAN_MAX];
    uint32_t copy_pre[MOE_DOORBELL_PLAN_MAX];
    volatile uint32_t copy_ready[MOE_DOORBELL_PLAN_MAX];
    uint64_t dst_gate;
    uint64_t dst_up;
    uint64_t dst_down;
    uint64_t stage;
};

extern "C" {
void * moe_doorbell_plan_host(int il);
void * moe_doorbell_plan_dev(int il);
int moe_doorbell_plan_cap();
bool moe_doorbell_stage_init(size_t stride, int n_slots);
void * moe_doorbell_stage_host();
void * moe_doorbell_stage_dev();
size_t moe_doorbell_stage_stride();
int moe_doorbell_stage_cap();
}

static std::atomic<bool> g_doorbell_running{false};
static std::thread g_doorbell_thread;
static std::atomic<bool> g_hot_rank_valid{false};

// Refresh the per-layer frequency rank from the decode counters. The doorbell
// worker is the only place that counts decodes in that mode, so it owns this.
static void moe_rank_refresh(llama_context * ctx) {
    const int n_layer = ctx->moe.n_layer;
    const int n_exp   = ctx->moe.n_expert;
    if (ctx->moe.exp_rank.size() != (size_t) n_layer * n_exp) return;
    if (ctx->moe.dec_freq.size() != (size_t) n_layer * n_exp) return;
    std::vector<int> order((size_t) n_exp);
    for (int il = 0; il < n_layer; il++) {
        for (int e = 0; e < n_exp; e++) order[(size_t) e] = e;
        const uint32_t * d = ctx->moe.dec_freq.data() + (size_t) il * n_exp;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return d[a] > d[b]; });
        uint16_t * rk = ctx->moe.exp_rank.data() + (size_t) il * n_exp;
        for (int k = 0; k < n_exp; k++) rk[order[(size_t) k]] = (uint16_t) k;
    }
    g_hot_rank_valid.store(true, std::memory_order_relaxed);
}

static void start_doorbell_worker(llama_context * ctx) {
    if (g_doorbell_running.load()) return;
    g_doorbell_running.store(true);
    g_doorbell_thread = std::thread([ctx]() {
        LayerDoorbell * dbs = moe_doorbell_get_host_ptr();
        if (!dbs) {
            g_doorbell_running.store(false);
            return;
        }
        const int n_layer = ctx->moe.n_layer;
        const int n_exp   = ctx->moe.n_expert;
        const int n_used  = ctx->moe.n_used;
        char * stage_host = (char *) moe_doorbell_stage_host();
        const size_t stride = moe_doorbell_stage_stride();
        const int stage_cap = moe_doorbell_stage_cap();
        if (!stage_host || stride == 0 || stage_cap <= 0) {
            fprintf(stderr, "[doorbell_worker] no pinned stage, worker disabled\n");
            g_doorbell_running.store(false);
            return;
        }
        int fd_gguf = -1;
        if (!ctx->pager.pack_mode()) {
            fd_gguf = open(ctx->pager.path(), O_RDONLY);
        }
        // Buffered reads: the page cache holds the hot expert set, and io_uring
        // submits the whole layer at once so the reads overlap.
        struct io_uring ring;
        bool ring_ok = io_uring_queue_init(512, &ring, 0) == 0;
        fprintf(stderr, "[doorbell_worker] STARTED for %d layers, stage %d x %zu, uring=%d\n",
                n_layer, stage_cap, stride, ring_ok ? 1 : 0);

        std::vector<char>    pinned;
        std::vector<int32_t> old_keys, misses, sel;
        std::vector<moe_pager::moe_pack_loc> locs;
        std::vector<int>     left;
        std::vector<uint32_t> exp_len;
        std::vector<char>    read_fail, done;
        long served = 0, failed = 0;
        long n_hit = 0, n_sel = 0;
        double t_read = 0.0, t_wait = 0.0;
        long n_read = 0;

        while (g_doorbell_running.load(std::memory_order_relaxed)) {
            bool handled = false;
            for (int il = 0; il < n_layer; ++il) {
                if (dbs[il].state != 1) continue;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                dbs[il].state = 2;

                DoorbellPlan * pl = (DoorbellPlan *) moe_doorbell_plan_host(il);
                if (!pl) continue;

                const int s_pl = moe_get_layer_slots(il, n_layer);
                const int base = moe_get_layer_base_slot(il, n_layer);
                int n = (int) pl->n;
                if (n < 0) n = 0;
                if (n > MOE_DOORBELL_PLAN_MAX) n = MOE_DOORBELL_PLAN_MAX;

                // real LRU: the tick was frozen in doorbell mode before
                if (++ctx->moe.lru.tick == 0) {
                    ctx->moe.lru.init(ctx->moe.lru.last.size(), 1);
                    ctx->moe.lru.tick = 2;
                }

                pinned.assign((size_t) s_pl, 0);
                old_keys.resize((size_t) s_pl);
                for (int s = 0; s < s_pl; ++s) {
                    old_keys[(size_t) s] = ctx->moe.slot_key[(size_t) base + s];
                }

                sel.resize((size_t) n);
                misses.clear();
                misses.reserve((size_t) n);
                for (int j = 0; j < n; ++j) {
                    const int32_t e = pl->sel[j];
                    sel[(size_t) j] = e;
                    if (e < 0 || e >= n_exp) continue;
                    n_sel++;
                    const int cur = ctx->moe.slot_of[(size_t) il * n_exp + e];
                    if (cur >= base && cur < base + s_pl) {
                        pinned[(size_t) cur - (size_t) base] = 1;
                        ctx->moe.lru.hit((size_t) cur);
                        n_hit++;
                    } else {
                        misses.push_back(e);
                    }
                }
                std::sort(misses.begin(), misses.end());
                misses.erase(std::unique(misses.begin(), misses.end()), misses.end());

                const size_t gb = ctx->moe.host_gate[il];
                const size_t ub = ctx->moe.host_up[il];
                const size_t db = ctx->moe.host_down[il];
                pl->stage        = (uint64_t) (uintptr_t) moe_doorbell_stage_dev();
                pl->stage_stride = (uint32_t) stride;
                pl->gate_bytes   = (uint32_t) gb;
                pl->up_bytes     = (uint32_t) ub;
                pl->down_bytes   = (uint32_t) db;
                pl->s_pl         = (uint32_t) s_pl;
                pl->dst_gate     = (uint64_t) (uintptr_t) ctx->moe.pool_gate[il]->data;
                pl->dst_up       = (uint64_t) (uintptr_t) ctx->moe.pool_up[il]->data;
                pl->dst_down     = (uint64_t) (uintptr_t) ctx->moe.pool_down[il]->data;

                // pick victims and publish the read plan before the reads start,
                // so the kernel can copy each expert as soon as its read lands
                locs.clear();
                int ncopy = 0;
                const int warm_pad = ctx->hot_k;
                const bool rank_ok = warm_pad > 0 &&
                    ctx->moe.exp_rank.size() == (size_t) n_layer * n_exp &&
                    g_hot_rank_valid.load(std::memory_order_relaxed);
                std::vector<char> is_cold;
                std::vector<int>  rfd;
                for (size_t mi = 0; mi < misses.size() && ncopy < stage_cap; ++mi) {
                    const int e = misses[mi];
                    const size_t v = ctx->moe.lru.victim(pinned.data(), (size_t) base, (size_t) s_pl);
                    if (v == (size_t) -1) break;
                    pinned[v - (size_t) base] = 1;
                    const int old_key = ctx->moe.slot_key[v];
                    if (old_key >= 0) ctx->moe.slot_of[(size_t) old_key] = -1;
                    ctx->moe.slot_of[(size_t) il * n_exp + e] = (int) v;
                    ctx->moe.slot_key[v] = il * n_exp + e;
                    ctx->moe.lru.miss(v);
                    pl->copy_slot[ncopy] = (int32_t) (v - (size_t) base);
                    pl->copy_ready[ncopy] = 0;
                    // twin-FD split: warm misses (rank in [S_L, S_L+W)) go through
                    // the buffered fd so they ride in the page cache; cold-tail
                    // misses go O_DIRECT so they never evict the warm tier.
                    bool cold = false;
                    if (rank_ok) {
                        const uint16_t rk = ctx->moe.exp_rank[(size_t) il * n_exp + e];
                        cold = rk >= (uint16_t)(s_pl + warm_pad);
                    }
                    is_cold.push_back((char) cold);
                    if (ctx->pager.pack_mode()) {
                        const moe_pager::moe_pack_loc loc = ctx->pager.pack_loc(il, e);
                        locs.push_back(loc);
                        pl->copy_pre[ncopy] = loc.pre;
                        rfd.push_back(cold && ctx->pager.pack_fd() >= 0 ? ctx->pager.pack_fd() : loc.fd);
                    } else {
                        locs.push_back(moe_pager::moe_pack_loc{-1, 0, 0, 0});
                        pl->copy_pre[ncopy] = 0;
                        rfd.push_back(fd_gguf);
                    }
                    ncopy++;
                }
                pl->n_copy = (uint32_t) ncopy;

                for (int j = 0; j < n; ++j) {
                    const int32_t e = sel[(size_t) j];
                    const int key = il * n_exp + e;
                    int cur = -1;
                    if (e >= 0 && e < n_exp) {
                        cur = ctx->moe.slot_of[(size_t) key];
                    }
                    const int local = (cur >= base && cur < base + s_pl) ? cur - base : s_pl;
                    pl->assigned[j] = local;
                    const int old = (local >= 0 && local < s_pl) ? old_keys[(size_t) local] : -1;
                    pl->evicted[j] = (old >= 0 && old != key) ? old - il * n_exp : -1;
                }

                if (n == n_used) {
                    for (int j = 0; j < n; ++j) {
                        const int32_t e = sel[(size_t) j];
                        if (e < 0 || e >= n_exp) continue;
                        ctx->moe.step_sel[(size_t) il * n_used + j] = e;
                        ctx->moe.dec_freq[(size_t) il * n_exp + e]++;
                    }
                }

                // publish the plan and ring the doorbell
                std::atomic_thread_fence(std::memory_order_seq_cst);
                moe_doorbell_signal_unblock(il, nullptr);

                left.assign((size_t) ncopy, 0);
                exp_len.assign((size_t) ncopy * 3, 0);
                read_fail.assign((size_t) ncopy, 0);
                done.assign((size_t) ncopy, 0);

                auto publish_ready = [&](int ci) {
                    pl->copy_ready[ci] = 1;
#if defined(__x86_64__) || defined(_M_X64)
                    _mm_clflush((const void *) (uintptr_t) &pl->copy_ready[ci]);
                    _mm_sfence();
#endif
                    done[(size_t) ci] = 1;
                    // The expert now lives in the VRAM pool, so its page-cache
                    // copy is a duplicate. Drop it to keep the two tiers
                    // exclusive and leave the RAM for the warm tail.
                    if (ctx->pager.pack_mode() && ctx->hot_k > 0) {
                        const moe_pager::moe_pack_loc & l = locs[(size_t) ci];
                        if (l.fd >= 0) {
                            posix_fadvise(l.fd, (off_t) l.off, (off_t) l.len, POSIX_FADV_DONTNEED);
                        }
                    }
                };
                auto fail_ci = [&](int ci) {
                    if (done[(size_t) ci]) return;
                    // drop the mapping and make this expert read the sentinel
                    // row instead of stale pool bytes
                    const int e = misses[(size_t) ci];
                    const int cslot = pl->copy_slot[ci];
                    ctx->moe.slot_of[(size_t) il * n_exp + e] = -1;
                    if (cslot >= 0 && cslot < s_pl) {
                        ctx->moe.slot_key[(size_t) cslot + (size_t) base] = -1;
                    }
                    for (int j = 0; j < n; ++j) {
                        if (sel[(size_t) j] == e) pl->assigned[j] = s_pl;
                    }
                    pl->copy_slot[ci] = -1;
#if defined(__x86_64__) || defined(_M_X64)
                    _mm_clflush((const void *) &pl->copy_slot[ci]);
                    for (int j = 0; j < n; ++j) {
                        if (sel[(size_t) j] == e) _mm_clflush((const void *) &pl->assigned[j]);
                    }
                    _mm_sfence();
#endif
                    failed++;
                    publish_ready(ci);
                };

                // validate and fill the per-expert geometry first, so a bad
                // entry never leaves a hole the kernel would spin on
                for (int ci = 0; ci < ncopy; ++ci) {
                    if (ctx->pager.pack_mode()) {
                        const moe_pager::moe_pack_loc & loc = locs[(size_t) ci];
                        const int rfd_ci = rfd[(size_t) ci];
                        if (rfd_ci < 0 || loc.len > stride) { read_fail[(size_t) ci] = 1; continue; }
                        exp_len[(size_t) ci * 3] = loc.len;
                        left[(size_t) ci] = 1;
                    } else if (fd_gguf >= 0) {
                        exp_len[(size_t) ci * 3 + 0] = (uint32_t) gb;
                        exp_len[(size_t) ci * 3 + 1] = (uint32_t) ub;
                        exp_len[(size_t) ci * 3 + 2] = (uint32_t) db;
                        left[(size_t) ci] = 3;
                    } else {
                        read_fail[(size_t) ci] = 1;
                    }
                }

                const auto t_read0 = std::chrono::high_resolution_clock::now();
                if (!ring_ok) {
                    // no io_uring: serve the layer with synchronous reads
                    for (int ci = 0; ci < ncopy; ++ci) {
                        if (left[(size_t) ci] == 0) { fail_ci(ci); continue; }
                        char * dst = stage_host + (size_t) ci * stride;
                        bool ok = true;
                        if (ctx->pager.pack_mode()) {
                            const moe_pager::moe_pack_loc & loc = locs[(size_t) ci];
                            size_t got = 0;
                            while (got < loc.len) {
                                const ssize_t r = pread(rfd[(size_t) ci], dst + got, loc.len - got,
                                        (off_t) (loc.off + got));
                                if (r <= 0) break;
                                got += (size_t) r;
                            }
                            ok = got == loc.len;
                        } else {
                            const moe_expert_slice * s = ctx->pager.slices(il, misses[(size_t) ci]);
                            const size_t want[3] = { gb, ub, db };
                            const size_t off[3] = { 0, gb, gb + ub };
                            for (int r = 0; r < 3 && ok; ++r) {
                                size_t got = 0;
                                while (got < want[r]) {
                                    const ssize_t rr = pread(fd_gguf, dst + off[r] + got,
                                            want[r] - got, (off_t) (s[r].off + s[r].pre + got));
                                    if (rr <= 0) break;
                                    got += (size_t) rr;
                                }
                                ok = got == want[r];
                            }
                        }
                        if (ok) publish_ready(ci);
                        else { read_fail[(size_t) ci] = 1; fail_ci(ci); }
                    }
                } else {
                    int n_sqe = 0;
                    bool submit_ok = true;
                    for (int ci = 0; ci < ncopy && submit_ok; ++ci) {
                        if (left[(size_t) ci] == 0) continue;
                        char * dst = stage_host + (size_t) ci * stride;
                        if (ctx->pager.pack_mode()) {
                            const moe_pager::moe_pack_loc & loc = locs[(size_t) ci];
                            struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
                            if (!sqe) { submit_ok = false; break; }
                            io_uring_prep_read(sqe, rfd[(size_t) ci], dst, loc.len, (off_t) loc.off);
                            io_uring_sqe_set_data(sqe, (void *) (uintptr_t) ((size_t) ci * 3));
                            n_sqe++;
                        } else {
                            const moe_expert_slice * s = ctx->pager.slices(il, misses[(size_t) ci]);
                            const size_t want[3] = { gb, ub, db };
                            const size_t off[3] = { 0, gb, gb + ub };
                            for (int r = 0; r < 3; ++r) {
                                struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
                                if (!sqe) { submit_ok = false; break; }
                                io_uring_prep_read(sqe, fd_gguf, dst + off[r], want[r],
                                        (off_t) (s[r].off + s[r].pre));
                                io_uring_sqe_set_data(sqe, (void *) (uintptr_t) ((size_t) ci * 3 + (size_t) r));
                                n_sqe++;
                            }
                        }
                    }
                    bool submit_fail = !submit_ok;
                    if (!submit_fail && n_sqe > 0) {
                        int submitted = 0;
                        while (submitted < n_sqe) {
                            const int r = io_uring_submit(&ring);
                            if (r <= 0) { submit_fail = true; break; }
                            submitted += r;
                        }
                        if (submitted < n_sqe) submit_fail = true;
                    }
                    if (submit_fail) {
                        for (int ci = 0; ci < ncopy; ++ci) {
                            if (left[(size_t) ci] > 0) { read_fail[(size_t) ci] = 1; left[(size_t) ci] = 0; }
                        }
                        io_uring_queue_exit(&ring);
                        ring_ok = false;
                        fprintf(stderr, "[doorbell_worker] io_uring submit failed, falling back to pread\n");
                    }

                    // pre-failed experts must release the kernel now
                    for (int ci = 0; ci < ncopy; ++ci) {
                        if (left[(size_t) ci] == 0) fail_ci(ci);
                    }

                    // reap completions and publish each expert as it lands; the
                    // kernel copies it while the later reads are still in flight
                    int remaining = n_sqe;
                    while (remaining > 0 && ring_ok) {
                        struct io_uring_cqe * cqe = nullptr;
                        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
                        const uint64_t ud = (uint64_t) (uintptr_t) io_uring_cqe_get_data(cqe);
                        const int ci = (int) (ud / 3);
                        const int r  = (int) (ud % 3);
                        const int res = cqe->res;
                        io_uring_cqe_seen(&ring, cqe);
                        remaining--;
                        if (ci < 0 || ci >= ncopy) continue;
                        if (left[(size_t) ci] == 0) continue;
                        if (res != (int) exp_len[(size_t) ci * 3 + (size_t) r]) {
                            // rare short read: retry this range synchronously
                            // once before dropping the expert to the sentinel
                            bool ok = false;
                            const size_t want = exp_len[(size_t) ci * 3 + (size_t) r];
                            char * dst = stage_host + (size_t) ci * stride;
                            if (ctx->pager.pack_mode()) {
                                const moe_pager::moe_pack_loc & loc = locs[(size_t) ci];
                                size_t got = 0;
                                while (got < want) {
                                    const ssize_t rr = pread(rfd[(size_t) ci], dst + got, want - got,
                                            (off_t) (loc.off + got));
                                    if (rr <= 0) break;
                                    got += (size_t) rr;
                                }
                                ok = got == want;
                            } else {
                                const moe_expert_slice * s = ctx->pager.slices(il, misses[(size_t) ci]);
                                const size_t off[3] = { 0, gb, gb + ub };
                                size_t got = 0;
                                while (got < want) {
                                    const ssize_t rr = pread(fd_gguf, dst + off[r] + got, want - got,
                                            (off_t) (s[r].off + s[r].pre + got));
                                    if (rr <= 0) break;
                                    got += (size_t) rr;
                                }
                                ok = got == want;
                            }
                            if (!ok) {
                                fprintf(stderr, "[doorbell] short read layer=%d expert=%d role=%d res=%d want=%zu\n",
                                        il, misses[(size_t) ci], r, res, want);
                                read_fail[(size_t) ci] = 1;
                            }
                        }
                        if (--left[(size_t) ci] == 0) {
                            if (read_fail[(size_t) ci]) fail_ci(ci);
                            else publish_ready(ci);
                        }
                    }
                }
                // a failed wait must still release the kernel
                for (int ci = 0; ci < ncopy; ++ci) {
                    if (!done[(size_t) ci]) fail_ci(ci);
                }
                const auto t_read1 = std::chrono::high_resolution_clock::now();
                t_read += std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();
                n_read += ncopy;

                handled = true;
                served++;

                // refresh the page-cache ranking from the live decode counters;
                // overlaps with the kernel copy below
                if ((served % 6144) == 0) {
                    moe_rank_refresh(ctx);
                    if ((served % 24576) == 0) {
                        moe_cache_scan(ctx, "run", true);
                    }
                }

                // The kernel copies from the stage buffer. Wait until it is done
                // before the next layer reuses those slots.
                while (dbs[il].state != 0) {
#if defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
#else
                    std::this_thread::yield();
#endif
                }
                t_wait += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_read1).count();

                if ((served % 512) == 0) {
                    fprintf(stderr, "[doorbell] served=%ld layer=%d n=%d miss=%zu copy=%d failed=%ld "
                            "hit=%.1f%% read=%.2f ms/expert wait=%.2f ms/expert\n",
                            served, il, n, misses.size(), ncopy, failed,
                            n_sel ? 100.0 * (double) n_hit / (double) n_sel : 0.0,
                            n_read ? t_read / n_read : 0.0,
                            n_read ? t_wait / n_read : 0.0);
                    t_read = t_wait = 0.0;
                    n_read = 0;
                }
            }
            if (!handled) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            }
        }
        if (ring_ok) {
            io_uring_queue_exit(&ring);
        }
        if (fd_gguf >= 0) {
            close(fd_gguf);
        }
    });
}

static void stop_doorbell_worker() {
    if (!g_doorbell_running.load()) return;
    g_doorbell_running.store(false);
    if (g_doorbell_thread.joinable()) {
        g_doorbell_thread.join();
    }
}

static void moe_slot_resolve_sync(
    void * stream,
    const int32_t * topk,
    const int32_t * table,
    int32_t * out,
    int64_t nk,
    int64_t nt,
    size_t topk_nb0,
    size_t topk_nb1,
    size_t table_nb0,
    size_t out_nb0,
    size_t out_nb1,
    const int32_t * full,
    int64_t fnk,
    int64_t fnt,
    size_t full_nb1,
    const int32_t * look,
    int64_t lnk,
    int32_t il,
    int32_t n_exp,
    int32_t sentinel);

static void moe_slot_resolve_register_ctx(llama_context * ctx) {
    g_moe_resolve_ctx = ctx;
    ggml_cuda_moe_slot_resolve_register(moe_slot_resolve_sync);
}

static void moe_slot_resolve_sync(
    void * stream,
    const int32_t * topk,
    const int32_t * table,
    int32_t * out,
    int64_t nk,
    int64_t nt,
    size_t topk_nb0,
    size_t topk_nb1,
    size_t table_nb0,
    size_t out_nb0,
    size_t out_nb1,
    const int32_t * full,
    int64_t fnk,
    int64_t fnt,
    size_t full_nb1,
    const int32_t * look,
    int64_t lnk,
    int32_t il,
    int32_t n_exp,
    int32_t sentinel) {
    (void) table;
    (void) table_nb0;
    (void) out_nb0;
    (void) out_nb1;
    llama_context * ctx = g_moe_resolve_ctx;
    const int64_t total = nk * nt;
    if (!ctx || !ctx->moe_enabled() || il < 0 || il >= ctx->moe.n_layer || total <= 0 || total > 80) {
        return;
    }
    (void) full; (void) fnk; (void) fnt; (void) full_nb1;
    static long s_nres = 0, s_nmiss = 0, s_ntok = 0, s_host = 0;
    static double s_tres = 0.0, s_ta = 0.0, s_tio = 0.0;
    cudaStream_t st = (cudaStream_t) stream;
    auto t_a = std::chrono::high_resolution_clock::now();
    cudaStreamSynchronize(st);
    std::vector<int32_t> ids((size_t) total);
    if (topk_nb0 == sizeof(int32_t) && topk_nb1 == (size_t) nk * sizeof(int32_t)) {
        cudaMemcpy(ids.data(), topk, (size_t) total * sizeof(int32_t), cudaMemcpyDeviceToHost);
    } else {
        for (int64_t t = 0; t < nt; t++) {
            cudaMemcpy(ids.data() + (size_t) t * nk,
                       (const char *) topk + (size_t) t * topk_nb1,
                       (size_t) nk * sizeof(int32_t), cudaMemcpyDeviceToHost);
        }
    }
    const int n_layer = ctx->moe.n_layer;
    const int s_pl = moe_get_layer_slots(il, n_layer);
    const int base = moe_get_layer_base_slot(il, n_layer);
    if (look && lnk > 0 && nt == 1 && il + 1 < n_layer) {
        static int look_pf = -1;
        if (look_pf < 0) { const char * e = getenv("MOE_LOOK_PF"); look_pf = (e && atoi(e) != 0) ? 1 : 0; }
        std::vector<int32_t> pred((size_t) std::min<int64_t>(lnk, 6));
        cudaMemcpy(pred.data(), look, pred.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for (int c = 0; c < 6; c++) {
            ctx->look_pred[(size_t) (il + 1) * 6 + c] = c < (int) pred.size() ? pred[(size_t) c] : -1;
        }
        ctx->look_fresh[(il + 1)]++;
        if (look_pf && ctx->or_run.load(std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> lock(ctx->or_mtx);
            for (size_t i = 0; i < pred.size() && ctx->or_q.size() < 4096; i++) {
                const int32_t f = pred[i];
                if (f < 0 || f >= n_exp) continue;
                if (ctx->moe.slot_of[(size_t) (il + 1) * n_exp + f] >= 0) continue;
                ctx->or_q.push_back(((uint32_t) (il + 1) << 16) | (uint32_t) f);
                ctx->look_push++;
            }
        }
    }
    if (nt == 1 && ctx->look_fresh[il]) {
        for (int64_t i = 0; i < nk; i++) {
            ctx->look_need++;
            for (int c = 0; c < 6; c++) {
                if (ctx->look_pred[(size_t) il * 6 + c] == ids[(size_t) i]) { ctx->look_hit++; break; }
            }
        }
        if ((ctx->look_need % 20000) < (long) nk) {
            fprintf(stderr, "[moe_look] need=%ld acc=%.1f%% pushed=%ld\n",
                ctx->look_need, 100.0 * ctx->look_hit / std::max(1L, ctx->look_need), ctx->look_push);
        }
    }
    auto t_b = std::chrono::high_resolution_clock::now();
    s_ta += std::chrono::duration<double, std::milli>(t_b - t_a).count();

    std::vector<int32_t> slot_of((size_t) total, sentinel);
    std::vector<char> pinned((size_t) s_pl, 0);
    std::vector<int32_t> uniq;
    uniq.reserve((size_t) total);
    std::vector<char> seen((size_t) n_exp > 0 ? (size_t) n_exp : 1, 0);
    const bool is_decode = (nt == 1);
    if (is_decode) {
        if (++ctx->moe.lru.tick == 0) {
            ctx->moe.lru.init(ctx->moe.lru.last.size(), 1);
            ctx->moe.lru.tick = 2;
        }
    }
    for (int64_t i = 0; i < total; i++) {
        const int32_t e = ids[(size_t) i];
        if (e < 0 || e >= n_exp) continue;
        const int cur = ctx->moe.slot_of[(size_t) il * n_exp + e];
        if (cur >= base && cur < base + s_pl) {
            slot_of[(size_t) i] = cur - base;
            pinned[(size_t) cur - (size_t) base] = 1;
            ctx->moe.n_hit++;
            if (is_decode) ctx->moe.lru.hit((size_t) cur);
        } else if (!seen[(size_t) e]) {
            seen[(size_t) e] = 1;
            uniq.push_back(e);
        }
    }
    if (nt == 1 && nk == ctx->moe.n_used) {
        for (int64_t i = 0; i < nk; i++) {
            const int32_t e = ids[(size_t) i];
            if (e < 0 || e >= n_exp) continue;
            ctx->moe.step_sel[(size_t) il * ctx->moe.n_used + i] = e;
            ctx->moe.dec_freq[(size_t) il * n_exp + e]++;
        }
        if (il == 0) {
            static long s_dec_steps = 0;
            if (++s_dec_steps % 128 == 0) {
                ctx->moe_hot_save();
            }
        }
        if (!ctx->moe.pred16.empty()) {
            ctx->moe.pred_fresh++;
            const uint16_t * p = ctx->moe.pred16.data() + (size_t) il * 16;
            for (int64_t i = 0; i < nk; i++) {
                const int32_t e = ids[(size_t) i];
                for (int c = 0; c < 16; c++) {
                    if (p[c] == (uint16_t) e) { ctx->moe.pred_hit++; break; }
                }
            }
        }
    }
    std::sort(uniq.begin(), uniq.end());
    ctx->moe.count_freq = (nt == 1);
    if (is_decode && nk == ctx->moe.n_used) {
        if (il == 0) {
            if (++ctx->co_tok == 0) {
                std::fill(ctx->co_hint.begin(), ctx->co_hint.end(), 0);
                ctx->co_tok = 1;
            }
        }
        if (il > 0 && !ctx->coact.empty()) {
            for (size_t i = 0; i < uniq.size(); i++) {
                const int32_t f = uniq[i];
                for (int64_t k = 0; k < nk; k++) {
                    const int32_t e = ctx->moe.step_sel[(size_t) (il - 1) * ctx->moe.n_used + k];
                    if (e < 0 || e >= n_exp) continue;
                    const size_t b = ((size_t) (il - 1) * n_exp + (size_t) e) * 4;
                    int slot = -1;
                    uint16_t minc = 0xffff;
                    for (int c = 0; c < 4; c++) {
                        if (ctx->coact[b + c] == (uint16_t) f) { slot = c; break; }
                        if (ctx->coact_cnt[b + c] < minc) { minc = ctx->coact_cnt[b + c]; slot = c; }
                    }
                    if (ctx->coact[b + slot] != (uint16_t) f) {
                        ctx->coact[b + slot] = (uint16_t) f;
                        ctx->coact_cnt[b + slot] = 0;
                    }
                    if (ctx->coact_cnt[b + slot] < 60000) ctx->coact_cnt[b + slot]++;
                    else {
                        for (int c = 0; c < 4; c++) ctx->coact_cnt[b + c] /= 2;
                    }
                }
            }
        }
        if (il + 1 < ctx->moe.n_layer && !ctx->coact.empty()) {
            static int co_on = -1;
            if (co_on < 0) { const char * e = getenv("MOE_COACT"); co_on = (e && atoi(e) != 0) ? 1 : 0; }
            for (size_t i = 0; i < uniq.size(); i++) {
                ctx->co_need++;
                if (ctx->co_hint[(size_t) il * n_exp + uniq[i]] == ctx->co_tok) ctx->co_cover++;
            }
            if (co_on && ctx->or_run.load(std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> lock(ctx->or_mtx);
                for (int64_t k = 0; k < nk && ctx->or_q.size() < 4096; k++) {
                    const int32_t e = ids[(size_t) k];
                    if (e < 0 || e >= n_exp) continue;
                    const size_t b = ((size_t) il * n_exp + (size_t) e) * 4;
                    for (int c = 0; c < 4 && ctx->or_q.size() < 4096; c++) {
                        const int32_t f = ctx->coact[b + c];
                        if (f < 0 || f >= n_exp) continue;
                        uint32_t & st = ctx->co_hint[(size_t) (il + 1) * n_exp + f];
                        if (st == ctx->co_tok) continue;
                        st = ctx->co_tok;
                        if (ctx->moe.slot_of[(size_t) (il + 1) * n_exp + f] >= 0) continue;
                        ctx->or_q.push_back(((uint32_t) (il + 1) << 16) | (uint32_t) f);
                        ctx->co_push++;
                    }
                }
            }
        }
        {
            static long xl_n = 0;
            xl_n += (long) uniq.size();
            if (xl_n >= 20000) {
                xl_n = 0;
                fprintf(stderr, "[moe_co] need=%ld cover=%.1f%% pushed=%ld ra_done=%ld\n",
                    ctx->co_need, ctx->co_need ? 100.0 * ctx->co_cover / ctx->co_need : 0.0,
                    ctx->co_push, ctx->moe.ra_done.load());
            }
        }
    }
    auto t_res0 = std::chrono::high_resolution_clock::now();
    bool host_ok = ctx->moe_page_host_batch(il, uniq.data(), uniq.size(), pinned, base, s_pl, st);
    s_host += host_ok ? 1 : 0;
    s_tio += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_res0).count();
    if (!host_ok) {
        std::vector<moe_batch_target> batch_targets;
        std::vector<int> target_slots;
        batch_targets.reserve(uniq.size());
        target_slots.reserve(uniq.size());

        const size_t g_slot = ggml_nbytes(ctx->moe.pool_gate[il]) / (size_t) (s_pl + 1);
        const size_t u_slot = ggml_nbytes(ctx->moe.pool_up[il])   / (size_t) (s_pl + 1);
        const size_t d_slot = ggml_nbytes(ctx->moe.pool_down[il]) / (size_t) (s_pl + 1);

        for (size_t i = 0; i < uniq.size(); ++i) {
            const int32_t e = uniq[i];
            const size_t v = ctx->moe.lru.victim((const char *) pinned.data(), (size_t) base, (size_t) s_pl);
            if (v == (size_t) -1) continue;
            int slot = (int) v;
            pinned[(size_t) slot - (size_t) base] = 1;

            const int local = slot - base;
            moe_batch_target tgt;
            tgt.expert = e;
            tgt.dst_gate = (char *) ctx->moe.pool_gate[il]->data + (size_t) local * g_slot;
            tgt.gate_bytes = g_slot;
            tgt.dst_up   = (char *) ctx->moe.pool_up[il]->data   + (size_t) local * u_slot;
            tgt.up_bytes = u_slot;
            tgt.dst_down = (char *) ctx->moe.pool_down[il]->data + (size_t) local * d_slot;
            tgt.down_bytes = d_slot;
            batch_targets.push_back(tgt);
            target_slots.push_back(slot);
        }

        if (!batch_targets.empty()) {
            for (size_t i = 0; i < batch_targets.size(); ++i) {
                int slot = target_slots[i];
                int e = batch_targets[i].expert;
                if (!ctx->pager.fetch_direct_io(il, e,
                    (char *) ctx->moe.pool_gate[il]->data + (size_t) slot * g_slot, g_slot,
                    (char *) ctx->moe.pool_up[il]->data   + (size_t) slot * u_slot, u_slot,
                    (char *) ctx->moe.pool_down[il]->data + (size_t) slot * d_slot, d_slot,
                    nullptr, slot)) {
                    ctx->moe.prefill_miss++;
                    continue;
                }
                ctx->moe.prefill_miss++;
                if (ctx->moe.count_freq) {
                    ctx->moe.slot_freq[(size_t) slot]++;
                    ctx->moe.lru.miss((size_t) slot);
                } else {
                    ctx->moe.lru.prefill_miss((size_t) slot);
                }
                ctx->moe_publish(il, e, slot);
                ctx->moe.n_miss++;
            }
        }
    }
    s_tres += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_res0).count();
    s_nres++;
    s_ntok += nt;
    s_nmiss += (long) uniq.size();
    if (s_nres % 1024 == 0) {
        static long p_warm = 0, p_pack = 0;
        const long d_warm = ctx->moe.n_warm_hit - p_warm;
        const long d_pack = ctx->moe.n_pack_hit - p_pack;
        p_warm = ctx->moe.n_warm_hit; p_pack = ctx->moe.n_pack_hit;
        const long wus = ctx->io_warm_us.load(), wn = ctx->io_warm_n.load();
        const long cus = ctx->io_cold_us.load(), cn = ctx->io_cold_n.load();
        fprintf(stderr,
                "[moe_res] n=%ld tok=%ld nt=%ld sync_ms=%.1f io_ms=%.1f tot_ms=%.1f miss=%ld host=%ld warm=%ld/%ld warm_ms=%.2f cold_ms=%.2f\n",
                s_nres, s_ntok, nt, s_ta, s_tio, s_tres, s_nmiss, s_host, d_warm, d_warm + d_pack,
                wn ? (double) wus / wn / 1000.0 : 0.0, cn ? (double) cus / cn / 1000.0 : 0.0);
        s_tres = 0.0; s_ta = 0.0; s_tio = 0.0; s_nres = 0; s_ntok = 0; s_nmiss = 0; s_host = 0;
    }
    if (!uniq.empty()) {
        cudaStream_t ws = (cudaStream_t) moe_pager::worker_stream();
        if (ws && ws != st) {
            thread_local cudaEvent_t s_ev = nullptr;
            if (!s_ev) cudaEventCreateWithFlags(&s_ev, cudaEventDisableTiming);
            cudaEventRecord(s_ev, ws);
            cudaStreamWaitEvent(st, s_ev, 0);
        }
    }
    for (int64_t i = 0; i < total; i++) {
        const int32_t e = ids[(size_t) i];
        if (e < 0 || e >= n_exp) continue;
        const int cur = ctx->moe.slot_of[(size_t) il * n_exp + e];
        if (cur >= base && cur < base + s_pl) slot_of[(size_t) i] = cur - base;
    }
    cudaMemcpyAsync(out, slot_of.data(), (size_t) total * sizeof(int32_t), cudaMemcpyHostToDevice, st);

}


// Page-cache read-ahead pool. The NVMe needs 4+ concurrent 1.4 MiB requests to
// reach full bandwidth, but one MoE layer only offers ~2 misses, so the queue
// must be filled from the predictor's cross-layer candidates instead. These
// threads issue readahead() only: no CUDA, no VRAM, no slot tables.
static void moe_ra_worker(llama_context * ctx) {
    const int fd = ctx->pager.pack_fd_buf();
    if (fd < 0) return;
    std::vector<uint32_t> batch;
    batch.reserve(256);
    while (ctx->or_run.load(std::memory_order_relaxed)) {
        batch.clear();
        {
            std::lock_guard<std::mutex> lock(ctx->or_mtx);
            const size_t n = ctx->or_q.size();
            if (n > 0) {
                const size_t take = std::min(n, (size_t) 256);
                batch.assign(ctx->or_q.end() - take, ctx->or_q.end());
                ctx->or_q.erase(ctx->or_q.end() - take, ctx->or_q.end());
            }
        }
        if (batch.empty()) {
            // idle: walk the frequency-ranked overflow set so the experts the
            // decode path keeps missing stay in the page cache
            if (ctx->hot_cycles) ctx->moe_hot_cycle(fd);
            else std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        for (size_t i = 0; i < batch.size(); i++) {
            const int il = (int) (batch[i] >> 16);
            const int e = (int) (batch[i] & 0xffff);
            if (il < 0 || il >= ctx->moe.n_layer || e < 0 || e >= ctx->moe.n_expert) continue;
            const moe_expert_slice * s = ctx->pager.slices(il, e);
            readahead(fd, (off_t) s->off, (size_t) s->len);
            ctx->moe.ra_done++;
        }
    }
}

static void moe_oracle_start(llama_context * ctx) {
    static int nth = [] {
        const char * e = getenv("MOE_RA_THREADS");
        if (e && atoi(e) == 0) return 0;
        int v = e && atoi(e) > 0 ? atoi(e) : 1;
        return std::min(v, 32);
    }();
    if (ctx->or_run.load() || nth <= 0) {
        LLAMA_LOG_WARN("%s: page-cache read-ahead DISABLED\n", __func__);
        return;
    }
    // The page-cache warmer is opt-in (MOE_HOT_K = warm experts per layer).
    // Measured net loss on this box: the demand stream already keeps the device
    // busy, so a warmer can only steal bandwidth from it.
    const char * hk = getenv("MOE_HOT_K");
    ctx->hot_k = hk ? atoi(hk) : 0;
    ctx->hot_cycles = ctx->hot_k > 0;
    LLAMA_LOG_INFO("%s: page-cache read-ahead on, %d threads, hot-cycle K=%d\n",
                   __func__, nth, ctx->hot_k);
    ctx->or_run.store(true);
    for (int i = 0; i < nth; i++) ctx->or_thr.emplace_back(moe_ra_worker, ctx);
}

static void moe_oracle_stop(llama_context * ctx) {
    if (!ctx->or_run.load()) return;
    ctx->or_run.store(false);
    for (auto & th : ctx->or_thr) if (th.joinable()) th.join();
    ctx->or_thr.clear();
}

// Worker for the staging-ring preads. Creating threads per resolve put thread
// spawn cost on the decode critical path, so the pool is persistent.
static void moe_io_worker(llama_context * ctx) {
    unsigned long long seen = 0;
    while (ctx->io_run.load(std::memory_order_relaxed)) {
        {
            std::unique_lock<std::mutex> lock(ctx->io_mtx);
            ctx->io_cv.wait(lock, [&]() { return ctx->io_gen != seen || !ctx->io_run.load(); });
            if (!ctx->io_run.load()) return;
            seen = ctx->io_gen;
        }
        const size_t stride = ctx->io_stride;
        const size_t per = ctx->io_per;
        const size_t nth = ctx->io_nth;
        const size_t total = std::min(ctx->io_jobs.size(), per * nth);
        long local_fail = 0;
        for (;;) {
            const size_t i = ctx->io_next.fetch_add(1);
            if (i >= total) break;
            const int fd = ctx->io_jobs[i].fd;
            const uint64_t off = ctx->io_jobs[i].off;
            const uint32_t len = ctx->io_jobs[i].len;
            const int rslot = ctx->io_jobs[i].rslot;
            char * dst = ctx->moe.host_pool + (size_t) rslot * stride;
            if (fd < 0) { local_fail++; continue; }
            size_t left = len, done = 0;
            const auto t_io0 = std::chrono::high_resolution_clock::now();
            while (left > 0) {
                ssize_t r = pread(fd, dst + done, left, (off_t) (off + done));
                if (r <= 0) break;
                done += (size_t) r;
                left -= (size_t) r;
            }
            const long dus = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t_io0).count();
            if (ctx->io_jobs[i].warm) { ctx->io_warm_us.fetch_add(dus); ctx->io_warm_n.fetch_add(1); }
            else { ctx->io_cold_us.fetch_add(dus); ctx->io_cold_n.fetch_add(1); }
            if (done != len) local_fail++;
        }
        if (local_fail) ctx->io_fail.fetch_add(local_fail);
        // Only the last worker out publishes completion: workers with no work
        // must not release the batch while others are still filling the ring.
        if (ctx->io_finished.fetch_add(1) + 1 == (long) ctx->io_thr_active) {
            ctx->io_gen_done.store(seen);
            ctx->io_done_cv.notify_all();
        }
    }
}

// Page-cache warmer budget per thread. The demand stream already keeps the
// device busy, so the warmer must stay well under the device rate.
static constexpr double MOE_WARM_RATE_MB = 200.0;

// 1 when the whole expert range is already in the page cache. Used to keep the
// background warmer from re-issuing reads for data that is already there.
static bool moe_ra_resident(int fd, off_t off, size_t len) {
    static std::mutex mtx;
    static int     cfd = -1;
    static char *  map = nullptr;
    static size_t  map_len = 0;
    std::lock_guard<std::mutex> lock(mtx);
    if (fd != cfd) {
        if (map) { munmap(map, map_len); map = nullptr; map_len = 0; }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) return false;
        map = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) { map = nullptr; return false; }
        map_len = (size_t) st.st_size;
        cfd = fd;
    }
    if (!map || (size_t) off + len > map_len) return false;
    const size_t page = 4096;
    const off_t  a = off & ~(off_t) (page - 1);
    const size_t n = (size_t) ((off + (off_t) len - a + (off_t) page - 1) / (off_t) page);
    unsigned char vec[512];
    if (n > sizeof(vec)) return false;
    if (mincore(map + a, n * page, vec) != 0) return false;
    for (size_t i = 0; i < n; i++) {
        if (!(vec[i] & 1)) return false;
    }
    return true;
}

// Residency census over the pack file: how much of the VRAM pool is duplicated
// in the page cache, and how much of the warm tail the page cache actually
// holds. `drop_dup` frees the duplicates so the RAM tier can use that space.
static void moe_cache_scan(llama_context * ctx, const char * tag, bool drop_dup) {
    const int n_layer = ctx->moe.n_layer;
    const int n_exp   = ctx->moe.n_expert;
    if (!ctx->pager.pack_mode()) return;
    long n_vram = 0, n_ram = 0, n_both = 0, n_warm = 0, n_warm_ram = 0;
    double dropped = 0.0;
    const int pad = ctx->hot_k;
    for (int il = 0; il < n_layer; il++) {
        const int s_l = moe_get_layer_slots(il, n_layer);
        for (int e = 0; e < n_exp; e++) {
            const moe_pager::moe_pack_loc loc = ctx->pager.pack_loc(il, e);
            if (loc.fd < 0) continue;
            const bool vram = ctx->moe.slot_of[(size_t) il * n_exp + e] >= 0;
            const bool ram  = moe_ra_resident(loc.fd, (off_t) loc.off, (size_t) loc.len);
            const bool warm = !vram && pad > 0 && g_hot_rank_valid.load(std::memory_order_relaxed) &&
                ctx->moe.exp_rank.size() == (size_t) n_layer * n_exp &&
                ctx->moe.exp_rank[(size_t) il * n_exp + e] < (uint16_t) (s_l + pad);
            if (vram) n_vram++;
            if (ram)  n_ram++;
            if (vram && ram) n_both++;
            if (warm) { n_warm++; if (ram) n_warm_ram++; }
            if (drop_dup && vram && ram) {
                if (posix_fadvise(loc.fd, (off_t) loc.off, (off_t) loc.len, POSIX_FADV_DONTNEED) == 0) {
                    dropped += (double) loc.len;
                }
            }
        }
    }
    fprintf(stderr, "[moe_cache] %s vram=%ld ram=%ld dup=%ld warm=%ld warm_ram=%ld (%.1f%%) dropped=%.2f GiB\n",
            tag, n_vram, n_ram, n_both, n_warm, n_warm_ram,
            n_warm ? 100.0 * (double) n_warm_ram / (double) n_warm : 0.0,
            dropped / 1073741824.0);
}

// Walk the frequency-ranked experts that overflow the VRAM slots so the page
// cache keeps them resident. Ranking (exp_rank) is refreshed by moe_hot_save()
// every 128 decode tokens; resident experts are skipped.
void llama_context::moe_hot_cycle(int fd) {
    const int k_hot = hot_k;
    if (k_hot <= 0 || moe.exp_rank.empty() || !g_hot_rank_valid.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return;
    }
    const int n_layer = moe.n_layer;
    const int n_exp = moe.n_expert;
    size_t issued = 0;
    for (int t = 0; t < k_hot; t++) {
        const int e = hot_pos % n_exp;
        const int il = (hot_pos / n_exp) % n_layer;
        hot_pos++;
        if (moe.slot_of[(size_t) il * n_exp + e] >= 0) continue;
        // k_hot is a per-layer pad on top of the VRAM slots, so layers with a
        // larger pool warm a proportionally smaller tail
        const int s_l = moe_get_layer_slots(il, n_layer);
        const uint16_t * rk = moe.exp_rank.data() + (size_t) il * n_exp;
        if (rk[e] >= (uint16_t) (s_l + k_hot)) continue;
        int rfd = fd;
        off_t roff = 0;
        size_t rlen = 0;
        if (pager.pack_mode()) {
            // the doorbell reads the pack blob, not the GGUF
            const moe_pager::moe_pack_loc loc = pager.pack_loc(il, e);
            if (loc.fd < 0) continue;
            rfd = loc.fd; roff = (off_t) loc.off; rlen = (size_t) loc.len;
        } else {
            const moe_expert_slice * s = pager.slices(il, e);
            roff = (off_t) s->off; rlen = (size_t) s->len;
        }
        // skip experts that are already in the page cache: re-issuing
        // readahead for resident data only churns the cache
        if (moe_ra_resident(rfd, roff, rlen)) continue;
        readahead(rfd, roff, rlen);
        moe.ra_done++;
        issued += rlen;
    }
    // The warmer is a background filler, not a second demand stream. Pace it
    // well below the device rate so it never starves the doorbell reads.
    if (issued == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } else {
        const double us = (double) issued / (MOE_WARM_RATE_MB * 1048576.0) * 1e6;
        std::this_thread::sleep_for(std::chrono::microseconds((long long) us));
    }
}

// sparse MoE paging callback (defined near the moe_page helpers below)
static bool llama_moe_page_cb(struct ggml_tensor * t, bool ask, void * ud);


// Non-uniform "Arrival Slack" slot allocation (Pillar 2)
static int g_moe_slots_total = 0; // 0 = fixed table; else scaled total
static inline int moe_fixed_slots(int il, int n_layer) {
    if (il >= 0 && il < 5) return 220;
    if (il >= 5 && il < 43) return 92;
    if (il >= 43 && il < n_layer) return 80;
    return 104;
}
static inline int moe_get_layer_slots(int il, int n_layer) {
    const int base = moe_fixed_slots(il, n_layer);
    if (g_moe_slots_total <= 0) return base;
    int total_fixed = 0;
    for (int l = 0; l < n_layer; l++) total_fixed += moe_fixed_slots(l, n_layer);
    if (total_fixed <= 0) return base;
    int scaled = (base * g_moe_slots_total + total_fixed / 2) / total_fixed;
    return std::max(16, scaled);
}

static inline int moe_get_layer_base_slot(int il, int n_layer) {
    int base = 0;
    for (int l = 0; l < il && l < n_layer; l++) {
        base += moe_get_layer_slots(l, n_layer);
    }
    return base;
}

static inline int moe_get_total_slots(int n_layer) {
    int total = 0;
    for (int l = 0; l < n_layer; l++) {
        total += moe_get_layer_slots(l, n_layer);
    }
    return total;
}

//
// llama_context
//

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.moe_direct_io = params.moe_direct_io;
    cparams.moe_slots     = params.moe_slots;
    cparams.moe_pack_dir  = params.moe_pack_dir;
    cparams.moe_chunk_max = params.moe_chunk_max;
    cparams.moe_gguf_path = params.moe_gguf_path;
    cparams.moe_profile_path = params.moe_profile_path;
    if (params.moe_direct_io) {
        if (params.moe_slots > 0 && params.moe_slots != 4096) {
            g_moe_slots_total = params.moe_slots;
        } else {
            g_moe_slots_total = 0;
        }
    }

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);
    if (params.moe_direct_io) {
        cparams.n_ubatch = std::min(cparams.n_ubatch, 64u);
        // The slot table is static for the whole ubatch, so the widest batch
        // must fit the smallest layer pool: min_slots >= n_tokens * top-k.
        // Larger user ubatches are split into these safe chunks internally.
        const int n_layer = (int) hparams.n_layer();
        const int n_used  = std::max(1, (int) hparams.n_expert_used());
        int min_slots = INT_MAX;
        for (int il = 0; il < n_layer; ++il) {
            min_slots = std::min(min_slots, moe_get_layer_slots(il, n_layer));
        }
        const uint32_t safe = (uint32_t) std::max(1, min_slots / n_used);
        if (cparams.n_ubatch > safe) {
            LLAMA_LOG_INFO("%s: moe slot pool caps n_ubatch at %u (requested %u, min slots %d, top-k %d)\n",
                    __func__, safe, cparams.n_ubatch, min_slots, n_used);
            cparams.n_ubatch = safe;
        }
    }

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // ---- sparse MoE direct-IO slot pool (3.2A) -----------------------
        // Metadata-only ggml_new_tensor calls create nothing; physical VRAM is
        // claimed below on the GPU backend and its size is verified.
        if (cparams.moe_direct_io && cparams.moe_gguf_path) {
            const int n_layer = (int) model.hparams.n_layer();
            const int n_exp   = (int) model.hparams.n_expert;
            const int n_used  = (int) model.hparams.n_expert_used();

            std::string perr;
            if (!pager.init(cparams.moe_gguf_path, n_layer, n_exp, n_used, perr)) {
                LLAMA_LOG_WARN("%s: moe_direct_io OFF: %s\n", __func__, perr.c_str());
            } else {
                // optional single-blob pack: one contiguous read per expert.
                // verified byte-identical to the GGUF's gate|up|down, so this
                // only changes IO, not the weights.
                if (cparams.moe_pack_dir) {
                    std::string dir(cparams.moe_pack_dir);
                    std::string bin = dir + "/experts.bin";
                    std::string idx = dir + "/native_experts.txt";
                    std::string e2;
                    if (!pager.use_pack(bin.c_str(), idx.c_str(), e2)) {
                        LLAMA_LOG_WARN("%s: moe pack OFF: %s\n", __func__, e2.c_str());
                    } else {
                        LLAMA_LOG_INFO("%s: moe pack ON: one blob read per expert (%d layers x %d experts)\n",
                                       __func__, n_layer, n_exp);
                        const char * wp = getenv("MOE_WARM_PACK");
                        const char * wi = getenv("MOE_WARM_INDEX");
                        std::string wbin = wp && wp[0] ? wp : "/home/zyxciss/moe_warm_btrfs.bin";
                        std::string widx = wi && wi[0] ? wi : "/home/zyxciss/moe_warm_btrfs.idx";
                        std::string e3;
                        if (pager.use_warm_pack(wbin.c_str(), widx.c_str(), e3)) {
                            LLAMA_LOG_INFO("%s: moe warm pack ON: %s\n", __func__, wbin.c_str());
                        } else {
                            LLAMA_LOG_WARN("%s: moe warm pack OFF: %s\n", __func__, e3.c_str());
                        }
                    }
                }
                ggml_init_params ip = {
                    ggml_tensor_overhead()*(size_t)(n_layer*4 + 8) + ggml_graph_overhead(),
                    nullptr, /*no_alloc=*/true };
                moe.ctx = ggml_init(ip);
                moe.pool_gate.resize(n_layer); moe.pool_up.resize(n_layer);
                moe.pool_gate_up.resize(n_layer);
                moe.pool_down.resize(n_layer); moe.slot_table.resize(n_layer);

                const ggml_tensor * gt0 = model.get_tensor("blk.0.ffn_gate_exps.weight");
                const ggml_tensor * dt0 = model.get_tensor("blk.0.ffn_down_exps.weight");
                if (!gt0 || !dt0) {
                    LLAMA_LOG_ERROR("%s: expert tensors not found -> moe_direct_io OFF\n", __func__);
                } else {
                // Mixed quants across layers: each layer's pool takes its type
                // (and ne) from its OWN blk.{il} tensors, not blk.0.

                size_t expect = 0;
                bool pool_ok = true;
                // One extra all-zero slot per layer, always armed: any expert
                // without a resident slot must remap here. Mapping it to -1
                // instead makes get_rows read out of bounds and silently
                // corrupts the whole layer.
                moe.sentinel_slot = true;
                for (int il = 0; il < n_layer; il++) {
                    const int s_pl_il = moe_get_layer_slots(il, n_layer);
                    const int64_t n_slots = s_pl_il + 1;
                    char nb_g[128], nb_u[128], nb_d[128];
                    snprintf(nb_g, sizeof nb_g, "blk.%d.ffn_gate_exps.weight", il);
                    snprintf(nb_u, sizeof nb_u, "blk.%d.ffn_up_exps.weight", il);
                    snprintf(nb_d, sizeof nb_d, "blk.%d.ffn_down_exps.weight", il);
                    const ggml_tensor * gl = model.get_tensor(nb_g);
                    const ggml_tensor * ul = model.get_tensor(nb_u);
                    const ggml_tensor * dl = model.get_tensor(nb_d);
                    if (!gl || !ul || !dl) {
                        pool_ok = false;
                        moe_ready = false;
                        break;
                    }
                    moe.pool_gate[il] = ggml_new_tensor_3d(moe.ctx, gl->type,
                            (int64_t) gl->ne[0], (int64_t) gl->ne[1], n_slots);
                    moe.pool_up[il]   = ggml_new_tensor_3d(moe.ctx, ul->type,
                            (int64_t) ul->ne[0], (int64_t) ul->ne[1], n_slots);
                    moe.pool_down[il] = ggml_new_tensor_3d(moe.ctx, dl->type,
                            (int64_t) dl->ne[0], (int64_t) dl->ne[1], n_slots);
                    moe.slot_table[il] = ggml_new_tensor_2d(moe.ctx, GGML_TYPE_I32, 1, n_exp);
                    expect += ggml_nbytes(moe.pool_gate[il]) + ggml_nbytes(moe.pool_up[il])
                            + ggml_nbytes(moe.pool_down[il]) + ggml_nbytes(moe.slot_table[il]);
                }
                if (!pool_ok) {
                    LLAMA_LOG_ERROR("%s: pool incomplete -> moe_direct_io OFF\n", __func__);
                } else {
                ggml_backend_t pool_be = nullptr;
                for (auto & b : backends) {
                    if (b && ggml_backend_get_device(b.get()) &&
                        ggml_backend_dev_type(ggml_backend_get_device(b.get())) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        pool_be = b.get(); break;
                    }
                }
                if (!pool_be) pool_be = backends.empty() ? nullptr : backends[0].get();

                if (pool_be) {
                    moe.buf = ggml_backend_alloc_ctx_tensors(moe.ctx, pool_be);
                }
                if (moe.buf) {
                    moe.buf_bytes = ggml_backend_buffer_get_size(moe.buf);
                    const size_t tol = expect / 20;
                    if (moe.buf_bytes + tol < expect) {
                        LLAMA_LOG_ERROR("%s: moe pool short: got %zu, need %zu -> OFF\n",
                                        __func__, moe.buf_bytes, expect);
                        ggml_backend_buffer_free(moe.buf); moe.buf = nullptr;
                    } else {
                        moe.n_layer = n_layer; moe.n_expert = n_exp;
                        const int total_slots = moe_get_total_slots(n_layer);
                        moe.n_layer = n_layer; moe.n_expert = n_exp;
                        moe.n_used = n_used;  moe.s_per_layer = 104;
                        moe.n_slots = total_slots;
                        moe.slot_of.assign((size_t) n_layer*n_exp, -1);
                        moe.slot_key.assign((size_t) total_slots, -1);
                        moe.slot_freq.assign((size_t) total_slots, 0);
                        moe.lru.init((size_t) total_slots, 1);
                        moe.dec_freq.assign((size_t) n_layer * n_exp, 0);
                        coact.assign((size_t) (n_layer - 1) * n_exp * 4, 0);
                        coact_cnt.assign((size_t) (n_layer - 1) * n_exp * 4, 0);
                        co_hint.assign((size_t) n_layer * n_exp, 0);
                        look_pred.assign((size_t) n_layer * 6, -1);
                        look_fresh.assign((size_t) n_layer, 0);
                        moe.exp_rank.assign((size_t) n_layer * n_exp, 0);
                        moe.step_sel.assign((size_t) n_layer * n_used, -1);
                        moe.step_topk.assign((size_t) n_layer, nullptr);
                        // staging for demand plus lookahead: 2 layers of layer-ahead
                        // plus token-ahead, all potentially unclaimed (65% predictor
                        // miss), plus margin so demand never starves
                        const char * cm = getenv("MOE_CACHE_MIB");
                        const size_t cache_bytes = cm ? (size_t) atoi(cm) * 1048576u : 0u;
                        if (!pager.reserve_staging_split(384, 64)) {
                            LLAMA_LOG_ERROR("%s: staging reserve failed -> moe_direct_io OFF\n", __func__);
                        } else if (cache_bytes && !pager.init_cache(cache_bytes)) {
                            LLAMA_LOG_WARN("%s: Tier 1 host cache unavailable (%zu MiB asked)\n",
                                           __func__, cache_bytes / 1048576);
                        } else {
                            for (int il = 0; il < n_layer; il++) {
                                const int s_pl_il = moe_get_layer_slots(il, n_layer);
                                std::vector<int32_t> tab((size_t) n_exp, s_pl_il);
                                ggml_backend_tensor_set(moe.slot_table[il], tab.data(), 0,
                                                        n_exp*sizeof(int32_t));
                            }
                            if (moe.sentinel_slot) {
                                for (int il = 0; il < moe.n_layer; il++) {
                                    const int s_pl_now = moe_get_layer_slots(il, moe.n_layer);
                                    const size_t gb = ggml_nbytes(moe.pool_gate[il]) / (s_pl_now + 1);
                                    const size_t ub = ggml_nbytes(moe.pool_up[il])   / (s_pl_now + 1);
                                    const size_t db = ggml_nbytes(moe.pool_down[il]) / (s_pl_now + 1);
                                    const size_t sent = std::max(std::max(gb, ub), db);
                                    std::vector<uint8_t> zero(sent, 0);
                                    ggml_backend_tensor_set(moe.pool_gate[il], zero.data(), (size_t) s_pl_now * gb, gb);
                                    ggml_backend_tensor_set(moe.pool_up[il],   zero.data(), (size_t) s_pl_now * ub, ub);
                                    ggml_backend_tensor_set(moe.pool_down[il], zero.data(), (size_t) s_pl_now * db, db);
                                }
                                LLAMA_LOG_INFO("%s: sentinel slots armed per layer (L0-4: %d, L5-42: %d, L43-47: %d)\n",
                                               __func__, moe_get_layer_slots(0, moe.n_layer),
                                               moe_get_layer_slots(5, moe.n_layer),
                                               moe_get_layer_slots(43, moe.n_layer));
                            }
                            moe_ready = true;
                            LLAMA_LOG_INFO("%s: moe pool %zu bytes (%.2f MiB) for %d layers, %d total slots\n",
                                           __func__, moe.buf_bytes, moe.buf_bytes/1048576.0,
                                           n_layer, total_slots);
                            {
                                size_t per = 0;
                                moe.host_gate.assign(n_layer, 0);
                                moe.host_up.assign(n_layer, 0);
                                moe.host_down.assign(n_layer, 0);
                                for (int il = 0; il < n_layer; il++) {
                                    const int s_pl_il = moe_get_layer_slots(il, n_layer);
                                    const size_t gb = ggml_nbytes(moe.pool_gate[il]) / (size_t) (s_pl_il + 1);
                                    const size_t ub = ggml_nbytes(moe.pool_up[il])   / (size_t) (s_pl_il + 1);
                                    const size_t db = ggml_nbytes(moe.pool_down[il]) / (size_t) (s_pl_il + 1);
                                    moe.host_gate[il] = gb; moe.host_up[il] = ub; moe.host_down[il] = db;
                                    per = std::max(per, gb + ub + db);
                                }
                                size_t blobmax = 0;
                                for (int il = 0; il < n_layer; il++) {
                                    for (int e = 0; e < n_exp; e++) {
                                        blobmax = std::max(blobmax, (size_t) pager.slices(il, e)->len);
                                    }
                                }
                                per = std::max(per, blobmax);
                                per = (per + 4095) & ~(size_t) 4095;
                                const char * he = getenv("MOE_HOST_SLOTS");
                                int hslots = (he && atoi(he) > 0) ? atoi(he) : 80;
                                moe.host_slot_bytes = per;
                                moe.host_odirect = false;
                                if (cudaHostAlloc((void **) &moe.host_pool,
                                        (size_t) hslots * per, cudaHostAllocPortable) == cudaSuccess) {
                                    moe.host_nslots = hslots;
                                    moe.host_odirect = (((uintptr_t) moe.host_pool & 4095) == 0);
                                    moe.host_rr = 0;
                                    moe.host_ev.assign((size_t) hslots, nullptr);
                                    for (int i = 0; i < hslots; i++) {
                                        cudaEvent_t ev = nullptr;
                                        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                                        moe.host_ev[(size_t) i] = (void *) ev;
                                    }
                                    LLAMA_LOG_INFO("%s: moe host staging %d slots x %zu bytes (%.2f MiB)\n",
                                                   __func__, hslots, per, (double) hslots * per / 1048576.0);
                                } else {
                                    moe.host_pool = nullptr;
                                    LLAMA_LOG_WARN("%s: moe host pool unavailable, NVMe direct only\n", __func__);
                                }
                                if (moe_doorbell_on(this)) {
                                    int min_slots = INT_MAX;
                                    for (int il = 0; il < n_layer; ++il) {
                                        min_slots = std::min(min_slots, moe_get_layer_slots(il, n_layer));
                                    }
                                    const int stage_slots = std::max(min_slots, n_used) + n_used;
                                    if (!moe_doorbell_stage_init(per, stage_slots)) {
                                        LLAMA_LOG_WARN("%s: doorbell pinned stage unavailable\n", __func__);
                                    }
                                }
                            }
                            moe_seed_hot();
                            if (moe_doorbell_on(this) && pager.pack_mode()) {
                                // tiers are exclusive: free the page-cache copies
                                // of the VRAM set before the warmer runs
                                moe_cache_scan(this, "init", true);
                            }
                            // RouterPredictorV2 measured 1.2% top-16 hit rate on this
                            // model, so it is off by default: it costs 120 MiB of VRAM
                            // that the slot pool can use instead. MOE_PRED=1 re-arms it.
                            const char * pred_on = getenv("MOE_PRED");
                            if (pred_on && atoi(pred_on)) {
                                const char * p_bin = getenv("MOE_PREDICTOR_BIN");
                                std::string pred_bin = p_bin ? p_bin : "/home/zyxciss/llama.cpp/data/router_predictor_v2.bin";
                                if (moe_predictor_init(pred_bin.c_str(), moe.n_layer, 2048, 512, 512)) {
                                    LLAMA_LOG_INFO("%s: RouterPredictorV2 (60M) armed from %s\n", __func__, pred_bin.c_str());
                                }
                            } else {
                                LLAMA_LOG_WARN("%s: RouterPredictorV2 skipped (MOE_PRED=0), +120 MiB for slots\n", __func__);
                            }
                            if (moe_doorbell_on(this)) {
                                if (moe_doorbell_init(moe.n_layer, moe.n_expert, moe.s_per_layer)) {
                                    LLAMA_LOG_INFO("%s: Zero-Split Stream-Ordered Doorbell armed (%d layers, n_splits = 1)\n",
                                                   __func__, moe.n_layer);
                                    start_doorbell_worker(this);
                                }
                            } else {
                                moe_slot_resolve_register_ctx(this);
                            }
                            moe_oracle_start(this);
                            io_run.store(true);
                            for (int i = 0; i < 16; i++) io_thr.emplace_back(moe_io_worker, this);
                            {
                                static int drop = -1;
                                if (drop < 0) {
                                    const char * e = getenv("MOE_DROP_MMAP");
                                    drop = (e && atoi(e) != 0) ? 1 : 0;
                                }
                                model.audit_residency();
                                model.advise_mmap_random();
                                if (drop) {
                                    const size_t dropped = model.drop_mmap_cache();
                                    LLAMA_LOG_INFO("%s: GGUF mmap page cache released (%.2f MiB mapped)\n",
                                                   __func__, (double) dropped / 1048576.0);
                                }
                            }
                        } // staging reserved
                    } // buf ok
                } else {
                    LLAMA_LOG_ERROR("%s: moe pool alloc failed -> moe_direct_io OFF\n", __func__);
                } // pool_ok
                } // outer else (pager ok)
            } // pager init else
        } // moe_direct_io if
        } // moe_direct_io if

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    stop_doorbell_worker();
    if (moe_doorbell_is_active()) {
        moe_doorbell_free();
    }
    moe_oracle_stop(this);
    io_run.store(false);
    io_cv.notify_all();
    for (auto & th : io_thr) if (th.joinable()) th.join();
    io_thr.clear();
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
    if (g_moe_resolve_ctx == this) {
        g_moe_resolve_ctx = nullptr;
    }
    // release the sparse-MoE slot pool (fit-probe contexts must not leak VRAM)
    if (moe.buf) { ggml_backend_buffer_free(moe.buf); moe.buf = nullptr; }
    if (moe.host_pool) {
        for (size_t i = 0; i < moe.host_ev.size(); i++) {
            if (moe.host_ev[i]) cudaEventDestroy((cudaEvent_t) moe.host_ev[i]);
        }
        moe.host_ev.clear();
        cudaFreeHost(moe.host_pool); moe.host_pool = nullptr; moe.host_nslots = 0;
    }
    if (moe.ctx) { ggml_free(moe.ctx); moe.ctx = nullptr; }
    if (moe_predictor_is_ready()) {
        moe_predictor_free();
    }

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.hparams.no_alloc);
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
    //sched_need_reserve = true;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    // prefill->decode edge: a long prefill churns the seeded slots, so
    // restore the hot profile once before decode token 0 begins. pure
    // decode streams (ubatch 1 -> 1) skip this entirely.
    if (moe_ready && moe.last_ntok > 1 && ubatch.n_tokens == 1) {
        // Direction 1's table is static by design: never refresh it mid-run
        static int keep_static = -1;
        if (keep_static < 0) keep_static = getenv("MOE_STATIC_HOT") ? atoi(getenv("MOE_STATIC_HOT")) : 0;
        // the doorbell LRU already adapts to the prefill's routing, and the
        // reseed would stall the first token for seconds; skip it there
        if (!keep_static && !moe_doorbell_on(this)) moe_reseed_decode();
    }
    if (moe_ready) moe.last_ntok = ubatch.n_tokens;
    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched.get());
        if (moe_ready) {
            ggml_backend_sched_set_eval_callback(sched.get(), llama_moe_page_cb, this);
        } else {
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        }

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    // ---- diagnostic timers (MOE_DIAG=1) ------------------------------------
    // T1 = inside graph_compute (the CUDA graph launch + any host wait)
    // T2 = host-side table/copy work (moe_publish + tensor_set), measured in
    //      moe_publish and accumulated into a thread-safe counter
    const bool diag = [] {
        static int v = -1;
        if (v < 0) v = getenv("MOE_DIAG") ? atoi(getenv("MOE_DIAG")) : 0;
        return v != 0;
    }();
    const auto t_gpu0 = std::chrono::high_resolution_clock::now();

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status = %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    {
        static bool seen_tg = false, seen_pp = false;
        static double pf_ms = 0.0, dc_ms = 0.0;
        static long pf_tok = 0, dc_tok = 0;
        static int pf_splits = 0, dc_splits = 0;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_gpu0).count();
        if (ubatch.n_tokens == 1 && !seen_tg) {
            seen_tg = true;
            fprintf(stderr, "%s: first decode splits=%d nodes=%d n_tokens=%u n_ubatch=%u compute=%.2f ms\n",
                __func__, ggml_backend_sched_get_n_splits(sched.get()),
                ggml_graph_n_nodes(res->get_gf()), ubatch.n_tokens, cparams.n_ubatch, ms);
        } else if (ubatch.n_tokens > 1 && !seen_pp) {
            seen_pp = true;
            fprintf(stderr, "%s: first prefill splits=%d nodes=%d n_tokens=%u n_ubatch=%u compute=%.2f ms\n",
                __func__, ggml_backend_sched_get_n_splits(sched.get()),
                ggml_graph_n_nodes(res->get_gf()), ubatch.n_tokens, cparams.n_ubatch, ms);
            {
                std::map<std::string, long> cpu_ops, cuda_ops;
                struct ggml_cgraph * gf = res->get_gf();
                for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                    struct ggml_tensor * t = ggml_graph_node(gf, i);
                    if (!t || !t->op) continue;
                    const char * op = ggml_op_name(t->op);
                    ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), t);
                    const char * bn = be ? ggml_backend_name(be) : "none";
                    const bool is_cpu = bn && (strstr(bn, "CPU") != nullptr);
                    (is_cpu ? cpu_ops : cuda_ops)[op ? op : "?"]++;
                }
                for (const auto & kv : cpu_ops) {
                    fprintf(stderr, "[moe_be] prefill CPU op=%s n=%ld\n", kv.first.c_str(), kv.second);
                }
                fprintf(stderr, "[moe_be] prefill cuda ops=%zu cpu ops=%zu\n", cuda_ops.size(), cpu_ops.size());
            }
        }
        if (ubatch.n_tokens > 1) {
            pf_ms += ms; pf_tok += ubatch.n_tokens;
            pf_splits = ggml_backend_sched_get_n_splits(sched.get());
        } else {
            dc_ms += ms; dc_tok += ubatch.n_tokens;
            dc_splits = ggml_backend_sched_get_n_splits(sched.get());
        }
        if (pf_tok + dc_tok >= 2048) {
            fprintf(stderr, "[moe_ub] prefill tok=%ld ms=%.0f (%.1f ms/tok) splits=%d | decode tok=%ld ms=%.0f (%.1f ms/tok) splits=%d\n",
                pf_tok, pf_ms, pf_ms / std::max(1L, pf_tok), pf_splits,
                dc_tok, dc_ms, dc_ms / std::max(1L, dc_tok), dc_splits);
            pf_ms = 0.0; dc_ms = 0.0; pf_tok = 0; dc_tok = 0;
        }
    }

    if (moe_ready && ubatch.n_tokens == 1) {
        moe_predict_and_prefetch();
        if (!moe_doorbell_on(this)) {
            moe_post_refresh();
        }
    }

    if (diag) {
        const double t1 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_gpu0).count();
        static double s_t1 = 0.0, s_t2 = 0.0;
        static long s_n = 0;
        s_t1 += t1;
        s_t2 += moe_diag_publish_ms;
        moe_diag_publish_ms = 0.0;
        if (++s_n == 20) {
            LLAMA_LOG_INFO("%s: DIAG steps=%ld T1_compute=%.2f ms T2_publish=%.2f ms per step\n",
                           __func__, s_n, s_t1 / s_n, s_t2 / s_n);
            s_t1 = s_t2 = 0.0;
            s_n = 0;
        }
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        n_sampling_nodes += n_nodes;
        if (cparams.n_outputs_max_per_seq > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }

    const uint32_t n_sampling_outputs_max = std::min<uint64_t>(
            std::min(n_tokens, cparams.n_outputs_max),
            (uint64_t) cparams.n_seq_max * cparams.n_outputs_max_per_seq);

    res += n_sampling_nodes;
    if (n_sampling_outputs_max > 1) {
        res += (n_sampling_outputs_max - 1) * n_sampling_nodes_max;
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.prec_policy =*/ &model.prec_policy,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.moe_gate     =*/ moe_ready ? (const ggml_tensor * const *) moe.pool_gate.data()     : nullptr,
        /*.moe_gate_up  =*/ moe_ready ? (const ggml_tensor * const *) moe.pool_gate_up.data() : nullptr,
        /*.moe_up       =*/ moe_ready ? (const ggml_tensor * const *) moe.pool_up.data()     : nullptr,
        /*.moe_down     =*/ moe_ready ? (const ggml_tensor * const *) moe.pool_down.data()  : nullptr,
        /*.moe_table    =*/ moe_ready ? (const ggml_tensor * const *) moe.slot_table.data() : nullptr,
        /*.moe_n_layer  =*/ moe_ready ? moe.n_layer : 0,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }
        if (moe_ready && strcmp(name, "ffn_moe_topk") == 0 && il >= 0 && il < moe.n_layer) {
            const_cast<llama_context*>(this)->moe.step_topk[(size_t) il] = cur;
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.moe_direct_io              =*/ false,
        /*.moe_slots                  =*/ 4096,
        /*.moe_gguf_path              =*/ nullptr,
        /*.moe_pack_dir               =*/ nullptr,
        /*.moe_chunk_max              =*/ 64,
        /*.moe_profile_path           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    if (ctx && ctx->get_cparams().moe_direct_io) {
        return 1;
    }
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}

// ----------------------------------------------------------------------------
// Ensure expert `e` of layer `il` is resident in that layer's pool and publish
// the mapping. `pinned[local]` marks slots that must not be evicted (experts
// still needed by the current batch); may be nullptr. Returns the layer-local
// slot, or -1 on failure.
int llama_context::moe_page_pinned(int il, int e, const bool * pinned) {
    if (!moe_ready || il < 0 || il >= moe.n_layer || e < 0 || e >= moe.n_expert) {
        return -1;
    }
    const int n_exp = moe.n_expert;
    const int key = il * n_exp + e;
    const int cur = moe.slot_of[key];
    moe_adapt_note(il, e);   // every demand feeds the decay histogram
    if (cur >= 0) {
        moe.n_hit++;
        if (moe.count_freq) { moe.slot_freq[(size_t) cur]++; moe.lru.hit((size_t) cur); }
        return cur;
    }
    const int s_pl = moe_get_layer_slots(il, moe.n_layer);
    const int base = moe_get_layer_base_slot(il, moe.n_layer);
    int slot = -1;
    {
        std::vector<char> lp;
        const char * lpinned = nullptr;
        if (pinned) {
            lp.assign((size_t) s_pl, 0);
            for (int t = 0; t < s_pl; t++) lp[(size_t) t] = pinned[t] ? 1 : 0;
            lpinned = lp.data();
        }
        const size_t v = moe.lru.victim(lpinned, (size_t) base, (size_t) s_pl);
        if (v != (size_t) -1) slot = (int) v;
    }
    if (slot < 0) {
        moe.n_page_fail++;
        return -1;
    }

    // DMA into the expert's OWN slot, not the pool base
    const int local = slot - base;
    const size_t g_slot = ggml_nbytes(moe.pool_gate[il]) / (size_t) (s_pl + 1);
    const size_t u_slot = ggml_nbytes(moe.pool_up[il])   / (size_t) (s_pl + 1);
    const size_t d_slot = ggml_nbytes(moe.pool_down[il]) / (size_t) (s_pl + 1);
    if (!pager.fetch(il, e,
                     (char *) moe.pool_gate[il]->data + (size_t) local * g_slot, g_slot,
                     (char *) moe.pool_up[il]->data   + (size_t) local * u_slot, u_slot,
                     (char *) moe.pool_down[il]->data + (size_t) local * d_slot, d_slot)) {
        moe.n_page_fail++;
        return -1;
    }
    if (moe.count_freq) { moe.slot_freq[(size_t) slot]++; moe.lru.miss((size_t) slot); }
    else moe.lru.prefill_miss((size_t) slot);
    moe_publish(il, e, slot);
    moe.n_miss++;
    return slot;
}

int llama_context::moe_page(int il, int e) {
    return moe_page_pinned(il, e, nullptr);
}

// Page a whole top-k set for one layer, pinning residents first so a later
// miss in the same batch can never evict an expert the batch still needs.
// Corrupt ids (< 0 or >= n_expert) are skipped and left unmapped.
void llama_context::moe_page_batch(int il, const int32_t * sel, int n) {
    if (!moe_ready || il < 0 || il >= moe.n_layer) return;
    const int n_exp = moe.n_expert;
    const int s_pl = moe_get_layer_slots(il, moe.n_layer);
    const int base = moe_get_layer_base_slot(il, moe.n_layer);
    std::vector<char> pinned((size_t) s_pl, 0);
    for (int k = 0; k < n; k++) {
        const int32_t e = sel[k];
        if (e < 0 || e >= n_exp) continue;
        const int cur = moe.slot_of[(size_t) il * n_exp + e];
        if (cur >= 0 && cur >= base && cur < base + s_pl) pinned[(size_t) cur - (size_t) base] = 1;
    }
    for (int k = 0; k < n; k++) {
        const int32_t e = sel[k];
        if (e < 0 || e >= n_exp) continue;
        const int got = moe_page_pinned(il, e, (const bool *) pinned.data());
        // a later miss in this same batch must not evict what an earlier miss
        // just paged: pinned was built before any of them ran
        if (got >= 0) {
            const int local = got - base;
            pinned[(size_t) local] = 1;
        }
    }
}

// Sorted variant for prefill micro-chunks and decode: dedup the batch, count
// resident hits for the zero-I/O fast path, then page misses in ascending
// expert order. Ascending order is monotonic in experts.bin, so the pager
// issues sequential reads at full bandwidth instead of random seeks.
bool llama_context::moe_page_host_batch(int il, const int32_t * uniq, size_t nunq,
        std::vector<char> & pinned, int base, int s_pl, void * stream) {
    if (!moe_ready || !moe.host_pool || moe.host_nslots <= 0) return false;
    if (!pager.pack_mode() || pager.pack_fd_buf() < 0) return false;
    if (nunq > (size_t) moe.host_nslots) return false;
    const int n_exp = moe.n_expert;
    cudaStream_t st = (cudaStream_t) stream;
    const size_t gb = moe.host_gate[il], ub = moe.host_up[il], db = moe.host_down[il];
    const size_t stride = moe.host_slot_bytes;
    std::vector<moe_io_job> & jobs = io_jobs_scratch;
    jobs.clear();
    jobs.reserve(nunq);
    for (size_t i = 0; i < nunq; i++) {
        const int32_t e = uniq[i];
        if (e < 0 || e >= n_exp) continue;
        int vslot = -1;
        {
            const size_t v = moe.lru.victim(pinned.data(), (size_t) base, (size_t) s_pl);
            if (v != (size_t) -1) vslot = (int) v;
        }
        if (vslot < 0) continue;
        pinned[(size_t) vslot - (size_t) base] = 1;
        const int rslot = moe.host_ring % moe.host_nslots;
        moe.host_ring++;
        const moe_pager::moe_pack_loc L = moe.host_odirect ? pager.pack_loc(il, e) : pager.pack_loc_buf(il, e);
        if (L.fd < 0) {
            pinned[(size_t) vslot - (size_t) base] = 0;
            continue;
        }
        jobs.push_back({e, vslot, rslot, L.fd, L.off, L.len, L.pre, L.fd == pager.warm_fd()});
        if (L.fd == pager.warm_fd()) moe.n_warm_hit++; else moe.n_pack_hit++;
    }
    if (io_thr.empty()) {
        for (size_t i = 0; i < jobs.size(); i++) {
            char * dst = moe.host_pool + (size_t) jobs[i].rslot * stride;
            if (cudaEventSynchronize((cudaEvent_t) moe.host_ev[(size_t) jobs[i].rslot]) != cudaSuccess) {
                for (size_t k = 0; k < jobs.size(); k++) pinned[(size_t) (jobs[k].vslot - base)] = 0;
                return false;
            }
            size_t left = jobs[i].len, done = 0;
            while (left > 0) {
                ssize_t r = pread(jobs[i].fd, dst + done, left, (off_t) (jobs[i].off + done));
                if (r <= 0) break;
                done += (size_t) r;
                left -= (size_t) r;
            }
            if (done != jobs[i].len) {
                for (size_t k = 0; k < jobs.size(); k++) pinned[(size_t) (jobs[k].vslot - base)] = 0;
                return false;
            }
        }
    } else {
        for (size_t i = 0; i < jobs.size(); i++) {
            if (cudaEventSynchronize((cudaEvent_t) moe.host_ev[(size_t) jobs[i].rslot]) != cudaSuccess) {
                for (size_t k = 0; k < jobs.size(); k++) pinned[(size_t) (jobs[k].vslot - base)] = 0;
                return false;
            }
        }
        std::unique_lock<std::mutex> lock(io_mtx);
        io_jobs   = jobs;
        io_il     = il;
        io_stride = stride;
        io_fail.store(0);
        io_per    = (jobs.size() + 15) / 16;
        io_nth    = std::min<size_t>(16, jobs.size());
        io_next.store(0);
        io_thr_active = (long) io_thr.size();
        io_finished.store(0);
        io_gen++;
        io_cv.notify_all();
        io_done_cv.wait(lock, [&]() { return io_gen_done.load() >= io_gen; });
        if (io_fail > 0) {
            for (size_t i = 0; i < jobs.size(); i++) pinned[(size_t) (jobs[i].vslot - base)] = 0;
            return false;
        }
    }
    const size_t vg = ggml_nbytes(moe.pool_gate[il]) / (size_t) (s_pl + 1);
    const size_t vu = ggml_nbytes(moe.pool_up[il])   / (size_t) (s_pl + 1);
    const size_t vd = ggml_nbytes(moe.pool_down[il]) / (size_t) (s_pl + 1);
    for (size_t i = 0; i < jobs.size(); i++) {
        const int local = jobs[i].vslot - base;
        const char * src = moe.host_pool + (size_t) jobs[i].rslot * stride;
        const size_t pre = jobs[i].pre;
        cudaMemcpyAsync((char *) moe.pool_gate[il]->data + (size_t) local * vg, src + pre,
                        std::min(vg, gb), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync((char *) moe.pool_up[il]->data + (size_t) local * vu, src + pre + gb,
                        std::min(vu, ub), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync((char *) moe.pool_down[il]->data + (size_t) local * vd, src + pre + gb + ub,
                        std::min(vd, db), cudaMemcpyHostToDevice, st);
        cudaEventRecord((cudaEvent_t) moe.host_ev[(size_t) jobs[i].rslot], st);
        const int key = il * n_exp + jobs[i].e;
        const int old = moe.slot_key[(size_t) jobs[i].vslot];
        if (old >= 0 && old != key) moe.slot_of[(size_t) old] = -1;
        const int prev = moe.slot_of[(size_t) key];
        if (prev >= 0 && prev != jobs[i].vslot) moe.slot_key[(size_t) prev] = -1;
        moe.slot_key[(size_t) jobs[i].vslot] = key;
        moe.slot_of[(size_t) key] = jobs[i].vslot;
        if (moe.count_freq) { moe.slot_freq[(size_t) jobs[i].vslot]++; moe.lru.miss((size_t) jobs[i].vslot); }
        else moe.lru.prefill_miss((size_t) jobs[i].vslot);
        moe.n_miss++;
        moe_adapt_note(il, jobs[i].e);
    }
    return true;
}

void llama_context::moe_page_batch_sorted(int il, const int32_t * sel, int n) {
    if (!moe_ready || il < 0 || il >= moe.n_layer) return;
    const int n_exp = moe.n_expert;
    const int s_pl = moe_get_layer_slots(il, moe.n_layer);
    const int base = moe_get_layer_base_slot(il, moe.n_layer);
    std::vector<char> seen((size_t) n_exp, 0);
    std::vector<int32_t> uniq;
    uniq.reserve((size_t) n > 64 ? 64 : (size_t) n);
    for (int k = 0; k < n; k++) {
        const int32_t e = sel[k];
        if (e < 0 || e >= n_exp) continue;
        if (seen[(size_t) e]) continue;
        seen[(size_t) e] = 1;
        uniq.push_back(e);
    }
    std::vector<char> pinned((size_t) s_pl, 0);
    for (size_t k = 0; k < uniq.size(); k++) {
        const int cur = moe.slot_of[(size_t) il * n_exp + uniq[k]];
        if (cur >= 0 && cur >= base && cur < base + s_pl) {
            pinned[(size_t) cur - (size_t) base] = 1;
            moe.prefill_resident++;
        }
    }
    std::sort(uniq.begin(), uniq.end());
    std::vector<moe_batch_target> batch_targets;
    std::vector<int> target_slots;
    batch_targets.reserve(uniq.size());
    target_slots.reserve(uniq.size());

    const size_t g_slot = ggml_nbytes(moe.pool_gate[il]) / (size_t) (s_pl + 1);
    const size_t u_slot = ggml_nbytes(moe.pool_up[il])   / (size_t) (s_pl + 1);
    const size_t d_slot = ggml_nbytes(moe.pool_down[il]) / (size_t) (s_pl + 1);

    for (size_t k = 0; k < uniq.size(); k++) {
        const int32_t e = uniq[k];
        if (moe.slot_of[(size_t) il * n_exp + e] >= 0) continue;
        const size_t v = moe.lru.victim((const char *) pinned.data(), (size_t) base, (size_t) s_pl);
        if (v == (size_t) -1) continue;
        int slot = (int) v;
        pinned[(size_t) slot - (size_t) base] = 1;

        const int local = slot - base;
        moe_batch_target tgt;
        tgt.expert = e;
        tgt.dst_gate = (char *) moe.pool_gate[il]->data + (size_t) local * g_slot;
        tgt.gate_bytes = g_slot;
        tgt.dst_up   = (char *) moe.pool_up[il]->data   + (size_t) local * u_slot;
        tgt.up_bytes = u_slot;
        tgt.dst_down = (char *) moe.pool_down[il]->data + (size_t) local * d_slot;
        tgt.down_bytes = d_slot;
        batch_targets.push_back(tgt);
        target_slots.push_back(slot);
    }

    if (!batch_targets.empty()) {
        for (size_t i = 0; i < batch_targets.size(); ++i) {
            int slot = target_slots[i];
            int e = batch_targets[i].expert;
            if (!this->pager.fetch_direct_io(il, e,
                (char *) moe.pool_gate[il]->data + (size_t) slot * g_slot, g_slot,
                (char *) moe.pool_up[il]->data   + (size_t) slot * u_slot, u_slot,
                (char *) moe.pool_down[il]->data + (size_t) slot * d_slot, d_slot,
                nullptr, slot)) {
                moe.prefill_miss++;
                continue;
            }
            moe.prefill_miss++;
            if (moe.count_freq) {
                moe.slot_freq[(size_t) slot]++;
                moe.lru.miss((size_t) slot);
            } else {
                moe.lru.prefill_miss((size_t) slot);
            }
            moe_publish(il, e, slot);
            moe.n_miss++;
        }
        std::vector<int32_t> tab((size_t) n_exp, s_pl);
        for (int e = 0; e < n_exp; ++e) {
            int s = moe.slot_of[(size_t) il * n_exp + e];
            if (s >= base && s < base + s_pl) {
                tab[e] = s - base;
            }
        }
        ggml_backend_tensor_set(moe.slot_table[il], tab.data(), 0, n_exp * sizeof(int32_t));
    }
}

// Seed every layer's slots with the offline hot profile so the first decode
// step starts warm. Runs synchronously inside pool setup; each page is a
// normal demand fetch through the pager, failures tolerated (slot stays -1).
// Round-robin over layers (top-1 of every layer, then top-2, ...) so a tight
// budget still covers all layers uniformly: ~2000 pages in 4s give top-40
// per layer ≈ 70% cold-hit coverage (trace-measured).
void llama_context::moe_hot_save() {
    if (!moe_ready) return;
    const char * learned = getenv("MOE_HOT_PROFILE");
    const char * lpath = (learned && learned[0]) ? learned : "data/moe_hot_profile.bin";
    const int n_exp = moe.n_expert;
    const int per = 240;
    FILE * f = fopen(lpath, "wb");
    if (!f) return;
    uint32_t nl = (uint32_t) moe.n_layer, nt = (uint32_t) per;
    fwrite(&nl, 4, 1, f);
    fwrite(&nt, 4, 1, f);
    std::vector<int> order((size_t) n_exp);
    for (int il = 0; il < moe.n_layer; il++) {
        for (int e = 0; e < n_exp; e++) order[(size_t) e] = e;
        const uint32_t * d = moe.dec_freq.data() + (size_t) il * n_exp;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return d[a] > d[b];
        });
        if (moe.exp_rank.size() != (size_t) moe.n_layer * n_exp) {
            moe.exp_rank.assign((size_t) moe.n_layer * n_exp, 0);
        }
        uint16_t * rk = moe.exp_rank.data() + (size_t) il * n_exp;
        for (int k = 0; k < n_exp; k++) rk[order[(size_t) k]] = (uint16_t) k;
        for (int k = 0; k < per; k++) {
            uint16_t v = (uint16_t) order[(size_t) k];
            fwrite(&v, 2, 1, f);
        }
    }
    fclose(f);
}

void llama_context::moe_seed_hot() {
    if (!moe_ready) return;
    // file profile (--moe-profile) wins; else learned profile; else built-in.
    // format: u32 n_layer, u32 per_layer, then u16 expert ids in rank order.
    std::vector<uint16_t> file_hot;
    int file_nt = 0;
    const char * learned = getenv("MOE_HOT_PROFILE");
    const char * lpath = (learned && learned[0]) ? learned : "/home/zyxciss/llama.cpp/data/moe_hot_profile.bin";
    FILE * test_f = fopen(lpath, "rb");
    if (test_f) { fclose(test_f); }
    else { lpath = "/home/zyxciss/fluxaeros-data/data/moe_hot_profile.bin"; }
    if ((!cparams.moe_profile_path || !cparams.moe_profile_path[0])) {
        FILE * lf = fopen(lpath, "rb");
        if (lf) {
            uint32_t nl = 0, nt = 0;
            bool ok = fread(&nl, 4, 1, lf) == 1 && fread(&nt, 4, 1, lf) == 1 &&
                      nl == (uint32_t) moe.n_layer && nt > 0 && nt <= (uint32_t) moe.n_expert;
            if (ok) {
                file_hot.resize((size_t) nl * nt);
                ok = fread(file_hot.data(), 2, file_hot.size(), lf) == file_hot.size();
            }
            fclose(lf);
            if (ok) {
                file_nt = (int) nt;
                LLAMA_LOG_INFO("%s: using learned profile %s\n", __func__, lpath);
            }
        }
    }
    if (cparams.moe_profile_path && cparams.moe_profile_path[0]) {
        FILE * f = fopen(cparams.moe_profile_path, "rb");
        if (!f) {
            LLAMA_LOG_WARN("%s: cannot open %s, using built-in profile\n",
                           __func__, cparams.moe_profile_path);
        } else {
            uint32_t nl = 0, nt = 0;
            bool ok = fread(&nl, 4, 1, f) == 1 && fread(&nt, 4, 1, f) == 1 &&
                      nl == (uint32_t) moe.n_layer && nt > 0 && nt <= (uint32_t) moe.n_expert;
            if (ok) {
                file_hot.resize((size_t) nl * nt);
                ok = fread(file_hot.data(), 2, file_hot.size(), f) == file_hot.size();
            }
            fclose(f);
            if (ok) { file_nt = (int) nt; }
            else LLAMA_LOG_WARN("%s: bad profile %s, using built-in\n", __func__, cparams.moe_profile_path);
        }
    }
#include "moe_hot_profile.h"
    static_assert(MOE_HOT_NLAYER == 48, "hot profile layer count");
    const int64_t t0 = ggml_time_us();
    const int64_t budget_us = [] {
        const char * e = getenv("MOE_SEED_BUDGET_MS");
        int64_t ms = e && e[0] ? atoi(e) : 4000;
        // Direction 1 needs the full profile: a partial seed leaves cold
        // experts permanently mapped to the (zero) sentinel, which silently
        // degrades every token. Give it the 15s full-seed budget.
        if (getenv("MOE_STATIC_HOT") && atoi(getenv("MOE_STATIC_HOT"))) {
            if (!(e && e[0]) || ms < 8000) ms = 15000;
        }
        return ms * 1000;
    }();
    // the profile carries 72 hot experts per layer, but the pool may hold fewer
    // slots: clamp to what exists, or slot base+k runs past the layer and the
    // DMA below writes outside the tensor.
    const int max_slots = 240;
    int seeded = 0;
    const int total = moe_get_total_slots(moe.n_layer);
    // seeded experts are a prior, not recent use: rank them below any real
    // access so the LRU drops the coldest seed first
    moe.lru.tick = (uint64_t) max_slots + 8;
    for (int k = 0; k < max_slots; k++) {
        for (int il = 0; il < moe.n_layer; il++) {
            const int s_pl_il = moe_get_layer_slots(il, moe.n_layer);
            if (k >= s_pl_il) continue;
            if (ggml_time_us() - t0 > budget_us) {
                LLAMA_LOG_WARN("%s: budget hit at round %d, %d seeded (profile: %s)\n", __func__, k, seeded,
                               file_nt > 0 ? cparams.moe_profile_path : "built-in");
                return;
            }
            int32_t e = -1;
            if (file_nt > 0 && k < file_nt) {
                e = file_hot[(size_t) il * file_nt + k];
            } else if (k < MOE_HOT_PER_LAYER) {
                e = moe_hot_experts[il][k];
            } else {
                e = (k * 7 + il * 13) % moe.n_expert;
            }
            if (e < 0 || e >= moe.n_expert) continue;
            const int base = moe_get_layer_base_slot(il, moe.n_layer);
            const int slot = base + k;
            const int key  = il * moe.n_expert + e;
            if (moe.slot_of[key] >= 0) continue;

            const int local = slot - base;
            const size_t g_slot = ggml_nbytes(moe.pool_gate[il]) / (size_t) (s_pl_il + 1);
            const size_t u_slot = ggml_nbytes(moe.pool_up[il])   / (size_t) (s_pl_il + 1);
            const size_t d_slot = ggml_nbytes(moe.pool_down[il]) / (size_t) (s_pl_il + 1);
            if (!pager.fetch(il, e,
                    (char *) moe.pool_gate[il]->data + (size_t) local * g_slot, g_slot,
                    (char *) moe.pool_up[il]->data   + (size_t) local * u_slot, u_slot,
                    (char *) moe.pool_down[il]->data + (size_t) local * d_slot, d_slot)) {
                continue;
            }
            moe_publish(il, e, slot);
            moe.lru.last[(size_t) slot] = (uint64_t) (max_slots - k) + 1;
            seeded++;
        }
    }
    if (moe.sentinel_slot) {
        for (int il = 0; il < moe.n_layer; il++) {
            const int s_pl_il = moe_get_layer_slots(il, moe.n_layer);
            const int base = moe_get_layer_base_slot(il, moe.n_layer);
            std::vector<int32_t> tab((size_t) moe.n_expert, s_pl_il);
            for (int s = 0; s < s_pl_il; s++) {
                const int k = moe.slot_key[(size_t) base + s];
                if (k >= 0) tab[(size_t) (k - il * moe.n_expert)] = s;
            }
            ggml_backend_tensor_set(moe.slot_table[il], tab.data(), 0,
                                    moe.n_expert * (int) sizeof(int32_t));
        }
        LLAMA_LOG_INFO("%s: Arrival-Slack static table published: %d hot experts seeded\n",
                       __func__, seeded);
    }
    LLAMA_LOG_INFO("%s: loaded %d layers x %d hot experts from %s, seeded %d/%d in %.2fs\n",
                   __func__, moe.n_layer, max_slots,
                   file_nt > 0 ? cparams.moe_profile_path : "built-in",
                   seeded, total, (ggml_time_us() - t0) / 1000000.0);
}

bool llama_context::moe_adapt_on() const {
    static int v = -1;
    if (v < 0) v = getenv("MOE_ADAPT") ? atoi(getenv("MOE_ADAPT")) : 0;
    return moe_ready && v != 0;
}

void llama_context::moe_adapt_note(int il, int e) {
    if (!moe_adapt_on()) return;
    if (il < 0 || il >= moe.n_layer || e < 0 || e >= moe.n_expert) return;
    const size_t k = (size_t) il * moe.n_expert + e;
    if (moe.adapt_decay.empty()) {
        moe.adapt_decay.assign((size_t) moe.n_layer * moe.n_expert, 0.0f);
        moe.adapt_last.assign((size_t) moe.n_layer * moe.n_expert, -1);
        moe.adapt_pend_e.assign((size_t) moe.n_layer * moe.n_expert, -1);
    }
    moe.adapt_decay[k] += 1.0f;
    moe.adapt_last[k] = (int32_t) moe.adapt_pass;
}

// Decay the demand histogram, then swap the coldest resident experts out of
// VRAM for the hottest non-resident ones. Runs once every MOE_ADAPT_EVERY
// decode steps, from the LAST layer's topk callback, at which point every
// layer's routing for this step is already in moe.step_sel.
//
// Reads are issued through the pager's async prefetch path and only DMA'd into
// the victim slot on the next pass, so the GPU stream never waits on the SSD.
// Page every layer's experts for the step that just finished and republish the
// tables, entirely between graph launches. The split-per-layer path exists only
// so the host can rewrite a table mid-graph; doing it after the graph lets the
// whole decode run as one unbroken CUDA graph, which is worth ~30 tok/s.
void llama_context::moe_post_refresh() {
    static int on = -1;
    if (on < 0) on = getenv("MOE_POST_REFRESH") ? atoi(getenv("MOE_POST_REFRESH")) : 1;
    if (!on || !moe_ready) return;

    const int n_exp  = moe.n_expert;
    const int n_used = moe.n_used;

    for (int il = 0; il < moe.n_layer; il++) {
        ggml_tensor * t = moe.step_topk[(size_t) il];
        if (!t) continue;
        std::vector<int32_t> ids((size_t) n_used);
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
        moe_page_batch(il, ids.data(), n_used);
        // remember this step's routing: the next one is prefetched from it, so a
        // miss costs a concurrent read instead of a stall in front of the graph
        for (int k = 0; k < n_used; k++) {
            moe.step_sel[(size_t) il * n_used + k] = ids[k];
        }
    }

    // Queue reads for what the previous step routed to. routing is independent
    // across layers, so only a layer's own previous ids predict it, and the
    // staging arena is far smaller than 48 layers x n_used: without a budget
    // these reads occupy every stage and the next step's demand fetch cannot
    // get one at all.
    int budget = std::max(0, pager.n_free_staging() - 2 * n_used);
    for (int il = 0; il < moe.n_layer && budget > 0; il++) {
        for (int k = 0; k < n_used && budget > 0; k++) {
            const int32_t e = moe.step_sel[(size_t) il * n_used + k];
            if (e < 0 || e >= n_exp) continue;
            if (moe.slot_of[(size_t) il * n_exp + e] >= 0) continue;
            if (pager.prefetch(il, e)) budget--;
        }
    }
}

void llama_context::moe_predict_and_prefetch() {
    if (!moe_ready || !moe_predictor_is_ready()) return;

    const int n_layer = moe.n_layer;
    const int n_exp   = moe.n_expert;
    const int n_used  = moe.n_used;

    // Rolling history buffer [4, n_layer, n_used]
    static std::vector<uint16_t> history(4 * 48 * 10, 0);

    // Shift previous 3 steps: [0..2] <- [1..3]
    memmove(history.data(), history.data() + n_layer * n_used, 3 * n_layer * n_used * sizeof(uint16_t));

    // Append newly completed decode step's routing into slice 3
    for (int il = 0; il < n_layer; il++) {
        for (int k = 0; k < n_used; k++) {
            const int32_t e = moe.step_sel[(size_t) il * n_used + k];
            history[(3 * n_layer + il) * n_used + k] = (e >= 0 && e < n_exp) ? (uint16_t) e : 0;
        }
    }

    static std::vector<uint16_t> cold_top16(n_layer * 16, 0);
    static float s_lat_total = 0.0f;
    static long  s_pred_steps = 0;
    static long  s_pf_enqueued = 0;

    // Pass moe.slot_of.data() to mask out resident experts and return pure COLD candidates
    const float lat_ms = moe_predictor_predict(history.data(), cold_top16.data(), moe.slot_of.data());
    s_lat_total += lat_ms;
    s_pred_steps++;

    if (moe.pred16.size() != (size_t) n_layer * 16) moe.pred16.resize((size_t) n_layer * 16);
    memcpy(moe.pred16.data(), cold_top16.data(), moe.pred16.size() * sizeof(uint16_t));

    // The predictor only hits 1.2% of the routed experts, so feeding the
    // read-ahead pool from it just burns NVMe bandwidth. It stays off unless
    // MOE_RA_FROM_PRED explicitly asks for it.
    const bool ra_from_pred = getenv("MOE_RA_FROM_PRED") && atoi(getenv("MOE_RA_FROM_PRED"));
    const int max_cold_per_layer = 8;
    int step_enqueued = 0;
    if (ra_from_pred) {
        std::lock_guard<std::mutex> lock(or_mtx);
        for (int il = 0; il < n_layer && or_q.size() < 3072; il++) {
            for (int k = 0; k < max_cold_per_layer && or_q.size() < 3072; k++) {
                const int32_t e = cold_top16[(size_t) il * 16 + k];
                if (e < 0 || e >= n_exp) continue;
                if (moe.slot_of[(size_t) il * n_exp + e] >= 0) continue;
                or_q.push_back(((uint32_t) il << 16) | (uint32_t) e);
                step_enqueued++;
            }
        }
        s_pf_enqueued += step_enqueued;
    }

    if (s_pred_steps == 1 || (s_pred_steps % 25 == 0)) {
        const double acc = moe.pred_fresh > 0 ? 100.0 * (double) moe.pred_hit /
                (double) (moe.pred_fresh * n_used) : 0.0;
        fprintf(stderr, "[moe_predictor] step=%ld lat=%.4f ms (mean=%.4f ms), enq=%d total_enq=%ld "
                "acc=%.1f%% ra_done=%ld\n",
            s_pred_steps, lat_ms, s_lat_total / s_pred_steps, step_enqueued, s_pf_enqueued,
            acc, moe.ra_done.load());
        fflush(stderr);
    }
}

void llama_context::moe_adapt_run() {
    if (!moe_adapt_on()) return;
    const int n_exp = moe.n_expert;
    const int s_pl  = moe.s_per_layer;

    moe.adapt_pass++;
    if (moe.adapt_decay.empty()) return;

    const int every = [] { const char * e = getenv("MOE_ADAPT_EVERY"); return (e && atoi(e) > 0) ? atoi(e) : 4; }();
    if (moe.adapt_pass % every) return;

    // decay: a demand 4 passes ago is worth ~2^-2 of a fresh one at 0.8
    for (auto & d : moe.adapt_decay) d *= 0.8f;

    // DMA the previous pass's candidates into their victim slots
    for (int il = 0; il < moe.n_layer; il++) {
        for (int e = 0; e < n_exp; e++) {
            const int slot = moe.adapt_pend_e[(size_t) il * n_exp + e];
            if (slot < 0) continue;
            moe.adapt_pend_e[(size_t) il * n_exp + e] = -1;
            const int base  = il * s_pl;                 // adapt_pend_e holds a
            const int local = slot - base;                // GLOBAL slot index
            if (local < 0 || local >= s_pl) continue;
            const size_t g_slot = ggml_nbytes(moe.pool_gate[il]) / (size_t) std::max(1, moe.n_slots);
            const size_t u_slot = ggml_nbytes(moe.pool_up[il])   / (size_t) std::max(1, moe.n_slots);
            const size_t d_slot = ggml_nbytes(moe.pool_down[il]) / (size_t) std::max(1, moe.n_slots);
            if (!pager.commit(il, e,
                    (char *) moe.pool_gate[il]->data + (size_t) local * g_slot, g_slot,
                    (char *) moe.pool_up[il]->data   + (size_t) local * u_slot, u_slot,
                    (char *) moe.pool_down[il]->data + (size_t) local * d_slot, d_slot)) {
                moe.adapt_cold++;         // recompute next pass
                continue;
            }
            moe.adapt_bytes += (long)(g_slot + u_slot + d_slot);
            // drop any slot the demand path gave this expert in the meantime,
            // so one expert never ends up mapped from two slots
            const int stale = moe.slot_of[(size_t) il * n_exp + e];
            if (stale >= 0 && stale != slot) moe.slot_key[(size_t) stale] = -1;
            moe_publish(il, e, slot);
        }
    }

    const int protect = every * 4;   // never evict what a recent batch may still want
    const int n_swap  = 2;           // swaps per layer per pass
    // The swap pass is opt-in: it measurably degrades the reply (repeated
    // tokens), so the default path is the demand-paged pool with the offline
    // hot profile. MOE_ADAPT_SWAP=1 re-enables it for experiments.
    if (![] { const char * e = getenv("MOE_ADAPT_SWAP"); return e && atoi(e) != 0; }()) {
        return;
    }

    for (int il = 0; il < moe.n_layer; il++) {
        const size_t lb = (size_t) il * n_exp;
        const int    base = il * s_pl;
        // residents with their demand score
        std::vector<std::pair<float, int>> res;   // (score, slot)
        for (int s = 0; s < s_pl; s++) {
            const int k = moe.slot_key[(size_t) il * s_pl + s];
            if (k < 0) continue;
            res.emplace_back(moe.adapt_decay[(size_t) k], s);
        }
        if (res.empty()) continue;
        std::sort(res.begin(), res.end());        // coldest first
        // hottest non-residents
        std::vector<std::pair<float, int>> cand;  // (score, expert)
        for (int e = 0; e < n_exp; e++) {
            if (moe.slot_of[lb + e] >= 0) continue;
            cand.emplace_back(moe.adapt_decay[lb + e], e);
        }
        if (cand.empty()) continue;
        const size_t nc = std::min((size_t) n_swap, cand.size());
        std::partial_sort(cand.begin(), cand.begin() + nc, cand.end(),
                          std::greater<std::pair<float, int>>());

        for (size_t i = 0; i < nc && i < (size_t) n_swap; i++) {
            const int victim_slot = res[i].second;
            const int victim_exp  = moe.slot_key[(size_t) il * s_pl + victim_slot] - (int) (lb);
            const float v_score   = res[i].first;
            const float c_score   = cand[i].first;
            // only swap when the candidate is strictly hotter, and never evict
            // an expert this or a very recent batch may still need
            if (victim_exp >= 0 && moe.adapt_last[lb + victim_exp] > (int32_t)(moe.adapt_pass - protect)) continue;
            if (c_score <= v_score) continue;
            // reserve the victim slot until the swap lands: freeing it here would
            // let the demand path reassign it before the commit DMA arrives
            moe.adapt_pend_e[lb + cand[i].second] = base + victim_slot;   // global
            if (pager.prefetch(il, cand[i].second)) moe.adapt_swaps++;
        }
    }
}

// Restore the hot profile once, right when decoding starts after a prefill
// phase. Micro-chunk paging during a long prefill walks moe_rr through every
// layer's slots, so decode token 0 would otherwise start cold. Re-seeding
// replays the profile top-down; experts the prefill already made resident
// are hits (no reread), so this only fetches what prefill evicted.
// Deliberately NOT called for pure decode streams (last_ntok already 1).
void llama_context::moe_reseed_decode() {
    if (!moe_ready) return;
    const int64_t t0 = ggml_time_us();
    moe_seed_hot();
    LLAMA_LOG_INFO("%s: decode-start reseed done in %.2fs (hit=%ld miss=%ld)\n",
                   __func__, (ggml_time_us() - t0) / 1000000.0, moe.n_hit, moe.n_miss);
}

// Queue async reads for experts the next layer(s) will likely need. Runs at
// reads overlap the GPU compute of the layers between il and the target.
// Predictor: previous-step recency per layer (measured overlap 35% at t-1,
// 54% at t-1..t-4). Experts already resident or in flight are skipped, and
// reap() recycles completions.
// Always on: admission control (below) plus the pager's victim-drop keep a
// prefetch burst from ever starving a demand fetch.
void llama_context::moe_prefetch_ahead(int il, const int32_t * sel, int n) {
    // prefetch is always on: with admission control below, a prefetch burst
    // can never starve a demand fetch, and unconverted reads are dropped by
    // the pager's victim-drop when staging fills
    if (!moe_ready || il < 0 || il >= moe.n_layer) return;
    const bool do_issue = true;
    const int n_exp = moe.n_expert;
    const int n_used = moe.n_used;
    // remember this step's routing for the token-ahead predictor below
    // (top-k width may exceed n_used on some layers; keep what fits)
    for (int k = 0; k < n && k < n_used; k++) {
        const int32_t e = sel[k];
        moe.step_sel[(size_t) il * n_used + k] = (e >= 0 && e < n_exp) ? e : -1;
    }
    // admission control: keep at least demand-reserve free stages so a
    // prefetch burst can never starve a demand fetch (which would corrupt text)
    const int reserve = 2 * n_used + 4;
    // layer-ahead: prefetch the PREVIOUS step's top-k for the next layers
    // (temporal recency: measured 35% overlap at t-1, 54% at t-1..t-4).
    // same-step/same-id across layers carries no signal (routing is
    // independent per layer), so this step's own ids are NOT reused here.
    for (int d = 1; d <= 2 && do_issue && pager.n_free_staging() > reserve; d++) {
        const int jl = il + d;
        if (jl >= moe.n_layer) break;
        for (int k = 0; k < n_used; k++) {
            const int32_t e = moe.step_sel[(size_t) jl * n_used + k];
            if (e < 0 || e >= n_exp) continue;
            if (moe.slot_of[(size_t) jl * n_exp + e] >= 0) continue;
            if (!pager.prefetch(jl, e)) break;
            if (pager.n_free_staging() <= reserve) break;
        }
    }
    // token-ahead: at the last layer, warm up layer 0/1 with their own most
    // recent routing (what token t+1's start most likely repeats)
    if (do_issue && il == moe.n_layer - 1) {
        for (int jl = 0; jl < 2 && jl < moe.n_layer; jl++) {
            for (int k = 0; k < n_used; k++) {
                const int32_t e = moe.step_sel[(size_t) jl * n_used + k];
                if (e < 0 || e >= n_exp) continue;
                if (moe.slot_of[(size_t) jl * n_exp + e] >= 0) continue;
                if (!pager.prefetch(jl, e)) break;
                if (pager.n_free_staging() <= reserve) break;
            }
        }
    }
    pager.reap();
}

void llama_context::moe_publish(int il, int e, int slot) {
    if (!moe_ready || il < 0 || il >= moe.n_layer) return;
    const int n_exp = moe.n_expert;
    const int s_pl = moe_get_layer_slots(il, moe.n_layer);
    const int base = moe_get_layer_base_slot(il, moe.n_layer);
    const int key = il * n_exp + e;
    if (slot < base || slot >= base + s_pl) return;
    const int old = moe.slot_key[slot];
    if (old >= 0 && old != key) moe.slot_of[old] = -1;
    // an expert may never be reachable from two slots: the pool would keep
    // stale weights for one of them and the swap pass would pick the same
    // victim twice
    const int prev = moe.slot_of[key];
    if (prev >= 0 && prev != slot) moe.slot_key[(size_t) prev] = -1;
    moe.slot_key[slot] = key;
    moe.slot_of[key]   = slot;
    // remap the whole layer table: expert -> local slot. An expert with no
    // resident slot must map to the all-zero sentinel row: get_rows would
    // otherwise read index -1, which is out of bounds and silently corrupts
    // every layer that still has a cold expert.
    std::vector<int32_t> tab((size_t) n_exp, s_pl);
    for (int s = 0; s < s_pl; s++) {
        const int k = moe.slot_key[(size_t) base + s];
        if (k >= 0) tab[k - il * n_exp] = s;
    }
    if (!moe_doorbell_on(this)) {
        ggml_backend_tensor_set(moe.slot_table[il], tab.data(), 0, n_exp * sizeof(int32_t));
    }
}

const ggml_tensor * llama_context::moe_pool_tensor(const char * which, int il) const {
    if (!moe_ready || il < 0 || il >= moe.n_layer) return nullptr;
    if (!strcmp(which, "gate"))  return moe.pool_gate[il];
    if (!strcmp(which, "up"))    return moe.pool_up[il];
    if (!strcmp(which, "down"))  return moe.pool_down[il];
    if (!strcmp(which, "table")) return moe.slot_table[il];
    return nullptr;
}

// ----------------------------------------------------------------------------
// sparse MoE demand pager: installed as the scheduler eval callback when
// --moe-direct-io is on. On ffn_moe_topk-<il> (data phase) we read the selected
// expert ids, page any that are not resident into the layer's pool via the
// O_DIRECT pager, and update the layer slot table before MUL_MAT_ID runs.
// Chains to any user-supplied cb_eval afterwards.
static bool llama_moe_page_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * ctx = (llama_context *) ud;
    const char * nm = ggml_get_name(t);
    if (!nm) {
        return false;
    }
    // MOE_TRACE_LOGITS: dump the continuous router scores (the tensor feeding
    // argsort_top_k) so a logit-space transition model can be trained offline.
    // ffn_moe_probs[_masked|_biased] are all [n_expert, n_tokens] F32.
    if (!ask && getenv("MOE_TRACE_LOGITS")
        && (!strcmp(nm, "ffn_moe_probs") || !strcmp(nm, "ffn_moe_probs_masked")
            || !strcmp(nm, "ffn_moe_probs_biased"))) {
        static FILE * lf = nullptr;
        static bool  linit = false;
        if (!linit) {
            linit = true;
            const char * p = getenv("MOE_TRACE_LOGITS");
            if (p && p[0]) lf = fopen((std::string(p) + ".logits.f32").c_str(), "wb");
        }
        if (lf) {
            // name is ffn_moe_probs[_masked|_biased]-<il>
            int il = atoi(strrchr(nm, '-') + 1);
            std::vector<float> buf((size_t) t->ne[0] * t->ne[1]);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            int32_t h = il;
            fwrite(&h, sizeof(h), 1, lf);
            fwrite(buf.data(), sizeof(float), buf.size(), lf);
            fflush(lf);
        }
    }
    if (!ctx || !ctx->moe_enabled()) {
        return false;
    }

    if (strncmp(nm, "ffn_moe_topk-", 13) != 0) {
        return false;
    }
    int il = atoi(nm + 13);
    if (il < 0 || il >= ctx->moe.n_layer) {
        return false;
    }
    if (moe_doorbell_on(ctx)) {
        // The resolve kernel runs on the compute stream and pages cold experts
        // through the host doorbell. No graph split is needed for any batch.
        return false;
    }
    if (ask) {
        static int mode = -1;
        if (mode < 0) {
            if (getenv("MOE_NOSPLIT") && atoi(getenv("MOE_NOSPLIT"))) mode = 2;
            else if (getenv("MOE_POST_REFRESH") && atoi(getenv("MOE_POST_REFRESH"))) mode = 4;
            else if (getenv("MOE_STATIC_HOT") && atoi(getenv("MOE_STATIC_HOT"))) mode = 3;
            else mode = 0;
        }
        if (mode == 2) return false; // n_splits = 1
        if (mode == 4 && t->ne[1] == 1) return false;
        if (mode == 3) return false;
        return true; // Mode 0
    }
    // data phase: page the selected experts (I32 [n_used, n_tokens]).
    // the graph only reads the pool when the slot_ids remap node exists;
    // detect it by name so full-tensor prefill never pays paging stalls.
    // (slot remap engages for decode and for micro-chunks <= 8 tokens.)
    {
        static int m2 = -1;
        if (t->ne[1] == 1) {
            return true; // Decode: page experts on host during graph build so GPU kernel finds them resident → fast path → no doorbell → no deadlock
        }
        if (m2 < 0) {
            if (getenv("MOE_STATIC_HOT") && atoi(getenv("MOE_STATIC_HOT"))) m2 = 3;
            else if (getenv("MOE_ADAPT") && atoi(getenv("MOE_ADAPT"))) m2 = 4;
            else if (getenv("MOE_BOUNDARY") && atoi(getenv("MOE_BOUNDARY"))) m2 = 1;
            else if (getenv("MOE_NOSPLIT") && atoi(getenv("MOE_NOSPLIT"))) m2 = 2;
            else m2 = 0;
        }
        if (m2 == 3) return true;          // Direction 1: table is static on device
        if (m2 == 1) return true;          // boundary: paged from the boundary refresh
        // mode 4 (MOE_SPLIT_LAYER): the split only exists at one layer, so only
        // that layer can refresh the table; the rest read what is resident.
        // mode 4 (adaptive) falls through: every selected expert is demand-paged,
        // so no routing ever reads the zero sentinel slot.
    }
    const int64_t ntok = t->ne[1];
    const int64_t nw = t->ne[0];
    std::vector<int32_t> ids((size_t) nw * ntok);
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (moe_doorbell_on(ctx)) {
        if (ntok == 1) {
            return true;
        }
        ctx->moe_page_batch_sorted(il, ids.data(), (int) (nw * ntok));
        return true;
    }
    ctx->moe_page_batch_sorted(il, ids.data(), (int) (nw * ntok));
    // else: large prefill batch runs on the full tensors; skip paging so no
    // QD1 fetch stalls land on the prompt path
    // layer-ahead + token-ahead async reads for the layers after this one;
    // decode-only: prefill rows predict the wrong step and would pollute
    // staging with useless reads. admission-controlled inside, never blocks.
    if (ntok == 1) {
        ctx->moe.step_topk[(size_t) il] = t;   // for moe_post_refresh()
        for (int64_t j = 0; j < ntok; j++) {
            ctx->moe_prefetch_ahead(il, ids.data() + (size_t) j * nw, (int) nw);
        }
        // adaptive swap tier: at the last layer every layer's routing for this
        // step is collected, so the decay/swap pass has the full picture
        if (ctx->moe_adapt_on() && il == ctx->moe.n_layer - 1) {
            ctx->moe_adapt_run();
        }
        // optional routing trace (MOE_TRACE_DIR=file prefix): one u16 row
        // [n_used] per layer per decode step for offline profile building.
        // layers run 0..47 in order; flush a full step at the last layer.
        // zero cost when unset.
        static FILE * tf = nullptr;
        static bool tinit = false;
        if (!tinit) {
            tinit = true;
            const char * p = getenv("MOE_TRACE_DIR");
            if (p && p[0]) {
                tf = fopen((std::string(p) + ".ids.u16").c_str(), "wb");
            }
        }
        if (tf) {
            static std::vector<uint16_t> tbuf;
            if (tbuf.empty()) tbuf.resize((size_t) ctx->moe.n_layer * ctx->moe.n_used, 0);
            for (int64_t k = 0; k < nw && k < ctx->moe.n_used; k++) {
                const int32_t e = ids[(size_t) k];
                tbuf[(size_t) il * ctx->moe.n_used + k] = (e >= 0 && e < 65536) ? (uint16_t) e : 0;
            }
            if (il == ctx->moe.n_layer - 1) {
                fwrite(tbuf.data(), 2, tbuf.size(), tf);
                fflush(tf);
            }
        }
    }
    {
        // prefetch telemetry: hit rate of demand commits against in-flight
        // prefetches, plus demand hit/miss totals and prefill fast-path rate
        static int n_rep = 0;
        if ((++n_rep % 200) == 0) {
            LLAMA_LOG_INFO("%s: pf issued=%ld hit=%ld demand hit=%ld miss=%ld fail=%ld iofail=%ld prefill resident=%ld miss=%ld\n",
                __func__, ctx->pager.stats_pf_issued(), ctx->pager.stats_pf_hit(),
                ctx->moe.n_hit, ctx->moe.n_miss, ctx->moe.n_page_fail,
                ctx->pager.stats_io_fail(), ctx->moe.prefill_resident, ctx->moe.prefill_miss);
        }
    }
    return true;   // MUST return true in the data phase (per cb_eval contract)
}
