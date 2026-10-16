// moe_pager.cpp -- see moe_pager.h.
//
// Indexes every (layer, expert, role) slice of the GGUF once at startup, then
// reads those slices with O_DIRECT through io_uring into pinned staging and
// DMAs them to the caller. Nothing is cached on disk; the kernel page cache is
// never consulted for the streamed bytes.
#include "moe_pager.h"

#include "ggml.h"
#include "gguf.h"

#include <liburing.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

static double now_s() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

moe_pager::~moe_pager() {
    if (ring_) { io_uring_queue_exit((struct io_uring *) ring_); ring_ = nullptr; }
    if (stage_) { cudaFreeHost(stage_); stage_ = nullptr; }
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
}

bool moe_pager::init(const char * gguf_path, int n_layer, int n_expert, int n_used, std::string & err) {
    path_ = gguf_path;
    n_layer_ = n_layer; n_expert_ = n_expert; n_used_ = n_used;

    struct gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct gguf_context * gctx = gguf_init_from_file(gguf_path, gp);
    if (!gctx) { err = "gguf_init_from_file failed"; return false; }

    const int64_t data_off = gguf_get_data_offset(gctx);
    idx_.assign((size_t) n_layer * n_expert * 3, moe_expert_slice{0, 0, 0});
    role_bytes_.assign((size_t) n_layer * 3, 0);
    size_t per_expert_bytes = 0;

    // role 0 = gate, role 1 = up, role 2 = down. qwen4exp on this box ships
    // SEPARATE gate/up tensors (no fused gate_up).
    for (int il = 0; il < n_layer; il++) {
        char n_g[128], n_u[128], n_dn[128];
        snprintf(n_g,  sizeof n_g,  "blk.%d.ffn_gate_exps.weight", il);
        snprintf(n_u,  sizeof n_u,  "blk.%d.ffn_up_exps.weight", il);
        snprintf(n_dn, sizeof n_dn, "blk.%d.ffn_down_exps.weight", il);

        const int64_t t_g  = gguf_find_tensor(gctx, n_g);
        const int64_t t_u  = gguf_find_tensor(gctx, n_u);
        const int64_t t_dn = gguf_find_tensor(gctx, n_dn);
        if (t_g < 0 || t_u < 0 || t_dn < 0) {
            char m[128];
            snprintf(m, sizeof m, "missing expert tensors for layer %d", il);
            err = m; gguf_free(gctx); return false;
        }
        const int64_t tids[3] = { t_g, t_u, t_dn };

        for (int e = 0; e < n_expert; e++) {
            for (int r = 0; r < 3; r++) {
                const int64_t tid = tids[r];
                const size_t tbytes = (size_t) gguf_get_tensor_size(gctx, tid);
                if (tbytes % (size_t) n_expert) {
                    err = "expert tensor size not divisible by n_expert";
                    gguf_free(gctx); return false;
                }
                const size_t per = tbytes / (size_t) n_expert;
                role_bytes_[(size_t) il * 3 + (size_t) r] = per;
                // experts of a tensor are contiguous: slice e begins at
                // base + e*per. Align DOWN for O_DIRECT and keep the pre-pad;
                // align the LENGTH up so the read is a whole number of blocks
                // (a short O_DIRECT read would otherwise DMA stale staging).
                const uint64_t fo = (uint64_t)(data_off + gguf_get_tensor_offset(gctx, tid)
                                        + (int64_t) e * (int64_t) per);
                const uint64_t pre = (ALIGN - (fo % ALIGN)) % ALIGN;
                moe_expert_slice & sl = idx_[((size_t) il * n_expert + e) * 3 + r];
                sl.off = fo - pre;
                sl.len = (uint32_t)(((per + pre + ALIGN - 1) / ALIGN) * ALIGN);
                sl.pre = (uint32_t) pre;
                if (il == 0 && e == 0) { per_expert_bytes += per; }
            }
        }
    }
    gguf_free(gctx);
    bytes_per_expert_ = per_expert_bytes;

    fd_ = open(gguf_path, O_RDONLY | O_DIRECT);
    if (fd_ < 0) { err = "open O_DIRECT failed"; return false; }

    // one staging slot must hold ALL THREE role reads of one expert: the
    // submit path lays them out back-to-back (boff), so sizing by the max
    // single-slice len would let concurrent requests overwrite each other.
    stage_bytes_ = 0;
    for (int jil = 0; jil < n_layer; jil++) {
        for (int je = 0; je < n_expert; je++) {
            const moe_expert_slice * se = &idx_[((size_t) jil * n_expert + je) * 3];
            stage_bytes_ = std::max(stage_bytes_,
                (size_t) se[0].len + se[1].len + se[2].len);
        }
    }
    if (cudaHostAlloc(&stage_, stage_bytes_ * NSTAGE, cudaHostAllocDefault) != cudaSuccess) {
        err = "cudaHostAlloc staging failed"; return false;
    }
    for (int i = NSTAGE - 1; i >= 0; i--) free_stage_.push_back(i);

    ring_ = new io_uring();
    if (io_uring_queue_init(32, (struct io_uring *) ring_, 0) < 0) {
        err = "io_uring_queue_init failed"; return false;
    }
    return true;
}

bool moe_pager::use_pack(const char * experts_bin, const char * index_txt, std::string & err) {
    if (n_layer_ <= 0 || n_expert_ <= 0) { err = "pager not indexed"; return false; }
    if (pack_mode_) { err = "pack already active"; return false; }
    std::ifstream f(index_txt);
    if (!f) { err = "cannot open pack index"; return false; }
    if (fd_pack_ < 0) {
        fd_pack_ = open(experts_bin, O_RDONLY | O_DIRECT);
        if (fd_pack_ < 0) { err = "open experts.bin O_DIRECT failed"; return false; }
    }
    bsrc_.assign((size_t) n_layer_ * n_expert_ * 3, 0);
    // build into temporaries: a rejected pack must leave the GGUF index intact
    std::vector<moe_expert_slice> new_idx = idx_;
    std::vector<uint32_t>         new_bsrc(n_layer_ * n_expert_ * 3, 0);
    size_t max_blob = 0;
    std::string line;
    int rows = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        int il, gu, dt;
        long long base = 0, blob = 0;
        if (!(is >> il >> gu >> dt >> base >> blob)) { err = "bad pack index line"; return false; }
        if (base < 0 || blob <= 0) { err = "bad pack index offsets"; return false; }
        if (il < 0 || il >= n_layer_) { err = "pack index layer out of range"; return false; }
        const size_t g = role_bytes_[(size_t) il * 3 + 0];
        const size_t u = role_bytes_[(size_t) il * 3 + 1];
        const size_t d = role_bytes_[(size_t) il * 3 + 2];
        if (g + u + d != (size_t) blob) {
            err = "pack blob_bytes do not match the GGUF gate|up|down sizes";
            return false;
        }
        for (int e = 0; e < n_expert_; e++) {
            const uint64_t fbo  = (uint64_t) base + (uint64_t) e * (uint64_t) blob;
            const uint64_t pre  = (ALIGN - (fbo % ALIGN)) % ALIGN;
            const size_t   alen = (size_t) (((blob + pre + ALIGN - 1) / ALIGN) * ALIGN);
            moe_expert_slice * s = &new_idx[((size_t) il * n_expert_ + e) * 3];
            s[0].off = fbo - pre;
            s[0].len = (uint32_t) alen;
            s[0].pre = (uint32_t) pre;
            s[1] = s[2] = moe_expert_slice{0, 0, 0};
            uint32_t * b = &new_bsrc[((size_t) il * n_expert_ + e) * 3];
            b[0] = (uint32_t) pre;
            b[1] = (uint32_t) (pre + g);
            b[2] = (uint32_t) (pre + g + u);
            if (pre + blob > alen) { err = "pack blob read underruns"; return false; }
            max_blob = std::max(max_blob, alen);
        }
        rows++;
    }
    if (rows != n_layer_) {
        char m[128];
        snprintf(m, sizeof m, "pack index has %d rows, expected %d", rows, n_layer_);
        err = m;
        return false;
    }
    idx_.swap(new_idx);
    bsrc_.swap(new_bsrc);
    pack_mode_ = true;
    stage_bytes_ = max_blob;
    return true;
}

void moe_pager::attach_gpu(void * backend, size_t, size_t) { backend_ = backend; }

int moe_pager::find_(int key) const {
    for (int i = (int) reqs_.size() - 1; i >= 0; i--) {
        if (reqs_[i].used && reqs_[i].il * n_expert_ + reqs_[i].e == key) return i;
    }
    return -1;
}

int moe_pager::submit_(int key, bool demand) {
    // No free stage: drop the oldest COMPLETED-but-unclaimed prefetch (its
    // data is stale by definition - nobody demanded it while it sat ready).
    // Never drop in-flight reads: a later commit may be waiting on them.
    std::vector<int> & pool = demand ? free_demand_ : free_stage_;
    if (pool.empty()) {
        // fall back to stealing a slot from the OTHER pool only if it is free;
        // never evict an in-flight read, a later commit may be waiting on it
        const int need_ok = pack_mode_ ? 1 : 3;   // pack mode completes in 1 read
        for (size_t i = 0; i < reqs_.size(); i++) {
            if (reqs_[i].used && reqs_[i].done >= need_ok && reqs_[i].demand == demand) {
                pool.push_back(reqs_[i].stage);
                reqs_[i].used = false;
                break;
            }
        }
        if (pool.empty()) return -1;
    }
    const int il = key / n_expert_, e = key % n_expert_;
    const moe_expert_slice * s = slices(il, e);

    const int st = pool.back(); pool.pop_back();
    const int ridx = (int) reqs_.size();
    int pending = 0;
    long got_self[3] = { -1, -1, -1 };
    size_t boff = 0;
    // pack mode: the whole expert is one blob, so one read covers all roles
    const int nread = pack_mode_ ? 1 : 3;
    for (int r = 0; r < nread; r++) {
        struct io_uring_sqe * sqe = io_uring_get_sqe((struct io_uring *) ring_);
        while (!sqe) {
            io_uring_submit((struct io_uring *) ring_);
            struct io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe((struct io_uring *) ring_, &cqe) == 0) {
                const uint64_t ud = cqe->user_data;
                const int res = cqe->res;
                io_uring_cqe_seen((struct io_uring *) ring_, cqe);
                const int rr = (int)(ud / 4);
                const int role = (int)(ud % 4);
                if (rr == ridx) { pending++; if (role >= 0 && role < 3) got_self[role] = res; }
                else if (rr < (int) reqs_.size() && reqs_[rr].used) {
                    reqs_[rr].done++;
                    if (role >= 0 && role < 3) reqs_[rr].got[role] = res;
                }
            }
            sqe = io_uring_get_sqe((struct io_uring *) ring_);
        }
        io_uring_prep_read(sqe, pack_mode_ ? fd_pack_ : fd_,
                           stage_ + (size_t) st * stage_bytes_ + boff,
                           s[r].len, (off_t) s[r].off);
        sqe->user_data = (uint64_t)((ridx * 4) + r);
        boff += s[r].len;
    }
    Req q; q.il = il; q.e = e; q.stage = st; q.done = pending; q.used = true; q.demand = demand;
    q.got[0] = got_self[0]; q.got[1] = got_self[1]; q.got[2] = got_self[2];
    reqs_.push_back(q);
    int nsub = io_uring_submit((struct io_uring *) ring_);
    if (nsub <= 0) { fprintf(stderr, "pager submit failed il=%d e=%d nsub=%d\n", il, e, nsub); }
    if (pack_mode_) { io_ops_ += 1; io_bytes_ += (long) boff; n_pack_reads_ += 1; }
    else            { io_ops_ += 3; io_bytes_ += (long) boff; n_gguf_reads_ += 3; }
    return (int) reqs_.size() - 1;
}

// returns true when every read this request needs transferred its full length
bool moe_pager::wait_ok_(int r) {
    if (r < 0) return false;
    const moe_expert_slice * s = slices(reqs_[r].il, reqs_[r].e);
    const int need = pack_mode_ ? 1 : 3;
    while (reqs_[r].done < need) {
        struct io_uring_cqe * cqe = nullptr;
        if (io_uring_wait_cqe((struct io_uring *) ring_, &cqe) < 0) break;
        const uint64_t ud = cqe->user_data;
        const int res = cqe->res;
        io_uring_cqe_seen((struct io_uring *) ring_, cqe);
        const int rr = (int)(ud / 4);
        const int role = (int)(ud % 4);
        if (rr < (int) reqs_.size() && reqs_[rr].used) {
            reqs_[rr].done++;
            if (role >= 0 && role < 3) reqs_[rr].got[role] = res;
        }
    }
    for (int k = 0; k < need; k++) {
        if (reqs_[r].got[k] != (long) s[k].len) return false;
    }
    return true;
}

double moe_pager::wait_(int r) {
    if (r < 0) return 0.0;
    const double t0 = now_s();
    wait_ok_(r);
    return now_s() - t0;
}
// Balanced submit: prefetch()ed reads stay in flight until a demand commit()
// consumes them or reap() recycles a completed-but-unclaimed one. reap()
// must run often enough that free_stage_ never empties under prefetch load;
// each call also surfaces stuck completions for in-flight requests.
int moe_pager::reap() {
    std::lock_guard<std::mutex> lock(mtx_);
    int n = 0;
    struct io_uring * ring = (struct io_uring *) ring_;
    for (;;) {
        struct io_uring_cqe * cqe = nullptr;
        if (io_uring_peek_cqe(ring, &cqe) != 0 || !cqe) break;
        const uint64_t ud = cqe->user_data;
        const int res = cqe->res;
        io_uring_cqe_seen(ring, cqe);
        const int rr = (int)(ud / 4);
        const int role = (int)(ud % 4);
        if (rr < (int) reqs_.size() && reqs_[rr].used) {
            reqs_[rr].done++;
            if (role >= 0 && role < 3) reqs_[rr].got[role] = res;
            n++;
        }
    }
    return n;
}

bool moe_pager::reserve_staging(int n) {
    // Grow the staging arena to `n` slots. Only valid on a quiet ring (no
    // in-flight reads reference the old arena); the pool reserves once at
    // startup right after init, when the ring is guaranteed quiet.
    if (n <= 0) return true;
    for (size_t i = 0; i < reqs_.size(); i++) {
        if (reqs_[i].used) return false;
    }
    char * stage = nullptr;
    if (cudaHostAlloc(&stage, stage_bytes_ * (size_t) n, cudaHostAllocDefault) != cudaSuccess) {
        return false;
    }
    cudaFreeHost(stage_);
    stage_ = stage;
    free_stage_.clear();
    for (int i = n - 1; i >= 0; i--) free_stage_.push_back(i);
    return true;
}

bool moe_pager::reserve_staging_split(int n_total, int n_demand) {
    // Grow the arena to `n_total` slots, then split it into a demand-only pool
    // and a prefetch pool. Only valid on a quiet ring.
    if (n_total <= 0) return true;
    for (size_t i = 0; i < reqs_.size(); i++) {
        if (reqs_[i].used) return false;
    }
    char * stage = nullptr;
    if (cudaHostAlloc(&stage, stage_bytes_ * (size_t) n_total, cudaHostAllocDefault) != cudaSuccess) {
        return false;
    }
    cudaFreeHost(stage_);
    stage_ = stage;
    if (n_demand < 0) n_demand = 0;
    if (n_demand > n_total) n_demand = n_total;
    demand_stages_ = n_demand;
    free_stage_.clear();
    free_demand_.clear();
    // prefetch gets the tail, demand the head: nothing is shared
    for (int i = n_total - 1; i >= n_demand; i--) free_stage_.push_back(i);
    for (int i = n_demand - 1; i >= 0; i--) free_demand_.push_back(i);
    return true;
}

// ---- Tier 1: pinned host RAM cache --------------------------------------
bool moe_pager::init_cache(size_t bytes) {
    if (bytes == 0) return true;
    if (stage_bytes_ == 0) return false;                  // need the expert size
    // the arena must hold a WHOLE expert (gate+up+down), not one role: sizing
    // it by the largest role makes every fill overrun its slot into the next
    size_t per = 0;
    if (pack_mode_) {
        for (int il = 0; il < n_layer_; il++) {
            per = std::max(per, role_bytes_[(size_t) il * 3 + 0]
                              + role_bytes_[(size_t) il * 3 + 1]
                              + role_bytes_[(size_t) il * 3 + 2]);
        }
    } else {
        const moe_expert_slice * s = slices(0, 0);
        per = s[0].len + s[1].len + s[2].len;
    }
    if (per == 0) return false;
    cache_bytes_ = per;
    cache_slots_ = (int) (bytes / per);
    if (cache_slots_ <= 0) return false;
    if (cudaHostAlloc(&cache_, cache_bytes_ * (size_t) cache_slots_, cudaHostAllocDefault) != cudaSuccess) {
        cache_ = nullptr; cache_slots_ = 0; return false;
    }
    fprintf(stderr, "moe_pager: Tier 1 host cache %.2f GiB, %d slots x %zu B\n",
            (double) bytes / (1024.0*1024.0*1024.0), cache_slots_, cache_bytes_);
    cache_idx_.assign((size_t) n_layer_ * n_expert_, -1);
    cache_used_.assign((size_t) cache_slots_, 0);
    cache_stamp_.assign((size_t) cache_slots_, 0);
    return true;
}

bool moe_pager::prefetch(int il, int e) {
    std::lock_guard<std::mutex> lock(mtx_);
    const int key = il * n_expert_ + e;
    if (find_(key) >= 0) return true;
    const int r = submit_(key, false);
    if (r >= 0) n_pf_issued_++;
    return r >= 0;
}

bool moe_pager::commit(int il, int e, void * dst_gate, size_t gate_bytes,
                                  void * dst_up,     size_t up_bytes,
                                  void * dst_down,   size_t down_bytes) {
    std::lock_guard<std::mutex> lock(mtx_);
    return commit_locked(il, e, dst_gate, gate_bytes, dst_up, up_bytes, dst_down, down_bytes);
}

bool moe_pager::commit_locked(int il, int e, void * dst_gate, size_t gate_bytes,
                                  void * dst_up,     size_t up_bytes,
                                  void * dst_down,   size_t down_bytes) {
    const int key = il * n_expert_ + e;
    cudaStream_t ws = (cudaStream_t) worker_stream();
    if (!ws) {
        n_io_fail_++;
        return false;
    }

    // Tier 1 first: a resident host copy is a PCIe copy, not an NVMe read
    if (cache_ && cache_slots_ > 0) {
        const int cslot = cache_idx_[(size_t) key];
        if (cslot >= 0) {
            const moe_expert_slice * cs = slices(il, e);
            const char * csrc = cache_ + (size_t) cslot * cache_bytes_;
            // the cache holds the expert COMPACT (gate|up|down, no O_DIRECT
            // pre-pad), so the offsets are the role sizes, not bsrc_[], which
            // index the aligned staging read
            size_t coff = 0;
            void * cdst[3] = { dst_gate, dst_up, dst_down };
            size_t cwant[3] = { gate_bytes, up_bytes, down_bytes };
            for (int k = 0; k < 3; k++) {
                const size_t cav = pack_mode_
                    ? role_bytes_[(size_t) il * 3 + (size_t) k]
                    : ((size_t) cs[k].len - cs[k].pre);
                cudaMemcpyAsync(cdst[k], csrc + coff, std::min(cwant[k], cav),
                                cudaMemcpyHostToDevice, ws);
                coff += cav;
            }
            cache_stamp_[(size_t) cslot] = ++cache_tick_;
            n_cache_hit_++;
            cudaError_t se = cudaStreamSynchronize(ws);
            if (se != cudaSuccess) { fprintf(stderr, "pager cache sync failed: %d\n", (int) se); return false; }
            return true;
        }
        n_cache_miss_++;
    }

    int r = find_(key);
    const bool was_prefetch = (r >= 0); // in flight before demand arrived
    if (r < 0) r = submit_(key, true);
    if (r < 0) return false;
    if (!wait_ok_(r)) {
        n_io_fail_++;
        (reqs_[r].demand ? free_demand_ : free_stage_).push_back(reqs_[r].stage);
        reqs_[r].used = false;
        return false;
    }
    if (was_prefetch) n_pf_hit_++;
    const moe_expert_slice * s = slices(il, e);
    const uint32_t * bsrc = pack_mode_ ? &bsrc_[(size_t) (il * n_expert_ + e) * 3] : nullptr;
    char * src = stage_ + (size_t) reqs_[r].stage * stage_bytes_;
    // each slice starts at an aligned-down offset; skip the stored pre-pad
    // to reach the true expert data (len is aligned up for O_DIRECT).
    // pack mode: all three roles come out of the one blob read at bsrc[k]
    size_t off = 0;
    void * dst[3] = { dst_gate, dst_up, dst_down };
    size_t want[3] = { gate_bytes, up_bytes, down_bytes };
    for (int k = 0; k < 3; k++) {
        const size_t skip = pack_mode_ ? bsrc[k] : s[k].pre;
        // pack mode: the blob's role k spans role_bytes_[il][k]; GGUF mode
        // each slice carries its own length minus the pre-pad
        const size_t avail = pack_mode_
            ? role_bytes_[(size_t) il * 3 + (size_t) k]
            : ((size_t) s[k].len - skip);
        cudaMemcpyAsync(dst[k], src + (pack_mode_ ? 0 : off) + skip,
                   std::min(want[k], avail), cudaMemcpyHostToDevice, ws);
        if (!pack_mode_) off += s[k].len;
    }
    cudaError_t se = cudaStreamSynchronize(ws);
    if (se != cudaSuccess) {
        fprintf(stderr, "pager fetch sync failed: %d\n", (int) se);
        n_io_fail_++;
        (reqs_[r].demand ? free_demand_ : free_stage_).push_back(reqs_[r].stage);
        reqs_[r].used = false;
        return false;
    }

    // keep a host copy so the next time this expert is demanded it comes over
    // PCIe instead of the NVMe
    if (cache_ && cache_slots_ > 0 && cache_idx_[(size_t) key] < 0) {
        int cslot = -1;
        for (int i = 0; i < cache_slots_; i++) {
            if (!cache_used_[(size_t) i]) { cslot = i; break; }
        }
        if (cslot < 0) {   // evict the least recently used
            long oldest = std::numeric_limits<long>::max();
            for (int i = 0; i < cache_slots_; i++) {
                if (cache_stamp_[(size_t) i] < oldest) { oldest = cache_stamp_[(size_t) i]; cslot = i; }
            }
            const int old_key = (int) (cache_stamp_.size() ? -1 : -1);
            (void) old_key;
            for (size_t k = 0; k < cache_idx_.size(); k++) {
                if (cache_idx_[k] == cslot) { cache_idx_[k] = -1; break; }
            }
            n_cache_evict_++;
        }
        char * csrc = stage_ + (size_t) reqs_[r].stage * stage_bytes_;
        char * cdst = cache_ + (size_t) cslot * cache_bytes_;
        size_t at = 0;
        if (pack_mode_) {
            const uint32_t * cbs = &bsrc_[(size_t) key * 3];
            for (int k = 0; k < 3; k++) {
                const size_t nb = role_bytes_[(size_t) il * 3 + (size_t) k];
                memcpy(cdst + at, csrc + cbs[k], nb);
                at += nb;
            }
        } else {
            const moe_expert_slice * cs = slices(il, e);
            for (int k = 0; k < 3; k++) {
                const size_t nb = (size_t) cs[k].len - cs[k].pre;
                memcpy(cdst + at, csrc + cs[k].pre, nb);
                at += nb;
            }
        }
        cache_used_[(size_t) cslot] = 1;
        cache_stamp_[(size_t) cslot] = ++cache_tick_;
        cache_idx_[(size_t) key] = cslot;
        n_cache_fill_++;
    }

    (reqs_[r].demand ? free_demand_ : free_stage_).push_back(reqs_[r].stage);
    reqs_[r].used = false;
    return true;
}

void * moe_pager::worker_stream() {
    thread_local cudaStream_t ws = nullptr;
    thread_local bool tried = false;
    if (!ws && !tried) {
        tried = true;
        cudaError_t e = cudaStreamCreateWithFlags(&ws, cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            fprintf(stderr, "pager worker_stream create failed: %d\n", (int) e);
            ws = nullptr;
        }
    }
    return (void *) ws;
}



bool moe_pager::fetch(int il, int e, void * dst_gate, size_t gate_bytes,
                                void * dst_up,     size_t up_bytes,
                                void * dst_down,   size_t down_bytes) {
    std::lock_guard<std::mutex> lock(mtx_);
    return commit_locked(il, e, dst_gate, gate_bytes, dst_up, up_bytes, dst_down, down_bytes);
}
