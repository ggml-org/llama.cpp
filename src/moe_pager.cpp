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
#include <cstdio>
#include <cstring>
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

    stage_bytes_ = 0;
    for (size_t i = 0; i < idx_.size(); i++) stage_bytes_ = std::max(stage_bytes_, (size_t) idx_[i].len);
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

void moe_pager::attach_gpu(void * backend, size_t, size_t) { backend_ = backend; }

int moe_pager::find_(int key) const {
    for (int i = (int) reqs_.size() - 1; i >= 0; i--) {
        if (reqs_[i].used && reqs_[i].il * n_expert_ + reqs_[i].e == key) return i;
    }
    return -1;
}

int moe_pager::submit_(int key) {
    if (free_stage_.empty()) return -1;
    const int il = key / n_expert_, e = key % n_expert_;
    const moe_expert_slice * s = slices(il, e);

    const int st = free_stage_.back(); free_stage_.pop_back();
    const int ridx = (int) reqs_.size();
    int pending = 0;
    size_t boff = 0;
    for (int r = 0; r < 3; r++) {
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
                if (rr == ridx) { pending++; }
                else if (rr < (int) reqs_.size() && reqs_[rr].used) {
                    reqs_[rr].done++;
                    if (role >= 0 && role < 3) reqs_[rr].got[role] = res;
                }
            }
            sqe = io_uring_get_sqe((struct io_uring *) ring_);
        }
        io_uring_prep_read(sqe, fd_, stage_ + (size_t) st * stage_bytes_ + boff,
                           s[r].len, (off_t) s[r].off);
        sqe->user_data = (uint64_t)((ridx * 4) + r);
        boff += s[r].len;
    }
    Req q; q.il = il; q.e = e; q.stage = st; q.done = pending; q.used = true;
    q.got[0] = q.got[1] = q.got[2] = -1;
    reqs_.push_back(q);
    io_uring_submit((struct io_uring *) ring_);
    io_ops_ += 3; io_bytes_ += (long) boff;
    return (int) reqs_.size() - 1;
}

// returns true when all 3 role reads for req r transferred their full length
bool moe_pager::wait_ok_(int r) {
    if (r < 0) return false;
    const moe_expert_slice * s = slices(reqs_[r].il, reqs_[r].e);
    while (reqs_[r].done < 3) {
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
    for (int k = 0; k < 3; k++) {
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

bool moe_pager::prefetch(int il, int e) {
    const int key = il * n_expert_ + e;
    if (find_(key) >= 0) return true;
    return submit_(key) >= 0;
}

bool moe_pager::commit(int il, int e, void * dst_gate, size_t gate_bytes,
                                  void * dst_up,     size_t up_bytes,
                                  void * dst_down,   size_t down_bytes) {
    const int key = il * n_expert_ + e;
    int r = find_(key);
    if (r < 0) r = submit_(key);
    if (r < 0) return false;
    if (!wait_ok_(r)) {
        n_io_fail_++;
        free_stage_.push_back(reqs_[r].stage);
        reqs_[r].used = false;
        return false;
    }

    const moe_expert_slice * s = slices(il, e);
    char * src = stage_ + (size_t) reqs_[r].stage * stage_bytes_;
    // each slice starts at an aligned-down offset; skip the stored pre-pad
    // to reach the true expert data (len is aligned up for O_DIRECT).
    size_t off = 0;
    void * dst[3] = { dst_gate, dst_up, dst_down };
    size_t want[3] = { gate_bytes, up_bytes, down_bytes };
    for (int k = 0; k < 3; k++) {
        const size_t skip = s[k].pre;
        cudaMemcpy(dst[k], src + off + skip,
                   std::min(want[k], (size_t) s[k].len - skip), cudaMemcpyHostToDevice);
        off += s[k].len;
    }

    free_stage_.push_back(reqs_[r].stage);
    reqs_[r].used = false;
    return true;
}

bool moe_pager::fetch(int il, int e, void * dst_gate, size_t gate_bytes,
                                void * dst_up,     size_t up_bytes,
                                void * dst_down,   size_t down_bytes) {
    return commit(il, e, dst_gate, gate_bytes, dst_up, up_bytes, dst_down, down_bytes);
}