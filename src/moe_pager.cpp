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
    idx_.assign((size_t) n_layer * n_expert * 2, moe_expert_slice{0, 0});
    size_t per_expert_bytes = 0;

    // role 0 = expert projections (fused gate_up when present, else gate),
    // role 1 = down. qwen4exp on this box ships SEPARATE gate/up tensors.
    for (int il = 0; il < n_layer; il++) {
        char n_gu[128], n_gn[128], n_dn[128];
        snprintf(n_gu, sizeof n_gu, "blk.%d.ffn_gate_up_exps.weight", il);
        snprintf(n_gn, sizeof n_gn, "blk.%d.ffn_gate_exps.weight", il);
        snprintf(n_dn, sizeof n_dn, "blk.%d.ffn_down_exps.weight", il);

        int64_t t_gu = gguf_find_tensor(gctx, n_gu);
        if (t_gu < 0) t_gu = gguf_find_tensor(gctx, n_gn);
        const int64_t t_dn = gguf_find_tensor(gctx, n_dn);
        if (t_gu < 0 || t_dn < 0) {
            char m[128];
            snprintf(m, sizeof m, "missing expert tensors for layer %d", il);
            err = m; gguf_free(gctx); return false;
        }
        const int64_t tids[2] = { t_gu, t_dn };

        for (int e = 0; e < n_expert; e++) {
            for (int r = 0; r < 2; r++) {
                const int64_t tid = tids[r];
                const size_t tbytes = (size_t) gguf_get_tensor_size(gctx, tid);
                if (tbytes % (size_t) n_expert) {
                    err = "expert tensor size not divisible by n_expert";
                    gguf_free(gctx); return false;
                }
                const size_t per = tbytes / (size_t) n_expert;
                // experts of a tensor are contiguous: slice e begins at
                // base + e*per. Align DOWN for O_DIRECT and keep the pre-pad.
                const uint64_t fo = (uint64_t)(data_off + gguf_get_tensor_offset(gctx, tid)
                                        + (int64_t) e * (int64_t) per);
                const uint64_t pre = (ALIGN - (fo % ALIGN)) % ALIGN;
                moe_expert_slice & sl = idx_[((size_t) il * n_expert + e) * 2 + r];
                sl.off = fo - pre;
                sl.len = (uint32_t)(per + pre);
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
    for (int r = 0; r < 2; r++) {
        struct io_uring_sqe * sqe = io_uring_get_sqe((struct io_uring *) ring_);
        while (!sqe) {
            io_uring_submit((struct io_uring *) ring_);
            struct io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe((struct io_uring *) ring_, &cqe) == 0) {
                const uint64_t ud = cqe->user_data;
                io_uring_cqe_seen((struct io_uring *) ring_, cqe);
                const int rr = (int)(ud / 4);
                if (rr == ridx) pending++;
                else if (rr < (int) reqs_.size() && reqs_[rr].used) reqs_[rr].done++;
            }
            sqe = io_uring_get_sqe((struct io_uring *) ring_);
        }
        io_uring_prep_read(sqe, fd_, stage_ + (size_t) st * stage_bytes_ + boff,
                           s[r].len, (off_t) s[r].off);
        sqe->user_data = (uint64_t)((ridx * 4) + r);
        boff += s[r].len;
    }
    Req q; q.il = il; q.e = e; q.stage = st; q.done = pending; q.used = true;
    reqs_.push_back(q);
    io_uring_submit((struct io_uring *) ring_);
    io_ops_ += 2; io_bytes_ += (long) boff;
    return (int) reqs_.size() - 1;
}

double moe_pager::wait_(int r) {
    if (r < 0) return 0.0;
    const double t0 = now_s();
    while (reqs_[r].done < 2) {
        struct io_uring_cqe * cqe = nullptr;
        if (io_uring_wait_cqe((struct io_uring *) ring_, &cqe) < 0) break;
        const uint64_t ud = cqe->user_data;
        io_uring_cqe_seen((struct io_uring *) ring_, cqe);
        const int rr = (int)(ud / 4);
        if (rr < (int) reqs_.size() && reqs_[rr].used) reqs_[rr].done++;
    }
    return now_s() - t0;
}

bool moe_pager::prefetch(int il, int e) {
    const int key = il * n_expert_ + e;
    if (find_(key) >= 0) return true;
    return submit_(key) >= 0;
}

bool moe_pager::commit(int il, int e, void * dst_gu, size_t gu_bytes,
                                  void * dst_dn, size_t dn_bytes) {
    const int key = il * n_expert_ + e;
    int r = find_(key);
    if (r < 0) r = submit_(key);
    if (r < 0) return false;
    wait_(r);

    const moe_expert_slice * s = slices(il, e);
    char * src = stage_ + (size_t) reqs_[r].stage * stage_bytes_;
    // each slice carries an alignment pre-pad that must be skipped
    const size_t skip0 = (ALIGN - (s[0].off % ALIGN)) % ALIGN;
    const size_t skip1 = (ALIGN - (s[1].off % ALIGN)) % ALIGN;

    cudaMemcpy(dst_gu, src + skip0,
               std::min(gu_bytes, (size_t) s[0].len - skip0), cudaMemcpyHostToDevice);
    cudaMemcpy(dst_dn, src + s[0].len + skip1,
               std::min(dn_bytes, (size_t) s[1].len - skip1), cudaMemcpyHostToDevice);

    free_stage_.push_back(reqs_[r].stage);
    reqs_[r].used = false;
    return true;
}

bool moe_pager::fetch(int il, int e, void * dst_gu, size_t gu_bytes,
                                void * dst_dn, size_t dn_bytes) {
    return commit(il, e, dst_gu, gu_bytes, dst_dn, dn_bytes);
}