// moe_pager.h -- user-space O_DIRECT expert pager for sparse MoE streaming.
//
// Reads routed-expert weight slices straight out of the GGUF with O_DIRECT via
// io_uring into pinned staging, bypassing the kernel page cache entirely.
// The index is built once from the GGUF header; no expert bytes are cached on
// disk and nothing is written.
//
// This is deliberately self-contained: it knows nothing about llama.cpp graphs so
// it can be unit-tested on its own before being wired into build_moe_ffn.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct ggml_backend_buffer;
struct ggml_backend;

// one expert slice set: gate + up + down, raw quantised bytes.
// off is 4 KiB aligned-down; len covers the pre-pad plus the expert bytes,
// aligned UP to 4 KiB so O_DIRECT reads always transfer in full.
struct moe_expert_slice {
    uint64_t off;   // 4 KiB aligned-down file offset (read starts here)
    uint32_t len;   // read length, aligned UP to 4 KiB
    uint32_t pre;   // leading pad bytes before the true expert data (off + pre = true offset)
};

class moe_pager {
public:
    ~moe_pager();

    // Open the GGUF and index every (layer, expert) slice. Returns false and
    // fills `err` on failure. Does not allocate GPU memory.
    bool init(const char * gguf_path, int n_layer, int n_expert, int n_used, std::string & err);

    // Attach a CUDA backend + graph context so we can DMA into weight tensors.
    void attach_gpu(void * backend, size_t gate_up_elems, size_t down_elems);

    // Number of layers/experts this pager serves.
    int  n_layer()  const { return n_layer_; }
    int  n_expert() const { return n_expert_; }
    int  n_used()   const { return n_used_; }
    size_t bytes_per_expert() const { return bytes_per_expert_; }

    // Fetch expert `e` of layer `il` into pinned staging and DMA it into the
    // supplied device pointers (gate, then up, then down). Blocking.
    // Returns true on success.
    bool fetch(int il, int e, void * dst_gate, size_t gate_bytes,
                                  void * dst_up,     size_t up_bytes,
                                  void * dst_down,   size_t down_bytes);
    // Non-blocking prefetch: queue a read. Returns true if it was queued
    // (or already in flight).
    bool prefetch(int il, int e);
    // Wait for a queued prefetch and DMA it. Returns true on success.
    bool commit(int il, int e, void * dst_gate, size_t gate_bytes,
                                   void * dst_up,     size_t up_bytes,
                                   void * dst_down,   size_t down_bytes);
    // Drain (reap) any completed reads so their staging slots recycle.
    // Call periodically (e.g. once per layer) to bound staging pressure.
    // Returns the number of completions reaped.
    int reap();
    // Reserve `n` extra staging slots for in-flight prefetches beyond the
    // demand path's needs. Returns false (and keeps the old size) when the
    // allocation fails.
    bool reserve_staging(int n);
    // Free staging slots right now (for admission control).
    int n_free_staging() const { return (int) free_stage_.size(); }

    // public for verification: exact file range this pager reads for an expert
    const moe_expert_slice * slices(int il, int e) const { return &idx_[((size_t) il * n_expert_ + e) * 3]; }
    const char * path() const { return path_.c_str(); }

    long stats_io_ops()   const { return io_ops_; }
    long stats_io_bytes() const { return io_bytes_; }
    long stats_io_fail()  const { return n_io_fail_; }
    long stats_pf_issued() const { return n_pf_issued_; }
    // completed prefetch reads still unconsumed when a demand commit for the
    // same expert arrived (prefetch hit); cumulative
    long stats_pf_hit()   const { return n_pf_hit_; }

private:
    struct Req { int il, e, stage, done; bool used; long got[3]; };
    // public for verification: the exact file range this pager reads for an expert

    std::string path_;
    int n_layer_ = 0, n_expert_ = 0, n_used_ = 0;
    int fd_ = -1;
    std::vector<moe_expert_slice> idx_;   // n_layer * n_expert * 3
    size_t bytes_per_expert_ = 0;

    void * backend_ = nullptr;
    char * stage_ = nullptr;              // pinned staging, NSTAGE slots
    size_t stage_bytes_ = 0;
    static const int NSTAGE = 48;
    static const long ALIGN = 4096;

    std::vector<Req> reqs_;
    std::vector<int> free_stage_;
    void * ring_ = nullptr;
    long io_ops_ = 0, io_bytes_ = 0, n_io_fail_ = 0;
    long n_pf_issued_ = 0, n_pf_hit_ = 0;

    int submit_(int key);
    int find_(int key) const;
    double wait_(int r);
    bool wait_ok_(int r); // true when all 3 roles moved their full length
};