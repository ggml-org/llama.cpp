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
#include <mutex>
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
    // Split the arena: `n_demand` slots serve blocking demand fetches only and
    // the rest serve prefetches, so a prefetch burst can never take the last
    // slot a demand read needs. That starvation is what made split-free decode
    // fall through to the zero sentinel and emit garbage.
    bool reserve_staging_split(int n_total, int n_demand);
    int n_free_staging() const { return (int) free_stage_.size() + (int) free_demand_.size(); }
    int n_free_demand()  const { return (int) free_demand_.size(); }

    // ---- Tier 1: pinned host RAM cache -----------------------------------
    // A cold expert read once is kept in host memory and served from there over
    // PCIe (~20 GB/s) instead of the NVMe (~2.8 GB/s). Sized by `bytes`; 0 off.
    bool init_cache(size_t bytes);
    size_t cache_bytes() const { return cache_bytes_; }
    long stats_cache_hit()   const { return n_cache_hit_; }
    long stats_cache_miss()  const { return n_cache_miss_; }
    long stats_cache_fill()  const { return n_cache_fill_; }
    long stats_cache_evict() const { return n_cache_evict_; }

    // public for verification: exact file range this pager reads for an expert
    const moe_expert_slice * slices(int il, int e) const { return &idx_[((size_t) il * n_expert_ + e) * 3]; }
    const char * path() const { return path_.c_str(); }

    // Optional single-blob expert pack (iq_pack.py output): experts.bin holds
    // all n_expert experts of a layer back-to-back as gate|up|down, so one
    // expert is ONE contiguous read instead of three fragmented GGUF ranges.
    // The blob bytes are identical to the GGUF's, so this is an IO change only.
    // native_experts.txt gives each layer's base offset and blob_bytes.
    bool use_pack(const char * experts_bin, const char * index_txt, std::string & err);
    bool pack_mode() const { return pack_mode_; }
    long stats_gguf_reads() const { return n_gguf_reads_; }
    long stats_pack_reads() const { return n_pack_reads_; }

    long stats_io_ops()   const { return io_ops_; }
    long stats_io_bytes() const { return io_bytes_; }
    long stats_io_fail()  const { return n_io_fail_; }
    long stats_pf_issued() const { return n_pf_issued_; }
    // completed prefetch reads still unconsumed when a demand commit for the
    // same expert arrived (prefetch hit); cumulative
    long stats_pf_hit()   const { return n_pf_hit_; }
    static void * worker_stream();

private:
    struct Req { int il, e, stage, done; bool used, demand; long got[3]; };
    // public for verification: the exact file range this pager reads for an expert

    std::string path_;
    int n_layer_ = 0, n_expert_ = 0, n_used_ = 0;
    int fd_ = -1;
    std::vector<moe_expert_slice> idx_;   // n_layer * n_expert * 3
    size_t bytes_per_expert_ = 0;
    // per-(layer,role) byte length taken from the GGUF index; the pack blob is
    // the same gate|up|down concatenation, so these also size the pack layout
    std::vector<size_t> role_bytes_;

    // single-blob pack mode (see use_pack)
    bool   pack_mode_ = false;
    int    fd_pack_   = -1;                 // fd_ stays the GGUF, pack reads use this
    std::vector<uint32_t> bsrc_;            // n_layer*n_expert*3: src offset inside the blob read
    long n_gguf_reads_ = 0, n_pack_reads_ = 0;

    void * backend_ = nullptr;
    std::mutex mtx_;
    char * stage_ = nullptr;              // pinned staging, NSTAGE slots
    size_t stage_bytes_ = 0;
    static const int NSTAGE = 48;
    static const long ALIGN = 4096;

    std::vector<Req> reqs_;
    std::vector<int> free_stage_;    // prefetch-eligible staging slots
    std::vector<int> free_demand_;   // demand-only staging slots
    long              demand_stages_ = 0;

    // Tier 1: pinned host arena holding whole experts, indexed by (il,e)
    char * cache_ = nullptr;
    size_t cache_bytes_ = 0;             // per-expert bytes in the arena
    int    cache_slots_ = 0;
    std::vector<int>  cache_idx_;        // il*n_expert+e -> slot, -1 = absent
    std::vector<char> cache_used_;       // slot occupied
    std::vector<long> cache_stamp_;      // slot last-use tick (LFU)
    long cache_tick_ = 0;
    long n_cache_hit_ = 0, n_cache_miss_ = 0, n_cache_fill_ = 0, n_cache_evict_ = 0;
    void * ring_ = nullptr;
    long io_ops_ = 0, io_bytes_ = 0, n_io_fail_ = 0;
    long n_pf_issued_ = 0, n_pf_hit_ = 0;

    int submit_(int key, bool demand);
    int find_(int key) const;
    double wait_(int r);
    bool wait_ok_(int r); // true when all 3 roles moved their full length
    bool commit_locked(int il, int e, void * dst_gate, size_t gate_bytes,
        void * dst_up, size_t up_bytes, void * dst_down, size_t down_bytes);
};