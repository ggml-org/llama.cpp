#pragma once

// --pin-hotexperts N
//
// Tracks, GLOBALLY across all MoE layers, how often each (layer, expert)
// pair gets selected by the router (via ggml_backend_sched's eval callback
// on the "ffn_moe_topk-<il>" nodes) and keeps the top N experts **per layer**
// (total capacity = N * num_moe_layers) locked in RAM with mlock()/VirtualLock(),
// IN PLACE inside the model's own weight tensors (ffn_gate_exps / ffn_up_exps /
// ffn_down_exps).
//
// Rationale: with --pin-hotexperts, all MoE expert tensors are assumed to
// live in host (CPU) memory (only the dense/router parts are typically
// offloaded to VRAM). If the model was loaded via mmap (the default) and
// does not fit entirely in RAM, or --mlock was not used, the OS is free to
// evict a cold expert's pages and re-fault them in from disk the next time
// it is selected. This does NOT copy any data anywhere and does NOT change
// what ggml_mul_mat_id() reads: it locks the exact bytes that are already
// used by the compute graph, so the benefit is real and unconditional
// whenever those pages would otherwise be evictable.
//
// Pin/evict is fully ONLINE (no periodic "refresh" pass, no sorting):
// a single global ordered set tracks the hottest (layer, expert) pairs
// across ALL layers, keyed by usage count, with the coldest pinned expert
// at the front. The total pin budget is N * num_moe_layers. Every time an
// expert is selected:
//   - if it's already pinned, its position in the ordered set is updated
//   - else if there's still a free pin slot, it is pinned immediately
//   - else if its count just overtook the coldest pinned expert, that
//     expert is evicted (unlocked) and this one is pinned in its place
// This is O(log N) per observation and never needs to re-rank the whole
// model, so there is no "N most used experts of the last K decisions"
// window -- ranking is exact and always up to date.

#include "ggml.h"
#include "llama-mmap.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

struct llama_model;

class llama_hot_expert_cache {
  public:
    // n_pin_experts: number of hottest experts to keep mlock'd per layer (N in --pin-hotexperts N)
    //                 total global capacity = N * num_moe_layers, ranked globally
    // budget_bytes:  hard cap on total bytes locked across ALL layers combined (0 = unlimited, NOT recommended)
    // stats_interval: print_stats() is called automatically every `stats_interval` router
    //                 observations (0 = disabled, only the destructor prints a final summary)
    llama_hot_expert_cache(const llama_model & model,
                           int32_t             n_pin_experts,
                           uint64_t            budget_bytes,
                           uint64_t            stats_interval = 200);
    ~llama_hot_expert_cache();

    llama_hot_expert_cache(const llama_hot_expert_cache &)             = delete;
    llama_hot_expert_cache & operator=(const llama_hot_expert_cache &) = delete;

    // ggml_backend_sched_eval_callback-compatible entry point.
    // Pass `this` as user_data when installing.
    static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

    // true if `expert_id` in layer `il` is currently mlock'd in place
    bool is_pinned(int il, int32_t expert_id) const;

    // Prints a summary (bytes locked, per-layer breakdown, router observations)
    // directly to stderr with fprintf. Deliberately bypasses LLAMA_LOG_* / the
    // ggml log callback: at destruction time (process teardown, or a caller
    // that already tore down its own log sink) those can silently swallow
    // output, so this is a best-effort guaranteed-visible dump.
    void print_stats() const;

  private:
    // Unique key identifying a specific expert in a specific layer
    struct expert_key {
        int     layer;
        int32_t expert_id;

        bool operator==(const expert_key & o) const { return layer == o.layer && expert_id == o.expert_id; }
    };

    struct expert_key_hash {
        std::size_t operator()(const expert_key & k) const {
            return std::hash<int>()(k.layer) ^ (std::hash<int32_t>()(k.expert_id) << 1);
        }
    };

    struct mlock_deleter {
        void operator()(llama_mlock * p) const {
            if (p) {
                p->unlock();
                delete p;
            }
        }
    };

    // holds the mlock guards keeping one expert's rows resident for each
    // relevant weight tensor; unlock() is called via custom deleter on destruction
    struct pinned_expert {
        std::unique_ptr<llama_mlock, mlock_deleter> gate_lock;
        std::unique_ptr<llama_mlock, mlock_deleter> up_lock;
        std::unique_ptr<llama_mlock, mlock_deleter> down_lock;
        std::unique_ptr<llama_mlock, mlock_deleter> gate_up_lock;
        size_t                                      nbytes_locked = 0;
    };

    struct layer_state {
        const ggml_tensor * t_gate    = nullptr;
        const ggml_tensor * t_up      = nullptr;
        const ggml_tensor * t_down    = nullptr;
        const ggml_tensor * t_gate_up = nullptr;

        bool tensors_are_host = false;  // false => experts live on a non-CPU backend, pinning is a no-op
        bool resolved_tensors = false;
    };

    void on_topk_tensor(int il, const struct ggml_tensor * t);
    void resolve_tensors(int il, layer_state & ls);

    // called once per (layer, selected expert) observation; updates global counts and
    // pins/evicts on the fly against the global top-N set
    void observe_expert(int il, layer_state & ls, int32_t expert_id);

    // Returns true if at least some bytes were actually locked.
    bool pin_expert(int il, layer_state & ls, int32_t expert_id);
    void unpin_expert(int il, layer_state & ls, int32_t expert_id);

    // locks the byte range of `expert_id`'s row within tensor `w` in place, honoring
    // the remaining global budget; returns bytes ACTUALLY locked (0 on failure/skip/no
    // budget left).
    size_t lock_expert_row(const struct ggml_tensor *                    w,
                           int32_t                                       expert_id,
                           std::unique_ptr<llama_mlock, mlock_deleter> & out_lock);

    const llama_model & model;

    const int32_t  n_pin;            // N experts per layer
    int32_t        n_pin_total = 0;  // N * num_moe_layers (global cap)
    const uint64_t budget_bytes;     // 0 = unlimited
    const uint64_t stats_interval;   // 0 = disabled periodic printing

    mutable std::mutex                   mu;
    std::unordered_map<int, layer_state> layers;

    // Global tracking across all layers
    std::unordered_map<expert_key, uint64_t, expert_key_hash> counts;  // (layer, expert_id) -> times selected

    // Global pinned set: (count, layer, expert_id) ordered ascending by count
    // begin() is always the coldest pinned expert globally
    std::set<std::tuple<uint64_t, int, int32_t>>                   pinned_rank;
    std::unordered_map<expert_key, pinned_expert, expert_key_hash> pinned;

    uint64_t n_eval_calls   = 0;
    uint64_t n_bytes_locked = 0;  // sum of llama_mlock::size(), i.e. bytes ACTUALLY locked, across all layers
};
