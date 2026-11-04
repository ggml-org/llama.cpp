#pragma once

// next-layer expert prediction for MoE models whose routed experts are streamed from disk (lazy, mmap'ed)
//
// while layer il computes, the experts layer il+1 will most likely select are guessed from layer il+1's router
// applied to layer il's router input, plus the experts that were recently popular in layer il+1; a background
// worker hints the kernel to read them (POSIX_MADV_WILLNEED), after the experts layer il actually selected
// the number of guesses adapts per layer from how often the outermost guesses turn out to be needed, and the
// learned state is saved to a file so that a restarted server starts warm
//
// enabled with LLAMA_MOE_PREDICT=1 (observe: statistics only) or 2 (prefetch)

#include "ggml.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_moe_predictor {
    enum mode_t { MODE_OBSERVE = 1, MODE_PREFETCH = 2 };

    struct params {
        int      mode      = MODE_OBSERVE;
        int      n_layer   = 0;
        int      n_expert  = 0;
        int      n_used    = 0;
        uint64_t model_id  = 0;     // identifies the model in the stats file
        std::string stats_path;     // empty: no persistence
        int      n_workers = 1;     // 1..2
        float    grow      = 0.5f;  // grow a slot group if its next 4 slots would be at least this useful
        float    shrink    = 0.3f;  // shrink if its outermost 4 slots are less useful than this
        int      cap       = 32;    // max router + popularity slots per layer
        // false: mincore cannot be trusted for the model files (neither owned nor writable), the HIGH loads decide
        bool     mincore_ok = true;
        // for tests: start with the workers paused, and record the order of processed work
        bool     paused    = false;
        bool     trace     = false;
        // for tests: bytes a load reads from disk (instead of /proc/thread-self/io), and expert residency (instead of mincore)
        std::function<uint64_t(int il, int e, bool high)> test_load_bytes;
        std::function<bool(int il, int e)>                test_resident;
    };

    // the three expert tensors of a MoE layer (nullptr for dense layers)
    struct layer_tensors {
        const ggml_tensor * exps[3] = { nullptr, nullptr, nullptr };
    };

    llama_moe_predictor(const params & p, std::vector<layer_tensors> layers);
    ~llama_moe_predictor(); // final save, joins the workers

    llama_moe_predictor(const llama_moe_predictor &) = delete;
    llama_moe_predictor & operator=(const llama_moe_predictor &) = delete;

    // called once per MoE layer and step from the graph (one thread, takes microseconds):
    // sel:  [n_used, n_tok] I32 actual selection of layer il (strides in bytes)
    // pred: [n_expert, n_tok] F32 router logits of layer il_next for the same tokens, bias: [n_expert]; may be null
    void on_layer(int il, const int32_t * sel, size_t sel_nb0, size_t sel_nb1, int n_tok,
                  int il_next, const float * pred, size_t pred_nb1, const float * bias);

    // per-layer userdata for the graph op; stable addresses owned by the predictor
    struct hint_ud {
        llama_moe_predictor * pred;
        int il;
        int il_next;
    };
    hint_ud * get_hint_ud(int il, int il_next);

    int mode()   const { return par.mode; }
    int n_used() const { return par.n_used; }

    // introspection (tests, logs)
    struct layer_view {
        int n_r;
        int n_p;
        double recall;     // hits / actual over the life of the predictor
        uint64_t tokens;
    };
    layer_view view(int il) const;
    uint32_t   checksum() const;      // of the persisted part of the state
    bool       loaded() const { return loaded_ok; }
    bool       save();                // synchronous save (also used by the worker)

    // for tests
    struct counters {
        uint64_t high_done;
        uint64_t low_done;
        uint64_t low_drops;
        uint64_t high_drops;
    };
    counters get_counters() const;
    void pause();
    void resume();
    void wait_idle();                 // until both queues are empty and no work is in progress
    std::vector<int> get_trace();     // +(il+1) for a HIGH expert, -(il+1) for a LOW one, in processing order

    // lifetime counters for the performance panel (relaxed atomics, any thread)
    // bytes: what the prefetch loads read from disk; a guess that is not used stays pending until it is used
    // (credited if still resident) or known to be gone (wasted)
    struct stats {
        int      mode;
        int      layer_seen;
        uint64_t step_seen;
        uint64_t hits, actual, guesses;
        uint64_t bytes_hit, bytes_wasted, bytes_late, bytes_actual, bytes_read_predictor, bytes_pending, bytes_credited;
        uint64_t probes; // residency probes (3 mincore calls each) on the graph thread
    };
    stats get_stats() const;

    // the process read_bytes and the page cache size, published about once a second; also expires the pending
    // guesses whose pages the page cache has surely evicted since they were read
    void set_io_clock(uint64_t read_bytes, uint64_t cached_bytes);

    // false if mincore cannot be trusted for these files (it reports every page resident for files neither owned nor writable)
    static bool mincore_trusted(const std::vector<std::string> & paths);

    // ranking used for the router guesses: sigmoid(logit) + bias, descending (exposed for tests)
    static void rank_router(const float * logits, const float * bias, int n_expert, int n, std::vector<int> & out);

private:
    struct layer_state {
        std::vector<float> pop;          // decayed selection counts
        int n_r = 16;
        int n_p = 0;
        // last prediction for this layer: router ranks [0, n_r + 4), popularity ranks [0, n_p + 4)
        std::vector<int> pred_r;
        std::vector<int> pred_p;
        bool has_pred = false;
        // usefulness accounting since the last adaptation
        uint64_t hit_r_outer = 0, hit_r_next = 0, hit_p_outer = 0, hit_p_next = 0;
        uint32_t n_eval = 0;
        // lifetime
        uint64_t hits = 0, actual = 0, tokens = 0;
        uint64_t wasted = 0; // predicted experts that were not selected
        // counterfactual recall grid (observe logs)
        uint64_t cf_hits[4] = {}; // router top 16, 24, 32, 48
        uint64_t cf_mix_hits[3] = {}; // router top T-4 + popularity top 4, T = 16, 24, 32
        std::vector<int> pred_r48;
        std::vector<int> pred_p_all;  // most popular overall (for the counterfactual mix)
        std::vector<int> guess;       // the guesses issued for this layer (router, then popularity)
    };

    struct work {
        int il;
        int e;
        uint64_t step;
        int64_t  t_us; // enqueue time
    };

    void enqueue(bool high, int il, const std::vector<int> & experts);
    void worker_loop();
    void prefetch_expert(int il, int e, bool low, uint64_t step);
    uint64_t load_bytes(int il, int e, bool high);
    bool resident(int il, int e);
    void account_load(int il, int e, bool low, uint64_t step, uint64_t b);
    void resolve_pending(size_t i, bool used);
    bool pop_high(work & w);
    void drain_high();
    void adapt(layer_state & s);
    void log_stats_locked();
    void serialize_locked(std::vector<uint8_t> & out) const;
    bool load();

    params par;
    std::vector<layer_tensors> lt;
    std::vector<layer_state>   ls;
    std::vector<std::unique_ptr<hint_ud>> uds;
    float decay;
    size_t page_size = 4096;
    // set when ggml splits matrix rows over NUMA nodes: reads each node's rows of an expert into that node's memory
    void (*numa_willneed)(const void * data, size_t row_size, int64_t nrows) = nullptr;

    mutable std::mutex mtx;           // state
    bool     unlocked   = false;      // a single-token step has run (under mtx)
    uint64_t step       = 0;          // decode steps seen (incremented at the first MoE layer)
    int      first_moe  = -1;
    int      cur_layer  = -1;         // last layer observed in the current step
    uint64_t last_save  = 0;
    uint64_t last_log   = 0;
    bool     loaded_ok  = false;

    // worker
    std::mutex              qmtx;
    std::condition_variable qcv;
    std::deque<work>        q_high, q_low;
    bool                    stop = false;
    bool                    want_save = false;
    std::vector<std::thread> workers;
    std::atomic<uint64_t>   busy_us{0}, high_wait_us{0}, high_n{0}, low_drops{0};
    std::atomic<uint64_t>   step_seen{0};
    std::atomic<int>        layer_seen{-1};
    std::atomic<int>        n_high_pending{0};
    std::atomic<uint64_t>   low_done{0};
    std::atomic<uint64_t>   high_drops{0};
    bool                    paused = false;  // under qmtx
    int                     in_progress = 0; // under qmtx
    std::condition_variable idle_cv;
    std::vector<int>        trace;           // under qmtx
    int64_t                 t_start_us = 0;

    std::vector<char> scratch_mark; // n_expert, used under mtx
    std::vector<float> scratch_score;
    std::vector<int>  scratch_rank_r; // n_expert: rank in the router guesses (pred_r48), -1 if none
    std::vector<int>  scratch_rank_p; // n_expert: rank in the popularity guesses (pred_p), -1 if none

    // byte accounting, prefetch mode only: per (layer, expert), index il*n_expert + e
    std::unique_ptr<std::atomic<uint64_t>[]> pred_tag;    // step + 1 of the current guess, 0: none
    std::unique_ptr<std::atomic<uint64_t>[]> pred_bytes;  // read for the current guess
    std::unique_ptr<std::atomic<uint64_t>[]> pend_bytes;  // read for an earlier guess that was not used
    std::unique_ptr<std::atomic<uint64_t>[]> pend_clock;  // io clock when it became pending
    std::unique_ptr<std::atomic<uint64_t>[]> pend_cached; // page cache size then

    std::atomic<uint64_t> n_hits{0}, n_actual{0}, n_guesses{0};
    std::atomic<uint64_t> bytes_hit{0}, bytes_wasted{0}, bytes_late{0}, bytes_actual{0}, bytes_read_predictor{0};
    std::atomic<uint64_t> bytes_pending{0}, bytes_credited{0}, n_probes{0};
    std::atomic<uint64_t> io_clock{0}, io_cached{0};
    uint64_t io_floor = 0;   // sampler thread only: clock when the sampler resumed
    int64_t  io_last_us = 0; // sampler thread only
};

// the graph op: thread 0 forwards the layer's selection (src[1]) and, if present, the next layer's router logits
// (src[2]) and bias (src[3]) to llama_moe_predictor::on_layer; src[0] only pins the op to the CPU backend
void llama_moe_predictor_hint_op(struct ggml_tensor * dst, int ith, int nth, void * userdata);
