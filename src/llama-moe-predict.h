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
        // for tests: start with the workers paused, and record the order of processed work
        bool     paused    = false;
        bool     trace     = false;
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
    void resume();
    void wait_idle();                 // until both queues are empty and no work is in progress
    std::vector<int> get_trace();     // +(il+1) for a HIGH expert, -(il+1) for a LOW one, in processing order

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
    };

    struct work {
        int il;
        int e;
        uint64_t step;
        int64_t  t_us; // enqueue time
        int      rank;  // guess rank for the trace (ggml-trace.h), -1 for an actual selection
        bool     trace; // enqueued by a traced step: its hints are traced (not those of prompt batches)
    };

    // ranks: per expert, for the trace; null for an actual selection
    void enqueue(bool high, int il, const std::vector<int> & experts, bool trace = false, const std::vector<int> * ranks = nullptr);
    void worker_loop(int idx);
    void prefetch_expert(int il, int e, bool low, int rank, bool trace);
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
};

// the graph op: thread 0 forwards the layer's selection (src[1]) and, if present, the next layer's router logits
// (src[2]) and bias (src[3]) to llama_moe_predictor::on_layer; src[0] only pins the op to the CPU backend
void llama_moe_predictor_hint_op(struct ggml_tensor * dst, int ith, int nth, void * userdata);
