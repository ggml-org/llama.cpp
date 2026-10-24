#pragma once

#include "ggml-backend.h"
#include "llama.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;

// reads a Strata expert profile (STRP): the (layer, expert) pairs ranked by routing frequency, hottest first
// false with err when the file is not one or was made for another n_layer x n_expert
bool llama_moe_cache_read_profile(const std::string & path, uint32_t n_layer, uint32_t n_expert,
        std::vector<std::pair<int32_t, int32_t>> & ranked, std::string & err);

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // with a profile, the cache starts filled with the hottest experts of each group of layers instead of empty
    // hybrid: those experts stay for good and the experts the cache does not hold are computed on the CPU, see build_hybrid_ids
    // adaptive (with hybrid): the cache counts the experts the router picks and swaps the hottest it lacks in for the coldest
    // it holds, every few dozen generated tokens, so it follows the conversation; the profile is only the start
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            const std::string & profile = "", bool hybrid = false, bool adaptive = false);
    ~llama_moe_cache();

    // adaptive: swap now, at most max_swaps, any expert hotter than a resident one (no hysteresis); the number of swaps,
    // -1 if the cache is not adaptive. Waits for the device to be idle.
    int32_t rebalance(int32_t max_swaps);

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the device copy of the expert bias b of a cached layer (read with the cache's experts), nullptr if there is none
    ggml_tensor * get_bias(const ggml_tensor * b) const;

    // hybrid: splits the experts selected in layer il, I32 [n_expert_used, n_tokens], into an I32 [n_expert_used, n_tokens, 4]:
    //   [.., .., 0] the cache slot of each expert the cache holds, -1 for the others (CUDA's one-token mul_mat_id skips them)
    //   [.., .., 1] each expert the cache does not hold, -1 for the others (the CPU's mul_mat_id skips them)
    //   [.., .., 2] 1 where the cache holds the expert, 0 where the CPU computes it
    //   [.., .., 3] as 0, but a slot the ubatch reads anyway in place of -1 (for the GPU kernels that do not skip)
    // a CPU op that reads the static slot map in place; nullptr if layer il is not served this way for n_tokens tokens
    ggml_tensor * build_hybrid_ids(ggml_context * ctx, ggml_tensor * selected_experts, int32_t il) const;

    // hybrid, one token: the GPU and the CPU work on a layer at once, handing data over through a mailbox (ggml-mailbox.h)
    // instead of the scheduler's synchronous copies. The GPU sends the layer's input and the selected experts, then computes
    // the experts it holds while the CPU computes the others; the CPU sends its result back.
    struct mailbox_graph {
        ggml_tensor * send;      // GPU op: stores x and the ids in the mailbox, raises the flag (its output is the flag)
        ggml_tensor * x;         // the input in the mailbox, for the CPU
        ggml_tensor * ids;       // the selected experts in the mailbox, for the CPU
        ggml_tensor * cold_ids;  // CPU op: waits for the flag, then the ids of the experts the cache does not hold, -1 for the others
        ggml_tensor * hot_slots; // I32 [k, n_tokens]: the cache slot of each expert it holds, -1 for the others
        ggml_tensor * hit;       // F32 [1, k, n_tokens]: 1 where the cache holds the expert
    };
    bool mailbox_ready(int32_t il) const;
    // false if x or ids do not fit the layer's mailbox
    bool build_mailbox(ggml_context * ctx, int32_t il, ggml_tensor * x, ggml_tensor * ids, mailbox_graph & out) const;
    // the CPU's experts (F32 [n_out, k, 1]) back on the GPU: `copy` is the CPU op that stores them in the mailbox,
    // `recv` the GPU op that reads them
    void build_mailbox_recv(ggml_context * ctx, int32_t il, ggml_tensor * e_cold, ggml_tensor *& copy, ggml_tensor *& recv) const;
    // before each graph compute: a new epoch, which the GPU's sends stamp on their flags
    void begin_step();
    // after each graph compute: true if a hand-over from the GPU timed out in it (the CPU then computed none of its
    // experts, so the step's results are wrong and it must fail); clears the error
    bool take_mailbox_error();

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // counters since the context was created, see llama_moe_cache_get_stats
    llama_moe_cache_stats get_stats() const;
    int32_t get_layer_stats(llama_moe_cache_layer_stats * stats, int32_t n_max) const;
    int32_t get_resident(int32_t il, int32_t * experts, int32_t n_max) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
