#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llm_graph_context;

// the experts of a MoE layer, as llm_graph_context::build_moe_ffn passes them to the MoE cache
struct llama_moe_cache_layer {
    int32_t       il;
    ggml_tensor * up_exps;
    ggml_tensor * gate_exps;
    ggml_tensor * down_exps;
    ggml_tensor * gate_up_exps;
    bool          weight_before_ffn; // the router weights scale the expert input
    bool          biases;            // the experts have biases
};

// the lookup of the selected experts of a layer in the MoE cache, see llama_moe_cache::build_lookup
struct llama_moe_cache_lookup {
    int32_t       il      = -1;
    ggml_tensor * slots   = nullptr; // [n_expert_used, n_tokens] slots of the selected experts in the MoE cache, nullptr if the layer does not read the cache
    ggml_tensor * cpu_ids = nullptr; // [n_expert_used, n_tokens] misses computed from the host experts, -1 for the other experts, or nullptr

    // the misses computed within the graph:
    ggml_tensor * slots_miss = nullptr; // [n_expert_used, n_tokens] two passes: the slots of the misses the device caches, -1 for the others
    bool          cpu_graph  = false;   // the result of the CPU part is merged after the cached experts, see llama_moe_cache::set_cpu_graph
};

// the CPU part of a MoE layer whose misses the MoE cache computes within the graph, see llama_moe_cache::set_cpu_graph
struct llama_moe_cache_cpu_part {
    int32_t       il;
    ggml_tensor * input;    // [n_embd, 1, n_tokens]
    ggml_tensor * selected; // [n_expert_used, n_tokens]
    ggml_tensor * ids;      // [n_expert_used, n_tokens] experts of the CPU, -1 for the others
    ggml_tensor * result;   // [n_embd, n_expert_used, n_tokens]
};

// what the MoE cache keeps with a graph, see llm_graph_result
struct llama_moe_cache_graph {
    std::vector<llama_moe_cache_cpu_part> cpu_parts; // not in the graph, the MoE cache computes them, see llama_moe_cache::build_experts
    std::vector<int32_t> publish;                    // the publish ops of the graph, see llama_moe_cache::check_graph
    bool main = false;                               // the thread that queues the graph computes its CPU parts, see llama_moe_cache::main_ok
};

// builds the experts selected by mm_ids up to the down projection, from the MoE cache if cached: [n_embd, n_expert_used, n_tokens]
// cur is the expert input and ids are the selected experts, for the biases and scales (see llm_graph_context::build_moe_ffn)
using llama_moe_cache_build_experts_t = ggml_tensor * (*)(ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * mm_ids, bool cached, void * user_data);

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    // n_threads is the number of CPU threads that compute the misses of small batches within the graph
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, int32_t n_threads);
    ~llama_moe_cache();

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the graph of a MoE layer, built by llm_graph_context::build_moe_ffn in two steps:

    // 1. the lookup of the selected experts in the cache, built after the expert input cur and before the expert matmuls
    //    returns no slots if the layer does not read its experts from the cache for this ubatch
    //    ubatches of up to 8 tokens compute the misses within the graph, without slot lookups (see llama-moe-cpu-misses.h):
    //    the device looks up the slots, the CPU computes a share of the misses, and the other misses are cached by the device or between the graphs
    llama_moe_cache_lookup build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * cur, ggml_tensor * selected_experts) const;

    // 2. the experts of a layer with slots, from the cache and from the host experts: [n_embd, n_expert_used, n_tokens]
    //    the CPU parts that the MoE cache computes while the graph runs go to the graph result (see llama_moe_cache_graph)
    // calls build synchronously; does not retain user_data
    ggml_tensor * build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
            ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // the CPU part of layer il (see llama_moe_cache_cpu_part), before the scheduler allocates the graph
    // the graph of the result is copied and the tensors are not used after the call
    void set_cpu_graph(int32_t il, ggml_tensor * input, ggml_tensor * selected, ggml_tensor * ids, ggml_tensor * result);

    // aborts if the graph has both publish ops and slot lookups, before the scheduler allocates it
    // publish: (layer, experts per token, tokens) of each publish op of the graph, for begin()
    void check_graph(ggml_cgraph * graph, std::vector<int32_t> & publish) const;

    // the thread that queues the graph can compute the CPU parts: no CPU op runs after the first publish op
    bool main_ok(ggml_backend_sched_t sched, ggml_cgraph * graph, ggml_backend_t cpu) const;

    // called before each graph compute, returns false if the CPU part of the last graph failed
    // main: the thread that queues the graph computes its CPU parts in serve(), else a CPU thread of the cache
    // publish: as given by check_graph() for this graph, or nullptr to search the graph
    bool begin(ggml_cgraph * graph, bool main, const std::vector<int32_t> * publish = nullptr);

    // called after the graph is queued: the host replays the slots of the last graph, then serves the CPU parts
    void serve(ggml_backend_t cpu);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
