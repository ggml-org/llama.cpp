#include "llama-context.h"

#include "llama-model.h"
#include "llama-memory.h"
#include "llama-pipe-shard.h"

#include "ggml-backend.h"

#include <unordered_map>
#include <vector>

void pshard_assign_tensors(
        ggml_backend_sched_t                              sched,
        const llama_model                               & model,
        llama_memory_i                                  * memory,
        const std::vector<ggml_backend_ptr>             & backends,
        const pshard_dev_layout                         & layout,
        ggml_cgraph                                     * gf) {
    const auto & tbids = model.get_tensor_backend_ids();

    for (const auto & [tensor, bid] : tbids) {
        if (bid >= 0 && bid < (int32_t)backends.size()) {
            ggml_backend_sched_set_tensor_backend(sched, tensor, backends[bid].get());
        }
    }

    // catches compute nodes never named via cb()
    // (e.g. wo matmul inside build_attn)
    if (gf) {
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * cur = ggml_graph_node(gf, i);
            if (cur->view_src) continue;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (!cur->src[j]) continue;
                auto it = tbids.find(cur->src[j]);
                if (it != tbids.end() && it->second >= 0 && it->second < (int32_t)backends.size()) {
                    ggml_backend_sched_set_tensor_backend(sched, cur, backends[it->second].get());
                    break;
                }
            }
        }
    }

    if (memory) {
        const auto & lbids = model.get_layer_backend_ids();
        for (auto * ps : memory->get_pipe_shards()) {
            ps->assign_tensors(sched, lbids, backends, layout);
        }
    }
}
