#include "llama-context.h"

#include "llama-model.h"
#include "llama-memory.h"
#include "llama-pipe-shard.h"
#include "llama-pshard-plan.h"
#include "llama-impl.h"

#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <vector>

void pshard_assign_tensors(
        ggml_backend_sched_t                              sched,
        const llama_model                               & model,
        llama_memory_i                                  * memory,
        const std::vector<ggml_backend_ptr>             & backends,
        const pshard_dev_layout                         & layout) {
    const auto & tbids = model.get_tensor_backend_ids();
    const auto & lbids = model.get_layer_backend_ids();

    for (const auto & [tensor, bid] : tbids) {
        if (bid >= 0 && bid < (int32_t)backends.size()) {
            ggml_backend_sched_set_tensor_backend_hint(sched, tensor, backends[bid].get());
        }
    }

    if (memory) {
        for (auto * ps : memory->get_pipe_shards()) {
            ps->assign_tensors(sched, lbids, backends, layout);
        }
    }
}
