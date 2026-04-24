#include "llama-moe-expert-manager.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-mmap.h"
#include "llama-hparams.h"

#include "ggml.h"

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <set>

// find which mmap region contains a given data pointer and return the slice info
static llama_expert_slice find_expert_slice(
        const struct ggml_tensor * tensor,
        int expert_id,
        llama_mmaps * mappings) {
    const uint8_t * expert_data = (const uint8_t *)tensor->data + expert_id * tensor->nb[2];
    const size_t    expert_size = tensor->nb[2];

    if (!mappings) {
        return { nullptr, 0, expert_size };
    }

    for (auto & mapping : *mappings) {
        const uint8_t * mmap_start = (const uint8_t *)mapping->addr();
        const uint8_t * mmap_end   = mmap_start + mapping->size();

        if (expert_data >= mmap_start && expert_data < mmap_end) {
            size_t offset = expert_data - mmap_start;
            return { mapping.get(), offset, expert_size };
        }
    }

    // tensor data not in any mmap region (e.g. GPU-offloaded)
    return { nullptr, 0, expert_size };
}

bool llama_moe_expert_manager::init(llama_model & model) {
    active = false;

    if (!model.has_moe_lazy_experts()) {
        return false;
    }

    const auto & hparams = model.hparams;

    if (hparams.n_expert == 0 || hparams.n_expert_used >= hparams.n_expert) {
        LLAMA_LOG_INFO("%s: MoE lazy experts disabled: model has %u experts, %u used per token\n",
                __func__, hparams.n_expert, hparams.n_expert_used);
        return false;
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: MoE lazy experts requires mmap support, disabling\n", __func__);
        return false;
    }

    llama_mmaps * mappings = model.get_mappings();
    if (!mappings || mappings->empty()) {
        LLAMA_LOG_WARN("%s: MoE lazy experts requires mmap, but no mappings found (use_mmap may be disabled)\n", __func__);
        return false;
    }

    const int n_expert = hparams.n_expert;
    const int n_layer  = hparams.n_layer;

    // find the max layer id to size the lookup table
    layer_id_to_idx.assign(n_layer, -1);

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];

        // check if this layer has MoE expert tensors
        if (!layer.ffn_gate_exps && !layer.ffn_gate_up_exps) {
            continue;
        }

        int moe_idx = (int)layers.size();
        layer_id_to_idx[il] = moe_idx;

        moe_layer_data ld;
        ld.layer_id = il;
        ld.n_expert = n_expert;
        ld.expert_slices.resize(n_expert);
        ld.expert_states.resize(n_expert);

        // collect expert tensors for this layer
        std::vector<const struct ggml_tensor *> tensors;
        if (layer.ffn_gate_exps)    { tensors.push_back(layer.ffn_gate_exps); }
        if (layer.ffn_up_exps)      { tensors.push_back(layer.ffn_up_exps); }
        if (layer.ffn_down_exps)    { tensors.push_back(layer.ffn_down_exps); }
        if (layer.ffn_gate_up_exps) { tensors.push_back(layer.ffn_gate_up_exps); }

        for (int e = 0; e < n_expert; e++) {
            ld.expert_slices[e].reserve(tensors.size());
            for (const auto * t : tensors) {
                ld.expert_slices[e].push_back(find_expert_slice(t, e, mappings));
            }
            ld.expert_states[e] = { 0, true }; // initially all considered resident
        }

        layers.push_back(std::move(ld));
    }

    if (layers.empty()) {
        LLAMA_LOG_INFO("%s: no MoE layers found in model\n", __func__);
        return false;
    }

    // init per-query usage tracking
    current_usage.resize(layers.size());
    for (size_t i = 0; i < layers.size(); i++) {
        current_usage[i].assign(layers[i].n_expert, false);
    }

    active = true;

    LLAMA_LOG_INFO("%s: MoE lazy expert loading enabled: %zu MoE layers, %d experts (%u used/token), "
                   "eviction after %llu unused queries\n",
            __func__, layers.size(), n_expert, hparams.n_expert_used,
            (unsigned long long)EVICTION_THRESHOLD);

    return true;
}

void llama_moe_expert_manager::begin_query() {
    for (auto & usage : current_usage) {
        std::fill(usage.begin(), usage.end(), false);
    }
}

void llama_moe_expert_manager::record_topk(int layer_id, const int32_t * ids, int n_expert_used, int n_tokens) {
    if (layer_id < 0 || layer_id >= (int)layer_id_to_idx.size()) {
        return;
    }

    int moe_idx = layer_id_to_idx[layer_id];
    if (moe_idx < 0) {
        return;
    }

    auto & usage = current_usage[moe_idx];
    const int n_expert = layers[moe_idx].n_expert;

    for (int i = 0; i < n_expert_used * n_tokens; i++) {
        int eid = ids[i];
        if (eid >= 0 && eid < n_expert) {
            usage[eid] = true;
        }
    }
}

void llama_moe_expert_manager::end_query() {
    query_counter++;

    // update last-used timestamps from current query
    for (size_t li = 0; li < layers.size(); li++) {
        auto & ld = layers[li];
        for (int e = 0; e < ld.n_expert; e++) {
            if (current_usage[li][e]) {
                ld.expert_states[e].last_used_at = query_counter;
                if (!ld.expert_states[e].resident) {
                    ld.expert_states[e].resident = true;
                }
            }
        }
    }

    evict_stale_experts();
}

void llama_moe_expert_manager::evict_stale_experts() {
    // don't evict until we have enough history
    if (query_counter <= EVICTION_THRESHOLD) {
        return;
    }

    for (auto & ld : layers) {
        for (int e = 0; e < ld.n_expert; e++) {
            auto & state = ld.expert_states[e];

            if (!state.resident) {
                continue;
            }

            if (query_counter - state.last_used_at > EVICTION_THRESHOLD) {
                // evict this expert's pages
                for (const auto & slice : ld.expert_slices[e]) {
                    if (slice.mapping) {
                        slice.mapping->madvise_range(
                                slice.offset_in_mmap,
                                slice.offset_in_mmap + slice.size,
                                true /* dontneed */);
                    }
                }

                state.resident = false;

                LLAMA_LOG_DEBUG("%s: evicted expert %d from layer %d (unused for %llu queries)\n",
                        __func__, e, ld.layer_id,
                        (unsigned long long)(query_counter - state.last_used_at));
            }
        }
    }
}

bool llama_moe_expert_manager::eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * mgr = (llama_moe_expert_manager *)user_data;

    // check if this is a topk tensor: "ffn_moe_topk-{layer_id}"
    const bool is_topk = (strncmp(t->name, "ffn_moe_topk-", 13) == 0);

    if (ask) {
        if (is_topk) {
            return true; // we need this tensor's data
        }
        if (mgr->user_cb) {
            return mgr->user_cb(t, true, mgr->user_cb_data);
        }
        return false;
    }

    // ask == false: tensor was just computed
    if (is_topk) {
        int layer_id = atoi(t->name + 13);
        int n_expert_used = (int)t->ne[0];
        int n_tokens      = (int)t->ne[1];

        std::vector<int32_t> ids(n_expert_used * n_tokens);
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));

        mgr->record_topk(layer_id, ids.data(), n_expert_used, n_tokens);
    }

    if (mgr->user_cb) {
        return mgr->user_cb(t, false, mgr->user_cb_data);
    }
    return true;
}
