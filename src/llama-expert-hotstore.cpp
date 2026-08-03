#include "llama-expert-hotstore.h"
#include "llama-expert-heatmap.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <regex>

// matches the weight tensor of an expert tensor, e.g.:
//   blk.0.ffn_gate_exps.weight
//   blk.3.ffn_down_chexps.weight
// follows the same convention as LLM_FFN_EXPS_REGEX in common.h
static const std::regex g_re_exps_weight("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

llama_expert_hotstore::llama_expert_hotstore(
        const llama_model * model, int n_layers, int n_experts, int hot_s, int sync_period) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    bytes_per_slot(n_layers, 0),
    sync_period(sync_period) {
    if (n_layers <= 0) {
        return;
    }

    for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
        std::smatch m;
        if (std::regex_search(name, m, g_re_exps_weight)) {
            const int il = std::stoi(m[1].str());
            if (il >= 0 && il < n_layers && tensor->ne[2] > 0) {
                // a slot holds nbytes/n_experts of this tensor
                bytes_per_slot[il] += ggml_nbytes(tensor) / (size_t) tensor->ne[2];
                entries.push_back({il, tensor, nullptr});
            }
        }
    }

    // entries is fixed from here on; build a per-layer index of stable
    // pointers so copy/resync do not iterate the whole entries vector.
    entries_by_layer.assign(n_layers, {});
    for (auto & e : entries) {
        entries_by_layer[e.layer_idx].push_back(&e);
    }

    if (hot_s > 0) {
        slot_to_expert.assign(n_layers, std::vector<int>(hot_s, -1));
    }
}

bool llama_expert_hotstore::allocate(ggml_backend_buffer_type_t gpu_buft) {
    if (hot_s <= 0 || entries.empty()) {
        return false;
    }
    if (hot_s > n_experts) {
        throw std::runtime_error(format("%s: hot store S=%d exceeds n_experts=%d",
            __func__, hot_s, n_experts));
    }

    // a no_alloc context just for the hot tensor metadata
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * entries.size(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_context_ptr(ggml_init(params));
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: hot store: failed to create ggml context\n", __func__);
        return false;
    }

    // one hot tensor per model expert tensor, with hot_s expert slots plus
    // 1 sentinel slot (index hot_s) that stays zero so cold selections read
    // zeros via a valid in-range index (sentinel trick, oldtricks Trick 2).
    for (auto & e : entries) {
        e.dst = ggml_new_tensor_3d(ctx.get(), e.src->type, e.src->ne[0], e.src->ne[1], hot_s + 1);
    }

    // check whether the buffer would fit before committing any VRAM
    const size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), gpu_buft);
    if (need == 0) {
        LLAMA_LOG_ERROR("%s: hot store: zero-sized buffer, disabled\n", __func__);
        ctx.reset();
        return false;
    }

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(gpu_buft);
    if (dev) {
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    }
    if (dev && free_mem < need) {
        throw std::runtime_error(format("%s: not enough memory to allocate the GPU hot store of %d slots (%zu MiB needed, %zu MiB free on %s)", 
            __func__, hot_s, need / (1024 * 1024), free_mem / (1024 * 1024),
            ggml_backend_dev_name(dev)));
    }

    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), gpu_buft);
    if (b == nullptr) {
        throw std::runtime_error(format("%s: unable to allocate hot store buffer of %d slots (%zu MiB)", 
            __func__, hot_s, need / (1024 * 1024)));
    }
    buf = ggml_backend_buffer_ptr(b);
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // zero the whole buffer so the sentinel slot (index hot_s) AND every
    // not-yet-filled expert slot is zero; copy_top_s/resync_top_s only write
    // slots 0..hot_s-1, so slot hot_s stays zero for the lifetime of the store.
    ggml_backend_buffer_clear(buf.get(), 0);

    return true;
}

void llama_expert_hotstore::copy_top_s(const llama_expert_heatmap & heatmap) {
    if (is_filled || hot_s <= 0 || entries.empty()) {
        return;
    }

    for (int il = 0; il < n_layers; il++) {
        const std::vector<int> top = heatmap.get_top_s(il, hot_s);
        auto & ste = slot_to_expert[il];
        for (int p = 0; p < (int) top.size() && p < hot_s; p++) {
            ste[p] = top[p];
        }

        for (entry * e : entries_by_layer[il]) {
            const size_t slot = ggml_nbytes(e->src) / (size_t) e->src->ne[2];
            const char * src = e->src->data ? (const char *) ggml_get_data(e->src) : nullptr;
            if (!src) {
                continue;
            }
            for (int p = 0; p < hot_s; p++) {
                const int ex = ste[p];
                if (ex < 0) {
                    continue;
                }
                ggml_backend_tensor_set(e->dst, src + (size_t) ex * slot, (size_t) p * slot, slot);
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    is_filled = true;
    LLAMA_LOG("=== Expert hot store: top-S experts copied to GPU ===\n");
}

void llama_expert_hotstore::resync_top_s(const llama_expert_heatmap & heatmap) {
    if (!is_filled || hot_s <= 0) {
        return;
    }

    int swapped = 0;
    for (int il = 0; il < n_layers; il++) {
        const std::vector<int> top = heatmap.get_top_s(il, hot_s);
        auto & ste = slot_to_expert[il];

        // which experts are in the new ranking
        std::vector<char> in_new(n_experts, 0);
        for (int e : top) {
            if (e >= 0 && e < n_experts) {
                in_new[e] = 1;
            }
        }

        // evict: free slots whose resident expert fell out of the new top-S
        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0 && !in_new[ste[p]]) {
                ste[p] = -1;
            }
        }

        // collect new experts that are not currently resident
        std::vector<char> resident_set(n_experts, 0);
        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0) {
                resident_set[ste[p]] = 1;
            }
        }
        std::vector<int> to_place;
        for (int e : top) {
            if (e >= 0 && e < n_experts && !resident_set[e]) {
                to_place.push_back(e);
            }
        }

        // place each new expert into the first free slot, copy its weights
        for (int e : to_place) {
            int p = -1;
            for (int q = 0; q < hot_s; q++) {
                if (ste[q] < 0) {
                    p = q;
                    break;
                }
            }
            if (p < 0) {
                break; // no free slot (should not happen: evictions == new)
            }
            for (entry * ent : entries_by_layer[il]) {
                const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                const char * src = ent->src->data ? (const char *) ggml_get_data(ent->src) : nullptr;
                if (!src) {
                    continue;
                }
                ggml_backend_tensor_set(ent->dst, src + (size_t) e * slot, (size_t) p * slot, slot);
            }
            ste[p] = e;
            swapped++;
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    if (swapped > 0) {
        LLAMA_LOG("=== Expert hot store: re-sync swapped %d expert slots ===\n", swapped);
    }
}

void llama_expert_hotstore::maybe_resync(const llama_expert_heatmap & heatmap) {
    if (sync_period <= 0 || heatmap.tokens_total <= 0) {
        return;
    }
    if (heatmap.tokens_total / sync_period > last_sync_tokens / sync_period) {
        resync_top_s(heatmap);
    }
}

int llama_expert_hotstore::slot_of(int layer_idx, int expert_id) const {
    if (layer_idx < 0 || layer_idx >= n_layers || hot_s <= 0) {
        return -1;
    }
    const auto & ste = slot_to_expert[layer_idx];
    for (int p = 0; p < hot_s; p++) {
        if (ste[p] == expert_id) {
            return p;
        }
    }
    return -1;
}

void llama_expert_hotstore::log() const {
    LLAMA_LOG("=== Expert hotstore sizing (S=%d) ===\n", hot_s);
    size_t total = 0;
    for (int il = 0; il < n_layers; il++) {
        total += bytes_per_slot[il];
        LLAMA_LOG("  layer %3d: bytes/slot = %zu\n", il, bytes_per_slot[il]);
    }
    LLAMA_LOG("  total bytes/slot across all layers = %zu (%zu MiB)\n",
        total, total / (1024 * 1024));
    if (buf) {
        LLAMA_LOG("  GPU hot store allocated: %s, %zu bytes (%zu MiB) for %d+1 slots (%d expert + 1 sentinel)\n",
            ggml_backend_buffer_name(buf.get()),
            ggml_backend_buffer_get_size(buf.get()),
            ggml_backend_buffer_get_size(buf.get()) / (1024 * 1024),
            hot_s, hot_s);
    } else if (hot_s > 0) {
        LLAMA_LOG("  hot store DISABLED (%d slots requested)\n", hot_s);
    }
}
