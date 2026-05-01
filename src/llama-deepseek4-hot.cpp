#include "llama-deepseek4-hot.h"

#include "llama.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

using nlohmann::json;

namespace ds4_hot {

struct hot_manager::ggml_buffers {
    std::vector<ggml_context_ptr>       ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};

hot_manager::~hot_manager() = default;

const layer_hot_state * hot_manager::get(int il) const {
    if (il < 0 || (size_t) il >= layers.size()) return nullptr;
    return layers[il].get();
}

size_t hot_manager::total_gpu_bytes() const {
    if (!bufs) return 0;
    size_t total = 0;
    for (const auto & b : bufs->bufs) {
        if (b) total += ggml_backend_buffer_get_size(b.get());
    }
    return total;
}

bool hot_manager::load_profile(std::string path) {
    if (active) return true;

    if (path.empty()) {
        const char * env = std::getenv("DS4_HOT_PROFILE_JSON");
        if (!env || !*env) return false;
        path = env;
    }

    std::ifstream f(path);
    if (!f.good()) {
        LLAMA_LOG_ERROR("ds4-hot: failed to open profile %s\n", path.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("ds4-hot: failed to parse %s: %s\n", path.c_str(), e.what());
        return false;
    }

    if (!j.contains("hot") || !j.contains("k") || !j.contains("n_expert") || !j.contains("n_layer")) {
        LLAMA_LOG_ERROR("ds4-hot: profile missing required fields (hot, k, n_expert, n_layer)\n");
        return false;
    }

    n_layer  = j.value("n_layer", 0);
    n_expert = j.value("n_expert", 0);
    k        = j.value("k", 0);
    category = j.value("category", std::string{});

    if (k <= 0 || n_expert <= 0 || n_layer == 0) {
        LLAMA_LOG_ERROR("ds4-hot: invalid profile dimensions: n_layer=%zu n_expert=%d k=%d\n",
                n_layer, n_expert, k);
        return false;
    }

    layers.resize(n_layer);

    const auto & hot_obj = j["hot"];
    int loaded = 0;
    for (auto it = hot_obj.begin(); it != hot_obj.end(); ++it) {
        int il = std::atoi(it.key().c_str());
        if (il < 0 || (size_t) il >= n_layer) continue;
        if (!it.value().is_array()) continue;

        auto state = std::make_unique<layer_hot_state>();
        state->il = il;
        state->hot_ids.reserve(k);
        state->hot_set.reserve(k);
        for (const auto & v : it.value()) {
            int e = v.is_number_integer() ? v.get<int>() : -1;
            if (e < 0 || e >= n_expert) continue;
            state->hot_ids.push_back(e);
            state->hot_set.insert(e);
            if ((int) state->hot_ids.size() >= k) break;
        }
        state->k = (int) state->hot_ids.size();
        if (state->k <= 0) continue;

        // Build cold set and remap tables.
        state->remap_hot.assign(n_expert, -1);
        state->remap_cold.assign(n_expert, -1);
        for (int idx = 0; idx < state->k; ++idx) {
            state->remap_hot[state->hot_ids[idx]] = idx;
        }

        state->cold_ids.reserve(n_expert - state->k);
        int cold_idx = 0;
        for (int e = 0; e < n_expert; ++e) {
            if (state->hot_set.count(e) == 0) {
                state->cold_ids.push_back(e);
                state->cold_set.insert(e);
                state->remap_cold[e] = cold_idx++;
            }
        }

        layers[il] = std::move(state);
        loaded++;
    }

    LLAMA_LOG_INFO("ds4-hot: loaded profile %s category=%s k=%d n_layer=%zu n_expert=%d (entries=%d)\n",
            path.c_str(), category.c_str(), k, n_layer, n_expert, loaded);

    active = (loaded > 0);
    return active;
}

namespace {

// Track per-device allocations to avoid all hot tensors piling onto one GPU.
struct device_budget {
    ggml_backend_buffer_type_t buft;
    size_t reserved = 0; // bytes already targeted at this buft in current allocate() call
    size_t free_at_start = 0;
};

static std::vector<device_budget> g_budgets;

void init_budgets() {
    g_budgets.clear();
    const int n_dev = ggml_backend_dev_count();
    for (int i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        device_budget b;
        b.buft = ggml_backend_dev_buffer_type(dev);
        b.free_at_start = free;
        b.reserved = 0;
        g_budgets.push_back(b);
    }
}

ggml_backend_buffer_type_t pick_gpu_buft(size_t needed_bytes) {
    // Use 256 MiB safety margin to leave room for other allocations later.
    const size_t margin = 256 * (size_t) 1024 * 1024;

    ggml_backend_buffer_type_t best = nullptr;
    size_t best_remaining = 0;
    for (auto & b : g_budgets) {
        size_t avail = b.free_at_start - std::min(b.free_at_start, b.reserved + margin);
        if (avail < needed_bytes) continue;
        size_t remaining_after = avail - needed_bytes;
        if (remaining_after > best_remaining || best == nullptr) {
            best_remaining = remaining_after;
            best = b.buft;
        }
    }
    if (best) {
        for (auto & b : g_budgets) {
            if (b.buft == best) { b.reserved += needed_bytes; break; }
        }
    }
    return best;
}

} // namespace

bool hot_manager::allocate(const llama_model & model) {
    if (!active) return false;
    if (bufs && !bufs->bufs.empty()) return true; // already allocated

    init_budgets();

    bufs = std::make_unique<ggml_buffers>();

    const auto & m_layers = model.layers;
    if (m_layers.size() != n_layer) {
        LLAMA_LOG_WARN("ds4-hot: profile n_layer=%zu but model has %zu layers; tolerating mismatch\n",
                n_layer, m_layers.size());
    }

    // Build a per-buft ggml context map so we can allocate all hot tensors of
    // a layer that share a destination device into the same backing buffer.
    struct ctx_entry {
        ggml_context_ptr ctx;
        std::vector<std::pair<ggml_tensor **, std::vector<uint8_t>>> pending; // tensor slot + host data to upload
    };
    std::map<ggml_backend_buffer_type_t, ctx_entry> per_buft;

    auto get_ctx = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = per_buft.find(buft);
        if (it != per_buft.end()) return it->second.ctx.get();
        ggml_init_params p = {
            /*.mem_size   =*/ 16 * (size_t) ggml_tensor_overhead() * std::max<size_t>(n_layer, 1),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_owner(ggml_init(p));
        ctx_entry e;
        e.ctx = std::move(ctx_owner);
        ggml_context * raw = e.ctx.get();
        per_buft.emplace(buft, std::move(e));
        return raw;
    };

    int n_alloc_layers = 0;
    size_t total_bytes = 0;

    auto extract_subset = [&](const ggml_tensor * src, const std::vector<int32_t> & hot_ids,
                              const std::string & dest_name, ggml_tensor ** out_tensor) -> bool {
        if (!src) return false;
        if (!src->buffer) return false; // tensor not yet backed (e.g., during params-fit probe)
        if (ggml_n_dims(src) < 3) return false;

        const int64_t ne0 = src->ne[0];
        const int64_t ne1 = src->ne[1];
        const int64_t n_expert_src = src->ne[2];
        if (n_expert_src != n_expert) {
            LLAMA_LOG_WARN("ds4-hot: tensor %s has %ld experts, profile expects %d\n",
                    src->name, (long) n_expert_src, n_expert);
            return false;
        }

        const size_t per_expert_bytes = ggml_nbytes(src) / n_expert_src;
        const int64_t k_local = (int64_t) hot_ids.size();
        const size_t needed = per_expert_bytes * k_local;

        // Pick GPU device with enough room.
        ggml_backend_buffer_type_t buft = pick_gpu_buft(needed);
        if (!buft) {
            LLAMA_LOG_WARN("ds4-hot: no GPU has %.1f MiB free for %s; skipping\n",
                    needed / (1024.0 * 1024.0), src->name);
            return false;
        }

        // Pull source data from wherever it lives (CPU or GPU) into a host buffer
        // we can slice from. Most of the time the source is CPU-resident with -ncmoe.
        std::vector<uint8_t> host_data(ggml_nbytes(src));
        ggml_backend_tensor_get(src, host_data.data(), 0, host_data.size());

        // Build the slice in a separate host buffer.
        std::vector<uint8_t> slice(needed);
        for (int64_t r = 0; r < k_local; ++r) {
            const int32_t e = hot_ids[(size_t) r];
            const size_t src_off = per_expert_bytes * (size_t) e;
            const size_t dst_off = per_expert_bytes * (size_t) r;
            std::memcpy(slice.data() + dst_off, host_data.data() + src_off, per_expert_bytes);
        }

        // Create the destination tensor in the per-buft ggml context.
        ggml_context * ctx = get_ctx(buft);
        if (!ctx) return false;
        ggml_tensor * dst = ggml_new_tensor_3d(ctx, src->type, ne0, ne1, k_local);
        ggml_format_name(dst, "%s.hot", dest_name.c_str());

        // Defer the upload until after we allocate the buffer.
        per_buft[buft].pending.push_back({ out_tensor, std::move(slice) });
        *out_tensor = dst;
        total_bytes += needed;
        return true;
    };

    for (size_t il = 0; il < std::min(n_layer, m_layers.size()); ++il) {
        if (!layers[il]) {
            continue;
        }
        auto & state = *layers[il];
        const auto & lm = m_layers[il];

        // Skip layers that don't have the relevant tensors at all.
        if (!lm.ffn_gate_up_exps && !(lm.ffn_gate_exps && lm.ffn_up_exps)) {
            continue;
        }
        if (!lm.ffn_down_exps) {
            continue;
        }

        // Determine which form the model uses for this layer.
        const bool has_combined = (lm.ffn_gate_up_exps != nullptr);
        const ggml_tensor * probe = has_combined ? lm.ffn_gate_up_exps
                                                 : (lm.ffn_gate_exps ? lm.ffn_gate_exps : lm.ffn_up_exps);

        // Skip early init phases (e.g., the --fit memory probe) where tensors
        // don't have buffers yet.
        if (!probe || !probe->buffer) {
            continue;
        }

        // Skip layers whose tensors are already on a GPU device. The whole
        // point of hot pinning is offloading CPU-resident expert work; if the
        // probe tensor is already on GPU there is nothing to gain.
        bool buf_is_host = ggml_backend_buft_is_host(ggml_backend_buffer_get_type(probe->buffer));
        if (!buf_is_host) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(probe->buffer));
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                layers[il].reset();
                continue;
            }
        }

        bool ok_all = true;
        if (has_combined) {
            ok_all &= extract_subset(lm.ffn_gate_up_exps, state.hot_ids,
                                     "ds4_hot_gate_up_exps_l" + std::to_string(il),
                                     &state.hot_gate_up_exps);
        } else {
            ok_all &= extract_subset(lm.ffn_gate_exps, state.hot_ids,
                                     "ds4_hot_gate_exps_l" + std::to_string(il),
                                     &state.hot_gate_exps);
            ok_all &= extract_subset(lm.ffn_up_exps, state.hot_ids,
                                     "ds4_hot_up_exps_l" + std::to_string(il),
                                     &state.hot_up_exps);
        }
        ok_all &= extract_subset(lm.ffn_down_exps, state.hot_ids,
                                 "ds4_hot_down_exps_l" + std::to_string(il),
                                 &state.hot_down_exps);
        if (!ok_all) {
            // Free anything partially allocated for this layer.
            state.hot_gate_up_exps = nullptr;
            state.hot_gate_exps    = nullptr;
            state.hot_up_exps      = nullptr;
            state.hot_down_exps    = nullptr;
            layers[il].reset();
            continue;
        }
        n_alloc_layers++;
    }

    // Now allocate backing buffers and upload the slices.
    for (auto & [buft, e] : per_buft) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(e.ctx.get(), buft);
        if (!buf) {
            LLAMA_LOG_WARN("ds4-hot: could not allocate hot buffer for buft %s; skipping affected layers\n",
                    ggml_backend_buft_name(buft));
            // Null out the tensor pointers since they're not actually backed.
            for (auto & p : e.pending) {
                if (p.first) *p.first = nullptr;
            }
            continue;
        }
        for (auto & p : e.pending) {
            if (!*p.first) continue;
            ggml_backend_tensor_set(*p.first, p.second.data(), 0, p.second.size());
        }
        bufs->bufs.emplace_back(buf);
        bufs->ctxs.emplace_back(std::move(e.ctx));
    }

    // Recount: a layer is fully usable only if all its tensor pointers are non-null after upload.
    int n_usable = 0;
    for (auto & lp : layers) {
        if (!lp) continue;
        const bool combined = lp->hot_gate_up_exps != nullptr;
        const bool separate = lp->hot_gate_exps != nullptr && lp->hot_up_exps != nullptr;
        if (lp->hot_down_exps && (combined || separate)) {
            n_usable++;
        } else {
            lp.reset();
        }
    }

    LLAMA_LOG_INFO("ds4-hot: pinned hot experts for %d/%d CPU-MoE layers, ~%.1f MiB on GPU across %zu buffers (k=%d, category=%s)\n",
            n_usable, n_alloc_layers, total_bytes / (1024.0 * 1024.0), bufs->bufs.size(), k, category.c_str());

    return n_usable > 0;
}

hot_manager & instance() {
    static hot_manager mgr;
    return mgr;
}

} // namespace ds4_hot
