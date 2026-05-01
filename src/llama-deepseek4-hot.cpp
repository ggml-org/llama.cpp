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

// Pick the GPU buffer type with the most remaining headroom that can fit
// `needed_bytes`. Reserves the bytes immediately so subsequent picks see
// the running total.
//
// If DS4_HOT_DEVICE is set in the environment, restrict picking to the
// matching CUDAN device (e.g. `DS4_HOT_DEVICE=CUDA0`). This is useful for
// debugging the dispatch path: pinning all hot tensors onto one GPU
// eliminates a class of multi-device scheduler interactions that have
// triggered illegal-memory-access crashes on certain prompts.
//
// Margin (default 1.5 GiB per device) is left untouched so prefill compute
// buffers can fit. DS4_HOT_MARGIN_MIB overrides this; use a larger value if
// you observe OOM errors during prefill of long prompts.
ggml_backend_buffer_type_t pick_gpu_buft(size_t needed_bytes) {
    static const size_t margin = []() -> size_t {
        const char * env = std::getenv("DS4_HOT_MARGIN_MIB");
        if (!env || !*env) return (size_t) 1536 * 1024 * 1024; // 1.5 GiB default
        long v = std::strtol(env, nullptr, 10);
        if (v <= 0) return (size_t) 1536 * 1024 * 1024;
        return (size_t) v * 1024 * 1024;
    }();

    static const char * const force_device = std::getenv("DS4_HOT_DEVICE");

    ggml_backend_buffer_type_t best = nullptr;
    size_t best_remaining = 0;
    for (auto & b : g_budgets) {
        if (force_device && force_device[0]) {
            if (std::strcmp(ggml_backend_buft_name(b.buft), force_device) != 0) {
                continue;
            }
        }
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

    // Pending uploads: a tensor pointer slot + the host bytes to copy into it.
    struct pending_upload {
        ggml_tensor ** slot;
        std::vector<uint8_t> data;
    };

    // Per-buft (i.e., per-GPU device) ggml_context that aggregates all hot
    // tensors + lookup tables targeted at that device. We allocate one backing
    // buffer per buft after the loop.
    struct ctx_entry {
        ggml_context_ptr ctx;
        std::vector<pending_upload> pending;
    };
    std::map<ggml_backend_buffer_type_t, ctx_entry> per_buft;

    auto get_ctx = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = per_buft.find(buft);
        if (it != per_buft.end()) return it->second.ctx.get();
        // Reserve enough space for ~16 tensors per layer (3 weight + 4 lookup + headroom).
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

    // Compute the total bytes one layer's hot tensors + lookup tables need so
    // we can reserve all of them on the SAME device. This is essential — if
    // gate_h, up_h, down_h end up on different GPUs the dual dispatch becomes
    // a multi-backend mess and we lose the placement benefit.
    auto layer_total_bytes = [&](int il, const llama_layer & lm) -> size_t {
        size_t total = 0;
        const layer_hot_state & st = *layers[il];
        const int64_t k_local = (int64_t) st.hot_ids.size();
        auto add_tensor = [&](const ggml_tensor * src) {
            if (!src) return;
            total += (ggml_nbytes(src) / src->ne[2]) * k_local;
        };
        if (lm.ffn_gate_up_exps) {
            add_tensor(lm.ffn_gate_up_exps);
        } else {
            add_tensor(lm.ffn_gate_exps);
            add_tensor(lm.ffn_up_exps);
        }
        add_tensor(lm.ffn_down_exps);
        // Lookup tables live in CPU buffer so they don't count against GPU budget.
        return total;
    };

    auto extract_subset = [&](ggml_backend_buffer_type_t buft, const ggml_tensor * src,
                              const std::vector<int32_t> & hot_ids,
                              const std::string & dest_name, ggml_tensor ** out_tensor) -> bool {
        if (!src) return false;
        if (!src->buffer) return false;

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
        // Allocate K + P + 1 experts:
        //   [0, K)        - real hot experts
        //   [K, K+P)      - per-pick dummy experts (zero-weighted; never collide
        //                   with real expert IDs across picks of a single token,
        //                   which fixes the CUDA mm_ids_helper dedup crash)
        //   [K+P]         - trailing prefetch padding slot (kernel reads ahead)
        const int64_t P = n_picks_;
        const int64_t k_alloc = k_local + P + 1;
        const size_t needed   = per_expert_bytes * k_alloc;

        // Pull source data from CPU into a host buffer we can slice from.
        std::vector<uint8_t> host_data(ggml_nbytes(src));
        ggml_backend_tensor_get(src, host_data.data(), 0, host_data.size());

        // Build the slice in a separate host buffer (zero-initialized so the
        // dummy experts and trailing prefetch slot all hold zeros).
        std::vector<uint8_t> slice(needed, 0);
        for (int64_t r = 0; r < k_local; ++r) {
            const int32_t e = hot_ids[(size_t) r];
            const size_t src_off = per_expert_bytes * (size_t) e;
            const size_t dst_off = per_expert_bytes * (size_t) r;
            std::memcpy(slice.data() + dst_off, host_data.data() + src_off, per_expert_bytes);
        }

        ggml_context * ctx = get_ctx(buft);
        if (!ctx) return false;
        ggml_tensor * dst = ggml_new_tensor_3d(ctx, src->type, ne0, ne1, k_alloc);
        ggml_format_name(dst, "%s.hot", dest_name.c_str());

        per_buft[buft].pending.push_back({ out_tensor, std::move(slice) });
        *out_tensor = dst;
        total_bytes += needed;
        return true;
    };

    auto add_lookup_f32 = [&](ggml_backend_buffer_type_t buft, const std::string & name,
                              const std::vector<float> & values, int64_t ne0, int64_t ne1,
                              ggml_tensor ** out_tensor) -> bool {
        ggml_context * ctx = get_ctx(buft);
        if (!ctx) return false;
        ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_format_name(dst, "%s", name.c_str());
        std::vector<uint8_t> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        per_buft[buft].pending.push_back({ out_tensor, std::move(bytes) });
        *out_tensor = dst;
        total_bytes += values.size() * sizeof(float);
        return true;
    };

    for (size_t il = 0; il < std::min(n_layer, m_layers.size()); ++il) {
        if (!layers[il]) {
            continue;
        }
        auto & state = *layers[il];
        const auto & lm = m_layers[il];

        if (!lm.ffn_gate_up_exps && !(lm.ffn_gate_exps && lm.ffn_up_exps)) {
            continue;
        }
        if (!lm.ffn_down_exps) {
            continue;
        }

        const bool has_combined = (lm.ffn_gate_up_exps != nullptr);
        const ggml_tensor * probe = has_combined ? lm.ffn_gate_up_exps
                                                 : (lm.ffn_gate_exps ? lm.ffn_gate_exps : lm.ffn_up_exps);

        if (!probe || !probe->buffer) {
            continue;
        }

        bool buf_is_host = ggml_backend_buft_is_host(ggml_backend_buffer_get_type(probe->buffer));
        if (!buf_is_host) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(probe->buffer));
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                layers[il].reset();
                continue;
            }
        }

        // Pick ONE GPU for all of this layer's hot tensors + lookup tables.
        const size_t needed = layer_total_bytes((int) il, lm);
        ggml_backend_buffer_type_t buft = pick_gpu_buft(needed);
        if (!buft) {
            LLAMA_LOG_WARN("ds4-hot: no GPU has %.1f MiB free for layer %zu hot pack; skipping\n",
                    needed / (1024.0 * 1024.0), il);
            layers[il].reset();
            continue;
        }

        // Hot-side lookup tables (hot_remap, is_hot) live on the SAME GPU as the
        // hot weights so the get_rows + mul_mat_id chain can run entirely on
        // that GPU without any cross-backend transfer of the per-pick IDs.
        // Cold-side tables (cold_remap, is_cold) live on CPU so the cold
        // mul_mat_id (CPU weights) consumes a CPU IDs tensor without sched
        // having to bounce data between backends each step.
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();

        // Build per-layer lookup tables.
        // For the hot path we use float arithmetic in the graph to construct
        // per-pick unique IDs:
        //   hot_id[k,t] = hot_remap_table[selected[k,t]] + is_cold[k,t] * hot_pick_arange[k]
        // For hot picks the arange contribution is 0 -> hot_id in [0, K).
        // For cold picks the base is K (sentinel) and the arange adds k -> id in [K, K+P).
        // This guarantees all P picks within a token map to distinct expert IDs,
        // which is required by the CUDA mm_ids_helper kernel (it dedups
        // (token, expert) pairs and the downstream quantize kernel reads
        // exactly P*T compact rows).
        const int P = n_picks_;

        std::vector<float> hot_remap_vals((size_t) n_expert, (float) state.k); // base sentinel = K
        std::vector<float> cold_remap_vals((size_t) n_expert, 0.0f);
        std::vector<float> is_hot_vals((size_t) n_expert, 0.0f);
        std::vector<float> is_cold_vals((size_t) n_expert, 1.0f);
        for (int32_t e : state.hot_ids) {
            hot_remap_vals[(size_t) e]  = (float) state.remap_hot[(size_t) e];
            is_hot_vals[(size_t) e]     = 1.0f;
            is_cold_vals[(size_t) e]    = 0.0f;
        }
        for (int32_t e : state.cold_ids) {
            cold_remap_vals[(size_t) e] = (float) e;
        }

        // Per-pick arange [0, 1, ..., P-1]
        std::vector<float> pick_arange_vals((size_t) P);
        for (int i = 0; i < P; ++i) pick_arange_vals[(size_t) i] = (float) i;

        // Per-pick cold sentinel: cold_ids[k % n_cold] for k in [0, P).
        // Each hot pick within a token gets a different cold expert as
        // sentinel, so the cold path's CPU mul_mat_id also sees per-token
        // unique IDs (defensive - the CPU helper might handle dedup
        // correctly, but making both paths uniform is safer).
        std::vector<float> cold_sentinel_vals((size_t) P);
        const int n_cold = (int) state.cold_ids.size();
        for (int i = 0; i < P; ++i) {
            cold_sentinel_vals[(size_t) i] = n_cold > 0 ? (float) state.cold_ids[(size_t) (i % n_cold)] : 0.0f;
        }

        bool ok_all = true;
        if (has_combined) {
            ok_all &= extract_subset(buft, lm.ffn_gate_up_exps, state.hot_ids,
                                     "ds4_hot_gate_up_exps_l" + std::to_string(il),
                                     &state.hot_gate_up_exps);
        } else {
            ok_all &= extract_subset(buft, lm.ffn_gate_exps, state.hot_ids,
                                     "ds4_hot_gate_exps_l" + std::to_string(il),
                                     &state.hot_gate_exps);
            ok_all &= extract_subset(buft, lm.ffn_up_exps, state.hot_ids,
                                     "ds4_hot_up_exps_l" + std::to_string(il),
                                     &state.hot_up_exps);
        }
        ok_all &= extract_subset(buft, lm.ffn_down_exps, state.hot_ids,
                                 "ds4_hot_down_exps_l" + std::to_string(il),
                                 &state.hot_down_exps);

        // Track per-layer pick count for downstream graph builder access.
        state.n_picks = P;

        ok_all &= add_lookup_f32(buft,     "ds4_hot_remap_l"  + std::to_string(il),
                                 hot_remap_vals, 1, n_expert, &state.hot_remap_table);
        ok_all &= add_lookup_f32(cpu_buft, "ds4_cold_remap_l" + std::to_string(il),
                                 cold_remap_vals, 1, n_expert, &state.cold_remap_table);
        ok_all &= add_lookup_f32(buft,     "ds4_is_hot_l"  + std::to_string(il),
                                 is_hot_vals, 1, n_expert, &state.is_hot_mask);
        ok_all &= add_lookup_f32(cpu_buft, "ds4_is_cold_l" + std::to_string(il),
                                 is_cold_vals, 1, n_expert, &state.is_cold_mask);
        // Per-pick constants live as [P, 1] tensors so they broadcast against
        // [P, T] when multiplied. Hot side on GPU, cold side on CPU.
        ok_all &= add_lookup_f32(buft,     "ds4_hot_pick_arange_l"   + std::to_string(il),
                                 pick_arange_vals, P, 1, &state.hot_pick_arange);
        ok_all &= add_lookup_f32(cpu_buft, "ds4_cold_pick_sentinel_l" + std::to_string(il),
                                 cold_sentinel_vals, P, 1, &state.cold_pick_sentinel);

        if (!ok_all) {
            state.hot_gate_up_exps   = nullptr;
            state.hot_gate_exps      = nullptr;
            state.hot_up_exps        = nullptr;
            state.hot_down_exps      = nullptr;
            state.hot_remap_table    = nullptr;
            state.cold_remap_table   = nullptr;
            state.is_hot_mask        = nullptr;
            state.is_cold_mask       = nullptr;
            state.hot_pick_arange    = nullptr;
            state.cold_pick_sentinel = nullptr;
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
            for (auto & p : e.pending) {
                if (p.slot) *p.slot = nullptr;
            }
            continue;
        }
        // Mark as model weights so the scheduler keeps the consuming ops on
        // this device (matches normal model-weight placement semantics).
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & p : e.pending) {
            if (!p.slot || !*p.slot) continue;
            ggml_backend_tensor_set(*p.slot, p.data.data(), 0, p.data.size());
        }
        bufs->bufs.emplace_back(buf);
        bufs->ctxs.emplace_back(std::move(e.ctx));
    }

    // Re-validate: a layer is fully usable only if EVERY required tensor and
    // lookup table is non-null after upload.
    int n_usable = 0;
    for (auto & lp : layers) {
        if (!lp) continue;
        if (lp->ready_for_dispatch()) {
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
