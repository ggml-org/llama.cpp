#include "llama-hot-experts.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <vector>

llama_hot_expert_cache::llama_hot_expert_cache(const llama_model & model,
                                               int32_t             n_pin_experts,
                                               uint64_t            budget_bytes,
                                               uint64_t            stats_interval) :
    model(model),
    n_pin(n_pin_experts),
    budget_bytes(budget_bytes),
    stats_interval(stats_interval) {
    // Count MoE layers by checking which layers have ffn_down_exps.weight
    int32_t n_moe_layers = 0;
    for (int32_t il = 0; il < (int32_t) model.hparams.n_layer(); il++) {
        const std::string name = "blk." + std::to_string(il) + ".ffn_down_exps.weight";
        if (model.get_tensor(name.c_str()) != nullptr) {
            n_moe_layers++;
        }
    }
    n_pin_total = n_pin * n_moe_layers;

    if (n_moe_layers == 0) {
        LLAMA_LOG_WARN("%s: no MoE layers detected in model, --pin-hotexperts has no effect\n", __func__);
    } else if (budget_bytes > 0) {
        LLAMA_LOG_INFO(
            "%s: pinning (mlock) up to %d hottest MoE experts per layer (%d MoE layers, "
            "%d total global slots) in place, budget %.2f MiB total, pin/evict on the fly\n",
            __func__, n_pin, n_moe_layers, n_pin_total, budget_bytes / (1024.0 * 1024.0));
    } else {
        LLAMA_LOG_WARN(
            "%s: --pin-hotexperts has NO memory budget cap (--pin-hotexperts-budget-mib "
            "was not set); with enough layers/experts this WILL try to lock more memory "
            "than physically fits and can be killed by the OOM killer. Setting an explicit "
            "budget that leaves headroom for the KV cache and compute buffers is strongly "
            "recommended.\n",
            __func__);
    }
    if (stats_interval > 0) {
        LLAMA_LOG_INFO("%s: printing pinning stats to stderr every %" PRIu64 " router observations\n", __func__,
                       stats_interval);
    }
    if (!llama_mlock::SUPPORTED) {
        LLAMA_LOG_WARN(
            "%s: mlock is not supported on this platform, --pin-hotexperts will only "
            "track usage statistics and will not actually lock any memory\n",
            __func__);
    }
}

llama_hot_expert_cache::~llama_hot_expert_cache() {
    print_stats();
}

void llama_hot_expert_cache::print_stats() const {
    std::lock_guard<std::mutex> lock(mu);

    size_t   total_distinct_seen  = counts.size();
    size_t   total_pinned         = pinned.size();
    uint64_t global_coldest_count = UINT64_MAX;
    uint64_t global_hottest_count = 0;

    // Per-layer breakdown of pinned experts
    std::unordered_map<int, size_t> pinned_per_layer;
    for (const auto & [key, pe] : pinned) {
        pinned_per_layer[key.layer]++;
    }

    if (!pinned_rank.empty()) {
        global_coldest_count = std::get<0>(*pinned_rank.begin());
        global_hottest_count = std::get<0>(*pinned_rank.rbegin());
    }

    // LLAMA_LOG_WARN (not INFO) so the message is visible with the default
    // verbosity threshold in llama-server (INFO maps to LOG_LEVEL_TRACE=4, which
    // is above the default threshold of 3). The user explicitly enabled this
    // feature, so WARN is the correct level for periodic stats output.
    LLAMA_LOG_WARN("[pin-hotexperts] obs=%" PRIu64
                   " | locked=%.2f MiB | moe_layers=%zu | "
                   "pinned=%zu/%d (global, N=%d x layers=%zu) | distinct (layer,expert) seen=%zu",
                   n_eval_calls, n_bytes_locked / (1024.0 * 1024.0), layers.size(), total_pinned, n_pin_total, n_pin,
                   layers.size(), total_distinct_seen);

    if (!pinned_rank.empty()) {
        LLAMA_LOG_CONT(" | pinned count range=[%" PRIu64 ", %" PRIu64 "]", global_coldest_count, global_hottest_count);
    }

    if (!pinned_per_layer.empty()) {
        LLAMA_LOG_CONT(" | per-layer: {");
        // Sort by layer index for readable output
        std::vector<std::pair<int, size_t>> sorted_layers(pinned_per_layer.begin(), pinned_per_layer.end());
        std::sort(sorted_layers.begin(), sorted_layers.end());
        for (size_t i = 0; i < sorted_layers.size(); i++) {
            const auto & [il, cnt] = sorted_layers[i];
            LLAMA_LOG_CONT("L%d=%zu%s", il, cnt, (i + 1 < sorted_layers.size()) ? ", " : "");
        }
        LLAMA_LOG_CONT("}");
    }
    LLAMA_LOG_CONT("\n");
}

bool llama_hot_expert_cache::eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = static_cast<llama_hot_expert_cache *>(user_data);

    // we only need the *values* of the top-k expert-selection tensor, named
    // "ffn_moe_topk-<il>" by llm_graph_context::cb() -> llama_context::graph_get_cb()
    static const char prefix[] = "ffn_moe_topk-";
    if (strncmp(t->name, prefix, sizeof(prefix) - 1) != 0) {
        return ask ? false : true;  // not interested, let the scheduler carry on either way
    }

    if (ask) {
        // request that the scheduler makes this tensor's data readable on the host
        return true;
    }

    const int il = atoi(t->name + sizeof(prefix) - 1);
    self->on_topk_tensor(il, t);

    return true;
}

void llama_hot_expert_cache::on_topk_tensor(int il, const struct ggml_tensor * t) {
    if (t->type != GGML_TYPE_I32) {
        return;  // unexpected, be defensive rather than misinterpret bytes
    }

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];
    const int64_t n_ids         = n_expert_used * n_tokens;

    std::vector<int32_t> ids(n_ids);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(ids.data(), t->data, n_ids * sizeof(int32_t));
    } else {
        ggml_backend_tensor_get(t, ids.data(), 0, n_ids * sizeof(int32_t));
    }

    uint64_t eval_calls_now = 0;
    {
        std::lock_guard<std::mutex> lock(mu);

        auto & ls = layers[il];
        if (!ls.resolved_tensors) {
            resolve_tensors(il, ls);
        }

        for (int32_t id : ids) {
            if (id < 0) {
                continue;  // padding / unused slot
            }
            observe_expert(il, ls, id);
        }

        n_eval_calls++;
        eval_calls_now = n_eval_calls;
    }  // lock released here

    // print_stats() takes the same mutex itself, so this must run outside the
    // scope above (the mutex is not recursive)
    if (stats_interval > 0 && eval_calls_now % stats_interval == 0) {
        print_stats();
    }
}

void llama_hot_expert_cache::resolve_tensors(int il, layer_state & ls) {
    const std::string base = "blk." + std::to_string(il) + ".";

    ls.t_gate    = model.get_tensor((base + "ffn_gate_exps.weight").c_str());
    ls.t_up      = model.get_tensor((base + "ffn_up_exps.weight").c_str());
    ls.t_down    = model.get_tensor((base + "ffn_down_exps.weight").c_str());
    ls.t_gate_up = model.get_tensor((base + "ffn_gate_up_exps.weight").c_str());

    ls.resolved_tensors = true;

    if (!ls.t_down || (!ls.t_gate && !ls.t_gate_up)) {
        LLAMA_LOG_WARN(
            "%s: layer %d does not look like a (supported) MoE FFN layer, "
            "hot-expert pinning disabled for this layer\n",
            __func__, il);
        return;
    }

    // pinning only makes sense (and is only safe to do in place) when the expert
    // tensors actually live in host memory, e.g. all experts kept on the CPU
    // while only the dense/router parts are offloaded to VRAM
    const ggml_tensor * repr = ls.t_down;
    ls.tensors_are_host      = ggml_backend_buffer_is_host(repr->buffer);

    if (!ls.tensors_are_host) {
        LLAMA_LOG_WARN(
            "%s: layer %d's MoE experts are not in host memory (offloaded to a "
            "device buffer), --pin-hotexperts has no effect for this layer\n",
            __func__, il);
    }
}

void llama_hot_expert_cache::observe_expert(int il, layer_state & ls, int32_t expert_id) {
    expert_key key{ il, expert_id };

    uint64_t &     count     = counts[key];  // default-constructs to 0
    const uint64_t old_count = count;
    count++;
    const uint64_t new_count = count;

    if (!ls.tensors_are_host || n_pin <= 0) {
        return;  // stats-only mode, nothing to pin
    }

    if (pinned.count(key)) {
        // already pinned: keep its ordered-set position up to date
        pinned_rank.erase({ old_count, il, expert_id });
        pinned_rank.insert({ new_count, il, expert_id });
        return;
    }

    if ((int32_t) pinned.size() < n_pin_total) {
        // a pin slot is still free: pin immediately, no eviction needed
        if (pin_expert(il, ls, expert_id)) {
            pinned_rank.insert({ new_count, il, expert_id });
        }
        return;
    }

    // all N slots are taken: only take over if we just overtook the coldest pinned expert
    const auto coldest = pinned_rank.begin();
    if (coldest != pinned_rank.end() && new_count > std::get<0>(*coldest)) {
        const uint64_t evict_count = std::get<0>(*coldest);
        const int      evict_layer = std::get<1>(*coldest);
        const int32_t  evict_id    = std::get<2>(*coldest);

        pinned_rank.erase(coldest);

        auto & evict_ls = layers[evict_layer];
        if (!evict_ls.resolved_tensors) {
            resolve_tensors(evict_layer, evict_ls);
        }
        unpin_expert(evict_layer, evict_ls, evict_id);

        if (pin_expert(il, ls, expert_id)) {
            pinned_rank.insert({ new_count, il, expert_id });
        } else {
            // New expert failed to lock anything (budget exhausted, mlock error, etc.).
            // Roll back the eviction: re-pin the old expert to keep the data structures
            // and actual locked pages consistent.
            pin_expert(evict_layer, evict_ls, evict_id);
            pinned_rank.insert({ evict_count, evict_layer, evict_id });
        }
    }
}

size_t llama_hot_expert_cache::lock_expert_row(const struct ggml_tensor *                    w,
                                               int32_t                                       expert_id,
                                               std::unique_ptr<llama_mlock, mlock_deleter> & out_lock) {
    if (!w || expert_id < 0 || expert_id >= w->ne[2] || !llama_mlock::SUPPORTED) {
        return 0;
    }
    if (!ggml_backend_buffer_is_host(w->buffer) || w->data == nullptr) {
        return 0;
    }

    const size_t nbytes = ggml_nbytes(w) / (size_t) w->ne[2];

    // enforce the global budget BEFORE touching any memory -- mlock() itself
    // faults pages in, so checking after the fact is too late to prevent an OOM
    if (budget_bytes > 0 && n_bytes_locked + nbytes > budget_bytes) {
        LLAMA_LOG_DEBUG(
            "%s: skipping expert %d, would exceed the %.2f MiB pin budget "
            "(%.2f MiB already locked)\n",
            __func__, expert_id, budget_bytes / (1024.0 * 1024.0), n_bytes_locked / (1024.0 * 1024.0));
        return 0;
    }

    const size_t offset = (size_t) expert_id * w->nb[2];
    void *       ptr    = (uint8_t *) w->data + offset;

    // lock the expert's rows IN PLACE inside the model's own tensor -- this is the
    // exact memory ggml_mul_mat_id() reads during build_moe_ffn(), so the lock
    // directly protects the data the compute graph actually uses.
    out_lock.reset(new llama_mlock());
    out_lock->init(ptr);
    out_lock->grow_to(nbytes);

    // only count what was ACTUALLY locked -- grow_to() silently stops (and logs a
    // warning) on failure rather than throwing, so size() may be less than nbytes
    const size_t locked = out_lock->size();
    if (locked < nbytes) {
        LLAMA_LOG_WARN(
            "%s: only locked %zu/%zu bytes for expert %d (system out of lockable "
            "memory?) -- consider lowering --pin-hotexperts N or its budget\n",
            __func__, locked, nbytes, expert_id);
    }
    if (locked == 0) {
        out_lock.reset();
    }

    n_bytes_locked += locked;  // update the running total immediately so sibling
                               // tensors of the SAME expert also respect the budget

    return locked;
}

bool llama_hot_expert_cache::pin_expert(int il, layer_state & ls, int32_t expert_id) {
    if (!ls.tensors_are_host) {
        return false;
    }

    expert_key key{ il, expert_id };
    if (pinned.count(key)) {
        return true;
    }

    pinned_expert pe;

    if (ls.t_gate_up) {
        pe.nbytes_locked += lock_expert_row(ls.t_gate_up, expert_id, pe.gate_up_lock);
    } else if (ls.t_gate) {
        pe.nbytes_locked += lock_expert_row(ls.t_gate, expert_id, pe.gate_lock);
    }
    if (ls.t_up) {
        pe.nbytes_locked += lock_expert_row(ls.t_up, expert_id, pe.up_lock);
    }
    pe.nbytes_locked += lock_expert_row(ls.t_down, expert_id, pe.down_lock);

    // n_bytes_locked was already updated incrementally inside lock_expert_row()
    // (so sibling tensors of this same expert see an up-to-date budget)

    if (pe.nbytes_locked == 0) {
        // Nothing was actually locked (budget exhausted, mlock not supported, etc.).
        // Do not insert a dead entry into the pinned map.
        return false;
    }

    pinned.emplace(key, std::move(pe));
    return true;
}

void llama_hot_expert_cache::unpin_expert(int il, layer_state & /*ls*/, int32_t expert_id) {
    expert_key key{ il, expert_id };
    auto       it = pinned.find(key);
    if (it == pinned.end()) {
        return;
    }

    n_bytes_locked -= it->second.nbytes_locked;
    pinned.erase(it);  // pinned_expert's destructor releases the mlock guards
}

bool llama_hot_expert_cache::is_pinned(int il, int32_t expert_id) const {
    std::lock_guard<std::mutex> lock(mu);

    expert_key key{ il, expert_id };
    return pinned.count(key) != 0;
}
