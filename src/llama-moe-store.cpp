#include "llama-moe-store.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// LRU map from (layer, expert) to a device slot
struct moe_store_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t> slot_of; // [n_layer*n_expert], -1 if not cached
    std::vector<int32_t> key_of;  // [n_slots], -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    // refcount of the layer currently planned, its slots are pinned against eviction
    std::vector<int32_t> refs; // [n_slots]
    std::vector<int32_t> pinned;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    // decaying per-expert use counts, eviction prefers the cold experts
    std::vector<float> heat; // [n_layer*n_expert]

    static constexpr float   decay_rate = 0.90f; // per graph compute, not per layer, otherwise deeper models forget faster
    static constexpr int32_t n_victims  = 16;    // LRU slots that compete for eviction

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_of.assign((size_t) n_layer*n_expert, -1);
        key_of.assign(n_slots, -1);
        refs.assign(n_slots, 0);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = n_slots > 0 ? 0 : -1;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
        heat.assign((size_t) n_layer*n_expert, 0.0f);
    }

    void pin(int32_t s) {
        refs[s]++;
        pinned.push_back(s);
    }

    void unpin_all() {
        for (int32_t s : pinned) {
            refs[s]--;
        }
        pinned.clear();
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
        int32_t evicted; // (layer, expert) key that was in this slot, -1 if the slot was empty
    };

    void decay() {
        for (float & h : heat) {
            h *= decay_rate;
        }
    }

    // the distinct experts of ids in uniq, sorted so that neighbors can share one upload
    void unique(const int32_t * ids, size_t n_ids) {
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        std::sort(uniq.begin(), uniq.end());
    }

    // the coldest of the first n_victims unpinned slots from the LRU head, empty slots first
    // fresh fills sit at the tail so they can't get kicked out right away. equal heat is just plain LRU
    int32_t evict() const {
        int32_t best = -1;
        int32_t n = 0;
        for (int32_t s = head; s >= 0 && n < n_victims; s = next[s]) {
            if (refs[s] != 0) {
                continue;
            }
            if (key_of[s] < 0) {
                return s;
            }
            if (best < 0 || heat[key_of[s]] < heat[key_of[best]]) {
                best = s;
            }
            n++;
        }
        return best;
    }

    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int32_t * remapped_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        unique(ids, n_ids);
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        const size_t base = (size_t) il*n_expert;

        for (size_t i = 0; i < n_ids; ++i) {
            heat[base + ids[i]] += 1.0f;
        }

        // hits go to the tail first, so that misses are evicted from the cold head
        for (int32_t e : uniq) {
            if (slot_of[base + e] >= 0) {
                touch(slot_of[base + e]);
                pin(slot_of[base + e]);
                n_hit++;
            }
        }
        for (int32_t e : uniq) {
            if (slot_of[base + e] >= 0) {
                continue;
            }
            const int32_t s = evict();
            if (s < 0) {
                return false;
            }
            const int32_t evicted = key_of[s];
            if (evicted >= 0) {
                slot_of[evicted] = -1;
            }
            key_of[s] = base + e;
            slot_of[base + e] = s;
            touch(s);
            pin(s);
            fills.push_back({ e, s, evicted });
        }

        for (size_t i = 0; i < n_ids; ++i) {
            remapped_ids[i] = slot_of[base + ids[i]];
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> moe_store_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool moe_store_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool moe_store_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_store::impl {
    // bigger batches route most experts, they use the expert copy of the scheduler instead of flushing the cache
    static constexpr int64_t max_batch = 32;

    // layers with the same expert layout share the banks and the LRU of a group
    struct group {
        std::vector<ggml_tensor *> ref; // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_store_lru lru;
    };

    struct binding {
        int32_t il;
        int32_t ig;
        ggml_tensor * src;    // host expert tensor
        ggml_tensor * bank;   // device storage of all slots
        ggml_tensor * cached; // view of the bank used in place of src
    };

    struct layer_state {
        std::vector<int32_t> bindings;
        uint64_t planned_epoch = 0;
        std::vector<int32_t> planned_ids;
        std::vector<int32_t> remapped_ids;
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    ggml_backend_t backend;
    bool no_alloc;
    int32_t n_expert;
    int32_t n_expert_used;

    uint64_t epoch = 0;
    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;

    std::vector<group> groups;
    std::vector<moe_store_lru::fill> fills;
    std::vector<int32_t> ids_data;
    std::vector<int32_t> ids_used;

    std::vector<binding> bindings;
    std::vector<layer_state> layers;
    std::unordered_map<const ggml_tensor *, int32_t> binding_of;

    ggml_context_ptr ctx;
    ggml_context_ptr ctx_copy; // the two plane views of a device copy
    ggml_backend_buffer_ptr buf;
    size_t buf_size = 0;

    impl(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
            int32_t n_cache_layers, size_t budget_bytes) :
            backend(backend), no_alloc(model.hparams.no_alloc), n_expert((int32_t) model.hparams.n_expert),
            n_expert_used((int32_t) model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);

        // only layers that keep all of their experts in host memory are managed
        size_t host_total = 0;
        int32_t n_managed = 0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = moe_store_layer_experts(model.layers[il]);
            if (experts.empty() || (int32_t) experts[0]->ne[2] != n_expert || model.dev_layer(il) != dev ||
                !std::all_of(experts.begin(), experts.end(), moe_store_is_host_weight)) {
                continue;
            }
            n_managed++;
            for (const ggml_tensor * t : experts) {
                host_total += ggml_nbytes(t);
            }
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return moe_store_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->ref = experts;
            }
            it->layers.push_back((int32_t) il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE store is disabled\n", __func__);
            return;
        }

        // auto spends the budget as is, a part of a layer is still worth caching
        size_t budget_total = 0;
        if (n_cache_layers < 0) {
            budget_total = std::min(budget_bytes, host_total);
            LLAMA_LOG_INFO("%s: --moe-cache-layers auto: %.0f MiB of device memory\n", __func__, budget_total/1024.0/1024.0);
        } else {
            if (n_cache_layers > n_managed) {
                LLAMA_LOG_WARN("%s: --moe-cache-layers %d exceeds the %d host-resident MoE layers, clamping\n",
                    __func__, n_cache_layers, n_managed);
            }
            budget_total = (size_t) ((double) host_total*std::min(n_cache_layers, n_managed)/n_managed);
        }

        // one extra plane at the end, CUDA MMQ can read past the last expert
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        auto alloc_size = [&](const group & g, int64_t n_planes) {
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*n_planes, alignment);
            }
            return res;
        };

        size_t n_tensors = 0;
        for (group & g : groups) {
            const size_t budget = (size_t) ((double) budget_total*g.host_bytes/host_total);
            const int32_t max_slots = (int32_t) (g.layers.size()*n_expert);
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            g.lru.init((int32_t) model.layers.size(), n_expert, g.n_slots);
            n_tensors += g.ref.size()*(1 + g.layers.size());
        }
        // a layer that runs from the cache needs a slot for each routed expert
        if (std::none_of(groups.begin(), groups.end(), [&](const group & g) { return g.n_slots >= n_expert_used; })) {
            LLAMA_LOG_WARN("%s: not enough device memory for the cache, MoE store is disabled\n", __func__);
            groups.clear();
            return;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        params.mem_size = 2*ggml_tensor_overhead();
        ctx_copy.reset(ggml_init(params));
        if (!ctx || !ctx_copy) {
            throw std::runtime_error("failed to create the MoE store context");
        }

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_store.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                const auto experts = moe_store_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_store.%s", experts[ip]->name);
                    binding_of[experts[ip]] = (int32_t) bindings.size();
                    layers[il].bindings.push_back((int32_t) bindings.size());
                    bindings.push_back({ (int32_t) il, (int32_t) ig, experts[ip], bank, cached });
                }
            }
            buf_size += alloc_size(g, g.n_slots + 1);
        }

        if (no_alloc) {
            // only measure the memory use, see llama_context::memory_breakdown
            buf.reset(ggml_backend_buft_alloc_buffer(buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf.get();
            }
        } else {
            buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
            if (!buf) {
                throw std::runtime_error("failed to allocate the MoE store buffer");
            }
            // not every backend initializes the view tensors when it allocates a context
            for (const binding & b : bindings) {
                if (b.cached->buffer == nullptr && ggml_backend_view_init(b.cached) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("failed to initialize the MoE store view");
                }
            }
            ggml_backend_buffer_clear(buf.get(), 0);
            buf_size = ggml_backend_buffer_get_size(buf.get());
        }

        LLAMA_LOG_INFO("%s: %10s MoE store size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
            ggml_backend_buft_name(buft), buf_size/1024.0/1024.0, host_total/1024.0/1024.0);
        for (const group & g : groups) {
            LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*n_expert));
        }
    }

    ~impl() {
        log_stats();
    }

    bool resolve(const ggml_tensor * node, ggml_backend_t target, ggml_tensor ** cached_weight, void ** cache_entry) {
        if (target != backend) {
            return false;
        }
        const auto it = binding_of.find(node->src[0]);
        if (it == binding_of.end()) {
            return false;
        }
        binding & b = bindings[it->second];

        // big batches and unions above the slot count are left to the expert copy of the scheduler
        const size_t n_ids = ggml_nelements(node->src[2]);
        if (n_ids > (size_t) max_batch*n_expert_used || std::min(n_ids, (size_t) n_expert) > (size_t) groups[b.ig].n_slots) {
            return false;
        }

        *cached_weight = b.cached;
        *cache_entry   = &b;
        return true;
    }

    void begin() {
        if (++epoch == 0) {
            epoch = 1;
            for (auto & l : layers) {
                l.planned_epoch = 0;
            }
        }
        for (group & g : groups) {
            g.lru.unpin_all();
            g.lru.decay();
        }
    }

    // device copy of n consecutive experts from planes of src to planes of dst
    void copy_planes(ggml_tensor * src, int32_t i_src, ggml_tensor * dst, int32_t i_dst, int32_t n) {
        ggml_reset(ctx_copy.get());
        ggml_tensor * a = ggml_view_3d(ctx_copy.get(), src, src->ne[0], src->ne[1], n, src->nb[1], src->nb[2], i_src*src->nb[2]);
        ggml_tensor * b = ggml_view_3d(ctx_copy.get(), dst, dst->ne[0], dst->ne[1], n, dst->nb[1], dst->nb[2], i_dst*dst->nb[2]);
        ggml_backend_view_init(a);
        ggml_backend_view_init(b);
        ggml_backend_tensor_copy_async(backend, backend, a, b);
    }

    bool prepare(const binding * entry, ggml_backend_t ids_backend, const ggml_tensor * ids, ggml_tensor * ids_copy) {
        const size_t n_ids = ggml_nelements(ids);
        if (n_ids == 0) {
            return true;
        }

        ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(ids_backend, ids, ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(ids_backend);

        ids_used.resize(n_ids);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                ids_used[i1*ids->ne[0] + i0] = ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
            }
        }

        layer_state & l = layers[entry->il];

        // a layer split over several splits uploads its experts once, the other projections reuse the plan
        if (l.planned_epoch == epoch) {
            if (l.planned_ids != ids_used) {
                return false;
            }
        } else {
            l.planned_ids  = ids_used;
            l.remapped_ids.resize(n_ids);

            group & g = groups[entry->ig];
            g.lru.unpin_all();

            size_t n_hit = 0;
            if (!g.lru.plan(entry->il, ids_used.data(), n_ids, l.remapped_ids.data(), fills, n_hit)) {
                return false;
            }

            // no wait needed before reusing a slot, the copies and the bank reads run in order on the store backend anyway
            size_t bytes = 0;
            for (int32_t ib : l.bindings) {
                const binding & b = bindings[ib];
                const size_t expert_size = b.src->nb[2];
                for (size_t i = 0; i < fills.size();) {
                    size_t n = 1;
                    while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                        n++;
                    }
                    ggml_backend_tensor_set_async(backend, b.bank, (const uint8_t *) b.src->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                    bytes += n*expert_size;
                    i += n;
                }
            }

            stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;
            st.hits   += n_hit;
            st.misses += fills.size();
            st.bytes  += bytes;

            l.planned_epoch = epoch;
        }

        ggml_backend_tensor_set_async(backend, ids_copy, l.remapped_ids.data(), 0, ggml_nbytes(ids_copy));
        return true;
    }

    bool copy_experts(ggml_backend_t target, const ggml_tensor * src, ggml_tensor * dst, const std::vector<bool> & used) {
        if (target != backend) {
            return false;
        }
        const auto it = binding_of.find(src);
        if (it == binding_of.end()) {
            return false;
        }
        const binding & b = bindings[it->second];
        const moe_store_lru & lru = groups[b.ig].lru;
        const size_t base = (size_t) b.il*n_expert;
        const size_t expert_size = src->nb[2];

        // runs of experts that sit in consecutive slots, or that are all not cached, go in one copy
        // each run also fills the start of the next expert, so that CUDA MMQ does not read NaNs past it
        for (int32_t e = 0; e < n_expert;) {
            if (!used[e]) {
                e++;
                continue;
            }
            const int32_t s = lru.slot_of[base + e];
            int32_t n = 1;
            while (e + n < n_expert && used[e + n] && lru.slot_of[base + e + n] == (s >= 0 ? s + n : -1)) {
                n++;
            }
            const bool pad = e + n < n_expert;
            if (s >= 0) {
                copy_planes(b.bank, s, dst, e, n + pad);
                stats_large.hits += n;
            } else {
                const size_t size = n*expert_size + (pad ? std::min<size_t>(expert_size, 512) : 0);
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + e*expert_size, e*expert_size, size);
                stats_large.misses += n;
                stats_large.bytes  += size;
            }
            e += n;
        }
        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_store: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
    }

    static bool sched_resolve(const ggml_tensor * node, ggml_backend_t backend, ggml_tensor ** cached, void ** handle, void * user_data) {
        return static_cast<impl *>(user_data)->resolve(node, backend, cached, handle);
    }

    static bool sched_prepare(void * handle, ggml_backend_t ids_backend, const ggml_tensor * ids, ggml_tensor * ids_copy, void * user_data) {
        return static_cast<impl *>(user_data)->prepare(static_cast<const binding *>(handle), ids_backend, ids, ids_copy);
    }
};

llama_moe_store::llama_moe_store(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
        int32_t n_cache_layers, size_t budget_bytes) :
    pimpl(new impl(model, backend, buft, n_cache_layers, budget_bytes)) {
}

llama_moe_store::~llama_moe_store() = default;

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_store::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    if (pimpl->buf) {
        res[ggml_backend_buffer_get_type(pimpl->buf.get())] = pimpl->buf_size;
    }
    return res;
}

void llama_moe_store::attach(ggml_backend_sched_t sched) {
    if (!pimpl->groups.empty()) {
        ggml_backend_sched_set_moe_store(sched, pimpl->backend, impl::sched_resolve, impl::sched_prepare, pimpl.get());
    }
}

void llama_moe_store::begin() {
    pimpl->begin();
}

bool llama_moe_store::copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<bool> & used) {
    return !pimpl->groups.empty() && pimpl->copy_experts(backend, src, dst, used);
}
