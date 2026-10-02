#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// LRU map from (layer, expert) to a cache slot
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t> slot_of; // [n_layer*n_expert], -1 if not cached
    std::vector<int32_t> key_of;  // [n_slots], -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_of.assign((size_t) n_layer*n_expert, -1);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
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
    };

    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int32_t * remapped_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

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
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        const size_t base = (size_t) il*n_expert;

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slot_of[base + e] >= 0) {
                touch(slot_of[base + e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slot_of[base + e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_of[key_of[s]] = -1;
            }
            key_of[s] = base + e;
            slot_of[base + e] = s;
            touch(s);
            fills.push_back({ e, s });
        }

        for (size_t i = 0; i < n_ids; ++i) {
            remapped_ids[i] = slot_of[base + ids[i]];
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
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

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // layers with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        std::vector<ggml_tensor *> ref; // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
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

    static constexpr int64_t max_batch = 32;

    ggml_backend_t backend;
    bool no_alloc;
    int32_t n_expert_used;

    uint64_t epoch = 0;
    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;

    std::vector<group> groups;
    std::vector<moe_cache_lru::fill> fills;

    std::vector<binding> bindings;
    std::vector<layer_state> layers;
    std::unordered_map<const ggml_tensor *, int32_t> binding_of;

    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    size_t buf_size = 0;

    impl(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size) :
            backend(backend), no_alloc(model.hparams.no_alloc), n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const auto dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        const int32_t n_expert = model.hparams.n_expert;

        // only cache layers that keep all of their experts in host memory
        size_t host_bytes = 0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || model.dev_layer(il) != dev ||
                !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes += ggml_nbytes(t);
                host_bytes     += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // split the budget by the size of the experts, so each group caches the same fraction of its experts
        size_t n_tensors = 0;
        for (group & g : groups) {
            const size_t budget = (size_t) ((double) size*g.host_bytes/host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors += g.ref.size()*(1 + g.layers.size());
        }
        if (n_tensors == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        ggml_init_params params = {
            /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create the MoE cache context");
        }

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                const auto experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", experts[ip]->name);
                    binding_of[experts[ip]] = bindings.size();
                    layers[il].bindings.push_back(bindings.size());
                    bindings.push_back({ il, (int32_t) ig, experts[ip], bank, cached });
                }
            }
            buf_size += alloc_size(g, g.n_slots);
        }

        if (no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            buf.reset(ggml_backend_buft_alloc_buffer(buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf.get();
            }
        } else {
            buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
            if (!buf) {
                throw std::runtime_error("failed to allocate the MoE cache buffer");
            }
            ggml_backend_buffer_clear(buf.get(), 0);
            buf_size = ggml_backend_buffer_get_size(buf.get());
        }

        LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
            ggml_backend_buft_name(buft), buf_size/1024.0/1024.0, host_bytes/1024.0/1024.0);
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

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        const int64_t n_tokens = node->src[2]->ne[1];
        if (n_tokens > max_batch || std::min(n_tokens*node->src[2]->ne[0], b.src->ne[2]) > groups[b.ig].n_slots) {
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
    }

    bool prepare(const binding * entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids) {
        layer_state & l = layers[entry->il];

        // the experts of all projections of the layer are uploaded on the first call, the others reuse the plan
        if (l.planned_epoch == epoch) {
            if (l.planned_ids.size() != n_ids || !std::equal(l.planned_ids.begin(), l.planned_ids.end(), ids)) {
                return false;
            }
            *remapped_ids = l.remapped_ids.data();
            return true;
        }

        l.planned_ids.assign(ids, ids + n_ids);
        l.remapped_ids.resize(n_ids);

        size_t n_hit = 0;
        if (!groups[entry->ig].lru.plan(entry->il, ids, n_ids, l.remapped_ids.data(), fills, n_hit)) {
            return false;
        }

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
        *remapped_ids = l.remapped_ids.data();
        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size) :
    pimpl(new impl(model, backend, buft, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend() const {
    return pimpl->backend;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    if (pimpl->buf) {
        res[ggml_backend_buffer_get_type(pimpl->buf.get())] = pimpl->buf_size;
    }
    return res;
}

bool llama_moe_cache::sched_resolve(void * user_data, const ggml_tensor * node, ggml_backend_t backend, ggml_tensor ** cached_weight, void ** cache_entry) {
    return static_cast<llama_moe_cache *>(user_data)->pimpl->resolve(node, backend, cached_weight, cache_entry);
}

void llama_moe_cache::sched_begin(void * user_data) {
    static_cast<llama_moe_cache *>(user_data)->pimpl->begin();
}

bool llama_moe_cache::sched_prepare(void * user_data, void * cache_entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids) {
    return static_cast<llama_moe_cache *>(user_data)->pimpl->prepare(static_cast<const impl::binding *>(cache_entry), ids, n_ids, remapped_ids);
}
