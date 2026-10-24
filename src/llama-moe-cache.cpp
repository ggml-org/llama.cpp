#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-mailbox.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty
    std::vector<uint8_t>   from_profile; // [n_slots] 1 while the slot holds the expert the profile put there

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        from_profile.assign(n_slots, 0);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    // the i-th hottest expert of a profile takes slot n_slots-1-i: the list runs from the head (least recently used) to
    // the tail, so the coldest prefilled experts are evicted first; -1 when the slots are full or e is cached already
    int32_t prefill(int32_t il, int32_t e, int32_t i) {
        if (i >= n_slots || e >= n_expert || slot_map[il][e] >= 0) {
            return -1;
        }
        const int32_t s = n_slots - 1 - i;
        slot_map[il][e] = s;
        key_of[s] = il*n_expert + e;
        from_profile[s] = 1;
        return s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    // n_hit_profile: the hits on experts the profile put in the cache that were never evicted since
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, size_t & n_hit_profile) {
        fills.clear();
        n_hit = 0;
        n_hit_profile = 0;

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

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
                n_hit_profile += from_profile[slots[e]];
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            from_profile[s] = 0;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
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

// the expert biases of a layer in host memory (--cpu-moe's pattern matches them too)
static std::vector<ggml_tensor *> llama_moe_cache_layer_biases(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps_b, layer.ffn_gate_exps_b, layer.ffn_up_exps_b, layer.ffn_down_exps_b }) {
        if (t != nullptr && llama_moe_cache_is_host_weight(t)) {
            res.push_back(t);
        }
    }
    return res;
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        size_t bias_bytes = 0;            // device copies of the layers' host expert biases, out of the budget
        int32_t n_slots = 0;
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits         = 0;
        size_t hits_profile = 0; // hits on experts the profile put in the cache
        size_t misses       = 0;
        size_t bytes        = 0; // uploaded
        size_t bytes_cpu    = 0; // hybrid: computed on the CPU
    };

    // hybrid: what the CPU op of one layer needs, at a stable address (the op's userdata)
    struct hybrid_layer {
        impl * self = nullptr;
        int32_t il  = -1;
        size_t expert_bytes = 0;     // one expert of the layer, all of its tensors
        std::vector<uint32_t> seen;  // [n_expert] for the distinct-expert counts
        uint32_t seen_gen = 0;
    };

    // hybrid, one token: a layer's hand-overs between the GPU and the CPU, through a mailbox (ggml-mailbox.h)
    struct mailbox_layer {
        ggml_tensor * flag      = nullptr; // I32 [1]: the output of the GPU's send, in the mailbox
        ggml_tensor * ids       = nullptr; // I32 [k, 1]: the selected experts
        ggml_tensor * x         = nullptr; // F32 [n_in, 1, 1]: the layer's input
        ggml_tensor * e         = nullptr; // F32 [n_out, k, 1]: the CPU's experts
        ggml_tensor * slots_dev = nullptr; // I32 [1, n_expert]: the slot map, on the device
        volatile uint32_t * host_flag = nullptr;
        ggml_mailbox_op send = {};
        ggml_mailbox_op recv = {};
    };

    static constexpr int64_t max_batch = 32;

    // adaptive: how the cache follows the conversation (measured on gpt-oss-120b, 400 generated tokens: swapping more often and
    // more per round lifts the hit rate from ~32% to 41-45% from any start at no measurable cost in ms/token)
    // A free slot is worth filling at once: on Qwen3-Coder-Next (16 GB of cache for 36 GB of experts) going from 16 experts a
    // token to 512 lifts the first 256 generated tokens from 21 to 30 t/s, and anything above 512 gains under 2% more.
    static constexpr int32_t rebalance_every      = 32;  // generated tokens between two rebalances
    static constexpr int32_t rebalance_swaps      = 64;  // swaps at most per rebalance...
    static constexpr int32_t rebalance_fill       = 512; // ...while the cache has free slots: every token that missed fills up to this many
    static constexpr float   heat_decay       = 0.998f; // per generated token: a half-life of ~350 tokens
    static constexpr float   swap_margin      = 1.5f;  // an expert comes in when its heat exceeds this times the resident's...
    static constexpr float   swap_min_heat    = 2.0f;  // ...plus this many selections

    int32_t n_expert_used;
    size_t  n_prefilled = 0; // slots filled from the profile at load
    bool    hybrid;
    bool    adaptive;
    std::vector<float>  heat;                // [sum of the layers' experts] decayed selection counts, in units of heat_inc
    std::vector<size_t> heat_off;            // [n_layer + 1]
    float   heat_inc = 1.0f;                 // the weight of a selection now: it grows, which is how older ones decay
    int32_t steps_since_rebalance = 0;
    bool    decode_seen = false;             // a CPU op of the hybrid ran in the last graph
    size_t  n_missed = 0;                    // experts the cache lacked in the graphs since the last fill: candidates for the free slots
    std::atomic<bool> mailbox_error{false};  // a hand-over timed out in the last graph, see take_mailbox_error
    size_t  n_swap = 0;
    // mailbox: how long the CPU waited for the GPU's hand-overs (the time the GPU spends before the experts), and how many
    uint64_t wait_ns = 0;
    uint64_t wait_n  = 0;
    std::vector<hybrid_layer> hybrid_layers; // [n_layer]
    std::vector<uint8_t> layer_full;         // [n_layer] hybrid: the cache holds every expert of the layer

    std::vector<mailbox_layer> mailbox;      // [n_layer], empty without mailboxes
    ggml_backend_buffer_ptr    mailbox_buf;
    ggml_context_ptr           ctx_mailbox;
    ggml_backend_buffer_ptr    buf_slots_dev;
    volatile uint32_t *        epoch_host = nullptr;
    uint32_t                   epoch = 0;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches
    std::vector<stats> stats_layer; // [n_layer] lookups of both batch sizes

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, ggml_tensor *> biases; // host expert biases -> device copies
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            const std::string & profile, bool hybrid, bool adaptive) :
            n_expert_used(model.hparams.n_expert_used_max()), hybrid(hybrid), adaptive(adaptive && hybrid), hybrid_layers(model.layers.size()),
            stats_layer(model.layers.size()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
            for (const ggml_tensor * b : llama_moe_cache_layer_biases(model.layers[il])) {
                it->bias_bytes += GGML_PAD(ggml_nbytes(b), ggml_backend_buft_get_alignment(devices[id].buft));
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) + g.bias_bytes <= budget) {
                g.n_slots++;
            }
            // the LRU needs a slot for each expert of a token; the hybrid never uploads, so any slot is a hit it would not have had
            if (g.n_slots < (hybrid ? 1 : n_expert_used)) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size()) + 4*g.layers.size(); // banks, views, biases
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);

                // the expert biases whole, next to the banks: the ops of the cache's experts run on the device, and a
                // host bias sends each of them back to the CPU, a round trip per bias (gpt-oss: 4 a layer for 1)
                for (ggml_tensor * b : llama_moe_cache_layer_biases(model.layers[il])) {
                    ggml_tensor * copy = ggml_dup_tensor(ctx, b);
                    ggml_format_name(copy, "moe_cache.%s", b->name);
                    biases[b] = copy;
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots) + g.bias_bytes;
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }

            size_t bias_bytes = 0;
            for (const auto & [b, copy] : biases) {
                ggml_backend_tensor_set(copy, b->data, 0, ggml_nbytes(b));
                bias_bytes += ggml_nbytes(b);
            }
            if (!biases.empty()) {
                LLAMA_LOG_INFO("%s: MoE cache holds %zu expert bias tensors (%.2f MiB)\n", __func__, biases.size(), bias_bytes/1024.0/1024.0);
            }

            if (!profile.empty()) {
                prefill(profile, (uint32_t) model.layers.size(), model.hparams.n_expert);
            }
            if (hybrid) {
                if (n_prefilled == 0 && this->adaptive) {
                    LLAMA_LOG_INFO("%s: adaptive hybrid MoE cache without a profile starts empty and fills itself while generating\n", __func__);
                } else if (n_prefilled == 0) {
                    LLAMA_LOG_WARN("%s: hybrid MoE cache without a profile holds no experts: the CPU computes all of them\n", __func__);
                }
                if (this->adaptive) {
                    heat_off.assign(layers.size() + 1, 0);
                    for (size_t il = 0; il < layers.size(); ++il) {
                        heat_off[il + 1] = heat_off[il] + (layers[il].slot_map ? layers[il].slot_map->ne[1] : 0);
                    }
                    heat.assign(heat_off.back(), 0.0f);
                }
                layer_full.assign(layers.size(), 0);
                int32_t n_full = 0;
                for (size_t il = 0; il < layers.size(); ++il) {
                    const layer & l = layers[il];
                    if (l.ig < 0) {
                        continue;
                    }
                    // a layer the cache holds whole needs no CPU at all: it takes the cache's GPU path, no round trip
                    const int32_t * slots = (const int32_t *) l.slot_map->data;
                    layer_full[il] = std::all_of(slots, slots + l.slot_map->ne[1], [](int32_t s) { return s >= 0; });
                    n_full += layer_full[il];
                    hybrid_layer & h = hybrid_layers[il];
                    h.self = this;
                    h.il   = (int32_t) il;
                    for (const ggml_tensor * t : l.experts) {
                        h.expert_bytes += t->nb[2];
                    }
                    h.seen.assign(l.slot_map->ne[1], 0);
                }
                LLAMA_LOG_INFO("%s: hybrid MoE cache: %d layers held whole (GPU only), the others split between GPU and CPU\n",
                    __func__, n_full);
                setup_mailbox(model);
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    // each group takes the hottest experts of its layers from the profile, as many as it has slots
    void prefill(const std::string & path, uint32_t n_layer, uint32_t n_expert) {
        std::vector<std::pair<int32_t, int32_t>> ranked;
        std::string err;
        if (!llama_moe_cache_read_profile(path, n_layer, n_expert, ranked, err)) {
            LLAMA_LOG_WARN("%s: %s, the MoE cache starts empty\n", __func__, err.c_str());
            return;
        }
        size_t n_filled = 0;
        size_t bytes    = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            int32_t i = 0;
            for (const auto & [il, e] : ranked) {
                if (i >= g.n_slots) {
                    break;
                }
                if (layers[il].ig != (int32_t) ig) {
                    continue;
                }
                const int32_t s = g.slru.prefill(il, e, i);
                if (s < 0) {
                    continue;
                }
                i++;
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t expert_size = l.experts[ip]->nb[2];
                    upload(devices[g.id], g.banks[ip], (const uint8_t *) l.experts[ip]->data + e*expert_size, s*expert_size, expert_size);
                    bytes += expert_size;
                }
            }
            n_filled += i;
        }
        flush_uploads();
        n_prefilled = n_filled;
        LLAMA_LOG_INFO("%s: MoE cache filled with %zu experts (%.2f MiB) from %s\n", __func__, n_filled, bytes/1024.0/1024.0, path.c_str());
    }

    // the prefill and the adaptive swaps put expert bytes into the banks through a pinned staging buffer: each upload is queued
    // on the device's stream while the CPU copies the next expert in, instead of a blocking copy from pageable memory per
    // tensor (0.47 ms an expert of Qwen3-Coder-Next's, 2.6 MB). flush_uploads() waits for them; the buffer is then free again.
    // Without pinned host memory (no such buffer type) the copies stay blocking.
    ggml_backend_t          up_backend = nullptr;
    ggml_backend_buffer_ptr up_buf;
    uint8_t *               up_base = nullptr;
    size_t                  up_off  = 0;
    static constexpr size_t up_cap  = 64*1024*1024;

    void upload(device & d, ggml_tensor * bank, const void * src, size_t offset, size_t size) {
        if (up_backend != d.backend) {
            flush_uploads();
            up_backend = d.backend;
            up_buf.reset();
            up_base = nullptr;
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
            if (buft != nullptr) {
                up_buf.reset(ggml_backend_buft_alloc_buffer(buft, up_cap));
                up_base = up_buf ? (uint8_t *) ggml_backend_buffer_get_base(up_buf.get()) : nullptr;
            }
        }
        if (up_base == nullptr || size > up_cap) {
            ggml_backend_tensor_set(bank, src, offset, size);
            return;
        }
        if (up_off + size > up_cap) {
            ggml_backend_synchronize(up_backend);
            up_off = 0;
        }
        std::memcpy(up_base + up_off, src, size);
        ggml_backend_tensor_set_async(up_backend, bank, up_base + up_off, offset, size);
        up_off += size;
    }

    void flush_uploads() {
        if (up_backend != nullptr && up_off > 0) {
            ggml_backend_synchronize(up_backend);
        }
        up_off = 0;
    }

    // mailboxes for the layers the hybrid splits between the GPU and the CPU, if the one device that has host experts has them
    // (CUDA); without, those layers keep the scheduler's copies.
    void setup_mailbox(const llama_model & model) {
        device * dev = nullptr;
        int n_dev = 0;
        for (device & d : devices) {
            if (d.host_bytes > 0 && d.buf) {
                dev = &d;
                n_dev++;
            }
        }
        if (n_dev != 1) {
            return;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(dev->backend));
        auto alloc = (ggml_backend_mailbox_buffer_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_mailbox_buffer");
        if (alloc == nullptr) {
            return;
        }

        struct place { int32_t il; size_t flag, ids, x, e; };
        std::vector<place> places;
        const size_t k = model.hparams.n_expert_used_max();
        size_t size = 64; // the epoch
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0 || is_full((int32_t) il)) {
                continue;
            }
            const size_t n_in  = l.experts.front()->ne[0];
            const size_t n_out = l.experts.back()->ne[1];
            place p = { (int32_t) il, size, 0, 0, 0 };
            p.ids = p.flag + 64;
            p.x   = p.ids + GGML_PAD(4*k, 64);
            p.e   = p.x   + GGML_PAD(4*n_in, 64);
            size  = GGML_PAD(p.e + 4*n_out*k, 4096);
            places.push_back(p);
        }
        if (places.empty()) {
            return;
        }

        void * dev_base = nullptr;
        mailbox_buf.reset(alloc(size, &dev_base));
        if (!mailbox_buf) {
            LLAMA_LOG_WARN("%s: no mailbox memory, the hybrid keeps the scheduler's copies\n", __func__);
            return;
        }
        uint8_t * host = (uint8_t *) ggml_backend_buffer_get_base(mailbox_buf.get());
        epoch_host = (volatile uint32_t *) host;

        ggml_init_params params = { /*.mem_size =*/ 4*places.size()*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ctx_mailbox.reset(ggml_init(params));
        ggml_init_params params_slots = { /*.mem_size =*/ places.size()*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ggml_context_ptr ctx_slots(ggml_init(params_slots));
        if (!ctx_mailbox || !ctx_slots) {
            mailbox_buf.reset();
            return;
        }

        mailbox.assign(layers.size(), mailbox_layer());
        for (const place & p : places) {
            const layer & l = layers[p.il];
            const int64_t n_in  = l.experts.front()->ne[0];
            const int64_t n_out = l.experts.back()->ne[1];
            mailbox_layer & m = mailbox[p.il];
            m.flag = ggml_new_tensor_1d(ctx_mailbox.get(), GGML_TYPE_I32, 1);
            m.ids  = ggml_new_tensor_2d(ctx_mailbox.get(), GGML_TYPE_I32, (int64_t) k, 1);
            m.x    = ggml_new_tensor_3d(ctx_mailbox.get(), GGML_TYPE_F32, n_in, 1, 1);
            m.e    = ggml_new_tensor_3d(ctx_mailbox.get(), GGML_TYPE_F32, n_out, (int64_t) k, 1);
            ggml_format_name(m.flag, "moe_cache.mailbox.flag-%d", p.il);
            ggml_format_name(m.ids,  "moe_cache.mailbox.ids-%d",  p.il);
            ggml_format_name(m.x,    "moe_cache.mailbox.x-%d",    p.il);
            ggml_format_name(m.e,    "moe_cache.mailbox.e-%d",    p.il);
            ggml_backend_tensor_alloc(mailbox_buf.get(), m.flag, host + p.flag);
            ggml_backend_tensor_alloc(mailbox_buf.get(), m.ids,  host + p.ids);
            ggml_backend_tensor_alloc(mailbox_buf.get(), m.x,    host + p.x);
            ggml_backend_tensor_alloc(mailbox_buf.get(), m.e,    host + p.e);
            m.host_flag = (volatile uint32_t *) (host + p.flag);

            m.send.magic     = GGML_MAILBOX_MAGIC;
            m.send.kind      = GGML_MAILBOX_SEND;
            m.send.n_f32     = (int32_t) n_in;
            m.send.n_i32     = (int32_t) k;
            m.send.dev_f32   = (uint8_t *) dev_base + p.x;
            m.send.dev_i32   = (uint8_t *) dev_base + p.ids;
            m.send.dev_flag  = (volatile uint32_t *) ((uint8_t *) dev_base + p.flag);
            m.send.dev_epoch = (const volatile uint32_t *) dev_base;
            m.recv           = m.send;
            m.recv.kind      = GGML_MAILBOX_RECV;
            m.recv.n_f32     = (int32_t) (n_out*k);
            m.recv.dev_f32   = (uint8_t *) dev_base + p.e;


            m.slots_dev = ggml_new_tensor_2d(ctx_slots.get(), GGML_TYPE_I32, 1, l.slot_map->ne[1]);
            ggml_format_name(m.slots_dev, "moe_cache.slots_dev-%d", p.il);
        }
        buf_slots_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_slots.get(), dev->buft));
        if (!buf_slots_dev) {
            mailbox.clear();
            ctx_mailbox.reset();
            mailbox_buf.reset();
            return;
        }
        ggml_backend_buffer_set_usage(buf_slots_dev.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const place & p : places) {
            refresh_slots_dev(p.il);
        }
        ctx_slots_dev = std::move(ctx_slots);
        LLAMA_LOG_INFO("%s: hybrid MoE cache: %zu layers hand over to the CPU through a mailbox (%.2f MiB)\n",
            __func__, places.size(), size/1024.0/1024.0);
    }

    ggml_context_ptr ctx_slots_dev; // keeps the slot maps' tensors alive

    // the device's copy of layer il's slot map, after the host's changed
    void refresh_slots_dev(int32_t il) {
        const layer & l = layers[il];
        ggml_backend_tensor_set(mailbox[il].slots_dev, l.slot_map->data, 0, ggml_nbytes(l.slot_map));
    }

    bool mailbox_ready(int32_t il) const {
        return il >= 0 && il < (int32_t) mailbox.size() && mailbox[il].flag != nullptr;
    }

    // the CPU op of the mailbox path: wait for this step's send of the layer, then split the ids it brought into the experts
    // the cache holds (the GPU has already looked them up on its own) and the ones the CPU computes; the counters as in
    // hybrid_split
    static void mailbox_split(ggml_tensor * dst, int ith, int nth, void * userdata) {
        GGML_UNUSED(nth);
        if (ith != 0) {
            return;
        }
        hybrid_layer & h = *(hybrid_layer *) userdata;
        impl & self = *h.self;
        const mailbox_layer & m = self.mailbox[h.il];
        const uint32_t epoch = self.epoch;
        const int64_t k = m.ids->ne[0];
        int32_t * cold = (int32_t *) dst->data;

        // an earlier layer of this step timed out already: the step fails anyway, do not wait 20 s for every layer
        if (self.mailbox_error) {
            std::fill(cold, cold + k, -1);
            return;
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (uint64_t spins = 0; *m.host_flag != epoch; ++spins) {
            if ((spins & 0xFFFFF) == 0xFFFFF && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(20)) {
                // a stalled GPU fails the step, not the process: the CPU computes no expert (its GPU-side receive is
                // launched after this split anyway) and graph_compute turns the error into a failed decode. A late send
                // is stamped with this epoch, which the next step's no longer matches
                LLAMA_LOG_ERROR("%s: the GPU did not hand layer %d over within 20 s, failing the step\n", __func__, h.il);
                std::fill(cold, cold + k, -1);
                self.mailbox_error = true;
                return;
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        self.wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        self.wait_n++;

        const int32_t * ids   = (const int32_t *) m.ids->data;
        const int32_t * slots = (const int32_t *) dst->src[2]->data;

        // consumed: a graph computed without begin_step() would wait for a new send instead of reading these ids again
        *m.host_flag = 0;

        if (++h.seen_gen == 0) {
            std::fill(h.seen.begin(), h.seen.end(), 0);
            h.seen_gen = 1;
        }
        size_t n_hit = 0;
        size_t n_hit_profile = 0;
        size_t n_cpu = 0;
        const moe_cache_slru & slru = self.groups[self.layers[h.il].ig].slru;
        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = ids[j];
            GGML_ASSERT(e >= 0 && e < (int32_t) h.seen.size());
            const int32_t s = slots[e];
            cold[j] = s >= 0 ? -1 : e;
            if (h.seen[e] != h.seen_gen) {
                h.seen[e] = h.seen_gen;
                (s >= 0 ? n_hit : n_cpu)++;
                n_hit_profile += s >= 0 ? slru.from_profile[s] : 0;
                if (self.adaptive) {
                    self.heat[self.heat_off[h.il] + e] += self.heat_inc;
                }
            }
        }
        self.decode_seen = true;
        self.n_missed += n_cpu;

        stats & st = self.stats_small;
        for (stats * s : { &st, &self.stats_layer[h.il] }) {
            s->hits         += n_hit;
            s->hits_profile += n_hit_profile;
            s->misses       += n_cpu;
            s->bytes_cpu    += n_cpu*h.expert_bytes;
        }
    }

    bool build_mailbox(ggml_context * ctx, int32_t il, ggml_tensor * x, ggml_tensor * ids, llama_moe_cache::mailbox_graph & out) const {
        if (!mailbox_ready(il)) {
            return false;
        }
        const mailbox_layer & m = mailbox[il];
        if (x->type != GGML_TYPE_F32 || ggml_nelements(x) != ggml_nelements(m.x) || !ggml_is_contiguous(x) ||
            ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[0] != m.ids->ne[0] || !ggml_is_contiguous(ids)) {
            return false;
        }
        const int64_t k = ids->ne[0];

        // the GPU: store x and the ids in the mailbox and raise the flag
        ggml_tensor * send_args[2] = { x, ids };
        out.send = ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, send_args, 2, ggml_mailbox_cpu_stub, 1, const_cast<ggml_mailbox_op *>(&m.send));
        ggml_backend_tensor_alloc(mailbox_buf.get(), out.send, (void *) m.flag->data);
        out.x   = m.x;
        out.ids = m.ids;

        // the CPU: the ids of the experts the cache does not hold
        ggml_tensor * split_args[3] = { out.send, m.ids, layers[il].slot_map };
        out.cold_ids = ggml_custom_4d(ctx, GGML_TYPE_I32, k, 1, 1, 1, split_args, 3, mailbox_split, 1, const_cast<hybrid_layer *>(&hybrid_layers[il]));
        out.cold_ids->flags |= GGML_TENSOR_FLAG_NO_SYNC;

        // the GPU looks up its own slots, from its copy of the slot map
        ggml_tensor * rows = ggml_get_rows(ctx, m.slots_dev, ids); // [1, k, 1]
        out.hot_slots = ggml_reshape_2d(ctx, rows, k, 1);
        ggml_tensor * hit = ggml_clamp(ctx, ggml_scale_bias(ctx, ggml_cast(ctx, out.hot_slots, GGML_TYPE_F32), 1.0f, 1.0f), 0.0f, 1.0f);
        out.hit = ggml_reshape_3d(ctx, hit, 1, k, 1);
        return true;
    }

    void build_mailbox_recv(ggml_context * ctx, int32_t il, ggml_tensor * e_cold, ggml_tensor *& copy, ggml_tensor *& recv) const {
        const mailbox_layer & m = mailbox[il];
        // the CPU stores its experts in the mailbox; the GPU reads them from there
        copy = ggml_cpy(ctx, e_cold, m.e);
        ggml_tensor * recv_args[1] = { copy };
        recv = ggml_custom_4d(ctx, GGML_TYPE_F32, m.e->ne[0], m.e->ne[1], 1, 1, recv_args, 1, ggml_mailbox_cpu_stub, 1, const_cast<ggml_mailbox_op *>(&m.recv));
    }

    void begin_step() {
        if (epoch_host != nullptr) {
            epoch++;
            std::atomic_thread_fence(std::memory_order_release);
            *epoch_host = epoch;
        }
        // adaptive: a generated token has passed since the last graph; fill the free slots with the experts it missed, and once
        // none are left, every rebalance_every tokens swap
        if (adaptive && decode_seen) {
            decode_seen = false;
            heat_inc /= heat_decay;
            if (heat_inc > 1e12f) { // the counts are kept in units of the newest selection: scale them back before they overflow
                for (float & h : heat) {
                    h /= heat_inc;
                }
                heat_inc = 1.0f;
            }
            int32_t n_free = 0;
            for (const group & g : groups) {
                n_free += g.n_slots > 0 ? (int32_t) std::count(g.slru.key_of.begin(), g.slru.key_of.end(), -1) : 0;
            }
            if (n_free > 0 && n_missed > 0) {
                const int32_t max_fill = std::min(n_free, rebalance_fill);
                if (do_rebalance(max_fill, /*eager =*/ false, /*fill_only =*/ true) < max_fill) {
                    n_missed = 0; // every candidate is in; a full round leaves the rest for the next token
                }
            }
            if (++steps_since_rebalance >= rebalance_every) { // a group that is full swaps even while another still fills
                steps_since_rebalance = 0;
                do_rebalance(rebalance_swaps, /*eager =*/ false);
            }
        }
    }

    int32_t rebalance_now(int32_t max_swaps) {
        return adaptive ? do_rebalance(max_swaps, /*eager =*/ true) : -1;
    }

    // adaptive: swap the hottest experts the cache lacks in for the coldest it holds (and into free slots first), at most
    // max_swaps. The device must be idle - the banks and its slot maps change - which it is between two generated tokens;
    // the copies are blocking, ~1.2 ms an expert. A layer the cache holds whole is left alone. eager: no hysteresis.
    // fill_only: free slots only, nothing is swapped out.
    // A swap rewrites key_of and the slot maps but not the SLRU lists: adaptive implies hybrid, which never calls
    // plan(), so the list order is unused. Anything that mixes the two would have to move() the reused slots.
    int32_t do_rebalance(int32_t max_swaps, bool eager, bool fill_only = false) {
        if (!adaptive || heat.empty() || max_swaps <= 0) {
            return 0;
        }
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_synchronize(d.backend);
            }
        }
        const int64_t t0 = ggml_time_us();

        struct cand { float h; int32_t il; int32_t e; };
        std::vector<uint8_t> touched(layers.size(), 0);
        int32_t done = 0;
        for (size_t ig = 0; ig < groups.size() && done < max_swaps; ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            std::vector<cand> in;  // the experts the cache lacks that were selected
            std::vector<cand> out; // the experts it holds
            for (int32_t il : g.layers) {
                if (is_full(il)) {
                    continue;
                }
                const int32_t * slots = g.slru.slot_map[il];
                for (int32_t e = 0; e < (int32_t) layers[il].slot_map->ne[1]; ++e) {
                    const float h = heat[heat_off[il] + e];
                    if (slots[e] >= 0) {
                        out.push_back({ h, il, e });
                    } else if (h > 0.0f) {
                        in.push_back({ h, il, e });
                    }
                }
            }
            std::sort(in.begin(),  in.end(),  [](const cand & a, const cand & b) { return a.h > b.h; });
            std::sort(out.begin(), out.end(), [](const cand & a, const cand & b) { return a.h < b.h; });

            std::vector<int32_t> free_slots;
            for (int32_t s = 0; s < g.n_slots; ++s) {
                if (g.slru.key_of[s] < 0) {
                    free_slots.push_back(s);
                }
            }
            size_t i_free = 0;
            size_t i_out  = 0;
            for (size_t i_in = 0; i_in < in.size() && done < max_swaps; ++i_in) {
                const cand & c = in[i_in];
                int32_t s;
                if (i_free < free_slots.size()) {
                    s = free_slots[i_free++];
                } else if (fill_only) {
                    break; // this group is full
                } else if (i_out < out.size()) {
                    const cand & v = out[i_out];
                    if (!(eager ? c.h > v.h : c.h > swap_margin*v.h + swap_min_heat*heat_inc)) {
                        break; // the rest of the candidates are colder, and the rest of the residents hotter
                    }
                    s = g.slru.slot_map[v.il][v.e];
                    g.slru.slot_map[v.il][v.e] = -1;
                    touched[v.il] = 1;
                    i_out++;
                } else {
                    break;
                }
                for (size_t ip = 0; ip < layers[c.il].experts.size(); ++ip) {
                    const size_t size = layers[c.il].experts[ip]->nb[2];
                    upload(devices[g.id], g.banks[ip], (const uint8_t *) layers[c.il].experts[ip]->data + c.e*size, s*size, size);
                }
                g.slru.slot_map[c.il][c.e] = s;
                g.slru.key_of[s]       = c.il*g.slru.n_expert + c.e;
                g.slru.from_profile[s] = 0;
                touched[c.il] = 1;
                done++;
            }
        }
        flush_uploads();
        for (size_t il = 0; il < layers.size(); ++il) {
            if (touched[il] && mailbox_ready((int32_t) il)) {
                refresh_slots_dev((int32_t) il);
            }
        }
        n_swap += done;
        if (done > 0) {
            LLAMA_LOG_DEBUG("%s: %d experts swapped in %.1f ms\n", __func__, done, (ggml_time_us() - t0)/1000.0);
        }
        return done;
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // hybrid: the cache holds every expert of layer il (false before the slot maps exist, e.g. without allocation)
    bool is_full(int32_t il) const {
        return !layer_full.empty() && layer_full[il];
    }

    ggml_tensor * build_hybrid_ids(ggml_context * ctx, ggml_tensor * selected_experts, int32_t il) const {
        const int64_t n_tokens = selected_experts->ne[1];
        if (!hybrid || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || is_full(il) || n_tokens == 0 || n_tokens > max_batch) {
            return nullptr;
        }
        // the op runs on the CPU (the only backend with custom ops) and reads the host slot map in place, so the copy
        // callback never sees it: nothing is uploaded
        ggml_tensor * ids = ggml_is_contiguous(selected_experts) ? selected_experts : ggml_cont(ctx, selected_experts);
        ggml_tensor * args[2] = { ids, layers[il].slot_map };
        return ggml_custom_4d(ctx, GGML_TYPE_I32, selected_experts->ne[0], n_tokens, 4, 1, args, 2, hybrid_split, 1,
                const_cast<hybrid_layer *>(&hybrid_layers[il]));
    }

    // the CPU op of build_hybrid_ids, one task
    static void hybrid_split(ggml_tensor * dst, int ith, int nth, void * userdata) {
        GGML_UNUSED(nth);
        if (ith != 0) {
            return;
        }
        hybrid_layer & h = *(hybrid_layer *) userdata;
        const ggml_tensor * sel   = dst->src[0];
        const int32_t     * slots = (const int32_t *) dst->src[1]->data;
        const int64_t k = sel->ne[0];
        const int64_t n = sel->ne[1];
        auto id = [sel](int64_t j, int64_t t) {
            return *(const int32_t *) ((const char *) sel->data + t*sel->nb[1] + j*sel->nb[0]);
        };
        int32_t * hot      = (int32_t *) dst->data;
        int32_t * cold     = hot  + k*n;
        int32_t * hit      = cold + k*n;
        int32_t * hot_any  = hit  + k*n;

        // the GPU's stand-in for the CPU's experts: a slot the ubatch reads anyway, so a masked row adds a row, not an expert
        int32_t any_slot = -1;
        size_t n_hit = 0;
        size_t n_hit_profile = 0;
        size_t n_cpu = 0;
        const moe_cache_slru & slru = h.self->groups[h.self->layers[h.il].ig].slru;
        if (++h.seen_gen == 0) {
            std::fill(h.seen.begin(), h.seen.end(), 0);
            h.seen_gen = 1;
        }
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t j = 0; j < k; ++j) {
                const int32_t e = id(j, t);
                GGML_ASSERT(e >= 0 && e < (int32_t) h.seen.size());
                const int32_t s = slots[e];
                if (s >= 0 && any_slot < 0) {
                    any_slot = s;
                }
                if (h.seen[e] != h.seen_gen) {
                    h.seen[e] = h.seen_gen;
                    (s >= 0 ? n_hit : n_cpu)++;
                    n_hit_profile += s >= 0 ? slru.from_profile[s] : 0;
                    if (h.self->adaptive) {
                        h.self->heat[h.self->heat_off[h.il] + e] += h.self->heat_inc;
                    }
                }
            }
        }
        h.self->decode_seen = true;
        h.self->n_missed += n_cpu;
        if (any_slot < 0) {
            any_slot = 0;          // nothing hit: slot 0, all masked
        }
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t j = 0; j < k; ++j) {
                const int32_t e = id(j, t);
                const int32_t s = slots[e];
                hot    [t*k + j] = s >= 0 ? s : -1;       // CUDA's one-token mul_mat_id skips -1, like the CPU's
                cold   [t*k + j] = s >= 0 ? -1 : e;       // the CPU's mul_mat_id skips -1: no compute, no read
                hit    [t*k + j] = s >= 0;
                hot_any[t*k + j] = s >= 0 ? s : any_slot; // for the other GPU kernels: a stand-in, masked out
            }
        }

        stats & st = k*n <= 8*h.self->n_expert_used ? h.self->stats_small : h.self->stats_large;
        for (stats * s : { &st, &h.self->stats_layer[h.il] }) {
            s->hits         += n_hit;
            s->hits_profile += n_hit_profile;
            s->misses       += n_cpu;
            s->bytes_cpu    += n_cpu*h.expert_bytes;
        }
    }

    ~impl() {
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        // the hybrid uploads nothing: build_hybrid_ids serves its small batches, except in the layers it holds whole,
        // which take this path - every lookup a hit, so the copy callback never uploads
        if (hybrid && !is_full(il)) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    ggml_tensor * get_bias(const ggml_tensor * b) const {
        const auto it = biases.find(b);
        return it != biases.end() ? it->second : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the hybrid's slot maps never change and it never uploads, so a layer it holds whole needs no look at the
        // selected experts: copy the map as it is, without the sync that reading them costs (~45 us a layer under WSL).
        // Every lookup is a hit, counted from the lookup's size (the distinct experts exactly for one token, at most
        // that for more). Without the selected experts, a hit is known to be on a profile expert only when every
        // expert of the layer still is one (an adaptive fill clears from_profile): otherwise none is counted, a lower bound
        if (hybrid) {
            const int64_t n_ids = ggml_nelements(lookup->src[1]);
            const size_t  n_hit = (size_t) std::min<int64_t>(n_ids, l.slot_map->ne[1]);
            const int32_t * slots = g.slru.slot_map[il];
            bool all_profile = slots != nullptr;
            for (int32_t e = 0; all_profile && e < g.slru.n_expert; ++e) {
                all_profile = slots[e] >= 0 && g.slru.from_profile[slots[e]];
            }
            stats & st = n_ids <= 8*n_expert_used ? stats_small : stats_large;
            for (stats * s : { &st, &stats_layer[il] }) {
                s->hits         += n_hit;
                s->hits_profile += all_profile ? n_hit : 0;
            }
            ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
            return true;
        }

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        size_t n_hit_profile = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit, n_hit_profile)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        for (stats * s : { &st, &stats_layer[il] }) {
            s->hits         += n_hit;
            s->hits_profile += n_hit_profile;
            s->misses       += fills.size();
            s->bytes        += bytes;
        }

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [this](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            // with a profile: how many of the hits are experts it put there, still in the cache
            if (hybrid) {
                LLAMA_LOG_INFO("llama_moe_cache: %s: hybrid: hits = %zu, computed on the CPU = %zu, hit rate = %.2f%%, CPU read %.2f MiB\n",
                    name, st.hits, st.misses, 100.0*st.hits/n, st.bytes_cpu/1024.0/1024.0);
                return;
            }
            const std::string from_profile = n_prefilled == 0 ? "" :
                format(" (%zu from the profile, %.2f%% of lookups)", st.hits_profile, 100.0*st.hits_profile/n);
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu%s, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, from_profile.c_str(), st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (adaptive) {
            LLAMA_LOG_INFO("llama_moe_cache: adaptive: %zu experts swapped in\n", n_swap);
        }
        if (wait_n > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: mailbox: the CPU waited %.1f us on average for each of %llu hand-overs from the GPU\n",
                wait_ns/1000.0/wait_n, (unsigned long long) wait_n);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        // the spread over the layers: a cache that serves some layers well and others not at all says so here
        int32_t lo = -1;
        int32_t hi = -1;
        auto rate = [this](int32_t il) { const stats & s = stats_layer[il]; return (double) s.hits/(s.hits + s.misses); };
        for (int32_t il = 0; il < (int32_t) stats_layer.size(); ++il) {
            if (stats_layer[il].hits + stats_layer[il].misses == 0) {
                continue;
            }
            lo = lo < 0 || rate(il) < rate(lo) ? il : lo;
            hi = hi < 0 || rate(il) > rate(hi) ? il : hi;
        }
        if (lo >= 0 && lo != hi) {
            LLAMA_LOG_INFO("llama_moe_cache: per layer: hit rate from %.2f%% (layer %d) to %.2f%% (layer %d)\n",
                100.0*rate(lo), lo, 100.0*rate(hi), hi);
        }
    }

    int32_t get_layer_stats(llama_moe_cache_layer_stats * out, int32_t n_max) const {
        int32_t n = 0;
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            if (n < n_max) {
                out[n] = { il, (int32_t) groups[layers[il].ig].n_slots, stats_layer[il].hits, stats_layer[il].hits_profile, stats_layer[il].misses };
            }
            n++;
        }
        return n;
    }

    int32_t get_resident(int32_t il, int32_t * out, int32_t n_max) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || layers[il].slot_map->data == nullptr) {
            return 0;
        }
        const ggml_tensor * map   = layers[il].slot_map;
        const int32_t     * slots = (const int32_t *) map->data;
        int32_t n = 0;
        for (int32_t e = 0; e < (int32_t) map->ne[1]; ++e) {
            if (slots[e] >= 0) {
                if (n < n_max) {
                    out[n] = e;
                }
                n++;
            }
        }
        return n;
    }

    // the lookups of both batch sizes (llama_moe_cache_get_stats)
    llama_moe_cache_stats get_stats() const {
        llama_moe_cache_stats res = {};
        for (const stats * st : { &stats_small, &stats_large }) {
            res.n_hit          += st->hits;
            res.n_hit_profile  += st->hits_profile;
            res.n_miss         += st->misses;
            res.n_bytes_upload += st->bytes;
            res.n_bytes_cpu    += st->bytes_cpu;
        }
        res.hybrid = hybrid;
        res.n_copy       = stats_copy.hits;
        res.n_bytes_copy = stats_copy.bytes;
        for (const group & g : groups) {
            res.n_slots += g.n_slots;
        }
        res.n_prefilled = n_prefilled;
        res.n_swap      = n_swap;
        for (const device & d : devices) {
            res.n_bytes_cache += d.buf_size;
        }
        return res;
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        const std::string & profile, bool hybrid, bool adaptive) :
    pimpl(new impl(model, backends, bufts, size, profile, hybrid, adaptive)) {
}

int32_t llama_moe_cache::rebalance(int32_t max_swaps) {
    return pimpl->rebalance_now(max_swaps);
}

ggml_tensor * llama_moe_cache::get_bias(const ggml_tensor * b) const {
    return pimpl->get_bias(b);
}

bool llama_moe_cache::mailbox_ready(int32_t il) const {
    return pimpl->mailbox_ready(il);
}

bool llama_moe_cache::build_mailbox(ggml_context * ctx, int32_t il, ggml_tensor * x, ggml_tensor * ids, mailbox_graph & out) const {
    return pimpl->build_mailbox(ctx, il, x, ids, out);
}

void llama_moe_cache::build_mailbox_recv(ggml_context * ctx, int32_t il, ggml_tensor * e_cold, ggml_tensor *& copy, ggml_tensor *& recv) const {
    pimpl->build_mailbox_recv(ctx, il, e_cold, copy, recv);
}

void llama_moe_cache::begin_step() {
    pimpl->begin_step();
}

bool llama_moe_cache::take_mailbox_error() {
    return pimpl->mailbox_error.exchange(false);
}

ggml_tensor * llama_moe_cache::build_hybrid_ids(ggml_context * ctx, ggml_tensor * selected_experts, int32_t il) const {
    return pimpl->build_hybrid_ids(ctx, selected_experts, il);
}

llama_moe_cache_stats llama_moe_cache::get_stats() const {
    return pimpl->get_stats();
}

int32_t llama_moe_cache::get_layer_stats(llama_moe_cache_layer_stats * stats, int32_t n_max) const {
    return pimpl->get_layer_stats(stats, n_max);
}

int32_t llama_moe_cache::get_resident(int32_t il, int32_t * experts, int32_t n_max) const {
    return pimpl->get_resident(il, experts, n_max);
}

bool llama_moe_cache_read_profile(const std::string & path, uint32_t n_layer, uint32_t n_expert,
        std::vector<std::pair<int32_t, int32_t>> & ranked, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    char magic[4] = {};
    uint32_t hdr[5] = {}; // version, n_layer, n_expert, slots (informational: the cache size it was planned for), n_ranked
    if (!f.read(magic, sizeof(magic)) || std::memcmp(magic, "STRP", 4) != 0 || !f.read((char *) hdr, sizeof(hdr))) {
        err = path + ": not a Strata expert profile";
        return false;
    }
    if (hdr[0] != 1) {
        err = format("%s: profile version %u, expected 1", path.c_str(), hdr[0]);
        return false;
    }
    if (hdr[1] != n_layer || hdr[2] != n_expert) {
        err = format("%s: made for %u layers x %u experts, the model has %u x %u", path.c_str(), hdr[1], hdr[2], n_layer, n_expert);
        return false;
    }
    if ((uint64_t) hdr[4] > (uint64_t) n_layer*n_expert) {
        err = format("%s: ranks %u pairs, the model has only %u x %u", path.c_str(), hdr[4], n_layer, n_expert);
        return false;
    }
    ranked.clear();
    ranked.reserve(hdr[4]);
    for (uint32_t i = 0; i < hdr[4]; ++i) {
        uint16_t p[2];
        if (!f.read((char *) p, sizeof(p)) || p[0] >= n_layer || p[1] >= n_expert) {
            err = format("%s: pair %u is truncated or out of range", path.c_str(), i);
            return false;
        }
        ranked.emplace_back(p[0], p[1]);
    }
    return true;
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
