#include "llama-moe-cache.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-moe-cpu-misses.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
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

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a CRF >= 1, as after use(): log2 is >= 0 and the cast is the floor
    static int class_of_crf(float x) {
        return std::min(n_class - 1, (int) std::log2(x));
    }

    int class_of(int32_t key) const {
        return class_of_crf(crf[key]);
    }

    // the decay of a CRF after d tokens
    static float decay(int64_t d) {
        return std::exp2(-d/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    // fma rounds the same on all hosts and devices, a separate multiply and add may or may not be contracted
    void use(int32_t il, int32_t key) {
        crf[key]      = std::fma(crf[key], decay(clock[il] - last_use[key]), 1.0f);
        last_use[key] = clock[il];
    }

    // decay() for d < GGML_MOE_CPU_N_DECAY, then the least CRF of each class, see ggml_moe_cpu_layer::decay
    // a device reads them instead of computing exp2 and log2, its functions round differently
    static std::vector<float> device_tables() {
        std::vector<float> res(GGML_MOE_CPU_N_DECAY + n_class);
        for (int d = 0; d < GGML_MOE_CPU_N_DECAY; ++d) {
            res[d] = decay(d);
        }
        // log2 rounds up to c for some CRF below 2^c, so search for the least CRF of class c among the floats in [1, 2^n_class]
        auto bits  = [](float x)    { uint32_t u; memcpy(&u, &x, sizeof(u)); return u; };
        auto value = [](uint32_t u) { float x;    memcpy(&x, &u, sizeof(x)); return x; };
        res[GGML_MOE_CPU_N_DECAY] = 0.0f;
        for (int c = 1; c < n_class; ++c) {
            uint32_t lo = bits(1.0f); // class < c
            uint32_t hi = bits(std::ldexp(1.0f, n_class)); // class >= c
            while (hi - lo > 1) {
                const uint32_t mid = lo + (hi - lo)/2;
                (class_of_crf(value(mid)) >= c ? hi : lo) = mid;
            }
            res[GGML_MOE_CPU_N_DECAY + c] = value(hi);
        }
        // the decay after GGML_MOE_CPU_N_DECAY tokens is 0 for the device
        GGML_ASSERT(std::fma(std::ldexp(1.0f, n_class), decay(GGML_MOE_CPU_N_DECAY), 1.0f) == 1.0f);
        return res;
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    // replay the slots that a device gave to the experts selected by ids (n_tokens tokens) in layer il, in the order of plan():
    // the CRF of all selected experts is updated and hits are promoted, then the misses cached by the device are put in their slots in expert order
    // slots at n_slots or later are misses computed on the CPU, they are not cached
    void apply(int32_t il, const int32_t * ids, const int32_t * slots, size_t n_ids, int64_t n_tokens, size_t & n_hit, size_t & n_fill, size_t & n_cpu) {
        n_hit  = 0;
        n_fill = 0;
        n_cpu  = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear(); // the index of the first use of each expert
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert && slots[i] >= 0);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back((int32_t) i);
            }
        }

        int32_t * map = slot_map[il];

        misses.clear(); // the index of each miss cached by the device
        for (int32_t i : uniq) {
            const int32_t e   = ids[i];
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[i] >= n_slots) {
                map[e] = slots[i];
                n_cpu++;
            } else if (map[e] == slots[i]) {
                move(slots[i], class_of(key));
                n_hit++;
            } else {
                misses.push_back(i);
            }
        }
        std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) { return ids[a] < ids[b]; });
        for (int32_t i : misses) {
            const int32_t e   = ids[i];
            const int32_t s   = slots[i];
            const int32_t key = make_key(il, e);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            key_of[s] = key;
            map[e]    = s;
            move(s, class_of(key));
            n_fill++;
        }
        clock[il] += n_tokens;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
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

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        return true;
    }
};

// the device runs the same ClassLRU, see ggml_moe_cpu_layer::lru
static_assert(moe_cache_lru::n_class == GGML_MOE_CPU_N_CLASS, "n_class mismatch");
static_assert(moe_cache_lru::crf_half_life == GGML_MOE_CPU_CRF_HALF_LIFE, "crf_half_life mismatch");

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

static void llama_moe_cache_cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

// the request is in memory shared with the device, its fields are read and written as atomics
template <typename T>
static std::atomic<T> & llama_moe_cache_atomic(T & v) {
    static_assert(sizeof(std::atomic<T>) == sizeof(T) && std::atomic<T>::is_always_lock_free);
    return reinterpret_cast<std::atomic<T> &>(v);
}

// copy of the graph of t in ctx, the tensors in map are replaced, the host weights are kept
static ggml_tensor * llama_moe_cache_copy_graph(ggml_context * ctx, ggml_tensor * t, std::unordered_map<const ggml_tensor *, ggml_tensor *> & map) {
    if (t == nullptr) {
        return nullptr;
    }
    const auto it = map.find(t);
    if (it != map.end()) {
        return it->second;
    }
    ggml_tensor * res = t;
    if (t->op != GGML_OP_NONE || t->view_src != nullptr) {
        // the sources first: the context is allocated in order, and a view needs its source allocated
        ggml_tensor * src[GGML_MAX_SRC];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            src[j] = llama_moe_cache_copy_graph(ctx, t->src[j], map);
        }
        ggml_tensor * view_src = llama_moe_cache_copy_graph(ctx, t->view_src, map);

        res = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
        memcpy(res->nb, t->nb, sizeof(t->nb));
        memcpy(res->op_params, t->op_params, sizeof(t->op_params));
        res->op    = t->op;
        res->flags = t->flags;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            res->src[j] = src[j];
        }
        if (view_src != nullptr) {
            res->view_src  = view_src;
            res->view_offs = t->view_offs;
        }
        ggml_set_name(res, t->name);
    } else if (!llama_moe_cache_is_host_weight(t)) {
        throw std::runtime_error(std::string("MoE cache: the CPU part reads the tensor ") + t->name + ", which is not a host weight");
    }
    map[t] = res;
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
        int32_t n_slots = 0;
        moe_cache_lru lru;

        ggml_tensor * lru_dev = nullptr; // I32 on the device when the device caches misses, see ggml_moe_cpu_layer::lru
        std::vector<int32_t> lru_host;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;
    static constexpr int64_t max_single = 8; // tokens per graph that compute the misses within the graph

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // batches of up to max_single tokens compute the misses within the graph, see ggml-moe-cpu-misses.h
    struct single_graph {
        const ggml_moe_cpu_misses_api * api = nullptr;
        int32_t id = -1; // device

        ggml_tensor * maps  = nullptr; // I32 [n_expert, n_layer] on the device, the slot of each expert, -1 if not cached
        ggml_tensor * state = nullptr; // I32 [1 + n_layer] on the device, the request counter and the request of each layer
        ggml_tensor * carry = nullptr; // I32 [n_layer] on the device, see ggml_moe_cpu_layer::carry
        ggml_tensor * crf      = nullptr; // F32 [n_expert, n_layer] on the device, the CRF of the ClassLRU
        ggml_tensor * last_use = nullptr; // I32 [n_expert, n_layer] on the device
        ggml_tensor * clock    = nullptr; // I32 [n_layer] on the device
        ggml_tensor * decay    = nullptr; // F32 on the device, see moe_cache_lru::device_tables
        bool dev = false; // the device caches the misses that the CPU does not compute, it needs to read the host experts
        bool cpu = false; // a CPU thread computes a share of the misses
        std::vector<int32_t>  maps_host;
        std::vector<uint32_t> carry_host;
        std::vector<float>    crf_host;
        std::vector<int32_t>  last_use_host;
        std::vector<int32_t>  clock_host;
        bool maps_dirty = true; // the host changed the cache since the last upload
        std::vector<int32_t> max_two_pass; // [n_layer] see ggml_moe_cpu_misses_api::max_two_pass, 0 for one pass

        // request, inputs and result in mapped host memory
        void      * host      = nullptr;
        ggml_moe_cpu_request * req = nullptr;
        int32_t   * ids       = nullptr; // [n_expert_used, max_single]
        float     * input     = nullptr; // [n_in, max_single]
        float     * output    = nullptr; // [n_out, n_expert_used, max_single]
        int32_t   * trace     = nullptr; // [n_expert_used, max_single, n_layer]
        int32_t   * slots_log = nullptr; // [n_expert_used, max_single, n_layer] slots given by the device
        uint32_t  * carry_log = nullptr; // [n_layer]
        int64_t     n_in      = 0;
        int64_t     n_out     = 0;
        ggml_backend_buffer_ptr host_buf;
        std::vector<ggml_moe_cpu_layer> params; // [n_layer]

        // the CPU part of each layer, one per number of tokens
        struct cpu_graph {
            ggml_context_ptr ctx;
            ggml_backend_buffer_ptr buf;
            ggml_cgraph * graph = nullptr;
        };
        std::vector<cpu_graph> graphs; // [max_single, n_layer]

        std::vector<int32_t> layers;   // the layers of the last graph that compute the misses within the graph
        std::vector<int32_t> n_used;   // [n_layer] experts per token in the last graph
        std::vector<int32_t> n_tokens; // [n_layer] tokens in the last graph
        bool pending = false; // the misses of the last graph are not admitted yet

        // the slots given by the device in the last graph, the host replays them in its ClassLRU after the next graph is queued (see serve)
        // with defer_log, the device does not wait for the host, else right away
        struct slot_log {
            bool pending = false;
            std::vector<int32_t>  layers;
            std::vector<int32_t>  n_used;
            std::vector<int32_t>  n_tokens;
            std::vector<uint32_t> carry;
            std::vector<int32_t>  ids;   // the selected experts of each layer, one after the other
            std::vector<int32_t>  slots;
        } log;
        bool defer_log = false;
        bool failed  = false;
        bool main    = false; // the thread that queued the last graph serves its requests, else the CPU thread
        uint32_t last = 0;    // the last request served

        // the CPU thread
        ggml_backend_ptr cpu_backend;
        ggml_threadpool_t threadpool = nullptr;
        void (*threadpool_free)(ggml_threadpool_t) = nullptr;
        std::thread thread;
        std::atomic<bool> stop { false };
        std::mutex mutex;
        std::condition_variable cv;
        bool wake = false;
        std::atomic<bool> thread_on { false }; // the CPU thread serves the requests
        std::atomic<bool> helper_on { false }; // the CPU thread serves the requests until the thread that queued the graph takes over
        std::atomic<bool> in_handle { false };
        std::atomic<bool> done      { false }; // the request of the last layer was served
        int32_t il_last = -1;
    };
    single_graph sg;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, int32_t n_threads) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
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
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
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
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
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
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        // computing the misses within the graph needs the cached layers on one device
        int32_t single_id = -1;
        bool single_ok = !no_alloc;
        if (const char * env = getenv("LLAMA_MOE_CACHE_SINGLE"); env != nullptr && atoi(env) == 0) { // LLAMA_MOE_CACHE_SINGLE=0: the slot lookups
            single_ok = false;
        }
        for (const group & g : groups) {
            if (g.n_slots > 0) {
                single_ok = single_ok && (single_id < 0 || single_id == g.id);
                single_id = g.id;
            }
        }
        if (single_ok && single_id >= 0) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[single_id].backend));
            auto get_api = (ggml_moe_cpu_misses_get_api_t) ggml_backend_reg_get_proc_address(reg, "ggml_moe_cpu_misses_get_api");
            sg.api = get_api ? get_api() : nullptr;
        }
        if (sg.api != nullptr) {
            // the device caches the misses itself if it can read the host experts, otherwise the CPU computes all the misses
            sg.cpu = n_zero > 0;
            sg.dev = sg.api->host_readable != nullptr;
            for (const group & g : groups) {
                for (int32_t il : g.layers) {
                    for (const ggml_tensor * t : llama_moe_cache_layer_experts(model.layers[il])) {
                        sg.dev = sg.dev && (g.n_slots == 0 || sg.api->host_readable(t->data));
                    }
                }
            }
            if ((!sg.dev && !sg.cpu) || (sg.cpu && !sg.api->cpu_part)) {
                sg.api = nullptr;
            }
        }
        if (sg.api != nullptr) {
            sg.id = single_id;
            // the misses of a layer can be computed in a second pass for up to this many tokens
            sg.max_two_pass.assign(layers.size(), 0);
            for (const group & g : groups) {
                for (int32_t il : g.layers) {
                    int32_t m = sg.dev && sg.api->max_two_pass != nullptr ? (int32_t) max_single : 0;
                    for (const ggml_tensor * t : llama_moe_cache_layer_experts(model.layers[il])) {
                        m = std::min<int32_t>(m, m > 0 ? sg.api->max_two_pass(devices[single_id].backend, t->type) : 0);
                    }
                    sg.max_two_pass[il] = m;
                }
            }
            n_tensors[single_id] += 7;
            for (const group & g : groups) {
                n_tensors[single_id] += sg.dev && g.n_slots > 0;
            }
        }

        // the host replays the slots of the device after the next graph is queued
        sg.defer_log = sg.api != nullptr && sg.dev && !sg.cpu && trace == nullptr;

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

        if (sg.api != nullptr) {
            int64_t n_expert = 0;
            for (const group & g : groups) {
                n_expert = std::max(n_expert, g.ref[0]->ne[2]);
            }
            ggml_context * ctx = devices[sg.id].ctx.get();
            sg.maps  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert, layers.size());
            sg.state = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1 + layers.size());
            sg.carry    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, layers.size());
            sg.crf      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, layers.size());
            sg.last_use = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert, layers.size());
            sg.clock    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, layers.size());
            sg.decay    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GGML_MOE_CPU_N_DECAY + moe_cache_lru::n_class);
            ggml_set_name(sg.maps,     "moe_cache.maps");
            ggml_set_name(sg.state,    "moe_cache.state");
            ggml_set_name(sg.carry,    "moe_cache.carry");
            ggml_set_name(sg.crf,      "moe_cache.crf");
            ggml_set_name(sg.last_use, "moe_cache.last_use");
            ggml_set_name(sg.clock,    "moe_cache.clock");
            ggml_set_name(sg.decay,    "moe_cache.decay");
        }

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            if (sg.dev) {
                // lists, miss count and pairs, hit count, n_tokens and pairs, see ggml_moe_cpu_layer::lru
                g.lru_dev = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4*g.n_slots + 2*moe_cache_lru::n_class + 1 + 1 + 2*max_single*n_expert_used + 2 + 2*max_single*n_expert_used);
                ggml_format_name(g.lru_dev, "moe_cache.%zu.lru", ig);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                        g.lru.slot_map[il] = slots;
                    }
                }
            }

            if (sg.api != nullptr) {
                init_single(n_threads);
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
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
    }

    void init_single(int32_t n_threads) {
        ggml_backend_t backend = devices[sg.id].backend;
        const int64_t n_layer = layers.size();
        for (const layer & l : layers) {
            if (l.ig >= 0) {
                sg.n_in  = std::max(sg.n_in,  l.experts.front()->ne[0]);
                sg.n_out = std::max(sg.n_out, l.experts.back()->ne[1]);
            }
        }

        const size_t off_ids    = GGML_PAD(sizeof(ggml_moe_cpu_request), 64);
        const size_t off_input  = off_ids    + GGML_PAD(n_expert_used*max_single*sizeof(int32_t), 64);
        const size_t off_output = off_input  + GGML_PAD(sg.n_in*max_single*sizeof(float), 64);
        const size_t off_trace  = off_output + GGML_PAD(sg.n_out*n_expert_used*max_single*sizeof(float), 64);
        const size_t off_slots  = off_trace  + GGML_PAD(n_layer*n_expert_used*max_single*sizeof(int32_t), 64);
        const size_t off_carry  = off_slots  + GGML_PAD(n_layer*n_expert_used*max_single*sizeof(int32_t), 64);
        const size_t size       = off_carry  + GGML_PAD(n_layer*sizeof(uint32_t), 64);

        uint8_t * dev = (uint8_t *) sg.api->alloc(backend, size, &sg.host);
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_reg_t cpu_reg = cpu_dev ? ggml_backend_dev_backend_reg(cpu_dev) : nullptr;
        auto * threadpool_new = cpu_reg ? (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new") : nullptr;
        auto * set_threadpool = cpu_reg ? (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_set_threadpool") : nullptr;
        auto * set_n_threads  = cpu_reg ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads") : nullptr;
        sg.threadpool_free = cpu_reg ? (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free") : nullptr;
        if (dev == nullptr || (sg.cpu && (threadpool_new == nullptr || set_threadpool == nullptr || set_n_threads == nullptr || sg.threadpool_free == nullptr))) {
            LLAMA_LOG_WARN("%s: cannot compute the misses within the graph, small batches use the slot lookups\n", __func__);
            if (dev != nullptr) {
                sg.api->free(sg.host);
            }
            sg.api = nullptr;
            return;
        }

        uint8_t * host = (uint8_t *) sg.host;
        sg.req       = (ggml_moe_cpu_request *) host;
        sg.ids       = (int32_t  *) (host + off_ids);
        sg.input     = (float    *) (host + off_input);
        sg.output    = (float    *) (host + off_output);
        sg.trace     = (int32_t  *) (host + off_trace);
        sg.slots_log = (int32_t  *) (host + off_slots);
        sg.carry_log = (uint32_t *) (host + off_carry);
        sg.host_buf.reset(ggml_backend_cpu_buffer_from_ptr(host, size));

        if (sg.dev) {
            const std::vector<float> tables = moe_cache_lru::device_tables();
            ggml_backend_tensor_set(sg.decay, tables.data(), 0, ggml_nbytes(sg.decay));
        }

        // the device gives up when the CPU does not answer in time
        const uint64_t timeout_ms = 10000;

        // the addresses of the tensors for the kernels
        bool addr_ok = true;
        auto addr = [&](const ggml_tensor * t) -> void * {
            void * res = sg.api->addr != nullptr ? sg.api->addr(backend, t) : t->data;
            addr_ok = addr_ok && res != nullptr;
            return res;
        };

        sg.params.resize(n_layer);
        for (int32_t il = 0; il < n_layer; ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            auto & p = sg.params[il];
            p.req        = (ggml_moe_cpu_request *) dev;
            p.ids        = (int32_t *) (dev + off_ids);
            p.input      = (float   *) (dev + off_input);
            p.output     = (float   *) (dev + off_output);
            p.trace      = (int32_t *) (dev + off_trace) + il*n_expert_used*max_single;
            p.layer      = il;
            p.zero_slot  = groups[layers[il].ig].n_slots;
            p.timeout_ns = timeout_ms*1000000;
            if (sg.dev) {
                const group & g = groups[layers[il].ig];
                p.lru       = (int32_t  *) addr(g.lru_dev);
                p.maps      = (int32_t  *) addr(sg.maps);
                p.crf       = (float    *) addr(sg.crf);
                p.last_use  = (int32_t  *) addr(sg.last_use);
                p.clock     = (int32_t  *) addr(sg.clock);
                p.decay     = (const float *) addr(sg.decay);
                p.carry     = (uint32_t *) addr(sg.carry);
                p.share_q   = cpu_share_q;
                p.carry_log = (uint32_t *) (dev + off_carry);
                p.slots_log = (int32_t  *) (dev + off_slots) + il*n_expert_used*max_single;
                p.n_slots   = g.n_slots;
                p.n_zero    = n_zero;
                p.n_expert  = sg.maps->ne[0];
                p.n_ids_max = max_single*n_expert_used;
                p.n_banks   = layers[il].experts.size();
                GGML_ASSERT(p.n_banks <= GGML_MOE_CPU_MAX_BANKS);
                for (int32_t ip = 0; ip < p.n_banks; ++ip) {
                    p.bank[ip] = (char *) addr(g.banks[ip]);
                    p.host[ip] = (const char *) addr(layers[il].experts[ip]);
                    p.size[ip] = layers[il].experts[ip]->nb[2];
                }
            }
        }
        if (!addr_ok) {
            LLAMA_LOG_WARN("%s: the device cannot address the MoE cache, small batches use the slot lookups\n", __func__);
            sg.params.clear();
            sg.host_buf.reset();
            sg.api->free(sg.host);
            sg.api = nullptr;
            return;
        }
        sg.graphs.resize(n_layer*max_single);
        sg.n_used.assign(n_layer, 0);
        sg.n_tokens.assign(n_layer, 0);
        sg.maps_host.assign(ggml_nelements(sg.maps), -1);

        if (!sg.cpu) {
            int32_t two_pass_min = max_single;
            int32_t two_pass_max = 0;
            for (const layer & l : layers) {
                if (l.ig >= 0) {
                    const int32_t il = (int32_t) (&l - layers.data());
                    two_pass_min = std::min(two_pass_min, sg.max_two_pass[il]);
                    two_pass_max = std::max(two_pass_max, sg.max_two_pass[il]);
                }
            }
            LLAMA_LOG_INFO("%s: batches of up to %d tokens cache the misses within the graph, the hits first for up to %d-%d tokens\n", __func__,
                (int) max_single, (int) two_pass_min, (int) two_pass_max);
            return;
        }

        sg.cpu_backend.reset(ggml_backend_dev_init(cpu_dev, nullptr));
        ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        sg.threadpool = threadpool_new(&tpp);
        set_threadpool(sg.cpu_backend.get(), sg.threadpool);
        set_n_threads(sg.cpu_backend.get(), n_threads);

        sg.thread = std::thread([this] { service(); });

        LLAMA_LOG_INFO("%s: batches of up to %d tokens compute the misses within the graph, %s, on %d CPU threads\n", __func__, (int) max_single,
            sg.dev ? "a share on the CPU" : "all on the CPU", n_threads);
    }

    void stop_single() {
        if (sg.api == nullptr) {
            return;
        }
        // the device does not wait for the CPU anymore
        llama_moe_cache_atomic(sg.req->status).store(GGML_MOE_CPU_STOP, std::memory_order_release);
        sg.stop = true;
        {
            std::lock_guard<std::mutex> lock(sg.mutex);
            sg.wake = true;
        }
        sg.cv.notify_one();
        if (sg.thread.joinable()) {
            sg.thread.join();
        }
        sg.graphs.clear();
        sg.cpu_backend.reset();
        if (sg.threadpool != nullptr) {
            sg.threadpool_free(sg.threadpool);
        }
        sg.host_buf.reset();
        sg.api->free(sg.host);
        sg.api = nullptr;
    }

    // handles the last request, returns its layer, -1 if there is no new request, -2 if the CPU part failed
    // the device publishes a request for each layer but waits only for the requests with experts of the CPU, so it can publish several requests between two calls
    int32_t handle(ggml_backend_t cpu) {
        const uint32_t r = llama_moe_cache_atomic(sg.req->request).load(std::memory_order_acquire);
        if (r == sg.last) {
            return -1;
        }
        const int32_t il       = llama_moe_cache_atomic(sg.req->layer).load(std::memory_order_relaxed);
        const int32_t n_cpu    = llama_moe_cache_atomic(sg.req->n_cpu).load(std::memory_order_relaxed);
        const int32_t n_tokens = llama_moe_cache_atomic(sg.req->n_tokens).load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (llama_moe_cache_atomic(sg.req->request).load(std::memory_order_relaxed) != r) {
            return -1; // overwritten by a newer request
        }
        sg.last = r;
        if (n_cpu > 0) {
            const int64_t ic = il*max_single + n_tokens - 1;
            if (il < 0 || il >= (int32_t) layers.size() || n_tokens < 1 || n_tokens > max_single || sg.graphs[ic].graph == nullptr ||
                ggml_backend_graph_compute(cpu, sg.graphs[ic].graph) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: the CPU part of layer %d failed\n", __func__, il);
                llama_moe_cache_atomic(sg.req->status).store(GGML_MOE_CPU_STOP, std::memory_order_release);
                return -2;
            }
        }
        llama_moe_cache_atomic(sg.req->completed).store(r, std::memory_order_release);
        if (il == sg.il_last) {
            sg.done = true;
        }
        return il;
    }

    // the thread that queued the graph serves its requests until the last layer published
    // the CPU thread serves them while the graph is queued, the device backend can wait for the device there, e.g. to free memory
    void serve(ggml_backend_t cpu) {
        // the graph is queued: replay the slots of the last graph while the device computes this one
        flush_log();
        if (sg.api == nullptr || !sg.cpu || !sg.pending || !sg.main) {
            return;
        }
        sg.helper_on = false;
        while (sg.in_handle.load()) {
            llama_moe_cache_cpu_relax();
        }
        const int64_t timeout = sg.params[sg.il_last].timeout_ns/1000;
        int64_t t_last = ggml_time_us();
        while (!sg.done.load()) {
            const int32_t il = handle(cpu ? cpu : sg.cpu_backend.get());
            if (il == -2) {
                return;
            }
            if (il >= 0) {
                t_last = ggml_time_us();
                continue;
            }
            // the device failed or does not publish, the next begin reports it
            if (llama_moe_cache_atomic(sg.req->status).load(std::memory_order_acquire) != GGML_MOE_CPU_OK || ggml_time_us() - t_last > timeout) {
                return;
            }
            llama_moe_cache_cpu_relax();
        }
    }

    // the CPU thread serves the graphs that the thread that queued them cannot serve, and helps while a graph is queued
    // it spins while the requests come often
    void service() {
        int64_t t_last = ggml_time_us();
        while (!sg.stop.load()) {
            if (sg.thread_on.load() || sg.helper_on.load()) {
                sg.in_handle = true;
                const int32_t il = sg.thread_on.load() || sg.helper_on.load() ? handle(sg.cpu_backend.get()) : -1;
                sg.in_handle = false;
                if (il == -2) {
                    return;
                }
                if (il >= 0) {
                    t_last = ggml_time_us();
                    continue;
                }
                if (sg.helper_on.load() || ggml_time_us() - t_last < 20000) {
                    llama_moe_cache_cpu_relax();
                    continue;
                }
            }
            // idle: wait for the next graph
            std::unique_lock<std::mutex> lock(sg.mutex);
            sg.cv.wait_for(lock, std::chrono::milliseconds(1), [&] { return sg.wake; });
            if (sg.wake) {
                sg.wake = false;
                t_last  = ggml_time_us();
            }
        }
    }

    // the layer of a publish op of this cache, -1 for any other node
    // the op is a custom op of the backend (see ggml_moe_cpu_misses_api), its map argument is a view of the slot maps at the row of the layer
    int32_t publish_layer(const ggml_tensor * node) const {
        if (sg.api == nullptr || node->op != GGML_OP_CUSTOM || node->src[3] != sg.state || node->src[2] == nullptr || node->src[2]->view_src != sg.maps) {
            return -1;
        }
        return (int32_t) (node->src[2]->view_offs/sg.maps->nb[1]);
    }

    // the thread that queues the graph can serve its requests if no CPU op runs after the first publish op, the scheduler would wait for the device first
    bool main_ok(ggml_backend_sched_t sched, ggml_cgraph * graph, ggml_backend_t cpu) const {
        if (sg.api == nullptr || cpu == nullptr) {
            return false;
        }
        bool published = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (publish_layer(node) >= 0) {
                published = true;
            } else if (published && ggml_backend_sched_get_tensor_backend(sched, node) == cpu &&
                       node->op != GGML_OP_NONE && node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
                       node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE) {
                return false;
            }
        }
        return true;
    }

    // batches of up to 8 tokens compute the misses within the graph, without slot lookups (see ggml-moe-cpu-misses.h):
    // the device looks up the slots, the CPU computes a share of the misses, and the other misses are cached by the device or between the graphs
    // can_cpu_part: the graph can compute the misses of layer il on the CPU
    bool single(int32_t il, int64_t n_tokens, bool can_cpu_part) const {
        if (sg.api == nullptr || sg.failed || n_tokens < 1 || n_tokens > max_single || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return false;
        }
        if (sg.cpu && !can_cpu_part) {
            return false;
        }
        // same condition as the slot lookups
        const layer & l = layers[il];
        return groups[l.ig].id == sg.id && std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) <= groups[l.ig].n_slots;
    }

    // small ubatches (the backend sets the limit for the expert types of layer il) can compute the hits while the device copies the misses,
    // then the misses: the publish op then gives two slot planes [n_used, n_tokens, 2], the hits and the zero slots of the experts of the CPU
    // (-1 for the misses the device caches) and the misses the device caches (-1 for the others), build_join gives the second plane once
    // the misses are copied, and build_merge adds the result of the CPU after both passes
    bool two_pass(int32_t il, int64_t n_tokens) const {
        return sg.api != nullptr && il >= 0 && il < (int32_t) sg.max_two_pass.size() && n_tokens <= sg.max_two_pass[il];
    }

    ggml_tensor * build_join(ggml_context * ctx, ggml_tensor * slots, int32_t il) const {
        GGML_ASSERT(slots->type == GGML_TYPE_I32 && ggml_is_contiguous(slots));
        ggml_tensor * args[] = { slots };
        return ggml_custom_4d(ctx, GGML_TYPE_I32, slots->ne[0], slots->ne[1], slots->ne[2], slots->ne[3], args, 1,
            sg.api->join, 1, const_cast<ggml_moe_cpu_layer *>(&sg.params[il]));
    }

    // the slots of the experts selected in layer il, the misses of the CPU read zero slots
    ggml_tensor * build_publish(ggml_context * ctx, ggml_tensor * selected_experts, ggml_tensor * cur, int32_t il, bool two_pass) const {
        ggml_tensor * ids = ggml_is_contiguous(selected_experts) ? selected_experts : ggml_cont(ctx, selected_experts);
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx, cur);
        }
        GGML_ASSERT(ids->ne[0] <= n_expert_used && ggml_nrows(ids) <= max_single && ggml_nelements(cur) <= sg.n_in*ggml_nrows(ids));
        ids = ggml_reshape_2d(ctx, ids, ids->ne[0], ggml_nrows(ids));
        ggml_tensor * map = ggml_view_1d(ctx, sg.maps, layers[il].slot_map->ne[1], il*sg.maps->nb[1]);
        GGML_ASSERT(ids->type == GGML_TYPE_I32 && cur->type == GGML_TYPE_F32 && ggml_nrows(ids) == ggml_nrows(cur));
        ggml_tensor * args[] = { ids, cur, map, sg.state };
        return ggml_custom_4d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1], two_pass ? 2 : 1, 1, args, 4,
            sg.api->publish, 1, const_cast<ggml_moe_cpu_layer *>(&sg.params[il]));
    }

    // the result of the cached experts of layer il plus the result of the CPU part
    ggml_tensor * build_merge(ggml_context * ctx, ggml_tensor * experts, ggml_tensor * slots, int32_t il) const {
        GGML_ASSERT(experts->ne[0] <= sg.n_out);
        GGML_ASSERT(experts->type == GGML_TYPE_F32 && ggml_is_contiguous(experts) && slots->type == GGML_TYPE_I32 && ggml_is_contiguous(slots));
        GGML_ASSERT(ggml_nelements(slots) == ggml_nrows(experts));
        ggml_tensor * args[] = { experts, slots, sg.state };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, experts->ne[0], experts->ne[1], experts->ne[2], experts->ne[3], args, 3,
            sg.api->merge, 1, const_cast<ggml_moe_cpu_layer *>(&sg.params[il]));
    }

    // the CPU part of layer il for n_tokens tokens: input F32 [n_embd, 1, n_tokens], selected experts and the experts of the CPU I32 [n_used, n_tokens],
    // result F32 [n_embd, n_used, n_tokens]
    bool has_cpu_graph(int32_t il, int64_t n_tokens) const {
        return !sg.cpu || sg.graphs[il*max_single + n_tokens - 1].graph != nullptr;
    }

    void set_cpu_graph(int32_t il, ggml_tensor * input, ggml_tensor * selected, ggml_tensor * ids, ggml_tensor * result) {
        const int64_t n_tokens = ggml_nrows(selected);
        GGML_ASSERT(sg.cpu && n_tokens >= 1 && n_tokens <= max_single);
        auto & cg = sg.graphs[il*max_single + n_tokens - 1];
        if (cg.graph != nullptr) {
            return;
        }
        ggml_init_params params = {
            /*.mem_size   =*/ 256*ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        cg.ctx.reset(ggml_init(params));
        ggml_context * ctx = cg.ctx.get();

        // the inputs and the result are in mapped host memory
        std::unordered_map<const ggml_tensor *, ggml_tensor *> map;
        auto in_host = [&](ggml_tensor * t, void * data) {
            ggml_tensor * res = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
            GGML_ASSERT(ggml_backend_tensor_alloc(sg.host_buf.get(), res, data) == GGML_STATUS_SUCCESS);
            map[t] = res;
        };
        GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_nelements(input) <= sg.n_in*n_tokens);
        GGML_ASSERT(selected->ne[0] <= n_expert_used && ggml_are_same_shape(selected, ids));
        in_host(input,    sg.input);
        in_host(selected, sg.trace + il*n_expert_used*max_single);
        in_host(ids,      sg.ids);

        ggml_tensor * out = llama_moe_cache_copy_graph(ctx, result, map);
        GGML_ASSERT(out->type == GGML_TYPE_F32 && out->view_src == nullptr && ggml_is_contiguous(out));
        GGML_ASSERT(out->ne[0] <= sg.n_out && ggml_nrows(out) <= n_expert_used*n_tokens);
        GGML_ASSERT(ggml_backend_tensor_alloc(sg.host_buf.get(), out, sg.output) == GGML_STATUS_SUCCESS);

        cg.graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(cg.graph, out);
        cg.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type()));
    }

    // upload the misses in fills, consecutive experts going to consecutive slots are uploaded together
    size_t upload(ggml_backend_t backend, const layer & l, const group & g) const {
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
        return bytes;
    }

    // the share of the misses computed on the CPU, which applies once the cache of the group is full
    uint32_t share_q(const group & g) const {
        return g.lru.n_filled == g.lru.n_slots ? cpu_share_q : 0;
    }

    // the slots of all layers on the device
    void upload_maps() {
        const int64_t n_e = sg.maps->ne[0];
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            const group & g = groups[l.ig];
            const int32_t * slots = g.lru.slot_map[il];
            for (int64_t e = 0; e < l.slot_map->ne[1]; ++e) {
                sg.maps_host[il*n_e + e] = slots[e] < g.n_slots ? slots[e] : -1;
            }
        }
        ggml_backend_tensor_set_async(devices[sg.id].backend, sg.maps, sg.maps_host.data(), 0, ggml_nbytes(sg.maps));
    }

    // the experts selected by the last graph: hits are promoted, a share 1 - cpu_share of the misses is admitted for the next tokens, all of them until the cache is full
    void admit() {
        ggml_backend_t backend = devices[sg.id].backend;
        for (int32_t il : sg.layers) {
            layer & l = layers[il];
            group & g = groups[l.ig];
            const int32_t n_used = sg.n_used[il];
            const int32_t * sel = sg.trace + il*n_expert_used*max_single;
            ids.assign(sel, sel + n_used*sg.n_tokens[il]);

            if (trace != nullptr) {
                write_trace(il, n_used);
            }

            const uint32_t q = share_q(g);
            auto n_cpu = [&](size_t n_miss) {
                const uint64_t acc = (uint64_t) n_miss*q + l.cpu_acc;
                l.cpu_acc = acc & 0xffff;
                return (size_t) (acc >> 16);
            };
            size_t n_hit = 0;
            if (!g.lru.plan(il, ids.data(), ids.size(), sg.n_tokens[il], n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
                GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
            }

            stats_small.hits   += n_hit;
            stats_small.misses += fills.size() + cpu.size();
            stats_small.cpu    += fills.size() + cpu.size();
            stats_small.bytes  += upload(backend, l, g);

            sg.maps_dirty = sg.maps_dirty || !fills.empty();
        }
    }

    // copy the slots that the device gave to the selected experts of the last graph, the next graph overwrites them
    void save_log() {
        auto & lg = sg.log;
        GGML_ASSERT(!lg.pending);
        lg.layers.clear();
        lg.n_used.clear();
        lg.n_tokens.clear();
        lg.carry.clear();
        lg.ids.clear();
        lg.slots.clear();
        for (int32_t il : sg.layers) {
            const size_t    n_ids = (size_t) sg.n_used[il]*sg.n_tokens[il];
            const int32_t * sel   = sg.trace     + il*n_expert_used*max_single;
            const int32_t * slots = sg.slots_log + il*n_expert_used*max_single;
            lg.layers.push_back(il);
            lg.n_used.push_back(sg.n_used[il]);
            lg.n_tokens.push_back(sg.n_tokens[il]);
            lg.carry.push_back(sg.carry_log[il]);
            lg.ids.insert(lg.ids.end(), sel, sel + n_ids);
            lg.slots.insert(lg.slots.end(), slots, slots + n_ids);
        }
        lg.pending = true;
    }

    // replay the saved slots in the ClassLRU of the host
    void flush_log() {
        auto & lg = sg.log;
        if (!lg.pending) {
            return;
        }
        lg.pending = false;
        size_t off = 0;
        for (size_t i = 0; i < lg.layers.size(); ++i) {
            const int32_t il = lg.layers[i];
            layer & l = layers[il];
            group & g = groups[l.ig];
            const int32_t   n_used = lg.n_used[i];
            const size_t    n_ids  = (size_t) n_used*lg.n_tokens[i];
            const int32_t * sel    = lg.ids.data()   + off;
            const int32_t * slots  = lg.slots.data() + off;
            off += n_ids;

            if (trace != nullptr) {
                ids.assign(sel, sel + n_ids);
                write_trace(il, n_used);
            }

            size_t n_hit  = 0;
            size_t n_fill = 0;
            size_t n_cpu  = 0;
            g.lru.apply(il, sel, slots, n_ids, lg.n_tokens[i], n_hit, n_fill, n_cpu);

            size_t expert_size = 0;
            for (const ggml_tensor * w : l.experts) {
                expert_size += w->nb[2];
            }
            stats_small.hits   += n_hit;
            stats_small.misses += n_fill + n_cpu;
            stats_small.cpu    += n_cpu;
            stats_small.bytes  += n_fill*expert_size;

            l.cpu_acc = lg.carry[i];
        }
    }

    // the cache on the device, when the host changed it: the slot maps, the carries and the ClassLRU of each group
    // otherwise the device has the same state, the host replayed the slots that the device gave
    void upload_dev() {
        if (!sg.maps_dirty) {
            return;
        }
        ggml_backend_t backend = devices[sg.id].backend;
        const int64_t n_e = sg.maps->ne[0];
        upload_maps();

        const size_t n_layer = layers.size();
        sg.carry_host.assign(n_layer, 0);
        sg.crf_host.assign(n_layer*n_e, 0.0f);
        sg.last_use_host.assign(n_layer*n_e, -1);
        sg.clock_host.assign(n_layer, 0);
        for (size_t il = 0; il < n_layer; ++il) {
            const layer & l = layers[il];
            sg.carry_host[il] = l.cpu_acc;
            if (l.ig < 0) {
                continue;
            }
            const moe_cache_lru & lru = groups[l.ig].lru;
            GGML_ASSERT(lru.clock[il] < INT32_MAX);
            sg.clock_host[il] = (int32_t) lru.clock[il];
            for (int64_t e = 0; e < l.slot_map->ne[1]; ++e) {
                const int32_t key = lru.make_key(il, e);
                sg.crf_host     [il*n_e + e] = lru.crf[key];
                sg.last_use_host[il*n_e + e] = (int32_t) lru.last_use[key];
            }
        }
        ggml_backend_tensor_set_async(backend, sg.carry,    sg.carry_host.data(),    0, ggml_nbytes(sg.carry));
        ggml_backend_tensor_set_async(backend, sg.crf,      sg.crf_host.data(),      0, ggml_nbytes(sg.crf));
        ggml_backend_tensor_set_async(backend, sg.last_use, sg.last_use_host.data(), 0, ggml_nbytes(sg.last_use));
        ggml_backend_tensor_set_async(backend, sg.clock,    sg.clock_host.data(),    0, ggml_nbytes(sg.clock));

        for (group & g : groups) {
            if (g.lru_dev == nullptr) {
                continue;
            }
            const moe_cache_lru & lru = g.lru;
            const int32_t S = g.n_slots;
            std::vector<int32_t> & v = g.lru_host;
            v.assign(4*S + 2*moe_cache_lru::n_class + 1, 0);
            for (int32_t s = 0; s < S; ++s) {
                const int32_t key = lru.key_of[s];
                v[0*S + s] = key < 0 ? -1 : (int32_t) (lru.key_layer(key)*n_e + lru.key_expert(key));
                v[1*S + s] = lru.prev[s];
                v[2*S + s] = lru.next[s];
                v[3*S + s] = lru.list_of[s];
            }
            for (int c = 0; c < moe_cache_lru::n_class; ++c) {
                v[4*S + c]                         = lru.head[c];
                v[4*S + moe_cache_lru::n_class + c] = lru.tail[c];
            }
            v[4*S + 2*moe_cache_lru::n_class] = lru.n_filled;
            ggml_backend_tensor_set_async(backend, g.lru_dev, v.data(), 0, v.size()*sizeof(int32_t));
        }
        sg.maps_dirty = false;
    }

    // the slot lookups change the cache while the graph runs, after the device read it, so a graph cannot have both
    // called before the scheduler allocates the graph, it replaces the slot maps read by the lookups with their copies
    // publish: (layer, experts per token, tokens) of each publish op, for begin()
    void check_graph(ggml_cgraph * graph, std::vector<int32_t> & publish) const {
        publish.clear();
        bool lookups = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (const int32_t il = publish_layer(node); il >= 0) {
                publish.push_back(il);
                publish.push_back((int32_t) node->src[0]->ne[0]);
                publish.push_back((int32_t) node->src[0]->ne[1]);
            }
            lookups = lookups || (node->op == GGML_OP_GET_ROWS && layer_of.count(node->src[0]) > 0);
        }
        GGML_ASSERT(!(!publish.empty() && lookups) && "a graph cannot compute the MoE cache misses both within the graph and with slot lookups");
    }

    // before each graph compute: replay or admit the misses of the last graph, then upload the state that the next graph reads
    // publish: the publish ops of the graph as given by check_graph(), nullptr to search the graph
    bool begin(ggml_cgraph * graph, bool main, const std::vector<int32_t> * publish) {
        if (sg.api == nullptr) {
            return true;
        }
        flush_log(); // normally done by serve()

        if (sg.pending) {
            sg.pending = false;
            ggml_backend_synchronize(devices[sg.id].backend);

            if (sg.cpu) {
                // the device is done, the requests left have no experts of the CPU
                sg.thread_on = false;
                sg.helper_on = false;
                while (sg.in_handle.load()) {
                    llama_moe_cache_cpu_relax();
                }
                sg.last = llama_moe_cache_atomic(sg.req->request).load(std::memory_order_acquire);

                const uint32_t status = llama_moe_cache_atomic(sg.req->status).load(std::memory_order_acquire);
                if (status != GGML_MOE_CPU_OK) {
                    LLAMA_LOG_ERROR("%s: the CPU part of the MoE cache failed (%s), the results of the last graph are invalid\n", __func__,
                        status == GGML_MOE_CPU_TIMEOUT ? "timeout" : "CPU thread error");
                    sg.failed = true;
                    return false;
                }
            }
            if (sg.dev) {
                save_log();
            } else {
                admit();
            }
        }

        sg.layers.clear();
        if (publish != nullptr) {
            for (size_t i = 0; i + 2 < publish->size(); i += 3) {
                const int32_t il = (*publish)[i];
                sg.layers.push_back(il);
                sg.n_used[il]   = (*publish)[i + 1];
                sg.n_tokens[il] = (*publish)[i + 2];
            }
        } else {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const ggml_tensor * node = ggml_graph_node(graph, i);
                if (const int32_t il = publish_layer(node); il >= 0) {
                    sg.layers.push_back(il);
                    sg.n_used[il]   = node->src[0]->ne[0];
                    sg.n_tokens[il] = node->src[0]->ne[1];
                }
            }
        }
        // the host reads its ClassLRU while it computes a graph without publish ops, and before it uploads the cache
        if (sg.layers.empty() || !sg.defer_log || sg.maps_dirty) {
            flush_log();
        }
        if (sg.layers.empty()) {
            return true;
        }
        if (sg.failed) {
            return false;
        }

        if (sg.dev) {
            upload_dev();
        } else if (sg.maps_dirty) {
            upload_maps();
            sg.maps_dirty = false;
        }
        sg.pending = true;
        if (!sg.cpu) {
            return true;
        }
        sg.main    = main;
        sg.il_last = sg.layers.back();
        sg.done    = false;
        if (sg.main) {
            sg.helper_on = true;
        } else {
            sg.thread_on = true;
        }
        {
            std::lock_guard<std::mutex> lock(sg.mutex);
            sg.wake = true;
        }
        sg.cv.notify_one();
        return true;
    }

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        // the slots of the last graph, the device is idle and the replay only reads host memory
        flush_log();
        if (sg.api != nullptr && sg.pending && sg.dev) {
            save_log();
            flush_log();
        }
        stop_single();
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
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
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
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
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
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
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_lookup

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t q = has_cpu ? share_q(g) : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += upload(backend, l, g);

        // the device reads the changed cache before the next graph that computes the misses within the graph
        sg.maps_dirty = true;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, int32_t n_threads) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share, n_threads)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

llama_moe_cache_lookup llama_moe_cache::build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * cur, ggml_tensor * selected_experts) const {
    const int32_t il = layer.il;
    ggml_context * ctx0 = g.ctx0;
    const std::array<ggml_tensor *, 4> exps = { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps };

    // part of the misses may be computed from the host experts, but not with weights before the FFN, LoRAs or rotated experts
    bool cpu_part = !layer.weight_before_ffn && g.loras->empty();
    for (ggml_tensor * w : exps) {
        cpu_part = cpu_part && !(w != nullptr && g.hdmd != nullptr && g.hdmd->rot.count(w) > 0);
    }

    // the misses of small ubatches are computed within the graph
    // the ubatch decides for all the layers, a graph cannot also have slot lookups, and the last layer can have fewer tokens
    if (pimpl->single(il, g.n_tokens, cpu_part) &&
        std::all_of(exps.begin(), exps.end(), [&](ggml_tensor * w) { return w == nullptr || pimpl->get_experts(w) != nullptr; })) {
        if (selected_experts->ne[1] == 0) {
            // no rows (the last layer computes only the outputs, if any): the cached experts are not read
            return { il, selected_experts };
        }

        // the skipped experts of a pass give zero, which biases would change
        // TODO: [TAG_MUL_MAT_ID_SKIP] remove the one pass path once MUL_MAT_ID (all kernels and types) and ADD_ID skip negative ids
        //       on the backends that run the MoE cache ops, then every ubatch of the single graph can compute the hits first
        const bool two_pass = pimpl->two_pass(il, g.n_tokens) && !layer.biases &&
            std::all_of(exps.begin(), exps.end(), [&](ggml_tensor * w) { return w == nullptr || ggml_is_quantized(w->type); });
        ggml_tensor * published = pimpl->build_publish(ctx0, selected_experts, cur, il, two_pass);
        g.cb(published, "ffn_moe_publish", il);
        ggml_build_forward_expand(g.gf, published);

        llama_moe_cache_lookup res;
        res.il        = il;
        res.cpu_graph = pimpl->sg.cpu;
        if (two_pass) {
            res.slots      = ggml_view_2d(ctx0, published, published->ne[0], published->ne[1], published->nb[1], 0);
            res.slots_miss = pimpl->build_join(ctx0, ggml_view_2d(ctx0, published, published->ne[0], published->ne[1], published->nb[1], published->nb[2]), il);
            g.cb(res.slots_miss, "ffn_moe_join", il);
        } else {
            res.slots = published;
        }
        return res;
    }

    ggml_tensor * cpu_map  = cpu_part ? pimpl->get_cpu_map(il, selected_experts->ne[1]) : nullptr;
    ggml_tensor * slot_map = pimpl->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0], cpu_map != nullptr);
    if (slot_map == nullptr) {
        return {};
    }
    for (ggml_tensor * w : exps) {
        if (w == nullptr) {
            continue;
        }
        ggml_tensor * experts = pimpl->get_experts(w);
        if (experts == nullptr) {
            return {};
        }
        // the device of the cache must compute the matmul, else the scheduler copies the cache to another backend
        ggml_tensor * b = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, experts->ne[0], w == layer.down_exps ? selected_experts->ne[0] : 1, selected_experts->ne[1]);
        if (!ggml_backend_supports_op(pimpl->backend(il), ggml_mul_mat_id(ctx0, experts, b, selected_experts))) {
            return {};
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, uploads the missing ones and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(pimpl->backend(il), slots)) {
        return {};
    }
    ggml_backend_sched_set_tensor_backend(g.sched, slots, pimpl->backend(il));
    g.cb(slots, "ffn_moe_slots", il);

    // add the lookup now: the transform of a Hadamard-folded expert reads host weights and would start its split first
    ggml_build_forward_expand(g.gf, slots);

    // the misses computed from the host experts, read after the lookup has picked them
    ggml_tensor * cpu_ids = nullptr;
    if (cpu_map != nullptr) {
        cpu_ids = ggml_get_rows(ctx0, cpu_map, ids); // [1, n_expert_used*n_tokens]
        cpu_ids = ggml_reshape_2d(ctx0, cpu_ids, selected_experts->ne[0], selected_experts->ne[1]);
    }

    return { il, ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]), cpu_ids };
}

ggml_tensor * llama_moe_cache::build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
        ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const {
    GGML_ASSERT(lookup.slots != nullptr);

    ggml_tensor * experts = build(cur, selected_experts, lookup.slots, true, user_data);
    if (lookup.slots_miss != nullptr) {
        // the hits first, so that they are computed while the misses are copied
        ggml_build_forward_expand(g.gf, experts);
        experts = ggml_add(g.ctx0, experts, build(cur, selected_experts, lookup.slots_miss, true, user_data));
        g.cb(experts, "ffn_moe_down_two_pass", lookup.il);
    }
    if (lookup.cpu_graph) {
        // the experts of the CPU read zero slots in the hits pass and are skipped in the misses pass
        // the CPU part is not in the graph, the MoE cache computes it while the graph runs
        const int64_t n_tokens = selected_experts->ne[1];
        if (!pimpl->has_cpu_graph(lookup.il, n_tokens)) {
            ggml_tensor * input = ggml_new_tensor_3d(g.ctx0, GGML_TYPE_F32, cur->ne[0], 1, n_tokens);
            ggml_tensor * sel   = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_I32, selected_experts->ne[0], n_tokens);
            ggml_tensor * ids   = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_I32, selected_experts->ne[0], n_tokens);
            g.res->moe_cache.cpu_parts.push_back({ lookup.il, input, sel, ids, build(input, sel, ids, false, user_data) });
        }
        experts = pimpl->build_merge(g.ctx0, experts, lookup.slots, lookup.il);
        g.cb(experts, "ffn_moe_down_merged", lookup.il);
    } else if (lookup.cpu_ids != nullptr) {
        // the cached experts and the misses computed from the host experts, each reads the others as zero
        // the cached experts come first in the graph, so that they are computed while the misses are
        ggml_build_forward_expand(g.gf, experts);
        experts = ggml_add(g.ctx0, experts, build(cur, selected_experts, lookup.cpu_ids, false, user_data));
        g.cb(experts, "ffn_moe_down_merged", lookup.il);
    }

    return experts;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

void llama_moe_cache::set_cpu_graph(int32_t il, ggml_tensor * input, ggml_tensor * selected, ggml_tensor * ids, ggml_tensor * result) {
    pimpl->set_cpu_graph(il, input, selected, ids, result);
}

void llama_moe_cache::check_graph(ggml_cgraph * graph, std::vector<int32_t> & publish) const {
    pimpl->check_graph(graph, publish);
}

bool llama_moe_cache::main_ok(ggml_backend_sched_t sched, ggml_cgraph * graph, ggml_backend_t cpu) const {
    return pimpl->main_ok(sched, graph, cpu);
}

bool llama_moe_cache::begin(ggml_cgraph * graph, bool main, const std::vector<int32_t> * publish) {
    return pimpl->begin(graph, main, publish);
}

void llama_moe_cache::serve(ggml_backend_t cpu) {
    pimpl->serve(cpu);
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
