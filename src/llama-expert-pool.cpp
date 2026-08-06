#include "llama-expert-pool.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdlib>
#include <cstring>
#include <regex>

namespace llama_expert_pool {

namespace {
    bool g_requested = false;

    // C callback for the ggml cold op: resolve a per-expert slice by tensor
    static const uint8_t * pool_slice_cb(const struct ggml_tensor * src0, int expert) {
        return slice_for_tensor(src0, expert);
    }

    // matches an expert weight tensor, e.g. blk.0.ffn_gate_exps.weight
    const std::regex g_re_exps("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

    struct store {
        std::vector<entry> entries;
        // per entry: per-expert slice pointers (nullptr = freed / on GPU)
        std::vector<std::vector<uint8_t *>> slices;
        // number of experts per entry (from registration)
        std::vector<int> n_experts;
        bool is_active = false;
    };

    store g_store;
}

void mark_requested() {
    g_requested = true;
}

bool requested() {
    return g_requested;
}

bool match_exps(const char * name, int & layer_idx) {
    if (!name) {
        return false;
    }
    std::cmatch m;
    if (!std::regex_search(name, m, g_re_exps)) {
        return false;
    }
    layer_idx = std::stoi(m[1].str());
    return true;
}

bool register_tensor(const ggml_tensor * tensor, int layer_idx, int n_experts, size_t slice_bytes) {
    if (!g_requested || !tensor || n_experts <= 0 || slice_bytes == 0) {
        return false;
    }
    for (size_t i = 0; i < g_store.entries.size(); i++) {
        if (g_store.entries[i].tensor == tensor) {
            return true; // already registered
        }
    }
    g_store.entries.push_back({tensor, layer_idx, slice_bytes});
    g_store.n_experts.push_back(n_experts);
    g_store.slices.emplace_back(n_experts, nullptr);
    return true;
}

bool fill_tensor(const ggml_tensor * tensor, const uint8_t * data, size_t nbytes) {
    if (!g_requested || !tensor || !data) {
        return false;
    }
    for (size_t i = 0; i < g_store.entries.size(); i++) {
        if (g_store.entries[i].tensor != tensor) {
            continue;
        }
        const size_t slice_bytes = g_store.entries[i].slice_bytes;
        const int n_experts = g_store.n_experts[i];
        if (nbytes != (size_t) n_experts * slice_bytes) {
            return false;
        }
        g_store.is_active = true;
        ggml_mmid_cold_set_slice_fn(pool_slice_cb);
        for (int e = 0; e < n_experts; e++) {
            uint8_t * slice = (uint8_t *) std::malloc(slice_bytes);
            if (!slice) {
                return false;
            }
            std::memcpy(slice, data + (size_t) e * slice_bytes, slice_bytes);
            g_store.slices[i][e] = slice;
        }
        return true;
    }
    return false;
}

bool active() {
    return g_store.is_active;
}

size_t num_entries() {
    return g_store.entries.size();
}

const entry & get_entry(size_t idx) {
    return g_store.entries[idx];
}

uint8_t * get_slice(size_t entry_idx, int expert) {
    if (entry_idx >= g_store.slices.size()) {
        return nullptr;
    }
    auto & slices = g_store.slices[entry_idx];
    if (expert < 0 || expert >= (int) slices.size()) {
        return nullptr;
    }
    return slices[expert];
}

const uint8_t * slice_for_tensor(const ggml_tensor * tensor, int expert) {
    if (!g_store.is_active || !tensor || expert < 0) {
        return nullptr;
    }
    for (size_t i = 0; i < g_store.entries.size(); i++) {
        if (g_store.entries[i].tensor == tensor) {
            auto & slices = g_store.slices[i];
            if (expert >= (int) slices.size()) {
                return nullptr;
            }
            return slices[expert];
        }
    }
    return nullptr;
}

void free_slice(size_t entry_idx, int expert) {
    if (entry_idx >= g_store.slices.size()) {
        return;
    }
    auto & slices = g_store.slices[entry_idx];
    if (expert < 0 || expert >= (int) slices.size()) {
        return;
    }
    std::free(slices[expert]);
    slices[expert] = nullptr;
}

void set_slice(size_t entry_idx, int expert, const uint8_t * data) {
    if (entry_idx >= g_store.slices.size()) {
        return;
    }
    auto & slices = g_store.slices[entry_idx];
    if (expert < 0 || expert >= (int) slices.size()) {
        return;
    }
    const size_t slice_bytes = g_store.entries[entry_idx].slice_bytes;
    uint8_t * slice = (uint8_t *) std::malloc(slice_bytes);
    if (!slice) {
        return;
    }
    std::memcpy(slice, data, slice_bytes);
    std::free(slices[expert]); // replace any existing
    slices[expert] = slice;
}

} // namespace llama_expert_pool
