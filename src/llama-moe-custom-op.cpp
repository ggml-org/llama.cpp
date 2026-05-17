#include "llama-moe-custom-op.h"

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace {

std::mutex                                                       g_ud_mtx;
std::unordered_map<int, std::unique_ptr<moe_residency_layer_ud>> g_ud_map;

}  // namespace

moe_residency_layer_ud * moe_residency_get_layer_ud(int layer) {
    std::lock_guard<std::mutex> g(g_ud_mtx);
    auto                        it = g_ud_map.find(layer);
    if (it != g_ud_map.end()) {
        return it->second.get();
    }
    auto ud    = std::make_unique<moe_residency_layer_ud>();
    ud->layer  = layer;
    auto * raw = ud.get();
    g_ud_map.emplace(layer, std::move(ud));
    return raw;
}

void moe_residency_custom_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }

    auto * ud = (moe_residency_layer_ud *) userdata;
    GGML_ASSERT(ud);
    GGML_ASSERT(ud->n_slots_used > 0);

    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src);
    GGML_ASSERT(src->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));

    const int32_t * ids = (const int32_t *) src->data;
    int32_t *       out = (int32_t *) dst->data;
    GGML_ASSERT(ids && out);

    const int64_t n = ggml_nelements(src);

    // Resolve on the first slot — this fills `out` with slot_ids.
    moe_residency().resolve_ids(ud->slots[0], ids, out, n);

    // Resolve on the remaining slots without writing output (the LRU state
    // of all 4 tensors of this layer must evolve identically). We discard
    // their slot_ids because, by construction, they match the first.
    if (ud->n_slots_used > 1) {
        thread_local std::vector<int32_t> scratch;
        if ((int64_t) scratch.size() < n) {
            scratch.resize((size_t) n);
        }
        for (int i = 1; i < ud->n_slots_used; ++i) {
            moe_residency().resolve_ids(ud->slots[i], ids, scratch.data(), n);
            // Invariant: scratch == out (same eviction decisions). Optionally
            // verify in debug builds.
#ifdef LLAMA_MOE_DEBUG_LRU_LOCKSTEP
            for (int64_t k = 0; k < n; ++k) {
                GGML_ASSERT(scratch[k] == out[k] && "LRU lockstep broken");
            }
#endif
        }
    }
}
