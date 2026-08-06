#include "llama-expert-tier.h"

#include <mutex>
#include <unordered_map>

namespace {
    struct tier_entry {
        std::vector<ggml_tensor *> dst_hot; // per-device hot tensors
        std::vector<ggml_tensor *> hot_lut; // per-device LUTs
        std::vector<ggml_tensor *> mask_lut; // per-device sentinel masks
        ggml_tensor * cold_mask;
    };

    std::mutex g_mtx;
    std::unordered_map<ggml_tensor *, tier_entry> g_table;
}

void llama_expert_tier_register(ggml_tensor * src,
                                const std::vector<ggml_tensor *> & dst_hot,
                                const std::vector<ggml_tensor *> & hot_lut,
                                const std::vector<ggml_tensor *> & mask_lut,
                                ggml_tensor * cold_mask) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table[src] = {dst_hot, hot_lut, mask_lut, cold_mask};
}

void llama_expert_tier_clear() {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table.clear();
}

bool llama_expert_tier_has(ggml_tensor * w) {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_table.find(w) != g_table.end();
}

// Remap real expert ids through a LUT to slot indices, returning a 2d
// [n_expert_used, n_tokens] i32 tensor usable as ids for ggml_mul_mat_id.
// The lut is a [1, n_experts] table, so ggml_get_rows picks one scalar per
// id. ggml_cont guards against argsort views that may not be contiguous.
static ggml_tensor * remap_ids(ggml_context * ctx,
                              ggml_tensor * lut,
                              ggml_tensor * selected,
                              int n_experts,
                              int n_expert_used,
                              int n_tokens) {
    (void)n_experts;
    ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
        ggml_cont(ctx, selected), n_expert_used * n_tokens);
    ggml_tensor * r = ggml_get_rows(ctx, lut, flat_ids);
    return ggml_reshape_2d(ctx, r, n_expert_used, n_tokens);
}

ggml_tensor * llama_expert_tier_build(ggml_context * ctx,
                                      ggml_tensor * w,
                                      ggml_tensor * cur,
                                      ggml_tensor * ids,
                                      ggml_tensor * w_s) {
    // the CUDA mul_mat_id path assumes distinct expert ids per token; the
    // sentinel slot produces duplicates that go out of bounds there, so the
    // tier only engages for single-token decode.
    if (cur->ne[2] > 1) return nullptr;

    tier_entry ent;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_table.find(w);
        if (it == g_table.end()) {
            return nullptr;
        }
        ent = it->second;
    }

    const int n_experts     = (int) w->ne[2];
    const int n_expert_used = (int) ids->ne[0];
    const int n_tokens      = (int) cur->ne[2];

    // hot: for each device, remap real expert ids to that device's LOCAL slot
    // indices; experts whose slot lives on another device (or are cold) map to
    // this device's zeroed sentinel slot and contribute nothing. Sum the
    // per-device results (the scheduler inserts any cross-device copies).
    ggml_tensor * hot = nullptr;
    for (size_t g = 0; g < ent.dst_hot.size(); g++) {
        ggml_tensor * ids_hot = remap_ids(ctx, ent.hot_lut[g], ids, n_experts, n_expert_used, n_tokens);
        ggml_tensor * h = ggml_mul_mat_id(ctx, ent.dst_hot[g], cur, ids_hot);
        // zero the rows routed to the sentinel plane: index the mask LUT (0
        // at the sentinel) by ids_hot, repeat over the hidden dim, multiply
        ggml_tensor * m = ggml_get_rows(ctx, ent.mask_lut[g], ids_hot);
        h = ggml_mul(ctx, h, ggml_repeat(ctx, m, h));
        hot = hot ? ggml_add(ctx, hot, h) : h;
    }

    // cold: dedicated CPU op that computes only the cold-selected experts,
    // skipping hot ones via the integer zero-check on cold_mask.
    ggml_tensor * cold = ggml_mul_mat_id_cold(ctx, w, cur, ids, ent.cold_mask, nullptr, nullptr);

    (void)w_s; // per-expert quant scale is intentionally discarded on the tiered path

    return ggml_add(ctx, hot, cold);
}