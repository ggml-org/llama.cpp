#include "llama-expert-tier.h"

#include <mutex>
#include <unordered_map>

namespace {
    struct tier_entry {
        ggml_tensor * dst_hot;    // [ne0, ne1, hot_s + 1]
        ggml_tensor * hot_lut;    // i32 [n_experts]
        ggml_tensor * cold_mask;  // f32 [n_experts] (read as int zero-check by cold op)
    };

    std::mutex g_mtx;
    std::unordered_map<ggml_tensor *, tier_entry> g_table;
}

void llama_expert_tier_register(ggml_tensor * src,
                                ggml_tensor * dst_hot,
                                ggml_tensor * hot_lut,
                                ggml_tensor * cold_mask) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table[src] = {dst_hot, hot_lut, cold_mask};
}

void llama_expert_tier_clear() {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table.clear();
}

bool llama_expert_tier_has(ggml_tensor * w) {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_table.find(w) != g_table.end();
}

// Remap real expert ids through a LUT and produce a 2d [n_expert_used,
// n_tokens] i32 tensor usable as `ids` for ggml_mul_mat_id.
//
// Flatten the per-token ids to 1D, run a stock 1D ggml_get_rows against the
// per-expert LUT (reshape to [1, n_experts] so each row picked is 1 scalar),
// then reshape the [1, n_eu*n_tok] result back to 2D. Avoids ggml_repeat_4d
// + view_2d strides that the CUDA mul_mat_id kernel mishandles on multi-token
// ubatches. `ggml_cont` defends against argsort views that may not be
// contiguous across the n_expert_used * n_tokens layout.
static ggml_tensor * remap_ids(ggml_context * ctx,
                              ggml_tensor * lut,
                              ggml_tensor * selected,
                              int n_experts,
                              int n_expert_used,
                              int n_tokens) {
    ggml_tensor * lut_rows = ggml_reshape_2d(ctx, lut, 1, n_experts);    // [1, n_experts]
    // selected (argsort_top_k view) may be non-contiguous; cont first, then reshape
    ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
        ggml_cont(ctx, selected), n_expert_used * n_tokens);             // [n_eu*n_tok] i32
    ggml_tensor * r = ggml_get_rows(ctx, lut_rows, flat_ids);            // [1, n_eu*n_tok, 1, 1] i32
    return ggml_reshape_2d(ctx, r, n_expert_used, n_tokens);              // [n_eu, n_tok]
}

// Build a per-(expert_used, token) mask f32 [1, n_expert_used, n_tokens, 1]
// (broadcastable against a mul_mat_id output of shape [out, n_eu, n_tok, 1]).
// Same flatten pattern as remap_ids.
static ggml_tensor * remap_mask(ggml_context * ctx,
                              ggml_tensor * mask,
                              ggml_tensor * selected,
                              int n_experts,
                              int n_expert_used,
                              int n_tokens) {
    ggml_tensor * mask_rows = ggml_reshape_2d(ctx, mask, 1, n_experts);  // [1, n_experts] f32
    ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
        ggml_cont(ctx, selected), n_expert_used * n_tokens);             // [n_eu*n_tok] i32
    ggml_tensor * r = ggml_get_rows(ctx, mask_rows, flat_ids);           // [1, n_eu*n_tok, 1, 1] f32
    return ggml_reshape_3d(ctx, r, 1, n_expert_used, n_tokens);          // [1, n_eu, n_tok]
}

ggml_tensor * llama_expert_tier_build(ggml_context * ctx,
                                      ggml_tensor * w,
                                      ggml_tensor * cur,
                                      ggml_tensor * ids,
                                      ggml_tensor * w_s) {
    // BUG-3E3: CUDA MMQ mul_mat_id compaction assumes distinct expert ids per token;
    // our sentinel duplicates OOB there. Decode (n_tokens==1) is safe via mmvq.
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

    // hot path: GPU tier tensor. Remap real expert ids through hot_lut
    // -> hot slot indices (sentinel S for cold experts = zero contribution).
    ggml_tensor * ids_hot = remap_ids(ctx, ent.hot_lut, ids, n_experts, n_expert_used, n_tokens);
    ggml_tensor * hot = ggml_mul_mat_id(ctx, ent.dst_hot, cur, ids_hot);

    // cold path: dedicated CPU op that computes ONLY cold-selected experts.
    // ent.cold_mask is f32 [n_experts] with 1.0f = cold; the op treats it
    // as integer-zero-check (0.0f = hot = skip, non-zero = cold = compute).
    ggml_tensor * cold = ggml_mul_mat_id_cold(ctx, w, cur, ids, ent.cold_mask);

    // per-expert quant scale on both paths
    if (w_s) {
        ggml_tensor * s_rows = ggml_reshape_2d(ctx, w_s, 1, n_experts);
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
            ggml_cont(ctx, ids), n_expert_used * n_tokens);
        ggml_tensor * s = ggml_get_rows(ctx, s_rows, flat_ids);
        s = ggml_reshape_3d(ctx, s, 1, n_expert_used, n_tokens);
        hot  = ggml_mul(ctx, hot,  s);
        cold = ggml_mul(ctx, cold, s);
    }

    return ggml_add(ctx, hot, cold);
}