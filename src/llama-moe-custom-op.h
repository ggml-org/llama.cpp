#pragma once

#include "ggml.h"
#include "llama-moe-residency.h"

#include <array>

// Per-layer userdata for the moe_residency custom op.
//
// One instance per MoE layer, lives in a static map inside llama-moe-custom-op.cpp.
// Built once on the first build_moe_ffn for the layer and reused across forwards.
//
// `slots` holds the residency slot pointers for the (up to 4) expert weight
// tensors of this layer (gate_up, up, gate, down). Order is irrelevant as
// long as it stays stable: the custom op walks all of them with the same
// `ids`, evolving their per-slot LRU state in lockstep.
struct moe_residency_layer_ud {
    std::array<moe_residency_slot *, 4> slots        = { nullptr, nullptr, nullptr, nullptr };
    int                                 n_slots_used = 0;
    int                                 layer        = -1;
};

// Get or create the userdata for a given layer. Thread-safe.
moe_residency_layer_ud * moe_residency_get_layer_ud(int layer);

// Custom op callback. dst = i32[n_expert_used, n_tokens] slot_ids tensor;
// dst->src[0] = i32[n_expert_used, n_tokens] selected_experts (CPU-resident).
// Runs single-threaded (n_tasks=1 at build time) — we ignore ith/nth.
void moe_residency_custom_op(ggml_tensor * dst, int ith, int nth, void * userdata);
