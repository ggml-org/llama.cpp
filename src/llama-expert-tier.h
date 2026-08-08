#pragma once

#include "ggml.h"

// Expert tier hook: drop-in replacement for ggml_mul_mat_id on expert weight
// tensors that have a registered GPU hot store. Pure stock ggml ops, no
// custom kernel, no per-model code.
//
// How it works (matches oldtricks.md Trick 4 sentinel + LUT design):
//   registered src w (full expert tensor on CPU/mmap) -> {dst_hot (S+1 slots
//   on GPU), per-layer LUTs and masks}. When build_lora_mm_id is called with
//   w, the hook remaps real expert ids through hot_lut/cold_lut, runs two
//   mul_mat_id ops (hot on GPU slots, cold on the full CPU tensor), scales
//   and masks each per routed expert, sums them. Result is byte-identical
//   in shape to a stock mul_mat_id output and feeds straight back into the
//   caller's downstream ops (bias add, activation, second build_lora_mm_id
//   call for down, routing-weight mul).
//
// GPU sentinel slot (index S of dst_hot) is zero-filled once at allocate
// time and never written again, so cold experts routed to it return zero.
// On the CPU side, we cannot grow a mmap'd expert tensor, so hot experts
// are routed to a real dummy (expert 0) on the cold path; the dummy's
// contribution is zeroed by cold_mask before the result is used.
//
// Masking happens at every mul_mat_id output, BEFORE bias add and BEFORE
// activation. That lifts the no-biases constraint: at a hot slot the cold
// mask zeroes the dummy contribution pre-activation, so the bias and
// activation run on the correct real value for that slot.

// register one expert weight tensor -> its GPU hot tensor and per-layer LUTs.
// Called by llama_expert_hotstore::allocate() after creating dst_hot and
// the LUT/mask tensors. Multiple entries per layer share the same luts[i].
void llama_expert_tier_register(ggml_tensor * src,
                                ggml_tensor * dst_hot,
                                ggml_tensor * hot_lut,
                                ggml_tensor * cold_mask);

// drop the entire table (called by hotstore destructor)
void llama_expert_tier_clear();

// cheap check: is `w` registered? (used so callers can short-circuit lora)
bool llama_expert_tier_has(ggml_tensor * w);

// drop-in hook called from build_lora_mm_id. Returns the combined
// hot+cold output tensor when `w` is registered; returns nullptr to let the
// caller fall back to stock ggml_mul_mat_id.
//   ctx   : graph context (ctx0 of the calling llm_graph_context)
//   w     : expert weight tensor, ne = [in, out, n_experts], 3d, ne[3]==1
//   cur   : activation, ne = [in, 1, n_tokens], 3d
//   ids   : selected_experts, ne = [n_expert_used, n_tokens], 2d i32, REAL ids
//   w_s   : per-expert quant scale, ne = [n_experts], f32; nullptr if none
ggml_tensor * llama_expert_tier_build(ggml_context * ctx,
                                      ggml_tensor * w,
                                      ggml_tensor * cur,
                                      ggml_tensor * ids,
                                      ggml_tensor * w_s);