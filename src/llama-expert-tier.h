#pragma once

#include <vector>

#include "ggml.h"

// Expert tier hook: drop-in replacement for ggml_mul_mat_id on expert weight
// tensors that have a registered GPU hot store. Pure stock ggml ops, no
// custom kernels.
//
// A registered expert tensor w is split between a GPU hot store (the top-S
// experts, held in dst_hot with hot_s+1 slot planes) and the CPU cold store
// (the remaining experts, still inside w). build_lora_mm_id calls back into
// llama_expert_tier_build, which computes:
//   - hot:  expert ids remapped through hot_lut to slot indices, then a
//           mul_mat_id on dst_hot. Cold experts land on the zeroed sentinel
//           plane (index hot_s) and therefore contribute zero on the GPU.
//   - cold: mul_mat_id_cold on w, which skips hot experts entirely
//           (cold_mask[e] == 0) and computes only the cold-selected rows.
//   - result = hot + cold.
// The result has the same shape as a stock mul_mat_id output and feeds
// straight back into the caller's downstream ops.
//
// The per-expert quant scale w_s is discarded on the tiered path. It is an
// intentional approximation: applying it would add get_rows/mul nodes per
// layer, and the scale factors are close to 1.

// register one expert weight tensor -> its per-device GPU hot tensors and
// per-device LUTs. called by llama_expert_hotstore::allocate() after creating
// dst_hot/hot_lut and the cold_mask tensors. multiple entries per layer share
// the same luts[i].
void llama_expert_tier_register(ggml_tensor * src,
                                const std::vector<ggml_tensor *> & dst_hot,
                                const std::vector<ggml_tensor *> & hot_lut,
                                const std::vector<ggml_tensor *> & mask_lut,
                                ggml_tensor * cold_mask,
                                ggml_tensor * counts);

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
//   w_s   : per-expert quant scale, ne = [n_experts], f32; ignored by the
//           tiered path (see above)
ggml_tensor * llama_expert_tier_build(ggml_context * ctx,
                                      ggml_tensor * w,
                                      ggml_tensor * cur,
                                      ggml_tensor * ids,
                                      ggml_tensor * w_s);

// fused cold path (GGML_OP_MOE_COLD), one call pair per MoE layer:
// begin_fused is called before the layer's expert matmuls and makes
// llama_expert_tier_build return hot-only results; end_fused (after the down
// matmul) returns the cold contribution tensor to add to the down result, or
// nullptr when the fused path is not active. gate/up/down must all be
// registered. act = 0 (silu, separate gate/up) or 1 (gelu, fused gate_up).
bool llama_expert_tier_begin_fused(ggml_tensor * gate_w,
                                   ggml_tensor * up_w,
                                   ggml_tensor * down_w,
                                   ggml_tensor * ids);

ggml_tensor * llama_expert_tier_end_fused(ggml_context * ctx,
                                          ggml_tensor * gate_w,
                                          ggml_tensor * up_w,
                                          ggml_tensor * down_w,
                                          ggml_tensor * x,
                                          ggml_tensor * ids,
                                          int32_t        act);