// MoE expert-parallelism helpers for CPU tensor parallelism.
// Experts are sharded across ranks (each holds n_expert/size experts, a contiguous slice of the
// ffn_*_exps n_expert dim). The router (gate_inp) is replicated, so every rank computes the same
// global top-k selection. Each rank remaps the selected expert ids into its LOCAL index space and
// zeros the weights of non-local experts, runs the normal mul_mat_id FFN on its shard -> a PARTIAL
// moe_out, then the per-rank partials are combined with the inter-node all-reduce (llama_tp_allreduce_op).
//
// userdata for both ops packs this rank's local expert range into the pointer itself (no alloc):
//   ud = ((uintptr_t)count << 32) | (uintptr_t)base,  base = rank*count, count = n_expert/size.
#ifndef LLAMA_TP_MOE_H
#define LLAMA_TP_MOE_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// custom1 on selected_experts (i32): dst = (global - base) if local, else 0 (clamped).
// selected_experts is a strided top-k view -> pass a contiguous copy (ggml_cont).
void tp_moe_local_ids_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                         int ith, int nth, void * userdata);

// custom2 on (weights f32, selected_experts i32): dst = weights where the expert is local, else 0.
void tp_moe_mask_weights_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                            const struct ggml_tensor * b, int ith, int nth, void * userdata);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TP_MOE_H
