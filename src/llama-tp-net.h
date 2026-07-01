// Inter-node transport for CPU tensor parallelism (rank=node, 2-way), used by build_ffn to
// all-reduce the row-parallel ffn_down output across nodes. UCX/IB, no MPI. Optional: when
// built without LLAMA_TP_UCX, llama_tp_enabled() is false and the op is a no-op.
#ifndef LLAMA_TP_NET_H
#define LLAMA_TP_NET_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cheap env check (LLAMA_TP_SIZE > 1) used at graph-build time to decide whether to insert the
// all-reduce op. Does NOT bootstrap the network.
int llama_tp_enabled(void);

// Cheap env check (LLAMA_TP_SIZE > 1 && LLAMA_TP_ATTN) — true when attention is sharded (Phase 2)
// so build_attn inserts an all-reduce after the row-parallel wo. Does NOT bootstrap the network.
int llama_tp_attn_enabled(void);

// Cheap env check (LLAMA_TP_SIZE > 1 && LLAMA_TP_SSM) — true when the recurrent SSM/Mamba-2 mixer is
// sharded (heads/d_inner channel-parallel) so build_mamba2_layer all-reduces after ssm_out.
int llama_tp_ssm_enabled(void);

// Cheap env check — true when MoE experts are sharded across ranks (any mode) so build_moe_ffn
// shards the expert FFN and all-reduces the combine.
int llama_tp_moe_enabled(void);

// MoE-parallel mode: 0=off, 1=expert-parallel (sharded expert set, dynamic routing), 2=tensor-
// parallel (split each expert's n_ff like a dense FFN). Mirrors tp_moe_mode in llama-tp-shard.h.
int llama_tp_moe_mode(void);

// This rank's id and the TP world size (from the config, else LLAMA_TP_RANK / LLAMA_TP_SIZE).
int llama_tp_rank(void);
int llama_tp_size(void);

// Set the process-wide TP config from llama_model_params at model-load time. moe_mode uses the
// 0/1/2 encoding (off/expert/tensor). When size <= 1, the accessors fall back to LLAMA_TP_* env.
void llama_tp_set_config(int size, int rank, int moe_mode, int attn, int ssm, const char * peer, int port);

// rank-0 bootstrap address / port for the all-reduce transport (config, else LLAMA_TP_PEER/PORT env).
const char * llama_tp_peer(void);
int          llama_tp_port(void);

// ggml custom1 op (n_tasks=1): in-place inter-node SUM all-reduce of `dst` (the partial
// ffn_down output). On its first call (ggml thread 0) it lazily bootstraps the transport from
// env (LLAMA_TP_SIZE / LLAMA_TP_RANK / LLAMA_TP_PEER / LLAMA_TP_PORT), so the UCX worker is
// created and used by the same thread.
void llama_tp_allreduce_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                           int ith, int nth, void * userdata);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TP_NET_H
