// Load-time weight sharding math for CPU tensor parallelism.
// The crux of loader sharding is computing, per rank, the sharded tensor shape and the
// byte layout of this rank's slice inside the GGUF tensor data — correctly for QUANTIZED
// weights (block-aligned), including the strided per-row read needed for row-parallel.
//
// ggml weight layout: ne[0]=contraction (cols), ne[1]=output rows; row-major; a row of n
// elements occupies ggml_row_size(type,n) = type_size * (n / block_size) bytes.
//
//   Column-parallel (wq/wk/wv/ffn_up/ffn_gate): split ne[1] (whole output rows). One
//     contiguous read. Always block-safe (rows are whole).
//   Row-parallel (wo/ffn_down): split ne[0] (the contraction). Each output row contributes
//     a sub-range; the rank's data is STRIDED (one chunk per row). Requires ne0/size to be a
//     multiple of the quant block size, else it cuts a block and corrupts the weights.
#ifndef TP_SHARD_H
#define TP_SHARD_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// COLUMN: split ne[1] (output rows). ROW: split ne[0] (contraction, block-aligned, strided).
// EXPERT: split ne[2] (the n_expert dim of a 3D MoE expert tensor) — a contiguous slice of
//         whole experts (each expert is a contiguous ne0*ne1 block), so it's a single read.
typedef enum { TP_SHARD_NONE = 0, TP_SHARD_COLUMN, TP_SHARD_ROW, TP_SHARD_EXPERT } tp_shard_role;

// How a MoE model is parallelized across ranks (selectable, like LLAMA_SPLIT_MODE):
//   OFF      experts not sharded (single-process MoE).
//   EXPERT   expert parallelism: each rank owns n_expert/size whole experts (TP_SHARD_EXPERT).
//            Scales to many ranks (capacity); routing is dynamic so compute is imbalanced.
//   TENSOR   tensor parallelism on the experts: split each expert's intermediate (n_ff) like a
//            dense FFN (gate/up COLUMN, down ROW). Balanced, splits compute even at decode, but
//            limited by n_ff / quant-block size (down must stay block-aligned).
typedef enum { TP_MOE_OFF = 0, TP_MOE_EXPERT = 1, TP_MOE_TENSOR = 2 } tp_moe_mode;

typedef struct { int size; int rank; int enabled; int attn; int moe; tp_moe_mode moe_mode; int ssm; } tp_shard_config;

// Reads LLAMA_TP_SIZE and LLAMA_TP_RANK from the environment. enabled = (size > 1).
// attn = (enabled && LLAMA_TP_ATTN); when set, attention (wq/wk/wv/wo) is sharded too
// (Phase 2). When unset, only the FFN is sharded (Phase 1, attention replicated).
// moe  = (enabled && LLAMA_TP_MOE); when set, MoE expert tensors (ffn_*_exps) are sharded
// across ranks (expert parallelism): each rank holds n_expert/size experts.
tp_shard_config tp_shard_from_env(void);

// Max segments for an SSM concatenated-projection gather (z,x,B,C,dt = 5).
#define TP_MAX_SEG 6

// A load plan: `nrows` chunks of `chunk_bytes`, the i-th read from
//   src: base_off + i*src_stride   ->   dst: i*chunk_bytes (contiguous).
typedef struct {
    int64_t ne0, ne1, ne2; // sharded dimensions (ne2 only differs from full for EXPERT)
    int64_t nrows;        // number of chunks (1 for column/expert-parallel; ne1 for row-parallel)
    size_t  chunk_bytes;  // bytes per chunk
    size_t  base_off;     // byte offset of first chunk within the source tensor data
    size_t  src_stride;   // source stride between chunks
    size_t  total_bytes;  // == nrows * chunk_bytes (the sharded tensor size)
    // SSM multi-segment gather: when n_seg>0 the rank's data is the concatenation of n_seg
    // contiguous byte spans (one per kept sub-slice of a [z|x|B|C|dt]-style projection), packed
    // into dst in order. Overrides the nrows/base_off/src_stride single-stride read above.
    int     n_seg;
    size_t  seg_src_off[TP_MAX_SEG]; // byte offset of each span in the source tensor data
    size_t  seg_bytes[TP_MAX_SEG];   // bytes copied from each span (sum == total_bytes)
} tp_shard_plan;

// Compute the sharded shape + load plan. block = ggml_blck_size(type) (elements/block),
// type_size = ggml_type_size(type) (bytes/block). ne2_full is the 3rd dim (n_expert for MoE
// expert tensors; pass 1 for 2D weights). Returns 0 on success, <0 on a divisibility /
// block-alignment violation (caller should refuse to shard and fall back).
int tp_shard_plan_make(tp_shard_role role, int rank, int size,
                       int64_t ne0_full, int64_t ne1_full, int64_t ne2_full,
                       int64_t block, size_t type_size,
                       tp_shard_plan * out);

#ifdef __cplusplus
}
#endif

#endif // TP_SHARD_H
