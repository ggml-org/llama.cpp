#include "llama-tp-shard.h"
#include "llama-tp-net.h"

// Read the effective TP config: the values set from llama_model_params (llama_tp_set_config), or the
// legacy LLAMA_TP_* environment fallback — both resolved by the llama-tp-net accessors.
tp_shard_config tp_shard_from_env(void) {
    tp_shard_config c;
    c.size     = llama_tp_size();
    c.rank     = llama_tp_rank();
    c.enabled  = llama_tp_enabled();
    c.attn     = llama_tp_attn_enabled();
    c.moe_mode = (tp_moe_mode) llama_tp_moe_mode();
    c.moe      = (c.moe_mode != TP_MOE_OFF);
    c.ssm      = llama_tp_ssm_enabled();
    return c;
}

// bytes occupied by a row of `n` elements of a type with `block` elems/block, `type_size` bytes/block.
static size_t row_bytes(int64_t n, int64_t block, size_t type_size) {
    return (size_t)(n / block) * type_size;
}

int tp_shard_plan_make(tp_shard_role role, int rank, int size,
                       int64_t ne0_full, int64_t ne1_full, int64_t ne2_full,
                       int64_t block, size_t type_size,
                       tp_shard_plan * out) {
    out->n_seg = 0;        // single-stride plan (not an SSM multi-segment gather)
    out->ne2 = ne2_full;   // unchanged except for EXPERT sharding
    if (size <= 1 || role == TP_SHARD_NONE) {
        out->ne0 = ne0_full; out->ne1 = ne1_full; out->nrows = ne1_full;
        out->chunk_bytes = row_bytes(ne0_full, block, type_size);
        out->base_off = 0; out->src_stride = out->chunk_bytes;
        out->total_bytes = (size_t)ne1_full * out->chunk_bytes;
        return 0;
    }

    if (role == TP_SHARD_COLUMN) {
        // split ne[1] (output rows). Whole rows -> always block-safe. For a 2D weight this is one
        // contiguous read; for a 3D expert tensor (ne2 experts) it's one chunk per expert (each
        // expert keeps its full ne0, this rank's ne1 slice), read strided across experts.
        if (ne1_full % size != 0) return -1;
        int64_t ne1_s = ne1_full / size;
        size_t  rb    = row_bytes(ne0_full, block, type_size);
        out->ne0 = ne0_full; out->ne1 = ne1_s;
        out->nrows = ne2_full;                       // 1 for 2D weights
        out->chunk_bytes = (size_t)ne1_s * rb;
        out->base_off = (size_t)rank * ne1_s * rb;
        out->src_stride = (size_t)ne1_full * rb;     // full expert size (unused when nrows==1)
        out->total_bytes = (size_t)ne2_full * out->chunk_bytes;
        return 0;
    }

    if (role == TP_SHARD_EXPERT) {
        // split ne[2] (n_expert). Each expert is a contiguous ne0*ne1 block -> a single
        // contiguous read of this rank's expert slice. Whole experts -> always block-safe.
        if (ne2_full % size != 0) return -1;
        int64_t ne2_s = ne2_full / size;
        size_t  expert_bytes = (size_t)ne1_full * row_bytes(ne0_full, block, type_size);
        out->ne0 = ne0_full; out->ne1 = ne1_full; out->ne2 = ne2_s;
        out->nrows = 1;
        out->chunk_bytes = (size_t)ne2_s * expert_bytes;
        out->base_off = (size_t)rank * ne2_s * expert_bytes;
        out->src_stride = 0;
        out->total_bytes = out->chunk_bytes;
        return 0;
    }

    // TP_SHARD_ROW: split ne[0] (the contraction). Must be block-aligned per shard. One strided
    // chunk per output row; for a 3D expert tensor that's ne1*ne2 rows (experts are contiguous in
    // memory, so iterating rows across all experts with the full-row stride covers them all).
    if (ne0_full % (size * block) != 0) return -1;   // would cut a quant block -> corruption
    int64_t ne0_s = ne0_full / size;
    size_t  full_rb  = row_bytes(ne0_full, block, type_size);
    size_t  shard_rb = row_bytes(ne0_s,    block, type_size);
    out->ne0 = ne0_s; out->ne1 = ne1_full;
    out->nrows = ne1_full * ne2_full;      // one strided chunk per (output row, expert)
    out->chunk_bytes = shard_rb;
    out->base_off = (size_t)rank * shard_rb;
    out->src_stride = full_rb;
    out->total_bytes = (size_t)out->nrows * shard_rb;
    return 0;
}
