#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-graph.h"

#include <cstdint>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;

// GLM-5-Next indexer pooling. the position -> cell map is built host side because
// find_slot's cell order is arbitrary. no input may hold a negative index: ggml_set_rows
// asserts i1 >= 0 and ggml_get_rows has no sentinel, so unusable entries are clamped into
// range and neutralised by the additive masks instead.

// pool slots for `n_kv` cells shared by `n_seqs` sequences: n_kv/kpool, exact only while
// the sequences' cells are disjoint, plus 2 per sequence for rebasing.
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// select_k of modular_glm5_next.py, Glm5NextTextIndexer.forward. must run over POOLS, not
// cells: relu ties span pool boundaries, so a cell-level cut takes partial pools.
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

// `kv` must be the ATTENTION (MLA) cache; the indexer cache shares its slot layout.
//   cell_pool  I32 [n_kv, n_stream]                 per-cell view, optional, unused here
//   pool_cells I32 [kpool*n_pools, n_stream]        pool member -> cell, 0 if not resident
//   bias       F32 [n_kv, n_tps, n_stream]          per-cell view, optional, unused here
//   pool_bias  F32 [n_pools, n_tps, n_stream]       pool_valid & pool_visible, -INFINITY
//       outside the query's own sequence run; computed, not gathered from `bias` at the
//       last member, which an incomplete pool lacks and would inherit cell 0's validity
//   sel_mask   F16/F32 [n_kv, n_batch, 1, n_stream] 0.0f on the always-selected tail only
//   cand_mask  F16/F32 [n_kv, n_batch, 1, n_stream] max(bias, sel_mask); bounds the top-k
//       spills that a partial seq_rm would otherwise let escape the candidate set
// pool_reps / new_pool_cells / new_pool_reps are optional (nullptr when the pooled-key
// cache is off). an entry is emitted only for a pool with filled == kpool: an incomplete
// pool's last-member slot is 0, and cell 0 is a legitimate cell, so writing it would
// clobber another pool's cached key.
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * tail_cells,   // optional, D0 gathered decode
              ggml_tensor    * tail_valid,   // optional, with tail_cells
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        // strm_of[s] is the physical stream behind view s and kv_size the per-stream cell
        // count, so a global row is strm_of[s]*kv_size + cell. only read when pool_reps is
        // set. the indexer cache shares the attention cache's slot layout, so one cell
        // index addresses both.
        const uint32_t       * strm_of,
              int64_t          kv_size,
        // re-emit every complete pool, not only the ones this ubatch closed. set after a
        // position mutation; the graph is built with n_new_max == n_pools to hold them.
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool);

// One pooling map per ubatch; rebuilding it per indexer layer costs O(n_kv * n_tokens)
// host writes and dominates prefill. sharing is valid only while every indexer layer sees
// the same candidate set - true for glm5next (indexer_types all "full"), not for windowed.
class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(
            const llama_kv_cache_context * mctx_attn,
            const llama_kv_cache_context * mctx_idx,
            uint32_t kpool) : mctx_attn(mctx_attn), mctx_idx(mctx_idx), kpool(kpool) {}

    ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_idxs     = nullptr;   // I32 [n_tokens]
    ggml_tensor * pool_cells = nullptr;   // I32 [kpool*n_pools, n_stream]
    ggml_tensor * pool_bias  = nullptr;   // F32 [n_pools, n_tps, n_stream]

    // pooled-key cache. the pooled value of a pool lives in the third head of the row of
    // the cell holding its LAST member (pos % kpool == kpool-1), so it is a pure function
    // of cell content: independent of sequence and of pool ordinal, which is what lets a
    // seq_cp share it and a rebase leave it alone.
    ggml_tensor * pool_reps      = nullptr; // I32 [n_pools, n_stream]  stream-local rep cell
    ggml_tensor * new_pool_cells = nullptr; // I32 [kpool*n_new_max, n_stream] members to (re)pool
    ggml_tensor * new_pool_reps  = nullptr; // I64 [n_new_max*n_stream] GLOBAL dest row

    // n_new_max is fixed for the whole decode phase on purpose. making the shape depend on
    // how many pools happened to close this step would flip the graph topology every kpool
    // tokens and defeat graph reuse / force CUDA-graph recapture.
    uint32_t n_new_max = 0;

    // this graph was built to re-emit every pool after a position mutation, not just the
    // ones this ubatch closed. decided at build time because it sets n_new_max above.
    bool rebuild = false;

    // exact, since pool_bias only holds 0.0f or -INFINITY. nullptr if the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    // D0 gathered DSA (codex 056 / grokk 057): the always-selected incomplete-pool tail as
    // explicit cells so a gathered graph can concat them to top_k. slot t holds the cell at
    // pos tail_start+t; unused slots are cell 0 with tail_valid -INFINITY (get_rows has no
    // sentinel). tail length is (q+1)%kpool in [0, kpool-1].
    ggml_tensor * tail_cells = nullptr;   // I32 [kpool-1, n_tps, n_stream]
    ggml_tensor * tail_valid = nullptr;   // F32 [kpool-1, n_tps, n_stream]

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;

    const uint32_t kpool;
};
