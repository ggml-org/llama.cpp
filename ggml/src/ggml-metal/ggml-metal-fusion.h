// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table, so the two
// phases can never disagree about what can be fused.

#pragma once

#include "ggml-impl.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#define GGML_METAL_FUSION_MAX 16

typedef enum ggml_metal_fusion_mode {
    // structural checks only; used by the graph optimizer, at which point the graph
    // tensors are not allocated yet, so buffer placement cannot be verified
    GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_FULL,
} ggml_metal_fusion_mode;

// identifier of each fusion pattern so the op encoders know which kernel to use
typedef enum ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSION_NORM_MUL_ADD, // NORM/RMS_NORM + MUL + ADD
    GGML_METAL_FUSION_ADD_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
} ggml_metal_fusion_id;

struct ggml_metal_fusion {
    ggml_metal_fusion_id id;

    const enum ggml_op * ops;        // op sequence (fixed length)
    int                  n_ops;      // number of ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern
    bool (*check)(const struct ggml_metal_fusion   * fusion,
                  const struct ggml_tensor * const * nodes,
                        ggml_metal_fusion_mode       mode);
};

typedef struct ggml_metal_fusion ggml_metal_fusion;

// the single table of all fusions supported by the Metal backend
const ggml_metal_fusion * ggml_metal_fusion_all(int * n);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
struct ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug);
void ggml_metal_fusion_info_free(struct ggml_metal_fusion_info * finfo);

bool ggml_metal_fusion_info_enabled(const struct ggml_metal_fusion_info * finfo);
bool ggml_metal_fusion_info_stats  (const struct ggml_metal_fusion_info * finfo);
int  ggml_metal_fusion_info_debug  (const struct ggml_metal_fusion_info * finfo);

int          ggml_metal_fusion_info_n_fusions(const struct ggml_metal_fusion_info * finfo);
const char * ggml_metal_fusion_info_label    (const struct ggml_metal_fusion_info * finfo, int idx);
uint64_t     ggml_metal_fusion_info_count    (const struct ggml_metal_fusion_info * finfo, int idx);

void ggml_metal_fusion_info_count_fusion(struct ggml_metal_fusion_info * finfo, const struct ggml_metal_fusion * fusion);
void ggml_metal_fusion_info_set_enabled (struct ggml_metal_fusion_info * finfo, bool enabled);

void ggml_metal_fusion_info_stats_init (      struct ggml_metal_fusion_info * finfo);
void ggml_metal_fusion_info_stats_reset(      struct ggml_metal_fusion_info * finfo);
int  ggml_metal_fusion_info_stats_get  (const struct ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n);
void ggml_metal_fusion_info_labels_init(      struct ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode`.
// returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
const ggml_metal_fusion * ggml_metal_fusion_next(
        const struct ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. returns at least 1.
int ggml_metal_fusion_max(const struct ggml_cgraph * gf, int idx);

#ifdef __cplusplus
}
#endif
