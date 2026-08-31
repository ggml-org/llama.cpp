// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fuse entry in
// the table in ggml-metal-fuse.cpp. both the graph optimizer (ggml_metal_fuse_max)
// and the op encoders (ggml_metal_fuse_next) consult this same table, so the two
// phases can never disagree about what can be fused.

#pragma once

#include "ggml-impl.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#define GGML_METAL_FUSE_MAX 16

enum ggml_metal_fuse_mode {
    // structural checks only; used by the graph optimizer, at which point the graph
    // tensors are not allocated yet, so buffer placement cannot be verified
    GGML_METAL_FUSE_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSE_FULL,
};

// identifier of each fusion pattern so the op encoders know which kernel to use
enum ggml_metal_fuse_id {
    GGML_METAL_FUSE_NONE = 0,
    GGML_METAL_FUSE_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSE_NORM_MUL_ADD, // NORM/RMS_NORM + MUL + ADD
    GGML_METAL_FUSE_ADD_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSE_SNAKE,        // MUL + SIN + SQR + MUL + ADD
};

struct ggml_metal_fuse {
    enum ggml_metal_fuse_id id;
    const enum ggml_op *    ops;      // op sequence (fixed length)
    int                     n_ops;    // number of ops
    const int *             outputs;  // output node indices (absolute graph indices; nullptr => the last node)
    int                     n_outputs;// number of outputs (0 => default last node)
    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern
    bool (*check)(const struct ggml_tensor * const * nodes,
                  const struct ggml_metal_fuse *    fuse,
                  enum ggml_metal_fuse_mode         mode);
};

// the single table of all fusions supported by the Metal backend
const struct ggml_metal_fuse * ggml_metal_fuse_all(int * n);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode`.
// returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
const struct ggml_metal_fuse * ggml_metal_fuse_next(
        const struct ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        enum ggml_metal_fuse_mode mode,
        int * n_out);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. returns at least 1.
int ggml_metal_fuse_max(const struct ggml_cgraph * gf, int idx);

#ifdef __cplusplus
}
#endif
