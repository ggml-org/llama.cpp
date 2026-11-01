#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"

#include <unordered_map>

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_sycl_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);

// GGML_SYCL_GDN_STATE_GATHER: a GET_ROWS fetching one recurrent-state row that only a single-sequence
// GATED_DELTA_NET reads is skipped, and the GDN reads the row straight from the cache. Matched per graph,
// gather -> its GDN.
using ggml_sycl_gdn_state_gathers = std::unordered_map<const ggml_tensor *, const ggml_tensor *>;
ggml_sycl_gdn_state_gathers ggml_sycl_match_gdn_state_gathers(const ggml_cgraph * cgraph);

// in the compute loop: true for a matched gather, which is skipped; a matched GDN gets its cache rows in
// ctx.gdn_gather
bool ggml_sycl_gdn_state_gather_skip(ggml_backend_sycl_context & ctx, const ggml_tensor * node,
                                     const ggml_sycl_gdn_state_gathers & gathers);
