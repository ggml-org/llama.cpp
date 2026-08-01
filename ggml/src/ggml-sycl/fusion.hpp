#ifndef GGML_SYCL_FUSION_HPP
#define GGML_SYCL_FUSION_HPP

#include <initializer_list>

#include "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops);

#endif  // GGML_SYCL_FUSION_HPP
