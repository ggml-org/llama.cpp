#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

// whether a mat-vec node can use the fused MMVQ path
bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor);

// try and fuse nodes starting at node i and return the number of nodes to skip
int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i);

// add alloc deps for the fusions that need their inputs to stay alive (called from graph_optimize)
void ggml_cuda_fusion_add_alloc_deps(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params);
